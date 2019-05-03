#!/usr/bin/env python
# coding: utf-8

from geoimagenet_ml.constants import OPERATION, VISIBILITY
from geoimagenet_ml.utils import (
    now,
    localize_datetime,
    datetime2str,
    str2datetime,
    get_settings,
    get_log_fmt,
    get_log_datefmt,
    get_job_log_msg,
    get_error_fmt,
    fully_qualified_name,
    is_uuid,
    isclass,
)
from geoimagenet_ml.status import COMPLIANT, CATEGORY, STATUS, job_status_categories, map_status
from geoimagenet_ml.processes.types import process_mapping, PROCESS_WPS
from geoimagenet_ml.store import exceptions as ex
from geoimagenet_ml.ml.impl import load_model
from pyramid_celery import celery_app as app
# noinspection PyPackageRequirements
from dateutil.parser import parse
from datetime import datetime
from zipfile import ZipFile
import json
import boltons.tbutils
import logging
import shutil
import uuid
import six
import os
import io
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # noinspection PyPackageRequirements
    from geoimagenet_ml.typedefs import (   # noqa: F401
        AnyStr, AnyStatus, ErrorType, LevelType, List, LoggerType, Number, Union,
        Optional, InputType, OutputType, UUID, JSON, ParamsType, Type,
    )
    from pywps import Process as ProcessWPS  # noqa: F401


def _check_io_format(io_items):
    # type: (List[Union[InputType, OutputType]]) -> None
    """
    Basic validation of input/output for process and job results.

    :raises TypeError: on any invalid format encountered.
    :raises ValueError: on invalid value conditions encountered.
    """
    if not isinstance(io_items, list) or \
            not all(isinstance(_io, dict) and all(k in _io for k in ["id", "value"]) for _io in io_items):
        raise TypeError("Expect a list of items with id/value keys.")
    id_items = [_io["id"] for _io in io_items]
    if len(id_items) != len(set(id_items)):
        raise ValueError("Duplicate item id in list.")


class Base(dict):
    """
    Dictionary with extended attributes auto-``getter``/``setter`` for convenience.
    Explicitly overridden ``getter``/``setter`` attributes are called instead of ``dict``-key ``get``/``set``-item
    to ensure corresponding checks and/or value adjustments are executed before applying it to the sub-``dict``.

    .. code-block:: python

        b = Base()

        # all following cases will ensure validation of 'uuid' field with setter property, and therefore will raise
        b.uuid = "1234"
        b["uuid"] = "1234"
        setattr(b, "uuid", "1234")

        # these will not do any validation because no 'other' setter property exists
        b["other"] = "blah"
        setattr(b, "other", "blah")

        # same checks applies for getter properties
        b.uuid
        >> "<valid-uuid>"
        b["uuid"]
        >> "<valid-uuid>"
        getattr(b, "uuid", "1234")
        >> "<valid-uuid>"

    Property getter/setter implementations should use following methods to obtain the literal get/set after validation:

    .. code-block:: python

        value = dict.__getitem__(self, "<field>")
        dict.__setitem__(self, "<field>", <value>)

    """
    def __init__(self, *args, **kwargs):
        super(Base, self).__init__(*args, **kwargs)
        if "uuid" not in self:
            setattr(self, "uuid", uuid.uuid4())
        if "created" not in self:
            setattr(self, "created", now())

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __setitem__(self, item, value):
        self.__setattr__(item, value)

    def __setattr__(self, item, value):
        # use the existing property setter if defined
        prop = getattr(type(self), item)
        if isinstance(prop, property) and prop.fset is not None:
            # noinspection PyArgumentList
            prop.fset(self, value)
        elif item in self:
            dict.__setitem__(self, item, value)
        else:
            raise AttributeError("Can't set attribute '{}'.".format(item))

    def __getattr__(self, item):
        # use existing property getter if defined
        prop = getattr(type(self), item)
        if isinstance(prop, property) and prop.fget is not None:
            # noinspection PyArgumentList
            return prop.fget(self)
        elif item in self:
            return dict.__getitem__(self, item)
        else:
            raise AttributeError("Can't get attribute '{}'.".format(item))

    def __str__(self):
        # type: () -> AnyStr
        cls = type(self)
        return "{} <{}>".format(cls.__name__, self.uuid)

    def __repr__(self):
        # type: () -> AnyStr
        cls = type(self)
        rep = dict.__repr__(self)
        return "{0}.{1} ({2})".format(cls.__module__, cls.__name__, rep)

    def _get_params(self, extra_params=None):
        # type: (Optional[ParamsType]) -> ParamsType
        """
        Collects ``params`` fields defined by every child ``With<Field>`` and aggregates them for the parent class.
        """
        base_params = {}
        for cls in type(self).__bases__:
            # noinspection PyUnresolvedReferences
            base_params.update(cls.params.fget(self))
        # apply parent definitions last to guarantee they override any base definitions
        base_params.update(extra_params or {})
        return base_params

    def _get_json(self, extra_json=None):
        # type: (Optional[JSON]) -> JSON
        """
        Collects ``json`` fields defined by every child ``With<Field>`` and aggregates them for the parent class.
        """
        base_json = {}
        for cls in type(self).__bases__:
            # noinspection PyUnresolvedReferences
            base_json.update(cls.json(self))
        # apply parent definitions last to guarantee they override any base definitions
        base_json.update(extra_json or {})
        return base_json

    @property
    def uuid(self):
        # type: () -> UUID
        return dict.__getitem__(self, "uuid")

    @uuid.setter
    def uuid(self, _uuid):
        # type: (UUID) -> None
        if not is_uuid(_uuid):
            raise ValueError("Not a valid UUID: {!s}.".format(_uuid))
        dict.__setitem__(self, "uuid", str(_uuid))

    @property
    def created(self):
        # type: () -> datetime
        created = self.get("created")
        if not created:
            dict.__setitem__(self, "created", now())
        if isinstance(created, six.string_types):
            setattr(self, "created", parse(dict.__getitem__(self, "created")))
        return self.get("created")

    @created.setter
    def created(self, dt):
        # type: (datetime) -> None
        if not isinstance(dt, datetime):
            raise TypeError("Type 'datetime' expected.")
        dict.__setitem__(self, "created", localize_datetime(dt))

    @property
    def params(self):
        # type: () -> ParamsType
        """
        Method used to collect and convert parameters of the datatype to a format that the storage will accept.
        Overloading by parent classes should include a call to :method:`_get_params`.
        """
        return {
            "uuid": self.uuid,
            "created": self.created,
        }

    def json(self):
        # type: () -> JSON
        """
        Method used to collect and convert parameters of the datatype to a format that will be JSON-serializable.
        Overloading by parent classes should include a call to :method:`_get_json`.
        """
        return {
            "uuid": self.uuid,
            "created": datetime2str(self.created) if self.created else None
        }


class WithFinished(dict):
    """Adds property ``finished`` to corresponding class."""
    @property
    def finished(self):
        # type: () -> Optional[datetime]
        finished = self.get("finished")
        if isinstance(finished, six.string_types):
            finished = str2datetime(finished)
        if finished:
            return finished
        return None

    @finished.setter
    def finished(self, dt):
        # type: (datetime) -> None
        if not isinstance(dt, datetime):
            raise TypeError("Type 'datetime' required.")
        self["finished"] = localize_datetime(dt)

    def is_finished(self):
        # type: () -> bool
        return self.finished is not None

    @property
    def params(self):
        # type: () -> ParamsType
        return {"finished": self.finished}

    def json(self):
        # type: () -> JSON
        return {"finished": datetime2str(self.finished) if self.finished else None}


class WithName(dict):
    """Adds property ``name`` to corresponding class."""

    @property
    def name(self):
        # type: () -> AnyStr
        return self["name"]

    @name.setter
    def name(self, name):
        # type: (AnyStr) -> None
        self["name"] = name

    @property
    def params(self):
        # type: () -> ParamsType
        return {"name": self.name}

    def json(self):
        # type: () -> JSON
        return {"name": self.name}


class WithType(dict):
    """Adds property ``type`` to corresponding class."""
    @property
    def type(self):
        # type: () -> AnyStr
        return self["type"]

    @type.setter
    def type(self, _type):
        # type: (AnyStr) -> None
        self["type"] = _type

    @property
    def params(self):
        # type: () -> ParamsType
        return {"type": self.type}

    def json(self):
        # type: () -> JSON
        return {"type": self.type}


class WithUser(dict):
    """Adds property ``user`` to corresponding class."""
    @property
    def user(self):
        # type: () -> Optional[int]
        return self.get("user", None)

    @user.setter
    def user(self, user):
        # type: (Optional[int]) -> None
        if not isinstance(user, int) and user is not None:
            raise TypeError("Type 'int' or 'None' is required for '{}.user'".format(type(self)))
        self["user"] = user

    @property
    def params(self):
        # type: () -> ParamsType
        return {"user": self.user}

    def json(self):
        # type: () -> JSON
        return {"user": self.user}


class WithVisibility(dict):
    """Adds properties ``visibility`` to corresponding class."""

    # noinspection PyTypeChecker
    @property
    def visibility(self):
        # type: () -> VISIBILITY
        return VISIBILITY.get(self.get("visibility"), default=VISIBILITY.PRIVATE)

    @visibility.setter
    def visibility(self, visibility):
        # type: (Union[VISIBILITY, AnyStr]) -> None
        if isinstance(visibility, six.string_types):
            visibility = VISIBILITY.get(visibility)
        if visibility not in VISIBILITY:
            raise TypeError("Type 'VISIBILITY' required.")
        self["visibility"] = visibility

    @property
    def params(self):
        # type: () -> ParamsType
        return {"visibility": self.visibility.name}

    def json(self):
        # type: () -> JSON
        return {"visibility": self.visibility.value}


class Dataset(Base, WithName, WithType, WithUser, WithFinished):
    """
    Dictionary that contains a dataset description for db storage.
    It always has keys ``name``, ``path``, ``type``, ``data`` and ``files``.
    """

    def __init__(self, *args, **kwargs):
        super(Dataset, self).__init__(*args, **kwargs)
        if "name" not in self:
            raise TypeError("Dataset 'name' is required.")
        if "type" not in self:
            raise TypeError("Dataset 'type' is required.")
        if "data" not in self:
            setattr(self, "data", dict())
        if "files" not in self:
            setattr(self, "files", list())
        if "status" not in self:
            setattr(self, "status", STATUS.RUNNING)

    @property
    def path(self):
        # type: () -> AnyStr
        """Retrieves the dataset path. Automatically creates the directory if not overridden during initialization."""
        if not self.get("path"):
            settings = get_settings(app)
            dataset_root = str(settings["geoimagenet_ml.ml.datasets_path"])
            if not os.path.isdir(dataset_root):
                raise RuntimeError("cannot find datasets root path")
            path = os.path.join(dataset_root, self.uuid)
            dict.__setitem__(self, "path", path)
            os.makedirs(path, exist_ok=False, mode=0o744)
        return dict.__getitem__(self, "path")

    @path.setter
    def path(self, path):
        # type: (AnyStr) -> None
        if not os.path.isdir(path):
            raise ValueError("Dataset path must be an existing directory.")
        dict.__setitem__(self, "path", path)

    def reset_path(self):
        """Clear all the 'path' content as regenerates a clean directory state."""
        if isinstance(self.path, six.string_types) and os.path.isdir(self.path):
            shutil.rmtree(self.path)
        dict.__setitem__(self, "path", None)
        if not os.path.isdir(self.path):  # trigger regeneration
            raise ValueError("Failed dataset path reset to clean state")

    def zip(self):
        # type: () -> AnyStr
        """
        Creates a ZIP file (if missing) of the dataset details and files.
        The content of the ZIP includes everything found inside its save directory.
        :returns: saved ZIP path
        """
        if not os.path.isdir(self.path):
            raise ValueError("Cannot generate ZIP from invalid dataset path: [{!s}]".format(self.path))
        dataset_zip_path = "{}.zip".format(self.path)
        if not os.path.isfile(dataset_zip_path):
            dataset_meta = os.path.join(self.path, "meta.json")
            if os.path.isfile(dataset_meta):
                os.remove(dataset_meta)
            # generate formatted JSON with substituted server paths
            settings = get_settings(app)
            base_path = str(settings["geoimagenet_ml.ml.datasets_path"])
            meta_str = json.dumps(self.data, indent=4)
            meta_str = meta_str.replace(self.path + "/", "")
            meta_str = meta_str.replace(base_path + "/", "")
            with open(dataset_meta, 'w') as f_meta:
                f_meta.write(meta_str)
            with ZipFile(dataset_zip_path, 'w') as f_zip:
                f_zip.write(dataset_meta, arcname=os.path.split(dataset_meta)[-1])
                for f in self.files:
                    f_zip.write(f, arcname=os.path.split(f)[-1])
        return dataset_zip_path

    @property
    def status(self):
        # type: () -> STATUS
        status = self.get("status")
        if not status:
            status = STATUS.RUNNING
            dict.__setitem__(self, "status", status)
        if isinstance(status, six.string_types):
            status = STATUS.get(status)
            if not isinstance(status, STATUS):
                raise TypeError("Invalid 'STATUS' enum value: '{}'.")
        return status

    @status.setter
    def status(self, status):
        # type: (AnyStatus) -> None
        status = map_status(status, COMPLIANT.LITERAL)
        if not isinstance(status, STATUS):
            raise TypeError("Type 'STATUS' enum is expected.")
        if status == STATUS.UNKNOWN:
            raise ValueError("Unknown status not allowed.")
        dict.__setitem__(self, "status", status)

    def mark_finished(self):
        # type: () -> None
        setattr(self, "finished", now())
        setattr(self, "status", STATUS.FINISHED)

    @property
    def data(self):
        # type: () -> JSON
        """Raw data contained in the dataset definition."""
        return dict.__getitem__(self, "data")

    @data.setter
    def data(self, data):
        # type: (Optional[JSON]) -> None
        dict.__setitem__(self, "data", data)

    @property
    def files(self):
        # type: () -> List[AnyStr]
        """All files referenced by the dataset."""
        return dict.__getitem__(self, "files")

    @files.setter
    def files(self, files):
        # type: (List[AnyStr]) -> None
        if not isinstance(files, list):
            raise TypeError("Type 'list' required.")
        dict.__setitem__(self, "files", files)

    @property
    def params(self):
        # type: () -> ParamsType
        return self._get_params({
            "name": self.name,
            "path": self.path,
            "type": self.type,
            "data": self.data,
            "files": self.files,
            "status": self.status.name,
        })

    def json(self):
        # type: () -> JSON
        return self._get_json({
            "name": self.name,
            "path": self.path,
            "type": self.type,
            "data": self.data,
            "files": self.files,
            "status": self.status.value,
        })

    def summary(self):
        # type: () -> JSON
        return {
            "uuid": self.uuid,
            "name": self.name,
        }


class Model(Base, WithName, WithUser, WithVisibility):
    """
    Dictionary that contains a model description for db storage.
    It always has ``name`` and ``path`` keys.
    """

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        if "name" not in self:
            raise TypeError("Model 'name' is required.")
        if "path" not in self:
            raise TypeError("Model 'path' is required.")

    @property
    def name(self):
        # type: () -> AnyStr
        return self["name"]

    @property
    def path(self):
        # type: () -> AnyStr
        """Original path specified for model creation."""
        return self["path"]

    @property
    def format(self):
        # type: () -> AnyStr
        """Original file format (extension)."""
        return os.path.splitext(self.path)[-1]

    @property
    def file(self):
        # type: () -> AnyStr
        """Stored file path of the created model."""
        return self.get("file")

    @property
    def data(self):
        # type: () -> Optional[ParamsType]
        """
        Retrieve the model's data config from the stored file.
        If data was already loaded and cached in the object, it is returned instead.
        """
        if self.get("data") is not None:
            return self["data"]
        if self.file:
            data, _ = self._load_check_data(self.file)
            self["data"] = data
            return data
        return None

    def save(self, location):
        """
        Saves the model data to specified storage ``location``, using the available ``file``, ``data`` or ``path``
        field, in this specific order of priority. Execute validation of data contents before saving.

        Cases:
            - ``file`` is available and exists, duplicate is created at ``location``
            - ``file`` is missing and ``data`` is *raw* checkpoint data (not loaded checkpoint config),
              it is dumped at ``location``
            - ``path`` is the only provided field and points to a valid path/URL, fetches and dumps
              its content to ``location``

        After calling this method, ``data`` is cleared (as required) and ``file`` is updated with ``location``.

        :returns: model safe to write to database (JSON serializable), with cleared `data` field and updated `file`.
        :raises ModelInstanceError:
            if none of the fields can help retrieve the model's data.
        :raises ModelLoadingError:
            if saving cannot be accomplished using provided fields because of invalid format or failing validation.
        :raises ModelRegistrationError:
            if the save location is invalid.
        :raises ModelConflictError:
            if the save location already exists.
        """
        if not isinstance(location, six.string_types):
            raise ex.ModelRegistrationError("Model save location has to be a string.")
        location = os.path.abspath(location)
        os.makedirs(os.path.dirname(location), exist_ok=True)

        if self.file and os.path.isfile(self.file) and self.file != location:
            if os.path.isfile(location):
                raise ex.ModelConflictError("Model save location already exists.")
            shutil.copyfile(self.file, location)
            self["file"] = location
            self["data"] = None
            return

        def _write_buffer(_buffer):
            # transfer loaded data buffer to storage file
            with open(location, 'wb') as model_file:
                _buffer.seek(0)
                model_file.write(_buffer.read())
                _buffer.close()

        if isinstance(self.data, io.BufferedIOBase):
            _write_buffer(self["data"])
            self["file"] = location
            self["data"] = None
            return

        if not self.file and not self.data and self.path:
            _, buffer = self._load_check_data(self.path)
            _write_buffer(buffer)
            self["file"] = location
            self["data"] = None
            return

        raise ex.ModelInstanceError("Model is expected to provided one of valid field: [file, data, path]")

    @staticmethod
    def _load_check_data(path):
        success, data, buffer, exception = load_model(path)  # nothrow operation
        if not success:
            if not exception:
                exception = "unknown reason"
            raise ex.ModelLoadingError("Failed loading model data: [{!r}].".format(exception))
        return data, buffer

    @property
    def params(self):
        # type: () -> ParamsType
        return self._get_params({
            # note: purposely avoid 'data' field to store only 'file' information
            "file": self.file,  # saved location (storage)
            "path": self.path,  # input location (submit)
        })

    def json(self):
        # type: () -> JSON
        return self._get_json({
            "path": self.path,
        })

    def summary(self):
        # type: () -> JSON
        return {
            "uuid": self.uuid,
            "name": self.name,
        }


class Process(Base, WithType, WithUser):
    """
    Dictionary that contains a process description for db storage.
    It always has ``uuid`` and ``identifier`` keys.

    Field ``type`` represents the wps, workflow, etc.
    """

    def __init__(self, *args, **kwargs):
        super(Process, self).__init__(*args, **kwargs)
        # use both 'id' and 'identifier' to support any call (WPS and recurrent 'id')
        if "identifier" not in self:
            raise TypeError("Process 'identifier' is required.")
        if "type" not in self:
            raise TypeError("Process 'type' is required.")

    def __str__(self):
        # type: () -> AnyStr
        return "{} ({})".format(super(Process, self).__str__(), self.title)

    @property
    def identifier(self):
        # type: () -> AnyStr
        return self["identifier"]

    @property
    def title(self):
        # type: () -> AnyStr
        return self.get("title", self.identifier)

    @property
    def abstract(self):
        # type: () -> AnyStr
        return self.get("abstract", "")

    @property
    def keywords(self):
        # type: () -> List[AnyStr]
        return self.get("keywords", [])

    @property
    def metadata(self):
        # type: () -> List[AnyStr]
        return self.get("metadata", [])

    @property
    def version(self):
        # type: () -> AnyStr
        return self.get("version")

    @property
    def inputs(self):
        # type: () -> List[InputType]
        return self.get("inputs", [])

    @inputs.setter
    def inputs(self, inputs):
        _check_io_format(inputs)
        dict.__setitem__(self, "inputs", inputs)

    @property
    def outputs(self):
        # type: () -> List[OutputType]
        return self.get("outputs", [])

    @outputs.setter
    def outputs(self, outputs):
        _check_io_format(outputs)
        dict.__setitem__(self, "outputs", outputs)

    @property
    def execute_endpoint(self):
        # type: () -> Optional[AnyStr]
        return self.get("execute_endpoint")

    @property
    def package(self):
        # type: () -> ParamsType
        return dict.__getitem__(self, "package")

    @property
    def limit_single_job(self):
        # type: () -> bool
        if "limit_single_job" not in self:
            setattr(self, "limit_single_job", False)
        return dict.__getitem__(self, "limit_single_job")

    @limit_single_job.setter
    def limit_single_job(self, value):
        if not isinstance(value, bool):
            raise TypeError("Invalid bool for 'limit_single_job'.")
        dict.__setitem__(self, "limit_single_job", False)

    @property
    def params(self):
        # type: () -> ParamsType
        return self._get_params({
            "identifier": self.identifier,
            "title": self.title,
            "abstract": self.abstract,
            "keywords": self.keywords,
            "metadata": self.metadata,
            "version": self.version,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "execute_endpoint": self.execute_endpoint,
            "limit_single_job": self.limit_single_job,
            "package": self.package,      # deployment specification (json body)
        })

    @property
    def params_wps(self):
        # type: () -> ParamsType
        """
        Values applicable to WPS Process ``__init__``
        """
        return {
            "identifier": self.identifier,
            "title": self.title,
            "abstract": self.abstract,
            "keywords": self.keywords,
            "metadata": self.metadata,
            "version": self.version,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }

    def json(self):
        # type: () -> JSON
        return self._get_json({
            "identifier": self.identifier,
            "title": self.title,
            "abstract": self.abstract,
            "keywords": self.keywords,
            "metadata": self.metadata,
            "version": self.version,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "execute_endpoint": self.execute_endpoint,
            "limit_single_job": self.limit_single_job,
        })

    def summary(self):
        # type: () -> JSON
        return {
            "uuid": self.uuid,
            "identifier": self.identifier,
            "title": self.title,
            "abstract": self.abstract,
            "keywords": self.keywords,
            "metadata": self.metadata,
            "version": self.version,
            "execute_endpoint": self.execute_endpoint,
        }

    def wps(self):
        # type: () -> ProcessWPS
        process_key = self.identifier
        if self.type != PROCESS_WPS:
            raise ex.ProcessInstanceError("Invalid WPS process call for '{}' of type '{}'."
                                          .format(process_key, self.type))
        if process_key not in process_mapping:
            raise ex.ProcessInstanceError("Unknown process '{}' in mapping".format(process_key))
        kwargs = self.params_wps
        return process_mapping[process_key](**kwargs)


class Job(Base, WithUser, WithFinished, WithVisibility):
    """
    Dictionary that contains a job description for db storage.
    It always has ``uuid`` and ``process`` keys.
    """

    def __init__(self, *args, **kwargs):
        super(Job, self).__init__(*args, **kwargs)
        if "uuid" not in self:
            raise TypeError("Job 'uuid' is required.")
        if "process" not in self:
            raise TypeError("Job 'process' is required.")

    def _get_log_msg(self, msg=None):
        # type: (Optional[AnyStr]) -> AnyStr
        if not msg:
            msg = self.status_message
        return get_job_log_msg(duration=self.duration, progress=self.progress, status=self.status, message=msg)

    def save_log(self, errors=None, logger=None, level=None):
        # type: (Optional[ErrorType], Optional[LoggerType], Optional[LevelType]) -> None
        if level is None:
            level = logging.INFO
        if isinstance(level, six.string_types):
            level = logging.getLevelName(level)
        if isinstance(errors, six.string_types):
            log_msg = [(logging.ERROR, self._get_log_msg())]
            self.exceptions.append(errors)
        elif isinstance(errors, Exception):
            log_msg = [(logging.ERROR, self._get_log_msg())]
            tb_info = boltons.tbutils.TracebackInfo.from_traceback()
            self.exceptions.extend([{
                "func_name": frame["func_name"],
                "line_detail": frame["line"].strip(),
                "line_number": frame["lineno"],
                "module_name": frame["module_name"],
                "module_path": frame["module_path"],
            } for frame in tb_info.to_dict()["frames"]])
        elif isinstance(errors, list):
            log_msg = [(logging.ERROR, self._get_log_msg(get_error_fmt().format(error)))
                       for error in errors]
            self.exceptions.extend([{
                "Code": error.code,
                "Locator": error.locator,
                "Text": error.text
            } for error in errors])
        else:
            log_msg = [(level, self._get_log_msg())]
        for lvl, msg in log_msg:
            # noinspection PyProtectedMember
            fmt_msg = get_log_fmt() % dict(asctime=datetime2str(fmt=get_log_datefmt()),
                                           levelname=logging.getLevelName(lvl),
                                           name=fully_qualified_name(self),
                                           message=msg)
            if len(self.logs) == 0 or self.logs[-1] != fmt_msg:
                self.logs.append(fmt_msg)
                if logger:
                    logger.log(lvl, msg)

    @property
    def task(self):
        # type: () -> UUID
        return self.get("task")

    @task.setter
    def task(self, task):
        # type: (UUID) -> None
        if not is_uuid(task):
            raise TypeError("Type 'UUID' is required for '{}.task'".format(type(self)))
        self["task"] = task

    @property
    def service(self):
        # type: () -> Optional[UUID]
        return self.get("service", None)

    @service.setter
    def service(self, service):
        # type: (UUID) -> None
        if not is_uuid(service):
            raise TypeError("Type 'UUID' is required for '{}.service'".format(type(self)))
        self["service"] = service

    @property
    def process(self):
        # type: () -> Optional[UUID]
        return self.get("process", None)

    @process.setter
    def process(self, process):
        # type: (UUID) -> None
        if not is_uuid(process):
            raise TypeError("Type 'UUID' is required for '{}.process'".format(type(self)))
        self["process"] = process

    def _get_inputs(self):
        # type: () -> List[JSON]
        if self.get("inputs") is None:
            dict.__setitem__(self, "inputs", list())
        return dict.__getitem__(self, "inputs")

    def _set_inputs(self, inputs):
        # type: (List[JSON]) -> None
        if not isinstance(inputs, list):
            raise TypeError("Type 'list' is required for '{}.inputs'".format(type(self)))
        if not all(isinstance(i, dict) for i in inputs):
            raise TypeError("Type 'dict' is required for elements of '{}.inputs'".format(type(self)))
        dict.__setitem__(self, "inputs", inputs)

    # allows to correctly update list by ref using 'job.inputs.extend()'
    inputs = property(_get_inputs, _set_inputs)

    @property
    def status(self):
        # type: () -> AnyStr
        return self.get("status", STATUS.UNKNOWN)

    @status.setter
    def status(self, status):
        # type: (STATUS) -> None
        status = map_status(status, COMPLIANT.LITERAL)
        if not isinstance(status, STATUS):
            raise TypeError("Type 'STATUS' enum is expected.")
        if status == STATUS.UNKNOWN:
            raise ValueError("Unknown status not allowed.")
        dict.__setitem__(self, "status", status)
        if status in job_status_categories[CATEGORY.EXECUTING]:
            self.mark_started()
        if status in job_status_categories[CATEGORY.FINISHED]:
            self.mark_finished()

    @property
    def status_message(self):
        # type: () -> AnyStr
        return self.get("status_message", "no message")

    @status_message.setter
    def status_message(self, message):
        # type: (Optional[AnyStr]) -> None
        if message is None:
            return
        if not isinstance(message, six.string_types):
            raise TypeError("Type 'str' is required for '{}.status_message'".format(type(self)))
        self["status_message"] = message

    @property
    def status_location(self):
        # type: () -> Optional[AnyStr]
        return self.get("status_location", None)

    @status_location.setter
    def status_location(self, location_url):
        # type: (Optional[AnyStr]) -> None
        if not isinstance(location_url, six.string_types) or location_url is None:
            raise TypeError("Type 'str' is required for '{}.status_location'".format(type(self)))
        self["status_location"] = location_url

    @property
    def execute_async(self):
        # type: () -> bool
        return self.get("execute_async", True)

    @execute_async.setter
    def execute_async(self, execute_async):
        # type: (bool) -> None
        if not isinstance(execute_async, bool):
            raise TypeError("Type 'bool' is required for '{}.execute_async'".format(type(self)))
        self["execute_async"] = execute_async

    @property
    def is_workflow(self):
        # type: () -> bool
        return self.get("is_workflow", False)

    @is_workflow.setter
    def is_workflow(self, is_workflow):
        # type: (bool) -> None
        if not isinstance(is_workflow, bool):
            raise TypeError("Type 'bool' is required for '{}.is_workflow'".format(type(self)))
        self["is_workflow"] = is_workflow

    @property
    def started(self):
        # type: () -> Optional[datetime]
        started = self.get("started")
        if isinstance(started, six.string_types):
            started = str2datetime(started)
        if started:
            return started
        return None

    @started.setter
    def started(self, dt):
        # type: (datetime) -> None
        if not isinstance(dt, datetime):
            raise TypeError("Type 'datetime' required.")
        self["started"] = localize_datetime(dt)

    def is_started(self):
        # type: () -> bool
        return self.started is not None

    def mark_started(self):
        # type: () -> None
        if not self.is_started():
            setattr(self, "started", now())

    def mark_finished(self):
        # type: () -> None
        if not self.is_finished():
            setattr(self, "finished", now())

    @property
    def duration(self):
        # type: () -> AnyStr
        if self.is_started():
            final_time = self.finished or now()
            duration = localize_datetime(final_time) - localize_datetime(self.started)
            dict.__setitem__(self, "duration", str(duration).split(".")[0])
        else:
            dict.__setitem__(self, "duration", None)
        return dict.__getitem__(self, "duration")

    @property
    def progress(self):
        # type: () -> Number
        return self.get("progress", 0)

    @progress.setter
    def progress(self, progress):
        # type: (Number) -> None
        if not isinstance(progress, (int, float)):
            raise TypeError("Number is required for '{}.progress'".format(type(self)))
        if progress < 0 or progress > 100:
            raise ValueError("Value must be in range [0,100] for '{}.progress'".format(type(self)))
        self["progress"] = progress

    def _get_results(self):
        # type: () -> List[JSON]
        if self.get("results") is None:
            self["results"] = list()
        return self["results"]

    def _set_results(self, results):
        # type: (List[JSON]) -> None
        _check_io_format(results)
        dict.__setitem__(self, "results", results)

    # allows to correctly update list by ref using 'job.results.extend()'
    results = property(_get_results, _set_results)

    def _get_exceptions(self):
        # type: () -> List[JSON]
        if self.get("exceptions") is None:
            self["exceptions"] = list()
        return self["exceptions"]

    def _set_exceptions(self, exceptions):
        # type: (List[JSON]) -> None
        if not isinstance(exceptions, list):
            raise TypeError("Type 'list' is required for '{}.exceptions'".format(type(self)))
        self["exceptions"] = exceptions

    # allows to correctly update list by ref using 'job.exceptions.extend()'
    exceptions = property(_get_exceptions, _set_exceptions)

    def _get_logs(self):
        # type: () -> List[JSON]
        if self.get("logs") is None:
            self["logs"] = list()
        return self["logs"]

    def _set_logs(self, logs):
        # type: (List[JSON]) -> None
        if not isinstance(logs, list):
            raise TypeError("Type 'list' is required for '{}.logs'".format(type(self)))
        self["logs"] = logs

    # allows to correctly update list by ref using 'job.logs.extend()'
    logs = property(_get_logs, _set_logs)

    def _get_tags(self):
        # type: () -> List[AnyStr]
        if self.get("tags") is None:
            self["tags"] = list()
        return self["tags"]

    def _set_tags(self, tags):
        # type: (List[AnyStr]) -> None
        if not isinstance(tags, list):
            raise TypeError("Type 'list' is required for '{}.tags'".format(type(self)))
        self["tags"] = tags

    # allows to correctly update list by ref using 'job.tags.extend()'
    tags = property(_get_tags, _set_tags)

    @property
    def request(self):
        # type: () -> Optional[AnyStr]
        """XML request for WPS execution submission as string."""
        return self.get("request", None)

    @request.setter
    def request(self, request):
        # type: (Optional[AnyStr]) -> None
        """XML request for WPS execution submission as string."""
        self["request"] = request

    @property
    def response(self):
        # type: () -> Optional[AnyStr]
        """XML status response from WPS execution submission as string."""
        return self.get("response", None)

    @response.setter
    def response(self, response):
        # type: (Optional[AnyStr]) -> None
        """XML status response from WPS execution submission as string."""
        self["response"] = response

    @property
    def params(self):
        # type: () -> ParamsType
        return self._get_params({
            "task": self.task,
            "service": self.service,
            "process": self.process,
            "inputs": self.inputs,
            "status": self.status.name,
            "status_message": self.status_message,
            "status_location": self.status_location,
            "execute_async": self.execute_async,
            "is_workflow": self.is_workflow,
            "started": self.started,
            "duration": self.duration,
            "progress": self.progress,
            "results": self.results,
            "exceptions": self.exceptions,
            "logs": self.logs,
            "tags": self.tags,
            "request": self.request,
            "response": self.response,
        })

    def json(self):
        # type: () -> JSON
        return self._get_json({
            "task": self.task,
            "service": self.service,
            "process": self.process,
            "inputs": self.inputs,
            "status": self.status.value,
            "status_message": self.status_message,
            "status_location": self.status_location,
            "execute_async": self.execute_async,
            "is_workflow": self.is_workflow,
            "started": datetime2str(self.started) if self.started else None,
            "duration": self.duration,
            "progress": self.progress,
            "tags": self.tags,
        })

    def summary(self):
        # type: () -> JSON
        return {
            "uuid": self.uuid,
            "process": self.process,
        }


class Action(Base, WithUser):
    """
    Dictionary that contains an action description for db storage.
    It always has ``uuid``, ``type`` and ``operation`` keys.
    """

    def __init__(self, *args, **kwargs):
        super(Action, self).__init__(*args, **kwargs)
        if "uuid" not in self:
            raise TypeError("Action 'uuid' is required.")
        if "type" not in self:
            raise TypeError("Action 'type' is required.")
        if "operation" not in self:
            raise TypeError("Action 'operation' is required.")
        # enforce type conversions
        for field in ["type", "operation"]:
            setattr(self, field, self[field])

    @staticmethod
    def _is_action_type(_type):
        return ((isclass(_type) and issubclass(_type, Base) and _type not in [Base, Action]) or
                (isinstance(_type, Base) and type(_type) not in [Base, Action]))

    @staticmethod
    def _to_action_type(_type):
        if isinstance(_type, six.string_types):
            for _class in [Dataset, Job, Model, Process]:
                if _class.__name__ == _type:
                    return _class
        return _type

    @property
    def type(self):
        # type: () -> Union[Base, Type[Base]]
        """Type of item affected by the action."""
        # enforce conversion in case loaded from db as string
        setattr(self, "type", self.get("type"))
        return dict.__getitem__(self, "type")

    @type.setter
    def type(self, _type):
        # type: (Union[Base, Type[Base]]) -> None
        _type = self._to_action_type(_type)
        if not self._is_action_type(_type):
            raise TypeError("Class or instance derived from 'Base' required.")
        # add 'item' automatically if not explicitly provided and is available
        if isclass(_type):
            dict.__setitem__(self, "type", _type)
        else:
            dict.__setitem__(self, "type", type(_type))
            dict.__setitem__(self, "item", self.item or _type.uuid)

    @property
    def item(self):
        # type: () -> Optional[UUID]
        """Reference to a specific item affected by the action."""
        return self.get("item", None)

    @item.setter
    def item(self, item):
        # type: (Optional[UUID]) -> None
        if item is not None and not is_uuid(item):
            raise TypeError("Item of type 'UUID' required.")
        self["item"] = str(item)

    # noinspection PyTypeChecker
    @property
    def operation(self):
        # type: () -> OPERATION
        """Operation accomplished by the action."""
        return OPERATION.get(dict.__getitem__(self, "operation"))

    @operation.setter
    def operation(self, operation):
        # type: (Union[OPERATION, AnyStr]) -> None
        if isinstance(operation, six.string_types):
            operation = OPERATION.get(operation)
        if not isinstance(operation, OPERATION):
            raise TypeError("Type 'OPERATION' required.")
        dict.__setitem__(self, "operation", operation)

    @property
    def path(self):
        # type: () -> Optional[AnyStr]
        """Request path on with the action was accomplished."""
        return self.get("path", None)

    @path.setter
    def path(self, path):
        # type: (Optional[AnyStr]) -> None
        if not isinstance(path, six.string_types) or path is None:
            raise TypeError("Type 'str' required.")
        dict.__setitem__(self, "path", path)

    @property
    def method(self):
        # type: () -> Optional[AnyStr]
        """Request path on with the action was accomplished."""
        return self.get("method", None)

    @method.setter
    def method(self, method):
        # type: (Optional[AnyStr]) -> None
        if not isinstance(method, six.string_types) or method is None:
            raise TypeError("Type 'str' required.")
        dict.__setitem__(self, "method", method)

    @property
    def params(self):
        # type: () -> ParamsType
        return self._get_params({
            "type": self.type.__name__,
            "item": self.item,
            "path": self.path,
            "method": self.method,
            "operation": self.operation.name,
        })
