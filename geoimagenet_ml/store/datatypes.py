#!/usr/bin/env python
# coding: utf-8

from geoimagenet_ml.constants import OPERATION
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
from geoimagenet_ml.store.exceptions import ModelLoadingError
from geoimagenet_ml.store.exceptions import ProcessInstanceError
from geoimagenet_ml.ml.impl import load_model
from pyramid_celery import celery_app as app
from pywps import Process as ProcessWPS
# noinspection PyPackageRequirements
from dateutil.parser import parse
from datetime import datetime       # noqa: F401
from zipfile import ZipFile
import json
import boltons.tbutils
import logging
import shutil
import uuid
import six
import os
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # noinspection PyPackageRequirements
    from geoimagenet_ml.typedefs import (   # noqa: F401
        AnyStr, AnyStatus, ErrorType, LevelType, List, LoggerType, Number, Union,
        Optional, InputType, OutputType, UUID, JSON, OptionType, Type,
    )


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
    """
    def __init__(self, *args, **kwargs):
        super(Base, self).__init__(*args, **kwargs)
        if "uuid" not in self:
            setattr(self, "uuid", uuid.uuid4())

    def __setattr__(self, item, value):
        # use the existing property setter if defined
        prop = getattr(type(self), item)
        if isinstance(prop, property) and prop.fset is not None:
            # noinspection PyArgumentList
            prop.fset(self, value)
        elif item in self:
            self[item] = value
        else:
            raise AttributeError("Can't set attribute '{}'.".format(item))

    def __getattr__(self, item):
        # use existing property getter if defined
        prop = getattr(type(self), item)
        if isinstance(prop, property) and prop.fget is not None:
            # noinspection PyArgumentList
            return prop.fget(self, item)
        elif item in self:
            return self[item]
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

    @property
    def uuid(self):
        # type: () -> UUID
        return self["uuid"]

    @uuid.setter
    def uuid(self, _uuid):
        # type: (UUID) -> None
        if not is_uuid(_uuid):
            raise ValueError("Not a valid UUID: {!s}.".format(_uuid))
        self["uuid"] = str(_uuid)

    @property
    def created(self):
        # type: () -> datetime
        created = self.get("created")
        if not created:
            setattr(self, "created", now())
        if isinstance(created, six.string_types):
            setattr(self, "created", parse(self["created"]))
        return self.get("created")

    @created.setter
    def created(self, dt):
        # type: (datetime) -> None
        if not isinstance(dt, datetime):
            raise TypeError("Type 'datetime' expected.")
        self["created"] = localize_datetime(dt)


class WithUser(dict):
    """Adds properties ``user`` to corresponding class."""
    @property
    def user(self):
        # type: () -> Optional[int]
        return self.get("user", None)

    @user.setter
    def user(self, user):
        # type: (Optional[int]) -> None
        if not isinstance(user, int) or user is None:
            raise TypeError("Type 'int' is required for '{}.user'".format(type(self)))
        self["user"] = user


class Dataset(Base, WithUser):
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
    def name(self):
        # type: () -> AnyStr
        return self["name"]

    @name.setter
    def name(self, name):
        # type: (AnyStr) -> None
        self["name"] = name

    @property
    def path(self):
        # type: () -> AnyStr
        """Retrieves the dataset path. Automatically creates the directory if not overridden during initialization."""
        if not self.get("path"):
            settings = get_settings(app)
            dataset_root = str(settings["geoimagenet_ml.ml.datasets_path"])
            if not os.path.isdir(dataset_root):
                raise RuntimeError("cannot find datasets root path")
            self["path"] = os.path.join(dataset_root, self.uuid)
            os.makedirs(self["path"], exist_ok=False, mode=0o744)
        return self["path"]

    @path.setter
    def path(self, path):
        # type: (AnyStr) -> None
        if not os.path.isdir(path):
            raise ValueError("Dataset path must be an existing directory.")
        self["path"] = path

    def reset_path(self):
        """Clear all the 'path' content as regenerates a clean directory state."""
        if isinstance(self.path, six.string_types) and os.path.isdir(self.path):
            shutil.rmtree(self.path)
        self["path"] = None
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
    def type(self):
        # type: () -> AnyStr
        return self["type"]

    @type.setter
    def type(self, _type):
        # type: (AnyStr) -> None
        self["type"] = _type

    @property
    def status(self):
        # type: () -> AnyStr
        status = self.get("status")
        if not status:
            status = STATUS.RUNNING
            setattr(self, "status", status)
        if isinstance(status, STATUS):
            status = status.value
        return status

    @status.setter
    def status(self, status):
        # type: (AnyStatus) -> None
        status = map_status(status, COMPLIANT.LITERAL)
        if not isinstance(status, STATUS):
            raise TypeError("Type 'STATUS' enum is expected.")
        if status == STATUS.UNKNOWN:
            raise ValueError("Unknown status not allowed.")
        self["status"] = status.value

    @property
    def finished(self):
        # type: () -> Optional[datetime]
        finished = self["finished"]
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

    def mark_finished(self):
        # type: () -> None
        setattr(self, "finished", now())
        setattr(self, "status", STATUS.FINISHED)

    @property
    def data(self):
        # type: () -> JSON
        """Raw data contained in the dataset definition."""
        return self["data"]

    @data.setter
    def data(self, data):
        # type: (Optional[JSON]) -> None
        self["data"] = data

    @property
    def files(self):
        # type: () -> List[AnyStr]
        """All files referenced by the dataset."""
        return self["files"]

    @files.setter
    def files(self, files):
        # type: (List[AnyStr]) -> None
        if not isinstance(files, list):
            raise TypeError("Type 'list' required.")
        self["files"] = files

    @property
    def params(self):
        # type: () -> JSON
        return {
            "uuid": self.uuid,
            "user": self.user,
            "name": self.name,
            "path": self.path,
            "type": self.type,
            "data": self.data,
            "files": self.files,
            "status": self.status,
            "created": self.created,
            "finished": self.finished,
        }

    def json(self):
        # type: () -> JSON
        return {
            "uuid": self.uuid,
            "name": self.name,
            "path": self.path,
            "type": self.type,
            "data": self.data,
            "files": self.files,
            "status": self.status,
            "created": datetime2str(self.created) if self.created else None,
            "finished": datetime2str(self.finished) if self.finished else None,
        }

    def summary(self):
        # type: () -> JSON
        return {
            "uuid": self.uuid,
            "name": self.name,
        }


class Model(Base, WithUser):
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
        self["created"] = datetime2str(self.created)

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
        # type: () -> Optional[OptionType]
        """
        Retrieve the model's data from the stored file.
        Data can be
        """
        if self["data"] is not None:
            return self["data"]
        if self.file:
            success, data, buffer, exception = load_model(self.file)
            if exception:
                raise exception
            if not success:
                raise ModelLoadingError("Failed retrieving model data.")
            return data
        return None

    @property
    def params(self):
        # type: () -> OptionType
        return {
            "uuid": self.uuid,
            "user": self.user,
            "name": self.name,
            "path": self.path,
            "file": self.file,
            "created": self.created,
        }

    def json(self):
        # type: () -> JSON
        return {
            "uuid": self.uuid,
            "name": self.name,
            "path": self.path,
            "created": datetime2str(self.created),
        }

    def summary(self):
        # type: () -> JSON
        return {
            "uuid": self.uuid,
            "name": self.name,
        }


class Process(Base, WithUser):
    """
    Dictionary that contains a process description for db storage.
    It always has ``uuid`` and ``identifier`` keys.
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

    # wps, workflow, etc.
    @property
    def type(self):
        # type: () -> AnyStr
        return self["type"]

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
        return self.get("inputs")

    @inputs.setter
    def inputs(self, inputs):
        _check_io_format(inputs)
        self["inputs"] = inputs

    @property
    def outputs(self):
        # type: () -> List[OutputType]
        return self.get("outputs")

    @outputs.setter
    def outputs(self, outputs):
        _check_io_format(outputs)
        self["outputs"] = outputs

    @property
    def execute_endpoint(self):
        # type: () -> AnyStr
        return self.get("execute_endpoint")

    @property
    def package(self):
        # type: () -> OptionType
        return self.get("package")

    @property
    def limit_single_job(self):
        # type: () -> bool
        return self.get("limit_single_job", False)

    @property
    def params(self):
        # type: () -> OptionType
        return {
            "uuid": self.uuid,
            "user": self.user,
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
            "type": self.type,
            "package": self.package,      # deployment specification (json body)
        }

    @property
    def params_wps(self):
        # type: () -> OptionType
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
        return {
            "uuid": self.uuid,
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
        }

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
            raise ProcessInstanceError("Invalid WPS process call for '{}' of type '{}'.".format(process_key, self.type))
        if process_key not in process_mapping:
            raise ProcessInstanceError("Unknown process '{}' in mapping".format(process_key))
        kwargs = self.params_wps
        return process_mapping[process_key](**kwargs)


class Job(Base, WithUser):
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
            self["inputs"] = list()
        return self["inputs"]

    def _set_inputs(self, inputs):
        # type: (List[JSON]) -> None
        if not isinstance(inputs, list):
            raise TypeError("Type 'list' is required for '{}.inputs'".format(type(self)))
        self["inputs"] = inputs

    # allows to correctly update list by ref using 'job.inputs.extend()'
    inputs = property(_get_inputs, _set_inputs)

    @property
    def status(self):
        # type: () -> AnyStr
        return self.get("status", STATUS.UNKNOWN.value)

    @status.setter
    def status(self, status):
        # type: (STATUS) -> None
        status = map_status(status, COMPLIANT.LITERAL)
        if not isinstance(status, STATUS):
            raise TypeError("Type 'STATUS' enum is expected.")
        if status == STATUS.UNKNOWN:
            raise ValueError("Unknown status not allowed.")
        self["status"] = status.value
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

    def mark_finished(self):
        # type: () -> None
        setattr(self, "finished", now())

    @property
    def duration(self):
        # type: () -> AnyStr
        final_time = self.finished or now()
        duration = localize_datetime(final_time) - localize_datetime(self.created)
        self["duration"] = str(duration).split(".")[0]
        return self["duration"]

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
        self["results"] = results

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
        # type: () -> JSON
        return {
            "uuid": self.uuid,
            "task": self.task,
            "service": self.service,
            "process": self.process,
            "inputs": self.inputs,
            "user": self.user,
            "status": self.status,
            "status_message": self.status_message,
            "status_location": self.status_location,
            "execute_async": self.execute_async,
            "is_workflow": self.is_workflow,
            "created": self.created,
            "finished": self.finished,
            "duration": self.duration,
            "progress": self.progress,
            "results": self.results,
            "exceptions": self.exceptions,
            "logs": self.logs,
            "tags": self.tags,
            "request": self.request,
            "response": self.response,
        }

    def json(self):
        # type: () -> JSON
        return {
            "uuid": self.uuid,
            "task": self.task,
            "service": self.service,
            "process": self.process,
            "user": self.user,
            "inputs": self.inputs,
            "status": self.status,
            "status_message": self.status_message,
            "status_location": self.status_location,
            "execute_async": self.execute_async,
            "is_workflow": self.is_workflow,
            "created": datetime2str(self.created) if self.created else None,
            "finished": datetime2str(self.finished) if self.finished else None,
            "duration": self.duration,
            "progress": self.progress,
            "tags": self.tags,
        }

    def summary(self):
        # type: () -> JSON
        return {
            "uuid": self.uuid,
            "process": self.process,
        }


class Action(Base):
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
        return self["type"]

    @type.setter
    def type(self, _type):
        # type: (Union[Base, Type[Base]]) -> None
        _type = self._to_action_type(_type)
        if not self._is_action_type(_type):
            raise TypeError("Class or instance derived from 'Base' required.")
        # add 'item' automatically if not explicitly provided and is available
        if isclass(_type):
            self["type"] = _type
        else:
            self["type"] = type(_type)
            setattr(self, "item", self.item or _type.uuid)

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
        return OPERATION.get(self["operation"])

    @operation.setter
    def operation(self, operation):
        # type: (Union[OPERATION, AnyStr]) -> None
        if isinstance(operation, six.string_types):
            operation = OPERATION.get(operation)
        if operation not in OPERATION:
            raise TypeError("Type 'OPERATION' required.")
        self["operation"] = operation.value

    @property
    def user(self):
        # type: () -> Optional[int]
        """User that accomplished the action."""
        return self.get("user", None)

    @user.setter
    def user(self, user):
        # type: (Optional[int]) -> None
        if not isinstance(user, int) and user is not None:
            raise TypeError("Type 'int' required.")
        self["user"] = user

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
        self["path"] = path

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
        self["method"] = method

    @property
    def params(self):
        # type: () -> JSON
        return {
            "uuid": self.uuid,
            "type": self.type.__name__,
            "item": self.item,
            "user": self.user,
            "path": self.path,
            "method": self.method,
            "operation": self.operation.name,
            "created": self.created,
        }
