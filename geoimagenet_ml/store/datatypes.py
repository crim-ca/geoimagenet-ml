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
    assert_sane_name,
    clean_json_text_body,
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
from distutils.version import StrictVersion
from zipfile import ZipFile
# noinspection PyPackageRequirements
from bson import ObjectId
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
        Any, AnyStr, AnyStatus, AnyUUID, Callable, Dict, ErrorType, LevelType, List, LoggerType, Number, Iterable,
        Union, Optional, InputType, OutputType, JSON, JsonValue, ParamsType, Type,
    )
    from pywps import Process as ProcessWPS  # noqa: F401


class Validator(object):

    def _is_of_type(self,
                    param_name,             # type: AnyStr
                    param_value,            # type: Any
                    param_types=None,       # type: Union[Type, Iterable[Type]]
                    allow_none=False,       # type: bool
                    sub_item=None,          # type: Optional[Union[Type, Iterable[Type]]]
                    sub_key=None,           # type: Optional[Union[Type, Iterable[Type]]]
                    sub_value=None,         # type: Optional[Union[Type, Iterable[Type]]]
                    valid_function=None,    # type: Optional[Callable[[Any], bool]]
                    ):                      # type: (...) -> None
        """
        Validates that ``param_value`` is one of the type(s) specified by ``param_types``.

        :param: param_name: name to display in the appropriate error message in case of invalid type.
        :param param_value: value to be evaluated for typing.
        :param param_types: type(s) for which the value must be validated. Also used for displaying appropriate message.
        :param allow_none: used to allow 'None' type, and adjust the message accordingly.
        :param sub_item: (optional) validates also that sub-item(s) type of a list/set/tuple `param_value` is respected.
        :param sub_key: (optional) validates also that sub-key(s) type of a dict `param_value` is respected.
        :param sub_value: (optional) validates also that sub-value(s) type of a dict `param_value` is respected.
        :param valid_function: (optional) alternative function to call for `param_value` validation.
        :raises TypeError: if type specification is not met.
        """
        # convert for backward compatibility and simplified calls with 'str'
        if param_types is str:
            param_types = six.string_types
        elif param_types != six.string_types and isinstance(param_types, (list, set, tuple)) and str in param_types:
            param_types = list(filter(lambda t: t is not str, param_types)) + list(six.string_types)

        # parameter validation
        if allow_none and param_value is None:
            return
        if not callable(valid_function):
            valid_param_type = tuple(param_types) if isinstance(param_types, (list, set)) else param_types
            valid_type = isinstance(param_value, valid_param_type)
        else:
            valid_type = valid_function(param_value)
        if not valid_type:
            if param_types == six.string_types:
                param_types = str  # convert back for display
            if param_types in (list, set, tuple):
                param_types = [t.__name__ for t in param_types]
                param_qualifier = "One of type"
            else:
                param_types = param_types.__name__
                param_qualifier = "Type"
            param_none = "or 'None' " if allow_none else ""
            raise TypeError("{} '{!s}' {}required for '{}.{}'.".format(
                param_qualifier, param_types, param_none, type(self), param_name)
            )

        # sub-element validation
        if sub_item is not None and isinstance(param_value, (list, set, tuple)):
            for param_item in param_value:
                self._is_of_type("{}[<item>]".format(param_name), param_item, sub_item)
        if sub_key is not None and isinstance(param_value, dict):
            for param_key in param_value.keys():
                self._is_of_type("{}[<key>]".format(param_name), param_key, sub_key)
        if sub_value is not None and isinstance(param_value, dict):
            for param_val in param_value.values():
                self._is_of_type("{}[<value>]".format(param_name), param_val, sub_value)


class Base(dict, Validator):
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

    Note that calling ``self.get(...)`` and ``self.setdefault(...)`` is also safe since they directly refer to the
    underlying dictionary methods (unless overridden in another parent class).

    To make a property `read-only`, it is possible to only define the ``@property`` method without its corresponding
    ``@<field>.setter` method. By doing so, both ``Base["<field>"] = value`` and ``Base.<field> = value`` will raise
    an ``AttributeError``.
    """
    def __init__(self, *args, **kwargs):
        super(Base, self).__init__()
        # apply any property check or field value adjustment during assignment
        for arg in args:
            kwargs.update(arg)
        for k, v in kwargs.items():
            self[k] = v
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
    def _id(self):
        # type: () -> Optional[ObjectId]
        """Internal ID assigned to this datatype by the db storage."""
        return self.get("_id")

    @_id.setter
    def _id(self, _id):
        # type: (ObjectId) -> None
        self._is_of_type("_id", _id, ObjectId)
        dict.__setitem__(self, "_id", _id)

    @property
    def uuid(self):
        # type: () -> AnyUUID
        return dict.__getitem__(self, "uuid")

    @uuid.setter
    def uuid(self, _uuid):
        # type: (AnyUUID) -> None
        self._is_of_type("uuid", _uuid, uuid.UUID, valid_function=is_uuid)
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


class WithFinished(dict, Validator):
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
        self._is_of_type("finished", dt, datetime, allow_none=True)
        dict.__setitem__(self, "finished", localize_datetime(dt) if dt else None)

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


class WithName(dict, Validator):
    """Adds property ``name`` to corresponding class."""

    @property
    def name(self):
        # type: () -> AnyStr
        return dict.__getitem__(self, "name")

    @name.setter
    def name(self, name):
        # type: (AnyStr) -> None
        self._is_of_type("name", name, str)
        dict.__setitem__(self, "name", name)

    @property
    def params(self):
        # type: () -> ParamsType
        return {"name": self.name}

    def json(self):
        # type: () -> JSON
        return {"name": self.name}


class WithType(dict, Validator):
    """Adds property ``type`` to corresponding class."""
    @property
    def type(self):
        # type: () -> AnyStr
        return dict.__getitem__(self, "type")

    @type.setter
    def type(self, _type):
        # type: (AnyStr) -> None
        self._is_of_type("type", _type, str)
        dict.__setitem__(self, "type", _type)

    @property
    def params(self):
        # type: () -> ParamsType
        return {"type": self.type}

    def json(self):
        # type: () -> JSON
        return {"type": self.type}


class WithUser(dict, Validator):
    """Adds property ``user`` to corresponding class."""
    @property
    def user(self):
        # type: () -> Optional[int]
        return self.get("user", None)

    @user.setter
    def user(self, user):
        # type: (Optional[int]) -> None
        self._is_of_type("user", user, int, allow_none=True)
        dict.__setitem__(self, "user", user)

    @property
    def params(self):
        # type: () -> ParamsType
        return {"user": self.user}

    def json(self):
        # type: () -> JSON
        return {"user": self.user}


class WithVisibility(dict, Validator):
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
        self._is_of_type("visibility", visibility, VISIBILITY)
        dict.__setitem__(self, "visibility", visibility)

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
    def path(self):
        # type: () -> AnyStr
        """Original path specified for model creation."""
        return dict.__getitem__(self, "path")

    @path.setter
    def path(self, path):
        # type: (AnyStr) -> None
        self._is_of_type("path", path, str)
        dict.__setitem__(self, "path", path)

    @property
    def format(self):
        # type: () -> AnyStr
        """Original file format (extension)."""
        return os.path.splitext(self.path)[-1]

    @format.setter
    def format(self, fmt):
        # type: (AnyStr) -> None
        self._is_of_type("format", fmt, str)
        dict.__setitem__(self, "format", fmt)

    @property
    def file(self):
        # type: () -> AnyStr
        """Stored file path of the created model."""
        return self.get("file")

    @file.setter
    def file(self, file):
        # type: (AnyStr) -> None
        self._is_of_type("file", file, str)
        dict.__setitem__(self, "file", file)

    @property
    def data(self):
        # type: () -> Optional[ParamsType]
        """
        Retrieve the model's data config from the stored file.
        If data was already loaded and cached in the object, it is returned instead.
        """
        if self.get("data") is not None:
            return dict.__getitem__(self, "data")
        if self.file:
            data, _ = self._load_check_data(self.file)
            dict.__setitem__(self, "data", data)
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

        # noinspection PyCallByClass
        def _reset_file_link():
            dict.__setitem__(self, "file", location)
            dict.__setitem__(self, "data", None)

        if self.file and os.path.isfile(self.file) and self.file != location:
            if os.path.isfile(location):
                raise ex.ModelConflictError("Model save location already exists.")
            shutil.copyfile(self.file, location)
            _reset_file_link()
            return

        def _write_buffer(_buffer):
            # transfer loaded data buffer to storage file
            with open(location, 'wb') as model_file:
                _buffer.seek(0)
                model_file.write(_buffer.read())
                _buffer.close()

        if isinstance(self.data, io.BufferedIOBase):
            _write_buffer(self["data"])
            _reset_file_link()
            return

        if not self.file and not self.data and self.path:
            _, buffer = self._load_check_data(self.path)
            _write_buffer(buffer)
            _reset_file_link()
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
        return dict.__getitem__(self, "identifier")

    @identifier.setter
    def identifier(self, identifier):
        # type: (AnyStr) -> None
        self._is_of_type("identifier", identifier, str)
        assert_sane_name(identifier)
        dict.__setitem__(self, "identifier", identifier)

    @property
    def title(self):
        # type: () -> AnyStr
        return self.get("title", self.identifier)

    @title.setter
    def title(self, title):
        self._is_of_type("title", title, str)
        dict.__setitem__(self, "title", title)

    @property
    def abstract(self):
        # type: () -> AnyStr
        return self.get("abstract", "")

    @abstract.setter
    def abstract(self, abstract):
        # type: (AnyStr) -> None
        self._is_of_type("abstract", abstract, str)
        dict.__setitem__(self, "abstract", clean_json_text_body(abstract))

    @property
    def keywords(self):
        # type: () -> List[AnyStr]
        return self.get("keywords", [])

    @keywords.setter
    def keywords(self, keywords):
        if not isinstance(keywords, list) or not all(isinstance(k, six.string_types) for k in keywords):
            raise TypeError("Type 'list[str]' required for '{}.keywords'.".format(type(self)))
        dict.__setitem__(self, "keywords", keywords)

    @property
    def metadata(self):
        # type: () -> List[AnyStr]
        return self.get("metadata", [])

    @metadata.setter
    def metadata(self, metadata):
        # type: (List[Dict[AnyStr, JsonValue]]) -> None
        self._is_of_type("metadata", metadata, list, sub_item=dict)
        for metainfo in metadata:
            self._is_of_type("metadata[<item>]", metainfo, sub_key=str, sub_value=(float, int, bool, str))
        dict.__setitem__(self, "metadata", metadata)

    @property
    def version(self):
        # type: () -> AnyStr
        return self.get("version")

    @version.setter
    def version(self, version):
        # type: (AnyStr) -> None
        self._is_of_type("version", version, str, allow_none=True)
        try:
            StrictVersion(version)
        except ValueError:
            raise ValueError("Invalid version '{!s}' for '{}.version'.".format(version, type(self)))
        dict.__setitem__(self, "version", version)

    @property
    def inputs(self):
        # type: () -> List[InputType]
        return self.get("inputs", [])

    @inputs.setter
    def inputs(self, inputs):
        self._is_of_type("inputs", inputs, list)
        for i in inputs:
            self._is_of_type("inputs[<item>]", i, dict, sub_key=str)
        dict.__setitem__(self, "inputs", inputs)

    @property
    def outputs(self):
        # type: () -> List[OutputType]
        return self.get("outputs", [])

    @outputs.setter
    def outputs(self, outputs):
        self._is_of_type("outputs", outputs, list)
        for o in outputs:
            self._is_of_type("outputs[<item>]", o, dict, sub_key=str)
        dict.__setitem__(self, "outputs", outputs)

    @property
    def execute_endpoint(self):
        # type: () -> Optional[AnyStr]
        return self.get("execute_endpoint")

    @execute_endpoint.setter
    def execute_endpoint(self, location):
        # type: (AnyStr) -> None
        self._is_of_type("execute_endpoint", location, str)
        if not location.startswith("http"):
            raise ValueError("Field 'execute_endpoint' must be an HTTP(S) location.")
        dict.__setitem__(self, "execute_endpoint", location)

    @property
    def package(self):
        # type: () -> Optional[JSON]
        return self.get("package", None)

    @package.setter
    def package(self, package):
        # type: (Optional[JSON]) -> None
        self._is_of_type("package", package, param_types=dict, sub_key=str, allow_none=True)
        dict.__setitem__(self, "package", package)

    @property
    def reference(self):
        # type: () -> Optional[AnyStr]
        return self.get("reference", None)

    @reference.setter
    def reference(self, reference):
        # type: (Optional[AnyStr]) -> None
        self._is_of_type("reference", reference, str, allow_none=True)
        if reference and not reference.startswith("http"):
            raise ValueError("Field 'reference' must be an HTTP(S) location.")
        dict.__setitem__(self, "reference", reference)

    @property
    def limit_single_job(self):
        # type: () -> bool
        if "limit_single_job" not in self:
            setattr(self, "limit_single_job", False)
        return dict.__getitem__(self, "limit_single_job")

    @limit_single_job.setter
    def limit_single_job(self, value):
        self._is_of_type("limit_single_job", value, bool)
        dict.__setitem__(self, "limit_single_job", value)

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
        # type: () -> AnyUUID
        return self.get("task")

    @task.setter
    def task(self, task):
        # type: (AnyUUID) -> None
        self._is_of_type("task", task, uuid.UUID, valid_function=is_uuid, allow_none=True)
        dict.__setitem__(self, "task", task)

    @property
    def service(self):
        # type: () -> Optional[AnyUUID]
        return self.get("service", None)

    @service.setter
    def service(self, service):
        # type: (Optional[AnyUUID]) -> None
        self._is_of_type("service", service, uuid.UUID, valid_function=is_uuid, allow_none=True)
        dict.__setitem__(self, "service", service)

    @property
    def process(self):
        # type: () -> Optional[AnyUUID]
        return self.get("process", None)

    @process.setter
    def process(self, process):
        # type: (AnyUUID) -> None
        self._is_of_type("process", process, uuid.UUID, valid_function=is_uuid)
        dict.__setitem__(self, "process", process)

    def _get_inputs(self):
        # type: () -> List[JSON]
        if self.get("inputs") is None:
            dict.__setitem__(self, "inputs", list())
        return dict.__getitem__(self, "inputs")

    def _set_inputs(self, inputs):
        # type: (List[JSON]) -> None
        self._is_of_type("inputs", inputs, list, sub_item=dict)
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
        self._is_of_type("status", status, STATUS)
        if status == STATUS.UNKNOWN:
            raise ValueError("Unknown status not allowed.")
        dict.__setitem__(self, "status", status)
        if status in job_status_categories[CATEGORY.EXECUTING]:
            self.update_started_datetime()
        if status in job_status_categories[CATEGORY.FINISHED]:
            self.update_finished_datetime()

    @property
    def status_message(self):
        # type: () -> AnyStr
        return self.get("status_message", "no message")

    @status_message.setter
    def status_message(self, message):
        # type: (Optional[AnyStr]) -> None
        if message is None:
            return
        self._is_of_type("status_message", message, str)
        dict.__setitem__(self, "status_message", message)

    @property
    def status_location(self):
        # type: () -> Optional[AnyStr]
        return self.get("status_location", None)

    @status_location.setter
    def status_location(self, location_url):
        # type: (Optional[AnyStr]) -> None
        self._is_of_type("status_location", location_url, str, allow_none=True)
        dict.__setitem__(self, "status_location", location_url)

    @property
    def execute_async(self):
        # type: () -> bool
        return self.get("execute_async", True)

    @execute_async.setter
    def execute_async(self, execute_async):
        # type: (bool) -> None
        self._is_of_type("execute_async", execute_async, bool)
        dict.__setitem__(self, "execute_async", execute_async)

    @property
    def is_workflow(self):
        # type: () -> bool
        return self.get("is_workflow", False)

    @is_workflow.setter
    def is_workflow(self, is_workflow):
        # type: (bool) -> None
        self._is_of_type("is_workflow", is_workflow, bool)
        dict.__setitem__(self, "is_workflow", is_workflow)

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
    def started(self, started):
        # type: (Union[AnyStr, datetime]) -> None
        self._is_of_type("started", started, datetime, allow_none=True)
        dict.__setitem__(self, "started", localize_datetime(started) if started else None)

    def is_started(self):
        # type: () -> bool
        return self.started is not None

    def update_started_datetime(self):
        # type: () -> None
        if not self.is_started():
            setattr(self, "started", now())

    def update_finished_datetime(self):
        # type: () -> None
        if not self.is_finished():
            setattr(self, "finished", now())

    @property
    def duration(self):
        # type: () -> Optional[AnyStr]
        """
        Execution time since ``started``, up to ``finished`` if job was marked as finished.
        If job has not yet started, returns ``None``.
        """
        if self.is_started():
            final_time = self.finished or now()
            duration = localize_datetime(final_time) - localize_datetime(self.started)
            return str(duration).split(".")[0]
        return None

    @property
    def progress(self):
        # type: () -> Number
        return self.get("progress", 0)

    @progress.setter
    def progress(self, progress):
        # type: (Number) -> None
        self._is_of_type("progress", progress, (int, float))
        if progress < 0 or progress > 100:
            raise ValueError("Value must be in range [0,100] for '{}.progress'".format(type(self)))
        dict.__setitem__(self, "progress", progress)

    @staticmethod
    def _check_results_io_format(io_items):
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

    def _get_results(self):
        # type: () -> List[JSON]
        if self.get("results") is None:
            self.results = []
        return dict.__getitem__(self, "results")

    def _set_results(self, results):
        # type: (List[JSON]) -> None
        self._check_results_io_format(results)
        dict.__setitem__(self, "results", results)

    # allows to correctly update list by ref using 'job.results.extend()'
    results = property(_get_results, _set_results)

    def _get_exceptions(self):
        # type: () -> List[Union[JSON, AnyStr]]
        if self.get("exceptions") is None:
            self.exceptions = []
        return dict.__getitem__(self, "exceptions")

    def _set_exceptions(self, exceptions):
        # type: (List[Union[JSON, AnyStr]]) -> None
        self._is_of_type("exceptions", exceptions, list, sub_item=(dict, str))
        dict.__setitem__(self, "exceptions", exceptions)

    # allows to correctly update list by ref using 'job.exceptions.extend()'
    exceptions = property(_get_exceptions, _set_exceptions)

    def _get_logs(self):
        # type: () -> List[AnyStr]
        if self.get("logs") is None:
            self.logs = []
        return dict.__getitem__(self, "logs")

    def _set_logs(self, logs):
        # type: (List[AnyStr]) -> None
        self._is_of_type("logs", logs, list, sub_item=str)
        dict.__setitem__(self, "logs", logs)

    # allows to correctly update list by ref using 'job.logs.extend()'
    logs = property(_get_logs, _set_logs)

    def _get_tags(self):
        # type: () -> List[AnyStr]
        if self.get("tags") is None:
            self.tags = []
        return dict.__getitem__(self, "tags")

    def _set_tags(self, tags):
        # type: (List[AnyStr]) -> None
        self._is_of_type("tags", tags, list, sub_item=str)
        dict.__setitem__(self, "tags", tags)

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
        dict.__setitem__(self, "request", request)

    @property
    def response(self):
        # type: () -> Optional[AnyStr]
        """XML status response from WPS execution submission as string."""
        return self.get("response", None)

    @response.setter
    def response(self, response):
        # type: (Optional[AnyStr]) -> None
        """XML status response from WPS execution submission as string."""
        dict.__setitem__(self, "response", response)

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
        # type: () -> Optional[AnyUUID]
        """Reference to a specific item affected by the action."""
        return self.get("item", None)

    @item.setter
    def item(self, item):
        # type: (Optional[AnyUUID]) -> None
        self._is_of_type("item", item, uuid.UUID, valid_function=is_uuid, allow_none=True)
        dict.__setitem__(self, "item", str(item) if item else None)

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
        self._is_of_type("operation", operation, OPERATION)
        dict.__setitem__(self, "operation", operation)

    @property
    def path(self):
        # type: () -> Optional[AnyStr]
        """Request path on with the action was accomplished."""
        return self.get("path", None)

    @path.setter
    def path(self, path):
        # type: (Optional[AnyStr]) -> None
        self._is_of_type("path", path, str, allow_none=True)
        dict.__setitem__(self, "path", path)

    @property
    def method(self):
        # type: () -> Optional[AnyStr]
        """Request path on with the action was accomplished."""
        return self.get("method", None)

    @method.setter
    def method(self, method):
        # type: (Optional[AnyStr]) -> None
        self._is_of_type("method", method, str, allow_none=True)
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
