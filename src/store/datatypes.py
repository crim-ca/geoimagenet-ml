#!/usr/bin/env python
# coding: utf-8

from geoimagenet_ml.utils import (
    now, localize_datetime,
    stringify_datetime,
    get_log_fmt,
    get_log_datefmt,
    get_job_log_msg,
    get_error_fmt,
    fully_qualified_name
)
from geoimagenet_ml.processes.status import (
    job_status_values,
    job_status_categories,
    STATUS_CATEGORY_FINISHED,
    STATUS_UNKNOWN,
)
from geoimagenet_ml.processes.types import process_mapping, PROCESS_WPS
from geoimagenet_ml.store.exceptions import ModelLoadingError
from geoimagenet_ml.store.exceptions import ProcessInstanceError
from geoimagenet_ml.ml.impl import load_model
from pywps import Process as ProcessWPS
# noinspection PyPackageRequirements
from dateutil.parser import parse
from datetime import datetime       # noqa: F401
import boltons.tbutils
import logging
import uuid
import six
import os
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # noinspection PyPackageRequirements
    from geoimagenet_ml.typedefs import (   # noqa: F401
        AnyStr, ErrorType, LevelType, List, LoggerType, Union,
        Optional, InputType, OutputType, UUID, JsonBody, OptionType
    )


class Base(dict):
    def __init__(self, *args, **kwargs):
        super(Base, self).__init__(*args, **kwargs)
        self['uuid'] = self.get('uuid', str(uuid.uuid4()))

    @property
    def uuid(self):
        # type: (...) -> UUID
        return self['uuid']


class Dataset(Base):
    """
    Dictionary that contains a dataset description for db storage.
    It always has ``name`` and ``path`` keys.
    """

    def __init__(self, *args, **kwargs):
        super(Dataset, self).__init__(*args, **kwargs)
        if 'name' not in self:
            raise TypeError("Dataset `name` is required.")
        if 'path' not in self:
            raise TypeError("Dataset `path` is required.")
        if 'type' not in self:
            raise TypeError("Dataset `type` is required.")

    @property
    def name(self):
        # type: (...) -> AnyStr
        return self['name']

    @property
    def path(self):
        # type: (...) -> AnyStr
        return self['path']

    @property
    def type(self):
        # type: (...) -> AnyStr
        return self['type']

    @property
    def parameters(self):
        # type: (...) -> JsonBody
        return self['params']

    @parameters.setter
    def parameters(self, params):
        # type: (Union[None, JsonBody]) -> None
        self['params'] = params

    @property
    def params(self):
        # type: (...) -> JsonBody
        return {
            'uuid': self.uuid,
            'name': self.name,
            'path': self.path,
            'type': self.type,
            'params': self.parameters,
        }

    def json(self):
        # type: (...) -> JsonBody
        return self.params

    def summary(self):
        # type: (...) -> JsonBody
        return {
            'uuid': self.uuid,
            'name': self.name,
        }


class Model(Base):
    """
    Dictionary that contains a model description for db storage.
    It always has ``name`` and ``path`` keys.
    """

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        if 'name' not in self:
            raise TypeError("Model `name` is required.")
        if 'path' not in self:
            raise TypeError("Model `path` is required.")
        self['created'] = stringify_datetime(self.created)

    @property
    def name(self):
        # type: (...) -> AnyStr
        return self['name']

    @property
    def path(self):
        # type: (...) -> AnyStr
        """Original path specified for model creation."""
        return self['path']

    @property
    def format(self):
        # type: (...) -> AnyStr
        """Original file format (extension)."""
        return os.path.splitext(self.path)[-1]

    @property
    def file(self):
        # type: (...) -> AnyStr
        """Stored file path of the created model."""
        return self.get('file')

    @property
    def data(self):
        # type: (...) -> Union[OptionType, None]
        """
        Retrieve the model's data from the stored file.
        Data can be
        """
        if self['data'] is not None:
            return self['data']
        if self.file:
            success, data, buffer, exception = load_model(self.file)
            if exception:
                raise exception
            if not success:
                raise ModelLoadingError("Failed retrieving model data.")
            return data
        return None

    @property
    def created(self):
        # type: (...) -> datetime
        created = self.get('created', None)
        if not created:
            self['created'] = now()
        if isinstance(created, six.string_types):
            self['created'] = parse(self['created'])
        return localize_datetime(self.get('created'))

    @property
    def params(self):
        # type: (...) -> OptionType
        return {
            'uuid': self.uuid,
            'name': self.name,
            'path': self.path,
            'file': self.file,
            'created': self.created,
        }

    def json(self):
        # type: (...) -> JsonBody
        return {
            'uuid': self.uuid,
            'name': self.name,
            'path': self.path,
            'created': stringify_datetime(self.created),
        }

    def summary(self):
        # type: (...) -> JsonBody
        return {
            'uuid': self.uuid,
            'name': self.name,
        }


class Process(Base):
    """
    Dictionary that contains a process description for db storage.
    It always has ``uuid`` and ``identifier`` keys.
    """

    def __init__(self, *args, **kwargs):
        super(Process, self).__init__(*args, **kwargs)
        # use both 'id' and 'identifier' to support any call (WPS and recurrent 'id')
        if 'identifier' not in self:
            raise TypeError("Process `identifier` is required.")
        if 'type' not in self:
            raise TypeError("Process `type` is required.")

    @property
    def identifier(self):
        # type: (...) -> AnyStr
        return self['identifier']

    # wps, workflow, etc.
    @property
    def type(self):
        # type: (...) -> AnyStr
        return self['type']

    @property
    def title(self):
        # type: (...) -> AnyStr
        return self.get('title', self.identifier)

    @property
    def abstract(self):
        # type: (...) -> AnyStr
        return self.get('abstract', '')

    @property
    def keywords(self):
        # type: (...) -> List[AnyStr]
        return self.get('keywords', [])

    @property
    def metadata(self):
        # type: (...) -> List[AnyStr]
        return self.get('metadata', [])

    @property
    def version(self):
        # type: (...) -> AnyStr
        return self.get('version')

    @property
    def inputs(self):
        # type: (...) -> List[InputType]
        return self.get('inputs')

    @property
    def outputs(self):
        # type: (...) -> List[OutputType]
        return self.get('outputs')

    # noinspection PyPep8Naming
    @property
    def jobControlOptions(self):
        # type: (...) -> AnyStr
        return self.get('jobControlOptions')

    # noinspection PyPep8Naming
    @property
    def outputTransmission(self):
        # type: (...) -> AnyStr
        return self.get('outputTransmission')

    # noinspection PyPep8Naming
    @property
    def executeEndpoint(self):
        # type: (...) -> AnyStr
        return self.get('executeEndpoint')

    @property
    def package(self):
        # type: (...) -> OptionType
        return self.get('package')

    def __str__(self):
        # type: (...) -> AnyStr
        return "Process <{0}> [{1}] ({2})".format(self.uuid, self.identifier, self.title)

    def __repr__(self):
        # type: (...) -> AnyStr
        cls = type(self)
        repr_ = dict.__repr__(self)
        return '{0}.{1}({2})'.format(cls.__module__, cls.__name__, repr_)

    @property
    def params(self):
        # type: (...) -> OptionType
        return {
            'uuid': self.uuid,
            'identifier': self.identifier,
            'title': self.title,
            'abstract': self.abstract,
            'keywords': self.keywords,
            'metadata': self.metadata,
            'version': self.version,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'jobControlOptions': self.jobControlOptions,
            'outputTransmission': self.outputTransmission,
            'executeEndpoint': self.executeEndpoint,
            'type': self.type,
            'package': self.package,      # deployment specification (json body)
        }

    @property
    def params_wps(self):
        # type: (...) -> OptionType
        """
        Values applicable to WPS Process __init__
        """
        return {
            'identifier': self.identifier,
            'title': self.title,
            'abstract': self.abstract,
            'keywords': self.keywords,
            'metadata': self.metadata,
            'version': self.version,
            'inputs': self.inputs,
            'outputs': self.outputs,
        }

    def json(self):
        # type: (...) -> JsonBody
        return {
            'uuid': self.uuid,
            'identifier': self.identifier,
            'title': self.title,
            'abstract': self.abstract,
            'keywords': self.keywords,
            'metadata': self.metadata,
            'version': self.version,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'jobControlOptions': self.jobControlOptions,
            'outputTransmission': self.outputTransmission,
            'executeEndpoint': self.executeEndpoint,
        }

    def summary(self):
        # type: (...) -> JsonBody
        return {
            'uuid': self.uuid,
            'identifier': self.identifier,
            'title': self.title,
            'abstract': self.abstract,
            'keywords': self.keywords,
            'metadata': self.metadata,
            'version': self.version,
            'jobControlOptions': self.jobControlOptions,
            'executeEndpoint': self.executeEndpoint,
        }

    def wps(self):
        # type: (...) -> ProcessWPS
        process_key = self.identifier
        if self.type != PROCESS_WPS:
            raise ProcessInstanceError("Invalid WPS process call for `{}` of type `{}`.".format(process_key, self.type))
        if process_key not in process_mapping:
            raise ProcessInstanceError("Unknown process `{}` in mapping".format(process_key))
        kwargs = self.params_wps
        return process_mapping[process_key](**kwargs)


class Job(Base):
    """
    Dictionary that contains a job description for db storage.
    It always has ``uuid`` and ``process_uuid`` keys.
    """

    def __init__(self, *args, **kwargs):
        super(Job, self).__init__(*args, **kwargs)
        if 'uuid' not in self:
            raise TypeError("Job `uuid` is required.")
        if 'process_uuid' not in self:
            raise TypeError("Job `process_uuid` is required.")

    def _get_log_msg(self, msg=None):
        # type: (Optional[AnyStr]) -> AnyStr
        if not msg:
            msg = self.status_message
        return get_job_log_msg(duration=self.duration, progress=self.progress, status=self.status, message=msg)

    def save_log(self, errors=None, logger=None, level=logging.INFO):
        # type: (Optional[ErrorType], Optional[LoggerType], Optional[LevelType]) -> None
        if isinstance(level, six.string_types):
            level = logging.getLevelName(level)
        if isinstance(errors, six.string_types):
            log_msg = [(logging.ERROR, self._get_log_msg())]
            self.exceptions.append(errors)
        elif isinstance(errors, Exception):
            log_msg = [(logging.ERROR, self._get_log_msg())]
            tb_info = boltons.tbutils.TracebackInfo.from_traceback()
            self.exceptions.extend([{
                'func_name': frame['func_name'],
                'line_detail': frame['line'].strip(),
                'line_number': frame['lineno'],
                'module_name': frame['module_name'],
                'module_path': frame['module_path'],
            } for frame in tb_info.to_dict()['frames']])
        elif isinstance(errors, list):
            log_msg = [(logging.ERROR, self._get_log_msg(get_error_fmt().format(error)))
                       for error in errors]
            self.exceptions.extend([{
                'Code': error.code,
                'Locator': error.locator,
                'Text': error.text
            } for error in errors])
        else:
            log_msg = [(level, self._get_log_msg())]
        for lvl, msg in log_msg:
            # noinspection PyProtectedMember
            fmt_msg = get_log_fmt() % dict(asctime=now().strftime(get_log_datefmt()),
                                           levelname=logging.getLevelName(lvl),
                                           name=fully_qualified_name(self),
                                           message=msg)
            if len(self.logs) == 0 or self.logs[-1] != fmt_msg:
                self.logs.append(fmt_msg)
                if logger:
                    logger.log(lvl, msg)

    @property
    def task_uuid(self):
        # type: (...) -> UUID
        return self.get('task_uuid')

    @task_uuid.setter
    def task_uuid(self, task_uuid):
        # type: (Union[None, UUID]) -> None
        if not isinstance(task_uuid, six.string_types) or task_uuid is None:
            raise TypeError("Type `str` is required for `{}.task_uuid`".format(type(self)))
        self['task_uuid'] = task_uuid

    @property
    def service_uuid(self):
        # type: (...) -> Union[None, UUID]
        return self.get('service_uuid', None)

    @service_uuid.setter
    def service_uuid(self, service_uuid):
        # type: (Union[None, UUID]) -> None
        if not isinstance(service_uuid, six.string_types) or service_uuid is None:
            raise TypeError("Type `str` is required for `{}.service_uuid`".format(type(self)))
        self['service_uuid'] = service_uuid

    @property
    def process_uuid(self):
        # type: (...) -> Union[None, UUID]
        return self.get('process_uuid', None)

    @process_uuid.setter
    def process_uuid(self, process_uuid):
        # type: (Union[None, UUID]) -> None
        if not isinstance(process_uuid, six.string_types) or process_uuid is None:
            raise TypeError("Type `str` is required for `{}.process_uuid`".format(type(self)))
        self['process_uuid'] = process_uuid

    def _get_inputs(self):
        # type: (...) -> List[Optional[JsonBody]]
        if self.get('inputs') is None:
            self['inputs'] = list()
        return self['inputs']

    def _set_inputs(self, inputs):
        # type: (List[Optional[JsonBody]]) -> None
        if not isinstance(inputs, list):
            raise TypeError("Type `list` is required for `{}.inputs`".format(type(self)))
        self['inputs'] = inputs

    # allows to correctly update list by ref using `job.inputs.extend()`
    inputs = property(_get_inputs, _set_inputs)

    @property
    def user_uuid(self):
        # type: (...) -> Union[None, AnyStr]
        return self.get('user_uuid', None)

    @user_uuid.setter
    def user_uuid(self, user_uuid):
        # type: (Union[None, AnyStr]) -> None
        if not isinstance(user_uuid, int) or user_uuid is None:
            raise TypeError("Type `int` is required for `{}.user_uuid`".format(type(self)))
        self['user_uuid'] = user_uuid

    @property
    def status(self):
        # type: (...) -> AnyStr
        return self.get('status', STATUS_UNKNOWN)

    @status.setter
    def status(self, status):
        # type: (AnyStr) -> None
        if not isinstance(status, six.string_types):
            raise TypeError("Type `str` is required for `{}.status`".format(type(self)))
        if status not in job_status_values:
            raise ValueError("Status `{0}` is not valid for `{1}.status`, must be one of {2!s}`"
                             .format(status, type(self), list(job_status_values)))
        self['status'] = status
        if status in job_status_categories[STATUS_CATEGORY_FINISHED]:
            self.mark_finished()

    @property
    def status_message(self):
        # type: (...) -> AnyStr
        return self.get('status_message', 'no message')

    @status_message.setter
    def status_message(self, message):
        # type: (Union[None, AnyStr]) -> None
        if message is None:
            return
        if not isinstance(message, six.string_types):
            raise TypeError("Type `str` is required for `{}.status_message`".format(type(self)))
        self['status_message'] = message

    @property
    def status_location(self):
        # type: (...) -> Union[None, AnyStr]
        return self.get('status_location', None)

    @status_location.setter
    def status_location(self, location_url):
        # type: (Union[None, AnyStr]) -> None
        if not isinstance(location_url, six.string_types) or location_url is None:
            raise TypeError("Type `str` is required for `{}.status_location`".format(type(self)))
        self['status_location'] = location_url

    @property
    def execute_async(self):
        # type: (...) -> bool
        return self.get('execute_async', True)

    @execute_async.setter
    def execute_async(self, execute_async):
        # type: (bool) -> None
        if not isinstance(execute_async, bool):
            raise TypeError("Type `bool` is required for `{}.execute_async`".format(type(self)))
        self['execute_async'] = execute_async

    @property
    def is_workflow(self):
        # type: (...) -> bool
        return self.get('is_workflow', False)

    @is_workflow.setter
    def is_workflow(self, is_workflow):
        # type: (bool) -> None
        if not isinstance(is_workflow, bool):
            raise TypeError("Type `bool` is required for `{}.is_workflow`".format(type(self)))
        self['is_workflow'] = is_workflow

    @property
    def created(self):
        # type: (...) -> datetime
        created = self.get('created', None)
        if not created:
            self['created'] = now()
        return localize_datetime(self.get('created'))

    @property
    def finished(self):
        # type: (...) -> Union[None, datetime]
        finished = self.get('finished')
        if finished:
            return localize_datetime(finished)
        return None

    def is_finished(self):
        # type: (...) -> bool
        return self.finished is not None

    def mark_finished(self):
        # type: (...) -> None
        self['finished'] = localize_datetime(now())

    @property
    def duration(self):
        # type: (...) -> AnyStr
        final_time = self.finished or now()
        duration = localize_datetime(final_time) - localize_datetime(self.created)
        self['duration'] = str(duration).split('.')[0]
        return self['duration']

    @property
    def progress(self):
        # type: (...) -> Union[int, float]
        return self.get('progress', 0)

    @progress.setter
    def progress(self, progress):
        # type: (Union[int, float]) -> None
        if not isinstance(progress, (int, float)):
            raise TypeError("Number is required for `{}.progress`".format(type(self)))
        if progress < 0 or progress > 100:
            raise ValueError("Value must be in range [0,100] for `{}.progress`".format(type(self)))
        self['progress'] = progress

    def _get_results(self):
        # type: (...) -> List[Optional[JsonBody]]
        if self.get('results') is None:
            self['results'] = list()
        return self['results']

    def _set_results(self, results):
        # type: (List[Optional[JsonBody]]) -> None
        if not isinstance(results, list):
            raise TypeError("Type `list` is required for `{}.results`".format(type(self)))
        self['results'] = results

    # allows to correctly update list by ref using `job.results.extend()`
    results = property(_get_results, _set_results)

    def _get_exceptions(self):
        # type: (...) -> List[Optional[JsonBody]]
        if self.get('exceptions') is None:
            self['exceptions'] = list()
        return self['exceptions']

    def _set_exceptions(self, exceptions):
        # type: (List[Optional[JsonBody]]) -> None
        if not isinstance(exceptions, list):
            raise TypeError("Type `list` is required for `{}.exceptions`".format(type(self)))
        self['exceptions'] = exceptions

    # allows to correctly update list by ref using `job.exceptions.extend()`
    exceptions = property(_get_exceptions, _set_exceptions)

    def _get_logs(self):
        # type: (...) -> List[Optional[JsonBody]]
        if self.get('logs') is None:
            self['logs'] = list()
        return self['logs']

    def _set_logs(self, logs):
        # type: (List[Optional[JsonBody]]) -> None
        if not isinstance(logs, list):
            raise TypeError("Type `list` is required for `{}.logs`".format(type(self)))
        self['logs'] = logs

    # allows to correctly update list by ref using `job.logs.extend()`
    logs = property(_get_logs, _set_logs)

    def _get_tags(self):
        # type: (...) -> List[Optional[AnyStr]]
        if self.get('tags') is None:
            self['tags'] = list()
        return self['tags']

    def _set_tags(self, tags):
        # type: (List[Optional[AnyStr]]) -> None
        if not isinstance(tags, list):
            raise TypeError("Type `list` is required for `{}.tags`".format(type(self)))
        self['tags'] = tags

    # allows to correctly update list by ref using `job.tags.extend()`
    tags = property(_get_tags, _set_tags)

    @property
    def request(self):
        # type: (...) -> Union[None, AnyStr]
        """XML request for WPS execution submission as string."""
        return self.get('request', None)

    @request.setter
    def request(self, request):
        # type: (Union[None, AnyStr]) -> None
        """XML request for WPS execution submission as string."""
        self['request'] = request

    @property
    def response(self):
        # type: (...) -> Union[None, AnyStr]
        """XML status response from WPS execution submission as string."""
        return self.get('response', None)

    @response.setter
    def response(self, response):
        # type: (Union[None, AnyStr]) -> None
        """XML status response from WPS execution submission as string."""
        self['response'] = response

    @property
    def params(self):
        # type: (...) -> JsonBody
        return {
            'uuid': self.uuid,
            'task_uuid': self.task_uuid,
            'service_uuid': self.service_uuid,
            'process_uuid': self.process_uuid,
            'inputs': self.inputs,
            'user_uuid': self.user_uuid,
            'status': self.status,
            'status_message': self.status_message,
            'status_location': self.status_location,
            'execute_async': self.execute_async,
            'is_workflow': self.is_workflow,
            'created': self.created,
            'finished': self.finished,
            'duration': self.duration,
            'progress': self.progress,
            'results': self.results,
            'exceptions': self.exceptions,
            'logs': self.logs,
            'tags': self.tags,
            'request': self.request,
            'response': self.response,
        }

    def json(self):
        # type: (...) -> JsonBody
        return {
            'uuid': self.uuid,
            'task_uuid': self.task_uuid,
            'service_uuid': self.service_uuid,
            'process_uuid': self.process_uuid,
            'user_uuid': self.user_uuid,
            'inputs': self.inputs,
            'status': self.status,
            'status_message': self.status_message,
            'status_location': self.status_location,
            'execute_async': self.execute_async,
            'is_workflow': self.is_workflow,
            'created': stringify_datetime(self.created) if self.created else None,
            'finished': stringify_datetime(self.finished) if self.finished else None,
            'duration': self.duration,
            'progress': self.progress,
            'tags': self.tags,
        }

    def summary(self):
        # type: (...) -> JsonBody
        return {
            'uuid': self.uuid,
            'process_uuid': self.process_uuid,
        }

    def __str__(self):
        # type: (...) -> AnyStr
        return 'Job <{}>'.format(self.uuid)

    def __repr__(self):
        # type: (...) -> AnyStr
        cls = type(self)
        rep = dict.__repr__(self)
        return '{0}.{1} ({2})'.format(cls.__module__, cls.__name__, rep)
