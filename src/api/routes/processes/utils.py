#!/usr/bin/env python
# coding: utf-8
from geoimagenet_ml.store.datatypes import Process, Job
from geoimagenet_ml.api import exceptions as ex, requests as r, schemas as s
from geoimagenet_ml.store.factories import database_factory
from geoimagenet_ml.store import exceptions as exc
from geoimagenet_ml.typedefs import AnyStr, Number, List, Union, Optional, UUID, TYPE_CHECKING       # noqa: F401
from geoimagenet_ml.processes.types import process_categories, PROCESS_ML
from geoimagenet_ml.processes.status import (
    map_status, STATUS_SUCCEEDED, STATUS_FAILED, STATUS_STARTED, STATUS_RUNNING
)
from geoimagenet_ml.processes.utils import get_base_url
from geoimagenet_ml.ml.impl import get_test_data_runner
from pyramid.request import Request
from pyramid.httpexceptions import (
    HTTPCreated,
    HTTPBadRequest,
    HTTPForbidden,
    HTTPNotFound,
    HTTPConflict,
    HTTPUnprocessableEntity,
    HTTPNotImplemented,
)
from pyramid_celery import celery_app as app
from celery.utils.log import get_task_logger
import numpy
import logging
import multiprocessing
LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    # noinspection PyProtectedMember
    from celery import Task                     # noqa: F401
    # noinspection PyPackageRequirements
    from owslib.wps import WPSException         # noqa: F401


def create_process(request):
    # type: (Request) -> Process
    """Creates the process based on the request after body inputs validation."""
    process_name = r.get_multiformat_post(request, 'process_name')
    process_type = r.get_multiformat_post(request, 'process_type')
    ex.verify_param(process_name, notNone=True, notEmpty=True, httpError=HTTPBadRequest, paramName='process_name',
                    msgOnFail=s.Processes_POST_BadRequestResponseSchema.description, request=request)
    ex.verify_param(process_type, isIn=True, httpError=HTTPBadRequest, paramName='process_type',
                    paramCompare=list(process_categories) + [None],  # allow None to use default value
                    msgOnFail=s.Processes_POST_BadRequestResponseSchema.description, request=request)
    new_process = None
    try:
        tmp_process = Process(identifier=process_name, type=process_type)
        new_process = database_factory(request.registry).processes_store.save_process(tmp_process, request=request)
        if not new_process:
            raise exc.ProcessRegistrationError
    except (exc.ProcessRegistrationError, exc.ProcessInstanceError):
        ex.raise_http(httpError=HTTPForbidden, request=request,
                      detail=s.Processes_POST_ForbiddenResponseSchema.description)
    except exc.ProcessConflictError:
        ex.raise_http(httpError=HTTPConflict, request=request,
                      detail=s.Processes_POST_ConflictResponseSchema.description)
    except exc.ProcessNotFoundError:
        ex.raise_http(httpError=HTTPNotFound, request=request,
                      detail=s.Processes_POST_NotFoundResponseSchema.description)
    return new_process


def get_process(request):
    # type: (Request) -> Process
    """Retrieves the process based on the request after body inputs validation."""
    process_uuid = request.matchdict.get('process_uuid')
    ex.verify_param(process_uuid, notNone=True, notEmpty=True, httpError=HTTPBadRequest, paramName='process_uuid',
                    msgOnFail=s.Process_GET_BadRequestResponseSchema.description, request=request)
    process = None
    try:
        process = database_factory(request.registry).processes_store.fetch_by_uuid(process_uuid)
        if not process:
            raise exc.ProcessNotFoundError
    except exc.ProcessInstanceError:
        ex.raise_http(httpError=HTTPForbidden, request=request,
                      detail=s.Process_GET_ForbiddenResponseSchema.description)
    except exc.ProcessNotFoundError:
        ex.raise_http(httpError=HTTPNotFound, request=request,
                      detail=s.Process_GET_NotFoundResponseSchema.description)
    return process


def get_job(request):
    # type: (Request) -> Job
    """Retrieves the job based on the request after body inputs validation."""
    job_uuid = request.matchdict.get('job_uuid')
    ex.verify_param(job_uuid, notNone=True, notEmpty=True, httpError=HTTPBadRequest, paramName='job_uuid',
                    msgOnFail=s.ProcessJob_GET_BadRequestResponseSchema.description, request=request)
    job = None
    try:
        job = database_factory(request.registry).jobs_store.fetch_by_uuid(job_uuid)
        if not job:
            raise exc.JobNotFoundError
    except exc.ProcessInstanceError:
        ex.raise_http(httpError=HTTPForbidden, request=request,
                      detail=s.ProcessJob_GET_ForbiddenResponseSchema.description)
    except exc.JobNotFoundError:
        ex.raise_http(httpError=HTTPNotFound, request=request,
                      detail=s.ProcessJob_GET_NotFoundResponseSchema.description)
    return job


def create_process_job(request, process):
    # type: (Request, Process) -> HTTPCreated
    """Creates a job for the requested process and dispatches it to the celery runner."""

    # validate body with expected JSON content and schema
    if 'application/json' not in request.content_type:
        raise HTTPBadRequest("Request 'Content-Type' header other than 'application/json' not supported.")
    try:
        json_body = request.json_body
    except Exception as e:
        raise HTTPBadRequest("Invalid JSON body cannot be decoded for job submission. [{}]".format(e))

    if 'inputs' not in json_body:
        raise HTTPBadRequest("Missing 'inputs' form JSON body.")
    job_inputs = json_body['inputs']

    # validation for specific processes
    if process.type != PROCESS_ML:
        # TODO: dispatch to wps as necessary
        raise HTTPNotImplemented("WPS process job execution dispatching not implemented.")
    for i in job_inputs:
        if not isinstance(i, dict):
            raise HTTPUnprocessableEntity("Invalid 'inputs' cannot be processed.")
    json_input_ids = [i['id'] for i in job_inputs]
    required_input_ids = ['dataset', 'model']
    if not all([input_id in json_input_ids for input_id in required_input_ids]):
        raise HTTPBadRequest("Missing at least one of required inputs {!s}".format(required_input_ids))

    # create job and dispatch it
    jobs_store = database_factory(request.registry).jobs_store
    job = Job(process_uuid=process.uuid, inputs=job_inputs)
    job = jobs_store.save_job(job)
    result = process_ml_job_runner.delay(job_uuid=job.uuid)
    job.status = map_status(result.status)  # pending or failure according to accepted celery task
    job = jobs_store.update_job(job)
    LOGGER.debug("Celery task `{}` for job `{}`.".format(result.id, job.uuid))

    location = '{base_url}{job_path}'.format(
        base_url=get_base_url(request.registry.settings),
        job_path=s.ProcessJobAPI.path.replace(s.ProcessVariableUUID, process.uuid).replace(s.JobVariableUUID, job.uuid))
    body_data = {
        'jobID': job.uuid,
        'status': job.status,
        'location': location
    }
    return HTTPCreated(location=location, json=body_data)


class CallbackIterator(object):
    def __init__(self):
        self._iter = 0

    def __call__(self, *args, **kwargs):
        self._iter = self._iter + 1

    @property
    def iteration(self):
        return self._iter


@app.task(bind=True)
def process_ml_job_runner(self, job_uuid):
    # type: (Task, UUID) -> AnyStr
    """Runs, monitor and updates the job based on it's execution progression and/or errors."""

    registry = app.conf['PYRAMID_REGISTRY']
    request = self.request
    jobs_store = database_factory(registry).jobs_store
    task_logger = get_task_logger(__name__)

    job = jobs_store.fetch_by_uuid(job_uuid, request=request)
    job.task_uuid = request.id
    job.tags.append(PROCESS_ML)
    job.status_location = '{base_url}{job_path}'.format(
        base_url=get_base_url(registry.settings),
        job_path=s.ProcessJobAPI.path
                  .replace(s.ProcessVariableUUID, job.process_uuid)
                  .replace(s.JobVariableUUID, job.uuid)
    )
    job = jobs_store.update_job(job, request=request)

    # note:
    #   if dataset loader uses multiple worker sub-processes to load batch samples, the process needs to be non-daemon
    #   to allow pool spawning of child processes since this task is already a child worker of the main celery app
    # see:
    #   ``geoimagenet_ml.ml.impl.test_loader_from_configs`` for corresponding override
    worker_count = registry.settings.get('geoimagenet_ml.ml.data_loader_workers', 0)
    worker_process = multiprocessing.current_process()
    # noinspection PyProtectedMember, PyUnresolvedReferences
    worker_process._config['daemon'] = not bool(worker_count)

    def _update_job_status(status, status_message, status_progress=None, errors=None):
        # type: (AnyStr, AnyStr, Optional[Number], Optional[Union[AnyStr, Exception, List[WPSException]]]) -> None
        """Updates the new job status."""
        job.status = map_status(status)
        job.status_message = "{} {}.".format(str(job), status_message)
        job.progress = status_progress if status_progress is not None else job.progress
        job.save_log(logger=task_logger, errors=errors)
        jobs_store.update_job(job, request=request)

    def _update_job_eval_progress(_job, _batch_iterator, start_percent=0, final_percent=100):
        # type: (Job, CallbackIterator, Optional[Number], Optional[Number]) -> None
        """
        Updates the job progress based on evaluation progress (after each batch).
        Called using callback of prediction metric.
        """
        metric = test_runner.test_metrics['predictions']
        total_sample_count = test_runner.test_loader.sample_count
        evaluated_sample_count = len(metric.predictions)    # gradually expanded on each evaluation callback
        batch_count = len(test_runner.test_loader)
        batch_index = _batch_iterator.iteration + 1
        progress = numpy.interp(batch_index / batch_count, [0, 100], [start_percent, final_percent])
        msg = "evaluating... [samples: {}/{}, batches: {}/{}]" \
              .format(evaluated_sample_count, total_sample_count, batch_index, batch_count)
        _update_job_status(STATUS_RUNNING, msg, progress)

        # update job results and add important fields
        _job.results = [{
            'identifier': 'predictions',
            'value': metric.predictions,
        }]
        if hasattr(test_runner, 'class_names'):
            _job.results.insert(0, {
                'identifier': 'classes',
                'value': test_runner.class_names
            })
        jobs_store.update_job(_job)

    try:
        _update_job_status(STATUS_STARTED, "initiation done", 1)

        _update_job_status(STATUS_RUNNING, "retrieving dataset definition", 2)
        dataset_uuid = [job_input['value'] for job_input in job.inputs if job_input['id'] == 'dataset'][0]
        dataset = database_factory(registry).datasets_store.fetch_by_uuid(dataset_uuid, request=request)

        _update_job_status(STATUS_RUNNING, "loading model from definition", 3)
        model_uuid = [job_input['value'] for job_input in job.inputs if job_input['id'] == 'model'][0]
        model = database_factory(registry).models_store.fetch_by_uuid(model_uuid, request=request)
        model_config = model.data     # calls loading method, raises failure accordingly

        _update_job_status(STATUS_RUNNING, "retrieving data loader for model and dataset", 4)
        test_runner = get_test_data_runner(job, model_config, model, dataset, registry.settings)

        # link the batch iteration with a callback for progress tracking
        batch_iter = CallbackIterator()
        test_runner.eval_iter_callback = batch_iter.__call__

        # link the predictions with a callback for progress update during evaluation
        pred_metric = test_runner.test_metrics['predictions']
        pred_metric.callback = lambda: _update_job_eval_progress(job, batch_iter, start_percent=5, final_percent=99)
        _update_job_status(STATUS_RUNNING, "starting test data prediction evaluation", 5)
        test_runner.eval()

        _update_job_status(STATUS_SUCCEEDED, "processing complete", 100)

    except Exception as task_exc:
        exception_class = "{}.{}".format(type(task_exc).__module__, type(task_exc).__name__)
        err_msg = "{0}: {1}".format(exception_class, str(task_exc))
        message = "failed to run {!s} [{!s}].".format(job, err_msg)
        _update_job_status(STATUS_FAILED, message, errors=task_exc)
    finally:
        _update_job_status(job.status, "done")

    return job.status
