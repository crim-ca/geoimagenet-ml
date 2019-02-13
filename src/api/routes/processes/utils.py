#!/usr/bin/env python
# coding: utf-8
from geoimagenet_ml.api import exceptions as ex, requests as r, schemas as s
from geoimagenet_ml.store import exceptions as exc
from geoimagenet_ml.store.datatypes import Process, Job
from geoimagenet_ml.store.factories import database_factory
from geoimagenet_ml.processes.types import process_mapping, process_categories, PROCESS_WPS
from geoimagenet_ml.processes.status import map_status
from geoimagenet_ml.utils import get_base_url
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
import logging
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # noinspection PyProtectedMember
    from celery import Task                                     # noqa: F401
    from pyramid.request import Request                         # noqa: F401
    from geoimagenet_ml.typedefs import AnyStr, UUID            # noqa: F401
    from geoimagenet_ml.processes.runners import ProcessRunner  # noqa: F401
LOGGER = logging.getLogger(__name__)


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
    if process_type is None:
        process_type = PROCESS_WPS
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

    for i in job_inputs:
        if not isinstance(i, dict):
            raise HTTPUnprocessableEntity("Invalid 'inputs' cannot be processed.")

    # TODO: dispatch other process runner as necessary
    if process.identifier not in process_mapping or process.type == PROCESS_WPS:
        raise HTTPNotImplemented("Process job execution not implemented for '{}' of type '{}'."
                                 .format(process.identifier, process.type))

    # validation for specific processes, create job and dispatch it to corresponding runner
    runner = process_mapping[process.identifier]
    missing_inputs = runner.check_inputs(job_inputs)
    if missing_inputs:
        raise HTTPBadRequest("Missing inputs '{}' for process of type '{}'.".format(missing_inputs, process.type))

    jobs_store = database_factory(request.registry).jobs_store
    job = Job(process_uuid=process.uuid, inputs=job_inputs)
    job = jobs_store.save_job(job)
    result = process_job_runner.delay(job_uuid=job.uuid, runner=runner)

    LOGGER.debug("Celery task `{}` for job `{}`.".format(result.id, job.uuid))
    job.status = map_status(result.status)  # pending or failure according to accepted celery task
    job.status_location = '{base_url}{job_path}'.format(
        base_url=get_base_url(request.registry.settings),
        job_path=s.ProcessJobAPI.path
                  .replace(s.ProcessVariableUUID, job.process_uuid)
                  .replace(s.JobVariableUUID, job.uuid)
    )
    job = jobs_store.update_job(job)
    body_data = {
        'jobID': job.uuid,
        'status': job.status,
        'location': job.status_location
    }
    return HTTPCreated(location=job.status_location, json=body_data)


@app.task(bind=True)
def process_job_runner(task, job_uuid, runner):
    # type: (Task, UUID, ProcessRunner) -> AnyStr
    registry = app.conf['PYRAMID_REGISTRY']
    return runner(task, registry, task.request, job_uuid)()
