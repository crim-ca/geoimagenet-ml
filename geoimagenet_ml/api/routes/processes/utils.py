#!/usr/bin/env python
# coding: utf-8
from geoimagenet_ml.api import exceptions as ex, requests as r, schemas as s
from geoimagenet_ml.constants import SORT, OPERATION, JOB_TYPE
from geoimagenet_ml.store import exceptions as exc
from geoimagenet_ml.store.datatypes import Process, Job
from geoimagenet_ml.store.factories import database_factory
from geoimagenet_ml.processes.types import process_mapping, process_categories, PROCESS_WPS
from geoimagenet_ml.status import map_status, STATUS, CATEGORY
from geoimagenet_ml.utils import get_base_url, get_user_id, is_uuid
from pyramid.httpexceptions import (
    HTTPAccepted,
    HTTPBadRequest,
    HTTPForbidden,
    HTTPNotFound,
    HTTPConflict,
    HTTPUnprocessableEntity,
    HTTPInternalServerError,
    HTTPNotImplemented,
)
from pyramid_celery import celery_app as app
from typing import TYPE_CHECKING
import logging
if TYPE_CHECKING:
    # noinspection PyProtectedMember
    from celery import Task                                     # noqa: F401
    from pyramid.request import Request                         # noqa: F401
    from pyramid.httpexceptions import HTTPException            # noqa: F401
    from typing import Optional                                 # noqa: F401
    from geoimagenet_ml.typedefs import Any, AnyStr, AnyUUID    # noqa: F401
LOGGER = logging.getLogger(__name__)


def create_process(request):
    # type: (Request) -> Optional[Process]
    """Creates the process based on the request after body inputs validation."""
    process_name = r.get_multiformat_post(request, "process_name")
    process_type = r.get_multiformat_post(request, "process_type")
    ex.verify_param(process_name, notNone=True, notEmpty=True, httpError=HTTPBadRequest, paramName="process_name",
                    msgOnFail=s.Processes_POST_BadRequestResponseSchema.description, request=request)
    ex.verify_param(process_type, isIn=True, httpError=HTTPBadRequest, paramName="process_type",
                    paramCompare=list(process_categories) + [None],  # allow None to use default value
                    msgOnFail=s.Processes_POST_BadRequestResponseSchema.description, request=request)
    if process_type is None:
        process_type = PROCESS_WPS
    try:
        db = database_factory(request)
        new_process = None
        tmp_process = Process(identifier=process_name, type=process_type, user=get_user_id(request))
        new_process = db.processes_store.save_process(tmp_process, request=request)
        if not new_process:
            raise exc.ProcessRegistrationError
        return new_process
    except (exc.ProcessRegistrationError, exc.ProcessInstanceError):
        ex.raise_http(httpError=HTTPForbidden, request=request,
                      detail=s.Processes_POST_ForbiddenResponseSchema.description)
    except exc.ProcessConflictError:
        ex.raise_http(httpError=HTTPConflict, request=request,
                      detail=s.Processes_POST_ConflictResponseSchema.description)
    except exc.ProcessNotFoundError:
        ex.raise_http(httpError=HTTPNotFound, request=request,
                      detail=s.Processes_POST_NotFoundResponseSchema.description)


def get_process(request):
    # type: (Request) -> Optional[Process]
    """Retrieves the process based on the request after body inputs validation."""
    process_uuid = request.matchdict.get(s.ParamProcessUUID)
    ex.verify_param(process_uuid, notNone=True, notEmpty=True, httpError=HTTPBadRequest, paramName=s.ParamProcessUUID,
                    msgOnFail=s.Process_GET_BadRequestResponseSchema.description, request=request)
    process = None
    try:
        db = database_factory(request)
        if is_uuid(process_uuid):
            LOGGER.debug("fetching process by uuid '{}'".format(process_uuid))
            process = db.processes_store.fetch_by_uuid(process_uuid)
        else:
            LOGGER.debug("fetching process by identifier '{}'".format(process_uuid))
            process = db.processes_store.fetch_by_identifier(process_uuid)
        if not process:
            raise exc.ProcessNotFoundError
        return process
    except exc.ProcessInstanceError:
        ex.raise_http(httpError=HTTPForbidden, request=request,
                      detail=s.Process_GET_ForbiddenResponseSchema.description)
    except exc.ProcessNotFoundError:
        ex.raise_http(httpError=HTTPNotFound, request=request,
                      detail=s.Process_GET_NotFoundResponseSchema.description)


def get_job(request):
    # type: (Request) -> Job
    """Retrieves the job based on the request after body inputs validation."""
    job_uuid = request.matchdict.get(s.ParamJobUUID)
    ex.verify_param(job_uuid, notNone=True, notEmpty=True, httpError=HTTPBadRequest, paramName=s.ParamJobUUID,
                    msgOnFail=s.ProcessJob_GET_BadRequestResponseSchema.description, request=request)
    job = None
    try:
        if job_uuid in JOB_TYPE.values() + JOB_TYPE.names():
            job = get_job_special(request, JOB_TYPE.get(job_uuid))
        else:
            job = database_factory(request).jobs_store.fetch_by_uuid(job_uuid)
        if not job:
            raise exc.JobNotFoundError
    except exc.ProcessInstanceError:
        ex.raise_http(httpError=HTTPForbidden, request=request,
                      detail=s.ProcessJob_GET_ForbiddenResponseSchema.description)
    except exc.JobNotFoundError:
        ex.raise_http(httpError=HTTPNotFound, request=request,
                      detail=s.ProcessJob_GET_NotFoundResponseSchema.description)
    return job


def get_job_special(request, job_type):
    # type: (Request, JOB_TYPE) -> Optional[Job]
    """
    Retrieves a job based on the request using keyword specifiers input validation.

    :raises TypeError: if ``job_type` is invalid.
    :raises HTTPException: corresponding error if applicable.
    :raises JobNotFoundError: in case of not found job that was not handled with HTTPException for special cases.
    """
    if not isinstance(job_type, JOB_TYPE):
        raise TypeError("Enum 'JOB_TYPE' value required.")
    jobs_store = database_factory(request).jobs_store
    proc = get_process(request)  # process required, raise if not specified
    if not proc.limit_single_job and job_type == JOB_TYPE.CURRENT:
        ex.raise_http(httpError=HTTPForbidden, request=request,
                      detail="Keyword '{}' is not allowed for multi-job process '{}'."
                      .format(JOB_TYPE.CURRENT.value, proc.identifier))

    status = [CATEGORY.RECEIVED, CATEGORY.EXECUTING] if job_type == JOB_TYPE.CURRENT else map_status(STATUS.SUCCESS)
    sort = SORT.FINISHED if job_type == JOB_TYPE.LATEST else SORT.CREATED
    try:
        jobs, count = jobs_store.find_jobs(process=proc.uuid, status=status, sort=sort)
    except exc.JobNotFoundError:
        if job_type == JOB_TYPE.CURRENT:
            ex.raise_http(httpError=HTTPNotFound, request=request,
                          detail="No current job pending execution or running for process.")
        raise  # other cases use default message
    if count > 1 and job_type == JOB_TYPE.CURRENT:
        ex.raise_http(httpError=HTTPInternalServerError, request=request,
                      detail="Found too many jobs ({}). Should only find one for single job process '{}'."
                             .format(count, proc.identifier))
    if count > 0:
        return jobs[0]
    return None


def update_job_params(request):
    # type: (Request) -> Optional[Job]
    """
    Updates a job's parameter from specified request input fields.
    Job is updated in storage if all input validation passed.

    :raises HTTPException: corresponding error if applicable.
    """
    job = get_job(request)

    def _apply(p, v):
        job[p] = v

    for param in ["visibility"]:
        value = r.get_multiformat_any(request, "visibility")
        ex.evaluate_call(lambda: _apply(param, value),
                         httpError=HTTPBadRequest,
                         msgOnFail=s.ProcessJob_PUT_BadRequestResponseSchema.description,
                         content={"param": {"name": param, "value": str(value)}})
    db = database_factory(request)
    job = ex.evaluate_call(lambda: db.jobs_store.update_job(job, request=request),
                           httpError=HTTPForbidden,
                           msgOnFail=s.ProcessJob_PUT_ForbiddenResponseSchema.description,
                           content={"param": {"name": "job", "value": job.uuid}})
    return job


# noinspection PyProtectedMember
def get_job_status_location(request, process, job):
    # type: (Request, Process, Job) -> AnyStr
    """Obtains the full URL of the job status location using the process ID variant specified by the request."""
    if request.path.startswith(s.ProcessJobsAPI.path.replace(s.VariableProcessUUID, process.identifier)):
        proc_id = process.identifier
    else:
        proc_id = process.uuid
    proc_job_path = s.ProcessJobAPI.path.replace(s.VariableProcessUUID, proc_id).replace(s.VariableJobUUID, job.uuid)
    return "{base}{path}".format(base=get_base_url(request.registry.settings), path=proc_job_path)


def create_process_job(request, process):
    # type: (Request, Process) -> HTTPException
    """Creates a job for the requested process and dispatches it to the celery runner."""

    # validate body with expected JSON content and schema
    if "application/json" not in request.content_type:
        raise HTTPBadRequest("Request 'Content-Type' header other than 'application/json' not supported.")
    try:
        json_body = request.json_body
    except Exception as e:
        raise HTTPBadRequest("Invalid JSON body cannot be decoded for job submission. [{}]".format(e))

    if "inputs" not in json_body:
        raise HTTPBadRequest("Missing 'inputs' form JSON body.")
    job_inputs = json_body["inputs"]

    for i in job_inputs:
        if not isinstance(i, dict):
            raise HTTPUnprocessableEntity("Invalid 'inputs' cannot be processed. Invalid list of objects format.")
        if "id" not in i:
            raise HTTPUnprocessableEntity("Invalid 'inputs' cannot be processed. Missing 'id' in input object.")

    # TODO: dispatch other process runner as necessary
    if process.identifier not in process_mapping or process.type == PROCESS_WPS:
        raise HTTPNotImplemented("Process job execution not implemented for '{}' of type '{}'."
                                 .format(process.identifier, process.type))

    if process.limit_single_job:
        job = None
        try:
            job = get_job_special(request, JOB_TYPE.CURRENT)
        except (exc.JobNotFoundError, HTTPNotFound):
            pass
        if job is not None:
            ex.raise_http(httpError=HTTPForbidden, request=request,
                          detail="Multiple jobs not allowed for [{!s}]. Job submission aborted.".format(process))

    # validation for specific processes, create job and dispatch it to corresponding runner
    runner_key = process.identifier
    runner = process_mapping[runner_key]
    missing_inputs = runner.check_inputs(job_inputs)
    if missing_inputs:
        raise HTTPBadRequest("Missing inputs '{}' for process of type '{}'.".format(missing_inputs, process.type))

    db = database_factory(request)
    jobs_store = db.jobs_store
    job = Job(process=process.uuid, inputs=job_inputs, user=get_user_id(request), status=STATUS.ACCEPTED)
    job = jobs_store.save_job(job)
    LOGGER.debug("Queuing new celery task for `{!s}`.".format(job))
    result = process_job_runner.delay(job_uuid=job.uuid, runner_key=runner_key)

    LOGGER.debug("Celery task '{}' for `{!s}`.".format(result.id, job))
    job.status = map_status(result.status)  # pending or failure according to accepted celery task
    job.status_location = get_job_status_location(request, process, job)
    job = jobs_store.update_job(job)
    job_json = job.json()
    body_data = {
        "job_uuid": job_json.get("uuid"),
        "status": job_json.get("status"),
        "location": job_json.get("status_location")
    }
    db.actions_store.save_action(job, OPERATION.SUBMIT, request=request)
    return HTTPAccepted(location=job.status_location, json=body_data)


@app.task(bind=True)
def process_job_runner(task, job_uuid, runner_key):
    # type: (Task, AnyUUID, AnyStr) -> Any
    LOGGER.debug("Celery task for job '{}' [{}] received.".format(job_uuid, runner_key))
    registry = app.conf["PYRAMID_REGISTRY"]
    runner = process_mapping[runner_key]
    return runner(task, registry, task.request, job_uuid)()
