#!/usr/bin/env python
# coding: utf-8
from geoimagenet_ml.api import exceptions as ex, schemas as s
from geoimagenet_ml.api.routes.processes.utils import (
    create_process, get_process, get_job, create_process_job, update_job_params
)
from geoimagenet_ml.constants import OPERATION
from geoimagenet_ml.utils import get_any_id, get_any_value, has_raw_value
from geoimagenet_ml.store.datatypes import Process, Job
from geoimagenet_ml.store.factories import database_factory
from pyramid.httpexceptions import (
    HTTPOk,
    HTTPCreated,
    HTTPForbidden,
    HTTPNotFound,
    HTTPInternalServerError,
    HTTPException,
)


@s.ProcessesAPI.get(tags=[s.TagProcesses], response_schemas=s.Processes_GET_responses)
def get_processes_view(request):
    """Get registered processes information."""
    db = database_factory(request)
    processes_list = ex.evaluate_call(lambda: db.processes_store.list_processes(request=request),
                                      fallback=lambda: db.rollback(), httpError=HTTPForbidden, request=request,
                                      msgOnFail=s.Processes_GET_ForbiddenResponseSchema.description)
    processes_json = ex.evaluate_call(lambda: [p.summary() for p in processes_list],
                                      fallback=lambda: db.rollback(), httpError=HTTPInternalServerError,
                                      request=request, msgOnFail=s.InternalServerErrorResponseSchema.description)
    db.actions_store.save_action(Process, OPERATION.INFO, request=request)
    return ex.valid_http(httpSuccess=HTTPOk, content={u"processes": processes_json},
                         detail=s.Processes_GET_OkResponseSchema.description, request=request)


@s.ProcessesAPI.post(tags=[s.TagProcesses],
                     schema=s.Processes_POST_RequestSchema(), response_schemas=s.Processes_POST_responses)
def post_processes_view(request):
    """Register a new process."""
    process = create_process(request)
    database_factory(request).actions_store.save_action(process, OPERATION.SUBMIT, request=request)
    return ex.valid_http(httpSuccess=HTTPCreated, content={u"process": process.json()},
                         detail=s.Processes_POST_CreatedResponseSchema.description, request=request)


@s.ProcessAPI.get(tags=[s.TagProcesses],
                  schema=s.ProcessEndpoint(), response_schemas=s.Process_GET_responses)
def get_process_view(request):
    """Get registered process information."""
    process = get_process(request)
    database_factory(request).actions_store.save_action(process, OPERATION.INFO, request=request)
    return ex.valid_http(httpSuccess=HTTPOk, content={u"process": process.json()},
                         detail=s.Process_GET_OkResponseSchema.description, request=request)


@s.ProcessJobsAPI.get(tags=[s.TagProcesses, s.TagJobs],
                      schema=s.ProcessJobsEndpoint(), response_schemas=s.ProcessJobs_GET_responses)
def get_process_jobs_view(request):
    """Get registered process jobs."""
    process = get_process(request)
    db = database_factory(request)
    job_list = ex.evaluate_call(lambda: db.jobs_store.list_jobs(request=request),
                                fallback=lambda: db.rollback(), httpError=HTTPForbidden, request=request,
                                msgOnFail=s.ProcessJobs_GET_ForbiddenResponseSchema.description)
    process_jobs_json = ex.evaluate_call(lambda: [j.summary() for j in
                                                  filter(lambda j: j.process == process.uuid, job_list)],
                                         fallback=lambda: db.rollback(), httpError=HTTPInternalServerError,
                                         request=request, msgOnFail=s.InternalServerErrorResponseSchema.description)
    db.actions_store.save_action(Job, OPERATION.LIST, request=request)
    return ex.valid_http(httpSuccess=HTTPOk, content={u"jobs": process_jobs_json},
                         detail=s.ProcessJobs_GET_OkResponseSchema.description, request=request)


@s.ProcessJobsAPI.post(tags=[s.TagProcesses, s.TagJobs],
                       schema=s.ProcessJobs_POST_RequestSchema(), response_schemas=s.ProcessJobs_POST_responses)
def post_process_jobs_view(request):
    """Execute a registered process job."""
    process = get_process(request)
    try:
        http = create_process_job(request, process)
        http_meta = ex.valid_http(httpSuccess=type(http), detail=s.ProcessJobs_POST_AcceptedResponseSchema.description,
                                  content=http.json, request=request)
        http_meta.location = http.location
        return http_meta
    except HTTPException as http_err:
        raise
    except Exception as err:
        detail = getattr(err, "detail") or getattr(err, "message")
        ex.raise_http(httpError=HTTPInternalServerError, detail=detail, request=request)


def get_process_job_handler(request):
    """Handles cases of Job UUID, latest and current."""
    process = get_process(request)
    job = get_job(request)
    ex.verify_param(job.process, paramCompare=process.uuid, isEqual=True, httpError=HTTPNotFound,
                    msgOnFail=s.ProcessJob_GET_NotFoundResponseSchema.description, request=request)
    database_factory(request).actions_store.save_action(job, OPERATION.INFO, request=request)
    return ex.valid_http(httpSuccess=HTTPOk, content={u"job": job.json()},
                         detail=s.Process_GET_OkResponseSchema.description, request=request)


@s.ProcessJobAPI.get(tags=[s.TagProcesses, s.TagJobs],
                     schema=s.ProcessJob_GET_Endpoint(), response_schemas=s.ProcessJob_GET_responses)
def get_process_job_view(request):
    """Get registered process job information."""
    return get_process_job_handler(request)


@s.ProcessJobCurrentAPI.get(tags=[s.TagProcesses, s.TagJobs],
                            schema=s.ProcessJobCurrentEndpoint(), response_schemas=s.ProcessJob_GET_responses)
def get_process_job_current_view(request):
    """Get currently running process job information."""
    return get_process_job_handler(request)


@s.ProcessJobLatestAPI.get(tags=[s.TagProcesses, s.TagJobs],
                           schema=s.ProcessJobLatestEndpoint(), response_schemas=s.ProcessJob_GET_responses)
def get_process_job_latest_view(request):
    """Get latest successful process job execution."""
    return get_process_job_handler(request)


@s.ProcessJobAPI.put(tags=[s.TagProcesses, s.TagJobs],
                     schema=s.ProcessJob_PUT_Endpoint(), response_schemas=s.ProcessJob_PUT_responses)
def put_process_job_view(request):
    """Update registered job information."""
    job = update_job_params(request)
    db = database_factory(request)
    db.actions_store.save_action(job, OPERATION.UPDATE, request=request)
    return ex.valid_http(httpSuccess=HTTPOk, content={"job": job.summary()},
                         detail=s.ProcessJob_PUT_OkResponseSchema.description, request=request)


@s.ProcessJobResultAPI.get(tags=[s.TagProcesses], renderer="json",
                           schema=s.ProcessJobResultEndpoint(), response_schemas=s.ProcessJobResult_GET_responses)
def get_process_job_results(request):
    """
    Retrieve the result of a job.
    """
    job = get_job(request)
    result = dict(outputs=[
        {
            "id": get_any_id(res),
            "value" if has_raw_value(res) else "href": get_any_value(res)
        }
        for res in job.results
    ])
    database_factory(request).actions_store.save_action(job, OPERATION.INFO, request=request)
    return ex.valid_http(httpSuccess=HTTPOk, content=result,
                         detail=s.ProcessJobResult_GET_OkResponseSchema.description, request=request)


@s.ProcessJobLogsAPI.get(tags=[s.TagProcesses], renderer="json",
                         schema=s.ProcessJobLogsEndpoint(), response_schemas=s.ProcessJobLogs_GET_responses)
def get_process_job_logs(request):
    """
    Retrieve the logs of a job.
    """
    job = get_job(request)
    database_factory(request).actions_store.save_action(job, OPERATION.INFO, request=request)
    return ex.valid_http(httpSuccess=HTTPOk, content={"logs": job.logs},
                         detail=s.ProcessJobLogs_GET_OkResponseSchema.description, request=request)


@s.ProcessJobExceptionsAPI.get(tags=[s.TagProcesses], renderer="json",
                               schema=s.ProcessJobExceptionsEndpoint(),
                               response_schemas=s.ProcessJobExceptions_GET_responses)
def get_process_job_exceptions(request):
    """
    Retrieve the exceptions of a job.
    """
    job = get_job(request)
    database_factory(request).actions_store.save_action(job, OPERATION.INFO, request=request)
    return ex.valid_http(httpSuccess=HTTPOk, content={"exceptions": job.exceptions},
                         detail=s.ProcessJobExceptions_GET_OkResponseSchema.description, request=request)
