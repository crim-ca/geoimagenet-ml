#!/usr/bin/env python
# coding: utf-8
from src.api.rest_api.processes.utils import create_process, get_process, get_job, create_process_job
from src.api.utils import get_any_id, get_any_value, has_raw_value
from src.api.routes import exceptions as ex, schemas as s
from src.store.factories import database_factory
from pyramid.httpexceptions import (
    HTTPOk,
    HTTPCreated,
    HTTPForbidden,
    HTTPNotFound,
    HTTPInternalServerError,
)


@s.ProcessesAPI.get(tags=[s.ProcessesTag], response_schemas=s.Processes_GET_responses)
def get_processes_view(request):
    """Get registered processes information."""
    processes_list = ex.evaluate_call(lambda: database_factory(request.registry)
                                      .processes_store.list_processes(request=request),
                                      fallback=lambda: request.db.rollback(), httpError=HTTPForbidden, request=request,
                                      msgOnFail=s.Processes_GET_ForbiddenResponseSchema.description)
    processes_json = ex.evaluate_call(lambda: [p.summary() for p in processes_list],
                                      fallback=lambda: request.db.rollback(), httpError=HTTPInternalServerError,
                                      request=request, msgOnFail=s.InternalServerErrorResponseSchema.description)
    return ex.valid_http(httpSuccess=HTTPOk, content={u'processes': processes_json},
                         detail=s.Processes_GET_OkResponseSchema.description, request=request)


@s.ProcessesAPI.post(tags=[s.ProcessesTag],
                     schema=s.Processes_POST_RequestSchema(), response_schemas=s.Processes_POST_responses)
def post_processes_view(request):
    """Register a new process."""
    process = create_process(request)
    return ex.valid_http(httpSuccess=HTTPCreated, content={u'process': process.json()},
                         detail=s.Processes_POST_CreatedResponseSchema.description, request=request)


@s.ProcessAPI.get(tags=[s.ProcessesTag],
                  schema=s.ProcessEndpoint(), response_schemas=s.Process_GET_responses)
def get_process_view(request):
    """Get registered process information."""
    process = get_process(request)
    return ex.valid_http(httpSuccess=HTTPOk, content={u'process': process.json()},
                         detail=s.Process_GET_OkResponseSchema.description, request=request)


@s.ProcessJobsAPI.get(tags=[s.ProcessesTag],
                      schema=s.ProcessJobsEndpoint(), response_schemas=s.ProcessJobs_GET_responses)
def get_process_jobs_view(request):
    """Get registered process jobs."""
    process = get_process(request)
    job_list = ex.evaluate_call(lambda: database_factory(request.registry).jobs_store.list_jobs(request=request),
                                fallback=lambda: request.db.rollback(), httpError=HTTPForbidden, request=request,
                                msgOnFail=s.ProcessJobs_GET_ForbiddenResponseSchema.description)
    process_jobs_json = ex.evaluate_call(lambda: [j.summary() for j in
                                                  filter(lambda j: j.process_uuid == process.uuid, job_list)],
                                         fallback=lambda: request.db.rollback(), httpError=HTTPInternalServerError,
                                         request=request, msgOnFail=s.InternalServerErrorResponseSchema.description)
    return ex.valid_http(httpSuccess=HTTPOk, content={u'jobs': process_jobs_json},
                         detail=s.ProcessJobs_GET_OkResponseSchema.description, request=request)


@s.ProcessJobsAPI.post(tags=[s.ProcessesTag],
                       schema=s.ProcessJobs_POST_RequestSchema(), response_schemas=s.ProcessJobs_POST_responses)
def post_process_jobs_view(request):
    """Execute a registered process job."""
    process = get_process(request)
    return create_process_job(request, process)


@s.ProcessJobAPI.get(tags=[s.ProcessesTag],
                     schema=s.ProcessJobEndpoint(), response_schemas=s.ProcessJob_GET_responses)
def get_process_job_view(request):
    """Get registered process job status."""
    process = get_process(request)
    job = get_job(request)
    ex.verify_param(job.process_uuid, paramCompare=process.uuid, isEqual=True, httpError=HTTPNotFound,
                    msgOnFail=s.ProcessJob_GET_NotFoundResponseSchema.description, request=request)
    return ex.valid_http(httpSuccess=HTTPOk, content={u'job': job.json()},
                         detail=s.Process_GET_OkResponseSchema.description, request=request)


@s.ProcessJobResultAPI.get(tags=[s.ProcessesTag], renderer='json',
                           schema=s.ProcessJobResultEndpoint(), response_schemas=s.ProcessJobResult_GET_responses)
def get_process_job_results(request):
    """
    Retrieve the result of a job.
    """
    job = get_job(request)
    result = dict(outputs=[
        {
            'id': get_any_id(res),
            'value' if has_raw_value(res) else 'href': get_any_value(res)
        }
        for res in job.results
    ])
    return ex.valid_http(httpSuccess=HTTPOk, content=result,
                         detail=s.ProcessJobResult_GET_OkResponseSchema.description, request=request)


@s.ProcessJobLogsAPI.get(tags=[s.ProcessesTag], renderer='json',
                         schema=s.ProcessJobLogsEndpoint(), response_schemas=s.ProcessJobLogs_GET_responses)
def get_process_job_logs(request):
    """
    Retrieve the logs of a job.
    """
    job = get_job(request)
    return ex.valid_http(httpSuccess=HTTPOk, content={'logs': job.logs},
                         detail=s.ProcessJobResult_GET_OkResponseSchema.description, request=request)


@s.ProcessJobExceptionsAPI.get(tags=[s.ProcessesTag], renderer='json',
                               schema=s.ProcessJobExceptionsEndpoint(),
                               response_schemas=s.ProcessJobExceptions_GET_responses)
def get_process_job_exceptions(request):
    """
    Retrieve the exceptions of a job.
    """
    job = get_job(request)
    return ex.valid_http(httpSuccess=HTTPOk, content={'exceptions': job.exceptions},
                         detail=s.ProcessJobResult_GET_OkResponseSchema.description, request=request)
