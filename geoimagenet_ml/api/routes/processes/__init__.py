#!/usr/bin/env python
# coding: utf-8
from geoimagenet_ml.api import schemas as s
from geoimagenet_ml.api.routes.processes import views as v
import logging
logger = logging.getLogger(__name__)


def includeme(config):
    logger.info("Adding routes processes...")
    config.add_route(**s.service_api_route_info(s.ProcessesAPI))
    config.add_route(**s.service_api_route_info(s.ProcessAPI))
    config.add_route(**s.service_api_route_info(s.ProcessJobAPI))
    config.add_route(**s.service_api_route_info(s.ProcessJobCurrentAPI))
    config.add_route(**s.service_api_route_info(s.ProcessJobLatestAPI))
    config.add_route(**s.service_api_route_info(s.ProcessJobsAPI))
    config.add_route(**s.service_api_route_info(s.ProcessJobResultAPI))
    config.add_route(**s.service_api_route_info(s.ProcessJobLogsAPI))
    config.add_route(**s.service_api_route_info(s.ProcessJobExceptionsAPI))
    config.add_view(v.get_processes_view, route_name=s.ProcessesAPI.name,
                    request_method="GET", renderer="json")
    config.add_view(v.post_processes_view, route_name=s.ProcessesAPI.name,
                    request_method="POST", renderer="json")
    config.add_view(v.get_process_view, route_name=s.ProcessAPI.name,
                    request_method="GET", renderer="json")
    config.add_view(v.put_process_job_view, route_name=s.ProcessJobAPI.name,
                    request_method="PUT", renderer="json")
    config.add_view(v.get_process_job_view, route_name=s.ProcessJobAPI.name,
                    request_method="GET", renderer="json")
    config.add_view(v.get_process_job_current_view, route_name=s.ProcessJobCurrentAPI.name,
                    request_method="GET", renderer="json")
    config.add_view(v.get_process_job_latest_view, route_name=s.ProcessJobLatestAPI.name,
                    request_method="GET", renderer="json")
    config.add_view(v.get_process_jobs_view, route_name=s.ProcessJobsAPI.name,
                    request_method="GET", renderer="json")
    config.add_view(v.post_process_jobs_view, route_name=s.ProcessJobsAPI.name,
                    request_method="POST", renderer="json")
    config.add_view(v.get_process_job_results, route_name=s.ProcessJobResultAPI.name,
                    request_method="GET", renderer="json")
    config.add_view(v.get_process_job_logs, route_name=s.ProcessJobLogsAPI.name,
                    request_method="GET", renderer="json")
    config.add_view(v.get_process_job_exceptions, route_name=s.ProcessJobExceptionsAPI.name,
                    request_method="GET", renderer="json")
