#!/usr/bin/env python
# coding: utf-8

"""
GeoImageNet ML API
"""
from geoimagenet_ml.store.factories import migrate_database_when_ready
from geoimagenet_ml.store.exceptions import JobExecutionError
from pyramid.config import Configurator
from pyramid.httpexceptions import HTTPClientError
from pyramid.security import NO_PERMISSION_REQUIRED
from pyramid.settings import asbool
import logging
import socket
import os
LOGGER = logging.getLogger(__name__)

sentry_dsn = os.getenv("GEOIMAGENET_ML_SENTRY_DSN")
sentry_debug = asbool(os.getenv("GEOIMAGENET_ML_SENTRY_DEBUG", False))
sentry_server_name = os.getenv("GEOIMAGENET_ML_SENTRY_SERVER_NAME", socket.gethostname())
if sentry_dsn:
    import sentry_sdk
    from sentry_sdk.integrations.celery import CeleryIntegration
    from sentry_sdk.integrations.pyramid import PyramidIntegration

    def before_send(event, hint):
        """
        Make Sentry filter and group more aggressively corresponding Job errors to avoid
        duplication of similar operations because of differing UUID or other unique fields.

        Also ignore expected HTTP Client errors (4xx) to only report unhandled HTTP Server errors (5xx).
        """
        task = event.get("extra", {}).get("celery-job", {})
        log_msg = event.get("logentry", {}).get("message")
        if not task or not log_msg:
            return event
        if "exc_info" in hint:
            exc_type, exc_value, tb = hint["exc_info"]
            if isinstance(exc_value, HTTPClientError):
                return  # empty drops the event reporting
        task_name = "geoimagenet_ml.api.routes.processes.utils.process_job_runner"
        if task.get("task_name") == task_name and "failed to run" in log_msg:
            try:
                # see format of logged job error for parsing
                task_msg = log_msg.split("[", 1)[-1].rsplit("]", 1)[0]
            except (IndexError, KeyError):
                task_msg = log_msg
            event["fingerprint"] = [JobExecutionError.__name__, task_msg]
            event["logentry"]["message"] = task_msg
        return event

    sentry_sdk.init(sentry_dsn, debug=sentry_debug, server_name=sentry_server_name,
                    integrations=[CeleryIntegration(), PyramidIntegration()], before_send=before_send)


# noinspection PyUnusedLocal
def main(global_config=None, **settings):
    """
    This function returns a Pyramid WSGI application.
    """

    # migrate db as required and check if database is ready
    migrate_database_when_ready(settings)

    protocol = os.getenv("GEOIMAGENET_ML_API_PROTOCOL", "http")
    hostname = os.getenv("GEOIMAGENET_ML_API_HOSTNAME") or os.getenv("HOSTNAME")
    port = os.getenv("GEOIMAGENET_ML_API_PORT")
    url = os.getenv("GEOIMAGENET_ML_API_URL")
    if url:
        settings["geoimagenet_ml.api.url"] = url
    elif hostname:
        # update variables from ini settings if not overridden in environment
        if protocol is None:
            protocol = settings["geoimagenet_ml.api.protocol"]
        if port is None:
            port = settings["geoimagenet_ml.api.port"]
        port = "{sep}{port}".format(port=port, sep=':' if port else '')
        api_url_template = "{protocol}://{hostname}{port}"
        settings["geoimagenet_ml.api.url"] = api_url_template.format(protocol=protocol, hostname=hostname, port=port)

    # setup logger override level
    log_lvl = os.getenv("GEOIMAGENET_ML_LOG_LEVEL")
    if log_lvl:
        root_logger = logging.getLogger("geoimagenet_ml")
        root_logger.setLevel(log_lvl)

    config = Configurator(settings=settings)
    config.include("geoimagenet_ml")
    config.set_default_permission(NO_PERMISSION_REQUIRED)

    wsgi_app = config.make_wsgi_app()
    return wsgi_app


if __name__ == "__main__":
    main()
