#!/usr/bin/env python
# coding: utf-8

"""
GeoImageNet ML API
"""
from geoimagenet_ml.store.factories import migrate_database_when_ready
from pyramid.config import Configurator
from pyramid.security import NO_PERMISSION_REQUIRED
import os
import logging
LOGGER = logging.getLogger(__name__)


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