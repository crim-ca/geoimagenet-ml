#!/usr/bin/env python
# coding: utf-8
from geoimagenet_ml.api import schemas as s
from geoimagenet_ml.api.routes import generic as g
import logging
logger = logging.getLogger(__name__)


def includeme(config):
    logger.info('Adding API routes...')
    config.include('geoimagenet_ml.api.routes.datasets')
    config.include('geoimagenet_ml.api.routes.models')
    config.include('geoimagenet_ml.api.routes.processes')

    config.add_route(**s.service_api_route_info(s.BaseAPI))
    config.add_route(**s.service_api_route_info(s.SwaggerJSON))
    config.add_route(**s.service_api_route_info(s.SwaggerAPI))
    config.add_route(**s.service_api_route_info(s.VersionsAPI))
    config.add_view(g.get_api_base_view, route_name=s.BaseAPI.name,
                    request_method='GET', renderer='json')
    config.add_view(g.api_swagger_json_view, route_name=s.SwaggerJSON.name,
                    request_method='GET', renderer='json')
    config.add_view(g.api_swagger_ui_view, route_name=s.SwaggerAPI.name,
                    request_method='GET', renderer='templates/swagger_ui.mako')
    config.add_view(g.get_version_view, route_name=s.VersionsAPI.name,
                    request_method='GET', renderer='json')
