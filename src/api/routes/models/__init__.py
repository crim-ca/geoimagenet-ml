#!/usr/bin/env python
# coding: utf-8
from geoimagenet_ml.api.rest_api import schemas as s
from geoimagenet_ml.api.rest_api.models import views as v
import logging
logger = logging.getLogger(__name__)


def includeme(config):
    logger.info('Adding routes models...')
    config.add_route(**s.service_api_route_info(s.ModelsAPI))
    config.add_route(**s.service_api_route_info(s.ModelAPI))
    config.add_route(**s.service_api_route_info(s.ModelDownloadAPI))
    config.add_view(v.get_models_view, route_name=s.ModelsAPI.name,
                    request_method='GET', renderer='json')
    config.add_view(v.post_models_view, route_name=s.ModelsAPI.name,
                    request_method='POST', renderer='json')
    config.add_view(v.get_model_view, route_name=s.ModelAPI.name,
                    request_method='GET', renderer='json')
    config.add_view(v.download_model_view, route_name=s.ModelDownloadAPI.name,
                    request_method='GET', renderer='')
