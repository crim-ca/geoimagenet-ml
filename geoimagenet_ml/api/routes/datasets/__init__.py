#!/usr/bin/env python
# coding: utf-8
from geoimagenet_ml.api import schemas as s
from geoimagenet_ml.api.routes.datasets import views as v
import logging
logger = logging.getLogger(__name__)


def includeme(config):
    logger.info("Adding routes datasets...")
    config.add_route(**s.service_api_route_info(s.DatasetsAPI))
    config.add_route(**s.service_api_route_info(s.DatasetAPI))
    config.add_route(**s.service_api_route_info(s.DatasetLatestAPI))
    config.add_route(**s.service_api_route_info(s.DatasetDownloadAPI))
    config.add_view(v.get_datasets_view, route_name=s.DatasetsAPI.name,
                    request_method="GET", renderer="json")
    config.add_view(v.post_datasets_view, route_name=s.DatasetsAPI.name,
                    request_method="POST", renderer="json")
    config.add_view(v.get_dataset_view, route_name=s.DatasetAPI.name,
                    request_method="GET", renderer="json")
    config.add_view(v.delete_dataset_view, route_name=s.DatasetAPI.name,
                    request_method="DELETE", renderer="json")
    config.add_view(v.get_dataset_latest_view, route_name=s.DatasetLatestAPI.name,
                    request_method="GET", renderer="json")
    config.add_view(v.download_dataset_view, route_name=s.DatasetDownloadAPI.name,
                    request_method="GET", renderer="json")
