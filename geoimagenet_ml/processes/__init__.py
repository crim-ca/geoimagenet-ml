#!/usr/bin/env python
# coding: utf-8
import os
import sys
import logging
LOGGER = logging.getLogger(__name__)


# noinspection PyUnusedLocal
def includeme(config):
    LOGGER.info('Adding processes...')
    from geoimagenet_ml import GEOIMAGENET_ML_CONFIG_DIR, GEOIMAGENET_ML_CONFIG_INI

    # help find celeryconfig.py for pyramid_celery
    config.include('geoimagenet_ml.config')
    sys.path.insert(0, os.path.abspath(GEOIMAGENET_ML_CONFIG_DIR))

    # include modules where celery tasks are defined to register them
    config.include('geoimagenet_ml.processes')

    # include celery
    config.include('pyramid_celery')
    config.configure_celery(GEOIMAGENET_ML_CONFIG_INI)
