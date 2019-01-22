#!/usr/bin/env python
# coding: utf-8
import os
import sys
import logging
LOGGER = logging.getLogger(__name__)


# noinspection PyUnusedLocal
def includeme(config):
    LOGGER.info('Adding processes...')
    from ccfb.api import CCFB_PROJECT_CONFIG_DIR

    # help find celeryconfig.py for pyramid_celery
    config.include('src.config')
    sys.path.insert(0, os.path.abspath(CCFB_PROJECT_CONFIG_DIR))

    # include modules where celery tasks are defined to register them
    config.include('src.api.processes')

    # include celery
    config.include('pyramid_celery')
    config.configure_celery(os.path.join(CCFB_PROJECT_CONFIG_DIR, 'src.ini'))
