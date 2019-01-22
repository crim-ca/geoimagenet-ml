#!/usr/bin/env python
# coding: utf-8
import os
import sys
import logging
logger = logging.getLogger(__name__)

# define common root paths and allow imports from top-level dirs
CCFB_API_DIR = os.path.abspath(os.path.dirname(__file__))
CCFB_PROJECT_DIR = os.path.abspath(os.path.dirname(CCFB_API_DIR))
CCFB_PROJECT_CONFIG_DIR = os.path.abspath(os.path.join(CCFB_PROJECT_DIR, 'config'))
CCFB_ROOT_DIR = os.path.abspath(os.path.dirname(CCFB_PROJECT_DIR))
CCFB_ROOT_CONFIG_DIR = os.path.abspath(os.path.join(CCFB_ROOT_DIR, 'config'))
try:
    import ccfb
except (ImportError, ModuleNotFoundError):
    # resolve rest_api location before install using local dir package structure
    sys.path.insert(0, CCFB_ROOT_DIR)
    sys.path.insert(0, CCFB_API_DIR)


def includeme(config):
    logger.info('Adding rest_api config modules...')
    config.include('cornice')
    config.include('cornice_swagger')
    config.include('pyramid_chameleon')
    config.include('pyramid_mako')
    config.include('src.api.definitions')
    config.include('src.api.processes')
    config.include('src.api.rest_api')
    config.include('src.api.store')
