#!/usr/bin/env python
# coding: utf-8
import os
import sys
import logging
logger = logging.getLogger(__name__)

# define common root paths and allow imports from top-level dirs
GEOIMAGENET_ML_API_DIR = os.path.abspath(os.path.dirname(__file__))
GEOIMAGENET_ML_PROJECT_DIR = os.path.abspath(os.path.dirname(GEOIMAGENET_ML_API_DIR))
GEOIMAGENET_ML_PROJECT_CONFIG_DIR = os.path.abspath(os.path.join(GEOIMAGENET_ML_PROJECT_DIR, 'config'))
GEOIMAGENET_ML_ROOT_DIR = os.path.abspath(os.path.dirname(GEOIMAGENET_ML_PROJECT_DIR))
GEOIMAGENET_ML_ROOT_CONFIG_DIR = os.path.abspath(os.path.join(GEOIMAGENET_ML_ROOT_DIR, 'config'))
try:
    # noinspection PyUnresolvedReferences
    import src
except (ImportError, ModuleNotFoundError):
    # resolve routes location before install using local dir package structure
    sys.path.insert(0, GEOIMAGENET_ML_ROOT_DIR)
    sys.path.insert(0, GEOIMAGENET_ML_API_DIR)


def includeme(config):
    logger.info('Adding API config modules...')
    config.include('cornice')
    config.include('cornice_swagger')
    config.include('pyramid_chameleon')
    config.include('pyramid_mako')
    config.include('geoimagenet_ml.api.routes')
