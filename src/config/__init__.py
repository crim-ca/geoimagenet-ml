import os
import logging
os.environ['CELERY_CONFIG_MODULE'] = 'src.config.celeryconfig'
LOGGER = logging.getLogger(__name__)


# noinspection PyUnusedLocal
def includeme(config):
    LOGGER.info('Adding config...')
