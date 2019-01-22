import logging
logger = logging.getLogger(__name__)


def includeme(config):
    logger.info('Adding geoimagenet_ml...')
    config.include('geoimagenet_ml.api')
    config.include('geoimagenet_ml.config')
    config.include('geoimagenet_ml.ml')
    config.include('geoimagenet_ml.processes')
    config.include('geoimagenet_ml.store')
