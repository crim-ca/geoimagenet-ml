import logging
logger = logging.getLogger(__name__)


def includeme(config):
    logger.info('Adding src...')
    config.include('src.api')
    config.include('src.config')
    config.include('src.ml')
    config.include('src.processes')
    config.include('src.store')
