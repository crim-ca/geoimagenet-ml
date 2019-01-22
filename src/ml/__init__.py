# noinspection PyPackageRequirements
from thelper import __version__ as thelper_version
import logging
logger = logging.getLogger("src")


__type__ = "thelper"
__version__ = thelper_version


# noinspection PyUnusedLocal
def includeme(config):
    logger.info('Adding ML config modules...')
