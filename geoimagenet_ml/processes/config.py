import os
import tempfile
import logging
LOGGER = logging.getLogger(__name__)


def _workdir(request):
    settings = request.registry.settings
    workdir = settings.get('geoimagenet_ml.workdir')
    workdir = workdir or tempfile.gettempdir()
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    LOGGER.debug('using workdir %s', workdir)
    return workdir


def _prefix(request):
    settings = request.registry.settings
    prefix = settings.get('geoimagenet_ml.prefix')
    prefix = prefix or 'GEOIMAGENET_ML_'
    return prefix


def includeme(config):
    LOGGER.debug("Loading processes configuration.")
    config.add_request_method(_workdir, 'workdir', reify=True)
    config.add_request_method(_prefix, 'prefix', reify=True)
