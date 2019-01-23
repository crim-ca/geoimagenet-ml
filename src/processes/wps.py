"""
pywps 4.x wrapper
"""

from geoimagenet_ml.typedefs import AnyStr, Union, Optional, SettingDict  # noqa: F401
from geoimagenet_ml.processes.types import PROCESS_WPS
from geoimagenet_ml.processes.utils import get_base_url
from pyramid.wsgi import wsgiapp2
from pyramid.threadlocal import get_current_request
from pyramid.registry import Registry  # noqa: F401
from pyramid.httpexceptions import HTTPInternalServerError
from pyramid_celery import celery_app as app
# noinspection PyPackageRequirements
from pywps import configuration as pywps_config
# noinspection PyPackageRequirements
from pywps.app.Service import Service
from six.moves.configparser import ConfigParser
import os
import six
import logging
import warnings
LOGGER = logging.getLogger(__name__)

# can be overridden with 'settings.wps-cfg'
DEFAULT_PYWPS_CFG = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'wps.cfg')
PYWPS_CFG = None


def _get_settings_or_wps_config(
        settings,                   # type: SettingDict
        project_setting_name,       # type: AnyStr
        config_setting_section,     # type: AnyStr
        config_setting_name,        # type: AnyStr
        default_not_found,          # type: AnyStr
        message_not_found,          # type: AnyStr
        ):                          # type: (...) -> AnyStr
    wps_path = settings.get(project_setting_name)
    if not wps_path:
        wps_cfg = get_wps_cfg_path(settings)
        config = ConfigParser()
        config.read(wps_cfg)
        wps_path = config.get(config_setting_section, config_setting_name)
    if not isinstance(wps_path, six.string_types):
        warnings.warn("{} not set in settings or WPS configuration, using default value.".format(message_not_found))
        wps_path = default_not_found
    return wps_path.rstrip('/').strip()


def get_wps_cfg_path(settings):
    # type: (SettingDict) -> AnyStr
    """
    Retrieves the WPS configuration file (`wps.cfg` by default or `geoimagenet_ml.wps_cfg` if specified).
    """
    return settings.get('geoimagenet_ml.wps_cfg', DEFAULT_PYWPS_CFG)


def get_wps_path(settings):
    # type: (SettingDict) -> AnyStr
    """
    Retrieves the WPS path (without hostname).
    Searches directly in settings, then `geoimagenet_ml.wps_cfg` file, or finally, uses the default values if not found.
    """
    wps_path = _get_settings_or_wps_config(settings, 'geoimagenet_ml.wps_path', 'server', 'url', '/wps', 'WPS path')
    GEOIMAGENET_ML_url = get_base_url(settings)
    if wps_path.startswith(GEOIMAGENET_ML_url):
        return wps_path.replace(GEOIMAGENET_ML_url, '')
    return wps_path


def get_wps_url(settings):
    # type: (SettingDict) -> AnyStr
    """
    Retrieves the full WPS URL (hostname + WPS path).
    Searches directly in settings, then `geoimagenet_ml.wps_cfg` file, or finally, uses the default values if not found.
    """
    return get_base_url(settings) + get_wps_path(settings)


def get_wps_output_path(settings):
    # type: (SettingDict) -> AnyStr
    """
    Retrieves the WPS output path directory where to write XML and result files.
    Searches directly in settings, then `geoimagenet_ml.wps_cfg` file, or finally, uses the default values if not found.
    """
    return _get_settings_or_wps_config(
        settings, 'geoimagenet_ml.wps_output_path', 'server', 'outputpath', '/tmp', 'WPS output path')


def get_wps_output_url(settings):
    # type: (SettingDict) -> AnyStr
    """
    Retrieves the WPS output URL that maps to WPS output path directory.
    Searches directly in settings, then `geoimagenet_ml.wps_cfg` file, or finally, uses the default values if not found.
    """
    wps_output_default = get_base_url(settings) + '/wpsoutputs'
    return _get_settings_or_wps_config(
        settings, 'geoimagenet_ml.wps_output_url', 'server', 'outputurl', wps_output_default, 'WPS output url')


def load_pywps_cfg(registry, config=None):
    # type: (Registry, Optional[Union[AnyStr, SettingDict]]) -> None
    global PYWPS_CFG

    if PYWPS_CFG is None:
        # get PyWPS config
        file_config = config if isinstance(config, six.string_types) or isinstance(config, list) else None
        pywps_config.load_configuration(file_config or get_wps_cfg_path(registry.settings))
        PYWPS_CFG = pywps_config

    # add additional config passed as dictionary of {'section.key': 'value'}
    if isinstance(config, dict):
        for key, value in config.items():
            section, key = key.split('.')
            PYWPS_CFG.CONFIG.set(section, key, value)
        # cleanup alternative dict 'PYWPS_CFG' which is not expected elsewhere
        if isinstance(registry.settings.get('PYWPS_CFG'), dict):
            del registry.settings['PYWPS_CFG']

    if 'geoimagenet_ml.wps_output_path' not in registry.settings:
        # ensure the output dir exists if specified
        out_dir_path = PYWPS_CFG.get_config_value('server', 'outputpath')
        if not os.path.isdir(out_dir_path):
            os.makedirs(out_dir_path)
        registry.settings['geoimagenet_ml.wps_output_path'] = out_dir_path

    if 'geoimagenet_ml.wps_output_url' not in registry.settings:
        output_url = PYWPS_CFG.get_config_value('server', 'outputurl')
        registry.settings['geoimagenet_ml.wps_output_url'] = output_url


# @app.task(bind=True)
@wsgiapp2
def pywps_view(environ, start_response):
    LOGGER.debug('pywps env: %s', environ.keys())

    try:
        registry = app.conf['PYRAMID_REGISTRY']

        # get config file
        pywps_cfg = environ.get('PYWPS_CFG') or registry.settings.get('PYWPS_CFG')
        if not pywps_cfg:
            environ['PYWPS_CFG'] = os.getenv('PYWPS_CFG') or get_wps_cfg_path(registry.settings)
        load_pywps_cfg(registry, config=pywps_cfg)

        # call pywps application with processes filtered according to the adapter's definition
        from geoimagenet_ml.store.factories import database_factory
        process_store = database_factory(registry).processes_store
        processes_wps = [process.wps() for process in
                         process_store.list_processes(request=get_current_request())
                         if process.type == PROCESS_WPS]
        service = Service(processes_wps)
    except Exception as ex:
        raise HTTPInternalServerError("Failed setup of PyWPS Service and/or Processes. Error [{}]".format(ex))

    return service(environ, start_response)


def includeme(config):
    LOGGER.info("WPS enabled.")
    wps_path = get_wps_path(config.registry.settings)
    config.add_route('wps', wps_path)
    config.add_view(pywps_view, route_name='wps')
    config.add_request_method(lambda req: get_wps_cfg_path(req.registry.settings), 'wps_cfg', reify=True)
