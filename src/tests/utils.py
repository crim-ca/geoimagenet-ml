import os
import six
import requests
import pyramid
# noinspection PyPackageRequirements
import pyramid.testing
from pyramid.config import Configurator
from pyramid.response import Response
from distutils.version import *
# noinspection PyPackageRequirements
from webtest import TestApp
# noinspection PyPackageRequirements
from webtest.response import TestResponse
from ccfb.api import __meta__
from ccfb.api.utils import settings_from_ini
from ccfb.api.definitions.typing_definitions import Any, AnyStr, Union, Optional, SettingDict
from ccfb.api.store.databases.types import MONGODB_TYPE


json_headers = [('Content-Type', 'application/json')]


def setup_test_app(settings=None, config=None):
    # type: (Optional[SettingDict], Optional[Configurator]) -> TestApp
    config = setup_config_from_settings(settings=settings, config=config)
    # scan dependencies
    config.include('src.api')
    # create the test application
    app = TestApp(config.make_wsgi_app())
    return app


def setup_config_from_settings(settings=None, config=None):
    # type: (Optional[SettingDict], Optional[Configurator]) -> Configurator
    ccfb_api_ini_path = os.getenv('CCFB_PROJECT_CONFIG_API_INI_PATH')
    if not isinstance(ccfb_api_ini_path, six.string_types):
        raise ValueError("API configuration file required for testing, please set 'CCFB_PROJECT_CONFIG_API_INI_PATH'.")
    if not os.path.isfile(ccfb_api_ini_path):
        raise ValueError("API configuration file cannot be retrieved for testing: [{!s}].".format(ccfb_api_ini_path))
    settings_ini = settings_from_ini(ccfb_api_ini_path, 'app:ccfb_app')
    if settings:
        settings_ini.update(settings)
    if config:
        settings_ini.update(config.registry.settings)
    config = pyramid.testing.setUp(settings=settings_ini)
    return config


def setup_config_with_mongodb(settings=None, config=None):
    # type: (Optional[SettingDict], Optional[Configurator]) -> Configurator
    settings_db = {
        'mongodb.host': '127.0.0.1',
        'mongodb.port': '27027',
        'mongodb.db_name': 'src-test',
        'src.api.db_factory': MONGODB_TYPE,
        'src.api.models_path': '/tmp'  # place models somewhere they will be deleted periodically
    }
    if settings:
        settings_db.update(settings)
    if config:
        settings_db.update(config.registry.settings)
    return setup_config_from_settings(settings=settings_db, config=config)


def request(app_or_url,             # type: Union[TestApp, AnyStr]
            method,                 # type: AnyStr
            path,                   # type: AnyStr
            timeout=5,              # type: Optional[int]
            allow_redirects=True,   # type: Optional[bool]
            **kwargs                # type: Any
            ):                      # type: (...) -> Union[TestResponse, Response]
    """
    Calls the request using either a `webtest.TestApp` instance or a `requests` instance from a string URL.
    :param app_or_url: `webtest.TestApp` instance of the test application or remote server URL to call with `requests`
    :param method: request method (GET, POST, PUT, DELETE)
    :param path: test path starting at base path
    :param timeout: request maximum timeout
    :param allow_redirects: allow following redirect responses
    :return: response of the request
    """
    method = method.upper()

    # obtain json body from any json/data/body/params kw and empty {} if not specified
    # reapply with the expected webtest/requests method kw afterward
    json_body = None
    for kw in ['json', 'data', 'body', 'params']:
        json_body = kwargs.get(kw, json_body)
        if kw in kwargs:
            kwargs.pop(kw)
    json_body = json_body or {}

    if isinstance(app_or_url, TestApp):
        # remove any 'cookies' keyword handled by the 'TestApp' instance
        if 'cookies' in kwargs:
            kwargs.pop('cookies')

        kwargs['params'] = json_body
        if method == 'GET':
            return app_or_url.get(path, **kwargs)
        elif method == 'POST':
            return app_or_url.post_json(path, **kwargs)
        elif method == 'PUT':
            return app_or_url.put_json(path, **kwargs)
        elif method == 'DELETE':
            return app_or_url.delete_json(path, **kwargs)
    else:
        kwargs['json'] = json_body
        url = '{url}{path}'.format(url=app_or_url, path=path)
        return requests.request(method, url, timeout=timeout, allow_redirects=allow_redirects, **kwargs)


def format_test_val_ref(val, ref, pre='Fail'):
    return '({0}) Test value: `{1}`, Reference value: `{2}`'.format(pre, val, ref)


def all_equal(iter_val, iter_ref, any_order=False):
    if not (hasattr(iter_val, '__iter__') and hasattr(iter_ref, '__iter__')):
        return False
    if len(iter_val) != len(iter_ref):
        return False
    if any_order:
        return all([it in iter_ref for it in iter_val])
    return all(it == ir for it, ir in zip(iter_val, iter_ref))


def check_all_equal(iter_val, iter_ref, any_order=False, msg=None):
    r_it_val = repr(iter_val)
    r_it_ref = repr(iter_ref)
    assert all_equal(iter_val, iter_ref, any_order), msg or format_test_val_ref(r_it_val, r_it_ref, pre='Equal Fail')


def check_val_equal(val, ref, msg=None):
    assert isinstance(ref, null) or val == ref, msg or format_test_val_ref(val, ref, pre='Equal Fail')


def check_val_not_equal(val, ref, msg=None):
    assert isinstance(ref, null) or val != ref, msg or format_test_val_ref(val, ref, pre='Equal Fail')


def check_val_is_in(val, ref, msg=None):
    assert isinstance(ref, null) or val in ref, msg or format_test_val_ref(val, ref, pre='Is In Fail')


def check_val_not_in(val, ref, msg=None):
    assert isinstance(ref, null) or val not in ref, msg or format_test_val_ref(val, ref, pre='Not In Fail')


def check_val_type(val, ref, msg=None):
    assert isinstance(val, ref), msg or format_test_val_ref(val, repr(ref), pre='Type Fail')


def check_response_basic_info(response, expected_code=200):
    """
    Validates basic API response metadata.
    :param response: response to validate.
    :param expected_code: status code to validate from the response.
    :return: json body of the response for convenience.
    """
    if isinstance(response, TestResponse):
        json_body = response.json
    else:
        json_body = response.json()
    content_types = [ct.strip() for ct in response.headers['Content-Type'].split(';')]
    check_val_is_in('application/json', content_types)
    check_val_equal(response.status_code, expected_code)
    check_val_is_in('meta', json_body)
    check_val_is_in('data', json_body)
    check_val_type(json_body['meta'], dict)
    check_val_type(json_body['data'], dict)
    check_val_equal(json_body['meta']['code'], expected_code)
    check_val_equal(json_body['meta']['type'], 'application/json')
    assert json_body['meta']['detail'] != ''
    return json_body


class null(object):
    """ Represents a null value to differentiate from None. """
    def __repr__(self):
        return '<Null>'


Null = null()


def check_error_param_structure(json_body, paramValue=Null, paramName=Null, paramCompare=Null,
                                isParamValueLiteralUnicode=False, paramCompareExists=False, version=None):
    """
    Validates error response 'param' information based on different Magpie version formats.
    :param json_body: json body of the response to validate.
    :param paramValue: expected 'value' of param, not verified if <Null>
    :param paramName: expected 'name' of param, not verified if <Null> or non existing for Magpie version
    :param paramCompare: expected 'compare'/'paramCompare' value, not verified if <Null>
    :param isParamValueLiteralUnicode: param value is represented as `u'{paramValue}'` for older Magpie version
    :param paramCompareExists: verify that 'compare'/'paramCompare' is in the body, not necessarily validating the value
    :param version: version of application/remote server to use for format validation, use local Magpie version if None
    :raise failing condition
    """
    check_val_type(json_body, dict)
    check_val_is_in('param', json_body)
    version = version or __meta__.__version__
    if LooseVersion(version) >= LooseVersion('0.6.3'):
        check_val_type(json_body['param'], dict)
        check_val_is_in('value', json_body['param'])
        check_val_is_in('name', json_body['param'])
        check_val_equal(json_body['param']['name'], paramName)
        check_val_equal(json_body['param']['value'], paramValue)
        if paramCompareExists:
            check_val_is_in('compare', json_body['param'])
            check_val_equal(json_body['param']['compare'], paramCompare)
    else:
        # unicode representation was explicitly returned in value only when of string type
        if isParamValueLiteralUnicode and isinstance(paramValue, six.string_types):
            paramValue = u'u\'{}\''.format(paramValue)
        check_val_equal(json_body['param'], paramValue)
        if paramCompareExists:
            check_val_is_in('paramCompare', json_body)
            check_val_equal(json_body['paramCompare'], paramCompare)
