#!/usr/bin/env python
# coding: utf-8

"""
CCFB API
"""
from ccfb.api.store.factories import migrate_database_when_ready
from pyramid.config import Configurator
from pyramid.exceptions import ConfigurationError
from pyramid.security import NO_PERMISSION_REQUIRED
import os
import logging
LOGGER = logging.getLogger(__name__)


# noinspection PyUnusedLocal
def main(global_config=None, **settings):
    """
    This function returns a Pyramid WSGI application.
    """

    # migrate db as required and check if database is ready
    migrate_database_when_ready(settings)

    protocol = os.getenv('CCFB_API_PROTOCOL', 'http')
    hostname = os.getenv('CCFB_API_HOSTNAME') or os.getenv('HOSTNAME')
    port = os.getenv('CCFB_API_PORT', '')
    url = os.getenv('CCFB_API_URL')
    if url:
        settings['src.api.url'] = url
    elif hostname:
        # update variables from ini settings if not overridden in environment
        if protocol is None:
            protocol = settings['src.api.protocol']
        if port is None:
            port = settings['src.api.port']
        port = '{sep}{port}'.format(port=port, sep=':' if port else '')
        ccfb_api_url_template = '{protocol}://{hostname}{port}'
        settings['src.api.url'] = ccfb_api_url_template.format(protocol=protocol, hostname=hostname, port=port)

    config = Configurator(settings=settings)
    config.include('src.api')
    config.set_default_permission(NO_PERMISSION_REQUIRED)

    wsgi_app = config.make_wsgi_app()
    return wsgi_app


if __name__ == '__main__':
    main()

