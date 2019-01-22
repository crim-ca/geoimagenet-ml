#!/usr/bin/env python
# coding: utf-8
from src.api import exceptions as ex, schemas as s
from src.store.factories import database_factory
from src import ml, __meta__
from pyramid.httpexceptions import (
    HTTPOk,
    HTTPNotFound,
    HTTPInternalServerError,
)
from pyramid.view import notfound_view_config, exception_view_config
from pyramid.renderers import render_to_response
from cornice_swagger.swagger import CorniceSwagger
from cornice.service import get_services


@notfound_view_config()
def not_found(request):
    return ex.raise_http(nothrow=True, httpError=HTTPNotFound, contentType='application/json',
                         detail=s.NotFoundResponseSchema.description, request=request)


@exception_view_config()
def internal_server_error(request):
    return ex.raise_http(nothrow=True, httpError=HTTPInternalServerError, contentType='application/json',
                         detail=s.InternalServerErrorResponseSchema.description, request=request)


@s.BaseAPI.get(tags=[s.APITag], response_schemas=s.Base_GET_responses)
def get_api_base_view(request):
    """
    GeoImageNet ML REST API information. Base path of the API.
    """
    content = {'title': __meta__.__title__,
               'description': __meta__.__description__,
               'version': __meta__.__version__,
               'url': __meta__.__url__,
               'author': __meta__.__author__,
               'email': __meta__.__email__,
               'docs': 'CCFB REST API documentation available under `{}`.'.format(s.SwaggerAPI.path)}
    return ex.valid_http(httpSuccess=HTTPOk, content=content, detail=s.Base_GET_OkResponseSchema.description,
                         contentType='application/json', request=request)


@s.VersionsAPI.get(tags=[s.APITag], response_schemas=s.Versions_GET_responses)
def get_version_view(request):
    """API version information."""
    db_info = database_factory(request.registry).get_information()
    content = {u'versions': [
        {u'name': u'api', u'version': __meta__.__version__},
        {u'name': u'db', u'version': db_info['version'], u'type': db_info['type']},
        {u'name': u'ml', u'version': ml.__version__, u'type': ml.__type__}
    ]}
    return ex.valid_http(httpSuccess=HTTPOk, content=content,
                         detail=s.Versions_GET_OkResponseSchema.description,
                         contentType='application/json', request=request)


@s.SwaggerJSON.get(tags=[s.APITag], renderer='json', response_schemas=s.SwaggerJSON_GET_responses)
def api_swagger_json_view(request):
    """REST API schema generation in JSON format."""
    swagger = CorniceSwagger(get_services())
    # function docstrings are used to create the route's summary in Swagger-UI
    swagger.summary_docstrings = True
    swagger.default_security = s.get_security
    swagger.swagger = s.SecurityDefinitionAPI
    base_path = request.registry.settings.get('src.api.url') or s.BaseAPI.path
    return swagger.generate(title=s.TitleAPI, version=__meta__.__version__, base_path=base_path)


@s.SwaggerAPI.get(tags=[s.APITag], response_schemas=s.SwaggerAPI_GET_responses)
def api_swagger_ui_view(request):
    """REST API swagger-ui schema documentation (this page)."""
    json_path = s.SwaggerJSON.path.lstrip('/')   # if path starts by '/', swagger-ui doesn't find it on remote
    data_mako = {'api_title': s.TitleAPI, 'api_swagger_json_path': json_path}
    return render_to_response('templates/swagger_ui.mako', data_mako, request=request)
