from ccfb.api.rest_api.models.utils import create_model, get_model
from ccfb.api.rest_api import exceptions as ex, schemas as s
from ccfb.api.store.factories import database_factory
from ccfb.api.definitions import pyramid_definitions as p
from pyramid.response import FileResponse


@s.ModelsAPI.get(tags=[s.ModelsTag], response_schemas=s.Models_GET_responses)
def get_models_view(request):
    """Get registered models."""
    models_list = ex.evaluate_call(lambda: database_factory(request.registry).models_store.list_models(),
                                   fallback=lambda: request.db.rollback(), httpError=p.HTTPForbidden, request=request,
                                   msgOnFail=s.Models_GET_ForbiddenResponseSchema.description)
    models_json = ex.evaluate_call(lambda: [m.summary() for m in models_list],
                                   fallback=lambda: request.db.rollback(), httpError=p.HTTPInternalServerError,
                                   request=request, msgOnFail=s.InternalServerErrorResponseSchema.description)
    return ex.valid_http(httpSuccess=p.HTTPOk, content={u'models': models_json},
                         detail=s.Models_GET_OkResponseSchema.description, request=request)


@s.ModelsAPI.post(tags=[s.ModelsTag], schema=s.Models_POST_RequestSchema(), response_schemas=s.Models_POST_responses)
def post_models_view(request):
    """Register a new model."""
    model = create_model(request)
    return ex.valid_http(httpSuccess=p.HTTPCreated, content={u'model': model.json()},
                         detail=s.Models_POST_CreatedResponseSchema.description, request=request)


@s.ModelAPI.get(tags=[s.ModelsTag],
                schema=s.ModelEndpoint(), response_schemas=s.Model_GET_responses)
def get_model_view(request):
    """Get registered model information."""
    model = get_model(request)
    return ex.valid_http(httpSuccess=p.HTTPOk, content={u'model': model.json()},
                         detail=s.Models_GET_OkResponseSchema.description, request=request)


@s.ModelDownloadAPI.get(tags=[s.ModelsTag],
                        schema=s.ModelDownloadEndpoint(), response_schemas=s.ModelDownload_GET_responses)
def download_model_view(request):
    """Get registered model information."""
    model = get_model(request)
    response = FileResponse(model.file, content_type="application/octet-stream")
    response.content_disposition = "attachment; filename={}{}".format(model.uuid, model.format)
    return response
