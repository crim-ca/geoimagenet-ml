from geoimagenet_ml.api.routes.models.utils import create_model, get_model, update_model
from geoimagenet_ml.api import exceptions as ex, schemas as s
from geoimagenet_ml.constants import OPERATION
from geoimagenet_ml.store.datatypes import Model
from geoimagenet_ml.store.factories import database_factory
from pyramid.response import FileResponse
from pyramid.httpexceptions import HTTPOk, HTTPCreated, HTTPForbidden, HTTPInternalServerError


@s.ModelsAPI.get(tags=[s.TagModels], response_schemas=s.Models_GET_responses)
def get_models_view(request):
    """Get registered models."""
    db = database_factory(request)
    models_list = ex.evaluate_call(lambda: db.models_store.list_models(),
                                   fallback=lambda: db.rollback(), httpError=HTTPForbidden, request=request,
                                   msgOnFail=s.Models_GET_ForbiddenResponseSchema.description)
    models_json = ex.evaluate_call(lambda: [m.summary() for m in models_list],
                                   fallback=lambda: db.rollback(), httpError=HTTPInternalServerError,
                                   request=request, msgOnFail=s.InternalServerErrorResponseSchema.description)
    db.actions_store.save_action(Model, OPERATION.LIST, request=request)
    return ex.valid_http(httpSuccess=HTTPOk, content={u"models": models_json},
                         detail=s.Models_GET_OkResponseSchema.description, request=request)


@s.ModelsAPI.post(tags=[s.TagModels], schema=s.Models_POST_RequestSchema(), response_schemas=s.Models_POST_responses)
def post_models_view(request):
    """Register a new model."""
    model = create_model(request)
    database_factory(request).actions_store.save_action(model, OPERATION.UPLOAD, request=request)
    return ex.valid_http(httpSuccess=HTTPCreated, content={u"model": model.json()},
                         detail=s.Models_POST_CreatedResponseSchema.description, request=request)


@s.ModelAPI.get(tags=[s.TagModels],
                schema=s.Model_GET_Endpoint(), response_schemas=s.Model_GET_responses)
def get_model_view(request):
    """Get registered model information."""
    model = get_model(request)
    db = database_factory(request)
    db.actions_store.save_action(model, OPERATION.INFO, request=request)
    _, download_count = db.actions_store.find_actions(item_type=Model, item=model.uuid, operation=OPERATION.DOWNLOAD)
    model_content = {
        "model": model.json(),
        "owner": model.user,
        "downloads": download_count,
    }
    return ex.valid_http(httpSuccess=HTTPOk, content=model_content,
                         detail=s.Model_GET_OkResponseSchema.description, request=request)


@s.ModelAPI.put(tags=[s.TagModels],
                schema=s.Model_PUT_Endpoint(), response_schemas=s.Model_PUT_responses)
def put_model_view(request):
    """Update registered model information."""
    model = update_model(request)
    db = database_factory(request)
    db.actions_store.save_action(model, OPERATION.UPDATE, request=request)
    return ex.valid_http(httpSuccess=HTTPOk, content={"model": model.summary()},
                         detail=s.Model_PUT_OkResponseSchema.description, request=request)


@s.ModelDownloadAPI.get(tags=[s.TagModels],
                        schema=s.ModelDownloadEndpoint(), response_schemas=s.ModelDownload_GET_responses)
def download_model_view(request):
    """Download registered model file."""
    model = get_model(request)
    response = FileResponse(model.file, content_type="application/octet-stream")
    response.content_disposition = "attachment; filename=model-{}{}".format(model.uuid, model.format)
    database_factory(request).actions_store.save_action(model, OPERATION.DOWNLOAD, request=request)
    return response
