from geoimagenet_ml.api.rest_api import exceptions as ex, schemas as s
from geoimagenet_ml.api.rest_api.datasets.utils import create_dataset, get_dataset
from geoimagenet_ml.store.factories import database_factory
from geoimagenet_ml.api.definitions.pyramid_definitions import *


@s.DatasetsAPI.get(tags=[s.DatasetsTag], response_schemas=s.Datasets_GET_responses)
def get_datasets_view(request):
    """Get registered datasets."""
    datasets_list = ex.evaluate_call(lambda: database_factory(request.registry).datasets_store.list_datasets(),
                                     fallback=lambda: request.db.rollback(), httpError=HTTPForbidden, request=request,
                                     msgOnFail=s.Datasets_GET_ForbiddenResponseSchema.description)
    datasets_json = ex.evaluate_call(lambda: [d.summary() for d in datasets_list],
                                     fallback=lambda: request.db.rollback(), httpError=HTTPInternalServerError,
                                     request=request, msgOnFail=s.InternalServerErrorResponseSchema.description)
    return ex.valid_http(httpSuccess=HTTPOk, content={u'datasets': datasets_json},
                         detail=s.Datasets_GET_OkResponseSchema.description, request=request)


@s.DatasetsAPI.post(tags=[s.DatasetsTag],
                    schema=s.Datasets_POST_RequestSchema(), response_schemas=s.Datasets_POST_responses)
def post_datasets_view(request):
    """Register a new dataset."""
    dataset = create_dataset(request)
    return ex.valid_http(httpSuccess=HTTPCreated, content={u'dataset': dataset.json()},
                         detail=s.Datasets_POST_CreatedResponseSchema.description, request=request)


@s.DatasetAPI.get(tags=[s.DatasetsTag],
                  schema=s.DatasetEndpoint(), response_schemas=s.Dataset_GET_responses)
def get_dataset_view(request):
    """Get registered dataset information."""
    dataset = get_dataset(request)
    return ex.valid_http(httpSuccess=HTTPOk, content={u'datasets': dataset.json()},
                         detail=s.Datasets_GET_OkResponseSchema.description, request=request)
