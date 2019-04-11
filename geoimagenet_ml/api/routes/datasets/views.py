from geoimagenet_ml.api import exceptions as ex, schemas as s
from geoimagenet_ml.api.routes.datasets.utils import create_dataset, get_dataset, delete_dataset
from geoimagenet_ml.constants import OPERATION
from geoimagenet_ml.store.datatypes import Dataset
from geoimagenet_ml.store.factories import database_factory
from pyramid.response import FileResponse
from pyramid.httpexceptions import HTTPOk, HTTPCreated, HTTPForbidden, HTTPInternalServerError


@s.DatasetsAPI.get(tags=[s.DatasetsTag], response_schemas=s.Datasets_GET_responses)
def get_datasets_view(request):
    """Get registered datasets."""
    db = database_factory(request)
    datasets_list = ex.evaluate_call(lambda: db.datasets_store.list_datasets(),
                                     fallback=lambda: db.rollback(), httpError=HTTPForbidden, request=request,
                                     msgOnFail=s.Datasets_GET_ForbiddenResponseSchema.description)
    datasets_json = ex.evaluate_call(lambda: [d.summary() for d in datasets_list],
                                     fallback=lambda: db.rollback(), httpError=HTTPInternalServerError,
                                     request=request, msgOnFail=s.InternalServerErrorResponseSchema.description)
    db.actions_store.save_action(Dataset, OPERATION.LIST, request=request)
    return ex.valid_http(httpSuccess=HTTPOk, content={u"datasets": datasets_json},
                         detail=s.Datasets_GET_OkResponseSchema.description, request=request)


@s.DatasetsAPI.post(tags=[s.DatasetsTag],
                    schema=s.Datasets_POST_RequestSchema(), response_schemas=s.Datasets_POST_responses)
def post_datasets_view(request):
    """Register a new dataset."""
    dataset = create_dataset(request)
    database_factory(request).actions_store.save_action(dataset, OPERATION.UPLOAD, request=request)
    return ex.valid_http(httpSuccess=HTTPCreated, content={u"dataset": dataset.json()},
                         detail=s.Datasets_POST_CreatedResponseSchema.description, request=request)


def get_dataset_handler(request):
    dataset = get_dataset(request)
    database_factory(request).actions_store.save_action(dataset, OPERATION.INFO, request=request)
    return ex.valid_http(httpSuccess=HTTPOk, content={u"datasets": dataset.json()},
                         detail=s.Dataset_GET_OkResponseSchema.description, request=request)


@s.DatasetAPI.get(tags=[s.DatasetsTag],
                  schema=s.DatasetEndpoint(), response_schemas=s.Dataset_GET_responses)
def get_dataset_view(request):
    """Get registered dataset information."""
    return get_dataset_handler(request)


@s.DatasetLatestAPI.get(tags=[s.DatasetsTag],
                        schema=s.DatasetLatestEndpoint(), response_schemas=s.Dataset_GET_responses)
def get_dataset_latest_view(request):
    """Get latest dataset information matching criterion."""
    return get_dataset_handler(request)


@s.DatasetAPI.delete(tags=[s.DatasetsTag],
                     schema=s.DatasetEndpoint(), response_schemas=s.Dataset_DELETE_responses)
def delete_dataset_view(request):
    """Get registered dataset information."""
    delete_dataset(request)
    database_factory(request).actions_store.save_action(Dataset, OPERATION.DELETE, request=request)
    return ex.valid_http(httpSuccess=HTTPOk, content={},
                         detail=s.Dataset_DELETE_OkResponseSchema.description, request=request)


@s.DatasetDownloadAPI.get(tags=[s.DatasetsTag],
                          schema=s.DatasetDownloadEndpoint(), response_schemas=s.DatasetDownload_GET_responses)
def download_dataset_view(request):
    """Download registered dataset file."""
    dataset = get_dataset(request)
    dataset_zip_path = dataset.zip()
    dataset_zip_name = "dataset-{}-{}.zip".format(dataset.name, dataset.uuid)
    response = FileResponse(dataset_zip_path, content_type="application/octet-stream")
    response.content_disposition = "attachment; filename={}".format(dataset_zip_name)
    database_factory(request).actions_store.save_action(dataset, OPERATION.DOWNLOAD, request=request)
    return response
