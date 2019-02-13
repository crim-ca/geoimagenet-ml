from geoimagenet_ml.api import exceptions as ex, schemas as s
from geoimagenet_ml.api.routes.datasets.utils import create_dataset, get_dataset
from geoimagenet_ml.store.factories import database_factory
from pyramid.response import FileResponse
from pyramid.httpexceptions import HTTPOk, HTTPCreated, HTTPForbidden, HTTPInternalServerError
from zipfile import ZipFile
from tempfile import gettempprefix
import os


@s.DatasetsAPI.get(tags=[s.DatasetsTag], response_schemas=s.Datasets_GET_responses)
def get_datasets_view(request):
    """Get registered datasets."""
    db = database_factory(request.registry)
    datasets_list = ex.evaluate_call(lambda: db.datasets_store.list_datasets(),
                                     fallback=lambda: db.rollback(), httpError=HTTPForbidden, request=request,
                                     msgOnFail=s.Datasets_GET_ForbiddenResponseSchema.description)
    datasets_json = ex.evaluate_call(lambda: [d.summary() for d in datasets_list],
                                     fallback=lambda: db.rollback(), httpError=HTTPInternalServerError,
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


@s.DatasetDownloadAPI.get(tags=[s.DatasetsTag],
                          schema=s.DatasetDownloadEndpoint(), response_schemas=s.DatasetDownload_GET_responses)
def get_dataset_view(request):
    """Download registered dataset file."""
    dataset = get_dataset(request)
    dataset_zip = os.path.join(gettempprefix(), 'dataset-{}-{}.zip'.format(dataset.name, dataset.uuid))
    if not os.path.isfile(dataset_zip):
        with ZipFile(dataset_zip, 'w') as f_zip:
            f_zip.write()
            dataset.parameters['']

    response = FileResponse(model.file, content_type="application/octet-stream")
    response.content_disposition = "attachment; filename={}{}".format(model.uuid, model.format)
    return response
