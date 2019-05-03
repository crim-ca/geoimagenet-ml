#!/usr/bin/env python
# coding: utf-8
from geoimagenet_ml.api import exceptions as ex, requests as r, schemas as s
from geoimagenet_ml.constants import SORT, ORDER
from geoimagenet_ml.status import STATUS
from geoimagenet_ml.store.datatypes import Dataset
from geoimagenet_ml.store.factories import database_factory
from geoimagenet_ml.store import exceptions as exc
from geoimagenet_ml.utils import get_user_id
from pyramid.httpexceptions import HTTPBadRequest, HTTPForbidden, HTTPNotFound, HTTPConflict, HTTPInternalServerError
from typing import TYPE_CHECKING
import six
if TYPE_CHECKING:
    from pyramid.request import Request  # noqa: F401


def create_dataset(request):
    # type: (Request) -> Dataset
    """
    Creates a new dataset instance and registers it to database.

    :returns: created dataset instance if all requirements are met.
    :raises HTTPException: corresponding status to error encountered.
    """
    dataset_name = r.get_multiformat_post(request, "dataset_name")
    dataset_path = r.get_multiformat_post(request, "dataset_path")
    dataset_type = r.get_multiformat_post(request, "dataset_type")
    dataset_params = r.get_multiformat_post(request, "dataset_params")
    ex.verify_param(dataset_name, notNone=True, notEmpty=True, ofType=six.string_types,
                    httpError=HTTPBadRequest, paramName="dataset_name",
                    msgOnFail=s.Datasets_POST_BadRequestResponseSchema.description, request=request)
    ex.verify_param(dataset_path, notNone=True, notEmpty=True, ofType=six.string_types,
                    httpError=HTTPBadRequest, paramName="dataset_path",
                    msgOnFail=s.Datasets_POST_BadRequestResponseSchema.description, request=request)
    ex.verify_param(dataset_type, notNone=True, notEmpty=True, ofType=six.string_types,
                    httpError=HTTPBadRequest, paramName="dataset_type",
                    msgOnFail=s.Datasets_POST_BadRequestResponseSchema.description, request=request)
    ex.verify_param(dataset_params, notNone=True, notEmpty=True, ofType=dict,
                    httpError=HTTPBadRequest, paramName="dataset_params",
                    msgOnFail=s.Datasets_POST_BadRequestResponseSchema.description, request=request)
    new_dataset = None
    try:
        tmp_dataset = Dataset(name=dataset_name, path=dataset_path, type=dataset_type,
                              params=dataset_params, user=get_user_id(request))
        new_dataset = database_factory(request).datasets_store.save_dataset(tmp_dataset, request=request)
        if not new_dataset:
            raise exc.DatasetRegistrationError
    except (exc.DatasetRegistrationError, exc.DatasetInstanceError):
        ex.raise_http(httpError=HTTPForbidden, request=request,
                      detail=s.Datasets_POST_ForbiddenResponseSchema.description)
    except exc.DatasetConflictError:
        ex.raise_http(httpError=HTTPConflict, request=request,
                      detail=s.Datasets_POST_ConflictResponseSchema.description)
    except exc.DatasetNotFoundError:
        ex.raise_http(httpError=HTTPNotFound, request=request,
                      detail=s.Datasets_POST_NotFoundResponseSchema.description)
    return new_dataset


def get_dataset(request):
    # type: (Request) -> Dataset
    """
    Searches for the dataset specified by path parameter.

    :returns: valid dataset instance if found.
    :raises HTTPException: corresponding status to error encountered.
    """
    dataset_uuid = request.matchdict.get(s.ParamDatasetUUID)
    ex.verify_param(dataset_uuid, notNone=True, notEmpty=True, httpError=HTTPBadRequest, paramName=s.ParamDatasetUUID,
                    msgOnFail=s.Dataset_GET_BadRequestResponseSchema.description, request=request)
    dataset = None
    try:
        datasets_store = database_factory(request).datasets_store
        if dataset_uuid == "latest":
            datasets, count = datasets_store.find_datasets(
                name=r.get_multiformat_any(request, "dataset_name"),
                type=r.get_multiformat_any(request, "dataset_type"),
                sort=SORT.FINISHED,
                order=ORDER.DESCENDING,
                status=STATUS.FINISHED,
                limit=1,
            )
            if len(datasets):
                dataset = datasets[0]
        else:
            dataset = datasets_store.fetch_by_uuid(dataset_uuid, request=request)
        if not dataset:
            raise exc.DatasetNotFoundError
    except exc.DatasetInstanceError:
        ex.raise_http(httpError=HTTPForbidden, request=request,
                      detail=s.Dataset_GET_ForbiddenResponseSchema.description)
    except exc.DatasetNotFoundError:
        ex.raise_http(httpError=HTTPNotFound, request=request,
                      detail=s.Dataset_GET_NotFoundResponseSchema.description)
    return dataset


def delete_dataset(request):
    # type: (Request) -> None
    """
    Deletes the dataset specified by path parameter.

    :returns: nothing if operation was successful.
    :raises HTTPException: corresponding status to error encountered.
    """
    dataset_uuid = request.matchdict.get(s.ParamDatasetUUID)
    ex.verify_param(dataset_uuid, notNone=True, notEmpty=True, httpError=HTTPBadRequest, paramName=s.ParamDatasetUUID,
                    msgOnFail=s.Dataset_DELETE_BadRequestResponseSchema.description, request=request)
    try:
        is_deleted = database_factory(request).datasets_store.delete_dataset(dataset_uuid, request=request)
        if not is_deleted:
            raise exc.DatasetNotFoundError
    except exc.DatasetInstanceError:
        ex.raise_http(httpError=HTTPInternalServerError, request=request,
                      detail=s.InternalServerErrorResponseSchema.description)
    except exc.DatasetNotFoundError:
        ex.raise_http(httpError=HTTPNotFound, request=request,
                      detail=s.Dataset_DELETE_NotFoundResponseSchema.description)
