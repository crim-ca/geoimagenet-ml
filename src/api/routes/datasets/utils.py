#!/usr/bin/env python
# coding: utf-8
from geoimagenet_ml.store.datatypes import Dataset
from geoimagenet_ml.api.rest_api import exceptions as ex, requests as r, schemas as s
from geoimagenet_ml.store.factories import database_factory
from geoimagenet_ml.store import exceptions as exc
from geoimagenet_ml.api.definitions.pyramid_definitions import *
import six


def create_dataset(request):
    dataset_name = r.get_multiformat_post(request, 'dataset_name')
    dataset_path = r.get_multiformat_post(request, 'dataset_path')
    dataset_type = r.get_multiformat_post(request, 'dataset_type')
    dataset_params = r.get_multiformat_post(request, 'dataset_params')
    ex.verify_param(dataset_name, notNone=True, notEmpty=True, ofType=six.string_types,
                    httpError=HTTPBadRequest, paramName='dataset_name',
                    msgOnFail=s.Datasets_POST_BadRequestResponseSchema.description, request=request)
    ex.verify_param(dataset_path, notNone=True, notEmpty=True, ofType=six.string_types,
                    httpError=HTTPBadRequest, paramName='dataset_path',
                    msgOnFail=s.Datasets_POST_BadRequestResponseSchema.description, request=request)
    ex.verify_param(dataset_type, notNone=True, notEmpty=True, ofType=six.string_types,
                    httpError=HTTPBadRequest, paramName='dataset_type',
                    msgOnFail=s.Datasets_POST_BadRequestResponseSchema.description, request=request)
    ex.verify_param(dataset_params, notNone=True, notEmpty=True, ofType=dict,
                    httpError=HTTPBadRequest, paramName='dataset_params',
                    msgOnFail=s.Datasets_POST_BadRequestResponseSchema.description, request=request)
    new_dataset = None
    try:
        tmp_dataset = Dataset(name=dataset_name, path=dataset_path, type=dataset_type, params=dataset_params)
        new_dataset = database_factory(request.registry).datasets_store.save_dataset(tmp_dataset)
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
    dataset_uuid = request.matchdict.get('dataset_uuid')
    ex.verify_param(dataset_uuid, notNone=True, notEmpty=True, httpError=HTTPBadRequest, paramName='dataset_uuid',
                    msgOnFail=s.Dataset_GET_BadRequestResponseSchema.description, request=request)
    dataset = None
    try:
        dataset = database_factory(request.registry).datasets_store.fetch_by_uuid(dataset_uuid)
        if not dataset:
            raise exc.DatasetNotFoundError
    except exc.DatasetInstanceError:
        ex.raise_http(httpError=HTTPForbidden, request=request,
                      detail=s.Dataset_GET_ForbiddenResponseSchema.description)
    except exc.DatasetNotFoundError:
        ex.raise_http(httpError=HTTPNotFound, request=request,
                      detail=s.Dataset_GET_NotFoundResponseSchema.description)
    return dataset
