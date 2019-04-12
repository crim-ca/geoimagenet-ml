#!/usr/bin/env python
# coding: utf-8
from geoimagenet_ml.api import exceptions as ex, requests as r, schemas as s
from geoimagenet_ml.store.datatypes import Model
from geoimagenet_ml.store.factories import database_factory
from geoimagenet_ml.store import exceptions as exc
from geoimagenet_ml.utils import get_user_id
from geoimagenet_ml.ml.impl import load_model
from pyramid.httpexceptions import HTTPBadRequest, HTTPForbidden, HTTPNotFound, HTTPConflict, HTTPUnprocessableEntity
from pyramid.request import Request  # noqa: F401


def create_model(request):
    # type: (Request) -> Model
    model_name = r.get_multiformat_post(request, "model_name")
    model_path = r.get_multiformat_post(request, "model_path")
    ex.verify_param(model_name, notNone=True, notEmpty=True, httpError=HTTPBadRequest, paramName="model_name",
                    msgOnFail=s.Models_POST_BadRequestResponseSchema.description, request=request)
    ex.verify_param(model_path, notNone=True, notEmpty=True, httpError=HTTPBadRequest, paramName="model_path",
                    msgOnFail=s.Models_POST_BadRequestResponseSchema.description, request=request)

    is_model_loaded, _, buffer, load_exception = load_model(model_path)
    if not is_model_loaded:
        ex.raise_http(httpError=HTTPUnprocessableEntity, request=request,
                      detail=s.Models_POST_UnprocessableEntityResponseSchema.description,
                      content={"exception": repr(load_exception)})

    new_model = None
    try:
        tmp_model = Model(name=model_name, path=model_path, user=get_user_id(request))
        new_model = database_factory(request).models_store.save_model(tmp_model, data=buffer, request=request)
        if not new_model:
            raise exc.ModelRegistrationError
    except (exc.ModelRegistrationError, exc.ModelInstanceError):
        ex.raise_http(httpError=HTTPForbidden, request=request,
                      detail=s.Models_POST_ForbiddenResponseSchema.description)
    except exc.ModelNotFoundError:
        ex.raise_http(httpError=HTTPNotFound, request=request,
                      detail=s.Models_POST_NotFoundResponseSchema.description)
    except exc.ModelConflictError:
        ex.raise_http(httpError=HTTPConflict, request=request,
                      detail=s.Models_POST_ConflictResponseSchema.description)
    return new_model


def get_model(request):
    # type: (Request) -> Model
    model_uuid = request.matchdict.get("model_uuid")
    ex.verify_param(model_uuid, notNone=True, notEmpty=True, httpError=HTTPBadRequest, paramName="model_uuid",
                    msgOnFail=s.Model_GET_BadRequestResponseSchema.description, request=request)
    model = None
    try:
        model = database_factory(request).models_store.fetch_by_uuid(model_uuid)
        if not model:
            raise exc.ModelNotFoundError
    except exc.ModelInstanceError:
        ex.raise_http(httpError=HTTPForbidden, request=request,
                      detail=s.Model_GET_ForbiddenResponseSchema.description)
    except exc.ModelNotFoundError:
        ex.raise_http(httpError=HTTPNotFound, request=request,
                      detail=s.Model_GET_NotFoundResponseSchema.description)
    return model
