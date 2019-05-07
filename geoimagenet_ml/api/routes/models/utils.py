#!/usr/bin/env python
# coding: utf-8
from geoimagenet_ml.api import exceptions as ex, requests as r, schemas as s
from geoimagenet_ml.store.datatypes import Model
from geoimagenet_ml.store.factories import database_factory
from geoimagenet_ml.store import exceptions as exc
from geoimagenet_ml.utils import get_user_id
from pyramid.httpexceptions import HTTPBadRequest, HTTPForbidden, HTTPNotFound, HTTPConflict, HTTPUnprocessableEntity
from pyramid.request import Request  # noqa: F401


def create_model(request):
    # type: (Request) -> Model
    """
    Creates a new model instance and registers it to database.

    :returns: created model instance if all requirements and data validation are met.
    :raises HTTPException: corresponding status to error encountered.
    """
    model_name = r.get_multiformat_post(request, "model_name")
    model_path = r.get_multiformat_post(request, "model_path")
    ex.verify_param(model_name, notNone=True, notEmpty=True, httpError=HTTPBadRequest, paramName="model_name",
                    msgOnFail=s.Models_POST_BadRequestResponseSchema.description, request=request)
    ex.verify_param(model_path, notNone=True, notEmpty=True, httpError=HTTPBadRequest, paramName="model_path",
                    msgOnFail=s.Models_POST_BadRequestResponseSchema.description, request=request)

    try:
        tmp_model = Model(name=model_name, path=model_path, user=get_user_id(request))
        new_model = database_factory(request).models_store.save_model(tmp_model, request=request)
        return new_model
    except exc.ModelRegistrationError as exception:
        ex.raise_http(httpError=HTTPForbidden, request=request,
                      detail=s.Models_POST_ForbiddenResponseSchema.description,
                      content={"exception": repr(exception)})
    except (exc.ModelLoadingError, exc.ModelInstanceError) as exception:
        ex.raise_http(httpError=HTTPUnprocessableEntity, request=request,
                      detail=s.Models_POST_UnprocessableEntityResponseSchema.description,
                      content={"exception": repr(exception)})
    except exc.ModelNotFoundError:
        ex.raise_http(httpError=HTTPNotFound, request=request,
                      detail=s.Models_POST_NotFoundResponseSchema.description)
    except exc.ModelConflictError:
        ex.raise_http(httpError=HTTPConflict, request=request,
                      detail=s.Models_POST_ConflictResponseSchema.description)


def get_model(request):
    # type: (Request) -> Model
    """
    Searches for the model specified by path parameter.

    :returns: valid model instance if found.
    :raises HTTPException: corresponding status to error encountered.
    """
    model_uuid = request.matchdict.get(s.ParamModelUUID)
    ex.verify_param(model_uuid, notNone=True, notEmpty=True, httpError=HTTPBadRequest, paramName=s.ParamModelUUID,
                    msgOnFail=s.Model_GET_BadRequestResponseSchema.description, request=request)
    try:
        return database_factory(request).models_store.fetch_by_uuid(model_uuid)
    except exc.ModelInstanceError:
        ex.raise_http(httpError=HTTPForbidden, request=request,
                      detail=s.Model_GET_ForbiddenResponseSchema.description)
    except exc.ModelNotFoundError:
        ex.raise_http(httpError=HTTPNotFound, request=request,
                      detail=s.Model_GET_NotFoundResponseSchema.description)


def update_model(request):
    # type: (Request) -> Model
    """
    Updates a matched model UUID in path with specified body parameter(s).

    :returns: valid model instance if found and updated.
    :raises HTTPException: corresponding status to error encountered.
    """
    model = get_model(request)
    field_names = ["name", "visibility"]
    model_fields = {field: r.get_multiformat_post(request, field) for field in field_names}
    model_fields = {f: v for f, v in model_fields.items() if v}
    ex.verify_param(len(model_fields), notEqual=True, paramCompare=0, httpError=HTTPBadRequest,
                    msgOnFail=s.Model_PUT_BadRequestResponseSchema.description, request=request)

    db = database_factory(request)
    return ex.evaluate_call(lambda: db.models_store.update_model(model, request=request, **model_fields),
                            fallback=lambda: db.rollback(), request=request, httpError=HTTPForbidden,
                            msgOnFail=s.Model_PUT_ForbiddenResponseSchema.description)
