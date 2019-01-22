#!/usr/bin/env python
# coding: utf-8
from geoimagenet_ml.api import exceptions as ex
from pyramid.httpexceptions import HTTPUnprocessableEntity, HTTPInternalServerError


def get_request_method_content(request):
    # 'request' object stores GET content into 'GET' property, while other methods are in 'POST' property
    method_property = 'GET' if request.method == 'GET' else 'POST'
    return getattr(request, method_property)


def get_multiformat_any(request, key):
    msg = "Key `{key}` could not be extracted from {method} of type `{type}`" \
          .format(key=repr(key), method=request.method, type=request.content_type)
    if request.content_type == 'application/json':
        return ex.evaluate_call(lambda: request.json.get(key),
                                httpError=HTTPInternalServerError, msgOnFail=msg)
    return ex.evaluate_call(lambda: get_request_method_content(request).get(key),
                            httpError=HTTPInternalServerError, msgOnFail=msg)


def get_multiformat_post(request, key):
    return get_multiformat_any(request, key)


def get_multiformat_put(request, key):
    return get_multiformat_any(request, key)


def get_multiformat_delete(request, key):
    return get_multiformat_any(request, key)


def get_value_multiformat_post_checked(request, key):
    val = get_multiformat_any(request, key)
    ex.verify_param(val, notNone=True, notEmpty=True, httpError=HTTPUnprocessableEntity,
                    content={str(key): str(val)}, msgOnFail="Invalid value specified.")
    return val
