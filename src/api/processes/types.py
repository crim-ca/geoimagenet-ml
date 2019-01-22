#!/usr/bin/env python
# coding: utf-8

from ccfb.api.processes.wps_test import HelloWPS

PROCESS_WPS = 'wps'
PROCESS_ML = 'ml'


process_mapping = {
    HelloWPS.__name__: HelloWPS
}

process_categories = frozenset(
    list(process_mapping.keys()) +
    list([
        PROCESS_WPS,
        PROCESS_ML,
    ])
)
