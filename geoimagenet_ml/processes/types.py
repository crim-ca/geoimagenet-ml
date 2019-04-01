#!/usr/bin/env python
# coding: utf-8

from geoimagenet_ml.processes.wps_test import HelloWPS
from geoimagenet_ml.processes.runners import ProcessRunnerModelTester, ProcessRunnerBatchCreator

PROCESS_WPS = 'wps'

process_mapping = {
    HelloWPS.identifier: HelloWPS,
    ProcessRunnerModelTester.identifier: ProcessRunnerModelTester,
    ProcessRunnerBatchCreator.identifier: ProcessRunnerBatchCreator,
}

# noinspection PyTypeChecker
process_categories = frozenset(
    list(process_mapping.keys()) +
    list([PROCESS_WPS])
)
