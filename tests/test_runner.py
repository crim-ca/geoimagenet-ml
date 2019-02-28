#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import sys
import os
# add parent directory of 'tests' to path to allow import although not a module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def filter_test_files(root, filename):
    return os.path.isfile(os.path.join(root, filename)) and filename.startswith('test') and filename.endswith('.py')


test_root_path = os.path.abspath(os.path.dirname(__file__))
test_root_name = os.path.split(test_root_path)[1]
test_files = os.listdir(test_root_path)
test_modules = [os.path.splitext(f)[0] for f in filter(lambda i: filter_test_files(test_root_path, i), test_files)]


def test_suite():
    suite = unittest.TestSuite()
    for t in test_modules:
        try:
            # If the module defines a suite() function, call it to get the suite.
            mod = __import__(t, globals(), locals(), ['suite'])
            suite_fn = getattr(mod, 'suite')
            suite.addTest(suite_fn())
        except (ImportError, AttributeError):
            # else, just load all the test cases from the module.
            suite.addTest(unittest.defaultTestLoader.loadTestsFromName(t))
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(test_suite())
