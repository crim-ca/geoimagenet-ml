#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src import __meta__    # noqa

CUR_DIR = os.path.dirname(__file__)

with open(os.path.join(CUR_DIR, 'README.rst')) as readme_file:
    README = readme_file.read()

with open(os.path.join(CUR_DIR, 'HISTORY.rst')) as history_file:
    HISTORY = history_file.read().replace('.. :changelog:', '')

REQUIREMENTS = set([])  # use set to have unique packages by name
with open(os.path.join(CUR_DIR, 'requirements.txt'), 'r') as requirements_file:
    [REQUIREMENTS.add(line.strip()) for line in requirements_file]
REQUIREMENTS = list(REQUIREMENTS)

TEST_REQUIREMENTS = set([])  # use set to have unique packages by name
with open(os.path.join(CUR_DIR, 'requirements-dev.txt'), 'r') as requirements_file:
    [TEST_REQUIREMENTS.add(line.strip()) for line in requirements_file]
TEST_REQUIREMENTS = list(TEST_REQUIREMENTS)

setup(
    # -- meta information --------------------------------------------------
    name=__meta__.__package__,
    version=__meta__.__version__,
    description=__meta__.__title__,
    long_description=README + '\n\n' + HISTORY,
    author=__meta__.__author__,
    author_email=__meta__.__email__,
    url=__meta__.__url__,
    platforms=__meta__.__platforms__,
    license=__meta__.__license__,
    keywords=' '.join(__meta__.__keywords__),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: ISC License (ISCL)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    # -- Package structure -------------------------------------------------
    packages=find_packages(),
    package_dir={__meta__.__package__: 'src'},
    include_package_data=True,
    install_requires=REQUIREMENTS,
    zip_safe=False,

    # -- self - tests --------------------------------------------------------
    test_suite='tests',
    tests_require=TEST_REQUIREMENTS,

    # -- script entry points -----------------------------------------------
    entry_points="""\
          [paste.app_factory]
          main = geoimagenet_ml.api.main:main
          [console_scripts]
          """,
)
