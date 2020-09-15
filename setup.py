#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import os
import sys
CUR_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(CUR_DIR))
from geoimagenet_ml import __meta__  # noqa

readme_file = os.path.join(CUR_DIR, "README.rst")
change_file = os.path.join(CUR_DIR, "HISTORY.rst")
description = None
if os.path.isfile(readme_file) and os.path.isfile(change_file):
    with open(readme_file) as readme_fd:
        README = readme_fd.read()
    with open(change_file) as change_fd:
        CHANGE = change_fd.read().replace(".. :changelog:", "")
    description = README + "\n\n" + CHANGE

REQUIREMENTS = set([])  # use set to have unique packages by name
with open(os.path.join(CUR_DIR, "requirements.txt"), 'r') as requirements_file:
    [REQUIREMENTS.add(line.strip()) for line in requirements_file]
REQUIREMENTS = list(REQUIREMENTS)

TEST_REQUIREMENTS = set([])  # use set to have unique packages by name
with open(os.path.join(CUR_DIR, "requirements-dev.txt"), 'r') as requirements_file:
    [TEST_REQUIREMENTS.add(line.strip()) for line in requirements_file]
TEST_REQUIREMENTS = list(TEST_REQUIREMENTS)

setup(
    # -- meta information --------------------------------------------------
    name=__meta__.__package__,
    version=__meta__.__version__,
    description=__meta__.__title__,
    long_description=description,
    author=__meta__.__author__,
    author_email=__meta__.__email__,
    url=__meta__.__url__,
    platforms=__meta__.__platforms__,
    license=__meta__.__license__,
    keywords=" ".join(__meta__.__keywords__),
    python_requires=">=3.6, <4",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: ISC License (ISCL)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Hydrology",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],

    # -- Package structure -------------------------------------------------
    packages=[__meta__.__package__, "scripts"],    # find_packages(),
    package_dir={__meta__.__package__: __meta__.__package__},
    include_package_data=True,
    install_requires=REQUIREMENTS,
    zip_safe=False,

    # --- extra requirements --------------------------------------------------
    extras_require={
        # default is to use mongodb, install PostgreSQL dependencies as required
        "postgres": [
            "psycopg2-binary>=2.7.1"
        ]
    },

    # -- tests --------------------------------------------------------
    test_suite="tests",
    tests_require=TEST_REQUIREMENTS,

    # -- script entry points -----------------------------------------------
    entry_points={
        "paste.app_factory": [
            "main = {}.api.main:main".format(__meta__.__package__)
        ],
        "console_scripts": [
            "gin_ml_run_model_tester = scripts.run_model_tester:main",
            "gin_ml_update_model_classes = scripts.update_model_classes:main"
        ]
    }
)
