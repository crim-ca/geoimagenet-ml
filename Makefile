define BROWSER_PYSCRIPT
import os, webbrowser, sys
try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT
BROWSER := python -c "$$BROWSER_PYSCRIPT"

CUR_DIR := $(abspath $(lastword $(MAKEFILE_LIST))/..)
APP_ROOT := $(CURDIR)
APP_NAME := geoimagenet-ml
SRC_DIR := $(CURDIR)/geoimagenet_ml

# Anaconda
ANACONDA_HOME ?= $(HOME)/anaconda
CONDA_ENV ?= geoimagenet-ml
CONDA_ENVS_DIR ?= $(HOME)/.conda/envs
CONDA_ENV_PATH := $(CONDA_ENVS_DIR)/$(CONDA_ENV)
DOWNLOAD_CACHE := $(APP_ROOT)/downloads
PYTHON_VERSION := 3.6

# choose anaconda installer depending on your OS
ANACONDA_URL = https://repo.continuum.io/miniconda
OS_NAME := $(shell uname -s 2>/dev/null || echo "unknown")
ifeq "$(OS_NAME)" "Linux"
FN := Miniconda3-latest-Linux-x86_64.sh
else ifeq "$(OS_NAME)" "Darwin"
FN := Miniconda3-latest-MacOSX-x86_64.sh
else
FN := unknown
endif


.DEFAULT_GOAL := help

.PHONY: all
all: help

.PHONY: help
help:
	@echo "bump             bump version using version specified as user input"
	@echo "bump-dry         bump version using version specified as user input (dry-run)"
	@echo "bump-tag         bump version using version specified as user input, tags it and commits change in git"
	@echo "clean            remove all build, test, coverage and Python artifacts"
	@echo "clean-build      remove build artifacts"
	@echo "clean-env        remove package environment"
	@echo "clean-ml         remove ML module build artifacts"
	@echo "clean-pyc        remove Python file artifacts"
	@echo "clean-test       remove test and coverage artifacts"
	@echo "coverage         check code coverage quickly with the default Python"
	@echo "dist             distribution of release package"
	@echo "docker-info      print the expected docker image tag"
	@echo "docker-build     build the docker image with the current version"
	@echo "docker-push      push the docker image with the current version"
	@echo "docs             generate Sphinx HTML documentation, including API docs"
	@echo "install          install the package to the active Python's site-packages"
	@echo "install-api      install API related components"
	@echo "install-ml       install ML related components"
	@echo "install-dev      install test related components"
	@echo "pep8             check code style"
	@echo "release          package and upload a release"
	@echo "start            start the installed application"
	@echo "test             run basic unit tests (not online)"
	@echo "test-all         run tests with every marker enabled"
	@echo "test-tox         run tests on every Python version with tox"
	@echo "update           same as 'install' but without conda packages installation"
	@echo "version          retrive the application version"

.PHONY: bump
bump:
	$(shell bash -c 'read -p "Version: " VERSION_PART; \
		source "$(CONDA_HOME)/bin/activate" "$(CONDA_ENV)"; \
		test -f "$(CONDA_ENV_PATH)/bin/bumpversion" || pip install bumpversion; \
		"$(CONDA_ENV_PATH)/bin/bumpversion" --config-file "$(CURDIR)/.bumpversion.cfg" \
			--verbose --allow-dirty --no-tag --new-version $$VERSION_PART patch;')

.PHONY: bump-dry
bump-dry:
	$(shell bash -c 'read -p "Version: " VERSION_PART; \
		source "$(CONDA_HOME)/bin/activate" "$(CONDA_ENV)"; \
		test -f "$(CONDA_ENV_PATH)/bin/bumpversion" || pip install bumpversion; \
		"$(CONDA_ENV_PATH)/bin/bumpversion" --config-file "$(CURDIR)/.bumpversion.cfg" \
			--verbose --allow-dirty --dry-run --tag --new-version $$VERSION_PART patch;')

.PHONY: bump-tag
bump-tag:
	$(shell bash -c 'read -p "Version: " VERSION_PART; \
		source "$(CONDA_HOME)/bin/activate" "$(CONDA_ENV)"; \
		test -f $(CONDA_ENV_PATH)/bin/bumpversion || pip install bumpversion; \
		"$(CONDA_ENV_PATH)/bin/bumpversion" --config-file "$(CURDIR)/.bumpversion.cfg" \
			--verbose --allow-dirty --tag --new-version $$VERSION_PART patch;')

.PHONY: clean
clean: clean-build clean-pyc clean-test clean-ml

.PHONY: clean-build
clean-build:
	@-rm -fr "$(CUR_DIR)/build/"
	@-rm -fr "$(CUR_DIR)/dist/"
	@-rm -fr "$(CUR_DIR)/.eggs/"
	@-find . -type d -name '*.egg-info' -exec rm -fr {} +
	@-find . -type f -name '*.egg-info' -exec rm -fr {} +
	@-find . -type f -name '*.egg' -exec rm -f {} +

.PHONY: clean-env
clean-env:
	@-test -d "$(CONDA_ENV_PATH)" && "$(ANACONDA_HOME)/bin/conda" remove -n "$(CONDA_ENV)" --yes --all

.PHONY: clean-pyc
clean-pyc:
	@-find . -type f -name '*.pyc' -exec rm -f {} +
	@-find . -type f -name '*.pyo' -exec rm -f {} +
	@-find . -type f -name '*~' -exec rm -f {} +
	@-find . -type f -name '__pycache__' -exec rm -fr {} +

.PHONY: clean-test
clean-test:
	@-rm -fr "$(CUR_DIR)/.tox/"
	@-rm -f "$(CUR_DIR)/.coverage"
	@-rm -fr "$(CUR_DIR)/coverage/"
	@-rm -fr "$(CUR_DIR)/.pytest_cache/"

.PHONY: clean-ml
clean-ml:
	# clean thelper sources left over from build
	@-rm -fr "$(CUR_DIR)/src" || true

.PHONY: pep8
pep8:
	@bash -c 'flake8 geoimagenet_ml && echo "All good!"'

.PHONY: test
test: install-dev
	@bash -c 'source "$(ANACONDA_HOME)/bin/activate" "$(CONDA_ENV)"; \
		"$(ANACONDA_HOME)/envs/$(CONDA_ENV)/bin/pytest" tests -vv -m "not online"'

.PHONY: test-all
test-all: install-dev
	@bash -c 'source "$(ANACONDA_HOME)/bin/activate" "$(CONDA_ENV)"; \
		"$(ANACONDA_HOME)/envs/$(CONDA_ENV)/bin/pytest" tests -vv'

.PHONY: test-tox
test-tox: install-dev
	@bash -c 'source "$(ANACONDA_HOME)/bin/activate" "$(CONDA_ENV)"; \
		tox'

.PHONY: coverage
coverage:
	@bash -c 'source "$(ANACONDA_HOME)/bin/activate" "$(CONDA_ENV)"; \
		coverage run --source api setup.py test; \
		coverage report -m; \
		coverage html -d coverage; \
		"$(BROWSER)" coverage/index.html;'

.PHONY: migrate
migrate:
	alembic upgrade head

.PHONY: docs
docs:
	@-rm -f "$(SRC_DIR)/docs/api.rst"
	@-rm -f "$(SRC_DIR)/docs/modules.rst"
	sphinx-apidoc -o "$(SRC_DIR)/docs/" "$(SRC_DIR)"
	@"$(MAKE)" -C "$(SRC_DIR)/docs" clean
	@"$(MAKE)" -C "$(SRC_DIR)/docs" html
	@"$(BROWSER)" "$(SRC_DIR)/docs/_build/html/index.html"

.PHONY: servedocs
servedocs: docs
	watchmedo shell-command -p '*.rst' -c '"$(MAKE)" -C docs html' -R -D .

.PHONY: release
release: clean
	python "$(CUR_DIR)/setup.py" sdist upload
	python "$(CUR_DIR)/setup.py" bdist_wheel upload

.PHONY: dist
dist: clean
	python "$(CUR_DIR)/setup.py" sdist
	python "$(CUR_DIR)/setup.py" bdist_wheel
	ls -l dist

.PHONY: install
install: install-ml install-api
	@echo "Installing all packages..."

.PHONY: install-api
install-api: clean conda-env
	@-bash -c 'source "$(ANACONDA_HOME)/bin/activate" "$(CONDA_ENV)"; pip install -r "$(CUR_DIR)/requirements.txt"'
	@-bash -c 'source "$(ANACONDA_HOME)/bin/activate" "$(CONDA_ENV)"; python "$(CUR_DIR)/setup.py" install'
	@-bash -c 'source "$(ANACONDA_HOME)/bin/activate" "$(CONDA_ENV)"; pip install "$(CUR_DIR)"'

.PHONY: install-ml
install-ml: clean conda-env
	@echo "Installing ML packages..."
	@echo "Installing packages that fail with pip using conda instead"
	@bash -c '"$(ANACONDA_HOME)/bin/conda" install -y -n "$(CONDA_ENV)" \
		--file "$(CUR_DIR)/requirements-gdal.txt" -c conda-forge'
	@bash -c 'source "$(ANACONDA_HOME)/bin/activate" "$(CONDA_ENV)"; pip install -r "$(CUR_DIR)/requirements-ml.txt"'
	$(MAKE) clean-ml

.PHONY: install-dev
install-dev: install
	@echo "Installing test dependencies..."
	@bash -c 'source "$(ANACONDA_HOME)/bin/activate" "$(CONDA_ENV)"; pip install -r "$(CUR_DIR)/requirements-dev.txt"'

.PHONY: update
update: clean
	@-bash -c 'source "$(ANACONDA_HOME)/bin/activate" "$(CONDA_ENV)"; pip install "$(CUR_DIR)"'

.PHONY: version
version:
	@echo "GeoImageNet ML version: $(APP_VERSION)"
	@python -c 'from geoimagenet_ml.__meta__ import __version__; print(__version__)'

## Docker targets

.PHONY: docker-info
docker-info:
	@echo "Will be built, tagged and pushed as: \
		$(DOCKER_REPO):`python -c 'from geoimagenet_ml.__meta__ import __version__; print(__version__)'`"

.PHONY: docker-build
docker-build:
	@bash -c "docker build $(CUR_DIR) \
		-t $(DOCKER_REPO):`python -c 'from geoimagenet_ml.__meta__ import __version__; print(__version__)'`"

.PHONY: docker-push
docker-push:
	@bash -c "docker push $(DOCKER_REPO):`python -c 'from geoimagenet_ml.__meta__ import __version__; print(__version__)'`"

## Supervisor targets

.PHONY: start
start:
	@echo "Starting supervisor service..."
	@-bash -c 'source "$(ANACONDA_HOME)/bin/activate" "$(CONDA_ENV)"; "$(SRC_DIR)/bin/supervisord" start'

.PHONY: stop
stop:
	@echo "Stopping supervisor service..."
	@-bash -c 'source "$(ANACONDA_HOME)/bin/activate" "$(CONDA_ENV)"; "$(SRC_DIR)/bin/supervisord" stop'

.PHONY: restart
restart:
	@echo "Restarting supervisor service..."
	@-bash -c 'source "$(ANACONDA_HOME)/bin/activate" "$(CONDA_ENV)"; "$(SRC_DIR)/bin/supervisord" restart'

.PHONY: status
status:
	@echo "Supervisor status..."
	@-bash -c 'source "$(ANACONDA_HOME)/bin/activate" "$(CONDA_ENV)"; "$(SRC_DIR)/bin/supervisord" status'

## Anaconda targets

.PHONY: anaconda
anaconda:
	@echo "Installing Anaconda..."
	@test -d "$(ANACONDA_HOME)" || curl "$(ANACONDA_URL)/$(FN)" --silent --insecure --output "$(DOWNLOAD_CACHE)/$(FN)"
	@test -d "$(ANACONDA_HOME)" || bash "$(DOWNLOAD_CACHE)/$(FN)" -b -p "$(ANACONDA_HOME)"
	@echo "Add '$(ANACONDA_HOME)/bin' to your PATH variable in '.bashrc'."

.PHONY: conda-config
conda-config: anaconda
	@echo "Update ~/.condarc"
	@"$(ANACONDA_HOME)/bin/conda" config --add envs_dirs $(CONDA_ENVS_DIR)
	@"$(ANACONDA_HOME)/bin/conda" config --set ssl_verify true
	@"$(ANACONDA_HOME)/bin/conda" config --set use_pip true
	@"$(ANACONDA_HOME)/bin/conda" config --set channel_priority true
	@"$(ANACONDA_HOME)/bin/conda" config --set auto_update_conda false
	# cannot mix 'conda-forge' and 'defaults', causes errors for gdal/osgeo related bin packages
	# @"$(ANACONDA_HOME)/bin/conda" config --add channels defaults
	@"$(ANACONDA_HOME)/bin/conda" config --append channels birdhouse
	@"$(ANACONDA_HOME)/bin/conda" config --append channels conda-forge

.PHONY: conda-env
conda-env: anaconda conda-config
	@echo "Update conda environment $(CONDA_ENV) using $(ANACONDA_HOME)..."
	@test -d "$(CONDA_ENV_PATH)" || "$(ANACONDA_HOME)/bin/conda" create -y -n "$(CONDA_ENV)" python=$(PYTHON_VERSION)
	"$(ANACONDA_HOME)/bin/conda" install -y -n "$(CONDA_ENV)" setuptools=$(SETUPTOOLS_VERSION)
	@-bash -c 'source "$(ANACONDA_HOME)/bin/activate" "$(CONDA_ENV)"; pip install --upgrade pip'
