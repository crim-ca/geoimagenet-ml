.. :changelog:

History
=======

Unreleased
---------------------

Changes
~~~~~~~~~~~~~~~~~~~~~
* remove git clone of ``thelper`` in ``Jenkinsfile`` as it is installed with requirements pinned version from PyPI

Fixes
~~~~~~~~~~~~~~~~~~~~~
* fix incorrect descriptions returned for ``GET`` requests of job logs and exceptions
* fix key error when parsing model task defined as literal string [GEOIM-245, GEOIM-247]
* pin ``gdal==2.4.2`` as ``gdal==3.0.1`` fails on import (https://github.com/OSGeo/gdal/issues/1759)

1.4.0 (2019-08-06)
---------------------

Changes
~~~~~~~~~~~~~~~~~~~~~
* add error logging in case of request generating an exception (voluntary or by execution problem)
* change ``gdal`` requirements and imports to avoid ``osgeo`` variant breaking too easily
* support older ``thelper`` model checkpoint definitions using string parameters instead of JSON [GEOIM-241]
* update to ``thelper>=0.3.7``, help better support old format for model task definitions [GEOIM-247]
* disable `Jenkins` online tests execution (download/upload model) as they now require login [GEOIM-180]

Fixes
~~~~~~~~~~~~~~~~~~~~~
* avoid key error of WPS-like process properties cleanup during initialization [GEOIM-225]
* better filter and group common job error message to Sentry [GEOIM-227]
* update test model checkpoint for unittest execution with more relevant ``thelper`` configuration [GEOIM-180]

1.3.0 (2019-06-18)
---------------------

Changes
~~~~~~~~~~~~~~~~~~~~~
* add additional metrics (Top-1, Top-5) in process `model-tester` execution results (GEOIM-163)
* add fetching of taxonomy definition during `batch-creation` process execution from input URL (GEOIM-161)
* save retrieved taxonomy definition to generated dataset batch (GEOIM-162)
* enforce ``thelper>=0.3.1`` to use new operation modes and back compatibility import methods
* add model checkpoint validation checks and raise ``ModelValidationError`` (``HTTP Forbidden [403]``) if invalid
    - disallow loading a model task defined as literal string, must be a JSON definition of parameters
    - disallow loading a model task not matched within existing job mapping (processing steps must be fully defined)
    - disallow loading a model task without minimal configuration parameters required during `model-tester` execution
* add named key variables to help understand corresponding items across dictionary definitions and generated results
* add more detail to produced results of `model-tester` process (GEOIM-163)

Fixes
~~~~~~~~~~~~~~~~~~~~~
* fix incorrectly returned message from ``POST /processes/{id}/jobs`` requests in case of error

1.2.0 (2019-06-11)
---------------------

Changes
~~~~~~~~~~~~~~~~~~~~~
* add ``BatchTestPatchesDatasetLoader`` implementation that loads a generated dataset of test patches from process
  `batch-creation` execution to evaluate them against a registered model using process `model-tester`.
* add ``geoimagenet_ml.ml.jobs_path`` configuration setting to indicate where process job logging should be stored.
  (mostly during process `model-tester` execution)
* adjust model and dataset definitions to employ the same `task` until resolved (GEOIM-153)

Fixes
~~~~~~~~~~~~~~~~~~~~~
* fix bug generated during `model-tester` process execution attempting to update ``Dataset`` parameters (GEOIM-149)

1.1.2 (2019-06-06)
---------------------

* add variable ``GEOIMAGENET_ML_SENTRY_SERVER_NAME`` to allow overriding ``server_name`` value reported by sentry.
* fix bug caused by invalid sub-item type checker of job exception field (GEOIM-146).

1.1.1 (2019-06-05)
---------------------

* fix bug caused by invalid list/tuple concatenation in job exception field (GEOIM-145).

1.1.0 (2019-05-16)
---------------------

Changes
~~~~~~~~~~~~~~~~~~~~~
* fetching ``latest`` job for processes where ``limit_single_job=False`` will not raise ``500`` if job count ``>1``
* fetching ``current`` job for processes where ``limit_single_job=False`` will raise ``403`` because of multiple matches
* fetching ``current`` job for processes where ``limit_single_job=True`` raises ``404`` with more appropriate message
* change job ``mark_started`` and ``mark_finished`` methods to ``update_started_datetime`` and
  ``update_finished_datetime`` respectively to be more specific since they do not actually set the ``status`` field
* use enum for ``current`` and ``latest`` keywords
* add additional input format validation during job submission
* add filtering of job search with multiple ``STATUS`` and/or ``CATEGORY`` simultaneously
* add tests for ``current`` and ``latest`` jobs use cases
* add tests for job submission input type validation
* update bump version ``Makefile`` targets and config
* enforce typing of enum string sub-type and unique constraint

Fixes
~~~~~~~~~~~~~~~~~~~~~
* fix process ``limit_single_job`` field incorrectly set in database
* fix process ``reference`` field incorrectly set in database
* fix double dot (``. .``) string ending not correctly cleaned up for response, notably process abstract field
  (requires process recreation or update if already inserted in the database)
* fix typing of ``ExtendedEnumMeta.get()`` method return value for expected corresponding enums

1.0.0
---------------------

Changes
~~~~~~~~~~~~~~~~~~~~~
* add visibility update ``PUT`` requests for ``Job`` and ``Model`` (GEOIM-137)
* add strong and enforced input validation of datatype parameters
* upgrade db to version ``"4"``, loading previous objects could cause errors (input validation failures)
* add more unittests for input validation
* add test for new visibility routes (GEOIM-137)
* add test for job submission
* add sentry-sdk integration (GEOIM-118)

Fixes
~~~~~~~~~~~~~~~~~~~~~
* fix returned body response from job submission to match rest of API format
* fix multiple API schema definitions

0.8.0
---------------------

Changes
~~~~~~~~~~~~~~~~~~~~~
* add request to store corresponding user-id to db if specified with ``MAGPIE_USER_URL``
* add statistics and action tracking of API requests
* add user creating a new dataset, model, job, process
* add started timestamp for jobs not immediately running (accepted but pending), duration based on it
* restructure enum components used across the project

Fixes
~~~~~~~~~~~~~~~~~~~~~
* fix rare race condition of job update caused by updated job details not retrieved from db
* fix incorrectly saved datetime as string in db
* fix API schemas and drop unused items

0.7.1
---------------------

Changes
~~~~~~~~~~~~~~~~~~~~~
* add pip check on install to ensure all package requirements/dependencies are met recursively

Fixes
~~~~~~~~~~~~~~~~~~~~~
* fix supervisor path reference to source
* fix db invalid index reference

0.7.0
---------------------

Changes
~~~~~~~~~~~~~~~~~~~~~
* rebase source directory from ``src`` to ``geoimagenet_ml`` to solve installation/debug issues
* add more validation of job inputs
* add and fix utility make targets

Fixes
~~~~~~~~~~~~~~~~~~~~~
* fix gdal package and unresolved symbol error
* fix typing and general code formatting
* fix and complete `batch-creation` job execution

0.6.x
---------------------

Changes
~~~~~~~~~~~~~~~~~~~~~
* Redefine most of the process creation procedure.
* Batch of patches creation process
* Model testing process
* Automatically create default processes on start if not available in db.
* Add dataset download route.

Fixes
~~~~~~~~~~~~~~~~~~~~~
* Fix typing and validations.

0.5.x
---------------------

Changes
~~~~~~~~~~~~~~~~~~~~~
* More refactoring and fixes for functional ML on server.

0.4.x
---------------------

Changes
~~~~~~~~~~~~~~~~~~~~~
* Full refactoring of project directories and imports.

0.3.x
---------------------

Changes
~~~~~~~~~~~~~~~~~~~~~
* Setup databases, datasets, models, processes and other interfaces with REST API.
* Setup API schemas for documentation.

0.2.x
---------------------

Changes
~~~~~~~~~~~~~~~~~~~~~
* Switch between mongodb/postgres databases (postgres schemas not all supported)

0.1.x
---------------------

* Initial release.
