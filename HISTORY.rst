.. :changelog:

History
=======

Unreleased
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
