.. :changelog:

History
=======

1.0.0
---------------------

* add visibility update ``PUT`` requests for ``Job`` and ``Model`` (GEOIM-137)
* add strong and enforced input validation of datatype parameters
* upgrade db to version ``"4"``, loading previous objects could cause errors (input validation failures)
* add more unittests for input validation
* add test for new visibility routes (GEOIM-137)
* add test for job submission
* fix returned body response from job submission to match rest of API format
* fix multiple API schema definitions

0.8.0
---------------------

* add request to store corresponding user-id to db if specified with ``MAGPIE_USER_URL``
* add statistics and action tracking of API requests
* add user creating a new dataset, model, job, process
* add started timestamp for jobs not immediately running (accepted but pending), duration based on it
* fix rare race condition of job update caused by updated job details not retrieved from db
* fix incorrectly saved datetime as string in db
* fix API schemas and drop unused items
* restructure enum components used across the project

0.7.1
---------------------

* add pip check on install to ensure all package requirements/dependencies are met recursively
* fix supervisor path reference to source
* fix db invalid index reference

0.7.0
---------------------

* rebase source directory from ``src`` to ``geoimagenet_ml`` to solve installation/debug issues
* add more validation of job inputs
* add and fix utility make targets
* fix gdal package and unresolved symbol error
* fix typing and general code formatting
* fix and complete `batch-creation` job execution

0.6.x
---------------------

* Redefine most of the process creation procedure.
* Batch of patches creation process
* Model testing process
* Automatically create default processes on start if not available in db.
* Add dataset download route.
* Typing and validations fixes.

0.5.x
---------------------

* More refactoring and fixes for functional ML on server.

0.4.x
---------------------

* Full refactoring of project directories and imports.

0.3.x
---------------------

* Setup databases, datasets, models, processes and other interfaces with REST API.
* Setup API schemas for documentation.

0.2.x
---------------------

* Switch between mongodb/postgres databases (postgres schemas not all supported)

0.1.x
---------------------

* Initial release.
