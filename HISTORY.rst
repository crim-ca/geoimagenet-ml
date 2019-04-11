.. :changelog:

History
=======

0.8.0
---------------------

* add ``Magpie`` request to store corresponding user-id to db
* add statistics and action tracking of api requests
* fix incorrectly saved datetime as string in db

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
