======================================
GeoImageNet ML
======================================

`GeoImageNet ML documentation`_
.. _geoimagenet_ml_latest: http://localhost:3000/api/?urls.primaryName=latest
.. _GeoImageNet ML documentation: REST API documentation: _geoimagenet_ml_latest


Build/push docker image
=======================

At the command line::

    $ make docker-build
    $ make docker-push

The image will be built and tagged with the current package version.
Pushing the image depends on the `DOCKER_REPO` value in the `Makefile`.

Build & install package
=======================

At the command line::

    $ make install

Start service
=============

At the command line::

    $ make start

