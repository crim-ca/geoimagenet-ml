#!/usr/bin/env python
# coding: utf-8


# noinspection PyUnusedLocal
def includeme(config):
    """
    Initialize the model for a Pyramid app.
    """
    from geoimagenet_ml.store.factories import database_factory, get_database_type, POSTGRES_TYPE

    if get_database_type(config.registry) == POSTGRES_TYPE:
        # use pyramid_tm to hook the transaction lifecycle to the request
        config.include('pyramid_tm')

        from geoimagenet_ml.store.interfaces import DatabaseInterface   # noqa: F401
        config.registry.db = database_factory(config.registry)  # type: DatabaseInterface
