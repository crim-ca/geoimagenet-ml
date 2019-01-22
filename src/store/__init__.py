#!/usr/bin/env python
# coding: utf-8


# noinspection PyUnusedLocal
def includeme(config):
    """
    Initialize the model for a Pyramid app.
    """

    # use pyramid_tm to hook the transaction lifecycle to the request
    config.include('pyramid_tm')

    from src.store.factories import database_factory
    from src.store.interfaces import DatabaseInterface
    config.registry.db = database_factory(config.registry)  # type: DatabaseInterface

    # make `request.db` available for use in Pyramid
    config.add_request_method(lambda r: config.registry.db, 'db', reify=True)
