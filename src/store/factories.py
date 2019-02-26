#!/usr/bin/env python
# coding: utf-8

from geoimagenet_ml.store.databases.memory import MemoryDatabase
from geoimagenet_ml.store.databases.mongodb import MongoDatabase
from geoimagenet_ml.store.databases.postgres import PostgresDatabase
from geoimagenet_ml.store.databases.types import MEMORY_TYPE, MONGODB_TYPE, POSTGRES_TYPE
from sqlalchemy.orm.session import Session
from pyramid.registry import Registry
import pymongo
import os
import time
import logging
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from geoimagenet_ml.typedefs import AnyStr, Union, SettingsType  # noqa: F401
    from geoimagenet_ml.store.interfaces import DatabaseInterface   # noqa: F401
logger = logging.getLogger(__name__)


def get_database_type(specification):
    # type: (Union[Session, Registry, SettingsType]) -> AnyStr
    """
    Return the db type string from a given specification or environment variable
    'GEOIMAGENET_ML_API_DB_FACTORY' if defined.
    :param specification: any of active `db_session`, pyramid registry or settings dictionary.
    :return: db type
    """
    db_type = os.getenv('GEOIMAGENET_ML_API_DB_FACTORY')
    if db_type:
        return db_type
    # noinspection PyUnresolvedReferences
    if isinstance(specification, pymongo.database.Database):
        return MONGODB_TYPE
    elif isinstance(specification, Session):
        return POSTGRES_TYPE
    elif isinstance(specification, Registry):
        return specification.settings.get('geoimagenet_ml.api.db_factory')
    elif isinstance(specification, dict):
        return specification.get('geoimagenet_ml.api.db_factory')
    raise NotImplementedError("Unknown type `{}` to retrieve database type.".format(type(specification)))


def database_factory(registry):
    # type: (Registry) -> DatabaseInterface
    db_type = get_database_type(registry)
    if db_type == MEMORY_TYPE:
        return MemoryDatabase(registry)
    if db_type == MONGODB_TYPE:
        return MongoDatabase(registry)
    if db_type == POSTGRES_TYPE:
        return PostgresDatabase(registry)
    raise NotImplementedError("Unknown db_factory type: `{}`.".format(db_type))


def migrate_database_when_ready(specification):
    # forge registry from settings as needed
    if isinstance(specification, dict):
        registry = Registry()
        registry.settings = specification
    else:
        registry = specification
    try:
        db = database_factory(registry)
        db.run_migration()
    except Exception as e:
        raise Exception('Database migration failed with error: [{}].'.format(str(e)))
    if not db.is_ready():
        time.sleep(2)
        raise Exception('Database not ready.')
