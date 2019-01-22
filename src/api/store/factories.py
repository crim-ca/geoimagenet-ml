#!/usr/bin/env python
# coding: utf-8

from ccfb.api.store.databases.memory import MemoryDatabase
from ccfb.api.store.databases.mongodb import MongoDatabase
from ccfb.api.store.databases.postgres import PostgresDatabase
from ccfb.api.store.databases.types import MEMORY_TYPE, MONGODB_TYPE, POSTGRES_TYPE
from ccfb.api.definitions.pyramid_definitions import Registry
from ccfb.api.definitions.sqlalchemy_definitions import Session
from ccfb.api.definitions.typing_definitions import *
import pymongo
import os
import time
import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ccfb.api.store.interfaces import DatabaseInterface


def get_database_type(specification):
    # type: (Union[Session, Registry, SettingDict]) -> AnyStr
    """
    Return the db type string from a given specification or environment variable 'CCFB_API_DB_FACTORY' if defined.
    :param specification: any of active `db_session`, pyramid registry or settings dictionary.
    :return: db type
    """
    db_type = os.getenv('CCFB_API_DB_FACTORY')
    if db_type:
        return db_type
    # noinspection PyUnresolvedReferences
    if isinstance(specification, pymongo.database.Database):
        return MONGODB_TYPE
    elif isinstance(specification, Session):
        return POSTGRES_TYPE
    elif isinstance(specification, Registry):
        return specification.settings.get('src.api.db_factory')
    elif isinstance(specification, dict):
        return specification.get('src.api.db_factory')
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
