#!/usr/bin/env python
# coding: utf-8

from geoimagenet_ml import GEOIMAGENET_ML_API_DIR
from geoimagenet_ml.store.databases import models
from geoimagenet_ml.store.databases.types import POSTGRES_TYPE
from geoimagenet_ml.store.interfaces import DatabaseInterface
from geoimagenet_ml.utils import isclass
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.orm import sessionmaker, configure_mappers
from sqlalchemy import engine_from_config
from zope.sqlalchemy import register
import sqlalchemy as sa
import transaction
import inspect
import os


# run configure_mappers after defining all of the models to ensure
# all relationships can be setup
configure_mappers()


class PostgresDatabase(DatabaseInterface):
    _db_session = None
    _settings = None

    def __init__(self, settings):
        super(PostgresDatabase, self).__init__(settings)
        self._settings = settings
        self._db_session = get_postgresdb_session_from_settings(settings)

    @property
    def datasets_store(self):
        raise NotImplementedError

    @property
    def models_store(self):
        raise NotImplementedError

    @property
    def processes_store(self):
        raise NotImplementedError

    @property
    def jobs_store(self):
        raise NotImplementedError

    def is_ready(self):
        return is_database_ready(self._settings)

    def run_migration(self):
        run_database_migration(self._settings)

    def rollback(self):
        self._db_session.rollback()

    def get_session(self):
        return self._db_session

    def get_information(self):
        s = sa.sql.select(['version_num'], from_obj='alembic_version')
        result = self._db_session.execute(s).fetchone()
        db_version = result['version_num']
        return {'version': db_version, 'type': POSTGRES_TYPE}


def get_postgresdb_url(settings=None):
    return "postgresql://%s:%s@%s:%s/%s" % (
        os.getenv("POSTGRES_USER", "geoimagenet") or settings.get('postgres.user'),
        os.getenv("POSTGRES_PASSWORD", "qwerty") or settings.get('postgres.password'),
        os.getenv("POSTGRES_HOST", "postgres") or settings.get('postgres.host'),
        os.getenv("POSTGRES_PORT", "5444") or settings.get('postgres.port'),
        os.getenv("POSTGRES_DB_NAME", "geoimagenet-ml") or settings.get('postgres.db_name'),
    )


def get_postgres_engine(settings, prefix='sqlalchemy.'):
    settings[prefix+'url'] = get_postgresdb_url(settings)
    return engine_from_config(settings, prefix)


def get_session_factory(engine):
    factory = sessionmaker()
    factory.configure(bind=engine)
    return factory


def get_tm_session(session_factory, transaction_manager):
    """
    Get a ``sqlalchemy.orm.Session`` instance backed by a transaction.

    This function will hook the session to the transaction manager which
    will take care of committing any changes.

    - When using pyramid_tm it will automatically be committed or aborted
      depending on whether an exception is raised.

    - When using scripts you should wrap the session in a manager yourself.
      For example::

          import transaction

          engine = get_engine(settings)
          session_factory = get_session_factory(engine)
          with transaction.manager:
              db_session = get_tm_session(session_factory, transaction.manager)

    """
    db_session = session_factory()
    register(db_session, transaction_manager=transaction_manager)
    return db_session


def get_alembic_ini_path():
    return '{path}/config/alembic.ini'.format(path=GEOIMAGENET_ML_API_DIR)


def run_database_migration(settings):
    import alembic
    if settings.get('geoimagenet_ml.api.db_factory') == POSTGRES_TYPE:
        alembic_args = ['-c', get_alembic_ini_path(), 'upgrade', 'heads']
        # noinspection PyUnresolvedReferences
        alembic.config.main(argv=alembic_args)  # noqa: F401


# noinspection PyUnusedLocal
def is_database_ready(settings):
    inspector = Inspector.from_engine(get_postgres_engine(dict()))
    table_names = inspector.get_table_names()

    for name, obj in inspect.getmembers(models):
        if isclass(obj):
            # noinspection PyBroadException
            try:
                curr_table_name = obj.__tablename__
                if curr_table_name not in table_names:
                    return False
            except Exception:
                continue
    return True


def get_postgresdb_session_from_settings(settings):
    session_factory = get_session_factory(get_postgres_engine(settings))
    db_session = get_tm_session(session_factory, transaction)
    return db_session
