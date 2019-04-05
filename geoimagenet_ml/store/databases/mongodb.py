#!/usr/bin/env python
# coding: utf-8

from geoimagenet_ml.processes.types import process_mapping
from geoimagenet_ml.store.interfaces import DatabaseInterface
from geoimagenet_ml.store.databases.types import MONGODB_TYPE
from geoimagenet_ml.store.adapters.mongodb import (
    MongodbDatasetStore, MongodbModelStore, MongodbProcessStore, MongodbJobStore
)
import pymongo
import os


class MongoDatabase(DatabaseInterface):
    _database = None
    _settings = None

    def __init__(self, registry):
        super(MongoDatabase, self).__init__(registry)
        self._database = MongoDB.get(registry)
        self._settings = registry.settings
        self.run_migration()

    @property
    def datasets_store(self):
        return MongodbDatasetStore(collection=self._database.datasets, settings=self._settings)

    @property
    def models_store(self):
        return MongodbModelStore(collection=self._database.models, settings=self._settings)

    @property
    def processes_store(self):
        store = MongodbProcessStore(collection=self._database.processes, settings=self._settings,
                                    default_processes=None if MongoDB.is_init() else process_mapping)
        MongoDB.mark_init()
        return store

    @property
    def jobs_store(self):
        return MongodbJobStore(collection=self._database.jobs, settings=self._settings)

    def is_ready(self):
        return self._database is not None and self._settings is not None

    def rollback(self):
        pass

    def get_session(self):
        return self._database

    def get_information(self):
        result = list(self._database.version.find().limit(1))[0]
        db_version = result["version_num"]
        return {"version": db_version, "type": MONGODB_TYPE}

    def run_migration(self):
        if self._database.version.count_documents({}) < 1:
            self._database.version.insert_one({"version_num": "3"})
        else:
            # TODO: do migration according to found version
            pass


class MongoDB:
    __db = None
    __init = False

    @classmethod
    def is_init(cls):
        return cls.__init

    @classmethod
    def mark_init(cls):
        cls.__init = True

    @classmethod
    def get(cls, registry):
        if not cls.__db:
            settings = registry.settings
            username = os.getenv("MONGODB_USER") or settings.get("mongodb.user")
            password = os.getenv("MONGODB_PASSWORD") or settings.get("mongodb.password")
            client = pymongo.MongoClient(
                os.getenv("MONGODB_HOST") or settings.get("mongodb.host"),
                int(os.getenv("MONGODB_PORT") or settings.get("mongodb.port")),
                # avoid empty string causing an error
                username=username if username else None,
                password=password if password else None,
            )
            cls.__db = client[os.getenv("MONGODB_DB_NAME") or settings.get("mongodb.db_name")]
            cls.__db.datasets.create_index("uuid", unique=True)
            cls.__db.models.create_index("uuid", unique=True)
            cls.__db.processes.create_index("uuid", unique=True)
            cls.__db.version.create_index("version_num", unique=True)
        return cls.__db
