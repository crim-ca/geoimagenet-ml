#!/usr/bin/env python
# coding: utf-8


from geoimagenet_ml.store.databases.types import MEMORY_TYPE
from geoimagenet_ml.store.interfaces import DatabaseInterface
from geoimagenet_ml.store.adapters.memory import (
    MemoryDatasetStore, MemoryModelStore, MemoryProcessStore, MemoryJobStore, MemoryActionStore
)


class MemoryDatabase(DatabaseInterface):

    def __init__(self, settings):
        super(MemoryDatabase, self).__init__(settings)
        _db = self._settings.get("_database")
        self._database = _db or {
            "datasets": MemoryDatasetStore(),
            "models": MemoryModelStore(),
            "processes": MemoryProcessStore(),
            "jobs": MemoryJobStore(),
            "actions": MemoryActionStore()
        }
        self._settings["_database"] = self._database

    @property
    def datasets_store(self):
        return self._database["datasets"]

    @property
    def models_store(self):
        return self._database["models"]

    @property
    def processes_store(self):
        return self._database["processes"]

    @property
    def jobs_store(self):
        return self._database["jobs"]

    @property
    def actions_store(self):
        return self._database["actions"]

    def is_ready(self):
        return True

    def run_migration(self):
        pass

    def rollback(self):
        pass

    def get_session(self):
        return None

    def get_information(self):
        return {"type": MEMORY_TYPE, "version": "0.0.0"}

    def get_revision(self):
        return "temp"
