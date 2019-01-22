#!/usr/bin/env python
# coding: utf-8


from ccfb.api.store.interfaces import DatabaseInterface


class MemoryDatabase(DatabaseInterface):

    def __init__(self, registry):
        super(MemoryDatabase, self).__init__(registry)

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
        raise NotImplementedError

    def run_migration(self):
        raise NotImplementedError

    def rollback(self):
        raise NotImplementedError

    def get_session(self):
        raise NotImplementedError

    def get_information(self):
        raise NotImplementedError

    def get_revision(self):
        raise NotImplementedError
