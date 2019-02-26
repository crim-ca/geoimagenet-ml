#!/usr/bin/env python
# coding: utf-8

from typing import TYPE_CHECKING, Any, AnyStr, List, Optional                   # noqa: F401
if TYPE_CHECKING:
    from geoimagenet_ml.store.datatypes import Dataset, Process, Model, Job     # noqa: F401
    from geoimagenet_ml.typedefs import OptionType, UUID                        # noqa: F401
    from pyramid.request import Request                                         # noqa: F401
    from io import BufferedIOBase                                               # noqa: F401


class DatabaseInterface(object):
    # noinspection PyUnusedLocal
    def __init__(self, registry):
        pass

    @property
    def datasets_store(self):
        # type: (...) -> DatasetStore
        raise NotImplementedError

    @property
    def models_store(self):
        # type: (...) -> ModelStore
        raise NotImplementedError

    @property
    def processes_store(self):
        # type: (...) -> ProcessStore
        raise NotImplementedError

    @property
    def jobs_store(self):
        # type: (...) -> JobStore
        raise NotImplementedError

    def is_ready(self):
        # type: (...) -> bool
        raise NotImplementedError

    def run_migration(self):
        # type: (...) -> None
        raise NotImplementedError

    def rollback(self):
        # type: (...) -> None
        """Rollback current database transaction."""
        raise NotImplementedError

    def get_session(self):
        # type: (...) -> Any
        raise NotImplementedError

    def get_information(self):
        # type: (...) -> OptionType
        """
        :returns: {'version': version, 'type': db_type}
        """
        raise NotImplementedError

    def get_revision(self):
        # type: (...) -> AnyStr
        return self.get_information().get('version')


class DatasetStore(object):
    """
    Storage for local datasets.
    """

    def save_dataset(self, dataset, request=None):
        # type: (Dataset, Optional[Request]) -> Dataset
        """
        Stores a dataset in storage.
        """
        raise NotImplementedError

    def delete_dataset(self, process_uuid, request=None):
        # type: (UUID, Optional[Request]) -> bool
        """
        Removes dataset from database.
        """
        raise NotImplementedError

    def list_datasets(self, request=None):
        # type: (Optional[Request]) -> List[Dataset]
        """
        Lists all datasets in database.
        """
        raise NotImplementedError

    def fetch_by_uuid(self, dataset_uuid, request=None):
        # type: (UUID, Optional[Request]) -> Dataset
        """
        Get dataset for given ``uuid`` from storage.
        """
        raise NotImplementedError


class ModelStore(object):
    """
    Storage for local models.
    """

    def save_model(self, model, data=None, request=None):
        # type: (Model, Optional[OptionType, BufferedIOBase], Optional[Request]) -> Model
        """
        Stores a model in storage.
        """
        raise NotImplementedError

    def delete_model(self, process_uuid, request=None):
        # type: (UUID, Optional[Request]) -> bool
        """
        Removes model from database.
        """
        raise NotImplementedError

    def list_models(self, request=None):
        # type: (Optional[Request]) -> List[Model]
        """
        Lists all models in database.
        """
        raise NotImplementedError

    def fetch_by_uuid(self, model_uuid, request=None):
        # type: (UUID, Optional[Request]) -> Model
        """
        Get model for given ``uuid`` from storage.
        """
        raise NotImplementedError

    def clear_models(self):
        # type: (...) -> bool
        """
        Deletes all models from storage.
        """
        raise NotImplementedError


class ProcessStore(object):
    """
    Storage for local WPS processes.
    """

    def save_process(self, process, overwrite=True, request=None):
        # type: (Process, Optional[bool], Optional[Request]) -> Process
        """
        Stores a WPS process in storage.
        """
        raise NotImplementedError

    def delete_process(self, process_id, request=None):
        # type: (UUID, Optional[Request]) -> bool
        """
        Removes process from database.
        """
        raise NotImplementedError

    def list_processes(self, request=None):
        # type: (Optional[Request]) -> List[Process]
        """
        Lists all processes in database.
        """
        raise NotImplementedError

    def fetch_by_uuid(self, process_id, request=None):
        # type: (UUID, Optional[Request]) -> Process
        """
        Get process for given ``uuid`` from storage.
        """
        raise NotImplementedError

    def fetch_by_identifier(self, process_identifier, request=None):
        # type: (AnyStr, Optional[Request]) -> Process
        """
        Get process for given ``identifier`` from storage.
        """
        raise NotImplementedError


class JobStore(object):
    """
    Storage for local jobs.
    """

    def save_job(self, job, request=None):
        # type: (Job, Optional[Request]) -> Job
        """
        Stores a job in storage.
        """
        raise NotImplementedError

    def update_job(self, job, request=None):
        # type: (Job, Optional[Request]) -> Job
        """
        Updates a job parameters in storage.
        """
        raise NotImplementedError

    def delete_job(self, process_uuid, request=None):
        # type: (UUID, Optional[Request]) -> bool
        """
        Removes job from database.
        """
        raise NotImplementedError

    def list_jobs(self, request=None):
        # type: (Optional[Request]) -> List[Job]
        """
        Lists all jobs in database.
        """
        raise NotImplementedError

    def fetch_by_uuid(self, job_uuid, request=None):
        # type: (UUID, Optional[Request]) -> Job
        """
        Get job for given ``uuid`` from storage.
        """
        raise NotImplementedError
