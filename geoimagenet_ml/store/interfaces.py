#!/usr/bin/env python
# coding: utf-8

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from geoimagenet_ml.constants import SORT, ORDER, OPERATION                                             # noqa: F401
    from geoimagenet_ml.status import STATUS, CATEGORY                                                      # noqa: F401
    from geoimagenet_ml.store.datatypes import Base, Dataset, Process, Model, Job, Action                   # noqa: F401
    from geoimagenet_ml.typedefs import OptionType, UUID, Any, AnyStr, List, Optional, Tuple, Type, Union   # noqa: F401
    from datetime import datetime                                                                           # noqa: F401
    from pyramid.request import Request                                                                     # noqa: F401
    from io import BufferedIOBase                                                                           # noqa: F401


class DatabaseInterface(object):
    # noinspection PyUnusedLocal
    def __init__(self, settings):
        pass

    @property
    def datasets_store(self):
        # type: () -> DatasetStore
        raise NotImplementedError

    @property
    def models_store(self):
        # type: () -> ModelStore
        raise NotImplementedError

    @property
    def processes_store(self):
        # type: () -> ProcessStore
        raise NotImplementedError

    @property
    def jobs_store(self):
        # type: () -> JobStore
        raise NotImplementedError

    @property
    def actions_store(self):
        # type: () -> ActionStore
        raise NotImplementedError

    def is_ready(self):
        # type: () -> bool
        raise NotImplementedError

    def run_migration(self):
        # type: () -> None
        raise NotImplementedError

    def rollback(self):
        # type: () -> None
        """Rollback current database transaction."""
        raise NotImplementedError

    def get_session(self):
        # type: () -> Any
        raise NotImplementedError

    def get_information(self):
        # type: () -> OptionType
        """
        :returns: {'version': version, 'type': db_type}
        """
        raise NotImplementedError

    def get_revision(self):
        # type: () -> AnyStr
        return self.get_information().get("version")


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

    def delete_dataset(self, dataset_uuid, request=None):
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

    # noinspection PyShadowingBuiltins
    def find_datasets(self,
                      name=None,        # type: Optional[AnyStr]
                      type=None,        # type: Optional[AnyStr]
                      status=None,      # type: Optional[Union[STATUS, CATEGORY]]
                      sort=None,        # type: Optional[SORT]
                      order=None,       # type: Optional[ORDER]
                      limit=None,       # type: Optional[int]
                      request=None,     # type: Optional[Request]
                      ):                # type: (...) -> Tuple[List[Dataset], int]
        """
        Finds all datasets in database matching search filters.
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
        # type: () -> bool
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

    def find_jobs(self,
                  page=0,           # type: int
                  limit=10,         # type: int
                  process=None,     # type: Optional[UUID]
                  service=None,     # type: Optional[UUID]
                  tags=None,        # type: Optional[List[AnyStr]]
                  user=None,        # type: Optional[int]
                  status=None,      # type: Optional[Union[STATUS, CATEGORY]]
                  sort=None,        # type: Optional[SORT]
                  order=None,       # type: Optional[ORDER]
                  request=None,     # type: Optional[Request]
                  ):                # type: (...) -> Tuple[List[Job], int]
        """
        Finds all jobs in database matching search filters.

        Returns a tuple of filtered ``items`` and their ``count``, where ``items`` can have paging and be limited
        to a maximum per page, but ``count`` always indicate the `total` number of matches.
        """
        raise NotImplementedError

    def fetch_by_uuid(self, job_uuid, request=None):
        # type: (UUID, Optional[Request]) -> Job
        """
        Get job for given ``uuid`` from storage.
        """
        raise NotImplementedError


class ActionStore(object):
    """
    Storage for local actions.
    """
    def save_action(self, type_or_item, operation, request=None):
        # type: (Union[Type[Base], Base], OPERATION, Optional[Request]) -> Action
        """Stores a new action in storage."""
        raise NotImplementedError

    def find_actions(self,
                     item_type=None,    # type: Optional[Any]
                     item=None,         # type: Optional[UUID]
                     operation=None,    # type: Optional[OPERATION]
                     user=None,         # type: Optional[int]
                     start=None,        # type: Optional[datetime]
                     end=None,          # type: Optional[datetime]
                     sort=None,         # type: Optional[SORT]
                     order=None,        # type: Optional[ORDER]
                     page=None,         # type: Optional[int]
                     limit=None,        # type: Optional[int]
                     ):                 # type: (...) -> Tuple[List[Action], int]
        """
        Get all matching actions from storage.

        Returns a tuple of filtered ``items`` and their ``count``, where ``items`` can have paging and be limited
        to a maximum per page, but ``count`` always indicate the `total` number of matches.
        """
        raise NotImplementedError
