"""
Read or write data from or to local memory.

Though not very valuable in a production setup, these store adapters are great
for testing purposes.
"""

from pyramid.request import Request

from geoimagenet_ml.store import exceptions as db_exc
from geoimagenet_ml.store.datatypes import Action
from geoimagenet_ml.store.interfaces import ActionStore, DatasetStore, JobStore, ModelStore, ProcessStore
from geoimagenet_ml.utils import get_sane_name, get_user_id


class MemoryActionStore(ActionStore):
    """
    Stores actions in memory. Useful for testing purposes.
    """
    memory = {}

    def find_actions(self, item_type=None, item=None, operation=None, user=None, start=None, end=None, sort=None,
                     order=None, page=None, limit=None):
        raise NotImplementedError

    def save_action(self, type_or_item, operation, request=None):
        try:
            action = Action(type=type_or_item, operation=operation)
        except Exception as exc:
            raise db_exc.ActionInstanceError("Invalid 'Action' instance raised an error: [{!r}].".format(exc))

        # automatically add additional request details
        if isinstance(request, Request):
            action.user = get_user_id(request)
            action.path = request.path
            action.method = request.method.upper()

        self.memory[action.uuid] = action


class MemoryDatasetStore(DatasetStore):
    """
    Stores datasets in memory. Useful for testing purposes.
    """
    memory = {}

    def save_dataset(self, dataset, overwrite=False, request=None):
        if dataset.uuid in self.memory and not overwrite:
            raise db_exc.DatasetConflictError("Dataset '{}' already exists.".format(dataset.uuid))
        self.memory[dataset.uuid] = dataset
        return dataset

    def delete_dataset(self, dataset_uuid, request=None):
        dataset = self.memory.pop(dataset_uuid, None)
        return dataset is not None

    def list_datasets(self, request=None):
        return list(self.memory.values())

    def fetch_by_uuid(self, dataset_uuid, request=None):
        dataset = self.memory.get(dataset_uuid)
        if not dataset:
            raise db_exc.DatasetNotFoundError("Dataset '{}' does not exist.".format(dataset_uuid))
        return dataset

    def find_datasets(self, name=None, type=None, status=None, sort=None, order=None, limit=None, request=None):
        raise NotImplementedError


class MemoryModelStore(ModelStore):
    """
    Stores models in memory. Useful for testing purposes.
    """
    memory = {}

    def update_model(self, model, request=None, **fields):
        self.memory[model.uuid] = model
        return model

    def delete_model(self, model_uuid, request=None):
        model = self.memory.pop(model_uuid, None)
        return model is not None

    def list_models(self, request=None):
        return list(self.memory.values())

    def fetch_by_uuid(self, model_uuid, request=None):
        model = self.memory.get(model_uuid)
        if not model:
            raise db_exc.ModelNotFoundError("Model '{}' does not exist.".format(model_uuid))
        return model

    def clear_models(self):
        self.memory = {}

    def save_model(self, model, request=None):
        if model.uuid in self.memory:
            raise db_exc.ModelConflictError("Model '{}' already exists.".format(model.uuid))
        self.memory[model.uuid] = model
        return model


class MemoryJobStore(JobStore):
    """
    Stores jobs in memory. Useful for testing purposes.
    """
    memory = {}

    def save_job(self, job, request=None):
        if job.uuid in self.memory:
            raise db_exc.JobConflictError("Job '{}' already exists.".format(job.uuid))
        self.memory[job.uuid] = job

    def update_job(self, job, allow_unmodified=False, request=None):
        self.memory[job.uuid] = job
        return job

    def delete_job(self, job_uuid, request=None):
        job = self.memory.pop(job_uuid, None)
        return job is not None

    def list_jobs(self, request=None):
        return list(self.memory.values())

    def fetch_by_uuid(self, job_uuid, request=None):
        job = self.memory.get(job_uuid)
        if not job:
            raise db_exc.JobNotFoundError("Job '{}' does not exist.".format(job_uuid))
        return job

    def find_jobs(self, page=0, limit=10, process=None, service=None, tags=None, user=None, status=None, sort=None,
                  order=None, request=None):
        raise NotImplementedError


class MemoryProcessStore(ProcessStore):
    """
    Stores WPS processes in memory. Useful for testing purposes.
    """

    def __init__(self, init_processes=None):
        self.name_index = {}
        if isinstance(init_processes, list):
            for process in init_processes:
                self.save_process(process)

    def save_process(self, process, overwrite=True, request=None):
        """
        Stores a WPS process in storage.
        """
        sane_name = get_sane_name(process.title)
        if not self.name_index.get(sane_name) or overwrite:
            process['title'] = sane_name
            self.name_index[sane_name] = process

    def delete_process(self, name, request=None):
        """
        Removes process from database.
        """
        sane_name = get_sane_name(name)
        if self.name_index.get(sane_name):
            del self.name_index[sane_name]

    def list_processes(self, request=None):
        """
        Lists all processes in database.
        """
        return [process.title for process in self.name_index]

    def fetch_by_uuid(self, process_uuid, request=None):
        """
        Get process for given ``uuid`` from storage.
        """
        sane_name = get_sane_name(process_uuid)
        process = self.name_index.get(sane_name)
        return process

    def fetch_by_identifier(self, process_id, request=None):
        """
        Get process for given ``identifier`` from storage.
        """
        sane_name = get_sane_name(process_id)
        process = filter(lambda p: self.name_index[p].identifier == sane_name, self.name_index)
        return process
