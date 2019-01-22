"""
Store adapters to read/write data to from/to mongodb using pymongo.
"""

from src.store import exceptions as ex
from src.store.datatypes import Dataset, Model, Process, Job
from src.store.interfaces import DatasetStore, ModelStore, ProcessStore, JobStore
from src.api.utils import islambda, get_sane_name
from pywps import Process as ProcessWPS
from pymongo.errors import DuplicateKeyError
import pymongo
import os
import io
import logging
LOGGER = logging.getLogger(__name__)


class MongodbStore(object):
    """
    Base class extended by all concrete store adapters.
    """

    def __init__(self, collection):
        self.collection = collection


# noinspection PyBroadException
class MongodbDatasetStore(DatasetStore, MongodbStore):
    """
    Registry for datasets. Uses mongodb to store attributes.
    """

    # noinspection PyUnusedLocal
    def __init__(self, collection, settings):
        super(MongodbDatasetStore, self).__init__(collection=collection)

    def save_dataset(self, dataset, request=None):
        if not isinstance(dataset, Dataset):
            raise ex.DatasetInstanceError("Unsupported dataset type `{}`".format(type(dataset)))
        try:
            result = self.collection.insert_one(dataset)
            if not result.acknowledged:
                raise Exception()
        except DuplicateKeyError:
            raise ex.DatasetConflictError("Dataset `{}` conflicts with an existing dataset.".format(dataset.name))
        except Exception as exc:
            LOGGER.exception("Dataset `{}` registration generated error: [{!r}].".format(dataset.name, exc))
            raise ex.DatasetRegistrationError("Dataset `{}` could not be registered.".format(dataset.name))
        return self.fetch_by_uuid(dataset.uuid)

    def delete_dataset(self, dataset_uuid, request=None):
        dataset_uuid = str(dataset_uuid)
        result = self.collection.delete_one({'uuid': dataset_uuid})
        return result.deleted_count == 1

    def fetch_by_uuid(self, dataset_uuid, request=None):
        dataset_uuid = str(dataset_uuid)
        dataset = None
        try:
            dataset = self.collection.find_one({'uuid': dataset_uuid})
        except Exception:
            ex.DatasetNotFoundError("Dataset `{}` could not be found.".format(dataset_uuid))
        if not dataset:
            raise ex.DatasetNotFoundError("Dataset `{}` could not be found.".format(dataset_uuid))
        try:
            dataset = Dataset(dataset)
        except Exception:
            raise ex.DatasetInstanceError("Dataset `{}` could not be generated.".format(dataset_uuid))
        return dataset

    def list_datasets(self, request=None):
        datasets = []
        try:
            for dataset in self.collection.find().sort('name', pymongo.ASCENDING):
                datasets.append(Dataset(dataset))
        except Exception:
            raise ex.DatasetInstanceError("Dataset could not be generated.")
        return datasets


# noinspection PyBroadException
class MongodbModelStore(ModelStore, MongodbStore):
    """
    Registry for models. Uses mongodb to store attributes.
    """
    _model_ext = ".ckpt"

    # noinspection PyUnusedLocal
    def __init__(self, collection, settings):
        super(MongodbModelStore, self).__init__(collection=collection)
        if not isinstance(settings, dict) or 'src.api.models_path' not in settings:
            raise LookupError("Settings with 'src.api.models_path' is mandatory.")
        self.models_path = settings.get('src.api.models_path')
        os.makedirs(self.models_path, exist_ok=True)

    def save_model(self, model, data=None, request=None):
        if not isinstance(model, Model):
            raise ex.ModelInstanceError("Unsupported model type `{}`".format(type(model)))
        data = data or model.data
        try:
            if isinstance(data, io.BufferedIOBase):
                # transfer loaded data buffer to storage file
                model_path = os.path.join(self.models_path, model.uuid + self._model_ext)
                with open(model_path, 'wb') as model_file:
                    data.seek(0)
                    model_file.write(data.read())
                    data.close()
                model['data'] = None        # force reload from stored file when calling `model.data` retrieved from db
                model['file'] = model_path
            elif isinstance(data, dict):
                model['data'] = data
                model['file'] = None
            else:
                raise ex.ModelInstanceError("Model data is expected to be a buffer or dict, got {!r}.".format(data))
            result = self.collection.insert_one(model)
            if not result.acknowledged:
                raise Exception()
        except DuplicateKeyError:
            raise ex.ModelConflictError("Model `{}` conflicts with an existing model.".format(model.name))
        except Exception as exc:
            LOGGER.exception("Model `{}` registration generated error: [{!r}].".format(model.name, exc))
            raise ex.ModelRegistrationError("Model `{}` could not be registered.".format(model.name))
        return self.fetch_by_uuid(model.uuid)

    def delete_model(self, model_uuid, request=None):
        model = self.fetch_by_uuid(model_uuid, request=request)
        try:
            os.remove(model.file)
        except Exception:
            pass
        result = self.collection.delete_one({'uuid': model_uuid})
        return result.deleted_count == 1

    def fetch_by_uuid(self, model_uuid, request=None):
        model_uuid = str(model_uuid)
        model = None
        try:
            model = self.collection.find_one({'uuid': model_uuid})
        except Exception:
            ex.ModelNotFoundError("Model `{}` could not be found.".format(model_uuid))
        if not model:
            raise ex.ModelNotFoundError("Model `{}` could not be found.".format(model_uuid))
        try:
            model = Model(model)
        except Exception:
            raise ex.ModelInstanceError("Model `{}` could not be generated.".format(model_uuid))
        return model

    def list_models(self, request=None):
        models = []
        try:
            for model in self.collection.find().sort('name', pymongo.ASCENDING):
                models.append(Model(model))
        except Exception:
            raise ex.ModelInstanceError("Model could not be generated.")
        return models

    def clear_models(self, request=None):
        success = all([self.delete_model(model.uuid, request=request) for model in self.list_models(request=request)])
        return self.collection.count_documents({}) == 0 and success


# noinspection PyBroadException
class MongodbProcessStore(ProcessStore, MongodbStore):
    """
    Registry for WPS processes. Uses mongodb to store processes and attributes.
    """

    def __init__(self, collection, settings, default_processes=None):
        from src.api.rest_api.schemas import ProcessJobsAPI
        super(MongodbProcessStore, self).__init__(collection=collection)
        self.default_host = settings.get('src.api.url')
        self.default_wps_endpoint_template = '{host}{path}'.format(host=self.default_host, path=ProcessJobsAPI.path)
        if default_processes:
            registered_processes = [process.identifier for process in self.list_processes()]
            for process in default_processes:
                sane_name = self._get_process_id(process)
                if sane_name not in registered_processes:
                    self._add_process(process)

    def _add_process(self, process):
        if isinstance(process, ProcessWPS):
            new_process = Process.from_wps(process)
        else:
            new_process = process
        if not isinstance(new_process, Process):
            raise ex.ProcessInstanceError("Unsupported process type `{}`.".format(type(process)))

        process_name = self._get_process_id(process)
        process_exec = self.default_wps_endpoint_template.replace('{process_uuid}', new_process.uuid)
        try:
            new_process['identifier'] = process_name
            new_process['type'] = self._get_process_type(process)
            new_process['executeEndpoint'] = process_exec
            self.collection.insert_one(new_process)
        except Exception as exc:
            raise ex.ProcessRegistrationError(
                "Process `{}` could not be registered. [{!r}]".format(process_name, exc))

    @staticmethod
    def _get_process_field(process, function_dict):
        """
        Takes a lambda expression or a dict of process-specific lambda expressions to retrieve a field.
        Validates that the passed process object is one of the supported types.
        """
        if isinstance(process, Process):
            if islambda(function_dict):
                return function_dict()
            return function_dict[Process]()
        elif isinstance(process, ProcessWPS):
            if islambda(function_dict):
                return function_dict()
            return function_dict[ProcessWPS]()
        else:
            raise ex.ProcessInstanceError("Unsupported process type `{}`".format(type(process)))

    def _get_process_id(self, process):
        return self._get_process_field(process, lambda: get_sane_name(process.identifier))

    def _get_process_type(self, process):
        return self._get_process_field(process, {Process: lambda: process.type, ProcessWPS: lambda: 'wps'}).lower()

    def save_process(self, process, overwrite=False, request=None):
        """
        Stores a WPS process in storage.
        """
        sane_name = self._get_process_id(process)
        if self.collection.count({'identifier': sane_name}) > 0:
            if overwrite:
                self.collection.delete_one({'identifier': sane_name})
            else:
                raise ex.ProcessConflictError("Process `{}` already registered.".format(sane_name))
        self._add_process(process)
        return self.fetch_by_identifier(sane_name)

    def delete_process(self, process_id, request=None):
        """
        Removes process from database.
        """
        sane_name = get_sane_name(process_id)
        result = self.collection.delete_one({'identifier': sane_name})
        return result.deleted_count == 1

    def list_processes(self, request=None):
        """
        Lists all processes in database.
        """
        db_processes = []
        for process in self.collection.find().sort('identifier', pymongo.ASCENDING):
            db_processes.append(Process(process))
        return db_processes

    def fetch_by_uuid(self, process_uuid, request=None):
        """
        Get process for given ``uuid`` from storage.
        """
        sane_name = get_sane_name(process_uuid)
        process = self.collection.find_one({'uuid': sane_name})
        if not process:
            raise ex.ProcessNotFoundError("Process `{}` could not be found.".format(sane_name))
        return Process(process)

    def fetch_by_identifier(self, process_identifier, request=None):
        """
        Get process for given ``identifier`` from storage.
        """
        sane_name = get_sane_name(process_identifier)
        process = self.collection.find_one({'identifier': sane_name})
        if not process:
            raise ex.ProcessNotFoundError("Process `{}` could not be found.".format(sane_name))
        return Process(process)


# noinspection PyBroadException
class MongodbJobStore(JobStore, MongodbStore):
    """
    Registry for jobs. Uses mongodb to store attributes.
    """

    # noinspection PyUnusedLocal
    def __init__(self, collection, settings):
        super(MongodbJobStore, self).__init__(collection=collection)

    def save_job(self, job, request=None):
        if not isinstance(job, Job):
            raise ex.JobInstanceError("Unsupported job type `{}`".format(type(job)))
        try:
            result = self.collection.insert_one(job)
            if not result.acknowledged:
                raise Exception()
        except DuplicateKeyError:
            raise ex.JobConflictError("Job `{}` conflicts with an existing job.".format(job.uuid))
        except Exception as exc:
            LOGGER.exception("Job `{}` registration generated error: [{!r}].".format(job.uuid, exc))
            raise ex.JobRegistrationError("Job `{}` could not be registered.".format(job.uuid))
        return self.fetch_by_uuid(job.uuid)

    def update_job(self, job, request=None):
        try:
            result = self.collection.update_one({'uuid': job.uuid}, {'$set': job.params})
            if result.acknowledged and result.modified_count == 1:
                return self.fetch_by_uuid(job.uuid)
        except Exception as exc:
            raise ex.JobUpdateError("Error occurred during job update: [{}]".format(repr(exc)))
        raise ex.JobUpdateError("Failed to update specified job: `{}`".format(str(job)))

    def delete_job(self, job_uuid, request=None):
        job_uuid = str(job_uuid)
        result = self.collection.delete_one({'uuid': job_uuid})
        return result.deleted_count == 1

    def fetch_by_uuid(self, job_uuid, request=None):
        job_uuid = str(job_uuid)
        job = None
        try:
            job = self.collection.find_one({'uuid': job_uuid})
        except Exception:
            ex.JobNotFoundError("Job `{}` could not be found.".format(job_uuid))
        if not job:
            raise ex.JobNotFoundError("Job `{}` could not be found.".format(job_uuid))
        try:
            job = Job(job)
        except Exception:
            raise ex.JobInstanceError("Job `{}` could not be generated.".format(job_uuid))
        return job

    def list_jobs(self, request=None):
        jobs = []
        try:
            for job in self.collection.find().sort('name', pymongo.ASCENDING):
                jobs.append(Job(job))
        except Exception:
            raise ex.JobInstanceError("Job could not be generated.")
        return jobs
