"""
Store adapters to read/write data to from/to mongodb using pymongo.
"""

from geoimagenet_ml.constants import ORDER, SORT, OPERATION
from geoimagenet_ml.store import exceptions as ex
from geoimagenet_ml.store.datatypes import Dataset, Model, Process, Job, Action
from geoimagenet_ml.store.interfaces import DatasetStore, ModelStore, ProcessStore, JobStore, ActionStore
from geoimagenet_ml.status import STATUS, CATEGORY, job_status_categories, map_status
from geoimagenet_ml.processes.types import PROCESS_WPS
from geoimagenet_ml.processes.runners import ProcessRunner
from geoimagenet_ml.utils import isclass, islambda, get_sane_name, is_uuid, get_user_id
from pyramid.request import Request
from pywps import Process as ProcessWPS
from pymongo.errors import DuplicateKeyError
from datetime import datetime
import pymongo
import shutil
import six
import os
import logging
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from geoimagenet_ml.typedefs import Any, AnyProcess, Callable, Dict, Union, Type  # noqa: F401

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

    def save_dataset(self, dataset, overwrite=False, request=None):
        if not isinstance(dataset, Dataset):
            raise ex.DatasetInstanceError("Unsupported dataset type '{}'".format(type(dataset)))
        if not len(dataset.files) or not len(dataset.data):
            raise ex.DatasetInstanceError("Incomplete dataset generation not allowed.")
        try:
            if overwrite:
                result = self.collection.update_one({"uuid": dataset.uuid}, {"$set": dataset.params}, upsert=True)
            else:
                result = self.collection.insert_one(dataset.params)
            if not result.acknowledged:
                raise Exception("Dataset insertion not acknowledged")
        except DuplicateKeyError:
            raise ex.DatasetConflictError("Dataset '{}' conflicts with an existing dataset.".format(dataset.name))
        except Exception as exc:
            LOGGER.exception("Dataset '{}' registration generated error: [{!r}].".format(dataset.name, exc))
            raise ex.DatasetRegistrationError("Dataset '{}' could not be registered.".format(dataset.name))
        return self.fetch_by_uuid(dataset.uuid)

    def delete_dataset(self, dataset_uuid, request=None):
        dataset_uuid = str(dataset_uuid)
        dataset = self.fetch_by_uuid(dataset_uuid)
        dataset_path = dataset.path
        result = self.collection.delete_one({"uuid": dataset_uuid})
        if result.deleted_count != 1:
            return False
        try:
            if os.path.isdir(dataset_path):
                dataset_base_dir = os.path.dirname(dataset_path)
                dataset_paths = [os.path.join(dataset_base_dir, p) for p in os.listdir(dataset_base_dir)
                                 if os.path.join(dataset_base_dir, p).startswith(dataset_path)]
                for path in dataset_paths:
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    elif os.path.isfile(path):
                        os.remove(path)
        except Exception as exc:
            LOGGER.exception("Dataset '{}' deletion generated error: [{!r}].".format(dataset.name, exc))
            raise ex.DatasetInstanceError("Dataset '{}' files could not be deleted.".format(dataset.name))
        return True

    def fetch_by_uuid(self, dataset_uuid, request=None):
        dataset_uuid = str(dataset_uuid)
        dataset = None
        try:
            dataset = self.collection.find_one({"uuid": dataset_uuid})
        except Exception:
            ex.DatasetNotFoundError("Dataset '{}' could not be found.".format(dataset_uuid))
        if not dataset:
            raise ex.DatasetNotFoundError("Dataset '{}' could not be found.".format(dataset_uuid))
        try:
            dataset = Dataset(dataset)
        except Exception:
            raise ex.DatasetInstanceError("Dataset '{}' could not be generated.".format(dataset_uuid))
        return dataset

    # noinspection PyShadowingBuiltins
    def find_datasets(self, name=None, type=None, status=None, sort=None, order=None, limit=None, request=None):
        search_filters = {}

        if isinstance(status, STATUS):
            search_filters["status"] = {"$in": [status.value]}
        elif isinstance(status, CATEGORY):
            search_filters["status"] = {"$in": [s.value for s in job_status_categories[status]]}

        if name is not None:
            search_filters["name"] = name
        if type is not None:
            search_filters["type"] = type

        if sort is None:
            sort = SORT.FINISHED
        if order is None:
            order = ORDER.DESCENDING if sort == SORT.FINISHED or sort == SORT.CREATED else ORDER.ASCENDING
        if not isinstance(sort, SORT):
            raise ex.DatasetNotFoundError("Invalid sorting method: '{}'".format(repr(sort)))
        if not isinstance(order, ORDER):
            raise ex.DatasetNotFoundError("Invalid ordering method: '{}'".format(repr(order)))

        sort_order = pymongo.DESCENDING if order == ORDER.DESCENDING else pymongo.ASCENDING
        sort_criteria = [(sort.value, sort_order)]
        found = self.collection.find(search_filters)
        count = self.collection.count_documents(search_filters)
        items = [Dataset(item) for item in list(found.limit(limit or count).sort(sort_criteria))]
        return items, count

    def list_datasets(self, request=None):
        datasets = []
        try:
            for dataset in self.collection.find().sort("uuid", pymongo.ASCENDING):
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
        if not isinstance(settings, dict) or "geoimagenet_ml.ml.models_path" not in settings:
            raise LookupError("Settings with 'geoimagenet_ml.ml.models_path' is mandatory.")
        self.models_path = settings.get("geoimagenet_ml.ml.models_path")
        os.makedirs(self.models_path, exist_ok=True)

    def save_model(self, model, request=None):
        if not isinstance(model, Model):
            raise ex.ModelInstanceError("Unsupported model type '{}'".format(type(model)))
        try:
            model_path = os.path.join(self.models_path, model.uuid + self._model_ext)
            model.save(model_path)  # apply the database's known storage location and prepare to write to db
            result = self.collection.insert_one(model.params)
            if not result.acknowledged:
                raise Exception("Model insertion not acknowledged")
        except ex.ModelError as exc:
            raise
        except DuplicateKeyError:
            raise ex.ModelConflictError("Model '{!s}' conflicts with an existing model.".format(model))
        except Exception as exc:
            msg_exc = "Model '{!s}' could not be registered. Unhandled error: [{!r}].".format(model, exc)
            LOGGER.exception(msg_exc)
            raise ex.ModelRegistrationError(msg_exc)
        return self.fetch_by_uuid(model.uuid)

    def update_model(self, model, request=None, **fields):
        if not isinstance(model, Model):
            raise ex.ModelInstanceError("Unsupported model type '{}'".format(type(model)))
        if len(fields) == 0:
            raise ex.ModelRegistrationError("No field specified for model update.")
        try:
            model_params = model.params
            for f in fields:
                if f not in model_params:
                    raise ex.ModelRegistrationError("Invalid field '{}' for model update.".format(f))
                # attempt setting field to enforce any validation rule
                model[f] = fields[f]
            result = self.collection.update_one({"uuid": model.uuid}, {"$set": fields})
            if result.modified_count != 1:
                raise ex.ModelRegistrationError("Expected only a single updated model instance. Got {}."
                                                .format(result.modified_count))
        except ex.ModelError:
            raise
        except Exception as exc:
            msg_exc = "Model '{!s}' could not be updated. Unhandled error: [{!r}].".format(model, exc)
            LOGGER.exception(msg_exc)
            raise ex.ModelRegistrationError(msg_exc)
        return self.fetch_by_uuid(model.uuid)

    def delete_model(self, model_uuid, request=None):
        model = self.fetch_by_uuid(model_uuid, request=request)
        try:
            os.remove(model.file)
        except Exception:
            pass
        result = self.collection.delete_one({"uuid": model_uuid})
        return result.deleted_count == 1

    def fetch_by_uuid(self, model_uuid, request=None):
        model_uuid = str(model_uuid)
        model = None
        try:
            model = self.collection.find_one({"uuid": model_uuid})
        except Exception:
            ex.ModelNotFoundError("Model '{}' could not be found.".format(model_uuid))
        if not model:
            raise ex.ModelNotFoundError("Model '{}' could not be found.".format(model_uuid))
        try:
            model = Model(model)
        except Exception:
            raise ex.ModelInstanceError("Model '{}' could not be generated.".format(model_uuid))
        return model

    def list_models(self, request=None):
        models = []
        try:
            for model in self.collection.find().sort("name", pymongo.ASCENDING):
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
        from geoimagenet_ml.api.schemas import ProcessJobsAPI
        super(MongodbProcessStore, self).__init__(collection=collection)
        self.default_host = settings.get("geoimagenet_ml.api.url")
        self.default_endpoint_template = "{host}{path}".format(host=self.default_host, path=ProcessJobsAPI.path)
        if default_processes:
            registered_processes = [process.identifier for process in self.list_processes()]
            for process in default_processes:
                if isinstance(default_processes, dict):
                    process = default_processes[process]
                sane_name = self._get_process_id(process)
                if sane_name not in registered_processes:
                    self._add_process(process)

    @staticmethod
    def _from_runner(runner_process, **extra_params):
        # type: (Union[ProcessRunner, Type[ProcessRunner]], Any) -> Process
        # NOTE:
        #   don't instantiate process because of missing init arguments, use class properties only
        process = {
            "type": runner_process.type,
            "identifier": runner_process.identifier,
            "inputs": runner_process.inputs,
            "outputs": runner_process.outputs,
            "abstract": runner_process.__doc__,
            "package": None,
            "reference": None,
            "limit_single_job": runner_process.limit_single_job,
        }
        process.update(**extra_params)
        return Process(process)

    @staticmethod
    def _from_wps(wps_process, **extra_params):
        # type: (Union[ProcessWPS, Type[ProcessWPS]], Any) -> Process
        # NOTE:
        #   instantiate process if it's only the type to populate json fields from defined metadata in init
        if isclass(wps_process):
            wps_process = wps_process()
        process = wps_process.json
        process_properties = Process.__dict__.keys()
        process_prop_to_rm = [p for p in process if p not in process_properties]
        for prop in process_prop_to_rm:
            process.pop(prop)
        process.update({
            "type": PROCESS_WPS,
            "package": None,
            "reference": None,
        })
        process.update(**extra_params)
        return Process(process)

    def _add_process(self, process):
        if isinstance(process, ProcessWPS) or (isclass(process) and issubclass(process, ProcessWPS)):
            new_process = self._from_wps(process)
        elif isinstance(process, ProcessRunner) or (isclass(process) and issubclass(process, ProcessRunner)):
            new_process = self._from_runner(process)
        else:
            new_process = process
        if not isinstance(new_process, Process):
            raise ex.ProcessInstanceError("Unsupported process type '{}'.".format(type(process)))

        process_name = self._get_process_id(process)
        process_exec = self.default_endpoint_template.replace("{process_uuid}", new_process.uuid)
        try:
            new_process["identifier"] = process_name
            new_process["type"] = self._get_process_type(process)
            new_process["execute_endpoint"] = process_exec
            self.collection.insert_one(new_process.params)
        except Exception as exc:
            raise ex.ProcessRegistrationError(
                "Process '{}' could not be registered. [{!r}]".format(process_name, exc))

    @staticmethod
    def _get_process_field(process, function_dict):
        # type: (AnyProcess, Union[Callable, Dict[AnyProcess, Callable]]) -> Any
        """
        Takes a lambda expression or a dict of process-specific lambda expressions to retrieve a field.
        Validates that the passed process object is one of the supported types.
        """
        # allow using class instances or direct class references
        process_type = process if isclass(process) else type(process)
        if not issubclass(process_type, (Process, ProcessWPS, ProcessRunner)):
            raise ex.ProcessInstanceError("Unsupported process type '{}'".format(process_type))
        if islambda(function_dict):
            return function_dict()
        # fix keys to use base class of derived ones
        if issubclass(process_type, ProcessWPS):
            process_type = ProcessWPS
        elif issubclass(process_type, ProcessRunner):
            process_type = ProcessRunner
        return function_dict[process_type]()

    def _get_process_id(self, process):
        return self._get_process_field(process, lambda: get_sane_name(process.identifier))

    def _get_process_type(self, process):
        return self._get_process_field(process, {
            Process: lambda: process.type,
            ProcessWPS: lambda: PROCESS_WPS,
            ProcessRunner: lambda: process.type,
        }).lower()

    def save_process(self, process, overwrite=False, request=None):
        """
        Stores a WPS process in storage.
        """
        sane_name = self._get_process_id(process)
        if self.collection.count_documents({"identifier": sane_name}) > 0:
            if overwrite:
                self.collection.delete_one({"identifier": sane_name})
            else:
                raise ex.ProcessConflictError("Process '{}' already registered.".format(sane_name))
        self._add_process(process)
        return self.fetch_by_identifier(sane_name)

    def delete_process(self, process_id, request=None):
        """
        Removes process from database.
        """
        sane_name = get_sane_name(process_id)
        result = self.collection.delete_one({"identifier": sane_name})
        return result.deleted_count == 1

    def list_processes(self, request=None):
        """
        Lists all processes in database.
        """
        db_processes = []
        for process in self.collection.find().sort("identifier", pymongo.ASCENDING):
            db_processes.append(Process(process))
        return db_processes

    def fetch_by_uuid(self, process_uuid, request=None):
        """
        Get process for given ``uuid`` from storage.
        """
        sane_name = get_sane_name(process_uuid)
        process = self.collection.find_one({"uuid": sane_name})
        if not process:
            raise ex.ProcessNotFoundError("Process '{}' could not be found.".format(sane_name))
        return Process(process)

    def fetch_by_identifier(self, process_identifier, request=None):
        """
        Get process for given ``identifier`` from storage.
        """
        sane_name = get_sane_name(process_identifier)
        process = self.collection.find_one({"identifier": sane_name})
        if not process:
            raise ex.ProcessNotFoundError("Process '{}' could not be found.".format(sane_name))
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
            raise ex.JobInstanceError("Unsupported job type '{}'".format(type(job)))
        try:
            result = self.collection.insert_one(job.params)
            if not result.acknowledged:
                raise Exception("Job insertion not acknowledged")
        except DuplicateKeyError:
            raise ex.JobConflictError("Job '{}' conflicts with an existing job.".format(job.uuid))
        except Exception as exc:
            LOGGER.exception("Job '{}' registration generated error: [{!r}].".format(job.uuid, exc))
            raise ex.JobRegistrationError("Job '{}' could not be registered.".format(job.uuid))
        return self.fetch_by_uuid(job.uuid)

    def update_job(self, job, request=None):
        try:
            result = self.collection.update_one({"uuid": job.uuid}, {"$set": job.params})
            if result.acknowledged and result.modified_count == 1:
                return self.fetch_by_uuid(job.uuid)
        except Exception as exc:
            raise ex.JobUpdateError("Error occurred during job update: [{}]".format(repr(exc)))
        raise ex.JobUpdateError("Failed to update specified job: '{}'".format(str(job)))

    def delete_job(self, job_uuid, request=None):
        job_uuid = str(job_uuid)
        result = self.collection.delete_one({"uuid": job_uuid})
        return result.deleted_count == 1

    def fetch_by_uuid(self, job_uuid, request=None):
        job_uuid = str(job_uuid)
        job = None
        try:
            job = self.collection.find_one({"uuid": job_uuid})
        except Exception:
            ex.JobNotFoundError("Job '{}' could not be found.".format(job_uuid))
        if not job:
            raise ex.JobNotFoundError("Job '{}' could not be found.".format(job_uuid))
        try:
            job = Job(job)
        except Exception:
            raise ex.JobInstanceError("Job '{}' could not be generated.".format(job_uuid))
        return job

    def list_jobs(self, request=None):
        jobs = []
        try:
            for job in self.collection.find().sort(SORT.UUID.value, pymongo.ASCENDING):
                jobs.append(Job(job))
        except Exception as exc:
            LOGGER.error(str(exc))
            raise ex.JobInstanceError("Job could not be generated.")
        return jobs

    def find_jobs(self, page=0, limit=10, process=None, service=None, tags=None,
                  user=None, status=None, sort=None, order=None, request=None):
        """
        Finds all jobs in mongodb storage matching search filters.
        """
        search_filters = {}

        if tags:
            search_filters["tags"] = {"$all": tags}

        if status:
            if not isinstance(status, list):
                status = [status]
            search_filters["status"] = {"$in": []}
            for s in status:
                if isinstance(s, STATUS):
                    cat_statuses = [s]
                elif isinstance(s, CATEGORY):
                    cat_statuses = job_status_categories[s]
                else:
                    raise ex.JobNotFoundError("Invalid status or category: '{}'".format(repr(s)))
                for cs in cat_statuses:
                    search_status = map_status(cs)
                    # search by name and value for back compatibility
                    search_filters["status"]["$in"].append(search_status.value)
                    search_filters["status"]["$in"].append(search_status.name)
            search_filters["status"]["$in"] = list(set(search_filters["status"]["$in"]))

        if process is not None:
            if not is_uuid(process):
                raise ex.JobNotFoundError("Invalid process UUID: '{!s}'".format(process))
            search_filters["process"] = process

        if service is not None:
            if not is_uuid(service):
                raise ex.JobNotFoundError("Invalid service UUID: '{!s}'".format(service))
            search_filters["service"] = service

        if sort is None:
            sort = SORT.CREATED
        if order is None:
            order = ORDER.DESCENDING if sort == SORT.FINISHED or sort == SORT.CREATED else ORDER.ASCENDING
        if not isinstance(sort, SORT):
            raise ex.JobNotFoundError("Invalid sorting method: '{}'".format(repr(sort)))
        if not isinstance(order, ORDER):
            raise ex.JobNotFoundError("Invalid ordering method: '{}'".format(repr(order)))

        sort_order = pymongo.DESCENDING if order == ORDER.DESCENDING else pymongo.ASCENDING
        sort_criteria = [(sort.value, sort_order)]
        found = self.collection.find(search_filters)
        count = self.collection.count_documents(search_filters)
        items = [Job(item) for item in list(found.skip(page * limit).limit(limit).sort(sort_criteria))]
        return items, count


class MongodbActionStore(ActionStore, MongodbStore):
    # noinspection PyUnusedLocal
    def __init__(self, collection, settings):
        super(MongodbActionStore, self).__init__(collection=collection)

    def save_action(self, type_or_item, operation, request=None):
        try:
            action = Action(type=type_or_item, operation=operation)
        except Exception as exc:
            raise ex.ActionInstanceError("Invalid 'Action' instance raised an error: [{!r}].".format(exc))

        # automatically add additional request details
        if isinstance(request, Request):
            action.user = get_user_id(request)
            action.path = request.path
            action.method = request.method.upper()

        try:
            result = self.collection.insert_one(action.params)
            if not result.acknowledged:
                raise Exception("Action insertion not acknowledged")
            return Action(self.collection.find_one({"uuid": action.uuid}))
        except DuplicateKeyError:
            raise ex.ActionRegistrationError("Action '{}' conflicts with an existing action.".format(action.uuid))
        except Exception as exc:
            LOGGER.exception("Action '{}' registration generated error: [{!r}].".format(action.uuid, exc))
            raise ex.ActionRegistrationError("Job '{}' could not be registered.".format(action.uuid))

    # noinspection PyShadowingBuiltins
    def find_actions(self, item_type=None, item=None, operation=None, user=None, start=None, end=None,
                     sort=None, order=None, page=None, limit=None):
        search_filters = {}
        if item_type:
            if isclass(item_type):
                item_type = item_type.__name__
            elif isclass(type(item_type)):
                item_type = item_type.__name__
            if not isinstance(item_type, six.string_types):
                raise TypeError("Invalid 'type' to search for 'Action'.")
            search_filters["type"] = item_type
        if item:
            if not is_uuid(item):
                raise TypeError("Invalid 'item' to search for 'Action'.")
            search_filters["item"] = str(item)

        if operation in OPERATION:
            search_filters["operation"] = operation.name

        if isinstance(user, int):
            search_filters["user"] = user

        if isinstance(start, datetime):
            search_filters["created"] = {"$gte": start}
        if isinstance(end, datetime):
            if "created" in search_filters:
                if start >= end:
                    raise ValueError("Invalid start/end datetimes.")
                search_filters["created"]["$lte"] = end
            else:
                search_filters["created"] = {"$lte": end}

        if sort is None:
            sort = SORT.CREATED
        if order is None:
            order = ORDER.DESCENDING if sort == SORT.CREATED else ORDER.ASCENDING
        if not isinstance(sort, SORT):
            raise ValueError("Invalid sorting method: '{}'".format(repr(sort)))
        if not isinstance(order, ORDER):
            raise ValueError("Invalid ordering method: '{}'".format(repr(order)))

        sort_order = pymongo.DESCENDING if order == ORDER.DESCENDING else pymongo.ASCENDING
        sort_criteria = [(sort.value, sort_order)]
        found = self.collection.find(search_filters)
        count = self.collection.count_documents(search_filters)
        if isinstance(limit, int):
            if isinstance(page, int):
                found = found.skip(page * limit)
            found = found.limit(limit)
        items = [Action(item) for item in list(found.sort(sort_criteria))]
        return items, count
