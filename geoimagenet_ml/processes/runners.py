from geoimagenet_ml.utils import classproperty, null, isnull, str2paths
from geoimagenet_ml.ml.impl import get_test_data_runner, create_batch_patches, retrieve_annotations
from geoimagenet_ml.processes.base import ProcessBase
from geoimagenet_ml.processes.status import map_status, STATUS
from geoimagenet_ml.store.constants import SORT, ORDER
from abc import abstractmethod
from pyramid.settings import asbool
from celery.utils.log import get_task_logger
import os
import six
import numpy
import logging
import multiprocessing
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from geoimagenet_ml.typedefs import (   # noqa: F401
        Any, AnyStr, ErrorType, LevelType, Number, Dict, List, JSON, Optional, UUID, Union
    )
    from geoimagenet_ml.store.datatypes import Job
    # noinspection PyProtectedMember
    from celery import Task                 # noqa: F401
    from pyramid.registry import Registry   # noqa: F401
    from pyramid.request import Request     # noqa: F401


class ProcessRunner(ProcessBase):
    """Runs, monitor and updates the job based on it's execution progression and/or errors."""

    @classproperty
    @abstractmethod
    def inputs(self):
        # type: () -> List[JSON]
        """
        Expected inputs of the class defined as list of ``{"id": '', <other-params>}``.
        Is returned as the `inputs` section of a `ProcessDescription` response.
        """
        raise NotImplementedError

    @classproperty
    @abstractmethod
    def outputs(self):
        # type: () -> List[JSON]
        """
        Expected outputs of the class defined as list of ``{"id": '', <other-params>}``.
        Is returned as the `outputs` section of a `ProcessDescription` response.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Process execution definition."""
        raise NotImplementedError

    def __init__(self, task, registry, request, job_uuid):
        # type: (Task, Registry, Request, UUID) -> None

        # imports to avoid circular references
        from geoimagenet_ml.store.factories import database_factory

        self.task = task
        self.logger = get_task_logger(self.identifier)
        self.request = request
        self.registry = registry
        self.db = database_factory(self.registry)
        self.job = self.setup_job(self.registry, self.request, job_uuid)

    @classmethod
    def check_inputs(cls, inputs):
        # type: (List[Dict[AnyStr, Any]]) -> List[AnyStr]
        """:returns: list of missing input IDs if any. Empty list returned if all required inputs are found."""
        # noinspection PyTypeChecker
        required_input_ids = [i["id"] for i in cls.inputs if i.get("minOccurs", 1) > 1]
        input_ids = [i["id"] for i in inputs]
        return list(set(required_input_ids) - set(input_ids))

    def get_input(self, input_id, default=null, one=False, required=False):
        # type: (AnyStr, Optional[Any], bool, bool) -> Union[List[Any], Any]
        """
        Retrieves a flattened list of input values matching any number of occurrences of ``input_id``.

        If ``input_id`` is not matched:
            - if ``default`` is specified, it is returned instead as a list.
            - if ``default`` is missing, but can be found in the process's ``input`` definition, it is used instead.

        If ``one=True``, the first element of any list from previous step is returned instead of the whole list.

        If ``required=True`` and no value can be found either by process input or default value, raises immediately.
        """
        inputs = [job_input["value"] for job_input in self.job.inputs if job_input["id"] == input_id]
        flattened_inputs = []
        for i in inputs:
            if isinstance(i, list):
                flattened_inputs.extend(i)
            else:
                flattened_inputs.append(i)
        if not flattened_inputs:
            if not isnull(default):
                flattened_inputs = [default]
            else:
                input_spec = [p_input for p_input in self.inputs if p_input["id"] == input_id]  # type: JSON
                if len(input_spec) < 1:
                    raise ValueError("Missing input '{}' not resolvable from process definition.".format(input_id))
                input_spec = input_spec[0]
                if 'default' in input_spec:
                    flattened_inputs = [input_spec['default']]
                else:
                    formats_defaults = [f['default'] for f in input_spec['formats'] if 'default' in f]
                    if formats_defaults:
                        flattened_inputs = [formats_defaults[0]]

        if required and len(flattened_inputs) < 1:
            raise ValueError("Missing input '{}' not matched from literal process input nor defaults.".format(input_id))
        return flattened_inputs[0] if one else flattened_inputs

    def save_results(self, outputs, status_message=None, status_progress=None):
        # type: (List[Dict[AnyStr, JSON]], Optional[AnyStr], Optional[Number]) -> None
        """
        Job output results to be saved to the database on successful process execution.

        Parameter ``outputs`` expects a list of ``{"id": "<output>", "value": <JSON>}`` to save.
        All output values must be JSON serializable.
        """
        optional_outputs = set(o["id"] for o in self.outputs if o.get("minOccurs", 1) == 0)
        required_outputs = set(o["id"] for o in self.outputs) - optional_outputs
        provided_outputs = set(o["id"] for o in outputs)
        missing_outputs = required_outputs - provided_outputs
        unknown_outputs = provided_outputs - (required_outputs | optional_outputs)
        if missing_outputs:
            raise ValueError("Missing required outputs. [{}]".format(missing_outputs))
        if unknown_outputs:
            raise ValueError("Unknown outputs specified. [{}]".format(unknown_outputs))
        self.job.results = outputs
        self.job.status_message = "{} {}.".format(str(self.job), status_message or "Updating output results.")
        self.job.progress = status_progress if status_progress is not None else self.job.progress
        self.db.jobs_store.update_job(self.job, request=self.request)

    def setup_job(self, registry, request, job_uuid):
        # type: (Registry, Request, UUID) -> Job
        job = self.db.jobs_store.fetch_by_uuid(job_uuid, request=request)
        job.task = request.id
        job.tags.append(self.type)
        job = self.db.jobs_store.update_job(job, request=request)
        return job

    def update_job_status(self, status, status_message, status_progress=None, errors=None, level=None):
        # type: (STATUS, AnyStr, Optional[Number], Optional[ErrorType], Optional[LevelType]) -> None
        """Updates the new job status."""
        self.job.status = map_status(status)
        self.job.status_message = "{} {}.".format(str(self.job), status_message)
        self.job.progress = status_progress if status_progress is not None else self.job.progress
        self.job.save_log(logger=self.logger, errors=errors, level=level)
        self.db.jobs_store.update_job(self.job, request=self.request)


class ProcessRunnerModelTester(ProcessRunner):
    """Tests an uploaded model with a defined dataset to retrieve prediction metrics."""

    @classproperty
    def identifier(self):
        return "model-tester"

    @classproperty
    def type(self):
        return "ml"

    @classproperty
    def inputs(self):
        return [
            {
                "id": "dataset",
                "formats": [{"mimeType": "text/plain"}],
                "type": "string",
                "minOccurs": 1,
                "maxOccurs": 1,
            },
            {
                "id": "model",
                "formats": [{"mimeType": "text/plain"}],
                "type": "string",
                "minOccurs": 1,
                "maxOccurs": 1,
            }
        ]

    @classproperty
    def outputs(self):
        return [
            {
                # FIXME: valid types?
                "id": "predictions",
                "type": "float",
                "minOccurs": 1,
                "maxOccurs": "unbounded",
            },
            {
                "id": "classes",
                "type": "string",
                "minOccurs": 0,
                "maxOccurs": "unbounded",
            }
        ]

    def __call__(self, *args, **kwargs):

        class CallbackIterator(object):
            def __init__(self):
                self._iter = 0

            def __call__(self, *it_args, **it_kwargs):
                self._iter = self._iter + 1

            @property
            def iteration(self):
                return self._iter

        def _update_job_eval_progress(_job, _batch_iterator, start_percent=0, final_percent=100):
            # type: (Job, CallbackIterator, Optional[Number], Optional[Number]) -> None
            """
            Updates the job progress based on evaluation progress (after each batch).
            Called using callback of prediction metric.
            """
            metric = test_runner.test_metrics["predictions"]
            total_sample_count = test_runner.test_loader.sample_count
            evaluated_sample_count = len(metric.predictions)  # gradually expanded on each evaluation callback
            batch_count = len(test_runner.test_loader)
            batch_index = _batch_iterator.iteration + 1
            progress = numpy.interp(batch_index / batch_count, [0, 100], [start_percent, final_percent])
            msg = "evaluating... [samples: {}/{}, batches: {}/{}]" \
                .format(evaluated_sample_count, total_sample_count, batch_index, batch_count)
            self.update_job_status(STATUS.RUNNING, msg, progress)

            # update job results and add important fields
            _job.results = [{
                "id": "predictions",
                "value": metric.predictions,
            }]
            if hasattr(test_runner, "class_names"):
                _job.results.insert(0, {
                    "id": "classes",
                    "value": test_runner.class_names
                })
            self.db.jobs_store.update_job(_job)

        try:
            # note:
            #   for dataset loader using multiple worker sub-processes to load samples by batch,
            #   process needs to be non-daemonic to allow pool spawning of child processes since this
            #   task is already a child worker of the main celery app
            # see:
            #   ``geoimagenet_ml.ml.impl.test_loader_from_configs`` for corresponding override
            worker_count = self.registry.settings.get('geoimagenet_ml.ml.data_loader_workers', 0)
            worker_process = multiprocessing.current_process()
            # noinspection PyProtectedMember, PyUnresolvedReferences
            worker_process._config['daemon'] = not bool(worker_count)

            self.update_job_status(STATUS.STARTED, "initiation done", 1)

            self.update_job_status(STATUS.RUNNING, "retrieving dataset definition", 2)
            dataset_uuid = self.get_input("dataset", one=True, required=True)
            dataset = self.db.datasets_store.fetch_by_uuid(dataset_uuid, request=self.request)

            self.update_job_status(STATUS.RUNNING, "loading model from definition", 3)
            model_uuid = self.get_input("model", one=True, required=True)
            model = self.db.models_store.fetch_by_uuid(model_uuid, request=self.request)
            model_config = model.data  # calls loading method, raises failure accordingly

            self.update_job_status(STATUS.RUNNING, "retrieving data loader for model and dataset", 4)
            test_runner = get_test_data_runner(self.job, model_config, model, dataset, self.registry.settings)

            # link the batch iteration with a callback for progress tracking
            batch_iter = CallbackIterator()
            test_runner.eval_iter_callback = batch_iter.__call__

            # link the predictions with a callback for progress update during evaluation
            pred_metric = test_runner.test_metrics["predictions"]
            pred_metric.callback = lambda: \
                _update_job_eval_progress(self.job, batch_iter, start_percent=5, final_percent=98)
            self.update_job_status(STATUS.RUNNING, "starting test data prediction evaluation", 5)
            results = test_runner.eval()
            # TODO:
            #   check if these results correspond to the ones updated gradually with the callback method
            #   (see: _update_job_eval_progress)
            self.save_results([{"id": "predictions", "value": results}], status_progress=99)
            self.update_job_status(STATUS.SUCCEEDED, "processing complete", 100)

        except Exception as task_exc:
            exception_class = "{}.{}".format(type(task_exc).__module__, type(task_exc).__name__)
            err_msg = "{0}: {1}".format(exception_class, str(task_exc))
            message = "failed to run {!s} [{!s}].".format(self.job, err_msg)
            self.update_job_status(STATUS.FAILED, message, errors=task_exc)
        finally:
            self.update_job_status(self.job.status, "done")

        return self.job.status


class ProcessRunnerBatchCreator(ProcessRunner):
    """
    Executes patches creation from a batch of annotated images with GeoJSON metadata.
    Uses with GeoImageNet API requests format for annotation data extraction.
    """

    @classproperty
    def dataset_type(self):
        return "geoimagenet-batch-patches"

    @classproperty
    def identifier(self):
        return "batch-creation"

    @classproperty
    def limit_single_job(self):
        return True

    @classproperty
    def type(self):
        return "ml"

    @classproperty
    def inputs(self):
        return [
            {
                "id": "name",
                "abstract": "Name to be applied to the batch (dataset) to be created.",
                "formats": [{"mimeType": "text/plain"}],
                "type": "string",
                "minOccurs": 1,
                "maxOccurs": 1,
            },
            {
                "id": "geojson_urls",
                "abstract": "List of request URL to GeoJSON annotations with patches geo-locations and metadata. "
                            "Multiple URL are combined into a single batch creation call, it should be used only for "
                            "paging GeoJSON responses. Coordinate reference systems must match between each URL.",
                "formats": [{"mimeType": "text/plain"}],
                "type": "string",
                "minOccurs": 1,
                "maxOccurs": None,
            },
            {
                "id": "crop_fixed_size",
                "abstract": "Overwrite an existing batch if it already exists.",
                "formats": [{"mimeType": "text/plain", "default": None}],
                "type": "integer",
                "minOccurs": 0,
                "maxOccurs": 1,
            },
            {
                "id": "split_ratio",
                "abstract": "Ratio to employ for train/test patch splits of the created batch.",
                "formats": [{"mimeType": "text/plain", "default": 0.90}],
                "type": "float",
                "minOccurs": 0,
                "maxOccurs": 1,
            },
            {
                "id": "incremental_batch",
                "abstract": "Base dataset UUID (string) to use for incremental addition of new patches to the batch. "
                            "If False (boolean), create the new batch from nothing, without using any incremental "
                            "patches from previous datasets. By default, searches and uses the 'latest' batch.",
                "formats": [{"mimeType": "text/plain", "default": None}],
                "types": ["string", "boolean"],
                "minOccurs": 0,
                "maxOccurs": 1,
            },
            {
                "id": 'overwrite',
                "abstract": "Overwrite an existing batch if it already exists.",
                "formats": [{"mimeType": "text/plain", "default": False}],
                "type": "boolean",
                "minOccurs": 0,
                "maxOccurs": 1,
            },
        ]

    @classproperty
    def outputs(self):
        return [
            {
                "id": "dataset",
                "type": "string",
                "abstract": "Dataset UUID corresponding to the generated batch of patches.",
                "minOccurs": 1,
                "maxOccurs": 1,
            }
        ]

    def find_batch(self, batch_id):
        if batch_id is None:
            datasets, count = self.db.datasets_store.find_datasets(
                type=self.dataset_type,
                sort=SORT.FINISHED,
                order=ORDER.DESCENDING,
                status=STATUS.FINISHED,
                limit=1,
            )
            if not datasets:
                self.update_job_status(
                    STATUS.RUNNING,
                    "Could not find latest dataset with [{!s}]. Building from scratch...".format(batch_id),
                    level=logging.WARNING,
                )
                return None
            return datasets[-1]
        elif isinstance(batch_id, six.string_types):
            dataset = self.db.datasets_store.fetch_by_uuid(batch_id)  # raises not found
            if dataset.type != self.dataset_type:
                raise ValueError("Invalid dataset type, found [{}], requires [{}]"
                                 .format(dataset.type, self.dataset_type))
            return dataset
        return None

    def __call__(self, *args, **kwargs):
        dataset = None
        try:
            # imports to avoid circular references
            from geoimagenet_ml.store.datatypes import Dataset

            self.update_job_status(STATUS.STARTED, "initializing configuration settings", 0)
            dataset_update_count = int(self.registry.settings.get("geoimagenet_ml.ml.datasets_update_patch_count", 32))
            self.update_job_status(STATUS.RUNNING, "initiation done", 1)

            self.update_job_status(STATUS.RUNNING, "creating dataset container for patches", 2)
            dataset_name = str(self.get_input("name", one=True))
            dataset = Dataset(name=dataset_name, type=self.dataset_type, status=STATUS.RUNNING)
            dataset_overwrite = asbool(self.get_input("overwrite", one=True))
            if dataset_overwrite and os.path.isdir(dataset.path) and len(os.listdir(dataset.path)):
                self.update_job_status(STATUS.RUNNING, "removing old dataset [{}] as required (override=True)"
                                       .format(dataset.uuid), 2, level=logging.WARNING)
                dataset.reset_path()

            self.update_job_status(STATUS.RUNNING, "obtaining references from process job inputs", 3)
            geojson_urls = self.get_input("geojson_urls", required=True)
            raster_paths = str2paths(self.registry.settings["geoimagenet_ml.ml.source_images_paths"], list_files=True)
            crop_fixed_size = self.get_input("crop_fixed_size", one=True)
            split_ratio = self.get_input("split_ratio", one=True)
            latest_batch = self.get_input("incremental_batch", one=True)

            self.update_job_status(STATUS.RUNNING, "fetching annotations using process job inputs", 4)
            annotations = retrieve_annotations(geojson_urls)

            self.update_job_status(STATUS.RUNNING, "looking for latest batch", 5)
            latest_batch = self.find_batch(latest_batch)

            self.update_job_status(STATUS.RUNNING, "starting batch patches creation", 6)
            dataset = create_batch_patches(annotations, raster_paths, self.db.datasets_store, dataset, latest_batch,
                                           dataset_update_count, crop_fixed_size,
                                           lambda s, p=None: self.update_job_status(STATUS.RUNNING, s, p),
                                           start_percent=7, final_percent=98, train_test_ratio=split_ratio)

            self.save_results([{"id": "dataset", "value": dataset.uuid}], status_progress=99)
            self.update_job_status(STATUS.SUCCEEDED, "processing complete", 100)

        except Exception as task_exc:
            exception_class = "{}.{}".format(type(task_exc).__module__, type(task_exc).__name__)
            err_msg = "{0}: {1}".format(exception_class, str(task_exc))
            message = "failed execution [{!s}].".format(err_msg)
            self.update_job_status(STATUS.FAILED, message, errors=task_exc)
        finally:
            self.update_job_status(self.job.status, "done")

        return self.job.status
