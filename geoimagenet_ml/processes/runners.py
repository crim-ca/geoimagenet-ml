from geoimagenet_ml.constants import SORT, ORDER
from geoimagenet_ml.utils import classproperty, null, isnull, str2paths
from geoimagenet_ml.ml.impl import (
    DATASET_CROP_MODES,
    get_test_data_runner,
    get_test_results,
    create_batch_patches,
    retrieve_annotations,
    retrieve_taxonomy,
)
from geoimagenet_ml.processes.base import ProcessBase
from geoimagenet_ml.status import map_status, STATUS
from abc import abstractmethod
from pyramid.settings import asbool
from celery.utils.log import get_task_logger
import os
import six
import logging
import multiprocessing
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from geoimagenet_ml.typedefs import (   # noqa: F401
        Any, AnyStr, AnyUUID, ErrorType, LevelType, Number, Dict, List, JSON, Optional, Union
    )
    from geoimagenet_ml.store.datatypes import Job  # noqa: F401
    # noinspection PyProtectedMember
    from celery import Task                 # noqa: F401
    from pyramid.registry import Registry   # noqa: F401
    from pyramid.request import Request     # noqa: F401
    import thelper                          # noqa: F401


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
        # type: (Task, Registry, Request, AnyUUID) -> None

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
                input_spec = [p_input for p_input in self.inputs if p_input["id"] == input_id]
                if len(input_spec) < 1:
                    raise ValueError("Missing input '{}' not resolvable from process definition.".format(input_id))
                input_spec = input_spec[0]
                if "default" in input_spec:
                    flattened_inputs = [input_spec["default"]]
                else:
                    formats_defaults = [f["default"] for f in input_spec["formats"] if "default" in f]
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
        # type: (Registry, Request, AnyUUID) -> Job
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
        self.job = self.db.jobs_store.update_job(self.job, request=self.request)


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
                "type": "string",
                "minOccurs": 1,
                "maxOccurs": 1,
            },
            {
                "id": "model",
                "type": "string",
                "minOccurs": 1,
                "maxOccurs": 1,
            }
        ]

    @classproperty
    def outputs(self):
        return [
            {
                "id": "summary",
                "type": ["string", "integer", "float", None],
                "abstract": "Additional prediction results information.",
                "minOccurs": 1,
                "maxOccurs": "unbounded",
            },
            {
                "id": "predictions",
                "type": "float",
                "abstract": "List of raw predictions produced by the model, for each test sample from the input "
                            "dataset (see 'samples'). Predictions per sample correspond to classes supported "
                            "by the model (see 'classes') that where matched against the corresponding input dataset "
                            "class IDs (see 'mapping'). Format of predictions depend on the model task type.",
                "minOccurs": 1,
                "maxOccurs": "unbounded",
            },
            {
                "id": "metrics",
                "type": "float",
                "abstract": "List of metrics evaluated across all retained dataset samples and class prediction "
                            "scores by the model.",
                "minOccurs": 1,
                "maxOccurs": "unbounded",
            },
            {
                "id": "classes",
                "type": ["string", "integer", None],
                "abstract": "Class ID ordering of the model (classes represented by corresponding output indices).",
                "minOccurs": 0,
                "maxOccurs": "unbounded",
            },
            {
                "id": "mapping",
                "type": ["string", "integer", None],
                "abstract": "Class ID mapping employed between the input dataset classes and the supported classes "
                            "by the model.",
                "minOccurs": 1,
                "maxOccurs": "unbounded",
            },
            {
                "id": "samples",
                "type": ["string", "integer"],
                "abstract": "List of samples retained from the input dataset for the model evaluation "
                            "(samples for which their class ID was matched with one of the supported model "
                            "class ID or their parent category).",
                "minOccurs": 1,
                "maxOccurs": "unbounded",
            },
        ]

    def __call__(self, *args, **kwargs):
        try:
            # note:
            #   for dataset loader using multiple worker sub-processes to load samples by batch,
            #   process needs to be non-daemonic to allow pool spawning of child processes since this
            #   task is already a child worker of the main celery app
            # see:
            #   ``geoimagenet_ml.ml.impl.test_loader_from_configs`` for corresponding override
            worker_count = self.registry.settings.get("geoimagenet_ml.ml.data_loader_workers", 0)
            worker_process = multiprocessing.current_process()
            # noinspection PyProtectedMember, PyUnresolvedReferences
            worker_process._config["daemon"] = not bool(worker_count)

            self.update_job_status(STATUS.STARTED, "initiation done", 1)

            self.update_job_status(STATUS.RUNNING, "retrieving dataset definition", 2)
            dataset_uuid = self.get_input("dataset", one=True, required=True)
            dataset = self.db.datasets_store.fetch_by_uuid(dataset_uuid, request=self.request)

            self.update_job_status(STATUS.RUNNING, "loading model from definition", 3)
            model_uuid = self.get_input("model", one=True, required=True)
            model = self.db.models_store.fetch_by_uuid(model_uuid, request=self.request)
            model_config = model.data  # calls loading method, raises failure accordingly

            self.update_job_status(STATUS.RUNNING, "retrieving data loader for model and dataset", 4)
            test_runner = get_test_data_runner(self, model_config, model, dataset, start_percent=5, final_percent=95)

            self.update_job_status(STATUS.RUNNING, "starting test data prediction evaluation", 5)
            # results obtained here are only for single-value monitor 'Metric' instances, loggers are dumped to file
            results = test_runner.eval()

            self.update_job_status(STATUS.RUNNING, "retrieving complete test data prediction results", 97)
            test_results = get_test_results(test_runner, results)

            self.update_job_status(STATUS.RUNNING, "preparing jobs outputs with prediction results", 98)
            outputs = [{"id": k, "value": v} for k, v in test_results.items()]

            self.save_results(outputs, status_progress=99)
            self.update_job_status(STATUS.SUCCEEDED, "processing complete", 100)

        except Exception as task_exc:
            exception_class = "{}.{}".format(type(task_exc).__module__, type(task_exc).__name__)
            err_msg = "{0}: {1}".format(exception_class, str(task_exc))
            message = "failed to run {!s} [{!s}].".format(self.job, err_msg)
            self.update_job_status(STATUS.FAILED, message, errors=task_exc)
        finally:
            self.update_job_status(self.job.status, "done")


class ProcessRunnerBatchCreator(ProcessRunner):
    """
    Executes patches creation from a batch of annotated images with GeoJSON metadata.
    Uses GeoImageNet API requests format for annotation data extraction.

    .. seealso::

        - `GeoImageNet API` source: https://www.crim.ca/stash/projects/GEO/repos/geoimagenet_api
        - `GeoImageNet API` swagger: https://geoimagenet.crim.ca/api/v1/docs
        - `GeoImageNet API` annotation example: https://geoimagenetdev.crim.ca/api/v1/batches/annotations
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
                "type": "string",
                "minOccurs": 1,
                "maxOccurs": 1,
            },
            {
                "id": "geojson_urls",
                "abstract": "List of request URL to GeoJSON annotations with patches geo-locations and metadata. "
                            "Multiple URL are combined into a single batch creation call, it should be used only for "
                            "paging GeoJSON responses. Coordinate reference systems must match between each URL.",
                "formats": [{"mimeType": "application/json"}],
                "type": "string",
                "minOccurs": 1,
                "maxOccurs": "unbounded",
            },
            {
                "id": "taxonomy_url",
                "abstract": "Request URL where to retrieve taxonomy classes metadata. "
                            "Class IDs should match result values found from specified 'geojson_url' input. "
                            "Parent/child class ID hierarchy should be available to support category grouping.",
                "formats": [{"mimeType": "application/json"}],
                "type": "string",
                "minOccurs": 1,
                "maxOccurs": 1,
            },
            {
                "id": "crop_fixed_size",
                "abstract": "Request creation of patch crops of fixed dimension (in meters) instead of original size. "
                            "These patch crops will be stored as 'fixed' type within the generated dataset. "
                            "Note that some annotation will not be fully contained within generated crops if the "
                            "provided dimension is smaller than the original feature square bounding box. "
                            "Features that are larger than the requested dimension will be cropped at their centroid "
                            "to generate the patch of fixed size. Smaller features will be padded with 'nodata'.",
                "type": ["integer", "float"],
                "default": None,
                "minOccurs": 0,
                "maxOccurs": 1,
            },
            {
                "id": "crop_mode",
                "abstract": "Method for handling the generation of bounding boxes of patch crops around features. "
                            "When 'crop_fixed_size' is provided, this parameter is ignored in favor of the requested "
                            "size. Otherwise, crops are processed according to the provided value. "
                            "When ('extend' or >0), the bounding box of the feature will be extended "
                            "along the smallest of the two dimensions with 'nodata' in order to generate the minimal "
                            "square crops that completely contains the original feature. When ('raw' or =0), the "
                            "minimal and original rectangle bounding box that contains the feature will be returned as "
                            "is without any padding. When ('reduce' or <0), the largest of the two dimensions of the "
                            "bounding box containing the feature will be reduced to match the smallest one in order to "
                            "from a square crop. No padding occurs in this case, but there is partial lost of feature "
                            "data due to the clipping.",
                "type": ["integer", "string"],
                "default": 1,
                "minOccurs": 0,
                "maxOccurs": 1,
                "allowedValues": list(DATASET_CROP_MODES)
            },
            {
                "id": "split_ratio",
                "abstract": "Ratio to employ for train/test patch splits of the created batch.",
                "type": "float",
                "default": 0.90,
                "minOccurs": 0,
                "maxOccurs": 1,
            },
            {
                "id": "incremental_batch",
                "abstract": "Base dataset UUID (string) to use for incremental addition of new patches to the batch. "
                            "If False (boolean), create the new batch from nothing, without using any incremental "
                            "patches from previous datasets. By default, searches and uses the 'latest' batch.",
                "type": ["string", "boolean"],
                "default": None,
                "minOccurs": 0,
                "maxOccurs": 1,
            },
            {
                "id": "overwrite",
                "abstract": "Overwrite an existing batch if it already exists.",
                "type": "boolean",
                "default": False,
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
            taxonomy_url = self.get_input("taxonomy_url", required=True, one=True)
            raster_paths = str2paths(self.registry.settings["geoimagenet_ml.ml.source_images_paths"],
                                     list_files=True, allowed_extensions=[".tif"])
            crop_fixed_size = self.get_input("crop_fixed_size", one=True)
            crop_mode = self.get_input("crop_mode", one=True)
            split_ratio = self.get_input("split_ratio", one=True)
            latest_batch = self.get_input("incremental_batch", one=True)

            self.update_job_status(STATUS.RUNNING, "fetching annotations using process job inputs", 4)
            annotations = retrieve_annotations(geojson_urls)

            self.update_job_status(STATUS.RUNNING, "fetching taxonomy using process job inputs", 5)
            taxonomy = retrieve_taxonomy(taxonomy_url)

            self.update_job_status(STATUS.RUNNING, "looking for latest batch", 6)
            latest_batch = self.find_batch(latest_batch)

            self.update_job_status(STATUS.RUNNING, "starting batch patches creation", 7)
            dataset = create_batch_patches(annotations, taxonomy, raster_paths, self.db.datasets_store,
                                           dataset, latest_batch, dataset_update_count, crop_fixed_size, crop_mode,
                                           lambda s, p=None: self.update_job_status(STATUS.RUNNING, s, p),
                                           start_percent=8, final_percent=98, train_test_ratio=split_ratio)

            self.save_results([{"id": "dataset", "value": dataset.uuid}], status_progress=99)
            self.update_job_status(STATUS.SUCCEEDED, "processing complete", 100)

        except Exception as task_exc:
            exception_class = "{}.{}".format(type(task_exc).__module__, type(task_exc).__name__)
            err_msg = "{0}: {1}".format(exception_class, str(task_exc))
            message = "failed execution [{!s}].".format(err_msg)
            self.update_job_status(STATUS.FAILED, message, errors=task_exc)
        finally:
            self.update_job_status(self.job.status, "done")
