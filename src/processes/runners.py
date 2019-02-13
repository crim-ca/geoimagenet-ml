from geoimagenet_ml.utils import classproperty
from geoimagenet_ml.ml.impl import get_test_data_runner
from geoimagenet_ml.processes.base import ProcessBase
from geoimagenet_ml.processes.status import (
    map_status, STATUS_SUCCEEDED, STATUS_FAILED, STATUS_STARTED, STATUS_RUNNING
)
from abc import abstractmethod
from celery.utils.log import get_task_logger
import os
import numpy
import multiprocessing
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from geoimagenet_ml.typedefs import Any, AnyStr, Error, Number, Dict, List, Optional, UUID  # noqa: F401
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
        # type: () -> List[Dict[AnyStr, Any]]
        """
        Expected inputs of the class defined as list of `{'id': '', <other-params>}`.
        Is returned as the 'inputs' of a Process description.
        """
        raise NotImplementedError

    @classmethod
    def check_inputs(cls, inputs):
        # type: (List[Dict[AnyStr, Any]]) -> List[AnyStr]
        """:returns: list of missing input IDs if any. Empty list returned if all required inputs are found."""
        # noinspection PyTypeChecker
        required_input_ids = [i['id'] for i in cls.inputs]
        input_ids = [i['id'] for i in inputs]
        return list(set(required_input_ids) - set(input_ids))

    @classmethod
    def get_input(cls, input_id):
        # type: (AnyStr) -> List[Any]
        return [job_input['value'] for job_input in self.job.inputs if job_input['id'] == input_id]

    def __init__(self, task, registry, request, job_uuid):
        # type: (Task, Registry, Request, UUID) -> None
        from geoimagenet_ml.store.factories import database_factory
        self.task = task
        self.logger = get_task_logger(self.identifier)
        self.request = request
        self.registry = registry
        self.jobs_store = database_factory(self.registry).jobs_store
        self.job = self.setup_job(self.registry, self.request, job_uuid)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def setup_job(self, registry, request, job_uuid):
        # type: (Registry, Request, UUID) -> Job
        job = self.jobs_store.fetch_by_uuid(job_uuid, request=request)
        job.task_uuid = request.id
        job.tags.append(self.type)
        job = self.jobs_store.update_job(job, request=request)
        return job

    def update_job_status(self, status, status_message, status_progress=None, errors=None):
        # type: (AnyStr, AnyStr, Optional[Number], Optional[Error]) -> None
        """Updates the new job status."""
        self.job.status = map_status(status)
        self.job.status_message = "{} {}.".format(str(self.job), status_message)
        self.job.progress = status_progress if status_progress is not None else self.job.progress
        self.job.save_log(logger=self.logger, errors=errors)
        self.jobs_store.update_job(self.job, request=self.request)


class ProcessRunnerModelTester(ProcessRunner):
    """Tests an uploaded model with a defined dataset to retrieve prediction metrics."""

    @classproperty
    def identifier(self):
        return 'model-tester'

    @classproperty
    def type(self):
        return 'ml'

    # TODO: make this cleaner
    @classproperty
    def inputs(self):
        return [
            {
                'id': 'dataset',
                'formats': [{'mimeType': 'text/plain', 'default': False}],
                'minOccurs': 1,
                'maxOccurs': 1,
            },
            {
                'id': 'model',
                'formats': [{'mimeType': 'text/plain', 'default': False}],
                'minOccurs': 1,
                'maxOccurs': 1,
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
            metric = test_runner.test_metrics['predictions']
            total_sample_count = test_runner.test_loader.sample_count
            evaluated_sample_count = len(metric.predictions)  # gradually expanded on each evaluation callback
            batch_count = len(test_runner.test_loader)
            batch_index = _batch_iterator.iteration + 1
            progress = numpy.interp(batch_index / batch_count, [0, 100], [start_percent, final_percent])
            msg = "evaluating... [samples: {}/{}, batches: {}/{}]" \
                .format(evaluated_sample_count, total_sample_count, batch_index, batch_count)
            self.update_job_status(STATUS_RUNNING, msg, progress)

            # update job results and add important fields
            _job.results = [{
                'identifier': 'predictions',
                'value': metric.predictions,
            }]
            if hasattr(test_runner, 'class_names'):
                _job.results.insert(0, {
                    'identifier': 'classes',
                    'value': test_runner.class_names
                })
            self.jobs_store.update_job(_job)

        try:
            from geoimagenet_ml.store.factories import database_factory

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

            self.update_job_status(STATUS_STARTED, "initiation done", 1)

            self.update_job_status(STATUS_RUNNING, "retrieving dataset definition", 2)
            dataset_uuid = self.get_input('dataset')[0]
            dataset = database_factory(self.registry).datasets_store.fetch_by_uuid(dataset_uuid, request=self.request)

            self.update_job_status(STATUS_RUNNING, "loading model from definition", 3)
            model_uuid = self.get_input('model')[0]
            model = database_factory(self.registry).models_store.fetch_by_uuid(model_uuid, request=self.request)
            model_config = model.data  # calls loading method, raises failure accordingly

            self.update_job_status(STATUS_RUNNING, "retrieving data loader for model and dataset", 4)
            test_runner = get_test_data_runner(self.job, model_config, model, dataset, self.registry.settings)

            # link the batch iteration with a callback for progress tracking
            batch_iter = CallbackIterator()
            test_runner.eval_iter_callback = batch_iter.__call__

            # link the predictions with a callback for progress update during evaluation
            pred_metric = test_runner.test_metrics['predictions']
            pred_metric.callback = lambda: _update_job_eval_progress(self.job, batch_iter,
                                                                     start_percent=5, final_percent=99)
            self.update_job_status(STATUS_RUNNING, "starting test data prediction evaluation", 5)
            test_runner.eval()

            self.update_job_status(STATUS_SUCCEEDED, "processing complete", 100)

        except Exception as task_exc:
            exception_class = "{}.{}".format(type(task_exc).__module__, type(task_exc).__name__)
            err_msg = "{0}: {1}".format(exception_class, str(task_exc))
            message = "failed to run {!s} [{!s}].".format(self.job, err_msg)
            self.update_job_status(STATUS_FAILED, message, errors=task_exc)
        finally:
            self.update_job_status(self.job.status, "done")

        return self.job.status


class ProcessRunnerBatchCreator(ProcessRunner):
    """Executes patches creation from a batch of annotated shapefiles and images."""

    @classproperty
    def identifier(self):
        return 'batch-creation'

    @classproperty
    def type(self):
        return 'ml'

    # TODO: make this cleaner
    @classproperty
    def inputs(self):
        return [
            {
                'id': 'name',
                'abstract': 'Name to be applied to the batch (dataset) to be created.',
                'formats': [{'mimeType': 'text/plain', 'default': False}],
                'minOccurs': 1,
                'maxOccurs': 1,
            },
            {
                'id': 'shapefiles',
                'abstract': 'List of shapefiles with annotated patches geo-locations.',
                'minOccurs': 1,
            },
            {
                'id': 'images',
                'abstract': 'List of images where to extract patches from.',
                'minOccurs': 1,
            }
        ]

    def __call__(self, *args, **kwargs):
        try:
            from geoimagenet_ml.store.datatypes import Dataset
            from geoimagenet_ml.store.factories import database_factory
            self.update_job_status(STATUS_STARTED, "initiation done", 1)

            self.update_job_status(STATUS_RUNNING, "creating dataset representation of patches", 2)
            dataset_name = str(self.get_input('name')[0])
            dataset_root = str(self.registry.settings['geoimagenet_ml.api.datasets_path'])
            if not os.path.isdir(dataset_root):
                raise RuntimeError("cannot find datasets root path")
            if not len(dataset_name):
                raise RuntimeError("invalid batch dataset name")
            dataset_path = os.path.join(dataset_root, dataset_name)
            os.makedirs(dataset_path, exist_ok=False, mode=0o644)
            dataset = Dataset(name=dataset_name, path=dataset_path, type='patches')

            self.update_job_status(STATUS_RUNNING, "obtaining references from process job inputs", 2)
            shapefiles = list().extend(s for s in self.get_input('shapefiles'))
            images = list().extend(i for i in self.get_input('images'))

            self.update_job_status(STATUS_RUNNING, "extracting patches for batch creation", 3)
            # TODO: IMPLEMENT ME!!

            self.update_job_status(STATUS_RUNNING, "updating completed dataset definition", 99)
            dataset = database_factory(self.registry).datasets_store.save_dataset(dataset, request=self.request)

            self.update_job_status(STATUS_SUCCEEDED, "processing complete", 100)

        except Exception as task_exc:
            exception_class = "{}.{}".format(type(task_exc).__module__, type(task_exc).__name__)
            err_msg = "{0}: {1}".format(exception_class, str(task_exc))
            message = "failed to run {!s} [{!s}].".format(self.job, err_msg)
            self.update_job_status(STATUS_FAILED, message, errors=task_exc)
        finally:
            self.update_job_status(self.job.status, "done")

        return self.job.status
