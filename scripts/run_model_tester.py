#!/usr/bin/env python
"""
Utility script that allows calling :class:`geoimagenet_ml.processes.runners.ProcessRunnerModelTester` directly without
having to worry about database connection and other process/job specific elements.

Using this script, you can evaluate a given model with some dataset as if they where stored in the database.
Note that results will not be saved as an executed job, but rather only *dumped* locally for manual evaluation.

You will need to provide some parameters to emulate the dataset and model objects that would normally be retrieved from
the database. Ensure that you called ``make install-dev`` also to have all dependencies installed.
"""
import argparse
import logging
import json
import os
import sys
import uuid
from typing import TYPE_CHECKING

from pyramid.registry import Registry
from pyramid.request import Request

from geoimagenet_ml.processes.runners import ProcessRunnerBatchCreator, ProcessRunnerModelTester
from geoimagenet_ml.status import STATUS
from geoimagenet_ml.store.datatypes import Dataset, Job, Model, Process
from geoimagenet_ml.store.databases.types import MEMORY_TYPE
from geoimagenet_ml.store.factories import database_factory

if TYPE_CHECKING:
    from geoimagenet_ml.typedefs import JSON, List, Optional, SettingsType, Tuple


SCRIPTS_PATH = os.path.abspath(os.path.dirname(__file__))
DEFAULT_SESSIONS_PATH = os.path.join(os.path.dirname(SCRIPTS_PATH), "sessions")


class MockRegistry(Registry):
    _settings = {
        "geoimagenet_ml.api.db_factory": MEMORY_TYPE
    }  # type: SettingsType

    def __init__(self, settings):   # noqa
        # type: (SettingsType) -> None
        self._settings.update(settings)

    @property
    def settings(self):
        return self._settings


class MockRequest(Request):
    def __init__(self, registry):  # noqa
        self.registry = registry
        self.id = uuid.uuid4()


def run_model_tester(dataset_data, model_path, dataset_root=None, output_root=DEFAULT_SESSIONS_PATH):
    # type: (JSON, str, Optional[str], str) -> Job
    """Runs :class:`geoimagenet_ml.processes.runners.ProcessRunnerModelTester` with provided script inputs."""

    # configure some registries that the process uses to retrieve data
    task = None
    process = Process(identifier=ProcessRunnerModelTester.identifier, type=ProcessRunnerModelTester.type)
    job = Job(process=process.uuid)
    registry = MockRegistry({"geoimagenet_ml.ml.jobs_path": output_root})
    request = MockRequest(registry)
    db = database_factory(registry)

    # data that will be used by script
    dataset = Dataset({
        "name": "script-test-model",
        "type": ProcessRunnerBatchCreator.dataset_type,
        "path": dataset_root,
        "data": dataset_data
    })
    model = Model({
        "name": "script-test-model",
        "path": model_path,
        "file": model_path,
    })
    job.inputs = [
        {"id": "dataset", "value": dataset.uuid},
        {"id": "model", "value": model.uuid}
    ]

    # dump mock data for retrieval by process
    db.processes_store.save_process(process)
    db.jobs_store.save_job(job)
    db.models_store.save_model(model)
    db.datasets_store.save_dataset(dataset)

    runner = ProcessRunnerModelTester(task, registry, request, job.uuid)  # noqa
    runner()  # call is wrapped in try/except block, so nothing will be raised (logged in job)
    return job


def update_logging(log_level, logger_names=None):
    # type: (int, Optional[List[str]]) -> None
    """Update logger levels that will be used by the script, which can provide more details during execution."""
    logger_names = logger_names or []
    logger_names.append("geoimagenet_ml")
    for name in logger_names:
        logger = logging.getLogger(name)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.setLevel(log_level)


def main():
    ap = argparse.ArgumentParser(prog=__file__, description=__doc__, add_help=True, # noqa
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("dataset_data", help="""
JSON file path with dataset 'data' field contents.
Expected Format: 
  {
    "taxonomy": [
      # top most should have COUV and OBJE for respectively land-cover and object-detection annotations 
      # See GeoImageNet API Taxonomy route 
      { "id": <int>, "name_fr": "<>", "name_en": "<>", "code": "<4-letter-code>", 
        "children": [<same-object-format-recursively>]
      },
      <...>
    ], 
    "patches": [
      { 
        "image": "<original-image-reference>",
        "class": <taxonomy-class-id>, 
        "feature": "<annotation.UUID>",   # identifier of original annotation
        "split": "<train|valid|test>",    # only 'test' are considered with this script
        # crops are a variation of each annotated patch (think of it as data augmentation or some other viewpoint)
        "crops": [
          {
            "type": "<raw|fixed>",
            "path": "<path-to-image-patch>"
            "shape": [<int>, <int>, <...>],
            "data_type": 1,               # enum-based (see Shapely, but basically indicates Polygon)
            "coordinates": [<double>],    # 6 values for lat/lon and relative position on Polygon
          },
          <...>   # other crops (if any)
        ]
      }
      <...>   # other patches
    ]
  }
    """)
    ap.add_argument("model_path", help="Path to the model to employ.")
    ap.add_argument("-d", "--dataset-root", dest="dataset_root",
                    help="Override dataset root of image patch location. "
                         "Use containing directory of dataset data JSON file by default.")
    ap.add_argument("-o", "--output", dest="output_path",
                    help="Path where to output job session results. Will always generate an unique job ID "
                         "(default: '<GIN_ML_ROOT>/sessions').",
                    default=DEFAULT_SESSIONS_PATH)
    ap_log = ap.add_mutually_exclusive_group()
    ap_log.add_argument("-v", "--verbose", action="store_const", const=logging.DEBUG,
                        help="Increase logging verbosity.")
    ap_log.add_argument("-w", "--warn", action="store_const", const=logging.WARNING,
                        help="Reduce logging verbosity to only warning and error.")
    ap_log.add_argument("-q", "--quiet", action="store_const", const=logging.ERROR,
                        help="Reduce logging verbosity to only errors.")
    args = ap.parse_args()
    log_level = args.verbose or args.warn or args.quiet or logging.INFO
    update_logging(log_level, [ProcessRunnerModelTester.identifier])
    with open(args.dataset_data, "r") as fd:
        data = json.load(fd)
    out_path = os.path.abspath(args.output_path)
    job = run_model_tester(data, args.model_path, args.dataset_root or os.path.split(args.dataset_data)[0], out_path)
    job_out_path = os.path.join(out_path, job.uuid)
    if job.status == STATUS.SUCCEEDED:
        with open(os.path.join(job_out_path, "results.json"), "w") as fd:
            json.dump(job.results, fd, indent=4, sort_keys=False)
        print(f"Success. Full results session location: [{job_out_path}]")
        sys.exit(0)
    print(f"Failure:\nLogs:\n{job.logs}\nExceptions:\n{job.exceptions}")


if __name__ == "__main__":
    main()
