#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_api
----------------------------------

Tests for `GeoImageNet ML API` module.
"""

from geoimagenet_ml import __meta__
from geoimagenet_ml.api import schemas
from geoimagenet_ml.constants import JOB_TYPE, VISIBILITY
from geoimagenet_ml.processes.types import process_mapping
from geoimagenet_ml.status import STATUS
from geoimagenet_ml.store.databases.types import MEMORY_TYPE, MONGODB_TYPE
from geoimagenet_ml.store.datatypes import Model, Process, Job
from geoimagenet_ml.store.exceptions import ModelNotFoundError
from geoimagenet_ml.store.factories import database_factory
from geoimagenet_ml.utils import now
from dateutil.parser import parse as dt_parse
from tests import utils
import pyramid.testing
import pytest
import mock
import unittest
import tempfile
import pyramid
import warnings
import json
import uuid
import six
import os

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geoimagenet_ml.store.databases.mongodb import MongoDatabase


class TestImportApi(unittest.TestCase):
    """Validate API startup with working imports.

    .. seealso::
        - :module:`geoimagenet_ml.ml.impl` for details.
    """

    def test_import_api(self):
        """No ``ImportError`` should occur on any import from :module:`geoimagenet_ml.ml.impl`."""
        try:
            from geoimagenet_ml.ml.impl import load_model
        except ImportError as ex:
            self.fail("Could not import API due to import error by ML modules. [{!r}]".format(ex))


class TestGenericApi(unittest.TestCase):
    """Test Generic API operations."""

    @classmethod
    def setUpClass(cls):
        cls.conf = utils.setup_config_with_mongodb()
        cls.app = utils.setup_test_app(config=cls.conf)
        cls.json_headers = [("Content-Type", schemas.ContentTypeJSON), ("Accept", schemas.ContentTypeJSON)]
        # cls.db = database_factory(cls.conf.registry)    # type: MongoDatabase

    # noinspection PyUnresolvedReferences
    @classmethod
    def tearDownClass(cls):
        pyramid.testing.tearDown()

    def test_GetVersion_valid(self):
        resp = utils.request(self.app, "GET", schemas.VersionsAPI.path, headers=self.json_headers)
        utils.check_response_basic_info(resp, 200)
        utils.check_val_equal(resp.json["data"]["versions"][0]["name"], "api")
        utils.check_val_type(resp.json["data"]["versions"][0]["version"], six.string_types)
        utils.check_val_equal(resp.json["data"]["versions"][0]["version"], __meta__.__version__)
        utils.check_val_equal(resp.json["data"]["versions"][1]["name"], "db")
        utils.check_val_type(resp.json["data"]["versions"][1]["version"], six.string_types)
        utils.check_val_type(resp.json["data"]["versions"][1]["type"], six.string_types)
        utils.check_val_is_in(resp.json["data"]["versions"][1]["type"], [MEMORY_TYPE, MONGODB_TYPE])
        utils.check_val_equal(resp.json["data"]["versions"][2]["name"], "ml")
        utils.check_val_type(resp.json["data"]["versions"][2]["version"], six.string_types)
        utils.check_val_type(resp.json["data"]["versions"][2]["type"], six.string_types)

    def test_GetAPI_valid(self):
        resp = utils.request(self.app, "GET", schemas.SwaggerJSON.path, headers=self.json_headers)
        assert resp.status_code == 200
        assert resp.content_type == schemas.ContentTypeJSON
        utils.check_val_is_in("info", resp.json)
        utils.check_val_equal(resp.json["info"]["version"], __meta__.__version__)
        utils.check_val_is_in("paths", resp.json)


class TestModelApi(unittest.TestCase):
    """Test Model API operations."""
    db = None

    @classmethod
    def setUpClass(cls):
        cls.conf = utils.setup_config_with_mongodb()
        cls.app = utils.setup_test_app(config=cls.conf)
        cls.json_headers = [("Content-Type", schemas.ContentTypeJSON), ("Accept", schemas.ContentTypeJSON)]
        cls.db = database_factory(cls.conf.registry)  # type: MongoDatabase
        cls.MODEL_BASE_PATH = cls.conf.registry.settings.get("geoimagenet_ml.ml.models_path")

        # url to existing remote model file definition
        cls.TEST_MODEL_URL = os.getenv("TEST_MODEL_URL")
        if not cls.TEST_MODEL_URL:
            raise LookupError("Missing required test environment variable: `TEST_MODEL_URL`.")

    @classmethod
    def tearDownClass(cls):
        cls.db.models_store.clear_models()
        pyramid.testing.tearDown()

    @staticmethod
    def make_model(name, data):
        tmp = tempfile.NamedTemporaryFile(mode="w+", delete=False)
        tmp.write(json.dumps(data))
        tmp.close()
        return Model(name=name, path=tmp.name)

    def delete_models(self):
        for m in [self.model_1, self.model_2]:
            try:
                self.db.models_store.delete_model(m.uuid)
            except ModelNotFoundError:
                pass

    def setUp(self):
        if not self.db.models_store.clear_models():
            warnings.warn("Models could not be cleared, future tests might fail due to unexpected values.", Warning)

        self.model_1 = self.make_model("model-1", data={"model": "test-1"})
        self.model_2 = self.make_model("model-2", data={"model": "test-2"})
        self.delete_models()

        def load_checkpoint_no_check(buffer):
            buffer.seek(0)
            return buffer.read()

        def valid_model_no_check(data):
            return True, None

        with mock.patch("thelper.utils.load_checkpoint", side_effect=load_checkpoint_no_check), \
                mock.patch("geoimagenet_ml.ml.impl.valid_model", side_effect=valid_model_no_check), \
                mock.patch("geoimagenet_ml.store.datatypes.valid_model", side_effect=valid_model_no_check):
            self.model_1 = self.db.models_store.save_model(self.model_1)
            self.model_2 = self.db.models_store.save_model(self.model_2)

        self.process = Process(uuid=uuid.uuid4(), type="test", identifier="test")
        self.db.processes_store.delete_process(self.process.identifier)
        self.db.processes_store.save_process(self.process)

    def tearDown(self):
        for f in [self.model_1.file, self.model_1.path, self.model_2.file, self.model_2.path]:
            if os.path.isfile(f):
                os.remove(f)
        self.delete_models()
        self.db.processes_store.delete_process(self.process.identifier)

    def test_GetModels_valid(self):
        resp = utils.request(self.app, "GET", schemas.ModelsAPI.path, headers=self.json_headers)
        utils.check_response_basic_info(resp, 200)
        utils.check_val_is_in("models", resp.json["data"])
        utils.check_val_type(resp.json["data"]["models"], list)
        for model in resp.json["data"]["models"]:
            utils.check_val_type(model, dict)
            utils.check_val_is_in("uuid", model)
            utils.check_val_is_in("name", model)
        models_uuid = [m["uuid"] for m in resp.json["data"]["models"]]
        utils.check_val_is_in(self.model_1.uuid, models_uuid)
        utils.check_val_is_in(self.model_2.uuid, models_uuid)

    @pytest.mark.online
    def test_PostModels_RemoteModel_valid(self):
        model_json = {
            "model_name": "new-test-model",
            "model_path": self.TEST_MODEL_URL
        }
        resp = utils.request(self.app, "POST", schemas.ModelsAPI.path, headers=self.json_headers, json=model_json)
        utils.check_response_basic_info(resp, 201)
        utils.check_val_is_in("uuid", resp.json["data"]["model"])
        utils.check_val_type(resp.json["data"]["model"]["uuid"], six.string_types)
        utils.check_val_is_in("name", resp.json["data"]["model"])
        utils.check_val_type(resp.json["data"]["model"]["name"], six.string_types)
        utils.check_val_is_in("path", resp.json["data"]["model"])
        utils.check_val_type(resp.json["data"]["model"]["path"], six.string_types)
        utils.check_val_is_in("created", resp.json["data"]["model"])
        utils.check_val_type(resp.json["data"]["model"]["created"], six.string_types)

        # validate that model can be retrieved from database after creation
        path = schemas.ModelAPI.path.replace(schemas.VariableModelUUID, resp.json["data"]["model"]["uuid"])
        resp = utils.request(self.app, "GET", path, headers=self.json_headers)
        utils.check_response_basic_info(resp, 200)
        utils.check_val_equal(resp.json["data"]["model"]["name"], model_json["model_name"])

        # validate that model file was registered to expected storage location
        # noinspection PyProtectedMember, PyUnresolvedReferences
        model_name = resp.json["data"]["model"]["uuid"] + self.db.models_store._model_ext
        saved_path = os.path.join(self.MODEL_BASE_PATH, model_name)
        assert os.path.isfile(saved_path)

        # validate that displayed path corresponds to uploaded model source path/URL
        assert resp.json["data"]["model"]["path"] == self.TEST_MODEL_URL

    @pytest.mark.online
    def test_DownloadModel_valid(self):
        # setup testing model
        model_json = {
            "model_name": "new-test-model",
            "model_path": self.TEST_MODEL_URL
        }
        resp = utils.request(self.app, "POST", schemas.ModelsAPI.path, headers=self.json_headers, json=model_json)
        utils.check_response_basic_info(resp, 201)

        # validate download
        path = schemas.ModelDownloadAPI.path.replace(schemas.VariableModelUUID, resp.json["data"]["model"]["uuid"])
        resp = utils.request(self.app, "GET", path)
        utils.check_val_equal(resp.status_code, 200)

    def test_UpdateModel(self):
        path = schemas.ModelAPI.path.replace(schemas.VariableModelUUID, self.model_1.uuid)
        resp = utils.request(self.app, "GET", path)
        utils.check_val_equal(resp.status_code, 200)
        assert resp.json["data"]["model"]["visibility"] == VISIBILITY.PRIVATE.value

        resp = utils.request(self.app, "PUT", path, body={"visibility": VISIBILITY.PUBLIC.value})
        utils.check_val_equal(resp.status_code, 200)
        resp = utils.request(self.app, "GET", path)
        utils.check_val_equal(resp.status_code, 200)
        assert resp.json["data"]["model"]["visibility"] == VISIBILITY.PUBLIC.value

        resp = utils.request(self.app, "PUT", path, body={"visibility": "RANDOM_VALUE!!"}, expect_errors=True)
        utils.check_val_is_in(resp.status_code, [400, 403])
        resp = utils.request(self.app, "GET", path)
        utils.check_val_equal(resp.status_code, 200)
        assert resp.json["data"]["model"]["visibility"] == VISIBILITY.PUBLIC.value

        model_new_name = "new-name"
        resp = utils.request(self.app, "PUT", path, body={"name": model_new_name})
        utils.check_val_equal(resp.status_code, 200)
        resp = utils.request(self.app, "GET", path)
        utils.check_val_equal(resp.status_code, 200)
        assert resp.json["data"]["model"]["name"] == model_new_name

        # missing update fields
        resp = utils.request(self.app, "PUT", path, expect_errors=True)
        utils.check_val_equal(resp.status_code, 400)


class TestProcessJobApi(unittest.TestCase):
    """Test Process/Job API operations."""
    db = None

    @classmethod
    def setUpClass(cls):
        cls.conf = utils.setup_config_with_mongodb()
        cls.app = utils.setup_test_app(config=cls.conf)
        cls.json_headers = [("Content-Type", schemas.ContentTypeJSON), ("Accept", schemas.ContentTypeJSON)]
        cls.db = database_factory(cls.conf.registry)  # type: MongoDatabase

    @classmethod
    def tearDownClass(cls):
        cls.delete_test_processes()
        cls.delete_test_jobs()
        pyramid.testing.tearDown()

    @classmethod
    def delete_test_processes(cls):
        test_processes = list(filter(lambda _p: _p.identifier not in process_mapping,
                                     cls.db.processes_store.list_processes()))
        for p in test_processes:
            cls.db.processes_store.delete_process(p.identifier)

    @classmethod
    def delete_test_jobs(cls):
        for j in cls.db.jobs_store.list_jobs():
            cls.db.jobs_store.delete_job(j.uuid)

    def setUp(self):
        self.process_single = Process(uuid=uuid.uuid4(), type="test", identifier="test-single", limit_single_job=True)
        self.process_multi = Process(uuid=uuid.uuid4(), type="test", identifier="test-multi", limit_single_job=False)
        self.delete_test_processes()
        self.delete_test_jobs()
        self.db.processes_store.save_process(self.process_single)
        self.db.processes_store.save_process(self.process_multi)

    def test_UpdateJob(self):
        job = Job(uuid=uuid.uuid4(), process=self.process_multi.uuid, status=STATUS.ACCEPTED)
        path = schemas.ProcessJobAPI.path \
            .replace(schemas.VariableProcessUUID, self.process_multi.uuid) \
            .replace(schemas.VariableJobUUID, job.uuid)
        self.db.jobs_store.save_job(job)

        resp = utils.request(self.app, "GET", path)
        utils.check_val_equal(resp.status_code, 200)
        assert resp.json["data"]["job"]["visibility"] == VISIBILITY.PRIVATE.value

        resp = utils.request(self.app, "PUT", path, body={"visibility": VISIBILITY.PUBLIC.value})
        utils.check_val_equal(resp.status_code, 200)
        resp = utils.request(self.app, "GET", path)
        utils.check_val_equal(resp.status_code, 200)
        assert resp.json["data"]["job"]["visibility"] == VISIBILITY.PUBLIC.value

        resp = utils.request(self.app, "PUT", path, body={"visibility": "RANDOM_VALUE!!"}, expect_errors=True)
        utils.check_val_is_in(resp.status_code, [400, 403])
        resp = utils.request(self.app, "GET", path)
        utils.check_val_equal(resp.status_code, 200)
        assert resp.json["data"]["job"]["visibility"] == VISIBILITY.PUBLIC.value

    @utils.mock_execute_process()
    def test_PostJob_BatchCreation(self):
        """
        Validate basic job submission is working and that corresponding routes return expected bodies.
        No job execution is executed here (assumed there is no Celery worker).
        """
        path_proc = schemas.ProcessAPI.path.replace(schemas.VariableProcessUUID, "batch-creation")
        path_jobs = schemas.ProcessJobsAPI.path.replace(schemas.VariableProcessUUID, "batch-creation")
        body = {
            "inputs": [
                {"id": "name", "value": "test-batch"},
                {"id": "geojson_urls", "value": ["https://geoimagenet.crim.ca/api/v1/batches/annotations"]},
                {"id": "overwrite", "value": True},
            ]
        }

        dt_before = now()
        resp = utils.request(self.app, "POST", path_jobs, body=body)
        dt_after = now()
        utils.check_val_equal(resp.status_code, 202)
        location = resp.json["data"]["location"]
        job_uuid = resp.json["data"]["job_uuid"]
        assert resp.headers["Location"] == location

        resp = utils.request(self.app, "GET", path_proc)
        utils.check_val_equal(resp.status_code, 200)
        process_uuid = resp.json["data"]["process"]["uuid"]

        resp = utils.request(self.app, "GET", location)
        utils.check_val_equal(resp.status_code, 200)
        dt_submit = dt_parse(resp.json["data"]["job"]["created"])
        assert dt_after > dt_submit > dt_before  # entry must indicate job creation timestamp although not executed yet
        assert resp.json["data"]["job"]["uuid"] == job_uuid
        assert resp.json["data"]["job"]["inputs"] == body["inputs"]
        assert resp.json["data"]["job"]["status"] == STATUS.ACCEPTED.value
        assert resp.json["data"]["job"]["process"] == process_uuid
        assert resp.json["data"]["job"]["visibility"] == VISIBILITY.PRIVATE.value
        assert resp.json["data"]["job"]["user"] is None  # no cookies to fetch id
        assert resp.json["data"]["job"]["task"] is None
        assert resp.json["data"]["job"]["started"] is None
        assert resp.json["data"]["job"]["finished"] is None
        assert resp.json["data"]["job"]["duration"] is None
        assert resp.json["data"]["job"]["progress"] == 0
        assert resp.json["data"]["job"]["service"] is None
        assert resp.json["data"]["job"]["tags"] == []
        assert resp.json["data"]["job"]["execute_async"] is True
        assert resp.json["data"]["job"]["is_workflow"] is False

    def test_submit_process_missing_inputs(self):
        path = schemas.ProcessJobsAPI.path.replace(schemas.VariableProcessUUID, self.process_single.uuid)
        resp = utils.request(self.app, "POST", path, body={}, expect_errors=True)
        utils.check_val_equal(resp.status_code, 400, msg="Expected bad request because of missing inputs.")

    def test_submit_process_invalid_inputs(self):
        path = schemas.ProcessJobsAPI.path.replace(schemas.VariableProcessUUID, self.process_single.uuid)
        resp = utils.request(self.app, "POST", path, body={"inputs": {"in": "put"}}, expect_errors=True)
        utils.check_val_equal(resp.status_code, 422, msg="Expected unprocessable inputs because not list of dicts.")

        resp = utils.request(self.app, "POST", path, body={"inputs": [1, 2, 3]}, expect_errors=True)
        utils.check_val_equal(resp.status_code, 422, msg="Expected unprocessable inputs because missing id.")

        resp = utils.request(self.app, "POST", path, body={"inputs": [{"in": "put"}]}, expect_errors=True)
        utils.check_val_equal(resp.status_code, 422, msg="Expected unprocessable inputs because missing id.")

    @utils.mock_execute_process(process_id="test-single")
    @utils.mock_execute_process(process_id="test-multi")
    def test_submit_process_job_single_or_multi_limit(self):
        path = schemas.ProcessJobsAPI.path.replace(schemas.VariableProcessUUID, self.process_single.uuid)
        resp = utils.request(self.app, "POST", path, body={"inputs": []})
        utils.check_val_equal(resp.status_code, 202)

        resp = utils.request(self.app, "POST", path, body={"inputs": []}, expect_errors=True)
        utils.check_val_equal(resp.status_code, 403, msg="Second job should be forbidden for single job process.")

        jobs, count = self.db.jobs_store.find_jobs(process=self.process_single.uuid)
        utils.check_val_equal(count, 1, msg="Should have only 1 job pending execution.")

        jobs[0].update_finished_datetime()
        jobs[0].status = STATUS.SUCCEEDED
        self.db.jobs_store.update_job(jobs[0])
        resp = utils.request(self.app, "POST", path, body={"inputs": []})
        utils.check_val_equal(resp.status_code, 202, msg="New job should be allowed when previous one was completed.")

        path = schemas.ProcessJobsAPI.path.replace(schemas.VariableProcessUUID, self.process_multi.uuid)
        resp = utils.request(self.app, "POST", path, body={"inputs": []})
        utils.check_val_equal(resp.status_code, 202)

        resp = utils.request(self.app, "POST", path, body={"inputs": []})
        utils.check_val_equal(resp.status_code, 202, msg="Second job should be allowed for non single job process.")

        jobs, count = self.db.jobs_store.find_jobs(process=self.process_multi.uuid)
        utils.check_val_equal(count, 2, msg="Should have 2 jobs pending execution.")

    @utils.mock_execute_process(process_id="test-single")
    def test_get_current_job_when_limit_single_job(self):
        path_proc = schemas.ProcessJobsAPI.path.replace(schemas.VariableProcessUUID, self.process_single.uuid)
        path_curr = schemas.ProcessJobAPI.path \
            .replace(schemas.VariableProcessUUID, self.process_single.uuid) \
            .replace(schemas.VariableJobUUID, JOB_TYPE.CURRENT.value)

        resp = utils.request(self.app, "POST", path_proc, body={"inputs": []})
        utils.check_val_equal(resp.status_code, 202)
        job_uuid = resp.json["data"]["job_uuid"]

        # 'ACCEPTED' job is the 'CURRENT'
        resp = utils.request(self.app, "GET", path_curr)
        utils.check_val_equal(resp.status_code, 200)
        utils.check_val_equal(resp.json["data"]["job"]["status"], STATUS.ACCEPTED.value,
                              msg="Job fetched by current should be the one accepted.")
        utils.check_val_equal(resp.json["data"]["job"]["uuid"], job_uuid,
                              msg="Job fetched by current should be the one accepted.")

        # update to 'RUNNING' and check 'CURRENT' also returns the same job
        job = self.db.jobs_store.fetch_by_uuid(job_uuid)
        job.status = STATUS.RUNNING
        self.db.jobs_store.update_job(job)
        resp = utils.request(self.app, "GET", path_curr)
        utils.check_val_equal(resp.status_code, 200)
        utils.check_val_equal(resp.json["data"]["job"]["status"], STATUS.RUNNING.value,
                              msg="Job fetched by current should be the one running.")
        utils.check_val_equal(resp.json["data"]["job"]["uuid"], job_uuid,
                              msg="Job fetched by current should be the one running.")

        # update to 'FINISHED' and check that 'CURRENT' doesn't exist
        job = self.db.jobs_store.fetch_by_uuid(job_uuid)
        job.update_finished_datetime()
        job.status = STATUS.SUCCEEDED
        self.db.jobs_store.update_job(job)
        resp = utils.request(self.app, "GET", path_curr, expect_errors=True)
        utils.check_val_equal(resp.status_code, 404)

    @utils.mock_execute_process(process_id="test-multi")
    def test_get_current_job_when_allowed_multi_jobs(self):
        # verify that 'CURRENT' is not allowed for multi-job processes
        path = schemas.ProcessJobsAPI.path.replace(schemas.VariableProcessUUID, self.process_multi.uuid)
        resp = utils.request(self.app, "POST", path, body={"inputs": []})
        utils.check_val_equal(resp.status_code, 202)
        job_uuid = resp.json["data"]["job_uuid"]
        path = schemas.ProcessJobAPI.path \
            .replace(schemas.VariableProcessUUID, self.process_multi.uuid) \
            .replace(schemas.VariableJobUUID, JOB_TYPE.CURRENT.value)
        resp = utils.request(self.app, "GET", path, expect_errors=True)
        utils.check_val_equal(resp.status_code, 403)

    # behaves the same for regardless of job limit
    @utils.mock_execute_process(process_id="test-single")
    def test_get_latest_job(self):
        path_proc = schemas.ProcessJobsAPI.path.replace(schemas.VariableProcessUUID, self.process_single.uuid)
        path_last = schemas.ProcessJobAPI.path \
            .replace(schemas.VariableProcessUUID, self.process_single.uuid) \
            .replace(schemas.VariableJobUUID, JOB_TYPE.LATEST.value)

        resp = utils.request(self.app, "POST", path_proc, body={"inputs": []})
        utils.check_val_equal(resp.status_code, 202)
        job1_uuid = resp.json["data"]["job_uuid"]

        # 'ACCEPTED' job doesn't exist for 'LATEST'
        resp = utils.request(self.app, "GET", path_last, expect_errors=True)
        utils.check_val_equal(resp.status_code, 404, msg="No job should be found when there is only a pending process.")

        # update to 'RUNNING' and check that it still doesn't exist
        job = self.db.jobs_store.fetch_by_uuid(job1_uuid)
        job.status = STATUS.RUNNING
        self.db.jobs_store.update_job(job)
        resp = utils.request(self.app, "GET", path_last, expect_errors=True)
        utils.check_val_equal(resp.status_code, 404, msg="No job should be found when there is only a running process.")

        # update to 'FINISHED' and check that 'LATEST' is found
        job = self.db.jobs_store.fetch_by_uuid(job1_uuid)
        job.update_finished_datetime()
        job.status = STATUS.SUCCEEDED
        self.db.jobs_store.update_job(job)
        resp = utils.request(self.app, "GET", path_last)
        utils.check_val_equal(resp.status_code, 200)
        utils.check_val_equal(resp.json["data"]["job"]["uuid"], job1_uuid)
        utils.check_val_equal(resp.json["data"]["job"]["status"], STATUS.SUCCEEDED.value,
                              msg="Job fetched by latest should be succeeded.")

        # create a new job and validate that it become the new 'LATEST' one
        resp = utils.request(self.app, "POST", path_proc, body={"inputs": []})
        utils.check_val_equal(resp.status_code, 202)
        job2_uuid = resp.json["data"]["job_uuid"]
        job = self.db.jobs_store.fetch_by_uuid(job2_uuid)
        job.update_finished_datetime()
        job.status = STATUS.SUCCEEDED
        self.db.jobs_store.update_job(job)
        resp = utils.request(self.app, "GET", path_last)
        utils.check_val_equal(resp.status_code, 200)
        utils.check_val_equal(resp.json["data"]["job"]["uuid"], job2_uuid,
                              msg="More recent job should be returned.")
        utils.check_val_equal(resp.json["data"]["job"]["status"], STATUS.SUCCEEDED.value,
                              msg="Job fetched by latest should be succeeded.")

        # mark 2nd job as 'FAILED' and check that 1st is now returned
        job = self.db.jobs_store.fetch_by_uuid(job2_uuid)
        job.status = STATUS.FAILED
        self.db.jobs_store.update_job(job)
        resp = utils.request(self.app, "GET", path_last)
        utils.check_val_equal(resp.status_code, 200)
        utils.check_val_equal(resp.json["data"]["job"]["uuid"], job1_uuid,
                              msg="Job failed should be omitted and most recent succeeded one should be returned.")
        utils.check_val_equal(resp.json["data"]["job"]["status"], STATUS.SUCCEEDED.value,
                              msg="Job fetched by latest should be succeeded.")

        # mark 1st job as 'FAILED' and check that none is returned
        job = self.db.jobs_store.fetch_by_uuid(job1_uuid)
        job.status = STATUS.FAILED
        self.db.jobs_store.update_job(job)
        resp = utils.request(self.app, "GET", path_last, expect_errors=True)
        utils.check_val_equal(resp.status_code, 404, msg="Failed jobs should all be omitted from search.")


if __name__ == "__main__":
    import sys

    sys.exit(unittest.main())
