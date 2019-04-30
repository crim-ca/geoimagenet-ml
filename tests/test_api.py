#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_api
----------------------------------

Tests for `GeoImageNet ML API` module.
"""

from geoimagenet_ml import __meta__
from geoimagenet_ml.api import schemas
from geoimagenet_ml.store.databases.types import MEMORY_TYPE, MONGODB_TYPE
from geoimagenet_ml.store.datatypes import Model
from geoimagenet_ml.store.factories import database_factory
from tests import utils
import pyramid.testing
# noinspection PyPackageRequirements
import pytest
# noinspection PyPackageRequirements
import mock
import unittest
import tempfile
import pyramid
import warnings
import json
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

    @classmethod
    def setUpClass(cls):
        cls.conf = utils.setup_config_with_mongodb()
        cls.app = utils.setup_test_app(config=cls.conf)
        cls.json_headers = [("Content-Type", schemas.ContentTypeJSON), ("Accept", schemas.ContentTypeJSON)]
        cls.db = database_factory(cls.conf.registry)    # type: MongoDatabase
        cls.MODEL_BASE_PATH = cls.conf.registry.settings.get("geoimagenet_ml.ml.models_path")

        # url to existing remote model file definition
        cls.TEST_MODEL_URL = os.getenv("TEST_MODEL_URL")
        if not cls.TEST_MODEL_URL:
            raise LookupError("Missing required test environment variable: `TEST_MODEL_URL`.")

    # noinspection PyUnresolvedReferences
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

    def setUp(self):
        if not self.db.models_store.clear_models():
            warnings.warn("Models could not be cleared, future tests might fail due to unexpected values.", Warning)

        self.model_1 = self.make_model("model-1", data={"model": "test-1"})
        self.model_2 = self.make_model("model-2", data={"model": "test-2"})

        def load_checkpoint_no_check(buffer):
            buffer.seek(0)
            return buffer.read()

        with mock.patch("thelper.utils.load_checkpoint", side_effect=load_checkpoint_no_check):
            self.db.models_store.save_model(self.model_1)
            self.db.models_store.save_model(self.model_2)

    def tearDown(self):
        for f in [self.model_1.file, self.model_1.path, self.model_2.file, self.model_2.path]:
            if os.path.isfile(f):
                os.remove(f)

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
        path = schemas.ModelAPI.path.replace("{model_uuid}", resp.json["data"]["model"]["uuid"])
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
        path = schemas.ModelDownloadAPI.path.replace("{model_uuid}", resp.json["data"]["model"]["uuid"])
        resp = utils.request(self.app, "GET", path)
        utils.check_val_equal(resp.status_code, 200)


if __name__ == "__main__":
    import sys
    sys.exit(unittest.main())
