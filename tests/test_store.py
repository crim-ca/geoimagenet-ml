# noinspection PyProtectedMember
from geoimagenet_ml.store.datatypes import _check_io_format, Dataset, Model, Process, Job, Action
from geoimagenet_ml.constants import OPERATION
# noinspection PyPackageRequirements
import pytest


# noinspection PyTypeChecker
def test_check_io_format_not_list():
    with pytest.raises(TypeError):
        _check_io_format({"id": "item", "value": "random"})


# noinspection PyTypeChecker
def test_check_io_format_no_id():
    with pytest.raises(TypeError):
        _check_io_format([{"id": "item"}])


# noinspection PyTypeChecker
def test_check_io_format_no_value():
    with pytest.raises(TypeError):
        _check_io_format([{"value": "random"}])


# noinspection PyTypeChecker
def test_check_io_format_duplicates():
    with pytest.raises(ValueError):
        _check_io_format([{"id": "item", "value": "random"}, {"id": "item", "value": "duplicate"}])


# noinspection PyTypeChecker, PyBroadException
def test_check_io_format_empty():
    try:
        _check_io_format([])
    except Exception:
        pytest.fail("should not raise")


# noinspection PyTypeChecker, PyBroadException
def test_check_io_format_valid():
    try:
        _check_io_format([
            {"id": "item", "value": "random"},
            {"id": "other", "value": 1},
            {"id": "list", "value": ["some", "stuff"]}
        ])
    except Exception:
        pytest.fail("should not raise")


def check_fields(item, expected_params_fields, expected_json_fields, expected_summary_fields):
    item_name = type(item).__name__

    params = item.params
    for p in expected_params_fields:
        assert p in params, "Missing expected 'params' field for '{}': '{}'".format(item_name, p)
        params.pop(p)
    assert len(params) == 0, "Additional 'params' fields not expected for '{}': '{}'".format(item_name, params)

    if hasattr(item, "json") and expected_json_fields is not None:
        json_body = item.json()
        for j in expected_json_fields:
            assert j in json_body, "Missing expected 'json' field for '{}': '{}'".format(item_name, j)
            json_body.pop(j)
        assert len(params) == 0, "Additional 'json' fields not expected for '{}': '{}'".format(item_name, json_body)

    if hasattr(item, "summary") and expected_json_fields is not None:
        summary = item.summary()
        for s in expected_summary_fields:
            assert s in summary, "Missing expected 'summary' field for '{}': '{}'".format(item_name, s)
            summary.pop(s)
        assert len(params) == 0, "Additional 'summary' fields not expected for '{}': '{}'".format(item_name, summary)


def test_dataset_fields():
    check_fields(
        Dataset(uuid="test", name="test", path="test", type="test", user=1),
        ["uuid", "name", "type", "path", "data", "files", "status", "user", "created", "finished"],
        ["uuid", "name", "type", "path", "data", "files", "status", "user", "created", "finished"],
        ["uuid", "name"],
    )


def test_model_fields():
    check_fields(
        Model(uuid="test", name="test", path="test", type="test", user=1),
        ["uuid", "name", "path", "data", "user", "created", "visibility"],
        ["uuid", "name", "path", "data", "user", "created", "visibility"],
        ["uuid", "name"],
    )


def test_process_fields():
    check_fields(
        Process(uuid="test", type="test", user=1, identifier="test"),
        ["uuid", "type", "user",  "created", "identifier", "title", "abstract", "keywords", "metadata", "version",
         "inputs", "outputs", "execute_endpoint", "limit_single_job", "package"],
        ["uuid", "type", "user", "created", "identifier", "title", "abstract", "keywords", "metadata", "version",
         "inputs", "outputs", "execute_endpoint", "limit_single_job"],
        ["uuid", "identifier", "title", "abstract", "keywords", "metadata", "version", "execute_endpoint"],
    )


def test_job_fields():
    check_fields(
        Job(uuid="test", process="test", user=1),
        ["uuid", "task", "service", "process", "user", "inputs", "status", "status_message", "status_location",
         "execute_async", "is_workflow", "started", "created", "finished", "duration", "progress", "tags",
         "results", "logs", "exceptions", "request", "response", "visibility"],
        ["uuid", "task", "service", "process", "user", "inputs", "status", "status_message", "status_location",
         "execute_async", "is_workflow", "started", "created", "finished", "duration", "progress", "tags"],
        ["uuid", "process"],
    )


def test_action_fields():
    check_fields(
        Action(uuid="test", type=Job, operation=OPERATION.INFO, user=1),
        ["uuid", "type", "item", "user", "path", "method", "operation", "created"],
        None,
        None,
    )
