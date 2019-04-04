# noinspection PyProtectedMember
from geoimagenet_ml.store.datatypes import _check_io_format
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
