from geoimagenet_ml import utils
import status
# noinspection PyPackageRequirements
import pytest
import os
CUR_DIR = os.path.abspath(os.path.dirname(__file__))


def test_str2paths_none():
    p = utils.str2paths()
    assert isinstance(p, list)
    assert len(p) == 0


def test_str2paths_single_existing():
    p = utils.str2paths(__file__)
    assert isinstance(p, list)
    assert len(p) == 1
    assert p[0] == __file__

    p = utils.str2paths(CUR_DIR)
    assert isinstance(p, list)
    assert len(p) == 1
    assert p[0] == CUR_DIR


def test_str2paths_single_missing():
    p = utils.str2paths("/random/missing/file.cfg")
    assert isinstance(p, list)
    assert len(p) == 0

    p = utils.str2paths("/random/missing/dir")
    assert isinstance(p, list)
    assert len(p) == 0


def test_str2paths_multi_existing():
    p = utils.str2paths(','.join([
        "/random/path",
        "/another/path",
        CUR_DIR,
        "/again/one/more",
        __file__,
    ]))
    assert isinstance(p, list)
    assert len(p) == 2
    assert p[0] == CUR_DIR
    assert p[1] == __file__


def test_str2paths_multi_missing():
    p = utils.str2paths(','.join([
        "/random/path",
        "/another/path",
        "/again/one/more",
    ]))
    assert isinstance(p, list)
    assert len(p) == 0


def test_str2paths_multi_mixed_exist_missing():
    p = utils.str2paths(','.join([
        "/random/path",
        "/another/path",
        CUR_DIR,
        "/again/one/more",
        __file__,
    ]))
    assert isinstance(p, list)
    assert len(p) == 2
    assert p[0] == CUR_DIR
    assert p[1] == __file__


def test_str2paths_multi_no_duplicates():
    p = utils.str2paths(','.join([
        CUR_DIR,
        CUR_DIR,
        __file__,
        CUR_DIR,
        __file__,
    ]))
    assert isinstance(p, list)
    assert len(p) == 2
    assert p[0] == CUR_DIR
    assert p[1] == __file__


def test_str2paths_multi_walk_files(tmp_path):
    f0 = tmp_path / "file0.txt"
    d1 = tmp_path / "sub"
    f1 = tmp_path / "file1.txt"
    d2 = d1 / "other"
    f2 = d2 / "file2.txt"
    d1.mkdir()
    d2.mkdir()
    f0.write_text("file0")
    f1.write_text("file1")
    f2.write_text("file2")
    p = utils.str2paths(','.join([
        str(tmp_path),
        str(f0),
        str(f2),
        str(d2),
    ]), list_files=True)
    assert isinstance(p, list)
    assert len(p) == 3
    assert p[0] == str(f0)
    assert p[1] == str(f1)
    assert p[2] == str(f2)


statuses_except_unknown = set(status.STATUS.__members__.values()) - {status.STATUS.UNKNOWN}


def get_status_variations(test_status):
    return [test_status.value.lower(),
            test_status.value.upper(),
            test_status.value.capitalize(),
            "Process" + test_status.value.capitalize()]


def test_map_status_ogc_compliant():
    for sv in statuses_except_unknown:
        for s in get_status_variations(sv):
            assert status.map_status(s, status.COMPLIANT.OGC) in \
                   status.job_status_categories[status.COMPLIANT.OGC]


def test_map_status_pywps_compliant():
    for sv in statuses_except_unknown:
        for s in get_status_variations(sv):
            assert status.map_status(s, status.COMPLIANT.PYWPS) in \
                   status.job_status_categories[status.COMPLIANT.PYWPS]


def test_map_status_owslib_compliant():
    for sv in statuses_except_unknown:
        for s in get_status_variations(sv):
            assert status.map_status(s, status.COMPLIANT.OWSLIB) in \
                   status.job_status_categories[status.COMPLIANT.OWSLIB]


def test_map_status_pywps_compliant_as_int_statuses():
    for s in range(len(status.WPS_STATUS)):
        if status.STATUS_PYWPS_MAP[s] != status.STATUS.UNKNOWN.value:
            assert status.map_status(s, status.COMPLIANT.PYWPS) in \
                   status.job_status_categories[status.COMPLIANT.PYWPS]


def test_map_status_pywps_back_and_forth():
    for s, i in status.STATUS_PYWPS_MAP.items():
        assert status.STATUS_PYWPS_IDS[i] == s


# noinspection PyTypeChecker
def test_get_sane_name_replace():
    kw = {"assert_invalid": False, "max_len": 25}
    assert utils.get_sane_name("Hummingbird", **kw) == "Hummingbird"
    assert utils.get_sane_name("MapMint Demo Instance", **kw) == "MapMint_Demo_Instance"
    assert utils.get_sane_name(None, **kw) is None
    assert utils.get_sane_name("12", **kw) is None
    assert utils.get_sane_name(" ab c ", **kw) == "ab_c"
    assert utils.get_sane_name("a_much_to_long_name_for_this_test", **kw) == "a_much_to_long_name_for_t"


def test_assert_sane_name():
    test_cases_invalid = [
        None,
        "12",   # too short
        " ab c ",
        "MapMint Demo Instance",
        "double--dashes_not_ok",
        "-start_dash_not_ok",
        "end_dash_not_ok-",
        "no_exclamation!point",
        "no_interrogation?point",
        "no_slashes/allowed",
        "no_slashes\\allowed",
    ]
    for test in test_cases_invalid:
        with pytest.raises(ValueError):
            utils.assert_sane_name(test)

    test_cases_valid = [
        "Hummingbird",
        "short",
        "a_very_long_name_for_this_test_is_ok_if_max_len_is_none",
        "AlTeRnAtInG_cApS"
        "middle-dashes-are-ok",
        "underscores_also_ok",
    ]
    for test in test_cases_valid:
        utils.assert_sane_name(test)
