from enum import Enum
# noinspection PyProtectedMember
from pywps.response.status import _WPS_STATUS, WPS_STATUS
from typing import TYPE_CHECKING
import six
if TYPE_CHECKING:
    from geoimagenet_ml.typedefs import AnyStatus  # noqa: F401
    from typing import AnyStr, Union  # noqa: F401


class COMPLIANT(Enum):
    LITERAL = "STATUS_COMPLIANT_LITERAL"
    OGC = "STATUS_COMPLIANT_OGC"
    PYWPS = "STATUS_COMPLIANT_PYWPS"
    OWSLIB = "STATUS_COMPLIANT_OWSLIB"
    CELERY = "STATUS_COMPLIANT_CELERY"


class CATEGORY(Enum):
    FINISHED = "STATUS_CATEGORY_FINISHED"
    RUNNING = "STATUS_CATEGORY_RUNNING"
    FAILED = "STATUS_CATEGORY_FAILED"


class STATUS(Enum):
    ACCEPTED = "accepted"
    STARTED = "started"
    PAUSED = "paused"
    SUCCESS = "success"
    SUCCEEDED = "succeeded"
    FINISHED = "finished"
    FAILED = "failed"
    FAILURE = "failure"
    RUNNING = "running"
    DISMISSED = "dismissed"
    EXCEPTION = "exception"
    RETRY = "retry"
    PENDING = "pending"
    UNKNOWN = "unknown"  # don't include in any below collections


job_status_categories = {
    # note:
    #   OGC compliant:  [Accepted, Running, Succeeded, Failed]
    #   PyWPS uses:     [Accepted, Started, Succeeded, Failed, Paused, Exception]
    #   OWSLib uses:    [Accepted, Running, Succeeded, Failed, Paused] (with 'Process' in front)
    #   Celery uses:    [PENDING, STARTED, RETRY, FAILURE, SUCCESS]
    # http://docs.opengeospatial.org/is/14-065/14-065.html#17
    # corresponding statuses are aligned vertically for 'COMPLIANT' groups
    COMPLIANT.OGC:       frozenset([STATUS.ACCEPTED, STATUS.RUNNING, STATUS.SUCCEEDED, STATUS.FAILED]),
    COMPLIANT.PYWPS:     frozenset([STATUS.ACCEPTED, STATUS.STARTED, STATUS.SUCCEEDED, STATUS.FAILED, STATUS.PAUSED, STATUS.EXCEPTION]),  # noqa: E501
    COMPLIANT.OWSLIB:    frozenset([STATUS.ACCEPTED, STATUS.RUNNING, STATUS.SUCCEEDED, STATUS.FAILED, STATUS.PAUSED]),                    # noqa: E501
    COMPLIANT.CELERY:    frozenset([STATUS.ACCEPTED, STATUS.STARTED, STATUS.SUCCEEDED, STATUS.FAILED, STATUS.PAUSED, STATUS.EXCEPTION]),  # noqa: E501
    # utility categories
    CATEGORY.RUNNING:    frozenset([STATUS.ACCEPTED, STATUS.RUNNING,   STATUS.STARTED,   STATUS.PAUSED]),
    CATEGORY.FINISHED:   frozenset([STATUS.FAILED,   STATUS.DISMISSED, STATUS.EXCEPTION, STATUS.SUCCEEDED]),
    CATEGORY.FAILED:     frozenset([STATUS.FAILED,   STATUS.DISMISSED, STATUS.EXCEPTION, STATUS.FAILURE])
}


# noinspection PyProtectedMember
STATUS_PYWPS_MAP = {s: _WPS_STATUS._fields[s].lower() for s in range(len(WPS_STATUS))}  # id -> str
STATUS_PYWPS_IDS = {k.lower(): v for v, k in STATUS_PYWPS_MAP.items()}                  # str -> id


def map_status(wps_status, compliant=COMPLIANT.OGC):
    # type: (AnyStatus, COMPLIANT) -> STATUS
    """
    Maps WPS statuses (``STATUS``, ``OWSLib`` or ``PyWPS``) to ``OWSLib``/``PyWPS`` compatible values.
    For each compliant combination, unsupported statuses are changed to corresponding ones (with closest logical match).
    Statuses are returned with ``STATUS`` format (lowercase and not preceded by 'Process').

    :param wps_status: one of ``STATUS`` to map to ``COMPLIANT`` standard or PyWPS ``int`` status.
    :param compliant: one of ``COMPLIANT.[...]`` values.
    :returns: mapped status complying to the requested compliant category, or ``STATUS.UNKNOWN`` if no match found.
    """

    # case of raw PyWPS status
    if isinstance(wps_status, int):
        return map_status(STATUS_PYWPS_MAP[wps_status], compliant)

    if isinstance(wps_status, six.string_types):
        # remove 'Process' from OWSLib statuses and lower for every compliant
        wps_status = wps_status.upper().replace("PROCESS", "")
        wps_status = STATUS[wps_status]

    job_status = wps_status

    if compliant != COMPLIANT.LITERAL:

        # celery to any WPS conversions
        if job_status == STATUS.FAILURE:
            job_status = STATUS.FAILED
        elif job_status in [STATUS.RETRY, STATUS.PENDING]:
            job_status = STATUS.ACCEPTED
        elif job_status in [STATUS.SUCCESS, STATUS.FINISHED]:
            job_status = STATUS.SUCCEEDED

        if compliant == COMPLIANT.OGC:
            if job_status in job_status_categories[CATEGORY.RUNNING]:
                if job_status in [STATUS.STARTED, STATUS.PAUSED]:
                    job_status = STATUS.RUNNING
            elif job_status in job_status_categories[CATEGORY.FAILED] and job_status != STATUS.FAILED:
                job_status = STATUS.FAILED

        elif compliant == COMPLIANT.PYWPS:
            if job_status == STATUS.RUNNING:
                job_status = STATUS.STARTED
            elif job_status == STATUS.DISMISSED:
                job_status = STATUS.FAILED

        elif compliant == COMPLIANT.OWSLIB:
            if job_status == STATUS.STARTED:
                job_status = STATUS.RUNNING
            elif job_status in job_status_categories[CATEGORY.FAILED] and job_status != STATUS.FAILED:
                job_status = STATUS.FAILED

    if str(job_status.value).upper() not in STATUS.__members__:
        job_status = STATUS.UNKNOWN
    return job_status
