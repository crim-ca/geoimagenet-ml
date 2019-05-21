from enum import Enum, EnumMeta, unique
from typing import TYPE_CHECKING
import six
if TYPE_CHECKING:
    from geoimagenet_ml.typedefs import Any, AnyKey, AnyStr, List, Optional  # noqa: F401
    # noinspection PyProtectedMember
    from typing import _TC  # noqa: F401


class ExtendedEnumMeta(EnumMeta):
    def names(cls):
        # type: () -> List[AnyStr]
        """Returns the member names assigned to corresponding enum elements."""
        return list(cls.__members__)

    def values(cls):
        # type: () -> List[AnyKey]
        """Returns the literal values assigned to corresponding enum elements."""
        return [m.value for m in cls.__members__.values()]

    def get(cls, key_or_value, default=None):
        # type: (AnyKey, Optional[Any]) -> Optional[_TC]
        """
        Finds a enum entry by defined name or its value.
        Returns the entry directly if it is already a valid enum.
        """
        if key_or_value in cls:
            return key_or_value
        for m_key, m_val in cls.__members__.items():
            if key_or_value == m_key or key_or_value == m_val.value:
                return m_val
        return default


# must match fields of corresponding object to use as search filters
@unique
class SORT(six.with_metaclass(ExtendedEnumMeta, str, Enum)):
    CREATED = "created"
    FINISHED = "finished"
    STATUS = "status"
    PROCESS = "process"
    SERVICE = "service"
    USER = "user"
    UUID = "uuid"


@unique
class ORDER(six.with_metaclass(ExtendedEnumMeta, str, Enum)):
    ASCENDING = "ascending"
    DESCENDING = "descending"


@unique
class OPERATION(six.with_metaclass(ExtendedEnumMeta, str, Enum)):
    SUBMIT = "submit"       # requests 'POST' with item submission (ex: Job)
    DELETE = "delete"       # requests 'DELETE' of an item
    UPDATE = "update"       # requests 'PUT' on item to modify it's content
    UPLOAD = "upload"       # requests 'POST' for created item with file
    DOWNLOAD = "download"   # requests 'GET' with file returned
    INFO = "info"           # requests 'GET' with single {} item returned
    LIST = "list"           # requests 'GET' with multiple {} items returned


@unique
class VISIBILITY(six.with_metaclass(ExtendedEnumMeta, str, Enum)):
    PUBLIC = "public"
    PRIVATE = "private"


@unique
class JOB_TYPE(six.with_metaclass(ExtendedEnumMeta, str, Enum)):
    CURRENT = "current"
    LATEST = "latest"
