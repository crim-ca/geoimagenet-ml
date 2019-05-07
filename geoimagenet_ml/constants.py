from enum import Enum, EnumMeta
from typing import TYPE_CHECKING
import six
if TYPE_CHECKING:
    from geoimagenet_ml.typedefs import Any, AnyKey, List, Optional, Type  # noqa: F401


class ExtendedEnumMeta(EnumMeta):
    def values(cls):
        # type: () -> List[AnyKey]
        """Returns the literal values assigned to each enum element."""
        return [m.value for m in cls.__members__.values()]

    def get(cls, key_or_value, default=None):
        # type: (AnyKey, Optional[Any]) -> Optional[Type[Enum]]
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
class SORT(six.with_metaclass(ExtendedEnumMeta, Enum)):
    CREATED = "created"
    FINISHED = "finished"
    STATUS = "status"
    PROCESS = "process"
    SERVICE = "service"
    USER = "user"
    UUID = "uuid"


class ORDER(six.with_metaclass(ExtendedEnumMeta, Enum)):
    ASCENDING = "ascending"
    DESCENDING = "descending"


class OPERATION(six.with_metaclass(ExtendedEnumMeta, Enum)):
    SUBMIT = "submit"       # requests 'POST' with item submission (ex: Job)
    DELETE = "delete"       # requests 'DELETE' of an item
    UPDATE = "update"       # requests 'PUT' on item to modify it's content
    UPLOAD = "upload"       # requests 'POST' for created item with file
    DOWNLOAD = "download"   # requests 'GET' with file returned
    INFO = "info"           # requests 'GET' with single {} item returned
    LIST = "list"           # requests 'GET' with multiple {} items returned


class VISIBILITY(six.with_metaclass(ExtendedEnumMeta, Enum)):
    PUBLIC = "public"
    PRIVATE = "private"
