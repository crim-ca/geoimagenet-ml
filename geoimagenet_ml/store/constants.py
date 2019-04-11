from enum import Enum


# must match fields of 'Job' to use as search filters
class SORT(Enum):
    CREATED = "created"
    FINISHED = "finished"
    STATUS = "status"
    PROCESS = "process"
    SERVICE = "service"
    USER = "user"
    UUID = "uuid"


class ORDER(Enum):
    ASCENDING = "ascending"
    DESCENDING = "descending"


class OPERATION(Enum):
    CREATE = "create"
    DELETE = "delete"
    DOWNLOAD = "download"
    UPLOAD = "upload"
    GET = "get"
