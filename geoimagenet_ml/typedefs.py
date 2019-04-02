# noinspection PyUnresolvedReferences
from typing import Any, AnyStr, Callable, Dict, List, Tuple, Optional, Union, Type, TYPE_CHECKING   # noqa: F401
if TYPE_CHECKING:
    from geoimagenet_ml.store.datatypes import Process as ProcessDB     # noqa: F401
    from geoimagenet_ml.processes.runners import ProcessRunner          # noqa: F401
    # noinspection PyPackageRequirements
    from owslib.wps import WPSException                                 # noqa: F401
    from pywps import Process as ProcessWPS                             # noqa: F401
    # noinspection PyProtectedMember
    from logging import _loggerClass                                    # noqa: F401
    from uuid import UUID as _UUID

    AnyProcess = Union[ProcessDB, ProcessWPS, ProcessRunner]
    Number = Union[float, int]
    JsonKey = Union[AnyStr, int]
    JsonValue = Union[AnyStr, Number, bool, None]
    JSON = Dict[JsonKey, Union[JsonValue, List['JSON'], Dict[JsonKey, 'JSON']]]
    SettingsType = Dict[AnyStr, JsonValue]
    OptionType = Dict[AnyStr, Any]
    InputType = OptionType
    OutputType = OptionType
    LoggerType = _loggerClass
    ErrorType = Union[AnyStr, Exception, List[WPSException]]
    LevelType = Union[AnyStr, int]
    UUID = Union[AnyStr, _UUID]
