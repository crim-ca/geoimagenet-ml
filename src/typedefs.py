# noinspection PyUnresolvedReferences
from typing import Any, AnyStr, Callable, Dict, List, Tuple, Optional, Union, Type, TYPE_CHECKING   # noqa: F401
if TYPE_CHECKING:
    # noinspection PyPackageRequirements
    from owslib.wps import WPSException                                 # noqa: F401
    from pywps import Process as ProcessWPS                             # noqa: F401
    from geoimagenet_ml.store.datatypes import Process as ProcessDB     # noqa: F401
    from geoimagenet_ml.processes.runners import ProcessRunner          # noqa: F401

    AnyProcess = Union[ProcessDB, ProcessWPS, ProcessRunner]
    Number = Union[float, int]
    SettingDict = Dict[AnyStr, AnyStr]
    OptionDict = Dict[AnyStr, Any]
    Output = OptionDict
    JsonValue = Union[AnyStr, Number, bool, None]
    JsonDict = Dict[AnyStr, Union[JsonValue, List[JsonValue], Dict[AnyStr, Union[JsonValue, 'JsonDict']]]]
    Error = Union[AnyStr, Exception, List[WPSException]]
    Input = OptionDict
    UUID = AnyStr
