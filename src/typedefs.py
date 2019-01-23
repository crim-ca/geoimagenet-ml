# noinspection PyUnresolvedReferences
from typing import Any, AnyStr, Dict, List, Tuple, Optional, Union, TYPE_CHECKING   # noqa: F401

Number = Union[float, int]
SettingDict = Dict[AnyStr, AnyStr]
OptionDict = Dict[AnyStr, Any]
JsonValue = Union[AnyStr, Number, bool, None]
JsonDict = Dict[AnyStr, Union[JsonValue, List[JsonValue], Dict[AnyStr, Union[JsonValue, 'JsonDict']]]]
Input = OptionDict
Output = OptionDict
UUID = AnyStr
