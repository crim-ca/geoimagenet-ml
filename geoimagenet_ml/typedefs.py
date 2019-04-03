# noinspection PyUnresolvedReferences
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from geoimagenet_ml.store.datatypes import Process as ProcessDB                                         # noqa: F401
    from geoimagenet_ml.processes.runners import ProcessRunner                                              # noqa: F401
    # noinspection PyUnresolvedReferences
    from typing import Any, AnyStr, Callable, Dict, Iterable, List, Tuple, Optional, Union, Type            # noqa: F401
    # noinspection PyPackageRequirements
    from owslib.wps import WPSException                                                                     # noqa: F401
    from pywps import Process as ProcessWPS                                                                 # noqa: F401
    # noinspection PyProtectedMember
    from logging import _loggerClass                                                                        # noqa: F401
    from uuid import UUID as _UUID

    UUID = Union[AnyStr, _UUID]
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

    # ML impl/utils types
    # noinspection PyPackageRequirements
    from shapely.geometry.base import BaseGeometry as GeometryType
    PointType = Tuple[Number, Number]
    GeoTransformType = Tuple[Number, Number, Number, Number, Number, Number]
    FeatureType = Dict[{"geometry": GeometryType, "properties": Dict["taxonomy_class_id": int]}]
    RasterDataType = Dict[{
        "srs": GeometryType,
        "geotransform": GeoTransformType,
        "offset_geotransform": GeoTransformType,
        "extent": List[PointType],
        "skew": PointType,
        "resolution": PointType,
        "band_count": int,
        "cols": int,
        "rows": int,
        "data_type": int,  # enum
        "local_roi": GeometryType,
        "global_roi": GeometryType,
        "file_path": AnyStr,
    }]
