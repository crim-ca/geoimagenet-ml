# noinspection PyUnresolvedReferences
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from geoimagenet_ml.store.datatypes import Process as ProcessDB
    from geoimagenet_ml.processes.status import STATUS
    from geoimagenet_ml.processes.runners import ProcessRunner
    from pyramid.config import Configurator
    from pyramid.registry import Registry
    from pyramid.request import Request
    from pyramid_celery import Celery
    # noinspection PyUnresolvedReferences
    from typing import Any, AnyStr, Callable, Dict, Iterable, List, Tuple, Optional, Union, Type            # noqa: F401
    # noinspection PyPackageRequirements
    from owslib.wps import WPSException                                                                     # noqa: F401
    from pywps import Process as ProcessWPS                                                                 # noqa: F401
    # noinspection PyProtectedMember
    from logging import _loggerClass                                                                        # noqa: F401
    from uuid import UUID as _UUID

    UUID = Union[AnyStr, _UUID]
    AnyKey = Union[AnyStr, int]
    AnyProcess = Union[ProcessDB, ProcessWPS, ProcessRunner]
    Number = Union[float, int]
    JsonValue = Union[AnyStr, Number, bool, None]
    JSON = Dict[AnyKey, Union[JsonValue, List["JSON"], Dict[AnyKey, "JSON"]]]
    OptionType = Dict[AnyStr, Any]
    InputType = Dict[AnyStr, JSON]
    OutputType = Dict[AnyStr, JSON]
    LoggerType = _loggerClass
    ErrorType = Union[AnyStr, Exception, List[WPSException]]
    LevelType = Union[AnyStr, int]
    AnyStatus = Union[STATUS, int, AnyStr]

    AnyContainer = Union[Configurator, Registry, Request, Celery]
    SettingValue = Union[AnyStr, Number, bool, None]
    SettingsType = Dict[AnyStr, SettingValue]
    AnySettingsContainer = Union[AnyContainer, SettingsType]
    AnyRegistryContainer = AnyContainer

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
