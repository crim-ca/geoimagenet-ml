from geoimagenet_ml.processes.status import map_status
from six.moves.configparser import ConfigParser
from datetime import datetime
# noinspection PyPackageRequirements
from dateutil.parser import parse
import collections
import time
import pytz
import types
import uuid
import six
import re
import os
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from geoimagenet_ml.typedefs import Any, AnyStr, Dict, List, Optional, Union, SettingsType  # noqa: F401


def get_base_url(settings):
    # type: (SettingsType) -> AnyStr
    return settings.get('geoimagenet_ml.api.url').rstrip('/').strip()


def get_any_id(info):
    # type: (Dict[AnyStr, AnyStr]) -> AnyStr
    """Retrieves a dictionary 'id'-like key using multiple common variations [id, identifier, _id].
    :param info: dictionary that potentially contains an 'id'-like key.
    :returns: value of the matched 'id'-like key."""
    return info.get('id', info.get('identifier', info.get('_id')))


def get_any_value(info):
    # type: (Dict[AnyStr, AnyStr]) -> Union[AnyStr, None]
    """Retrieves a dictionary 'value'-like key using multiple common variations [href, value, reference].
    :param info: dictionary that potentially contains a 'value'-like key.
    :returns: value of the matched 'id'-like key."""
    return info.get('href', info.get('value', info.get('reference', info.get('data'))))


def has_raw_value(info):
    # type: (Dict[AnyStr, AnyStr]) -> Union[bool, None]
    """Examines a dictionary of 'value'-like keys to determine if the value is directly accessible (True), or
    is specified as a reference (False). Invalid ``info`` (wrong format of not value/reference) returns ``None``.
    :param info: dictionary that potentially contains a 'value'-like key.
    :returns: indication of the presence of raw data."""
    if not isinstance(info, dict):
        return None
    if any([k in info for k in ['data', 'value']]):
        return True
    if any([k in info for k in ['href', 'reference']]):
        return False
    return None


def settings_from_ini(config_ini_file_path, ini_main_section_name):
    # type: (AnyStr, AnyStr) -> Dict[AnyStr, AnyStr]
    parser = ConfigParser()
    parser.read([config_ini_file_path])
    settings = dict(parser.items(ini_main_section_name))
    return settings


def str_2_path_list(str_list):
    # type: (Union[None, AnyStr]) -> List[AnyStr]
    """Obtains a list of existing and validated paths from a comma-separated string of potential paths."""
    if not isinstance(str_list, six.string_types) or not str_list:
        return []
    path_list = [p.strip() for p in str_list.split(',')]
    path_list = [p for p in path_list if os.path.isdir(p) or os.path.isfile(p)]


class null(object):
    """Represents a ``null`` value to differentiate from ``None`` when used as default value."""
    def __repr__(self):
        return '<Null>'


def isnull(item):
    # type: (Any) -> bool
    """Evaluates ``item`` for ``null`` type or instance."""
    return isinstance(item, null) or item is null


def islambda(func):
    # type: (Any) -> bool
    """Evaluates ``func`` for ``lambda`` type."""
    return isinstance(func, types.LambdaType) and func.__name__ == (lambda: None).__name__


def isclass(obj):
    # type: (Any) -> bool
    """Evaluates ``obj`` for ``class`` type (ie: class definition, not an instance nor any other type)."""
    return isinstance(obj, (type, six.class_types))


def is_uuid(item, version=4):
    # type: (Any, Optional[int]) -> bool
    """Evaluates if ``item`` is of type ``UUID``, or a string representing one."""
    if isinstance(item, uuid.UUID) and item.version == version:
        return True
    try:
        uuid_item = uuid.UUID(item, version=version)
    except ValueError:
        return False
    return str(uuid_item) == item


def get_sane_name(name, min_len=3, max_len=None, assert_invalid=True, replace_invalid=False):
    # type: (AnyStr, Optional[int], Optional[int], Optional[bool], Optional[bool]) -> Union[AnyStr, None]
    if assert_invalid:
        assert_sane_name(name, min_len, max_len)
    if name is None:
        return None
    name = name.strip()
    if len(name) < min_len:
        return None
    if replace_invalid:
        max_len = max_len or 25
        name = re.sub("[^a-z]", "_", name.lower()[:max_len])
    return name


def assert_sane_name(name, min_len=3, max_len=None):
    # type: (AnyStr, Optional[int], Optional[int]) -> None
    if name is None:
        raise ValueError('Invalid name : {0}'.format(name))
    name = name.strip()
    if '--' in name \
            or name.startswith('-') \
            or name.endswith('-') \
            or len(name) < min_len \
            or (max_len is not None and len(name) > max_len) \
            or not re.match(r"^[a-zA-Z0-9_\-]+$", name):
        raise ValueError('Invalid name : {0}'.format(name))


def fully_qualified_name(obj):
    # type: (Any) -> AnyStr
    return '.'.join([obj.__module__, type(obj).__name__])


def now():
    # type: (...) -> datetime
    return localize_datetime(datetime.utcnow())


def now_secs():
    # type: (...) -> int
    """
    Return the current time in seconds since the Epoch.
    """
    return int(time.time())


def expires_at(hours=1):
    return now_secs() + hours * 3600


def localize_datetime(dt, tz_name='UTC'):
    # type: (datetime, AnyStr) -> datetime
    """
    Provide a timezone-aware object for a given datetime and timezone name.
    """
    tz_aware_dt = dt
    if dt.tzinfo is None:
        utc = pytz.timezone('UTC')
        aware = utc.localize(dt)
        timezone = pytz.timezone(tz_name)
        tz_aware_dt = aware.astimezone(timezone)
    else:
        Warning('tzinfo already set')
    return tz_aware_dt


def stringify_datetime(dt=None, tz_name='UTC', fmt=None):
    # type: (Optional[datetime], AnyStr, Optional[AnyStr]) -> AnyStr
    """
    Obtain a localized and formatted datetime string from a datetime object.

    If ``fmt`` is provided as a format string, it is applied, otherwise applies `ISO-8601` by default.
    If ``dt`` is not provided, ``now()`` is employed.
    """
    dt_fmt = parse(str(localize_datetime(dt or now(), tz_name)))
    if fmt:
        return dt_fmt.strftime(fmt)
    return dt_fmt.isoformat()


def get_log_fmt():
    # type: (...) -> AnyStr
    return '[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s'


def get_log_datefmt():
    # type: (...) -> AnyStr
    return '%Y-%m-%d %H:%M:%S'


def get_job_log_msg(status, message, progress=0, duration=None):
    # type: (AnyStr, AnyStr, int, AnyStr) -> AnyStr
    return '{d} {p:3d}% {s:10} {m}'.format(
        d=duration or '',
        p=int(progress or 0),
        s=map_status(status).value,
        m=message
    )


def get_error_fmt():
    # type: (...) -> AnyStr
    return '{0.text} - code={0.code} - locator={0.locator}'


class classproperty(object):
    """
    Decorator allowing `@property`-like call to classes.

    Example ::

        >> class Foo(object):
               @classproperty
               def bar(cls):
                   return 'bar from Foo'

        >> Foo.bar
        'bar from Foo'

    """
    def __init__(self, getter):
        self.getter = getter

    def __get__(self, instance, owner):
        return self.getter(owner)


class ClassCounter(collections.Counter):
    """Counter with additional utility methods for class splitting."""
    def split(self, ratio):
        """Produces two :class:`ClassCounter` split by ``ratio``."""
        c1 = ClassCounter()
        c2 = ClassCounter()
        for c in self:
            c1[c] = int(self[c] * ratio)
            c2[c] = self[c] - c1[c]
        return c1, c2
