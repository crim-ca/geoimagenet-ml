from geoimagenet_ml.utils import classproperty
from typing import TYPE_CHECKING
from abc import abstractmethod
if TYPE_CHECKING:
    from typing import AnyStr  # noqa: F401


class ProcessBase(object):
    @classproperty
    def identifier(self):
        # type: () -> AnyStr
        return self.__name__

    @classproperty
    @abstractmethod
    def type(self):
        # type: () -> AnyStr
        raise NotImplementedError

    @classproperty
    def limit_single_job(self):
        # type: () -> bool
        """
        Specifies whether multiple parallel process executions are
        allowed in for corresponding process identifiers.
        """
        return False
