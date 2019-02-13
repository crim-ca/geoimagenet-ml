from geoimagenet_ml.utils import classproperty
from typing import AnyStr
from abc import abstractmethod


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
