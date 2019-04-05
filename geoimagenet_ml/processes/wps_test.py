from geoimagenet_ml.processes.base import ProcessBase
from geoimagenet_ml.utils import classproperty
from pywps import Process, LiteralInput, LiteralOutput
import os
import logging
LOGGER = logging.getLogger(__name__)


class HelloWPS(ProcessBase, Process):
    @classproperty
    def identifier(self):
        return 'hello'

    @classproperty
    def type(self):
        from geoimagenet_ml.processes.types import PROCESS_WPS
        return PROCESS_WPS

    def __init__(self):
        inputs = [
            LiteralInput('name', 'Your name', data_type='string')]
        outputs = [
            LiteralOutput('output', 'Output response',
                          data_type='string')]

        super(HelloWPS, self).__init__(
            self._handler,
            identifier=self.identifier,
            title='Says Hello',
            version='1.4',
            inputs=inputs,
            outputs=outputs,
            store_supported=True,
            status_supported=True
        )

    # noinspection PyMethodMayBeStatic
    def _handler(self, request, response):
        response.update_status("saying hello...", 0)
        LOGGER.debug("HOME=%s, Current Dir=%s", os.environ.get('HOME'), os.path.abspath(os.curdir))
        response.outputs['output'].file = 'Hello ' + request.inputs['name'][0].file
        return response
