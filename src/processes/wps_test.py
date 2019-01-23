from pywps import Process, LiteralInput, LiteralOutput
import os
import logging
LOGGER = logging.getLogger(__name__)


class HelloWPS(Process):
    def __init__(self):
        inputs = [
            LiteralInput('name', 'Your name', data_type='string')]
        outputs = [
            LiteralOutput('output', 'Output response',
                          data_type='string')]

        super(HelloWPS, self).__init__(
            self._handler,
            identifier='hello',
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
