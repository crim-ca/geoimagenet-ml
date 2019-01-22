

from ccfb.api.store.interfaces import ProcessStore
from ccfb.api.store.databases import models
from ccfb.api.store import exceptions as ex

# TODO: postgres store implementations...


class PostgresProcessStore(ProcessStore):

    def save_process(self, process, overwrite=True, request=None):
        try:
            process_name = process.identifier
        except:
            raise ex.ProcessInstanceError()
        try:
            process_check = models.Process.by_process_name(process_name, db_session=request.db)
        except:
            raise ex.ProcessNotFoundError()
        if process_check is not None:
            raise ex.ProcessConflictError()
        try:
            request.db.add(models.Process(name=process_name))
            new_process = models.Process.by_process_name(process_name, db_session=request.db)
        except:
            raise ex.ProcessRegistrationError()
        return new_process
