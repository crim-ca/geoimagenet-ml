

from geoimagenet_ml.store.interfaces import ProcessStore
from geoimagenet_ml.store.databases import models
from geoimagenet_ml.store import exceptions as ex

# TODO: postgres store implementations...


class PostgresProcessStore(ProcessStore):
    db = None

    def save_process(self, process, overwrite=True, request=None):
        try:
            process_name = process.identifier
        except Exception:
            raise ex.ProcessInstanceError()
        try:
            process_check = models.Process.by_process_name(process_name, db_session=self.db)
        except Exception:
            raise ex.ProcessNotFoundError()
        if process_check is not None:
            raise ex.ProcessConflictError()
        try:
            self.db.add(models.Process(name=process_name))
            new_process = models.Process.by_process_name(process_name, db_session=self.db)
        except Exception:
            raise ex.ProcessRegistrationError()
        return new_process

    def fetch_by_uuid(self, process_id, request=None):
        raise NotImplementedError

    def fetch_by_identifier(self, process_identifier, request=None):
        raise NotImplementedError

    def list_processes(self, request=None):
        raise NotImplementedError

    def delete_process(self, process_id, request=None):
        raise NotImplementedError
