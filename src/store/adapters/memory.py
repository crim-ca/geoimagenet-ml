"""
Read or write data from or to local memory.

Though not very valuable in a production setup, these store adapters are great
for testing purposes.
"""

from geoimagenet_ml.store.interfaces import ProcessStore
from geoimagenet_ml.api.utils import get_sane_name


class MemoryProcessStore(ProcessStore):
    """
    Stores WPS processes in memory. Useful for testing purposes.
    """

    def __init__(self, init_processes=None):
        self.name_index = {}
        if isinstance(init_processes, list):
            for process in init_processes:
                self.save_process(process)

    def save_process(self, process, overwrite=True, request=None):
        """
        Stores a WPS process in storage.
        """
        sane_name = get_sane_name(process.title)
        if not self.name_index.get(sane_name) or overwrite:
            process['title'] = sane_name
            self.name_index[sane_name] = process

    def delete_process(self, name, request=None):
        """
        Removes process from database.
        """
        sane_name = get_sane_name(name)
        if self.name_index.get(sane_name):
            del self.name_index[sane_name]

    def list_processes(self, request=None):
        """
        Lists all processes in database.
        """
        return [process.title for process in self.name_index]

    def fetch_by_uuid(self, process_uuid, request=None):
        """
        Get process for given ``uuid`` from storage.
        """
        sane_name = get_sane_name(process_uuid)
        process = self.name_index.get(sane_name)
        return process

    def fetch_by_identifier(self, process_id, request=None):
        """
        Get process for given ``identifier`` from storage.
        """
        sane_name = get_sane_name(process_id)
        process = filter(lambda p: self.name_index[p].identifier == sane_name, self.name_index)
        return process
