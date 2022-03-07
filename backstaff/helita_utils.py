import gc
import psutil
import pathlib
import logging
import numpy as np
from helita.sim.bifrost import BifrostData


class BifrostDataCache:
    def __init__(self, logger=logging):
        self.snaps = []
        self.fields = {}
        self._logger = logger

    def number_of_snaps(self):
        return len(self.snaps)

    def has_snap(self, snap):
        return snap in self.snaps

    def has_var(self, snap, var):
        return self.has_snap(snap) and var in self.fields[snap]

    def cache_field(self, snap, var, field):
        while not self.field_fits_in_memory(field):
            if self.number_of_snaps() == 0:
                self._logger.debug(f'No more snaps to remove, using memmap')
                return field
            else:
                removed_snap = self.snaps.pop(0)
                self._logger.debug(f'Removed snap {removed_snap} from cache')
                self.fields.pop(removed_snap)
                gc.collect()

        if not self.has_snap(snap):
            self.snaps.append(snap)
            self.fields[snap] = {}

        self._logger.debug(f'Cached {var} for snap {snap}')
        self.fields[snap][var] = np.array(field)
        return self.get_cached_field(snap, var)

    def get_cached_field(self, snap, var):
        self._logger.debug(f'Found {var} for snap {snap} in cache')
        return self.fields[snap][var]

    def field_fits_in_memory(self, field, buffer_factor=2):
        memory_requirement = field.size * field.dtype.itemsize
        available_memory = psutil.virtual_memory().available
        self._logger.debug(
            f'Required memory: {memory_requirement*1e-9:.2f} GB ({available_memory*1e-9:.2f} GB available)'
        )
        return buffer_factor * memory_requirement < available_memory


class CachingBifrostData(BifrostData):
    def __init__(self, *args, logger=logging, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = BifrostDataCache(logger=logger)
        self.root_name = pathlib.Path(self.file_root).name

    def get_var(self, var, *args, snap=None, **kwargs):
        active_snap = self.snap if snap is None else snap
        if self._cache.has_var(active_snap, var):
            return self._cache.get_cached_field(active_snap, var)
        else:
            return self._cache.cache_field(
                active_snap, var,
                super().get_var(var, *args, snap=snap, **kwargs))
