# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------

# imports from other packages
import gc
from functools import wraps
from os import listdir, path
from weakref import WeakValueDictionary

import numpy as np
from traits.api import Bool, Delegate, Dict, HasPrivateTraits, Instance

from .configuration import Config, config
from .fbeamform import BeamformerAdaptiveGrid, BeamformerBase, BeamformerSODIX, PointSpreadFunction
from .h5files import _get_cachefile_class


def cached_file(func):
    """Decorator that handles cache loading and storing results in the cache."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if isinstance(self, BeamformerBase):
            H5cache._get_bf_filecache(self)
        return func(self, *args, **kwargs)
    return wrapper


class HDF5Cache(HasPrivateTraits):
    """Cache class that handles opening and closing 'tables.File' objects."""

    config = Instance(Config)

    cache_dir = Delegate('config')

    busy = Bool(False)

    open_files = WeakValueDictionary()

    open_file_reference = Dict()

    def _idle_if_busy(self):
        while self.busy:
            pass

    def open_cachefile(self, filename, mode):
        file = _get_cachefile_class()
        return file(path.join(self.cache_dir, filename), mode)

    def close_cachefile(self, cachefile):
        self.open_file_reference.pop(get_basename(cachefile))
        cachefile.close()

    def get_filename(self, file):
        file_class = _get_cachefile_class()
        if isinstance(file, file_class):
            return get_basename(file)
        return 0

    def get_open_cachefiles(self):
        try:
            return self.open_files.itervalues()
        except AttributeError:
            return iter(self.open_files.values())

    def close_unreferenced_cachefiles(self):
        for cachefile in self.get_open_cachefiles():
            if not self.is_reference_existent(cachefile):
                #                print("close unreferenced File:",get_basename(cachefile))
                self.close_cachefile(cachefile)

    def is_reference_existent(self, file):
        exist_flag = False
        # inspect all refererres to the file object
        gc.collect()  # clear garbage before collecting referrers
        for ref in gc.get_referrers(file):
            # does the file object have a referrer that has a 'h5f'
            # attribute?
            if isinstance(ref, dict) and 'h5f' in ref:
                # file is still referred, must not be closed
                exist_flag = True
                break
        return exist_flag

    def is_cachefile_existent(self, filename):
        if filename in listdir(self.cache_dir):
            return True
        return False

    def _increase_file_reference_counter(self, filename):
        self.open_file_reference[filename] = self.open_file_reference.get(filename, 0) + 1

    def _decrease_file_reference_counter(self, filename):
        self.open_file_reference[filename] = self.open_file_reference[filename] - 1

    def _print_open_files(self):
        print(list(self.open_file_reference.items()))

    def get_cache_file(self, obj, basename, mode='a'):
        """Returns pytables .h5 file to h5f trait of calling object for caching."""
        self._idle_if_busy()  #
        self.busy = True

        filename = basename + '_cache.h5'
        obj_filename = self.get_filename(obj.h5f)

        if obj_filename:
            if obj_filename == filename:
                self.busy = False
                return
            self._decrease_file_reference_counter(obj_filename)

        if filename not in self.open_files:  # or tables.file._open_files.filenames
            if config.global_caching == 'readonly' and not self.is_cachefile_existent(
                filename,
            ):  # condition ensures that cachefile is not created in readonly mode
                obj.h5f = None
                self.busy = False
                #                self._print_open_files()
                return
            if config.global_caching == 'readonly':
                mode = 'r'
            f = self.open_cachefile(filename, mode)
            self.open_files[filename] = f

        obj.h5f = self.open_files[filename]
        self._increase_file_reference_counter(filename)

        # garbage collection
        self.close_unreferenced_cachefiles()

        self.busy = False
        self._print_open_files()


    @staticmethod
    def _get_bf_filecache(bf):
        """Function collects cached results from file depending on
        global/local caching behaviour. Returns (None, None) if no cachefile/data
        exist and global caching mode is 'readonly'.
        """
        nodename = bf.__class__.__name__ + bf.digest

        if not (  # if result caching is active
            config.global_caching == 'none' or (config.global_caching == 'individual' and not bf.cached)
        ):
            H5cache.get_cache_file(bf, bf.freq_data.basename)
            if not bf.h5f or (config.global_caching == 'readonly' and not bf.h5f.is_cached(nodename)):
                bf._ac = None
                bf._fr = None
                return
            #        print("collect filecache for nodename:",nodename)
            if config.global_caching == 'overwrite' and bf.h5f.is_cached(nodename):
                #            print("remove existing data for nodename",nodename)
                bf.h5f.remove_data(nodename)  # remove old data before writing in overwrite mode

            if not bf.h5f.is_cached(nodename):
                #            print("no data existent for nodename:", nodename)
                numfreq = bf.freq_data.fftfreq().shape[0]  # block_size/2 + 1steer_obj
                group = bf.h5f.create_new_group(nodename)
                bf.h5f.create_compressible_array(
                    'freqs',
                    (numfreq,),
                    'int8',  #'bool',
                    group,
                )
                if isinstance(bf, BeamformerAdaptiveGrid):
                    bf.h5f.create_compressible_array('gpos', (3, bf.size), 'float64', group)
                    bf.h5f.create_compressible_array('result', (numfreq, bf.size), bf.precision, group)
                elif isinstance(bf, BeamformerSODIX):
                    bf.h5f.create_compressible_array(
                        'result',
                        (numfreq, bf.steer.grid.size * bf.steer.mics.num_mics),
                        bf.precision,
                        group,
                    )
                else:
                    bf.h5f.create_compressible_array('result', (numfreq, bf.steer.grid.size), bf.precision, group)

            ac = bf.h5f.get_data_by_reference('result', '/' + nodename)
            fr = bf.h5f.get_data_by_reference('freqs', '/' + nodename)
            gpos = bf.h5f.get_data_by_reference('gpos', '/' + nodename) if isinstance(bf, BeamformerAdaptiveGrid) else None
            if gpos:
                bf._gpos = gpos
        if (ac and fr) and config.global_caching == 'readonly':
            (ac, fr) = (ac[:], fr[:])  # so never write back to disk
        bf._ac = ac
        bf._fr = fr
        return



H5cache = HDF5Cache(config=config)

class CacheObject(HasPrivateTraits):

    cached = Bool(True)

    # hdf5 cache file
    h5f = Instance(H5CacheFileBase, transient=True) # TODO: make this private?

    _cache_filename = Property()

    _nodename = Property()

    def _get__nodename(self):
        if isinstance(self, PointSpreadFunction):
            return ('Hz_%.2f' % self.freq).replace('.', '_')
        return self.__class__.__name__ + self.digest

    def _get__cache_filename(self):
        if isinstance(self, BeamformerBase):
            return self.freq_data.basename
        if isinstance(self, PointSpreadFunction):
            return 'psf' + self.digest
        return self.__class__.__name__ + self.digest

    def is_cached(self):
        return self.h5f.is_cached(self._nodename)

    def _remove_cache(self):
        if self.h5f:
            self.h5f.remove_data(self._nodename)

    def _init_cache(self):
        # must be overwritten by subclasses
        group = self.h5f.create_new_group(self.nodename)
        if isinstance(self, BeamformerBase):
            numfreq = self.freq_data.fftfreq().shape[0]  # block_size/2 + 1steer_obj
            self.h5f.create_compressible_array(
                'freqs',(numfreq,),'int8', group)
            self.h5f.create_compressible_array(
                'result', (numfreq, self.steer.grid.size), self.precision, group)
        elif isinstance(self, PointSpreadFunction):
            gs = self.steer.grid.size
            group = self.h5f.create_new_group(self.nodename)
            self.h5f.create_compressible_array('result', (gs, gs), self.precision, group)
            self.h5f.create_compressible_array(
                'gridpts', (gs,), 'int8', group)

    def _no_cache(self):
        if isinstance(self, (BeamformerBase, PointSpreadFunction)):
            ac = np.zeros((self._numfreq, self.steer.grid.size), dtype=self.precision)
            fr = np.zeros(self._numfreq, dtype='int8')
            return (ac, fr)
        return None

    def _get_cache(self):
        if isinstance(self, BeamformerBase):
            ac = self.h5f.get_data_by_reference('result', '/' + self.nodename)
            fr = self.h5f.get_data_by_reference('freqs', '/' + self.nodename)
            if self.global_caching == 'readonly':
                return (ac[:], fr[:])
            return (ac, fr)
        return None

    def handle_cache(self):
        status = config.global_caching
        if not (  # if result caching is active
            status == 'none' or  (status == 'individual' and \
                     not self.cached)
        ):
            H5cache.get_cache_file(self, self._cache_filename)
            if not self.h5f or (status == 'readonly' \
                and not self.is_cached):
                return self._no_cache()

        if status == 'overwrite' and self.is_cached:
            self._remove_cache()  # remove old data before writing in overwrite mode

        if not self.is_cached and status != 'readonly':
            self._init_cache()
        return self._get_cache()


def get_basename(file):
    return path.basename(file.filename)
