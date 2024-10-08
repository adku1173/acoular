# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------

# imports from other packages
import gc
from os import listdir, path
from weakref import WeakValueDictionary

from traits.api import Bool, Delegate, Dict, HasPrivateTraits, Instance, Property, Str, Union

from .configuration import Config, config
from .h5files import H5CacheFileBase, _get_cachefile_class


class CacheInterface(HasPrivateTraits):

    #: flag to enable/disable caching
    cached = Bool(True, desc='flag to enable/disable caching')

    #: file object containing the cached data
    h5f = Instance(H5CacheFileBase, transient=True, desc='HDF5 file object')

    #: default name of the cache file (should be overwritten by subclasses)
    default_cache_name = Property(desc='default name of the cache file')

    #: default name of the cache node
    default_node_name = Property(desc='default name of the cache node')

    _default_cache_name = Union(Str, None, default_value=None)

    _default_node_name = Union(Str, None, default_value=None)

    def init_cache(self, **kwargs):
        msg = 'Method init_no_cache must be implemented by subclass'
        raise NotImplementedError(msg)
        return kwargs

    def init_no_cache(self, **kwargs):
        msg = 'Method init_no_cache must be implemented by subclass'
        raise NotImplementedError(msg)
        return kwargs

    def get_cache(self, **kwargs):
        msg = 'Method init_no_cache must be implemented by subclass'
        raise NotImplementedError(msg)
        return kwargs

    def is_cached(self, **kwargs): # noqa ARG002
        return self.h5f.is_cached(self.default_node_name)

    def remove_cache(self, **kwargs): # noqa ARG002
        if self.h5f:
            self.h5f.remove_data(self.default_node_name)

    def _get_default_cache_name(self):
        if self._default_cache_name:
            return self._default_cache_name
        return self.basename  + '_cache.h5'

    def _set_default_cache_name(self, name):
        self._default_cache_name = name

    def _get_default_node_name(self):
        if self._default_node_name:
            return self._default_node_name
        return self.__class__.__name__ + self.digest

    def _set_default_node_name(self, name):
        self._default_node_name = name


    def handle_cache(self, **kwargs):
        """Handles caching of data.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments to be passed to the methods responsible for
            allocation and cache handling.

        Returns
        -------
        data : array
            The data array.
        is_cached : bool
            Flag indicating whether the data is cached.

        """
        status = config.global_caching
        H5cache.get_cache_file(self, self.default_cache_name)
        if not self.h5f:
            return self.init_no_cache(**kwargs), False
        # remove old data before writing in overwrite mode
        is_cached = self.is_cached(**kwargs)
        if status == 'overwrite' and is_cached:
            self.remove_cache(**kwargs)
            is_cached = False
        if not is_cached and status != 'readonly':
            self.init_cache(**kwargs)
            return self.get_cache(**kwargs), is_cached
        if status == 'readonly':
            cache = self.get_cache(**kwargs)
            if isinstance(cache, tuple):
                # construct tuple with copied data
                return tuple(data[:] for data in cache), is_cached
            return cache[:], is_cached
        return self.get_cache(**kwargs), is_cached



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

    def get_cache_file(self, obj, filename, mode='a'):
        """Returns pytables .h5 file to h5f trait of calling object for caching."""
        if config.global_caching == 'none' or (
            config.global_caching == 'individual' and not obj.cached
        ):
            obj.h5f = None
            return

        self._idle_if_busy()  #
        self.busy = True

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


H5cache = HDF5Cache(config=config)


def get_basename(file):
    return path.basename(file.filename)
