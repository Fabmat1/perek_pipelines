import os
import atexit
import pickle
import struct
from contextlib import contextmanager
from multiprocessing import Pool, cpu_count
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from astropy.stats import mad_std
from scipy.optimize import curve_fit
from resample_backend import resample


# ---------------------------------------------------------------------------
# Worker pool.
#
# macOS and Windows start workers with "spawn": each one is a fresh
# interpreter that re-imports the whole pipeline (~1.3 s) before it can run
# anything.  On top of that, a Pool built with `initargs` larger than the
# ~64 kB pipe buffer starts its workers *one at a time*, because the parent
# blocks writing the payload until that child has finished importing and
# starts reading.  Opening one pool of 11 workers therefore cost 11 x 1.3 s,
# and the pipeline opened six of them per frame: ~84 s of startup around ~6 s
# of work, which is why the cores looked idle.  Linux hid all of it behind
# fork.
#
# So the pool is opened once and kept for the whole run, and payloads reach
# the workers through shared memory instead of the initializer.  Workers then
# start concurrently (nothing large goes through the pipe) and share one copy
# of the detector frames instead of holding one each.
# ---------------------------------------------------------------------------

_WORKER_DATA = {}

# Worker side: the payload block currently mapped, {name: (SharedMemory, dict)}.
_ATTACHED = {}
_ATTACHED_NAME = [None]

# Parent side: one pool per requested worker count, reused across phases.
_POOLS = {}

_ALIGN = 64


def _default_ncpu():
    """Every core, unless PEREK_NCPU says otherwise."""
    env = os.environ.get("PEREK_NCPU")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    try:
        return max(1, cpu_count())
    except NotImplementedError:
        return 1


# How many worker processes the pools may use.
_NCPU = _default_ncpu()


def set_ncpu(n):
    """Set the worker count for every pool the pipeline opens."""
    global _NCPU
    n = max(1, int(n))
    if n != _NCPU:
        close_pools()
    _NCPU = n
    os.environ["PEREK_NCPU"] = str(_NCPU)
    return _NCPU


def get_ncpu():
    return _NCPU


def _get_pool(processes):
    """The pool with `processes` workers, opened on first use and kept.

    `None` means "no pool": one worker is the caller itself, and going
    through a process for that only costs pickling."""
    if processes <= 1:
        return None
    pool = _POOLS.get(processes)
    if pool is None:
        pool = Pool(processes=processes)
        _POOLS[processes] = pool
    return pool


def close_pools():
    """Shut the workers down.  Only needed to change the worker count."""
    for pool in list(_POOLS.values()):
        try:
            pool.terminate()
            pool.join()
        except Exception:
            pass
    _POOLS.clear()


atexit.register(close_pools)


def _align(n):
    return -(-n // _ALIGN) * _ALIGN


class _ArrayRef(object):
    """Marks where an array was lifted out of the payload."""
    __slots__ = ("index",)

    def __init__(self, index):
        self.index = index

    def __getstate__(self):
        return self.index

    def __setstate__(self, index):
        self.index = index


def _strip_arrays(obj, arrays):
    """Replace every numeric array in `obj` by a reference into `arrays`."""
    if isinstance(obj, np.ndarray) and obj.dtype.kind in "biufc":
        # ascontiguousarray would turn a 0-d array into a 1-d one
        arrays.append(obj if obj.flags.c_contiguous
                      else np.ascontiguousarray(obj))
        return _ArrayRef(len(arrays) - 1)
    if type(obj) is dict:
        return dict((k, _strip_arrays(v, arrays)) for k, v in obj.items())
    if type(obj) is list:
        return [_strip_arrays(v, arrays) for v in obj]
    if type(obj) is tuple:
        return tuple(_strip_arrays(v, arrays) for v in obj)
    return obj


def _restore_arrays(obj, arrays):
    if isinstance(obj, _ArrayRef):
        return arrays[obj.index]
    if type(obj) is dict:
        return dict((k, _restore_arrays(v, arrays)) for k, v in obj.items())
    if type(obj) is list:
        return [_restore_arrays(v, arrays) for v in obj]
    if type(obj) is tuple:
        return tuple(_restore_arrays(v, arrays) for v in obj)
    return obj


def _publish_segment(payload):
    """Copy `payload` into a fresh shared-memory block and return it.

    Layout: the length of the pickled skeleton, the skeleton itself, then the
    raw array data, each aligned so numpy gets aligned views."""
    arrays = []
    skeleton = _strip_arrays(payload, arrays)
    specs = []
    total = 0
    for a in arrays:
        total = _align(total)
        specs.append((total, a.shape, a.dtype.str))
        total += a.nbytes
    header = pickle.dumps((skeleton, specs), protocol=pickle.HIGHEST_PROTOCOL)
    start = _align(8 + len(header))
    shm = SharedMemory(create=True, size=max(1, start + total))
    struct.pack_into("<Q", shm.buf, 0, len(header))
    shm.buf[8:8 + len(header)] = header
    for a, (off, shape, dtype) in zip(arrays, specs):
        dst = np.ndarray(shape, dtype=dtype, buffer=shm.buf, offset=start + off)
        dst[...] = a
        del dst
    return shm


def _release_segments():
    """Worker side: drop the block from the previous phase."""
    for name in list(_ATTACHED):
        shm, _ = _ATTACHED.pop(name)
        try:
            shm.close()
        except BufferError:
            # something still holds a view into it; it goes at worker exit
            pass
    _ATTACHED_NAME[0] = None


def _attach_segment(name):
    """Worker side: map the block `name` and rebuild the payload from it."""
    _release_segments()
    try:
        shm = SharedMemory(name=name, track=False)
    except TypeError:
        # track= is 3.13+.  Before that, merely attaching registers the block
        # with the resource tracker, which then fights the parent over who
        # unlinks it (the tracker keeps a set, so the second unregister is a
        # KeyError).  The parent owns the block; the worker just maps it.
        from multiprocessing import resource_tracker
        register = resource_tracker.register
        resource_tracker.register = lambda *a, **k: None
        try:
            shm = SharedMemory(name=name)
        finally:
            resource_tracker.register = register
    hlen, = struct.unpack_from("<Q", shm.buf, 0)
    skeleton, specs = pickle.loads(bytes(shm.buf[8:8 + hlen]))
    start = _align(8 + hlen)
    arrays = []
    for off, shape, dtype in specs:
        a = np.ndarray(shape, dtype=dtype, buffer=shm.buf, offset=start + off)
        a.flags.writeable = False   # the block is shared: nobody may write
        arrays.append(a)
    payload = _restore_arrays(skeleton, arrays)
    _ATTACHED[name] = (shm, payload)
    _ATTACHED_NAME[0] = name
    return payload


def _dispatch(task):
    """Worker side: make the payload reachable through `shared()`, then work."""
    name, func, arg = task
    if name is not None and name != _ATTACHED_NAME[0]:
        _WORKER_DATA.clear()
        _WORKER_DATA.update(_attach_segment(name))
    return func(arg)


class _SharedPool(object):
    """`map`/`imap` over the shared pool, with a payload behind `shared()`.

    Same three methods the pipeline used on a plain Pool, so call sites do
    not change; `None` for the pool means run in this process."""

    def __init__(self, name, pool):
        self._name = name
        self._pool = pool

    def _tasks(self, func, iterable):
        return [(self._name, func, arg) for arg in iterable]

    def map(self, func, iterable, chunksize=None):
        if self._pool is None:
            return [func(arg) for arg in iterable]
        return self._pool.map(_dispatch, self._tasks(func, iterable), chunksize)

    def imap(self, func, iterable, chunksize=1):
        if self._pool is None:
            return (func(arg) for arg in iterable)
        return self._pool.imap(_dispatch, self._tasks(func, iterable), chunksize)

    def imap_unordered(self, func, iterable, chunksize=1):
        if self._pool is None:
            return (func(arg) for arg in iterable)
        return self._pool.imap_unordered(_dispatch, self._tasks(func, iterable),
                                         chunksize)


@contextmanager
def shared_pool(payload, processes=None):
    """A pool whose workers can reach `payload` (a dict) via `shared()`."""
    if processes is None:
        processes = _NCPU
    processes = max(1, int(processes))
    pool = _get_pool(processes)
    if pool is None:
        publish_shared(payload)
        yield _SharedPool(None, None)
        return
    shm = _publish_segment(payload)
    try:
        yield _SharedPool(shm.name, pool)
    finally:
        shm.close()
        try:
            shm.unlink()
        except FileNotFoundError:
            pass


def publish_shared(payload):
    """Make `payload` visible to `shared()` in the current process."""
    _WORKER_DATA.update(payload)


_MISSING = object()


def shared(key, default=_MISSING):
    """Read a value published to the workers by `shared_pool`."""
    value = _WORKER_DATA.get(key, _MISSING)
    if value is _MISSING:
        if default is _MISSING:
            raise KeyError("no %r in the worker payload (published: %s)"
                           % (key, ", ".join(sorted(_WORKER_DATA)) or "nothing"))
        return default
    return value

def polynomial(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d

def Gaussian(x, A, mu=0, sigma=1):
    return A * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def Gaussian_res(x, A, mu=0, sigma=1):
    oversample = 10
    xfull = np.linspace(np.min(x), np.max(x), len(x)*oversample)
    yfull = Gaussian(xfull, A, mu=mu, sigma=sigma)
    return resample(x, xfull, yfull, fill=0, verbose=False)

def fill_nan(y):
    '''replace nan values in 1-array by interpolated values'''

    y = np.array(y, dtype=float)
    nans = np.isnan(y)
    if not nans.any() or nans.all():
        return y
    x = lambda z: z.nonzero()[0]
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])

    return y

# sort and mask the sections based on flux thresholds
def mask_section(section, tlo=0.05, thi=0.05, return_mask=False):
    """Drop the lowest `tlo` and highest `thi` fraction of `section`."""
    section = np.asarray(section)
    lsec = len(section)
    lo_idx = int(tlo * lsec)
    hi_idx = int((1 - thi) * lsec)
    mask = np.zeros(lsec, dtype=bool)
    if hi_idx > lo_idx:
        mask[np.argsort(section, kind="stable")[lo_idx:hi_idx]] = True
    if return_mask:
        return mask
    else:
        return section[mask]

def pair_generation(arr1, arr2, thres_max=5.5):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    distance_matrix = np.abs(arr1[:, np.newaxis] - arr2[np.newaxis, :])
    # -> shape = (len(arr1), len(arr2))

    pairs = []
    pair_dists = []
    for i, row in enumerate(distance_matrix):
        next_index = np.argmin(row)
        min_dist = np.min(distance_matrix[:, next_index])
        if min_dist == row[next_index]:
            pairs.append([i, int(next_index)])
            pair_dists.append(min_dist)
        else:
            pairs.append([i, None])
            pair_dists.append(np.nan)

    mean_dist = np.nanmean(pair_dists)
    std_dist = np.nanstd(pair_dists)

    for i in range(len(pairs)):
        if (pair_dists[i] > mean_dist+thres_max*std_dist):
            pairs[i][1] = None

    return pairs

def rolling_median_quant(x, y, window_size, p=0.6827):
    # x must be sorted
    x = np.array(x)
    y = np.array(y)

    half_window = window_size / 2

    # quantiles
    pl = 0. + 0.5 * (1. - p)
    ph = 1. - 0.5 * (1. - p)

    rolling_median = np.full_like(x, np.nan, dtype=np.float64)
    rolling_qlo = np.full_like(x, np.nan, dtype=np.float64)
    rolling_qhi = np.full_like(x, np.nan, dtype=np.float64)

    start = 0
    for i in range(len(x)):
        # move the start pointer to maintain the window
        while x[i] - x[start] > half_window:
            start += 1

        # only include points within the window
        y_in_window = y[start:i+1]

        if len(y_in_window) > 0:
            rolling_median[i] = np.median(y_in_window)
            rolling_qlo[i] = np.quantile(y_in_window, pl)
            rolling_qhi[i] = np.quantile(y_in_window, ph)

    return rolling_median, rolling_qlo, rolling_qhi

def polyfit_reject(x, y, deg=1, thres=2, nit=3):
    x = np.array(x)
    y = np.array(y)

    for i in range(nit+1):
        if i == 0:
            xfit = x
            yfit = y
        else:
            if np.sum(mask) >= deg + 1:
                xfit = x[mask]
                yfit = y[mask]
            else:
                return coef
        coef = np.polyfit(xfit, yfit, deg)

        poly1d_fn = np.poly1d(coef)
        ydiff = np.abs(poly1d_fn(x) - y)
        ystd = mad_std(ydiff)
        mask = ydiff < ystd * thres

    return coef

def curve_fit_reject(x, y, function, thres=2, thres_max=None, **kwargs):

    if np.isscalar(thres):
        thres = [thres] * 2

    nit = len(thres)

    x = np.array(x)
    y = np.array(y)

    for i in range(nit):
        if i == 0:
            xfit = x
            yfit = y
            kwargs_fit = kwargs.copy()
        else:
            try:
                nparam = function.__code__.co_argcount - 1
            except AttributeError:
                nparam = 3
            if np.sum(mask) < max(3, nparam):
                return params, errs, mask
            else:
                xfit = x[..., mask]
                yfit = y[mask]
                for key in kwargs:
                    if type(kwargs[key]) == np.ndarray and \
                       len(kwargs[key]) == len(mask):
                        kwargs_fit[key] = kwargs[key][mask]

        params, errs = curve_fit(function, xfit, yfit, **kwargs_fit)
        errs = np.sqrt(np.diag(errs))

        ymod = function(x, *params)
        ydiff = np.abs(ymod - y)
        ystd = mad_std(ydiff)
        if thres_max is not None:
            mask = ydiff < max(ystd * thres[i], thres_max)
        else:
            mask = ydiff < ystd * thres[i]

    return params, errs, mask
