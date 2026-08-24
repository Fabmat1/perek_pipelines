import numpy as np
from astropy.stats import mad_std
from multiprocessing import Pool
from scipy.optimize import curve_fit
from resample_backend import resample

# ---------------------------------------------------------------------------
# Sharing bulk data with Pool workers.
#
# Python 3.14 switched the default start method on Linux from "fork" to
# "forkserver", so workers no longer inherit the parent's memory copy-on-write.
# Anything placed in a task tuple is then pickled through a pipe once per task,
# which for whole 2048x2048 detector frames dominates the run time. Passing the
# frames as `initargs` instead sends them once per worker process.
# ---------------------------------------------------------------------------

_WORKER_DATA = {}


def _init_worker(payload):
    _WORKER_DATA.update(payload)


def shared_pool(payload, processes=None):
    """A Pool whose workers can reach `payload` (a dict) via `shared()`."""
    return Pool(processes=processes, initializer=_init_worker,
                initargs=(payload,))


def publish_shared(payload):
    """Make `payload` visible to `shared()` in the current process.

    Needed when the same worker function is called directly instead of through
    a `shared_pool` (e.g. the sequential DEBUG_PLOTS path)."""
    _init_worker(payload)


def shared(key):
    """Read a value published to the workers by `shared_pool`."""
    return _WORKER_DATA[key]

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
        # nothing to do, or nothing to interpolate from -- np.interp raises
        # on an empty set of sample points
        return y
    x = lambda z: z.nonzero()[0]
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])

    return y

# sort and mask the sections based on flux thresholds
def mask_section(section, tlo=0.05, thi=0.05, return_mask=False):
    """Drop the lowest `tlo` and highest `thi` fraction of `section`.

    Selection is by rank rather than by comparing against the values at those
    ranks: with ties (a flat section) a value comparison rejects every element
    at once, and with tlo=0 it still discarded the single lowest element.
    """
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
            if np.sum(mask) < 3:
                # too few points survived clipping: keep the previous fit.
                # (also happens for a near-exact fit, where mad_std == 0)
                return params, errs, mask
            else:
                xfit = x[mask]
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
