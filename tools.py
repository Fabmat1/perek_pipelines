import numpy as np
from astropy.stats import mad_std
from scipy.optimize import curve_fit
try:
    from resample_spectres import resample
except ModuleNotFoundError:
    print("compile 'pyresample_spectres' like this:")
    print("python3 -m numpy.f2py -c -m pyresample_spectres resample_spectres.f90")
    try:
        from spectres import spectres as resample
    except ModuleNotFoundError:
        raise Exception("Need either 'spectres' or 'resample_spectres'")

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

    y = np.array(y)
    nans = np.isnan(y)
    x = lambda z: z.nonzero()[0]
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])

    return y

# sort and mask the sections based on flux thresholds
def mask_section(section, tlo=0.05, thi=0.05, return_mask=False):
    sorted_section = np.sort(section)
    lsec = len(section)
    mask = (section > sorted_section[int(tlo*lsec)]) & \
           (section < sorted_section[int((1-thi)*lsec)])
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
            if np.sum(mask) >= deg:
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

    if type(thres) == int:
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
                return params, errs
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
