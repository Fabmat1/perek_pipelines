import numpy as np
from astropy.stats import mad_std

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
