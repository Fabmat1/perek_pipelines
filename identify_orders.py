import os

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from scipy.ndimage import (minimum_filter, maximum_filter)

from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from tools import (mask_section, Gaussian, fill_nan, pair_generation,
                   polynomial, shared_pool, shared)
from orders import SpectralOrder

two_log_two = 2 * np.sqrt(2 * np.log(2))

MIN_WINDOW_DEFAULT = 15

class SpectralSlice:
    def __init__(self, x, ys, y_errs, widths):
        self.x = x
        self.ys = ys
        self.y_errs = y_errs
        self.widths = widths
        self.order_ownership = None
        self.next_slice = None
        self.previous_slice = None

    def gen_empty_ownership(self):
        self.order_ownership = np.full(len(self.ys), None)

    def clean_ownership(self):
        n = len(self.order_ownership)
        if all(o is None for o in self.order_ownership):
            raise ValueError("slice at x=%s has no order assignments to "
                             "extrapolate from" % self.x)
        for i, owner in enumerate(self.order_ownership):
            if owner is None:
                if i == 0:
                    j = 1
                    while (i + j) < n and self.order_ownership[i + j] is None:
                        j += 1
                    self.order_ownership[i] = self.order_ownership[i + j] - j
                else:
                    self.order_ownership[i] = self.order_ownership[i - 1] + 1

def slice_analysis(pixel, slice_x, slice_y, MIN_WINDOW=MIN_WINDOW_DEFAULT, MAX_WINDOW=15, NOISE_MEASURE_SECTION_WIDTH=0.05,
                   NOISE_CUTOFF=20, CUTTOFF_MARGIN=5, ORDER_GAUSS_THRESHOLD=0.6,
                   idx_peak_min=700, idx_peak_max=1750, DEBUG_PLOTS=False):
    """Locate the orders in one cross-dispersion cut of the frame."""
    if len(slice_x) != len(slice_y):
        raise ValueError("slice_x and slice_y must have the same length, got "
                         "%d and %d" % (len(slice_x), len(slice_y)))


    # remove "bias"
    bias_lvl = minimum_filter(slice_y, size=MIN_WINDOW)
    slice_y -= bias_lvl

    ny = len(slice_y)
    NOISE_WIDTH_IDX = int(ny * NOISE_MEASURE_SECTION_WIDTH)

    # select noise at top and bottom of the slice
    upper_section = slice_y[:NOISE_WIDTH_IDX]
    lower_section = slice_y[-NOISE_WIDTH_IDX:]

    upper_section = mask_section(upper_section, tlo=0.05, thi=0.15)
    lower_section = mask_section(lower_section, tlo=0.05, thi=0.15)

    # remove "bias"
    bias_lvl = (np.median(lower_section) + np.median(upper_section)) / 2
    slice_y -= bias_lvl
    lower_section -= bias_lvl
    upper_section -= bias_lvl

    noise_lvl = (np.std(lower_section) + np.std(upper_section)) / 2
    noise_lvl *= NOISE_CUTOFF

    noise_indices = np.where(slice_y > noise_lvl)[0]
    if len(noise_indices) < 2:
        raise ValueError("no orders stand out above the noise in the slice at "
                         "x=%s (%d pixels above cutoff)"
                         % (pixel, len(noise_indices)))

    def group_consecutive(arr):
        diffs = np.diff(arr)
        # identify the points where the difference is not 1, meaning the sequence breaks
        break_points = np.where(diffs != 1)[0] + 1
        # return indices to split the array at the break points
        isplit = np.split(np.arange(len(arr)), break_points)
        return isplit

    isplit_groups = group_consecutive(noise_indices)
    groups_ipeak = [noise_indices[i] for i in isplit_groups]
    groups_ipeak = [np.mean(i) for i in groups_ipeak]

    # estimate distances between orders
    def min_adjacent_distance(a):
        adjacent_diff = np.abs(np.diff(a))
        min_distances = np.zeros(len(a))
        min_distances[0] = adjacent_diff[0]
        min_distances[-1] = adjacent_diff[-1]
        min_distances[1:-1] = np.minimum(adjacent_diff[:-1], adjacent_diff[1:])
        return min_distances

    ipeak_dist = min_adjacent_distance(groups_ipeak)

    # remove outliers for standard deviation
    ipeak_dist_cut = mask_section(ipeak_dist, tlo=0, thi=0.1)
    median_dist = np.median(ipeak_dist_cut)
    std_dist = np.std(ipeak_dist_cut)

    # remove peaks that are isolated to more than 10 sigma
    thres_dist = median_dist + std_dist * 10
#    MAX_WINDOW = max(MAX_WINDOW, median_dist + std_dist * 4)

    # remove pixels that belong to a 'bad' group
    igroup_bad = np.where(ipeak_dist > thres_dist)[0]
    if DEBUG_PLOTS:
        print("bad groups:", igroup_bad)
    if (len(igroup_bad) > 0):
        isplit_groups = np.array(isplit_groups, dtype=object)
        ibad = np.concatenate(isplit_groups[igroup_bad])
        mask_good = np.ones(len(noise_indices)).astype(bool)
        mask_good[ibad] = False
        noise_indices = noise_indices[mask_good]

    # disabled
    if DEBUG_PLOTS and False:
        plt.xlabel("group distance  /  pix")
        plt.hist(np.diff(groups_ipeak), bins=15, range=(0,50))
        plt.axvline(median_dist)
        plt.axvline(median_dist+10*std_dist)
        plt.show()

    # assume that "real" orders only start at pixles > 'idx_peak_min'
    noise_indices = noise_indices[noise_indices>idx_peak_min]
    noise_indices = noise_indices[noise_indices<idx_peak_max]
    n_cross = 2
    if len(noise_indices) < 2 * n_cross + 1:
        raise ValueError(
            "only %d pixels above the noise cutoff in rows %d-%d of the slice "
            "at x=%s; adjust idx_peak_min/idx_peak_max for this detector"
            % (len(noise_indices), idx_peak_min, idx_peak_max, pixel))
    first_cross = noise_indices[n_cross] - n_cross
    last_cross = noise_indices[-1-n_cross] + n_cross

    lo_ind = first_cross - CUTTOFF_MARGIN
    hi_ind = last_cross + CUTTOFF_MARGIN

    if (DEBUG_PLOTS):
        plt.title("Order slice for X pixel %d" % pixel)
        plt.plot(slice_x, slice_y, label="X slice")
        plt.axhline(noise_lvl, color="orange", label="noise level")
        plt.axvline(lo_ind, color='r')
        plt.axvline(hi_ind, color='r')
        max_slice = maximum_filter(slice_y, size=MAX_WINDOW)
        max_slice = np.clip(max_slice, a_min=noise_lvl*2, a_max=np.inf)
        plt.plot(slice_x, max_slice, color="tab:green", label="max filter")
        plt.xlabel("Y pixel")
        plt.ylabel("Counts")
        plt.legend()
        plt.tight_layout()
        plt.show()

    slice_x = slice_x[lo_ind:hi_ind]
    slice_y = slice_y[lo_ind:hi_ind]

    max_slice = maximum_filter(slice_y, size=MAX_WINDOW)
    max_slice = np.clip(max_slice, a_min=noise_lvl*2, a_max=np.inf)
    slice_y /= max_slice

    filtered_indices = slice_x[slice_y > ORDER_GAUSS_THRESHOLD]
    peaks = np.split(filtered_indices, np.where(np.diff(filtered_indices) != 1)[0] + 1)
    peaks = [peak for peak in peaks if len(peak) > 1]

    peak_locations = [np.mean(p) for p in peaks]

    if len(peak_locations) > 1:
        xdist_med = np.median(np.diff(peak_locations))
        # filter peaks closer than xdist_med/3
        filtered_peak_locations = [peak_locations[0]]
        for loc in peak_locations[1:]:
            if loc - filtered_peak_locations[-1] >= xdist_med / 3:
                filtered_peak_locations.append(loc)
    else:
        filtered_peak_locations = peak_locations
    peak_locations = filtered_peak_locations

#    print(f"Identified {len(peak_locations)} orders @ x = {pixel}")

    if DEBUG_PLOTS and False:
        plt.title("Order slice for X pixel %d" % pixel)
        for idx, l in enumerate(peak_locations):
            plt.axvline(l, ymin=0, ymax=0.93, color="gray")
            plt.text(s=str(idx), x=l, y=1.1, rotation=90,
                     va="center", ha="center")
        plt.axhline(ORDER_GAUSS_THRESHOLD, color="orange")
        plt.plot(slice_x, slice_y)
        plt.ylim(-0.05, 1.15)
        plt.xlabel("Y pixel")
        plt.ylabel("Re-normalised counts")
        plt.tight_layout()
        plt.show()

    fit_params = []
    med_peak_distance = np.median(np.diff(peak_locations))

    refined_peak_locations = []
    location_uncertainties = []
    widths = []
    for i, peak_location in enumerate(peak_locations):
        if i == 0:
            mask = slice_x < peak_location + (peak_locations[i + 1] - peak_location) / 2
            bounds = [[0, 0, 0], [np.inf, peak_locations[i + 1], med_peak_distance / 2]]
        elif i == len(peak_locations) - 1:
            mask = slice_x > peak_location - (peak_location - peak_locations[i - 1]) / 2
            bounds = [[0, peak_locations[i - 1], 0], [np.inf, np.max(slice_x), med_peak_distance / 2]]
        else:
            mask = np.logical_and(slice_x < peak_location + (peak_locations[i + 1] - peak_location) / 2,
                                  slice_x > peak_location - (peak_location - peak_locations[i - 1]) / 2)
            bounds = [[0, peak_locations[i - 1], 0], [np.inf, peak_locations[i + 1], med_peak_distance / 2]]
        x_neighborhood = slice_x[mask]
        y_neighborhood = slice_y[mask]

        params, errs = curve_fit(Gaussian, x_neighborhood, y_neighborhood,
                                 [1, peak_location, med_peak_distance / 4],
                                 bounds=bounds,
                                 maxfev=100000)

        if DEBUG_PLOTS:
            plt.text(s=str(i), x=peak_location, y=1.1, rotation=90,
                     va="center", ha="center")
            plt.plot(x_neighborhood, Gaussian(x_neighborhood, *params),
                     color="r")
        fit_params.append(params)

        errs = np.sqrt(np.diag(errs))

        refined_peak_locations.append(params[1])
        location_uncertainties.append(errs[1])
        widths.append(two_log_two * params[2])

    if DEBUG_PLOTS:
        plt.title("X pixel %d" % pixel)
        plt.plot(slice_x, slice_y)
        plt.ylim(-0.05, 1.15)
        plt.xlabel("Y pixel")
        plt.ylabel("Re-normalised counts")
        plt.tight_layout()
        plt.show()

    return SpectralSlice(pixel,
                         np.array(refined_peak_locations),
                         np.array(location_uncertainties),
                         np.array(widths))

def process_slice(args):
    i, pixel, npix_x, DEBUG_PLOTS = args
    frame_for_slice = shared("frame_for_slice")
    xidx = np.arange(3) + pixel - 1
    xidx = xidx[(xidx>=0) & (xidx<npix_x)]
    slice_y = np.sum(frame_for_slice[:, xidx].astype(float), axis=1) / len(xidx)
    # slice_y runs along the cross-dispersion axis, so slice_x indexes rows
    slice_x = np.arange(frame_for_slice.shape[0])
    debug_slice = (i == 0) and DEBUG_PLOTS
    slice = slice_analysis(pixel, slice_x, slice_y, DEBUG_PLOTS=debug_slice)
    return slice

FALLBACK_WINDOWS = {"blue": (770, 800), "red": (1620, 1670),
                    "noise": (100, 600), "cols": (900, 1150)}


def column_profile(frame, bias=None, cols=None, col_frac=0.12):
    """Cross-dispersion profile of one frame, background removed."""
    frame = np.asarray(frame, dtype=float)
    if bias is not None:
        frame = frame - np.asarray(bias, dtype=float)
    ny, nx = frame.shape
    if cols is None:
        half = max(1, int(0.5 * col_frac * nx))
        cols = (max(0, nx // 2 - half), min(nx, nx // 2 + half))
    profile = frame[:, cols[0]:cols[1]].mean(axis=1)
    return profile - minimum_filter(profile, MIN_WINDOW_DEFAULT)


def trace_windows(profiles, flat_profile=None, verbose=False):
    """Where on the detector to measure blue signal, red signal and noise."""
    if not profiles:
        return dict(FALLBACK_WINDOWS), False
    ref = np.max(np.vstack(profiles), axis=0)
    ny = len(ref)

    med = np.median(ref)
    mad = 1.4826 * np.median(np.abs(ref - med))
    peak = float(np.max(ref) - med)
    if not np.isfinite(peak) or peak <= 0:
        return dict(FALLBACK_WINDOWS), False
    if np.isfinite(mad) and mad > 0:
        threshold = 5 * mad
    else:
        threshold = 0.01 * peak
    lit = np.where(ref - med > threshold)[0]
    if len(lit) < 20:
        return dict(FALLBACK_WINDOWS), False

    # robust extent, so one hot pixel outside the orders cannot stretch it
    lo = int(np.percentile(lit, 0.5))
    hi = int(np.percentile(lit, 99.5))
    span = hi - lo
    if span < 50:
        return dict(FALLBACK_WINDOWS), False

    # width of an end window: a couple of order spacings
    width = int(max(20, round(0.035 * span)))
    end_a = (lo, min(hi, lo + width))
    end_b = (max(lo, hi - width), hi)

    blue, red = end_a, end_b
    if flat_profile is not None and len(flat_profile) == ny:
        fa = float(np.median(flat_profile[end_a[0]:end_a[1]]))
        fb = float(np.median(flat_profile[end_b[0]:end_b[1]]))
        if np.isfinite(fa) and np.isfinite(fb) and fa > fb:
            blue, red = end_b, end_a

    # noise from the largest order-free block, above or below the orders
    below, above = lo, ny - hi
    if max(below, above) < 50:
        return dict(FALLBACK_WINDOWS), False
    if below >= above:
        noise = (int(0.15 * below), int(0.85 * below))
    else:
        noise = (int(hi + 0.15 * above), int(hi + 0.85 * above))

    if verbose:
        print("- order block spans rows %d-%d; blue %d-%d, red %d-%d, "
              "noise %d-%d" % (lo, hi, blue[0], blue[1], red[0], red[1],
                               noise[0], noise[1]))
    return {"blue": blue, "red": red, "noise": noise, "cols": None}, True


def window_signal(profile, rows, noise_rows):
    """Peak of `profile` in `rows`, in units of its scatter in `noise_rows`."""
    ny = len(profile)
    if rows[1] > ny or noise_rows[1] > ny or rows[0] >= rows[1] \
            or noise_rows[0] >= noise_rows[1]:
        return -np.inf
    noise = np.std(profile[noise_rows[0]:noise_rows[1]])
    if not np.isfinite(noise) or noise <= 0:
        return -np.inf
    peak = np.max(profile[rows[0]:rows[1]])
    if not np.isfinite(peak):
        return -np.inf
    return float(peak / noise)


def blue_order_signal(frame, bias=None, row_lo=None, row_hi=None,
                      col_lo=None, col_hi=None, noise_lo=None, noise_hi=None):
    """How well the bluest orders stand out in one frame, in units of noise."""
    w = FALLBACK_WINDOWS
    rows = (w["blue"][0] if row_lo is None else row_lo,
            w["blue"][1] if row_hi is None else row_hi)
    noise_rows = (w["noise"][0] if noise_lo is None else noise_lo,
                  w["noise"][1] if noise_hi is None else noise_hi)
    cols = (w["cols"][0] if col_lo is None else col_lo,
            w["cols"][1] if col_hi is None else col_hi)
    frame = np.asarray(frame, dtype=float)
    if frame.ndim != 2 or cols[1] > frame.shape[1]:
        return -np.inf
    return window_signal(column_profile(frame, bias=bias, cols=cols),
                         rows, noise_rows)


def select_trace_frames(frames, bias=None, nstack=4, verbose=False,
                        flat=None):
    """Pick the frames to trace the orders on."""
    profiles, kept = [], []
    for f in frames:
        arr = f
        if isinstance(f, str):
            try:
                with fits.open(f) as hdul:
                    arr = np.asarray(hdul[0].data)
            except Exception:
                continue
        arr = np.asarray(arr)
        if arr.ndim != 2:
            continue
        try:
            profiles.append(column_profile(arr, bias=bias))
        except Exception:
            continue
        kept.append(f)

    if not profiles:
        if verbose:
            print("- no frame could be scored; tracing on the flat")
        return []

    flat_profile = None
    if flat is not None:
        try:
            flat_profile = column_profile(flat, bias=bias)
        except Exception:
            flat_profile = None

    win, measured = trace_windows(profiles, flat_profile=flat_profile,
                                  verbose=verbose)
    if verbose and not measured:
        print("- could not locate the orders; using the built-in "
              "cross-dispersion windows")

    scored = []
    for prof, f in zip(profiles, kept):
        blue = window_signal(prof, win["blue"], win["noise"])
        red = window_signal(prof, win["red"], win["noise"])
        if np.isfinite(blue):
            scored.append((blue, red, f))

    if not scored:
        if verbose:
            print("- no frame could be scored; tracing on the flat")
        return []

    by_blue = sorted(scored, key=lambda t: t[0], reverse=True)
    chosen = [by_blue[0]]
    for cand in sorted(scored, key=lambda t: t[1], reverse=True):
        if len(chosen) >= nstack:
            break
        if cand[2] not in [c[2] for c in chosen]:
            chosen.append(cand)

    if verbose:
        print("- tracing orders on %d frame(s):" % len(chosen))
        for i, (blue, red, f) in enumerate(chosen):
            nm = os.path.basename(f) if isinstance(f, str) else "frame"
            print("    %s %-20s blue %6.0f sigma, red %6.0f sigma"
                  % ("bluest:" if i == 0 else "       ", nm, blue, red))

    return [f for _, _, f in chosen]


def find_slices(frame_for_slice, sampling=200, DEBUG_PLOTS=False):
    # Get orders and stuff from flat
    npix_x = frame_for_slice.shape[1]
    pixels = np.linspace(5, npix_x-5, sampling).astype(int)
    args_list = [(i, pixels[i], npix_x, DEBUG_PLOTS) for i in range(sampling)]
    with shared_pool({"frame_for_slice": frame_for_slice}) as pool:
        slices = list(tqdm(pool.imap(process_slice, args_list), total=sampling))

    if DEBUG_PLOTS and False:
        plt.imshow(frame_for_slice, zorder=1, cmap='gray', norm="log")
        for slice in slices:
            plt.scatter([np.repeat(slice.x, len(slice.ys))], slice.ys, marker="x", zorder=2)
        # allow stretching
        plt.gca().set_aspect('auto')
        plt.xlabel("X pixel")
        plt.ylabel("Y pixel")
        plt.tight_layout()
        plt.show()

    for i, slice in enumerate(slices):
        if i == 0:
            slice.next_slice = slices[1]
        elif i == len(slices) - 1:
            slice.previous_slice = slices[-2]
        else:
            slice.previous_slice = slices[i - 1]
            slice.next_slice = slices[i + 1]

    return slices


def assign_orders(slicelist: list[SpectralSlice], max_ind, DEBUG_PLOTS=False):

    # Forward loop
    slicelist[max_ind].order_ownership = np.arange(len(slicelist[max_ind].ys)) + 1

    curr_slice = slicelist[max_ind]

    while curr_slice.next_slice is not None:
        curr_slice.clean_ownership()

        # a slice (or column) is defined by a fixed x pixel and a number of y pixels
        pair_idx = pair_generation(curr_slice.ys, curr_slice.next_slice.ys)

        curr_slice.next_slice.gen_empty_ownership()

        for p in pair_idx:
            if p[1] is None:
                continue
            curr_slice.next_slice.order_ownership[p[1]] = curr_slice.order_ownership[p[0]]
            if DEBUG_PLOTS:
                plt.plot([curr_slice.x, curr_slice.next_slice.x],
                         [curr_slice.ys[p[0]], curr_slice.next_slice.ys[p[1]]], color="red")

        curr_slice = curr_slice.next_slice
    else:
        curr_slice.clean_ownership()

    curr_slice = slicelist[max_ind]

    while curr_slice.previous_slice is not None:
        curr_slice.clean_ownership()

        pair_idx = pair_generation(curr_slice.ys, curr_slice.previous_slice.ys)
        curr_slice.previous_slice.gen_empty_ownership()

        for p in pair_idx:
            if p[1] is None:
                continue
            curr_slice.previous_slice.order_ownership[p[1]] = curr_slice.order_ownership[p[0]]
            if DEBUG_PLOTS:
                plt.plot([curr_slice.x, curr_slice.previous_slice.x],
                         [curr_slice.ys[p[0]], curr_slice.previous_slice.ys[p[1]]],
                         color="blue", ls="--")

        curr_slice = curr_slice.previous_slice
    else:
        curr_slice.clean_ownership()

    if DEBUG_PLOTS:
        for s in slicelist:
            plt.scatter(s.x*np.ones(len(s.ys)), s.ys, color="k", marker="x")

        plt.gca().invert_yaxis()
        plt.title("Order assignment")
        plt.xlabel("X pixel")
        plt.ylabel("Y pixel")
        plt.tight_layout()
        plt.show()

    orders = {}
    for o in slicelist[max_ind].order_ownership:
        o = int(o)
        orders[o] = SpectralOrder(o)

    for s in slicelist:
        for i, y in enumerate(s.ys):
            this_owner = int(s.order_ownership[i])
            try:
                orders[this_owner].pixel_y.append(y)
                orders[this_owner].pixel_x.append(s.x)
                orders[this_owner].pixel_y_err.append(s.y_errs[i])
                orders[this_owner].order_width.append(s.widths[i])
            except KeyError:
                orders[this_owner] = SpectralOrder(this_owner)
                orders[this_owner].pixel_y.append(y)
                orders[this_owner].pixel_x.append(s.x)
                orders[this_owner].pixel_y_err.append(s.y_errs[i])
                orders[this_owner].order_width.append(s.widths[i])

    olist = list(orders.values())

    for o in olist:
        o.sort_self()

    return olist


def assign_orders_polyfit(orders, slicelist: list[SpectralSlice], thres_ydist = 0.5, DEBUG_PLOTS=False):
    # thres_ydist is in pixels

    y_best_plot = []
    for o in orders:
        if DEBUG_PLOTS:
            plt.scatter(o.pixel_x, o.pixel_y, marker="o")
        o.pixel_x = []
        o.pixel_y = []
        o.pixel_y_err = []
        o.order_width = []

        for s in slicelist:
            ypred = polynomial(s.x, *o.solution)
            ydist = np.abs(s.ys - ypred)
            idx_best = np.argmin(ydist)
            if ydist[idx_best] < thres_ydist:
                o.pixel_x.append(s.x)
                o.pixel_y.append(s.ys[idx_best])
                o.pixel_y_err.append(s.y_errs[idx_best])
                o.order_width.append(s.widths[idx_best])
            y_best_plot.append(ydist[idx_best])

        o.pixel_x = np.array(o.pixel_x)
        o.pixel_y = np.array(o.pixel_y)
        o.pixel_y_err = np.array(o.pixel_y_err)
        o.order_width = np.array(o.order_width)

    if DEBUG_PLOTS:
        crot = ["black", "gray"]
        for i, o in enumerate(orders):
            plt.scatter(o.pixel_x, o.pixel_y, marker="x", s=6**2, color=crot[i%len(crot)])
        plt.show()

    if DEBUG_PLOTS:
        plt.hist(y_best_plot, bins=100, range=(0, 10))
        plt.show()

    for o in orders:
        o.sort_self()

    return orders


def repair_short_traces(orders, min_coverage=0.5, max_shift=30.0,
                        verbose=False):
    """Re-shape a trace that was fitted over too little of the detector."""
    if len(orders) < 3:
        return 0

    spans = {}
    for o in orders:
        x = np.asarray(getattr(o, "pixel_x", []), float)
        spans[id(o)] = (x.min(), x.max(), len(x)) if x.size else (0.0, 0.0, 0)
    width = max(hi for _, hi, _ in spans.values()) or 1.0

    def coverage(o):
        lo, hi, n = spans[id(o)]
        return (hi - lo) / width if n else 0.0

    nfixed = 0
    for i, o in enumerate(orders):
        if o.solution is None or coverage(o) >= min_coverage:
            continue
        x = np.asarray(o.pixel_x, float)
        y = np.asarray(o.pixel_y, float)
        good = np.asarray(getattr(o, "pixel_mask_good", np.ones(len(x), bool)))
        if good.sum() < 4:
            continue

        # nearest neighbour that covers the detector well
        donor = None
        for j in sorted(range(len(orders)), key=lambda k: abs(k - i)):
            if j == i or orders[j].solution is None:
                continue
            if coverage(orders[j]) >= min_coverage:
                donor = orders[j]
                break
        if donor is None:
            continue

        shift = float(np.median(y[good] - polynomial(x[good], *donor.solution)))
        if not np.isfinite(shift) or abs(shift) > max_shift:
            continue
        resid = y[good] - (polynomial(x[good], *donor.solution) + shift)
        rms = float(np.sqrt(np.mean(resid ** 2)))
        if rms > max(1.0, 5.0 * (o.rms or 0.0)):
            continue            # not parallel: leave the order's own fit

        o.solution = list(donor.solution)
        o.solution[-1] = o.solution[-1] + shift
        o.rms = rms
        nfixed += 1
        if verbose:
            print("- order %s traced over only %.0f%% of the detector: shape "
                  "taken from a neighbour (offset %+.1f px, rms %.2f px)"
                  % (o.id, 100 * coverage(o), shift, rms))
    return nfixed


def find_orders(frame_for_slice,
                sampling=200,
                min_order_samples=6,
                DEBUG_PLOTS=False, verbose=False):

    # Get orders and stuff from flat
    if verbose: print("- identifying orders")
    slices = find_slices(frame_for_slice, sampling=sampling, DEBUG_PLOTS=DEBUG_PLOTS)

    # find slice with most order identifications, preferring the detector centre
    xcen = frame_for_slice.shape[1] / 2
    norders_slice = [len(s.ys) if abs(s.x - xcen) < 200 else len(s.ys)/2 for s in slices]
    max_slice = np.argmax(norders_slice)

    # max_slice is the slice that has the largest number of order identifications
    orders = assign_orders(slices, max_slice, DEBUG_PLOTS=DEBUG_PLOTS)

    if verbose: print(f"- {len(orders)} orders found")

    norders = len(orders)
    idx = np.arange(norders)
    for i in idx:
        o = orders[i]
        if len(o) > min_order_samples:
            o.generate_polynomial_solution(verbose=verbose, DEBUG_PLOTS=DEBUG_PLOTS)

    # remove bad orders
    orders = [o for o in orders if o.solution is not None]
    """
    # re-assign slices based on first polynomial fit
    orders = assign_orders_polyfit(orders, slices, max_slice, **kwargs)
    # re-fit orders with polynomials
    o_to_be_removed = []
    for i, o in enumerate(orders):
        if len(o) > min_order_samples:
            o.generate_polynomial_solution(verbose=verbose)
        else:
            o.solution = None
    # remove bad orders
    orders = [o for o in orders if o.solution is not None]
    """

    if verbose: print(f"- {len(orders)} orders identified")

    repair_short_traces(orders, verbose=verbose)

    times_sigma = 2
    if DEBUG_PLOTS:
        plt.imshow(frame_for_slice, zorder=1, cmap='gray', norm="log")
        for slice in slices:
            plt.scatter([np.repeat(slice.x, len(slice.ys))], slice.ys, marker="x", zorder=2)
        for o in orders:
            # plt.scatter(o.pixel_x, o.pixel_y, marker="x", zorder=2)
            x = np.arange(frame_for_slice.shape[1])
            o.generate_width_fcn()
            sigma = o.w_fcn(x) / two_log_two
            width = times_sigma * sigma
            width = np.clip(width, a_min=1, a_max=4)
            pc = plt.plot(x, o.evaluate(x))
            plt.plot(x, o.evaluate(x)-width, c=pc[0]._color, ls="--")
            plt.plot(x, o.evaluate(x)+width, c=pc[0]._color, ls="--")
        # allow imshow stretching
        plt.xlabel("X pixel")
        plt.ylabel("Y pixel")
        plt.gca().set_aspect('auto')
        plt.tight_layout()
        plt.show()

    return orders
