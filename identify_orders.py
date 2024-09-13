import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import (minimum_filter, maximum_filter)

from tools import (mask_section, Gaussian, fill_nan, pair_generation)
from orders import SpectralOrder

two_log_two = 2 * np.sqrt(2 * np.log(2))

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
        for i, owner in enumerate(self.order_ownership):
            if owner is None:
                if i == 0:
                    j = 1
                    while self.order_ownership[i + j] is None:
                        j += 1
                    self.order_ownership[i] = self.order_ownership[i + j] - j
                else:
                    self.order_ownership[i] = self.order_ownership[i - 1] + 1


def slice_analysis(pixel, slice_x, slice_y, MIN_WINDOW=15, MAX_WINDOW=15, NOISE_MEASURE_SECTION_WIDTH=0.05,
                   NOISE_CUTOFF=20, CUTTOFF_MARGIN=5, ORDER_GAUSS_THRESHOLD=0.6, DEBUG_PLOTS=False):

#    if abs(pixel-1714) < 2:
#        DEBUG_PLOTS = True
#    else:
#        DEBUG_PLOTS = False

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

    # hot pixels or other artefacts are usually isolated
    # -> find groups of high-flux pixels (the orders)
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
        print(igroup_bad)
    if (len(igroup_bad) > 0):
        isplit_groups = np.array(isplit_groups, dtype=object)
        ibad = np.concatenate(isplit_groups[igroup_bad])
        mask_good = np.ones(len(noise_indices)).astype(bool)
        mask_good[ibad] = False
        noise_indices = noise_indices[mask_good]

    if DEBUG_PLOTS:
        plt.xlabel("group distance  /  pix")
        plt.hist(np.diff(groups_ipeak), bins=15, range=(0,50))
        plt.axvline(median_dist)
        plt.axvline(median_dist+10*std_dist)
        plt.show()
#        exit()

    # assume that "real" orders only start at pixles > 'idx_peak_min'
    idx_peak_min = 700
    idx_peak_max = 1750
    noise_indices = noise_indices[noise_indices>idx_peak_min]
    noise_indices = noise_indices[noise_indices<idx_peak_max]
    n_cross = 2
    first_cross = noise_indices[n_cross] - n_cross
    last_cross = noise_indices[-1-n_cross] + n_cross

    lo_ind = first_cross - CUTTOFF_MARGIN
    hi_ind = last_cross + CUTTOFF_MARGIN

    if DEBUG_PLOTS:
        plt.title(pixel)
        plt.plot(slice_x, slice_y)
        plt.axhline(noise_lvl, color="orange")
        plt.axvline(lo_ind, color='r')
        plt.axvline(hi_ind, color='r')
        max_slice = maximum_filter(slice_y, size=MAX_WINDOW)
        max_slice = np.clip(max_slice, a_min=noise_lvl*2, a_max=np.inf)
        plt.plot(slice_x, max_slice, color="tab:green")
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
#    print(f"Identified {len(peaks)} orders @ x = {pixel}")

    if DEBUG_PLOTS:
        plt.title(str(pixel))
        for idx, l in enumerate(peak_locations):
            plt.axvline(l, ymin=0, ymax=0.93, color="gray")
            plt.text(s=str(idx), x=l, y=1.1, rotation=90,
                     va="center", ha="center")
        plt.axhline(ORDER_GAUSS_THRESHOLD, color="orange")
        plt.plot(slice_x, slice_y)
        plt.ylim(-0.05, 1.15)
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
            plt.plot(x_neighborhood, Gaussian(x_neighborhood, *params), color="r")
        fit_params.append(params)

        errs = np.sqrt(np.diag(errs))

        refined_peak_locations.append(params[1])
        location_uncertainties.append(errs[1])
        widths.append(two_log_two * params[2])

    if DEBUG_PLOTS:
        plt.title(pixel)
        plt.plot(slice_x, slice_y)
        plt.ylim(-0.05, 1.15)
        plt.tight_layout()
        plt.show()

    return SpectralSlice(pixel,
                         np.array(refined_peak_locations),
                         np.array(location_uncertainties),
                         np.array(widths))


def find_slices(frame_for_slice, sampling=200, DEBUG_PLOTS=False):
    # Get orders and stuff from flat
    npix_x = frame_for_slice.shape[1]
    pixels = np.linspace(5, npix_x-5, sampling).astype(int)
    slices = []
    for i in range(sampling):
        pixel = pixels[i]
#        slice_y = frame_for_slice[:, pixel - 1].astype(float)
        # average 3 pixels in x direction
        xidx = np.arange(3) + pixel - 1
        xidx = xidx[(xidx>=0) & (xidx<npix_x)]
        slice_y = np.sum(frame_for_slice[:, xidx].astype(float), axis=1) / len(xidx)
        slice_x = np.arange(frame_for_slice.shape[1])
        slice = slice_analysis(pixel - 1, slice_x, slice_y, DEBUG_PLOTS=DEBUG_PLOTS)
        slices.append(slice)

    if DEBUG_PLOTS:
        plt.imshow(frame_for_slice, zorder=1, cmap='gray', norm="log")
        for slice in slices:
            plt.scatter([np.repeat(slice.x, len(slice.ys))], slice.ys, marker="x", zorder=2)
        plt.tight_layout()
        plt.show()

    norders_slice = []
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
        lold = len(o.pixel_y)
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
        lnew = len(o.pixel_y)

    o.pixel_x = np.array(o.pixel_x)
    o.pixel_y = np.array(o.pixel_y)

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


def find_orders(frame_for_slice,
                sampling=200,
                min_order_samples=6,
                DEBUG_PLOTS=False, verbose=False):

    # Get orders and stuff from flat
    if verbose: print("- identifying orders")
    slices = find_slices(frame_for_slice, sampling=sampling, DEBUG_PLOTS=DEBUG_PLOTS)

    # find slice with most order identifications
    norders_slice = [len(s.ys) if abs(s.x - 1024) < 200 else len(s.ys)/2 for s in slices]
    max_slice = np.argmax(norders_slice)

    # max_slice is the slice that has the largest number of order identifications
    orders = assign_orders(slices, max_slice, DEBUG_PLOTS=DEBUG_PLOTS)

    if verbose: print(f"- {len(orders)} orders found")

    norders = len(orders)
    idx = np.arange(norders)
    # start from the middle, alternate each side (use later for starting parameters)
#    idx = norders // 2 + (idx + 1) // 2 * (-1) ** idx
    # fit orders with polynomials
    for i in idx:
        o = orders[i]
        if len(o) > min_order_samples:
            o.generate_polynomial_solution(verbose=verbose, DEBUG_PLOTS=DEBUG_PLOTS)

    # remove bad orders
    orders = [o for o in orders if o.solution is not None]

#    if DEBUG_PLOTS:
#        for o in orders:
#            plt.hist(o.rms, bins=10)
#        plt.show()

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

    times_sigma = 2
    if DEBUG_PLOTS:
        plt.imshow(frame_for_slice, zorder=1, cmap='gray')
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
        plt.tight_layout()
        plt.show()

    return slices, orders
