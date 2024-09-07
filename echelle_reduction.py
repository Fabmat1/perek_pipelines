import os
import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from scipy.constants import speed_of_light
from matplotlib import pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import minimum_filter, maximum_filter, median_filter, uniform_filter1d
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import curve_fit, least_squares
from scipy.signal import savgol_filter
from astropy.stats import sigma_clip, mad_std
from estimate_noise import estimate_noise
from tools import polyfit_reject, pair_generation
try:
    from resample_spectres import resample
except ModuleNotFoundError:
    print("compile 'pyresample_spectres' like this:")
    print("python3 -m numpy.f2py -c -m pyresample_spectres resample_spectres.f90")
    try:
        from spectres import spectres as resample
    except ModuleNotFoundError:
        raise Exception("Need either 'spectres' or 'resample_spectres'")

two_log_two = 2 * np.sqrt(2 * np.log(2))

def polynomial(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d

class SpectralOrder:
    def __init__(self, id):
        self.id = id
        self.order_width = []
        self.pixel_x = []
        self.pixel_y = []
        self.pixel_y_err = []
        self.solution = None
        self.solution_errors = None
        self.w_fcn = None

        self.pixel_y_cen = None

        self.science = None
        self.flat = None
        self.comparison = None
        self.bias = None

        self.wl = None
        self.pix = None

        self.cal_rms = None
        self.cal_pix = None
        self.cal_wl = None
        self.cal_pix_fwhm = None
        self.cal_wl_fwhm = None

    def __len__(self):
        return len(self.pixel_x)

    def generate_polynomial_solution(self, yerr_default=1.5, verbose=True):
        # yerr_default -> maximum rms (in pix) for an acceptable fit

        params, errs = curve_fit(polynomial, self.pixel_x, self.pixel_y, sigma=self.pixel_y_err)
        errs = np.sqrt(np.diag(errs))

        nresid = len(self.pixel_y)

        ypoly = polynomial(self.pixel_x, *params)
        # root mean squared
        rms = np.sqrt(np.sum(np.square(self.pixel_y - ypoly)) / nresid)

        self.rms = rms

        """
        # estimate reduced chi2
        nfree = 4
        dof = nresid - nfree
        # self.pixel_y_err
        resid = (self.pixel_y - ypoly) / yerr_default
        chi2 = np.sqrt(np.sum(np.square(resid)))
        rchi2 = chi2 / dof
        """

        # only save decent fits
        if rms < yerr_default:
            self.solution = params
            self.solution_errors = errs
        else:
            if verbose: print("- identification failed for order", self.id)
            self.solution = None
            self.solution_errors = None

    def generate_width_fcn(self):
        self.w_fcn = interp1d(self.pixel_x, self.order_width, bounds_error=False, fill_value=np.mean(self.order_width))

    def evaluate(self, x):
        return polynomial(x, *self.solution)

    def sort_self(self):
        # Combine all the lists into a single list of tuples
        combined = list(zip(self.pixel_x, self.order_width, self.pixel_y, self.pixel_y_err))

        # Sort the combined list based on the first element in each tuple (pixel_x)
        combined.sort(key=lambda x: x[0])

        # Unzip the sorted combined list back into the individual lists
        self.pixel_x, self.order_width, self.pixel_y, self.pixel_y_err = zip(*combined)

        self.pixel_x = np.array(list(self.pixel_x))
        self.order_width = np.array(list(self.order_width))
        self.pixel_y = np.array(list(self.pixel_y))
        self.pixel_y_err = np.array(list(self.pixel_y_err))

    def extract_along_order(self, image, type, times_sigma=2):
        if self.solution is None:
            raise Exception("Generate a solution first!")
        if self.w_fcn is None:
            self.generate_width_fcn()

        intensities = []

        for pixel in np.arange(image.shape[1]):
            sigma = self.w_fcn(pixel) / two_log_two
            width = times_sigma * sigma
            width = np.clip(width, a_min=1, a_max=4)
            y_ind = self.evaluate(pixel)
            top = int(np.floor(y_ind + width))
            bot = int(np.ceil(y_ind - width))

            fluxsum = np.dot(image[bot:top, int(pixel)], Gaussian(np.arange(bot, top), 1, y_ind, sigma))
            upperfraction = image[bot - 1, int(pixel)] * (np.abs(bot - (y_ind - width))) * Gaussian(bot - 1, 1, y_ind, sigma)
            lowerfraction = image[top + 1, int(pixel)] * (np.abs(top - (y_ind + width))) * Gaussian(top + 1, 1, y_ind, sigma)
            fluxsum = upperfraction + lowerfraction + fluxsum
            intensities.append(fluxsum)

        if type == "bias" or type == "zero":
            self.bias = np.array(intensities)
        elif type == "flat":
            self.flat = np.array(intensities)
        elif type == "comparison" or type == "comp":
            self.comparison = np.array(intensities)
        elif type == "science":
            self.science = np.array(intensities)
        else:
            raise Exception("Unknown frame type!")

    def plot_frame(self, type):
        if type == "bias" or type == "zero":
            data_y = self.bias
            data_x = self.wl if self.wl is not None else np.arange(len(data_y)) + 1
            plt.title("bias")
        elif type == "flat":
            data_y = self.flat
            data_x = self.wl if self.wl is not None else np.arange(len(data_y)) + 1
            plt.title("flat")
        elif type == "comparison" or type == "comp":
            data_y = self.comparison
            data_x = self.wl if self.wl is not None else np.arange(len(data_y)) + 1
            plt.title("comparison")
        elif type == "science":
            data_y = self.science
            data_x = self.wl if self.wl is not None else np.arange(len(data_y)) + 1
            plt.title("science")
        else:
            raise Exception("Unknown frame type!")

        plt.plot(data_x, data_y)

        plt.tight_layout()
        plt.show()

    def apply_corrections(self, med_win_size=25, min_win_size=15, max_win_size=15):
        self.flat -= self.bias
        self.science -= self.bias
        self.comparison -= self.bias

        norm_flat = median_filter(self.flat, size=med_win_size)
        # norm_flat /= norm_flat.max()

        self.flat = norm_flat

        self.science /= norm_flat

        self.comparison -= minimum_filter(self.comparison, size=min_win_size)
        self.comparison /= maximum_filter(self.comparison, size=max_win_size)


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


def Gaussian(x, A, mu=0, sigma=1):
    return A * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def Gaussian_res(x, A, mu=0, sigma=1):
    oversample = 10
    xfull = np.linspace(np.min(x), np.max(x), len(x)*oversample)
    yfull = Gaussian(xfull, A, mu=mu, sigma=sigma)
    return resample(x, xfull, yfull, fill=0, verbose=False)


def assign_orders(slicelist: list[SpectralSlice], max_ind, DEBUG_PLOTS=False, **kwargs):
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
#        print(lold, lnew)

    o.pixel_x = np.array(o.pixel_x)
    o.pixel_y = np.array(o.pixel_y)

    if DEBUG_PLOTS:
        crot = ["black", "gray"]
        for i, o in enumerate(orders):
            plt.scatter(o.pixel_x, o.pixel_y, marker="x", s=6**2, color=crot[i%len(crot)])
#            print(o.pixel_x)
#            print(o.pixel_y)
        plt.show()

    if DEBUG_PLOTS:
        plt.hist(y_best_plot, bins=100, range=(0, 10))
        plt.show()

    for o in orders:
        o.sort_self()

    return orders


def parse_idcomp(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    aplow = None
    aphigh = None
    table_data = []
    in_table = False

    for line in lines:
        # Strip leading/trailing whitespace
        line = line.strip()

        # Extract aplow and aphigh values
        if line.startswith('aplow'):
            aplow = float(line.split()[1])
        elif line.startswith('aphigh'):
            aphigh = float(line.split()[1])

        # Identify when the table starts
        if line.startswith('features'):
            in_table = True
            continue  # Skip the "features" line itself

        # Collect table data
        if in_table:
            # Check if the line is still part of the table (starts with a number)
            if line and line[0].isdigit():
                floats = line.split()
                if len(floats) == 7:
                    floats = floats[:-1]
                row = list(map(float, floats))
                table_data.append(row)
            else:
                # Table ends if we encounter a non-digit line
                in_table = False

    return aplow, aphigh, np.array(table_data)


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

def slice_analysis(pixel, slice_x, slice_y, MIN_WINDOW=15, MAX_WINDOW=15, NOISE_MEASURE_SECTION_WIDTH=0.05,
                   NOISE_CUTOFF=20, CUTTOFF_MARGIN=5, ORDER_GAUSS_THRESHOLD=0.6, DEBUG_PLOTS=False, **kwargs):

#    if abs(pixel-1437) < 2:
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
        plt.tight_layout()
        plt.show()

    slice_x = slice_x[lo_ind:hi_ind]
    slice_y = slice_y[lo_ind:hi_ind]

    max_slice = maximum_filter(slice_y, size=MAX_WINDOW)
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
        plt.plot(slice_x, slice_y)
        plt.ylim(-0.05, 1.15)
        plt.tight_layout()
        plt.show()

    fit_params = []
    avg_peak_distance = np.mean(np.diff(peak_locations))

    refined_peak_locations = []
    location_uncertainties = []
    widths = []
    for i, peak_location in enumerate(peak_locations):
        if i == 0:
            mask = slice_x < peak_location + (peak_locations[i + 1] - peak_location) / 2
            bounds = [[0, 0, 0], [np.inf, peak_locations[i + 1], avg_peak_distance / 2]]
        elif i == len(peak_locations) - 1:
            mask = slice_x > peak_location - (peak_location - peak_locations[i - 1]) / 2
            bounds = [[0, peak_locations[i - 1], 0], [np.inf, np.max(slice_x), avg_peak_distance / 2]]
        else:
            mask = np.logical_and(slice_x < peak_location + (peak_locations[i + 1] - peak_location) / 2,
                                  slice_x > peak_location - (peak_location - peak_locations[i - 1]) / 2)
            bounds = [[0, peak_locations[i - 1], 0], [np.inf, peak_locations[i + 1], avg_peak_distance / 2]]
        x_neighborhood = slice_x[mask]
        y_neighborhood = slice_y[mask]

        params, errs = curve_fit(Gaussian, x_neighborhood, y_neighborhood,
                                 [1, peak_location, avg_peak_distance / 4],
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


def open_or_coadd_frame(frame, get_add_info=False):
    if isinstance(frame, np.ndarray):
        if get_add_info:
            raise Exception("Spectrum MUST be path to fits file!")
        return frame
    elif isinstance(frame, list):
        frame = np.sum(frame, axis=0).astype(float) / len(frame)
        if get_add_info:
            raise Exception("Spectrum MUST be path to fits file!")
    else:
        frame = fits.open(frame)
        if get_add_info:
            header = frame[0].header
            header = dict(header)
            hreq = ["LATITUDE", "LONGITUD", "HEIGHT",
                    "RA", "DEC", "DATE-OBS", "UT"]
            if all(i in header for i in hreq):
                lat = header['LATITUDE']
                lon = header['LONGITUD']

                height = header['HEIGHT']
                RA = header['RA']
                DEC = header['DEC']

                time = Time(header["DATE-OBS"]+"T"+header["UT"], format='isot', scale='utc')

                location = EarthLocation(lat=lat, lon=lon, height=height)
                coord = SkyCoord(ra=RA, dec=DEC, unit=(u.hourangle, u.deg))
                radvel_corr = coord.radial_velocity_correction(obstime=time, location=location)
                radvel_corr = radvel_corr.to(u.km / u.s)
                radvel_corr = radvel_corr.value

                return frame[0].data, radvel_corr
            else:
                return frame[0].data, None

        else:
            frame = frame[0].data

    return frame

class WavelenthPixelTransform():
    def __init__(self, wstart, dwdp=None, dwdp2=None, dwdp3=None, dwdp4=None, polyorder=3):
        self.wstart = wstart  # wavelength at pixel 0
        self.dwdp = dwdp  # d(wavelength)/d(pixel)
        self.dwdp2 = dwdp2  # d(wavelength)^2/d(pixel)^2
        self.dwdp3 = dwdp3  # d(wavelength)^3/d(pixel)^3
        self.dwdp4 = dwdp4  # d(wavelength)^4/d(pixel)^4
        self.polyorder = polyorder

    def wl_to_px(self, wl_arr):
        pxspace = np.linspace(0, 2500, 2500)
        f = interp1d(self.px_to_wl(pxspace), pxspace, bounds_error=False, fill_value="extrapolate")
        return f(wl_arr)

    def px_to_wl(self, px_arr):
        if self.polyorder == 4:
            return line(px_arr, self.dwdp4, self.dwdp3, self.dwdp2, self.dwdp, self.wstart)
        elif self.polyorder == 3:
            return self.wstart + self.dwdp * px_arr + self.dwdp2 * px_arr ** 2 + self.dwdp3 * px_arr ** 3


def wlshift(wl, vel_corr):
    # wl_shift = vel_corr/speed_of_light * wl
    # return wl+wl_shift
    return wl / (1 - (vel_corr / (speed_of_light / 1000)))


def solve_wavelength(linetable, order: SpectralOrder, pixel_window=10, DEBUG_PLOT=False):
    line_wls = (linetable[:, 1] + linetable[:, 2]) / 2
    line_px = linetable[:, 0]

    initial_params, _ = curve_fit(polynomial, line_px, line_wls)

    pixels = np.arange(len(order.comparison)) + 1
    actual_positions = []
    actual_errors = []
    fwhm_pix = []

    for l in line_px:
        px_window = np.logical_and(pixels > l - pixel_window, pixels < l + pixel_window)
        pwin = pixels[px_window]
        intensities = order.comparison[px_window]

        # replace with Levenberg-Marquardt fit (lmfit)
        # remove bad fits (assume snr = 100), cut on rchi^2
        # possibly resample
        # extrapolate solutions, iterative identification with ThAr lines
        params, errs = curve_fit(Gaussian_res, pwin, intensities,
                                 p0=[1, l, pixel_window / 4],
                                 bounds=[
                                     [0, l - pixel_window / 2, 0],
                                     [np.inf, l + pixel_window / 2, pixel_window / 2]
                                 ],
                                 maxfev=100000)

        errs = np.sqrt(np.diag(errs))

        actual_positions.append(params[1])
        actual_errors.append(errs[1])
        fwhm_pix.append(params[2]*two_log_two)
        if DEBUG_PLOT:
            plt.plot(pwin, Gaussian_res(pwin, *params), color="red", zorder=20)

    actual_positions = np.array(actual_positions)
    actual_errors = np.array(actual_errors)
    fwhm_pix = np.array(fwhm_pix)

    if DEBUG_PLOT:
        plt.plot(pixels, order.comparison, zorder=10)
        plt.show()

    # this depends on the spectrograph!
    too_wide_pix = 6
    too_narrow_pix = 2.5

    mask_good = (fwhm_pix < too_wide_pix) & (fwhm_pix > too_narrow_pix)

    not_enough_lines = (np.sum(mask_good) < 5)
    if not_enough_lines:
        mask_good = (fwhm_pix < 9) & (fwhm_pix > 2)
    not_enough_lines = (np.sum(mask_good) < 5)
    if not_enough_lines:
        mask_good = np.ones(len(fwhm_pix)).astype(bool)
    if (np.sum(mask_good) < 5):
        raise Exception("Not enough calibration lines in order %d" % order.id)

    if DEBUG_PLOT:
        plt.scatter(actual_positions[~mask_good], fwhm_pix[~mask_good], zorder=10, color="gray")
        plt.scatter(actual_positions[mask_good], fwhm_pix[mask_good], zorder=11, color="black")
        plt.axhline(too_wide_pix, ls="--", c="red")
        plt.axhline(too_narrow_pix, ls="--", c="red")
        plt.show()

    actual_positions = actual_positions[mask_good]
    actual_errors = actual_errors[mask_good]
    fwhm_pix = fwhm_pix[mask_good]
    line_wls = line_wls[mask_good]

    # threshold = np.percentile(actual_errors, 0.9)
    # actual_positions = actual_positions[actual_errors <= threshold]
    # actual_errors = actual_errors[actual_errors <= threshold]

    params, errs = curve_fit(polynomial, actual_positions, line_wls, sigma=actual_errors)

    pix_width = [np.mean(np.diff(polynomial(np.arange(3)+i-1,*params))) for i in actual_positions]
    pix_width = np.abs(pix_width)
    fwhm_angstrom = fwhm_pix * np.array(pix_width)

    order.wl = polynomial(pixels, *params)
    order.cal_pix = actual_positions
    order.cal_wl = line_wls
    order.cal_pix_fwhm = fwhm_pix
    order.cal_wl_fwhm = fwhm_angstrom

    if DEBUG_PLOT:
        plt.scatter(order.cal_wl, order.cal_wl/fwhm_angstrom, zorder=10)
        plt.show()

def plot_order_list(olist: list[SpectralOrder]):
    for o in olist:
        plt.plot(o.wl, o.science)
    plt.tight_layout()
    plt.show()

def generate_wl_grid(wl, resolution=45000, sampling=2.7):
    temp = (2 * sampling * resolution + 1) / (2 * sampling * resolution - 1)
    nwave = np.ceil(np.log(wl.max() / wl.min()) / np.log(temp))
    if wl.min() > 0 and np.isfinite(nwave):
        t2 = np.arange(nwave)
        new_grid = temp ** t2 * wl.min()
    else:
        print("WARNING: could not find nwave, using old grid")
        new_grid = wl
    return new_grid

def mask_intervals(x, intervals):
    # Initialize a mask that is True for all elements
    mask = np.ones_like(x, dtype=bool)

    # Loop through each interval and update the mask
    for interval in intervals:
        lower_bound, upper_bound = interval
        mask &= ~((x >= lower_bound) & (x <= upper_bound))

    return mask

def poly_normalization(wls, flxs,
                         ignore_windows=[(4090, 4115), (4320, 4355),
                                         (4842, 4888), (6540, 6590),
                                         (6860, 6880),
                                         (6888.1, 6890.5), (6892,6893.6),
                                         (7590, 7617), (7622.8, 7625)],
                         DEBUG_PLOTS=False):

    for i, wl in enumerate(wls):
        flx = flxs[i]

        mask = mask_intervals(wl, ignore_windows)
        wl_for_interpol = wl[mask]
        flx_for_interpol = flx[mask]

        # clip some extreme outliers before the first fit
        mask = ~sigma_clip(flx_for_interpol,
                           sigma_lower=10,
                           sigma_upper=10,
                           axis=0, masked=True).mask

        sigmas_lo = [6, 4, 3, 2, 1.7]
        sigmas_hi = [6, 5, 4, 4, 3]
        nit = len(sigmas_lo)
        for k in range(nit):
            params, errs = curve_fit(polynomial,
                                     wl_for_interpol[mask],
                                     flx_for_interpol[mask])
            flx_cont = polynomial(wl_for_interpol, *params)
            mask = ~sigma_clip(flx_for_interpol/flx_cont,
                               sigma_lower=sigmas_lo[k],
                               sigma_upper=sigmas_hi[k],
                               masked=True,
                               axis=0).mask
        flx_cont = polynomial(wl, *params)
        flx /= flx_cont
        flxs[i] = flx

        if DEBUG_PLOTS:
            colors_good = ["navy", "black"]
            color_bad = ["cadetblue", "gray"]
            color_model = ["darkorange", "red"]
            plt.scatter(wl_for_interpol[~mask], flx_for_interpol[~mask],
                        marker=".", zorder=10, color=color_bad[i%2])
            plt.scatter(wl_for_interpol[mask], flx_for_interpol[mask],
                        marker=".", zorder=11, color=colors_good[i%2])
            plt.plot(wl, flx_cont, color=color_model[i%2], zorder=12)


        # hiflx = flx[int(0.8*len(flx)):int(0.9*len(flx))]
        # loflx = flx[int(0.1*len(flx)):int(0.2*len(flx))]
        #
        # hiwl = np.mean(wl[int(0.8*len(flx)):int(0.9*len(flx))])
        # lowl = np.mean(wl[int(0.1*len(flx)):int(0.2*len(flx))])
        #
        # himed = np.median(hiflx)
        # lomed = np.median(loflx)
        #
        # m = (himed-lomed)/(hiwl-lowl)
        # n = lomed - m * lowl
        #
        # flxs[i] /= m*wls[i]+n

        # nextflx = np.median(flxs[i+1][int(0.1*len(flx)):int(0.2*len(flx))])
        # nextwl = np.median(wls[i+1][int(0.1*len(flx)):int(0.2*len(flx))])
        #
        # shouldbe = nextwl*m+n
        #
        # fac = shouldbe/nextflx

        # flxs[i+1] *= fac

    if DEBUG_PLOTS:
        plt.show()

    return flxs


def merge_orders(olist: list[SpectralOrder], normalize=True, margin=2, max_wl=8900,
                 resolution=50000, DEBUG_PLOTS=False):
    common_wl = [o.wl[margin:-margin] for o in olist if o.wl.min() < max_wl]
    common_flx = [o.science[margin:-margin] for o in olist if o.wl.min() < max_wl]

    if DEBUG_PLOTS:
        for w, f in zip(common_wl, common_flx):
            plt.plot(w, f)
        plt.show()

    if normalize:
        common_flx = poly_normalization(common_wl, common_flx, DEBUG_PLOTS=DEBUG_PLOTS)

    # estimate noise for each order, to be used for the weights when merging
#    common_noise = [estimate_noise(common_wl[k], common_flx[k]) for k in range(len(common_flx))]

    common_wl = np.concatenate(common_wl)
    common_flx = np.concatenate(common_flx)

    mask = np.argsort(common_wl)
    common_wl = common_wl[mask]
    common_flx = common_flx[mask]

    mask = np.isfinite(common_flx)
    common_wl = common_wl[mask]
    common_flx = common_flx[mask]

    new_grid = generate_wl_grid(common_wl, resolution=resolution)
    # this can only compute a "combined noise", but does not weigh pixels by SNR
    common_flx  = resample(new_grid, common_wl, common_flx,
                           verbose=False)
    common_wl = new_grid

    if DEBUG_PLOTS:
        plt.plot(common_wl, common_flx)
        plt.tight_layout()
        plt.show()

    return common_wl, common_flx

def rolling_std(arr, window):
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    cumsum_sq = np.cumsum(np.insert(arr**2, 0, 0))
    mean = (cumsum[window:] - cumsum[:-window]) / window
    mean_sq = (cumsum_sq[window:] - cumsum_sq[:-window]) / window
    std = np.sqrt(mean_sq - mean**2)
    return np.concatenate((np.zeros(window-1), std))

def rmcosmics(wl, flx):
    mean_filtered = uniform_filter1d(flx, size=10)
    std_filtered = rolling_std(flx, window=50)
    mask = flx < 3 * std_filtered + mean_filtered
    return wl[mask], flx[mask]

def estimate_resolution(orders, verbose=False, DEBUG_PLOTS=False):
    p = 0.6827
    pl = 0. + 0.5 * (1. - p)
    ph = 1. - 0.5 * (1. - p)
    wl_med = []
    res_med = []
    res_qlo = []
    res_qhi = []
    res_all = []
    res_all_wl = []
    for o in orders:
        res_o = o.cal_wl/o.cal_wl_fwhm
        quantile_lo = np.quantile(res_o, pl)
        quantile_hi = np.quantile(res_o, ph)
        wl_med.append(np.median(o.cal_wl))
        res_med.append(np.median(res_o))
        res_qlo.append(quantile_lo)
        res_qhi.append(quantile_hi)
        res_all.extend(res_o)
        res_all_wl.extend(o.cal_wl)
    res_qlo = np.array(res_qlo)
    res_qhi = np.array(res_qhi)
    res_med = np.array(res_med)

    res_all = np.array(res_all)
    res_all_wl = np.array(res_all_wl)
    isort = np.argsort(res_all_wl)
    res_all = res_all[isort]
    res_all_wl = res_all_wl[isort]

#    window_size = 25 # AA
#    rmed, rqlo, rqhi = rolling_median_quant(res_all_wl, res_all, window_size, p=0.6827)

    coeffs = []
    for o in orders:
        res_o = o.cal_wl/o.cal_wl_fwhm
        if len(res_o) < 15:
            deg = 1
        else:
            deg = 2
        coeff = polyfit_reject(o.cal_wl, res_o, deg=deg, thres=2, nit=3)
        coeffs.append(coeff)

    if DEBUG_PLOTS:
        for i, o in enumerate(orders):
            res_o = o.cal_wl/o.cal_wl_fwhm
#            plt.scatter(o.cal_wl, o.cal_wl_fwhm, zorder=10, lw=1)
            xeval = np.linspace(np.min(o.cal_wl), np.max(o.cal_wl), 100)
            poly1d_fn = np.poly1d(coeffs[i])
            plt.plot(xeval, poly1d_fn(xeval), zorder=11, lw=2, color="blue", ls="-")

            plt.scatter(o.cal_wl, res_o, zorder=10, lw=1)
            plt.plot()

        plt.plot(wl_med, res_med, color="black", lw=2, zorder=20, ls="-")
        plt.plot(wl_med, res_qlo, color="black", lw=2, zorder=20, ls="--")
        plt.plot(wl_med, res_qhi, color="black", lw=2, zorder=20, ls="--")
        plt.xlabel(r"Wavelength  /  $\mathrm{\AA}$")
        plt.ylabel(r"$\lambda$  /  $\Delta \lambda$")
        plt.show()

    # estimate the median resolving power

    res_all_qlo = np.quantile(res_all, pl)
    res_all_qhi = np.quantile(res_all, ph)
    res_all_med = np.median(res_all)
    res_all_min = res_all_med - res_all_qlo
    res_all_max = res_all_qhi - res_all_med
    if verbose: print("- R = %.0f^{+%.0f}_{-%.0f}" % \
                     (res_all_med, res_all_max, res_all_min))

    dout = {"R_med": res_all_med,
            "R_lo": res_all_qlo,
            "R_hi": res_all_qhi,
            "coeffs": coeffs}

    return dout

def merge_resolution(wave_merged, orders, dres, DEBUG_PLOTS=False):
    res_poly = []
    res_poly_wl = []
    for i, o in enumerate(orders):
        coeff = dres["coeffs"][i]
        xeval = np.linspace(np.min(o.wl), np.max(o.wl), 200)
        poly1d_fn = np.poly1d(coeff)
        yeval = poly1d_fn(xeval)
        res_poly.extend(yeval)
        res_poly_wl.extend(xeval)
        if DEBUG_PLOTS: plt.plot(xeval, yeval, lw=1.5)
    res_poly = np.array(res_poly)
    res_poly_wl = np.array(res_poly_wl)
    isort = np.argsort(res_poly_wl)
    res_poly = res_poly[isort]
    res_poly_wl = res_poly_wl[isort]
#    res_merged = resample(wave_merged, res_poly_wl, res_poly)
    res_merged = np.interp(wave_merged, res_poly_wl, res_poly)
    res_med = dres["R_med"]
    res_min = dres["R_lo"] / 3
    res_merged[res_merged<=res_min] = res_min
    # smooth
    npix = 35
    res_merged = gaussian_filter1d(res_merged, npix,
                                   mode="constant", cval=res_med)

    if DEBUG_PLOTS:
        plt.plot(wave_merged, res_merged, ls="--", lw=2, color="black")
        plt.show()
    return res_merged

def extract_spectrum(spectrum, flats, comps, biases, idcomp_offset=-15,
                     normalize=True, idcomp_dir="idcomp",
                     sampling=200, min_order_samples=6,
                     apply_barycorr=True,
                     verbose=False,
                     DEBUG_PLOTS=False, **kwargs):

    spectrum, radvel = open_or_coadd_frame(spectrum, True)
    flats = open_or_coadd_frame(flats)
    comps = open_or_coadd_frame(comps)
    biases = open_or_coadd_frame(biases)

    slices = []

    # Get orders and stuff from flat
    if verbose: print("- identifying orders")
    npix_x = flats.shape[1]
    pixels = np.linspace(5, npix_x-5, sampling).astype(int)
    for i in range(sampling):
        pixel = pixels[i]
#        slice_y = flats[:, pixel - 1].astype(float)
        # average 3 pixels in x direction
        xidx = np.arange(3) + pixel - 1
        xidx = xidx[(xidx>=0) & (xidx<npix_x)]
        slice_y = np.sum(flats[:, xidx].astype(float), axis=1) / len(xidx)
        slice_x = np.arange(flats.shape[1])
        slice = slice_analysis(pixel - 1, slice_x, slice_y, DEBUG_PLOTS=DEBUG_PLOTS, **kwargs)
        slices.append(slice)

    if DEBUG_PLOTS:
        plt.imshow(flats, zorder=1, cmap='gray', norm="log")
        for slice in slices:
            plt.scatter([np.repeat(slice.x, len(slice.ys))], slice.ys, marker="x", zorder=2)
        plt.tight_layout()
        plt.show()

    norder = 0
    max_slice = 0
    for i, slice in enumerate(slices):
        if i == 0:
            slice.next_slice = slices[1]
        elif i == len(slices) - 1:
            slice.previous_slice = slices[-2]
        else:
            slice.previous_slice = slices[i - 1]
            slice.next_slice = slices[i + 1]

        if len(slice.ys) > norder:
            max_slice = i
            norder = len(slice.ys)

    # max_slice is the slice that has the largest number of order identifications
    orders = assign_orders(slices, max_slice, **kwargs)

    if verbose: print(f"- {len(orders)} orders found")

    # fit orders with polynomials
    for i, o in enumerate(orders):
        if len(o) > min_order_samples:
            o.generate_polynomial_solution(verbose=verbose)

    if DEBUG_PLOTS:
        for o in orders:
            plt.hist(o.rms, bins=10)
        plt.show()

    # remove bad orders
    orders = [o for o in orders if o.solution is not None]

    # estimate y-pos of each order at the center of the x axis
    ap_measure = []
    for o in orders:
        ap = float(polynomial(len(spectrum) / 2, *o.solution))
        o.pixel_y_cen = ap
        ap_measure.append(ap)

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
        plt.imshow(flats, zorder=1, cmap='gray')
        for slice in slices:
            plt.scatter([np.repeat(slice.x, len(slice.ys))], slice.ys, marker="x", zorder=2)
        for o in orders:
            # plt.scatter(o.pixel_x, o.pixel_y, marker="x", zorder=2)
            x = np.arange(flats.shape[1])
            o.generate_width_fcn()
            sigma = o.w_fcn(x) / two_log_two
            width = times_sigma * sigma
            width = np.clip(width, a_min=1, a_max=4)
            pc = plt.plot(x, o.evaluate(x))
            plt.plot(x, o.evaluate(x)-width, c=pc[0]._color, ls="--")
            plt.plot(x, o.evaluate(x)+width, c=pc[0]._color, ls="--")
        plt.tight_layout()
        plt.show()

    if verbose: print("- extracting orders")

    # Extract different spectra
    for o in orders:
        if verbose: print("- order", o.id, end="\r")
        o.extract_along_order(spectrum, "science", times_sigma=times_sigma)
        o.extract_along_order(flats, "flat", times_sigma=times_sigma)
        o.extract_along_order(biases, "bias", times_sigma=times_sigma)
        o.extract_along_order(comps, "comp", times_sigma=times_sigma)

        o.apply_corrections()
        # o.plot_frame("science")
        # o.plot_frame("flat")
        # o.plot_frame("bias")
        # o.plot_frame("comp")

    linelists = {}
    fp_idcomp = sorted(os.listdir(idcomp_dir))
    for file in fp_idcomp:
        if "idiazcomp" in file:
            aplo, aphi, table = parse_idcomp(idcomp_dir + "/" + file)
            avg_ap = (aplo + aphi) / 2
            linelists[avg_ap+idcomp_offset] = table
    avg_aps = np.array(list(linelists.keys()))
    nlist = len(avg_aps)

    if DEBUG_PLOTS:
        plt.imshow(flats)
        for key in linelists.keys():
            plt.scatter([len(spectrum) / 2], [key], marker="x", zorder=2, color="red")
        plt.show()

    # find the best-matching orders
    id_order_pairs = pair_generation(avg_aps, ap_measure, thres_max=np.inf)
    id_order_pairs = [p for p in id_order_pairs if (p[0] is not None) and (p[1] is not None)]
    if verbose: print("- found %d pairs" % len(id_order_pairs))

    if verbose: print("- solving dispersion relations")
    for p in id_order_pairs:
        idx_id = p[0]
        idx_order = p[1]
        o = orders[idx_order]
        if verbose: print("- order", o.id, end="\r")
        key = avg_aps[idx_id]
        linelist_o = linelists[key]
        solve_wavelength(linelist_o, o)

    orders = [o for o in orders if o.wl is not None]

    # sort wavelengths
    for o in orders:
        o.pix = np.arange(len(o.wl))
        isort = np.argsort(o.wl)
        o.wl = o.wl[isort]
        o.science = o.science[isort]
        o.flat = o.flat[isort]
        o.comparison = o.comparison[isort]
        o.bias = o.bias[isort]
        o.pix = o.pix[isort]

    if DEBUG_PLOTS:
        for o in orders:
            plt.plot(o.wl, o.science)
            plt.plot(o.wl, o.flat)
            plt.plot(o.wl, o.comparison)
            plt.plot(o.wl, o.bias)
#            print(o.wl)
        plt.show()

    if verbose: print("- done         ")

    if DEBUG_PLOTS:
        for o in orders:
            plt.scatter(o.cal_pix, np.log10(o.cal_wl), zorder=10, lw=1)
            plt.plot(o.pix, np.log10(o.wl), zorder=11, lw=1)
        plt.show()

    # estimate spectral resolving power
    dres = estimate_resolution(orders, verbose=verbose, DEBUG_PLOTS=DEBUG_PLOTS)

    if verbose: print("- merging orders")
    wave_merged, flux_merged = merge_orders(orders,
                                            normalize=normalize,
                                            resolution=dres["R_hi"],
                                            DEBUG_PLOTS=DEBUG_PLOTS)

    if apply_barycorr and (radvel is not None):
        wave_merged = wlshift(wave_merged, radvel)

    # construct resolving power column
    res_merged = merge_resolution(wave_merged, orders, dres, DEBUG_PLOTS=DEBUG_PLOTS)

    # this may remove all pixels ...
#    wave_merged, flux_merged = rmcosmics(wave_merged, flux_merged)

    mask = np.isfinite(flux_merged)
    mask[[0, -1]] = False
    wave_merged = wave_merged[mask]
    flux_merged = flux_merged[mask]
    res_merged = res_merged[mask]

    mask = mask_section(flux_merged, tlo=0, thi=0.01, return_mask=True)
    wave_merged = wave_merged[mask]
    flux_merged = flux_merged[mask]
    res_merged = res_merged[mask]

    noise = estimate_noise(wave_merged, flux_merged)

    dout = {"wave": wave_merged,
            "flux": flux_merged,
            "error": noise,
            "res": res_merged}

    return dout


if __name__ == "__main__":
    # spectrum, flats, comps, biases
    bp = "20240901/"
    idcomp_dir = "idcomp_2307/"
    verbose = True
    spec = extract_spectrum(spectrum=bp+"e202409010007.fit",
                            flats=bp+"e202409010019.fit",
                            comps=bp+"e202409010029.fit",
                            biases=bp+"e202409010033.fit",
                            idcomp_dir=idcomp_dir,
                            verbose=verbose)
    print(spec)

