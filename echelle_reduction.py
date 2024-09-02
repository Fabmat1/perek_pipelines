import os
import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from scipy.constants import speed_of_light
from spectres import spectres
from matplotlib import pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import minimum_filter, maximum_filter, median_filter, gaussian_filter, uniform_filter1d
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import curve_fit
from astropy.stats import sigma_clip
from estimate_noise import estimate_noise

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

        self.science = None
        self.flat = None
        self.comparison = None
        self.bias = None

        self.cal_pix = None
        self.cal_wl = None
        self.wl = None
        self.cal_pix_fwhm = None
        self.cal_wl_fwhm = None

    def __len__(self):
        return len(self.pixel_x)

    def generate_polynomial_solution(self):
        params, errs = curve_fit(polynomial, self.pixel_x, self.pixel_y, sigma=self.pixel_y_err)
        errs = np.sqrt(np.diag(errs))

        self.solution = params
        self.solution_errors = errs

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
        self.generate_width_fcn()

        intensities = []

        for pixel in np.arange(image.shape[1]):
            sigma = self.w_fcn(pixel) / two_log_two
            width = times_sigma * sigma
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
    return spectres(x, xfull, yfull, fill=0, verbose=False)

def pair_generation(arr1, arr2):
    pairs = []

    distance_matrix = np.abs(arr1[:, np.newaxis] - arr2[np.newaxis, :])

    for i, row in enumerate(distance_matrix):
        next_index = np.argmin(row)
        if np.min(distance_matrix[:, next_index]) == row[next_index]:
            pairs.append((i, int(next_index)))
        else:
            pairs.append((i, None))

    return pairs


def assign_orders(slicelist: list[SpectralSlice], max_ind, DEBUG_PLOTS=False, **kwargs):
    # Forward loop
    slicelist[max_ind].order_ownership = np.arange(len(slicelist[max_ind].ys)) + 1

    curr_slice = slicelist[max_ind]

    while curr_slice.next_slice is not None:
        curr_slice.clean_ownership()

        results = pair_generation(curr_slice.ys, curr_slice.next_slice.ys)

        curr_slice.next_slice.gen_empty_ownership()

        for p in results:
            if p[1] is None:
                continue
            curr_slice.next_slice.order_ownership[p[1]] = curr_slice.order_ownership[p[0]]
            if DEBUG_PLOTS:
                plt.plot([curr_slice.x, curr_slice.next_slice.x], [curr_slice.ys[p[0]], curr_slice.next_slice.ys[p[1]]], color="red")

        curr_slice = curr_slice.next_slice
    else:
        curr_slice.clean_ownership()

    curr_slice = slicelist[max_ind]

    while curr_slice.previous_slice is not None:
        curr_slice.clean_ownership()

        results = pair_generation(curr_slice.ys, curr_slice.previous_slice.ys)
        curr_slice.previous_slice.gen_empty_ownership()

        for p in results:
            if p[1] is None:
                continue
            curr_slice.previous_slice.order_ownership[p[1]] = curr_slice.order_ownership[p[0]]
            if DEBUG_PLOTS:
                plt.plot([curr_slice.x, curr_slice.previous_slice.x], [curr_slice.ys[p[0]], curr_slice.previous_slice.ys[p[1]]], color="red")

        curr_slice = curr_slice.previous_slice
    else:
        curr_slice.clean_ownership()

    orders = {}
    for o in slicelist[max_ind].order_ownership:
        o = int(o)
        orders[o] = SpectralOrder(o)

    for s in slicelist:
        for i, y in enumerate(s.ys):
            try:
                this_owner = int(s.order_ownership[i])
                orders[this_owner].pixel_y.append(y)
                orders[this_owner].pixel_x.append(s.x)
                orders[this_owner].pixel_y_err.append(s.y_errs[i])
                orders[this_owner].order_width.append(s.widths[i])
            except KeyError:
                this_owner = int(s.order_ownership[i])
                orders[this_owner] = SpectralOrder(this_owner)
                orders[this_owner].pixel_y.append(y)
                orders[this_owner].pixel_x.append(s.x)
                orders[this_owner].pixel_y_err.append(s.y_errs[i])
                orders[this_owner].order_width.append(s.widths[i])

    olist = list(orders.values())

    for o in olist:
        o.sort_self()

    return olist


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


def slice_analysis(pixel, slice_x, slice_y, MIN_WINDOW=15, MAX_WINDOW=15, NOISE_MEASURE_SECTION_WIDTH=0.05,
                   NOISE_CUTOFF=20, CUTTOFF_MARGIN=5, ORDER_GAUSS_THRESHOLD=0.6, DEBUG_PLOTS=False, **kwargs):
    min_slice = minimum_filter(slice_y, size=MIN_WINDOW)

    slice_y -= min_slice

    lower_section = slice_y[:int(len(slice_y) * NOISE_MEASURE_SECTION_WIDTH)]
    upper_section = slice_y[int(len(slice_y) * (1 - NOISE_MEASURE_SECTION_WIDTH)):]

    noise_lvl = (np.std(lower_section) + np.std(upper_section)) / 2
    noise_lvl *= NOISE_CUTOFF

    noise_indices = np.where(slice_y > noise_lvl)[0]
    first_cross = noise_indices[0]
    last_cross = noise_indices[-1]

    lo_ind = first_cross - CUTTOFF_MARGIN
    hi_ind = last_cross + CUTTOFF_MARGIN

    if DEBUG_PLOTS:
        plt.plot(slice_x, slice_y)
        plt.axhline(noise_lvl, color="orange")
        plt.axvline(lo_ind, color='r')
        plt.axvline(hi_ind, color='r')
        plt.tight_layout()
        plt.show()

    slice_y = slice_y[lo_ind:hi_ind]
    slice_x = slice_x[lo_ind:hi_ind]

    max_slice = maximum_filter(slice_y, size=MAX_WINDOW)
    slice_y /= max_slice

    filtered_indices = slice_x[slice_y > ORDER_GAUSS_THRESHOLD]
    peaks = np.split(filtered_indices, np.where(np.diff(filtered_indices) != 1)[0] + 1)
    peaks = [peak for peak in peaks if len(peak) > 1]

    peak_locations = [np.mean(p) for p in peaks]
#    print(f"Identified {len(peaks)} orders @ x = {pixel}")

    if DEBUG_PLOTS:
        for l in peak_locations:
            plt.axvline(l, color="gray")
        plt.plot(slice_x, slice_y)
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
            plt.plot(x_neighborhood, Gaussian(x_neighborhood, *params), color="r")
        fit_params.append(params)

        errs = np.sqrt(np.diag(errs))

        refined_peak_locations.append(params[1])
        location_uncertainties.append(errs[1])
        widths.append(two_log_two * params[2])

    if DEBUG_PLOTS:
        plt.plot(slice_x, slice_y)
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


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


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


def find_neighbors(arr, target):
    # Ensure the array is sorted
    arr = np.sort(arr)

    # Use np.searchsorted to find the index where the target would be inserted
    idx = np.searchsorted(arr, target)

    # Get the neighbors
    if idx == 0:
        # Target is smaller than the smallest element
        return None, arr[0]
    elif idx == len(arr):
        # Target is larger than the largest element
        return arr[-1], None
    else:
        # Normal case, the target is between two elements
        return arr[idx - 1], arr[idx]

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

def proper_normalization(wls, flxs,
                         ignore_windows=[(4090, 4110), (4320, 4355),
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


def merge_orders(olist: list[SpectralOrder], normalize=False, margin=20, uppermost_wl=8500,
                 resolution=50000, DEBUG_PLOTS=False):
    common_wl_array = [o.wl[margin:-margin] for o in olist if o.wl.min() < uppermost_wl]
    common_flx_array = [o.science[margin:-margin] for o in olist if o.wl.min() < uppermost_wl]

    if DEBUG_PLOTS:
        for w, f in zip(common_wl_array, common_flx_array):
            plt.plot(w, f)
    common_flx_array = proper_normalization(common_wl_array, common_flx_array)

    common_wl_array = np.concatenate(common_wl_array)
    common_flx_array = np.concatenate(common_flx_array)

    mask = np.argsort(common_wl_array)
    common_wl_array = common_wl_array[mask]
    common_flx_array = common_flx_array[mask]

    new_grid = generate_wl_grid(common_wl_array, resolution=resolution)
    common_flx_array = spectres(new_grid, common_wl_array, common_flx_array)
    common_wl_array = new_grid

    if DEBUG_PLOTS:
        plt.plot(common_wl_array, common_flx_array)
        plt.tight_layout()
        plt.show()

    return common_wl_array, common_flx_array

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

def extract_spectrum(spectrum, flats, comps, biases, idcomp_offset=-15,
                     normalize=False, idcomp_dir="idcomp",
                     sampling=75, min_order_samples=10,
                     apply_barycorr=True,
                     DEBUG_PLOTS=False, **kwargs):
    spectrum, radvel = open_or_coadd_frame(spectrum, True)
    flats = open_or_coadd_frame(flats)
    comps = open_or_coadd_frame(comps)
    biases = open_or_coadd_frame(biases)

    slices = []

    # Get orders and stuff from flat
    for i in range(sampling):
        pixel = i * int(flats.shape[1] / sampling) + int(flats.shape[1] / (2 * sampling)) + 1
        slice_y = flats[:, pixel - 1].astype(float)
        slice_x = np.arange(flats.shape[1])
        slices.append(slice_analysis(pixel - 1, slice_x, slice_y), **kwargs)

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

    orders = assign_orders(slices, max_slice, **kwargs)

    o_to_be_removed = []
    for i, o in enumerate(orders):
        if len(o) > min_order_samples:
            o.generate_polynomial_solution()
        else:
            o_to_be_removed.append(i)

    orders = [item for i, item in enumerate(orders) if i not in o_to_be_removed]

    print(f"{len(orders)} orders found")

    if DEBUG_PLOTS:
        plt.imshow(flats, zorder=1, cmap='gray')
        for slice in slices:
            plt.scatter([np.repeat(slice.x, len(slice.ys))], slice.ys, marker="x", zorder=2)
        for o in orders:
            # plt.scatter(o.pixel_x, o.pixel_y, marker="x", zorder=2)
            plt.plot(np.linspace(0, flats.shape[1], 2000), o.evaluate(np.linspace(0, flats.shape[1], 2000)))
        plt.tight_layout()
        plt.show()

    # Extract different spectra

    linelists = {}
    for file in os.listdir(idcomp_dir):
        if "idiazcomp" in file:
            aplo, aphi, table = parse_idcomp(idcomp_dir + "/" + file)

            avg_ap = (aplo + aphi) / 2

            linelists[avg_ap+idcomp_offset] = table

    avg_aps = np.array(list(linelists.keys()))

    if DEBUG_PLOTS:
        plt.imshow(flats)
        for key in linelists.keys():
            plt.scatter([len(spectrum) / 2], [key], marker="x", zorder=2, color="red")
        plt.show()

    for o in orders:
        o.extract_along_order(spectrum, "science")
        o.extract_along_order(flats, "flat")
        o.extract_along_order(biases, "bias")
        o.extract_along_order(comps, "comp")

        o.apply_corrections()

        # o.plot_frame("science")
        # o.plot_frame("flat")
        # o.plot_frame("bias")
        # o.plot_frame("comp")

        ap_measure = polynomial(len(spectrum) / 2, *o.solution)
        key = find_nearest(avg_aps, ap_measure)
        linelist = linelists[key]

        solve_wavelength(linelist, o)

    if DEBUG_PLOTS:
        for o in orders:
            plt.scatter(o.cal_pix, np.log10(o.cal_wl), zorder=10, lw=1)
            plt.plot(np.arange(len(o.wl))+1, np.log10(o.wl), zorder=11, lw=1)
        plt.show()

    p = 0.6827
    pl = 0.+0.5*(1.-p)
    ph = 1.-0.5*(1.-p)
    if DEBUG_PLOTS:
        wl_med = []
        res_med = []
        res_qlo = []
        res_qhi = []
        res = []
        for o in orders:
            res_o = o.cal_wl/o.cal_wl_fwhm

#            plt.scatter(o.cal_wl, o.cal_wl_fwhm, zorder=10, lw=1)
            plt.scatter(o.cal_wl, res_o, zorder=10, lw=1)

            quantile_lo = np.quantile(res_o, pl)
            quantile_hi = np.quantile(res_o, ph)
            wl_med.append(np.median(o.cal_wl))
            res_med.append(np.median(res_o))
            res_qlo.append(quantile_lo)
            res_qhi.append(quantile_hi)
        plt.plot(wl_med, res_med, color="black", lw=2, zorder=20, ls="-")
        plt.plot(wl_med, res_qlo, color="black", lw=2, zorder=20, ls="--")
        plt.plot(wl_med, res_qhi, color="black", lw=2, zorder=20, ls="--")
        plt.xlabel(r"Wavelength  /  $\mathrm{\AA}$")
        plt.ylabel(r"$\lambda$  /  $\Delta \lambda$")
        plt.show()


    # get rid of bad orders at edges
    order_ltrim = 0
#    order_rtrim = -10
    order_rtrim = 47
    orders = orders[order_ltrim:order_rtrim]
    print("> using %d orders" % len(orders))

    # estimate the median resolving power
    res = []
    for o in orders:
        res_o = o.cal_wl/o.cal_wl_fwhm
        res.extend(res_o)
    res_qlo = np.quantile(res, pl)
    res_qhi = np.quantile(res, ph)
    res_med = np.median(res)

    # plot_order_list(orders)
    print("R = %.0f^{+%.0f}_{-%.0f}" % (res_med, res_qhi-res_med, res_med-res_qlo))
    wl_final, wl_flux = merge_orders(orders, resolution=res_qhi, DEBUG_PLOTS=DEBUG_PLOTS)

    if apply_barycorr and (radvel is not None):
        wl_final = wlshift(wl_final, radvel)

    # this may remove all pixels ...
#    wl_final, wl_flux = rmcosmics(wl_final, wl_flux)

    mask = np.isfinite(wl_flux)

    wl_final = wl_final[mask]
    wl_flux = wl_flux[mask]

    return wl_final, wl_flux


if __name__ == "__main__":
    extract_spectrum("e202109060016.fit", "e202109060010.fit", "e202109060011.fit", "e202109060004.fit")
