import warnings

import numpy as np
from tools import (curve_fit_reject, polynomial, Gaussian, fill_nan, shared)
from scipy.interpolate import interp1d
from scipy.ndimage import (minimum_filter, maximum_filter, median_filter)
from matplotlib import pyplot as plt
from scipy.special import erf

def gaussian_pixel_weights(y_pixels, y0, sigma):
    """Compute normalized pixel-integrated Gaussian weights for each pixel
    index in y_pixels."""
    y_low = (y_pixels - 0.5 - y0) / (np.sqrt(2) * sigma)
    y_high = (y_pixels + 0.5 - y0) / (np.sqrt(2) * sigma)
    weights = 0.5 * (erf(y_high) - erf(y_low))
    # Normalize so total weight sums to 1
    if np.sum(weights) > 0:
        weights /= np.sum(weights)
    return weights

def gaussian_pixel_weights_2d(y_pixels, y0, sigma, valid):
    """Row-wise version of ``gaussian_pixel_weights``."""
    y_low = (y_pixels - 0.5 - y0) / (np.sqrt(2) * sigma)
    y_high = (y_pixels + 0.5 - y0) / (np.sqrt(2) * sigma)
    weights = 0.5 * (erf(y_high) - erf(y_low))
    weights = np.where(valid, weights, 0.0)
    # Normalize so each row sums to 1
    total = weights.sum(axis=1, keepdims=True)
    np.divide(weights, total, out=weights, where=total > 0)
    return weights


two_log_two = 2 * np.sqrt(2 * np.log(2))

class SpectralOrder:
    def __init__(self, id):
        self.id = id
        self.order_width = []
        self.pixel_x = []
        self.pixel_y = []
        self.pixel_y_err = []
        self.pixel_mask_good = None
        self.solution = None
        self.solution_errors = None
        self.w_fcn = None

        self.pixel_y_cen = None

        self.science = None
        self.flat = None
        self.comparison = None
        self.comparison_orig = None
        self.bias = None

        self.science_err = None
        self.science_var = None

        self.wl = None
        self.pix = None

        self.cal_rms = None
        self.cal_pix = None
        self.cal_wl = None
        self.cal_pix_fwhm = None
        self.cal_wl_fwhm = None

    def __len__(self):
        return len(self.pixel_x)

    def generate_polynomial_solution(self, yerr_default=1.5, verbose=True, DEBUG_PLOTS=False):
        # yerr_default -> maximum rms (in pix) for an acceptable fit


        x = self.pixel_x
        y = self.pixel_y
        kwargs = {"sigma": self.pixel_y_err}
        thres = [100, 5, 5, 3]
        params, errs, mask_good = curve_fit_reject(x, y, polynomial,
                                                   thres=thres, thres_max=0.2,
                                                   **kwargs)
        self.pixel_mask_good = mask_good


        ypoly = polynomial(x, *params)
        # root mean squared
        resid = (y - ypoly)
        nresid = len(resid[mask_good])
        rms = np.sqrt(np.sum(np.square(resid[mask_good])) / nresid)

        self.rms = rms

        # don't show all orders
        if DEBUG_PLOTS and ((self.id < 8) or (self.id > 55)):

            figsize = np.array([8, 6])
            fig, axs = plt.subplots(2, 1, sharex=True,
                                    height_ratios=[3, 1],
                                    figsize=figsize)
            fig.subplots_adjust(hspace=0)
            axs[0].set_ylabel("y  /  pix")
            axs[1].set_xlabel("x  /  pix")
            axs[1].set_ylabel("y - yfit  /  pix")
            axs[0].scatter(x[mask_good], y[mask_good], color="black")
            axs[0].scatter(x[~mask_good], y[~mask_good], color="gray")

            xeval = np.arange(np.min(x), np.max(x)+1)
            yeval = polynomial(xeval, *params)
            axs[0].plot(xeval, yeval, color="red", label="poly")

            rmax = np.max(resid)
            rmin = np.min(resid)
            rbuf = (rmax - rmin) * 0.1
            axs[1].set_ylim(bottom=rmin-rbuf, top=rmax+rbuf)
            axs[1].axhline(y=0, ls="--", color="gray", zorder=10)
            axs[1].scatter(x[mask_good], resid[mask_good], color="black", zorder=20)
            axs[1].scatter(x[~mask_good], resid[~mask_good], color="gray", zorder=20)

            axs[0].text(0.95, 0.9,
                    s="rms = %.3f (thres = %.3f)" % (rms, yerr_default),
                    ha='right', va='center',
                    transform=axs[0].transAxes)

            fig.suptitle("order %d (3rd order poly fit)" % self.id)

            plt.tight_layout()
            plt.show()
        """
        # estimate reduced chi2
        nfree = 4
        dof = nresid - nfree
        # self.pixel_y_err
        chi = resid / yerr_default
        chi2 = np.sqrt(np.sum(np.square(chi)))
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
        if self.pixel_mask_good is not None:
            self.pixel_mask_good = np.array(list(self.pixel_mask_good))

    def aperture(self, ny, nx, times_sigma=2):
        """Pixel indices and weights of the extraction aperture."""
        if self.solution is None:
            raise Exception("Generate a solution first!")
        if self.w_fcn is None:
            self.generate_width_fcn()

        columns = np.arange(nx)

        sigma = self.w_fcn(columns) / two_log_two
        width = times_sigma * sigma
        # extend the aperture to something safe (avoid clipping)
        half_height = np.ceil(width * 3)  # 3-sigma coverage

        y_ind = self.evaluate(columns)
        y_ind_round = np.round(y_ind).astype(int)
        y_min = np.floor(y_ind - half_height).astype(int)
        y_max = np.ceil(y_ind + half_height).astype(int)
        # limit to +-4 pixel; the OES orders are too close together
        ywidth = 4
        if self.id > 8:
            ywidth = 5
        y_min = y_ind_round - np.minimum(ywidth, y_ind_round - y_min)
        y_max = y_ind_round + np.minimum(ywidth, y_max - y_ind_round)

        offsets = np.arange(2 * ywidth)
        y_pixels = y_min[:, None] + offsets[None, :]
        in_aperture = y_pixels < y_max[:, None]
        in_aperture &= (y_pixels >= 0) & (y_pixels < ny)

        # compute proper Gaussian-integrated weights
        weights = gaussian_pixel_weights_2d(y_pixels, y_ind[:, None],
                                            sigma[:, None], in_aperture)
        return columns, y_pixels, weights

    def extract_along_order(self, image, type, times_sigma=2):
        ny, nx = image.shape
        columns, y_pixels, weights = self.aperture(ny, nx,
                                                   times_sigma=times_sigma)

        col = image[np.clip(y_pixels, 0, ny - 1), columns[:, None]].astype(float)

        intensities = np.einsum("ij,ij->i", col, weights)
        self._store(type, intensities)

    def extract_variance_along_order(self, raw, bias, gain, read_noise,
                                     times_sigma=2):
        """Variance of the extracted science spectrum, in ADU^2 per pixel."""
        raw = np.asarray(raw, dtype=float)
        bias = np.asarray(bias, dtype=float)
        ny, nx = raw.shape
        columns, y_pixels, weights = self.aperture(ny, nx,
                                                   times_sigma=times_sigma)

        yc = np.clip(y_pixels, 0, ny - 1)
        counts = raw[yc, columns[:, None]] - bias[yc, columns[:, None]]
        var_pix = np.maximum(counts, 0.0) / gain + (read_noise / gain) ** 2
        self.science_var = np.einsum("ij,ij->i", var_pix, np.square(weights))
        return self.science_var

    def _store(self, type, intensities):
        """File an extracted 1D spectrum under the frame type it came from."""
        if type == "bias" or type == "zero":
            self.bias = intensities
        elif type == "flat":
            self.flat = intensities
        elif type == "comparison" or type == "comp":
            self.comparison = intensities
        elif type == "science":
            self.science = intensities
        else:
            raise Exception("Unknown frame type!")

    def plot_frame_1d(self, type):
        if type == "bias" or type == "zero":
            data_y = self.bias
        elif type == "flat":
            data_y = self.flat
        elif type == "comparison" or type == "comp":
            data_y = self.comparison
        elif type == "comparison_orig" or type == "comp_orig":
            data_y = self.comparison_orig
        elif type == "science":
            data_y = self.science
        else:
            raise Exception("Unknown frame type!")

        data_x = np.arange(len(data_y)) + 1

        if self.wl is not None:
            figsize = np.array([9.5, 7.5])
            fig, axs = plt.subplots(2, 1, sharey=True,
                                    height_ratios=[1, 1],
                                    figsize=figsize)
            axs[0].plot(self.wl, data_y)
            axs[1].plot(data_x, data_y)
            axs[0].set_xlabel(r"$\lambda$  /  $\mathrm{\AA}$")
            axs[1].set_xlabel("x  /  pix")
            axs[1].invert_xaxis()
        else:
            fig, ax = plt.subplots()
            ax.plot(data_x, data_y)
            ax.invert_xaxis()

        fig.suptitle(type + " " + str(self.id))

        plt.tight_layout()
        plt.show()

    def apply_corrections(self, med_win_size=25, min_win_size=15, max_win_size=15,
                          comparison=False, gain=1.0, read_noise=0.0,
                          flat_noise_target=0.02, max_win_flat=601,
                          DEBUG_PLOTS=False):
        """`gain` in e-/ADU and `read_noise` in e- are only used by the
        fallback in the science-error branch; the real propagation happens
        in `extract_variance_along_order`, which sees the raw frame."""

        DEBUG_PLOTS = False

        if DEBUG_PLOTS and (self.bias is not None) and (self.wl is not None):
            plt.plot(self.wl, self.bias, label="bias", color="black")
        if DEBUG_PLOTS and (self.flat is not None) and (self.wl is not None):
            plt.plot(self.wl, self.flat, label="flat")
        if DEBUG_PLOTS and (self.science is not None) and (self.wl is not None):
            plt.plot(self.wl, self.science, label="science")

        if self.flat is not None:
            self.flat -= self.bias
        if self.science is not None:
            self.science -= self.bias
        if self.comparison is not None:
            self.comparison -= self.bias

        if DEBUG_PLOTS and (self.flat is not None) and (self.wl is not None):
            plt.plot(self.wl, self.flat, label="flat - bias")
            if (self.science is not None):
                plt.plot(self.wl, self.science, label="science - bias")
            plt.title("Bias correction of flat")
            plt.xlabel("Wavelength")
            plt.ylabel("Flux")
            plt.legend()
            plt.tight_layout()
            plt.show()

        if self.flat is not None:
            norm_flat = median_filter(self.flat, size=med_win_size)
            resid = self.flat - norm_flat
            flat_sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
            if not np.isfinite(flat_sigma) or flat_sigma <= 0:
                flat_sigma = 0.0

            level = float(np.median(norm_flat))
            if flat_sigma > 0 and level > 0:
                want = (1.253 * flat_sigma / (flat_noise_target * level)) ** 2
                win = int(np.clip(want, med_win_size, max_win_flat))
                if win > med_win_size:
                    norm_flat = median_filter(self.flat, size=win)
                    flat_sigma /= np.sqrt(win / med_win_size)
            self.flat = norm_flat

        if (self.science is not None) and (self.flat is not None):
            with np.errstate(divide="ignore", invalid="ignore"):
                if self.science_var is not None:
                    var_sci = np.asarray(self.science_var, dtype=float)
                else:
                    var_sci = np.abs(self.science) / gain \
                        + (read_noise / gain) ** 2
                var_flat = np.full_like(norm_flat, flat_sigma ** 2)
                ratio = np.divide(self.science, norm_flat,
                                  out=np.full_like(norm_flat, np.inf),
                                  where=norm_flat != 0)
                rel = np.sqrt(
                    np.divide(var_sci, np.square(self.science),
                              out=np.full_like(var_sci, np.inf),
                              where=self.science != 0)
                    + np.divide(var_flat, np.square(norm_flat),
                                out=np.full_like(var_flat, np.inf),
                                where=norm_flat != 0))
                self.science_err = np.abs(ratio) * rel
            self.science = ratio

        if comparison and self.comparison is not None:
            self.comparison_orig = self.comparison.copy()
            qhi = np.quantile(self.comparison_orig, 0.9)
            mask = self.comparison_orig < -qhi
            self.comparison_orig[mask] = np.nan
            self.comparison_orig = fill_nan(self.comparison_orig)

            self.comparison -= minimum_filter(self.comparison, size=min_win_size)
            peak = maximum_filter(self.comparison, size=max_win_size)
            noise = np.median(np.abs(self.comparison))
            if not np.isfinite(noise) or noise <= 0:
                noise = 1.0
            self.comparison /= np.maximum(peak, noise)

            qhi = np.quantile(self.comparison, 0.9)
            mask = self.comparison < -qhi
            self.comparison[mask] = np.nan
            self.comparison = fill_nan(self.comparison)

def extract_order(o_args):
    """Pool worker."""
    o, times_sigma = o_args
    spectrum, flats = shared("spectrum"), shared("flats")
    biases, comps = shared("biases"), shared("comps")
    gain = shared("gain", 1.0)
    read_noise = shared("read_noise", 0.0)
    o.extract_along_order(spectrum, "science", times_sigma=times_sigma)
    o.extract_along_order(flats, "flat", times_sigma=times_sigma)
    o.extract_along_order(biases, "bias", times_sigma=times_sigma)
    o.extract_along_order(comps, "comp", times_sigma=times_sigma)
    raw = shared("spectrum_raw", None)
    if raw is not None:
        o.extract_variance_along_order(raw, biases, gain, read_noise,
                                       times_sigma=times_sigma)
    o.apply_corrections(comparison=True, gain=gain, read_noise=read_noise,
                        DEBUG_PLOTS=False)
    return o

def extract_order_for_calib(args):
    """Pool worker; see `extract_order` for where the frames come from."""
    idx_order, o, times_sigma = args
    biases, comps = shared("biases"), shared("comps")
    o.extract_along_order(biases, "bias", times_sigma=times_sigma)
    o.extract_along_order(comps, "comp", times_sigma=times_sigma)
    o.apply_corrections(comparison=True)
    return idx_order, o
