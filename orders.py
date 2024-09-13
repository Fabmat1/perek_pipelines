import numpy as np
from tools import (curve_fit_reject, polynomial, Gaussian, fill_nan)
from scipy.interpolate import interp1d
from scipy.ndimage import (minimum_filter, maximum_filter, median_filter)

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

#        params, errs = curve_fit(polynomial, self.pixel_x, self.pixel_y, sigma=self.pixel_y_err)
#        errs = np.sqrt(np.diag(errs))

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

        if DEBUG_PLOTS:

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
            axs[0].plot(xeval, yeval, color="red")

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

            fig.suptitle("order %d" % self.id)

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
            self.pixel_mask_good = np.array(list(self.pixel_mask_goodr))

    def extract_along_order(self, image, type, times_sigma=2):
        if self.solution is None:
            raise Exception("Generate a solution first!")
        if self.w_fcn is None:
            self.generate_width_fcn()

        intensities = []

        for pixel in np.arange(image.shape[1]):
            sigma = self.w_fcn(pixel) / two_log_two
            width = times_sigma * sigma
            # limit the order y-width to 4 pixels
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

    def apply_corrections(self, med_win_size=25, min_win_size=15, max_win_size=15):
        self.flat -= self.bias
        self.science -= self.bias
        self.comparison -= self.bias

        norm_flat = median_filter(self.flat, size=med_win_size)
        # norm_flat /= norm_flat.max()

        self.flat = norm_flat

        self.science /= norm_flat

        self.comparison_orig = self.comparison.copy()
        qhi = np.quantile(self.comparison_orig, 0.9)
        mask = self.comparison_orig < -qhi
        self.comparison_orig[mask] = np.nan
        self.comparison_orig = fill_nan(self.comparison_orig)

        self.comparison -= minimum_filter(self.comparison, size=min_win_size)
        # should clip the filter here, force to exceed noise level
        self.comparison /= maximum_filter(self.comparison, size=max_win_size)

        qhi = np.quantile(self.comparison, 0.9)
        mask = self.comparison < -qhi
        self.comparison[mask] = np.nan
        self.comparison = fill_nan(self.comparison)
