import os
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from tools import Gaussian_res, polynomial, curve_fit_reject, pair_generation

two_log_two = 2 * np.sqrt(2 * np.log(2))

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

def fit_comparison(linetable, comparison, pixel_window=10, DEBUG_PLOTS=False):
    line_wls = (linetable[:, 1] + linetable[:, 2]) / 2
    line_px = linetable[:, 0]

    initial_params, _ = curve_fit(polynomial, line_px, line_wls)

    pixels = np.arange(len(comparison)) + 1
    actual_positions = []
    actual_errors = []
    fwhm_pix = []

    for l in line_px:
        px_window = np.logical_and(pixels > l - pixel_window, pixels < l + pixel_window)
        pwin = pixels[px_window]
        intensities = comparison[px_window]

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
        if DEBUG_PLOTS:
            plt.plot(pwin, Gaussian_res(pwin, *params), color="red", zorder=20)

    actual_positions = np.array(actual_positions)
    actual_errors = np.array(actual_errors)
    fwhm_pix = np.array(fwhm_pix)

    if DEBUG_PLOTS:
        plt.plot(pixels, comparison, zorder=10)
        plt.show()

    dout = {"actual_positions": actual_positions,
            "actual_errors": actual_errors,
            "fwhm_pix": fwhm_pix,
            "line_wls": line_wls}

    return dout

def mask_good_lines(actual_positions, fwhm_pix,
                    too_narrow_pix=2.5,
                    too_wide_pix=7.0,
                    DEBUG_PLOTS=False):

    mask_good = (fwhm_pix < too_wide_pix) & (fwhm_pix > too_narrow_pix)
#    mask_good = np.ones(len(fwhm_pix)).astype(bool)

    not_enough_lines = (np.sum(mask_good) < 5)
    if not_enough_lines:
        mask_good = (fwhm_pix < 9) & (fwhm_pix > 2)
    not_enough_lines = (np.sum(mask_good) < 5)
    if not_enough_lines:
        mask_good = np.ones(len(fwhm_pix)).astype(bool)
    if (np.sum(mask_good) < 5):
        raise Exception("Not enough calibration lines in order %d" % order.id)

    # threshold = np.percentile(actual_errors, 0.9)
    # actual_positions = actual_positions[actual_errors <= threshold]
    # actual_errors = actual_errors[actual_errors <= threshold]

    if DEBUG_PLOTS:
        plt.scatter(actual_positions[~mask_good], fwhm_pix[~mask_good], zorder=10, color="gray")
        plt.scatter(actual_positions[mask_good], fwhm_pix[mask_good], zorder=11, color="black")
        plt.axhline(too_wide_pix, ls="--", c="red")
        plt.axhline(too_narrow_pix, ls="--", c="red")
        plt.show()

    return mask_good

def fit_dispersion(x, y, yerr, DEBUG_PLOTS=False):

    isort = np.argsort(x)
    x = x[isort]
    y = y[isort]
    yerr = yerr[isort]

    kwargs = {"sigma": yerr}
    thres = [10, 5, 3, 2]
    thres_max = 0.1
    # TODO: extrapolate solutions to adjacent orders, iterative identification with ThAr lines
    params, errs, mask_good = curve_fit_reject(x, y, polynomial,
                                               thres=thres, thres_max=thres_max,
                                               **kwargs)

    ypoly = polynomial(x, *params)
    # root mean squared
    resid = (y - ypoly)
    nresid = len(resid)
    rms = np.sqrt(np.sum(np.square(resid)) / nresid)

    if DEBUG_PLOTS:
#    if True:

        figsize = np.array([8, 6])
        fig, axs = plt.subplots(2, 1, sharex=True,
                                height_ratios=[3, 1],
                                figsize=figsize)
        fig.subplots_adjust(hspace=0)
#        axs[0].invert_yaxis()
        axs[0].set_ylabel("y  /  Angstrom")
        axs[1].set_xlabel("x  /  pix")
        axs[1].set_ylabel("y - yfit  /  Angstrom")
        axs[0].scatter(x[mask_good], y[mask_good], color="black")
        axs[0].scatter(x[~mask_good], y[~mask_good], color="gray")
        axs[0].plot(x, ypoly, color="red")

        rmax = np.max(resid)
        rmin = np.min(resid)
        rbuf = (rmax - rmin) * 0.1
        axs[1].set_ylim(bottom=rmin-rbuf, top=rmax+rbuf)
        axs[1].axhline(y=0, ls="--", color="gray", zorder=10)
        axs[1].scatter(x[mask_good], resid[mask_good], color="black", zorder=20)
        axs[1].scatter(x[~mask_good], resid[~mask_good], color="gray", zorder=20)

        axs[0].text(0.95, 0.9,
                s="rms = %.3f" % (rms),
                ha='right', va='center',
                transform=axs[0].transAxes)

        fig.suptitle("order " + str(order.id))

        plt.tight_layout()
        plt.show()

    dout = {"params": params,
            "errs": errs,
            "mask_good": mask_good,
            "rms": rms}

    return dout


def solve_wavelength(linetable, order, pixel_window=10, DEBUG_PLOTS=False):

    # determine the arc line positions in pixels, given starting values
    lfit = fit_comparison(linetable, order.comparison,
                          pixel_window=10, DEBUG_PLOTS=DEBUG_PLOTS)

    actual_positions = lfit["actual_positions"]
    actual_errors = lfit["actual_errors"]
    fwhm_pix = lfit["fwhm_pix"]
    line_wls = lfit["line_wls"]
    pixels = np.arange(len(order.comparison)) + 1

    # this depends on the spectrograph!
    # OES has 7x sampling in the blue ...
    too_wide_pix = 7
    too_narrow_pix = 2.5
    mask_good = mask_good_lines(actual_positions, fwhm_pix,
                                too_narrow_pix=too_narrow_pix,
                                too_wide_pix=too_wide_pix,
                                DEBUG_PLOTS=DEBUG_PLOTS)

    if (np.sum(mask_good) < 5):
        raise Exception("Not enough calibration lines in order %d" % order.id)

    actual_positions = actual_positions[mask_good]
    actual_errors = actual_errors[mask_good]
    fwhm_pix = fwhm_pix[mask_good]
    line_wls = line_wls[mask_good]

    # solve the dispersion relation
    disp = fit_dispersion(x=actual_positions,
                          y=line_wls,
                          yerr=actual_errors,
                          DEBUG_PLOTS=DEBUG_PLOTS)

    params = disp["params"]
    mask_good = disp["mask_good"]

    actual_positions = actual_positions[mask_good]
    actual_errors = actual_errors[mask_good]
    fwhm_pix = fwhm_pix[mask_good]
    line_wls = line_wls[mask_good]

    # average pixel width in angstroms
    pix_width = [np.mean(np.diff(polynomial(np.arange(3)+i-1,*params))) for i in actual_positions]
    pix_width = np.abs(pix_width)
    fwhm_angstrom = fwhm_pix * np.array(pix_width)

    order.wl = polynomial(pixels, *params)
    order.cal_pix = actual_positions
    order.cal_wl = line_wls
    order.cal_pix_fwhm = fwhm_pix
    order.cal_wl_fwhm = fwhm_angstrom

    if DEBUG_PLOTS:
        # plot spectral resolution
        plt.scatter(order.cal_wl, order.cal_wl/fwhm_angstrom, zorder=10)
        plt.show()


def find_dispersion(orders, biases, comps,
                    idcomp_dir, idcomp_offset=-15,
                    verbose=False, DEBUG_PLOTS=False):

    npix_x = biases.shape[1]
    # estimate y-pos of each order at the center of the x axis
    ap_measure = []
    for o in orders:
        ap = float(polynomial(npix_x / 2, *o.solution))
        o.pixel_y_cen = ap
        ap_measure.append(ap)

    times_sigma = 2

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

    if verbose: print("- extracting orders")

    # Extract different spectra
#    for o in orders:
    for p in id_order_pairs:
        idx_order = p[1]
        o = orders[idx_order]
        if verbose: print("- order", o.id, end="\r")
        o.extract_along_order(biases, "bias", times_sigma=times_sigma)
        o.extract_along_order(comps, "comp", times_sigma=times_sigma)
        o.apply_corrections(comparison=True)

#        o.plot_frame_1d("comp_orig")

    if verbose: print("- solving dispersion relations")
    for p in id_order_pairs:
        idx_id = p[0]
        idx_order = p[1]
        o = orders[idx_order]
        if verbose: print("- order", o.id, end="\r")
        key = avg_aps[idx_id]
        linelist_o = linelists[key]
        solve_wavelength(linelist_o, o)
        o.pix = np.arange(len(o.wl))

#        o.plot_frame_1d("comp_orig")

    return orders
