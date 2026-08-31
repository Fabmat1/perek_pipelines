
import os
import warnings
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from tools import (Gaussian_res, polynomial, curve_fit_reject, pair_generation,
                   shared_pool, shared, publish_shared)
from orders import extract_order_for_calib
from grating import enforce_invariant
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from scipy.interpolate import interp1d


two_log_two = 2 * np.sqrt(2 * np.log(2))


def load_thar_list(*paths):
    """Read one or more ThAr line lists into the DataFrame the fit expects.

    Two formats are understood:

    * the cleaned Lovis & Pepe CSV, which already has a ``wave_air`` column;
    * the Murphy et al. (2007) atlas, whitespace-separated, whose second
      column is the air wavelength (its first column is the wavenumber, and
      1e8/wavenumber recovers the *vacuum* value -- the two differ by the
      refractive index of air, ~2.8e-4, which is 1.4 A at 5000 A and would
      quietly bias every wavelength if the wrong column were used).

    Lovis & Pepe ends at 6912 A, so the reddest OES orders have no lines in it
    at all. Passing both merges them: lines closer than 0.01 A are treated as
    the same line and only the first list's value is kept.

    Note when comparing runs: adding lines *lowers* the reported resolution,
    because R is measured from the fitted width of the calibration lines
    themselves. With six lines an order is calibrated on the few brightest and
    narrowest features; with twenty-six the sample includes normal-width lines
    and the median widens. The lower number is the more honest estimate of the
    instrument, and the fit it comes from is far better constrained.
    """
    import pandas as pd

    frames = []
    for path in paths:
        if path is None:
            continue
        with open(path) as fh:
            head = fh.readline()
        if "," in head and "wave_air" in head:
            frames.append(pd.read_csv(path))
            continue
        # Murphy: wavenumber, air wavelength, uncertainty, species, flag.
        # The third column is carried through as `wave_err_raw` and is
        # deliberately *not* called an Angstrom uncertainty: its values run
        # 1.4-4.5, which is far too large to be Angstroms and is consistent
        # with either 1e-3 cm^-1 or m/s. Nothing reads it. Check the paper
        # before using it as a fit weight -- taking it for Angstroms would
        # weight every line by a number three orders of magnitude too big.
        arr = np.loadtxt(path, usecols=(0, 1, 2))
        frames.append(pd.DataFrame({"wave_air": arr[:, 1],
                                    "wave_vac": 1e8 / arr[:, 0],
                                    "wave_err_raw": arr[:, 2]}))

    if not frames:
        return None
    if len(frames) == 1:
        return frames[0].sort_values("wave_air").reset_index(drop=True)

    merged = frames[0]
    for extra in frames[1:]:
        have = np.sort(merged["wave_air"].to_numpy())
        w = extra["wave_air"].to_numpy()
        idx = np.searchsorted(have, w)
        left = np.abs(w - have[np.clip(idx - 1, 0, len(have) - 1)])
        right = np.abs(w - have[np.clip(idx, 0, len(have) - 1)])
        new = np.minimum(left, right) > 0.01
        merged = pd.concat([merged, extra[new]], ignore_index=True)
    return merged.sort_values("wave_air").reset_index(drop=True)


def parse_idcomp(file_path):
    """Read one IRAF ``identify`` database file.

    IRAF appends a record every time the aperture is saved, so a file can hold
    several: the 2023 lists were trimmed to the last record before they were
    committed, the 2026 ones from the observatory were not. Each record
    supersedes the one before it, so only the last is kept -- concatenating
    them would feed the fit the same line several times over at slightly
    different centres.
    """
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
            # start of a new record: drop whatever the previous one held
            aplow = float(line.split()[1])
            aphigh = None
            table_data = []
            in_table = False
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

    if not table_data:
        # pixel, fitted wavelength, catalogue wavelength, weight, ...
        return aplow, aphigh, np.empty((0, 6), dtype=float)
    return aplow, aphigh, np.array(table_data, dtype=float)

def fit_comparison(linetable, comparison, pixel_window=8, DEBUG_PLOTS=False,
                   raw=None, saturation=60000.0):
    """Fit the arc lines of one order.

    `raw` is the same order's arc in counts, before the running-maximum
    normalisation. It is used only to drop saturated lines: a clipped line has
    a flat top, so its fitted centre is set by wherever the plateau happens to
    be brightest rather than by the line, and its width is overestimated. Which
    lines saturate depends on the lamp and the exposure, not on the line, so
    this has to be decided per exposure rather than from a fixed list.
    """
#    DEBUG_PLOTS = True

    line_wls = (linetable[:, 1] + linetable[:, 2]) / 2
    line_px = linetable[:, 0]
    mask_good = (line_px>=0) & (line_wls>0)
    line_px = line_px[mask_good]
    line_wls = line_wls[mask_good]

    if DEBUG_PLOTS:
        plt.vlines(line_px, ymin=1.1, ymax=1.15, color="tab:orange")

    pixels = np.arange(len(comparison)) + 1
    actual_positions = []
    actual_errors = []
    fwhm_pix = []

    kept = []
    for idx, l in enumerate(line_px):
        px_window = np.logical_and(pixels > l - pixel_window,
                                   pixels < l + pixel_window)
        pwin = pixels[px_window]
        intensities = comparison[px_window]

        if raw is not None:
            seg = np.asarray(raw, dtype=float)[px_window]
            if seg.size:
                top = float(np.max(seg))
                # at the detector ceiling, or flat-topped below it
                if top >= saturation or \
                   (top > 0 and np.sum(seg > 0.98 * top) >= 3):
                    continue

        psigma_ini = 3.5 / 2.355
        psigma_min = 1.5 / 2.355
        psigma_max = min(9.5, 2*pixel_window) / 2.355

        params, errs = curve_fit(Gaussian_res, pwin, intensities,
                                 p0=[1, l, psigma_ini],
                                 bounds=[
                                     [0, l - pixel_window / 2, psigma_min],
                                     [np.inf, l + pixel_window / 2, psigma_max]
                                 ],
                                 maxfev=100000)

        errs = np.sqrt(np.diag(errs))

        actual_positions.append(params[1])
        actual_errors.append(errs[1])
        fwhm_pix.append(params[2]*two_log_two)
        # skipping a saturated line above would otherwise leave `line_wls`
        # longer than the fitted arrays, silently pairing every later
        # measurement with the wrong catalogue wavelength. Keep indices, not
        # values: two lines can share a pixel position.
        kept.append(idx)
        if DEBUG_PLOTS:
            plt.plot(pwin, Gaussian_res(pwin, *params), color="red", zorder=20)

    actual_positions = np.array(actual_positions)
    actual_errors = np.array(actual_errors)
    fwhm_pix = np.array(fwhm_pix)
    if raw is not None:
        line_wls = line_wls[np.array(kept, dtype=int)] if kept \
            else line_wls[:0]

    if DEBUG_PLOTS:
        plt.plot(pixels, comparison, zorder=10)
        plt.title("Arc line fits")
        plt.xlabel("Extracted X pixel")
        plt.ylabel("Renormalised counts")
        plt.tight_layout()
        plt.show()

    dout = {"actual_positions": actual_positions,
            "actual_errors": actual_errors,
            "fwhm_pix": fwhm_pix,
            "line_wls": line_wls}

    return dout

def mask_good_lines(actual_positions, fwhm_pix,
                    too_narrow_pix=2.6,
                    too_wide_pix=7.0,
                    order_id=None,
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
        raise Exception("Not enough calibration lines in order %s" % order_id)

    # threshold = np.percentile(actual_errors, 0.9)
    # actual_positions = actual_positions[actual_errors <= threshold]
    # actual_errors = actual_errors[actual_errors <= threshold]

    if DEBUG_PLOTS:
        plt.scatter(actual_positions[~mask_good], fwhm_pix[~mask_good], zorder=10, color="gray")
        plt.scatter(actual_positions[mask_good], fwhm_pix[mask_good], zorder=11, color="black")
        plt.axhline(too_wide_pix, ls="--", c="red")
        plt.axhline(too_narrow_pix, ls="--", c="red")
        plt.title("Arc fit rejection")
        plt.xlabel("Extracted X pixel")
        plt.xlabel("Gaussian FWHM (in pix)")
        plt.show()

    return mask_good


def wavelength_to_pixel(wavelengths, params, x, polynomial):
    # evaluate polynomial across your pixel grid
    wl = polynomial(x, *params)
    # make a monotonic interpolator (λ→x)
    f = interp1d(wl, x, bounds_error=False, fill_value=np.nan)
    return f(wavelengths)


def _monotonic(wl, max_wrong=0.0):
    """True if `wl` runs one way across the whole detector.

    `max_wrong` is the tolerated fraction of steps against the trend.
    """
    dw = np.diff(np.asarray(wl, float))
    fin = np.isfinite(dw)
    if fin.sum() < 2:
        return True
    up = int(np.sum(dw[fin] > 0))
    dn = int(np.sum(dw[fin] < 0))
    if up + dn == 0:
        return False
    return min(up, dn) / float(up + dn) <= max_wrong


def fit_dispersion(x, y, yerr, thar_list=None, DEBUG_PLOTS=False):

    # sort for the fit and the residual plot, but remember the permutation:
    # the mask returned below has to line up with the caller's arrays, not with
    # the sorted copies made here
    isort = np.argsort(x)
    x = x[isort]
    y = y[isort]
    yerr = yerr[isort]

    kwargs = {"sigma": yerr}
    thres = [10, 5, 3, 2]
    thres_max = 0.025
    # TODO: extrapolate solutions to adjacent orders, iterative identification with ThAr lines
    params, errs, mask_good = curve_fit_reject(x, y, polynomial,
                                               thres=thres, thres_max=thres_max,
                                               **kwargs)

    # A dispersion relation cannot turn round, but the cubic is fitted to as
    # few as a dozen lines and will put a stationary point inside the detector
    # if they cluster. Such a fit has a normal median dispersion, so only the
    # sign changes reveal it. Lower the degree until it is monotonic; a line
    # never folds, so this terminates.
    npix = int(np.nanmax(x)) + 1 if np.isfinite(np.nanmax(x)) else 2048
    grid = np.arange(max(npix, 2), dtype=float)
    if not _monotonic(polynomial(grid, *params)):
        # `polynomial` takes a fixed four coefficients, so refit with polyfit
        # on the already-accepted lines and pad back to four.
        keep = mask_good if mask_good.sum() >= 4 else np.ones(len(x), bool)
        w = 1.0 / np.where(yerr[keep] > 0, yerr[keep], np.inf)
        for deg in (2, 1):
            if keep.sum() < deg + 2:
                continue
            try:
                coef = np.polyfit(x[keep], y[keep], deg, w=w)
            except Exception:
                continue
            trial = np.concatenate([np.zeros(4 - (deg + 1)), coef])
            if _monotonic(polynomial(grid, *trial)):
                params = trial
                errs = np.full(4, np.nan)
                warnings.warn(
                    "dispersion fit was not monotonic across the detector; "
                    "degree lowered to %d" % deg)
                break

    ypoly = polynomial(x, *params)
    # root mean squared
    resid = (y - ypoly)
    nresid = len(resid)
    rms = np.sqrt(np.sum(np.square(resid)) / nresid)

    if DEBUG_PLOTS:
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

#        fig.suptitle("order " + str(order.id))

        plt.tight_layout()
        plt.show()

    # `mask_good` indexes the sorted arrays; the caller still holds the
    # unsorted ones. Undo the permutation so that applying the mask there
    # keeps the lines the fit actually kept. Without this the clip silently
    # discards a different set of lines than the one it rejected -- harmless
    # while the input happens to be sorted by pixel, but the ThAr refinement
    # appends its predicted lines with `vstack` and so does not.
    mask_orig = np.zeros_like(mask_good)
    mask_orig[isort] = mask_good

    dout = {"params": params,
            "errs": errs,
            "mask_good": mask_orig,
            "rms": rms}

    return dout

def solve_wavelength(linetable, order,
                     pixel_window=8,
                     thar_list=None,
                     max_iterations=3,
                     DEBUG_PLOTS=False):
    """
    Fit ThAr lines in a spectral order to solve the dispersion relation.

    Parameters
    ----------
    linetable : array-like
        Initial guess of line positions.
    order : object
        Spectral order with .comparison array and .id attribute.
    pixel_window : int
        Pixel window to search around guessed positions.
    thar_list : DataFrame
        Reference ThAr line list (optional).
    max_iterations : int
        Maximum iterations for fitting with new ThAr lines.
    DEBUG_PLOTS : bool
        Whether to show debug plots.
    """

    # work on a copy: `linetable` is owned by the shared `linelists` dict and
    # is mutated below, which would leak between orders in a sequential run
    linetable = np.array(linetable, dtype=float, copy=True)

    pixels = np.arange(len(order.comparison)) + 1
    too_wide_pix = 7
    too_narrow_pix = 2.5

    ngoods = []
    for iteration in range(max_iterations):
        # Fit comparison lines
        lfit = fit_comparison(linetable, order.comparison,
                              raw=getattr(order, "comparison_orig", None),
                              pixel_window=pixel_window,
                              DEBUG_PLOTS=DEBUG_PLOTS)

        actual_positions = lfit["actual_positions"]
        actual_errors = lfit["actual_errors"]
        fwhm_pix = lfit["fwhm_pix"]
        line_wls = lfit["line_wls"]

        # Mask out bad lines
        mask_good = mask_good_lines(actual_positions, fwhm_pix,
                                    too_narrow_pix=too_narrow_pix,
                                    too_wide_pix=too_wide_pix,
                                    order_id=order.id,
                                    DEBUG_PLOTS=DEBUG_PLOTS)

        if np.sum(mask_good) < 5:
            raise Exception(f"Not enough calibration lines in order {order.id}")

        actual_positions = actual_positions[mask_good]
        actual_errors = actual_errors[mask_good]
        fwhm_pix = fwhm_pix[mask_good]
        line_wls = line_wls[mask_good]

        # Fit the dispersion relation
        disp = fit_dispersion(x=actual_positions,
                              y=line_wls,
                              yerr=actual_errors,
                              thar_list=thar_list,
                              DEBUG_PLOTS=DEBUG_PLOTS)
        params = disp["params"]
        mask_good_disp = disp["mask_good"]
        ngood = np.sum(mask_good_disp)
        ngoods.append(ngood)

        # Update pixel positions in linetable using current fit
        # Only first 3 entries matter; first is pixel, second/third are wavelength
        linetable[:, 0] = wavelength_to_pixel((linetable[:, 1]+linetable[:, 2])/2,
                                              params, pixels, polynomial)

        # If no ThAr list or last iteration, break
        if thar_list is None or iteration == max_iterations - 1:
            break

        # Predict ThAr positions from current fit
        ythar = thar_list["wave_air"].to_numpy()
        ypoly = polynomial(pixels, *params)
        wmin, wmax = np.min(ypoly) + 1, np.max(ypoly) - 1
        ythar_mask = np.logical_and(ythar > wmin, ythar < wmax)
        if np.sum(ythar_mask) == 0:
            break
        ythar = ythar[ythar_mask]
        xthar = wavelength_to_pixel(ythar, params, pixels, polynomial)

        # Remove ThAr lines too close to measured lines
        unmatched_mask = np.ones_like(xthar, dtype=bool)
        for i, xt in enumerate(xthar):
            if np.any(np.abs(actual_positions - xt) < 0.2):
                unmatched_mask[i] = False
        xthar = xthar[unmatched_mask]
        ythar = ythar[unmatched_mask]

        if len(xthar) == 0:
            break

        # Append predicted ThAr lines to linetable for next iteration
        lt_fake = [[xthar[k], ythar[k], ythar[k], 1.0, 1.0, 1.0] for k in range(len(xthar))]
        linetable = np.vstack((linetable, lt_fake))
#    print(ngoods)

    # Final mask and measurements
    actual_positions = actual_positions[mask_good_disp]
    actual_errors = actual_errors[mask_good_disp]
    fwhm_pix = fwhm_pix[mask_good_disp]
    line_wls = line_wls[mask_good_disp]

    # Pixel width in Angstroms
    pix_width = np.abs([np.mean(np.diff(polynomial(np.arange(3) + i - 1, *params)))
                        for i in actual_positions])
    fwhm_angstrom = fwhm_pix * np.array(pix_width)

    # Store results in order
    order.wl = polynomial(pixels, *params)
    order.cal_pix = actual_positions
    order.cal_wl = line_wls
    order.cal_pix_fwhm = fwhm_pix
    order.cal_wl_fwhm = fwhm_angstrom

    if DEBUG_PLOTS:
        plt.title(f"Order {order.id}")
        plt.scatter(order.cal_wl, order.cal_wl / fwhm_angstrom, zorder=10)
        plt.xlabel(r"$\lambda$ / $\mathrm{\AA}$")
        plt.ylabel(r"$R = \lambda / \Delta \lambda$")
        plt.show()

def process_dispersion(args):
    """
    wrapper for solving dispersion relations using multiprocessing
    """
    j, idx_id, idx_order, o, DEBUG_PLOTS = args
    key = shared("avg_aps")[idx_id]
    linelist_o = shared("linelists")[key]
    thar_list = shared("thar_list")

    # only plot one solution
    debug_solve = DEBUG_PLOTS and (j == 3)

    try:
        solve_wavelength(linelist_o, o, DEBUG_PLOTS=debug_solve, thar_list=thar_list)
    except Exception as exc:
        # too few usable lines in this order, or a fit that will not converge.
        # leave o.wl as None: the caller drops orders without a solution rather
        # than losing the whole night over one unusable order at the edge.
        warnings.warn("no dispersion solution for order %s: %s" % (o.id, exc))
        return idx_order, o
    o.pix = np.arange(len(o.wl))
#    o.plot_frame_1d("comp_orig")

    return idx_order, o


def _nearest_signed(ap_measure, ap_shifted):
    """Signed distance from each order to its nearest reference aperture."""
    d = np.asarray(ap_measure)[:, None] - np.asarray(ap_shifted)[None, :]
    j = np.argmin(np.abs(d), axis=1)
    return d[np.arange(len(d)), j]


def _plot_idcomp_offset(ap_idcomp, ap_measure, offs, cost, offset, residual,
                        quality, spacing):
    """Show how the idcomp offset was chosen, and what the alternatives look like.

    Top: the scan. Every candidate shift of the reference apertures, and how
    well the detected orders line up with them. The correct shift sits in a
    deep, narrow well; the shallow minima roughly one order spacing away are
    the off-by-one alignments, which are exactly the ones that used to be
    picked silently by a hardcoded offset.

    Bottom: the alignment itself. At the chosen offset every order sits within
    a fraction of a pixel of a reference aperture. Shifted by one order spacing
    the orders no longer land on the apertures -- and because the spacing
    varies across the detector, the mismatch grows across the frame instead of
    being a constant, which is what makes the correct shift identifiable.
    """
    figsize = np.array([8, 6])
    fig, axs = plt.subplots(2, 1, figsize=figsize)

    # --- top: the scan -------------------------------------------------
    axs[0].plot(offs, cost, color="black", lw=1)
    axs[0].axvline(offset, color="tab:green", lw=1.5,
                   label="chosen: %+.2f px (%.2f px)" % (offset, residual))
    far = np.abs(offs - offset) > 0.5 * spacing
    if np.any(far):
        alt = offs[far][np.argmin(cost[far])]
        axs[0].axvline(alt, color="tab:orange", ls="--", lw=1.2,
                       label="best alternative: %+.0f px (%.2f px)"
                             % (alt, cost[far].min()))
    axs[0].axhline(0.3 * spacing, color="tab:red", ls=":", lw=1,
                   label="warn above %.1f px" % (0.3 * spacing))
    axs[0].set_xlabel("shift applied to reference apertures  /  pix")
    axs[0].set_ylabel("median distance\norder -> aperture  /  pix")
    axs[0].set_title("idcomp offset scan: chosen minimum is %.1fx deeper "
                     "than any other" % quality)
    axs[0].legend(fontsize=8)

    # --- bottom: this alignment vs the off-by-one ones -------------------
    # orders beyond the ends of the reference set have no aperture to match and
    # would otherwise set the scale for the whole panel
    inside = ((np.asarray(ap_measure) > np.min(ap_idcomp) + offset - spacing)
              & (np.asarray(ap_measure) < np.max(ap_idcomp) + offset + spacing))
    alt_resid = []
    for delta, color, style, name in [
            (-spacing, "tab:orange", "^", "one order low"),
            (0.0, "tab:green", "o", "chosen"),
            (+spacing, "tab:purple", "v", "one order high")]:
        resid = _nearest_signed(ap_measure, np.asarray(ap_idcomp) + offset + delta)
        if delta != 0:
            alt_resid.append(np.abs(resid[inside]))
        axs[1].scatter(ap_measure, resid, s=14, color=color, marker=style,
                       label="%s (median |d| = %.2f px)"
                             % (name, np.median(np.abs(resid[inside]))))
    axs[1].axhline(0, ls="--", color="gray", zorder=0)
    # scale to the off-by-one alignments, not to the unmatchable edge orders
    span = 1.5 * np.percentile(np.concatenate(alt_resid), 90) if alt_resid else spacing
    axs[1].set_ylim(-span, span)
    axs[1].set_xlabel("order position on detector  /  pix")
    axs[1].set_ylabel("distance to nearest\naperture  /  pix")
    axs[1].set_title("orders off the top/bottom of this panel lie beyond the "
                     "reference set", fontsize=8)
    axs[1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def solve_idcomp_offset(ap_idcomp, ap_measure, search=None, coarse=0.5, fine=0.02,
                        verbose=False, DEBUG_PLOTS=False):
    """Measure the cross-dispersion shift between the idcomp reference and this night.

    The idcomp line lists are tied to the aperture positions of the night they
    were taken on. The spectrograph shifts in the cross-dispersion direction
    between runs, so orders can only be paired with their line list after that
    shift is applied. It used to be a hardcoded constant, which quietly pairs
    orders with a *neighbouring* aperture once the drift grows comparable to the
    order spacing (~15 px) -- the reduction still completes, but every order
    carries the wrong line list and the wavelength solution is wrong.

    Here the shift is measured from the data: scan candidate offsets and keep
    the one minimising the median distance from each detected order to the
    nearest reference aperture. Using the median makes this insensitive to
    spurious order detections, and the minimum is sharp because the order
    spacing varies across the detector -- a wrong-by-one alignment cannot be
    absorbed by a rigid shift, so it leaves a much larger residual.

    Returns
    -------
    offset : float
        Shift in pixels to add to the reference aperture positions.
    residual : float
        Median distance from an order to its reference aperture, in pixels.
    quality : float
        Residual of the best rejected alternative divided by ``residual``.
        Values near 1 mean the alignment is ambiguous.

    With ``DEBUG_PLOTS`` the scan is shown together with the alignment it
    produces, next to the off-by-one alignments it rejected.
    """
    ap_idcomp = np.asarray(ap_idcomp, dtype=float)
    ap_measure = np.asarray(ap_measure, dtype=float)

    spacing = np.median(np.diff(np.sort(ap_measure)))
    if search is None:
        # a few order spacings either side covers any realistic drift
        search = 4 * spacing

    def cost(offsets):
        # median nearest-neighbour distance, robust against spurious orders
        d = np.abs(ap_measure[:, None, None]
                   - (ap_idcomp[None, :, None] + offsets[None, None, :]))
        return np.median(d.min(axis=1), axis=0)

    offs = np.arange(-search, search + coarse, coarse)
    c = cost(offs)
    best = offs[np.argmin(c)]

    # sub-pixel refinement around the coarse minimum
    fine_offs = np.arange(best - coarse, best + coarse + fine, fine)
    fc = cost(fine_offs)
    offset = float(fine_offs[np.argmin(fc)])
    residual = float(fc.min())

    # how well separated is this minimum from the best alternative alignment?
    others = c[np.abs(offs - best) > 0.5 * spacing]
    quality = float(others.min() / residual) if len(others) and residual > 0 else np.inf

    if DEBUG_PLOTS:
        _plot_idcomp_offset(ap_idcomp, ap_measure, offs, c, offset, residual,
                            quality, spacing)

    if verbose:
        print("- idcomp offset = %+.2f px (residual %.2f px, %.1fx better than "
              "next candidate)" % (offset, residual, quality))
    if residual > 0.3 * spacing:
        warnings.warn("idcomp offset residual is %.2f px for an order spacing of "
                      "%.1f px: the detected orders do not line up with the "
                      "reference apertures." % (residual, spacing))
    elif quality < 2:
        warnings.warn("idcomp offset %+.2f px is only %.1fx better than the next "
                      "candidate: the order identification may be off by one."
                      % (offset, quality))
    return offset, residual, quality


def find_dispersion(orders, biases, comps,
                    idcomp_dir, idcomp_offset="auto",
                    thar_list=None,
                    verbose=False, DEBUG_PLOTS=False):

    npix_x = biases.shape[1]
    # estimate y-pos of each order at the center of the x axis
    ap_measure = []
    for o in orders:
        ap = float(polynomial(npix_x / 2, *o.solution))
        o.pixel_y_cen = ap
        ap_measure.append(ap)

    times_sigma = 2

    # Anything in the directory that parses as an identify record is a line
    # list. The names are not a convention we control -- the 2023 lists are
    # "idiazcomp.0001", the 2026 ones from the observatory are "idtzc01" --
    # so matching on the name would have silently found nothing and left the
    # night with no wavelength solution at all.
    raw_lists = {}
    skipped = []
    for file in sorted(os.listdir(idcomp_dir)):
        path = os.path.join(idcomp_dir, file)
        if not os.path.isfile(path):
            continue
        try:
            aplo, aphi, table = parse_idcomp(path)
        except (ValueError, UnicodeDecodeError):
            aplo = aphi = None
        if aplo is None or aphi is None or len(table) == 0:
            skipped.append(file)
            continue
        raw_lists[(aplo + aphi) / 2] = table
    if not raw_lists:
        raise ValueError("no idcomp line lists found in %s" % idcomp_dir)
    if verbose:
        print("- %d idcomp line lists from %s%s"
              % (len(raw_lists), idcomp_dir,
                 " (ignored %s)" % ", ".join(skipped) if skipped else ""))

    if isinstance(idcomp_offset, str) and idcomp_offset == "auto":
        idcomp_offset, _, _ = solve_idcomp_offset(
            list(raw_lists.keys()), ap_measure, verbose=verbose,
            DEBUG_PLOTS=DEBUG_PLOTS)
    elif verbose:
        print("- idcomp offset = %+.2f px (fixed)" % idcomp_offset)

    linelists = {ap + idcomp_offset: table for ap, table in raw_lists.items()}
    avg_aps = np.array(list(linelists.keys()))

#    if DEBUG_PLOTS:
#        plt.imshow(flats)
#        for key in linelists.keys():
#            plt.scatter([len(spectrum) / 2], [key], marker="x", zorder=2, color="red")
#        plt.show()

    # find the best-matching orders
    id_order_pairs = pair_generation(avg_aps, ap_measure, thres_max=np.inf)
    id_order_pairs = [p for p in id_order_pairs if (p[0] is not None) and (p[1] is not None)]
    if verbose: print("- found %d pairs" % len(id_order_pairs))

    # extract arc spectra from image
    if verbose: print("- extracting orders")
    args = [(p[1], orders[p[1]], times_sigma) for p in id_order_pairs]
    with shared_pool({"biases": biases, "comps": comps}) as pool:
        results = list(tqdm(pool.imap(extract_order_for_calib, args), total=len(args)))
    for idx_order, o in results:
        orders[idx_order] = o

    """
    for p in id_order_pairs:
        idx_order = p[1]
        o = orders[idx_order]
        if verbose: print("- order", o.id, end="\r")
        o.extract_along_order(biases, "bias", times_sigma=times_sigma)
        o.extract_along_order(comps, "comp", times_sigma=times_sigma)
        o.apply_corrections(comparison=True)
    """

    if verbose: print("- solving dispersion relations")
    args = [(j, p[0], p[1], orders[p[1]], DEBUG_PLOTS) \
        for j, p in enumerate(id_order_pairs)]
    calib_data = {"avg_aps": avg_aps, "linelists": linelists,
                  "thar_list": thar_list}

    if DEBUG_PLOTS:
        # Sequential processing when DEBUG_PLOTS is enabled (matplotlib compatibility)
        publish_shared(calib_data)
        results = []
        for arg in tqdm(args, total=len(args)):
            result = process_dispersion(arg)
            results.append(result)
    else:
        # Parallel processing when DEBUG_PLOTS is disabled
        with shared_pool(calib_data) as pool:
            results = list(tqdm(pool.imap(process_dispersion, args), total=len(args)))

    for idx_order, o in results:
        orders[idx_order] = o

    # Each order above was fitted in isolation. The spectrograph does not work
    # that way -- m*lambda is the same for every order to a part in 10^4 -- so
    # use that to catch and repair any order whose own fit has drifted. The
    # sparsest orders have thirteen lines against four free parameters and are
    # the ones this protects.
    try:
        enforce_invariant(orders, verbose=verbose)
    except Exception as exc:
        warnings.warn("grating-relation check skipped: %s" % exc)

    """
    for j, p in enumerate(id_order_pairs):
        idx_id = p[0]
        idx_order = p[1]
        o = orders[idx_order]
        if verbose: print("- order", o.id, end="\r")
        key = avg_aps[idx_id]
        linelist_o = linelists[key]
        # only plot one solution
        if DEBUG_PLOTS and j==3:
            debug_solve = True
        else:
            debug_solve = False
        solve_wavelength(linelist_o, o, DEBUG_PLOTS=debug_solve)
        o.pix = np.arange(len(o.wl))

#        o.plot_frame_1d("comp_orig")
    """


    return orders
