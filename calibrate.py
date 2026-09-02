
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
from scipy.signal import find_peaks


two_log_two = 2 * np.sqrt(2 * np.log(2))


def load_thar_list(*paths):
    """Read one or more ThAr line lists into the DataFrame the fit expects."""
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
    """Read one IRAF ``identify`` database file."""
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
    """Fit the arc lines of one order."""
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
    """True if `wl` runs one way across the whole detector."""
    dw = np.diff(np.asarray(wl, float))
    fin = np.isfinite(dw)
    if fin.sum() < 2:
        return True
    up = int(np.sum(dw[fin] > 0))
    dn = int(np.sum(dw[fin] < 0))
    if up + dn == 0:
        return False
    return min(up, dn) / float(up + dn) <= max_wrong


def fit_dispersion(x, y, yerr, thar_list=None, npix_detector=2048,
                   DEBUG_PLOTS=False):

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

    npix = max(int(np.nanmax(x)) + 1, npix_detector) \
        if np.isfinite(np.nanmax(x)) else npix_detector
    grid = np.arange(max(npix, 2), dtype=float)
    if not _monotonic(polynomial(grid, *params)):
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
                     seed_shift=0.0,
                     DEBUG_PLOTS=False):
    """Fit ThAr lines in a spectral order to solve the dispersion relation."""

    linetable = np.array(linetable, dtype=float, copy=True)
    if seed_shift:
        linetable[:, 0] = linetable[:, 0] + seed_shift

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
                              npix_detector=len(pixels),
                              DEBUG_PLOTS=DEBUG_PLOTS)
        params = disp["params"]
        mask_good_disp = disp["mask_good"]
        ngood = np.sum(mask_good_disp)
        ngoods.append(ngood)

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

def solve_dispersion_shift(line_px, comparison, search=30.0, coarse=0.25,
                           fine=0.02, min_lines=6):
    """Measure the shift along dispersion between the seed positions and this
    arc."""
    comparison = np.asarray(comparison, dtype=float)
    line_px = np.asarray(line_px, dtype=float)
    line_px = line_px[np.isfinite(line_px)]
    if len(line_px) < min_lines or len(comparison) < 50:
        return 0.0, np.nan, np.nan

    finite = comparison[np.isfinite(comparison)]
    if len(finite) < 50:
        return 0.0, np.nan, np.nan
    height = np.percentile(finite, 80)
    peaks, _ = find_peaks(np.nan_to_num(comparison, nan=-np.inf),
                          height=height, distance=3)
    if len(peaks) < min_lines:
        return 0.0, np.nan, np.nan
    peaks = peaks.astype(float) + 1.0        # fit_comparison counts from 1

    def cost(shifts):
        d = np.abs((line_px[:, None, None] + shifts[None, None, :])
                   - peaks[None, :, None])
        return np.median(d.min(axis=1), axis=0)

    offs = np.arange(-search, search + coarse, coarse)
    c = cost(offs)
    best = offs[int(np.argmin(c))]

    fine_offs = np.arange(best - coarse, best + coarse + fine, fine)
    fc = cost(fine_offs)
    shift = float(fine_offs[int(np.argmin(fc))])
    residual = float(fc.min())

    # how much better is this than the best genuinely different alignment?
    spacing = np.median(np.diff(np.sort(peaks))) if len(peaks) > 2 else 10.0
    others = c[np.abs(offs - best) > 0.5 * spacing]
    quality = float(others.min() / residual) if len(others) and residual > 0 \
        else np.inf

    return shift, residual, quality


def solve_dispersion_shifts(id_order_pairs, orders, linelists, avg_aps,
                            search=30.0, max_scatter=6.0, verbose=False):
    """Per-order dispersion shift, with the orders keeping each other honest.
"""
    raw = {}
    ap = {}
    for idx_id, idx_order in id_order_pairs:
        o = orders[idx_order]
        comp = getattr(o, "comparison", None)
        if comp is None:
            continue
        table = linelists[avg_aps[idx_id]]
        shift, residual, quality = solve_dispersion_shift(
            np.asarray(table, dtype=float)[:, 0], comp, search=search)
        if np.isfinite(residual) and quality > 1.5:
            raw[idx_order] = shift
            ap[idx_order] = getattr(o, "pixel_y_cen", float(idx_order))

    if not raw:
        if verbose:
            print("- dispersion shift: not measurable, seeds used as they are")
        return {}

    idxs = np.array(sorted(raw))
    vals = np.array([raw[i] for i in idxs])
    pos = np.array([ap[i] for i in idxs])

    med = np.median(vals)
    scatter = 1.4826 * np.median(np.abs(vals - med))
    keep = np.abs(vals - med) <= max(3 * scatter, max_scatter)
    if np.sum(keep) >= 4:
        coef = np.polyfit(pos[keep], vals[keep], 1)
        smooth = np.polyval(coef, pos)
        # only override the orders that disagree with the trend
        bad = np.abs(vals - smooth) > max(3 * scatter, max_scatter)
        vals = np.where(bad, smooth, vals)
    else:
        bad = ~keep
        vals = np.where(bad, med, vals)

    if verbose:
        print("- dispersion shift = %+.1f px (range %+.1f..%+.1f over %d "
              "orders, %d replaced by the trend)"
              % (np.median(vals), vals.min(), vals.max(), len(vals),
                 int(np.sum(bad))))
        if abs(np.median(vals)) > 4:
            print("  the idcomp seeds are further from the arc than "
                  "fit_comparison can reach on its own; removing the shift "
                  "first is what keeps this solution tied to the data")

    return {int(i): float(v) for i, v in zip(idxs, vals)}


def process_dispersion(args):
    """wrapper for solving dispersion relations using multiprocessing"""
    j, idx_id, idx_order, o, DEBUG_PLOTS = args
    key = shared("avg_aps")[idx_id]
    linelist_o = shared("linelists")[key]
    thar_list = shared("thar_list")
    seed_shift = shared("seed_shifts", {}).get(idx_order, 0.0)

    # only plot one solution
    debug_solve = DEBUG_PLOTS and (j == 3)

    try:
        solve_wavelength(linelist_o, o, DEBUG_PLOTS=debug_solve,
                         thar_list=thar_list, seed_shift=seed_shift)
    except Exception as exc:
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
    """Show how the idcomp offset was chosen, and what the alternatives look
    like."""
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
    """Measure the cross-dispersion shift between the idcomp reference and this
    night."""
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

    seed_shifts = solve_dispersion_shifts(id_order_pairs, orders, linelists,
                                          avg_aps, verbose=verbose)

    if verbose: print("- solving dispersion relations")
    args = [(j, p[0], p[1], orders[p[1]], DEBUG_PLOTS) \
        for j, p in enumerate(id_order_pairs)]
    calib_data = {"avg_aps": avg_aps, "linelists": linelists,
                  "thar_list": thar_list, "seed_shifts": seed_shifts}

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
