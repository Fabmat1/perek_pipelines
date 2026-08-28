import os
import warnings
from time import time
import numpy as np
from numpy.polynomial import legendre
import matplotlib.pyplot as plt

from scipy.constants import speed_of_light
from scipy.ndimage import (minimum_filter, maximum_filter,
                           median_filter, uniform_filter1d)
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy.io import fits
from astropy.stats import sigma_clip

from estimate_noise import estimate_noise
from tools import (polyfit_reject, curve_fit_reject, pair_generation,
                   mask_section, shared_pool, shared, get_ncpu,
                   fill_nan, Gaussian, Gaussian_res)
from calibrate import find_dispersion
from identify_orders import (SpectralSlice, find_orders)
from orders import (SpectralOrder, extract_order)
from scattered import remove_background

from resample_backend import resample
from paths import DEFAULT_IDCOMP_DIR, DEFAULT_DATA_DIR

two_log_two = 2 * np.sqrt(2 * np.log(2))


def get_barycorr(frame):
    """
    comute:
     - barycentric correction to the radial velocity
     - barycentrically corrected JD (-> BJD)
    """

    with fits.open(frame) as hdul:
        header = hdul[0].header
    header = dict(header)
    hreq = ["LATITUDE", "LONGITUD", "HEIGHT",
            "RA", "DEC", "DATE-OBS", "UT"]
    radvel_corr = None
    bjd = None
    if all(i in header for i in hreq):
        # Telescope location
        lat = header['LATITUDE']
        lon = header['LONGITUD']
        height = header['HEIGHT']
        location = EarthLocation(lat=lat, lon=lon, height=height)

        # Target coordinates
        RA = header['RA']
        DEC = header['DEC']
        coord = SkyCoord(ra=RA, dec=DEC, unit=(u.hourangle, u.deg))

        # Observation time
        otime = Time(header["DATE-OBS"]+"T"+header["UT"], format='isot', scale='utc', location=location)

        # Radial velocity correction (barycentric)
        radvel_corr = coord.radial_velocity_correction(obstime=otime)
        radvel_corr = radvel_corr.to(u.km / u.s).value

        # Barycentric Julian Date
        ltt_bary = otime.light_travel_time(coord)  # Light-travel time correction
        bjd = (otime.tdb + ltt_bary).jd           # BJD in days
    else:
        print("> failed to compute barycorr")

    return radvel_corr, bjd

def coadd_frames(frames):
    frame = np.sum(frames, axis=0).astype(float) / len(frames)
    return frame


def combine_arcs(comps, verbose=False):
    """Combine the comparison exposures of a night into one arc.

    Two things a plain mean gets wrong here:

    * a cosmic ray on one exposure survives averaging as a sharp, line-shaped
      artefact, which is exactly what the line fitter is looking for;
    * the ThAr lamp is still warming up on the first exposure of a set. On
      20260826 it is 19% fainter than the rest, which then agree to within 5%,
      so including it dilutes the stack.

    Median-combine instead, and drop the leading exposure when enough others
    remain. Arcs taken hours apart are still combined: the measured drift
    across a night is ~0.1 px in both directions, far below a resolution
    element.
    """
    arrays = [np.asarray(open_or_coadd_frame(c), dtype=float) for c in comps] \
        if isinstance(comps, list) else [np.asarray(comps, dtype=float)]

    if len(arrays) > 3:
        arrays = arrays[1:]
    if len(arrays) == 1:
        return arrays[0]
    if verbose:
        print("- combining %d arc exposures (median)" % len(arrays))
    return np.median(arrays, axis=0)

def open_or_coadd_frame_old(frame):
    if isinstance(frame, np.ndarray):
        return frame
    elif isinstance(frame, list) and \
         (type(frame[0]) != str):
        frame = coadd_frames(frame)
    else:
        with fits.open(frame) as hdul:
            frame = hdul[0].data
    return frame

def open_or_coadd_frame(frame):
    if isinstance(frame, np.ndarray):
        # Already a numpy array
        return frame
    elif isinstance(frame, list):
        if all(isinstance(f, np.ndarray) for f in frame):
            # List of numpy arrays -> coadd directly
            frame = coadd_frames(frame)
        elif all(isinstance(f, str) for f in frame):
            # List of filenames -> open and coadd
            arrays = []
            for f in frame:
                with fits.open(f) as hdul:
                    arrays.append(np.asarray(hdul[0].data))
            frame = coadd_frames(arrays)
        else:
            raise ValueError("List must contain either all ndarrays or all strings.")
    elif isinstance(frame, str):
        # Single filename
        with fits.open(frame) as hdul:
            frame = hdul[0].data
    else:
        raise TypeError("Input must be a numpy array, a filename, or a list of them.")
    return frame

def wlshift(wl, vel_corr):
    # wl_shift = vel_corr/speed_of_light * wl
    # return wl+wl_shift
    return wl / (1 - (vel_corr / (speed_of_light / 1000)))


def plot_order_list(olist: list[SpectralOrder]):
    for o in olist:
        plt.plot(o.wl, o.science)
    plt.tight_layout()
    plt.show()


def polynomial(x, *p):
    return np.polyval(p, x)

def mask_intervals(wl, intervals):
    mask = np.ones_like(wl, dtype=bool)
    for lo, hi in intervals:
        mask &= ~((wl > lo) & (wl < hi))
    return mask

def _fit_continuum(wl, flx, poly_order):
    """Polynomial continuum, clipped asymmetrically so lines pull it down less."""
    keep = np.ones(len(wl), dtype=bool)
    for lo, hi in ((3.5, 5), (2.5, 4), (2.0, 3.5), (1.8, 3)):
        params, *_ = curve_fit(lambda x, *p: polynomial(x, *p),
                               wl[keep], flx[keep],
                               p0=np.ones(poly_order + 1))
        ratio = flx / polynomial(wl, *params)
        keep = ~sigma_clip(ratio, sigma_lower=lo, sigma_upper=hi,
                           masked=True).mask
    return params


def red_edge_keep(wl, flx, width=1.5):
    """Drop the reddest `width` Angstrom of an order, and any non-finite pixel.

    There the flat holds almost no lamp signal, so dividing by it spikes the
    flux. A fixed cut rather than one that follows the spike: the flux-based
    detectors tried here needed constant tuning and still missed edges.
    """
    return np.isfinite(flx) & (wl < wl.max() - width)


def normalize_single_order(args):
    """Worker function for multiprocessing. Returns the continuum too, so the
    caller can divide the errors by it and keep each pixel's S/N."""
    (i, wl, flx, poly_order, ignore_windows, smooth_width, floor_width,
     edge_width, neighbours, runaway_range,
     extrapolated_out, extrapolate_max, DEBUG_PLOTS) = args

    keep = red_edge_keep(wl, flx, width=edge_width)

    # sigma_clip passes inf through, and one is enough to drag the continuum
    finite = np.isfinite(flx)
    mask = mask_intervals(wl, ignore_windows)
    mask2 = ~sigma_clip(np.where(finite, flx, np.nan),
                        sigma_lower=8, sigma_upper=8, masked=True).mask
    mask = mask & mask2 & keep

    own_wl = wl[mask]
    own_flx = median_filter(flx[mask], size=smooth_width, mode="nearest")

    # Only orders whose own fit runs away need a neighbour to anchor them;
    # for the rest it would just add the neighbour's noise.
    trial = _fit_continuum(own_wl, own_flx, poly_order)
    span = polynomial(wl, *trial)
    finite = np.isfinite(span) & (span > 0)
    runaway = (finite.sum() > 0 and
               span[finite].max() / span[finite].min() > runaway_range)

    wl_fit, flx_fit = [own_wl], [own_flx]
    for wl_n, flx_n in (neighbours or []) if runaway else []:
        sel = ((wl_n >= wl.min()) & (wl_n <= wl.max())
               & np.isfinite(flx_n) & mask_intervals(wl_n, ignore_windows)
               & red_edge_keep(wl_n, flx_n, width=edge_width))
        if sel.sum() < 20:
            continue
        # match the neighbour's throughput to this order before using it
        both = mask & (wl >= wl_n[sel].min()) & (wl <= wl_n[sel].max())
        if both.sum() < 20:
            continue
        own = np.median(flx[both])
        scale = np.median(np.interp(wl[both], wl_n[sel], flx_n[sel])) / own \
            if own != 0 else 0.0
        if not np.isfinite(scale) or scale <= 0:
            continue
        wl_fit.append(wl_n[sel])
        flx_fit.append(median_filter(flx_n[sel] / scale, size=smooth_width,
                                     mode="nearest"))

    wl_for_interpol = np.concatenate(wl_fit)
    flx_for_interpol = np.concatenate(flx_fit)
    isort = np.argsort(wl_for_interpol)
    params = _fit_continuum(wl_for_interpol[isort], flx_for_interpol[isort],
                            poly_order)

    flx_cont = polynomial(wl, *params)

    # Outside the fitted range use the tangent: a cubic with no data on one
    # side invents curvature there (0.61 against 1.04 at 3964 A on 20260826).
    own_fitted = own_wl
    for limit, side in ((own_fitted.min(), wl < own_fitted.min()),
                        (own_fitted.max(), wl > own_fitted.max())):
        if side.sum() == 0:
            continue
        edge_value = polynomial(limit, *params)
        slope = np.polyval(np.polyder(np.asarray(params)), limit)
        # bounded, or a steep slope run far enough drives the continuum to zero
        span = np.clip(wl[side] - limit, -extrapolate_max, extrapolate_max)
        flx_cont[side] = np.maximum(edge_value + slope * span,
                                    0.5 * abs(edge_value))

    # Extrapolated continuum is not measured: still anchors the neighbours'
    # fits, but a neighbour that did measure those wavelengths supplies them.
    # Bounded, or a broad line at an order end would drop most of the order.
    if extrapolated_out:
        keep = keep & (wl >= own_fitted.min() - extrapolate_max) \
                    & (wl <= own_fitted.max() + extrapolate_max)

    # broad median filter used as a floor under the polynomial continuum. Its
    # width is a fixed fraction of the order, not of however many pixels
    # survived masking above -- otherwise it varies from order to order.
    floor_size = max(1, int(len(flx) * floor_width))
    flx_smooth = median_filter(flx, size=floor_size, mode="nearest")
    flx_cont = np.maximum(flx_cont, flx_smooth)
    normalized_flx = flx / flx_cont

    return (i, normalized_flx, flx_cont, keep,
            (wl, flx, flx_smooth, flx_cont, keep) if DEBUG_PLOTS else None)

def poly_normalization(wls, flxs,
                       poly_order=3,
                       ignore_windows=[(3831.4, 3839.4),
                                       (3883, 3893), (3933-3, 3933+3),
                                       (3963.5, 3981),
                                       (4090, 4115), (4320, 4355),
                                       (4842, 4888), (6540, 6590),
                                       (6860, 6880),
                                       (6888.1, 6890.5), (6892, 6893.6),
                                       # the O2 A-band runs to ~7660, not 7617:
                                       # it is still 0.4 deep at 7627, so the
                                       # old windows left the trough's red half
                                       # in the fit and pulled the continuum
                                       # down into it
                                       (7590, 7660)],
                       smooth_width=31,
                       floor_width=0.25,
                       edge_width=1.5,
                       use_neighbours=True,
                       runaway_range=3.0,
                       extrapolated_out=True,
                       extrapolate_max=8.0,
                       DEBUG_PLOTS=False,
                       n_processes=None,
                       errs=None,
                       show_progress=True):
    """
    Normalize single spectral orders using low-order polynomials with multiprocessing.

    An order that cannot constrain its own continuum is fitted together with
    the overlapping parts of its neighbours, so adjacent continua agree.

    floor_width : fraction of the order used for the median filter that floors
        the continuum.
    edge_width : Angstrom cut from each order's red end.
    runaway_range : bring in the neighbours only when an order's own continuum
        spans more than this factor across the order.
    extrapolated_out : keep extrapolated pixels out of the merge.
    extrapolate_max : how far the tangent runs beyond the fitted range.
    errs : per-order uncertainties, divided by the same continuum as the flux.
        Modified in place.
    """
    # the list is wavelength-sorted, so list neighbours are detector neighbours
    def near(i):
        if not use_neighbours:
            return []
        return [(np.asarray(wls[j], float), np.asarray(flxs[j], float))
                for j in (i - 1, i + 1) if 0 <= j < len(wls)]

    args_list = [(i, wl, flxs[i], poly_order, ignore_windows, smooth_width,
                  floor_width, edge_width, near(i), runaway_range,
                  extrapolated_out, extrapolate_max, DEBUG_PLOTS)
                 for i, wl in enumerate(wls)]
    # `Pool(None)` asks for one worker per core, which ignores --ncpu and
    # oversubscribes the machine the flag exists to protect
    if n_processes is None:
        n_processes = get_ncpu()
    with Pool(max(1, int(n_processes))) as pool:
        if show_progress and len(args_list) > 10:
            results = list(tqdm(pool.imap(normalize_single_order, args_list),
                               total=len(args_list)))
        else:
            results = pool.map(normalize_single_order, args_list)

    keeps = [None] * len(wls)
    for i, normalized_flx, flx_cont, keep, debug_data in results:
        flxs[i] = normalized_flx
        keeps[i] = keep
        if errs is not None and errs[i] is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                errs[i] = np.divide(errs[i], flx_cont,
                                    out=np.full_like(np.asarray(errs[i], float),
                                                     np.inf),
                                    where=flx_cont != 0)
        if DEBUG_PLOTS and debug_data:
            wl, flx, flx_smooth, flx_cont, keep = debug_data
            obs, fit = ("black", "red") if i % 2 == 0 else ("gray", "tab:orange")
            plt.plot(wl, flx, "-", color=obs, lw=1)
            plt.plot(wl, flx_cont, "-", color=fit, lw=2)
            if not keep.all():
                plt.plot(wl[~keep], flx[~keep], "-", color="tab:blue", lw=2)
            # alternate the label height so neighbours do not overprint
            plt.annotate(str(i), (np.median(wl), np.nanmedian(flx_cont)),
                         textcoords="offset points",
                         xytext=(0, 12 if i % 2 == 0 else -16),
                         ha="center", fontsize=7, color=fit)

    if DEBUG_PLOTS:
        plt.title("Normalisation (blue = rising red edge, dropped)")
        plt.xlabel("Wavelength  /  Angstrom")
        plt.ylabel("Flux")
        plt.tight_layout()
        plt.show()

    return flxs, keeps

def legendre_normalization(wls, flxs,
                           poly_order=3,
                           ignore_windows=[(3831.4, 3839.4),
                                           (3883, 3893), (3963.5, 3981),
                                           (4090, 4115), (4320, 4355),
                                           (4842, 4888), (6540, 6590),
                                           (6860, 6880),
                                           (6888.1, 6890.5), (6892, 6893.6),
                                           (7590, 7617), (7622.8, 7625)],
                           smooth_width=31,
                           DEBUG_PLOTS=False):
    """
    Normalize single spectral orders using low-order Legendre polynomials.

    Parameters
    ----------
    wls : list of 1D arrays
        Wavelength arrays (one per order).
    flxs : list of 1D arrays
        Flux arrays (one per order).
    poly_order : int
        Degree of Legendre polynomial (1–2 is usually best).
    ignore_windows : list of (low, high) tuples
        Wavelength ranges to ignore during fitting.
    smooth_width : int
        Width of median filter for pre-smoothing.
    DEBUG_PLOTS : bool
        If True, diagnostic plots are shown.
    """

    def mask_intervals(wl, intervals):
        mask = np.ones_like(wl, dtype=bool)
        for lo, hi in intervals:
            mask &= ~((wl > lo) & (wl < hi))
        return mask

    def legendre_fit(x, *coeffs):
        """Evaluate Legendre polynomial on normalized domain [-1, 1]."""
        # Normalize wavelengths to [-1, 1]
        xn = 2 * (x - x.min()) / (x.max() - x.min()) - 1
        return legendre.legval(xn, coeffs)

    for i, wl in enumerate(wls):
        flx = flxs[i]

        # Pre-smooth flux to suppress noise/cosmic rays
        flx_smooth = median_filter(flx, size=smooth_width)

        # Mask out bad windows
        mask = mask_intervals(wl, ignore_windows)
        wl_for_fit = wl[mask]
        flx_for_fit = flx_smooth[mask]

        # Initial broad sigma clip
        mask2 = ~sigma_clip(flx_for_fit,
                            sigma_lower=8,
                            sigma_upper=8,
                            masked=True).mask

        # Iterative asymmetric clipping
        sigmas_lo = [3, 2.5, 2.0, 1.8]
        sigmas_hi = [5, 4, 4, 3]

        for k in range(len(sigmas_lo)):
            coeffs, _ = curve_fit(lambda x, *c: legendre_fit(x, *c),
                                  wl_for_fit[mask2],
                                  flx_for_fit[mask2],
                                  p0=np.zeros(poly_order + 1))
            flx_cont = legendre_fit(wl_for_fit, *coeffs)
            ratio = flx_for_fit / flx_cont
            mask2 = ~sigma_clip(ratio,
                                sigma_lower=sigmas_lo[k],
                                sigma_upper=sigmas_hi[k],
                                masked=True).mask

        # Final continuum
        flx_cont = legendre_fit(wl, *coeffs)

        # Enforce continuum ≥ smoothed flux
        flx_cont = np.maximum(flx_cont, flx_smooth)

        # Normalize
        flxs[i] = flx / flx_cont

        if DEBUG_PLOTS:
            plt.figure()
            plt.plot(wl, flx, "k-", alpha=0.6, label="original")
            plt.plot(wl, flx_smooth, "gray", alpha=0.6, label="smoothed")
            plt.plot(wl, flx_cont, "r-", lw=2, label="Legendre fit")
            plt.legend()
            plt.title(f"Order {i}")

    if DEBUG_PLOTS:
        plt.show()

    return flxs

def process_order(order_data):
    i, wave_order, flux_order, flux_err_order = order_data
    wave_new = shared("wave_new")

    # Sort and mask valid entries
    isort = np.argsort(wave_order)
    wave_order = wave_order[isort]
    flux_order = flux_order[isort]
    mask = np.isfinite(wave_order) & np.isfinite(flux_order)
    wave_order = wave_order[mask]
    flux_order = flux_order[mask]

    if flux_err_order is not None:
        flux_err_order = flux_err_order[isort][mask]

    if len(wave_order) == 0:
        # nothing usable in this order; contribute no pixels to the merge
        empty = np.zeros(0)
        return empty, empty, empty, np.zeros(0, dtype=int), (empty, empty)

    # Find indices in wave_new that fall within this order
    wmin_order = wave_order[0]
    wmax_order = wave_order[-1]
    widx_new = np.where((wave_new >= wmin_order) & (wave_new <= wmax_order))[0]
    wave_order_new = wave_new[widx_new]

    if len(widx_new) == 0:
        # this order lies entirely outside the output grid
        empty = np.zeros(0)
        return empty, empty, empty, np.zeros(0, dtype=int), (wave_order, flux_order)
    # kept for the caller to plot: pyplot in a worker draws into a figure
    # that is discarded when the process exits
    pre_resample = (wave_order, flux_order)

    # Resample flux and flux_err
    flux_order = resample(wave_order_new, wave_order, flux_order, fill=0, verbose=False)
    if flux_err_order is not None:
        flux_err_order = resample(wave_order_new, wave_order, flux_err_order, fill=0, verbose=False)
    else:
        flux_err_order = estimate_noise(wave_order_new, flux_order)

    mask_err = (~np.isfinite(flux_err_order)) | (flux_err_order <= 0)
    flux_err_order[mask_err] = np.inf

    return wave_order_new, flux_order, flux_err_order, widx_new, pre_resample


def resample_orders_parallel(wave_new, wave, flux, flux_err=None, plot=False, ncpu=4):
    norder = len(wave)

    # Prepare arguments for each process
    args_list = [
        (i, wave[i], flux[i], None if flux_err is None else flux_err[i])
        for i in range(norder)
    ]

    # Run in parallel; the common wavelength grid is shared, not re-sent per order
    with shared_pool({"wave_new": wave_new}, processes=ncpu) as pool:
        results = pool.map(process_order, args_list)

    # Unpack results
    wave_res, flux_res, err_res, widx_res, pre_res = zip(*results)

    if plot:
        colors_i = ["tab:orange", "tab:pink"]
        for i, (w, f) in enumerate(pre_res):
            plt.plot(w, f, color=colors_i[i % 2])

    return wave_res, flux_res, err_res, widx_res

def resample_orders(wave_new, wave, flux, flux_err=None,
                    plot=False):

    """
    wave_res = []
    flux_res = []
    err_res = []
    widx_res = []
    norder = len(wave)
    colors_i = ["tab:orange", "tab:pink"]
    for i in range(norder):
        wave_order = wave[i]
        flux_order = flux[i]
        isort = np.argsort(wave_order)
        wave_order = wave_order[isort]
        flux_order = flux_order[isort]
        mask = np.isfinite(flux_order) & np.isfinite(wave_order)
        wave_order = wave_order[mask]
        flux_order = flux_order[mask]
        if not (flux_err is None):
            flux_err_order = flux_err[i]
            flux_err_order = flux_err_order[isort]
            flux_err_order = flux_err_order[mask]
        wmin_order = wave_order[0]
        wmax_order = wave_order[-1]
        widx_new = (wave_new >= wmin_order) & (wave_new <= wmax_order)
        widx_new = np.where(widx_new)[0]
        wave_order_new = wave_new[widx_new]

        if plot:
            plt.plot(wave_order, flux_order, color=colors_i[i%2])

        flux_order = resample(wave_order_new, wave_order, flux_order,
                              fill=0, verbose=False)
        if not (flux_err is None):
            flux_err_order = resample(wave_order_new, wave_order, flux_err_order,
                                      fill=0, verbose=False)
        else:
            flux_err_order = estimate_noise(wave_order_new, flux_order)

#        wave_res.append(wave_order_new)
        mask_err = (~np.isfinite(flux_err_order)) | \
                   (flux_err_order <= 0)
        flux_err_order[mask_err] = np.inf

        flux_res.append(flux_order)
        err_res.append(flux_err_order)
        widx_res.append(widx_new)
    """

    ncpu = get_ncpu()
    wave_res, flux_res, err_res, widx_res = resample_orders_parallel(wave_new, wave, flux,
                                                                     flux_err=flux_err,
                                                                     plot=plot, ncpu=ncpu)

    # --> inverse-variance merging. np.bincount sums per output pixel and does
    # not care in which order the contributions arrive, so the flattened
    # per-order arrays can be accumulated directly.
    nwave = len(wave_new)
    widx_flat = np.concatenate(widx_res)
    flux_flat = np.concatenate(flux_res)
    err_flat = np.concatenate(err_res)

    weight = 1 / np.square(err_flat)
    wsum = np.bincount(widx_flat, weights=weight, minlength=nwave)
    weighted_flux_sum = np.bincount(widx_flat, weights=flux_flat * weight,
                                    minlength=nwave)

    # wavelengths no order contributes to (gaps between orders, and the ends of
    # the grid) have zero total weight: mark them instead of dividing by zero
    covered = wsum > 0
    flux_merge = np.full(nwave, np.nan)
    err_merge = np.full(nwave, np.inf)
    np.divide(weighted_flux_sum, wsum, out=flux_merge, where=covered)
    variance = np.zeros(nwave)
    np.divide(1.0, wsum, out=variance, where=covered)
    np.sqrt(variance, out=err_merge, where=covered)

    if plot:
        plt.plot(wave_new, flux_merge)
#        plt.plot(wave_new, err_merge, c="grey")
        err_est = estimate_noise(wave_new, flux_merge)
        plt.plot(wave_new, err_est, c="tab:gray")
        plt.title("Flux normalisation by order")
        plt.xlabel("Wavelength / Angstrom")
        plt.ylabel("Flux")
        plt.tight_layout()
        plt.show()

    return flux_merge, err_merge


def generate_wave_grid(wmin, wmax, resolution,
                       sampling=2.7):
    if not (0 < wmin < wmax):
        raise ValueError("need 0 < wmin < wmax, got wmin=%r wmax=%r" % (wmin, wmax))
    temp = (2 * sampling * resolution + 1) / (2 * sampling * resolution - 1)
    nwave = np.ceil(np.log(wmax / wmin) / np.log(temp))
    if not np.isfinite(nwave):
        raise ValueError("could not determine grid length for "
                         "wmin=%r wmax=%r resolution=%r" % (wmin, wmax, resolution))
    t2 = np.arange(nwave)
    return temp ** t2 * wmin


def align_normalization(wave, flux, DEBUG_PLOTS=False):
    """
    mulitiply fluxes for each order so that overlapping regions align
    """

    norder = len(flux)
    wmin = [np.min(i) for i in wave]
    wmax = [np.max(i) for i in wave]
    mleft = []
    mright = []
    wleft = []
    wright = []
    for i in range(norder):
        f = flux[i]
        w = wave[i]
        isort = np.argsort(w)
        w = w[isort]
        f = f[isort]
        if i==0:
            mask_right = w > wmin[i+1]
            if np.sum(mask_right) == 0:
                mask_right = np.zeros(len(w)).astype(bool)
                mask_right[-30:] = True
            mr = np.nanmedian(f[mask_right])
            wr = np.nanmedian(w[mask_right])
            ml = np.nan
            wl = np.nan
        elif i == norder-1:
            mask_left = w < wmax[i-1]
            if np.sum(mask_left) == 0:
                mask_left = np.zeros(len(w)).astype(bool)
                mask_left[:30] = True
            ml = np.nanmedian(f[mask_left])
            wl = np.nanmedian(w[mask_left])
            mr = np.nan
            wr = np.nan
        else:
            mask_right = w > wmin[i+1]
            mask_left = w < wmax[i-1]
            if np.sum(mask_left) == 0:
                mask_left = np.zeros(len(w)).astype(bool)
                mask_left[:30] = True
            if np.sum(mask_right) == 0:
                mask_right = np.zeros(len(w)).astype(bool)
                mask_right[-30:] = True
            ml = np.nanmedian(f[mask_left])
            wl = np.nanmedian(w[mask_left])
            mr = np.nanmedian(f[mask_right])
            wr = np.nanmedian(w[mask_right])
        mleft.append(ml)
        mright.append(mr)
        wleft.append(wl)
        wright.append(wr)

    mleft = np.array(mleft)
    mright = np.array(mright)
    factors = mright[:-1] / mleft[1:]
    factors = np.insert(factors, 0, 1)
    factors = np.cumprod(factors)

    if DEBUG_PLOTS:
        for i in range(norder):
            plt.plot(wave[i], flux[i])
            plt.plot(wave[i], flux[i]*factors[i], color="black")
        plt.scatter(wleft, mleft, c="blue", zorder=1000)
        plt.scatter(wright, mright, c="red", zorder=1000)
        plt.show()

    flux_renorm = [flux[i]*factors[i] for i in range(norder)]

    return flux_renorm


def merge_orders(olist: list[SpectralOrder], normalize=True, margin=2, max_wl=8900,
                 resolution=30000, DEBUG_PLOTS=False, verbose=True):
    keep = [o for o in olist if o.wl.min() < max_wl]
    wave = [o.wl[margin:-margin] for o in keep]
    flux = [o.science[margin:-margin] for o in keep]
    # Photon errors from before normalisation: a real measurement and a blue
    # edge divided by an almost-empty flat both come out near unity after it.
    errs = [None if o.science_err is None else o.science_err[margin:-margin]
            for o in keep]

    # sort orders by median wave
    wmed = [np.nanmedian(i) for i in wave]
    isort = np.argsort(wmed)
    wave = [np.asarray(wave[i], dtype=float) for i in isort]
    flux = [np.asarray(flux[i], dtype=float) for i in isort]
    errs = [None if errs[i] is None else np.asarray(errs[i], dtype=float)
            for i in isort]
    if all(e is None for e in errs):
        errs = None

    if DEBUG_PLOTS:
        for w, f in zip(wave, flux):
            plt.plot(w, f)
        plt.title("Extracted, calibrated, flat-fielded orders")
        plt.xlabel("Wavelength  /  Angstrom")
        plt.ylabel("Flux")
        plt.tight_layout()
        plt.show()

    if verbose: print("- normalising orders")
    keeps = None
    if normalize:
        flux, keeps = poly_normalization(wave, flux, errs=errs,
                                         DEBUG_PLOTS=DEBUG_PLOTS)
#        flux = legendre_normalization(wave, flux, DEBUG_PLOTS=DEBUG_PLOTS)
    else:
        flux_before = [f.copy() for f in flux]
        flux = align_normalization(wave, flux, DEBUG_PLOTS=DEBUG_PLOTS)
        if errs is not None:
            # align_normalization scales each order by a constant; errors must
            # follow or the weights no longer match
            for i, (before, after) in enumerate(zip(flux_before, flux)):
                if errs[i] is None:
                    continue
                nz = before != 0
                scale = np.median(after[nz] / before[nz]) if nz.any() else 1.0
                errs[i] = errs[i] * abs(scale)

    # never calibrated, so they must not reach the merge
    if keeps is not None and any(k is not None and not k.all() for k in keeps):
        ndrop = sum(int((~k).sum()) for k in keeps if k is not None)
        wave = [w[k] for w, k in zip(wave, keeps)]
        flux = [f[k] for f, k in zip(flux, keeps)]
        if errs is not None:
            errs = [None if e is None else e[k] for e, k in zip(errs, keeps)]
        if verbose:
            print("- dropped %d pixels on rising red order edges" % ndrop)

    # only to get min and max wavelength
    wave_flat = np.concatenate(wave)
    flux_flat = np.concatenate(flux)
    mask = np.isfinite(flux_flat)
    wave_flat = wave_flat[mask]
    flux_flat = flux_flat[mask]
    isort = np.argsort(wave_flat)
    wave_flat = wave_flat[isort]
    flux_flat = flux_flat[isort]
    wmin = wave_flat[0]
    wmax = wave_flat[-1]

    if verbose: print("- merging orders")
    common_wl = generate_wave_grid(wmin, wmax, resolution=resolution)
    plot_resample = False
    # Measured errors, not `process_order`'s local scatter: that cannot tell an
    # inflated blue edge from a real measurement.
    common_flx, common_err = resample_orders(common_wl, wave, flux,
                                             flux_err=errs,
                                             plot=plot_resample)

    if DEBUG_PLOTS:
        plt.plot(common_wl, common_flx)
        plt.title("Merged, normalised")
        plt.xlabel("Wavelength  /  Angstrom")
        plt.ylabel("Flux")
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
        plt.title("Resolving power per order")
        plt.xlabel(r"Wavelength  /  $\mathrm{\AA}$")
        plt.ylabel(r"$\lambda$  /  $\Delta \lambda$")
        plt.tight_layout()
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


def merge_resolution(wave_merged, orders, dres, npix=45, DEBUG_PLOTS=False):

    DEBUG_PLOTS_res = DEBUG_PLOTS
    # not necessary; already plotted
    DEBUG_PLOTS_res = False

    res_poly = []
    res_poly_wl = []
    for i, o in enumerate(orders):
        coeff = dres["coeffs"][i]
        xeval = np.linspace(np.min(o.wl), np.max(o.wl), 200)
        poly1d_fn = np.poly1d(coeff)
        yeval = poly1d_fn(xeval)
        res_poly.extend(yeval)
        res_poly_wl.extend(xeval)
        if DEBUG_PLOTS_res:
            plt.plot(xeval, yeval, lw=1.5)
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
    res_merged = gaussian_filter1d(res_merged, npix,
                                   mode="constant", cval=res_med)

    if DEBUG_PLOTS_res:
        plt.plot(wave_merged, res_merged, ls="--", lw=2, color="black")
        plt.title("Resolving power")
        plt.tight_layout()
        plt.show()

    return res_merged


def _clean_scattered(spectrum, flats, comps, biases, orders, verbose=False):
    """Take the grating's scattered-light halo off the three lit frames. The
    bias comes off before measuring and back on after, because
    `apply_corrections` subtracts it again later."""
    biases = np.asarray(biases, dtype=float)
    out = []
    for frame, label in ((spectrum, "science"), (flats, "flat"),
                         (comps, "arc")):
        clean = remove_background(np.asarray(frame, dtype=float) - biases,
                                  orders, verbose=verbose, label=label)
        out.append(clean + biases)
    return out


def extract_spectrum(spectrum, flats, comps, biases, idcomp_offset="auto",
                     frame_for_slice=None,
                     normalize=True, idcomp_dir=DEFAULT_IDCOMP_DIR,
                     sampling=200, min_order_samples=6,
                     apply_barycorr=True,
                     verbose=False,
                     orders=None,
                     thar_list=None,
                     remove_scattered=True,
                     DEBUG_PLOTS=False, **kwargs):

    radvel, bjd = get_barycorr(spectrum)

    spectrum = open_or_coadd_frame(spectrum)
    flats = open_or_coadd_frame(flats)
    # the arc gets its own combine: cosmics and the lamp warm-up matter here
    # in a way they do not for the flat or bias
    comps = combine_arcs(comps, verbose=verbose)
    biases = open_or_coadd_frame(biases)

    """
    # remove biases first
    spectrum = spectrum - biases
    flats = flats - biases
    comps = comps - biases
    biases -= biases
    """

    times_sigma = 2
    if orders is None:
        if frame_for_slice is None:
            frame_for_slice = flats
        else:
            # median-combine several trace frames rather than averaging them:
            # a cosmic ray on one frame is a bright, order-shaped artefact that
            # a mean carries straight into the trace
            if isinstance(frame_for_slice, list) and len(frame_for_slice) > 2:
                stack = [np.asarray(open_or_coadd_frame(f), dtype=float)
                         for f in frame_for_slice]
                frame_for_slice = np.median(stack, axis=0)
            else:
                frame_for_slice = open_or_coadd_frame(frame_for_slice)
            # cast first: raw frames are uint16, and uint16 + uint16 wraps
            # around in numpy instead of promoting, which silently corrupts
            # exactly the bright order cores we are trying to trace
            frame_for_slice = (frame_for_slice.astype(float)
                               + np.asarray(flats, dtype=float)) / 2
            """
            plt.imshow(frame_for_slice, norm="log")
            plt.gca().set_aspect('auto')
            plt.show()
            """

        # find orders in 2d image
        orders = find_orders(frame_for_slice, sampling=sampling,
                             min_order_samples=min_order_samples,
                             DEBUG_PLOTS=DEBUG_PLOTS, verbose=verbose)

#        for o in orders:
#            o.extract_along_order(spectrum, "science", times_sigma=times_sigma)

        # Must happen here: the halo is measured between the orders, and
        # extraction collapses those pixels away. Doing it before
        # find_dispersion keeps the arc consistent with what it calibrates.
        if remove_scattered:
            spectrum, flats, comps = _clean_scattered(
                spectrum, flats, comps, biases, orders, verbose=verbose)

        # extract calibration and solve dispersion relations for each identified order
        orders = find_dispersion(orders, biases, comps, idcomp_dir,
                                 idcomp_offset=idcomp_offset,
                                 thar_list=thar_list,
                                 verbose=verbose,
                                 DEBUG_PLOTS=DEBUG_PLOTS)

        # only keep orders that have a wavelength solution
        orders = [o for o in orders if o.wl is not None]
    elif remove_scattered:
        # geometry reused from an earlier frame, but this frame's halo is its own
        spectrum, flats, comps = _clean_scattered(
            spectrum, flats, comps, biases, orders, verbose=verbose)

    if verbose: print("- extracting orders")
    args = [(o, times_sigma) for o in orders]
    frames = {"spectrum": spectrum, "flats": flats,
              "biases": biases, "comps": comps}
    with shared_pool(frames) as pool:
        results = list(tqdm(pool.imap(extract_order, args), total=len(args)))
        orders = results

    if DEBUG_PLOTS:
        plt.title("Science, flat, arc, bias in each order")
        colors_sc = ["tab:red", "tab:blue"]
        colors_flat = ["tab:orange", "tab:pink"]
        colors_arc = ["tab:purple", "tab:green"]
        colors_bias = ["tab:gray", "black"]
        for k, o in enumerate(orders):
            plt.plot(o.wl, o.science, colors_sc[k%2])
            plt.plot(o.wl, o.flat, colors_flat[k%2])
            plt.plot(o.wl, o.comparison, colors_arc[k%2])
            plt.plot(o.wl, o.bias, colors_bias[k%2])
#            print(o.wl)
        plt.tight_layout()
        plt.show()

    if verbose: print("- done         ")

    if DEBUG_PLOTS:
        for o in orders:
            plt.scatter(o.cal_pix, np.log10(o.cal_wl), zorder=10, lw=1)
            plt.plot(o.pix, np.log10(o.wl), zorder=11, lw=1)
        plt.title("Dispersion relations")
        plt.xlabel("Extracted X pixel")
        plt.ylabel(r"log" "$\\lambda / \\mathrm{\\AA}$")
        plt.tight_layout()
        plt.show()

    # estimate spectral resolving power
    dres = estimate_resolution(orders, verbose=verbose, DEBUG_PLOTS=DEBUG_PLOTS)

    wave_merged, flux_merged = merge_orders(orders,
                                            normalize=normalize,
                                            resolution=dres["R_med"],
                                            DEBUG_PLOTS=DEBUG_PLOTS,
                                            verbose=verbose)

    if apply_barycorr and (radvel is not None):
        wave_merged = wlshift(wave_merged, radvel)

    # construct resolving power column
    res_merged = merge_resolution(wave_merged, orders, dres, DEBUG_PLOTS=DEBUG_PLOTS)

    # this may remove all pixels ...
#    wave_merged, flux_merged = rmcosmics(wave_merged, flux_merged)

    flux_median = np.nanmedian(flux_merged)
    mask = np.isfinite(flux_merged) & (flux_merged > -flux_median)
    mask[[0, -1]] = False
    wave_merged = wave_merged[mask]
    flux_merged = flux_merged[mask]
    res_merged = res_merged[mask]

    mask = mask_section(flux_merged, tlo=0, thi=0.005, return_mask=True)
    wave_merged = wave_merged[mask]
    flux_merged = flux_merged[mask]
    res_merged = res_merged[mask]

    noise = estimate_noise(wave_merged, flux_merged)

    dout = {"wave": wave_merged,
            "flux": flux_merged,
            "error": noise,
            "res": res_merged,
            "orders": orders,
            "bjd": bjd}

    return dout


if __name__ == "__main__":
    # spectrum, flats, comps, biases
    bp = DEFAULT_DATA_DIR + os.sep
    idcomp_dir = DEFAULT_IDCOMP_DIR
    verbose = True
    # e202409010033 is the science target (OBJECT = 'BD+26 2766'),
    # e202409010007 is the bias (OBJECT = 'zero')
    spec = extract_spectrum(spectrum=bp+"e202409010033.fit",
                            flats=bp+"e202409010019.fit",
                            comps=bp+"e202409010029.fit",
                            biases=bp+"e202409010007.fit",
                            frame_for_slice=bp+"e202409020033.fit",
                            idcomp_dir=idcomp_dir,
                            verbose=verbose)

    plt.plot(spec["wave"], spec["flux"])
    plt.show()

