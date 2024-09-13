import os
import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from scipy.constants import speed_of_light
from matplotlib import pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import (minimum_filter, maximum_filter,
                           median_filter, uniform_filter1d)
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from astropy.stats import sigma_clip

from estimate_noise import estimate_noise
from tools import (polyfit_reject, curve_fit_reject, pair_generation,
                   mask_section,
                   fill_nan, Gaussian, Gaussian_res, polynomial)
from calibrate import find_dispersion
from identify_orders import (SpectralSlice, find_orders)
from orders import SpectralOrder

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

def get_barycorr(frame):
    with fits.open(frame) as hdul:
        header = hdul[0].header
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
    return radvel_corr

def coadd_frames(frames):
    frame = np.sum(frames, axis=0).astype(float) / len(frames)
    return frame

def open_or_coadd_frame(frame):
    if isinstance(frame, np.ndarray):
        return frame
    elif isinstance(frame, list):
        frame = coadd_frames(frame)
    else:
        with fits.open(frame) as hdul:
            frame = hdul[0].data
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


def generate_wl_grid(wl, resolution=45000, sampling=2.6):
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
                       ignore_windows=[(3831.4, 3839.4),
                                       (3883, 3893), (3963.5, 3981),
                                       (4090, 4115), (4320, 4355),
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


def merge_resolution(wave_merged, orders, dres, npix=45, DEBUG_PLOTS=False):
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
    res_merged = gaussian_filter1d(res_merged, npix,
                                   mode="constant", cval=res_med)

    if DEBUG_PLOTS:
        plt.plot(wave_merged, res_merged, ls="--", lw=2, color="black")
        plt.show()
    return res_merged

def calibrate_orders(flat, comp, bias,
                     idcomp_dir="idcomp",
                     idcomp_offset=-15,
                     sampling=200, min_order_samples=6,
                     frame_for_slice=None,
                     verbose=False, DEBUG_PLOTS=False):

    if frame_for_slice is None:
        frame_for_slice = flat
    else:
        frame_for_slice = open_or_coadd_frame(frame_for_slice)
        frame_for_slice = (frame_for_slice + flat) / 2

    # find orders in 2d image
    orders = find_orders(frame_for_slice, sampling=sampling,
                         min_order_samples=min_order_samples,
                         DEBUG_PLOTS=DEBUG_PLOTS, verbose=verbose)

    # extract calibration and solve dispersion relations for each identified order
    orders = find_dispersion(orders, bias, comp, idcomp_dir,
                             idcomp_offset=idcomp_offset,
                             verbose=verbose,
                             DEBUG_PLOTS=DEBUG_PLOTS)

    orders = [o for o in orders if o.wl is not None]

    return orders

def extract_spectrum(spectrum, flats, comps, biases, idcomp_offset=-15,
                     frame_for_slice=None,
                     normalize=True, idcomp_dir="idcomp",
                     sampling=200, min_order_samples=6,
                     apply_barycorr=True,
                     verbose=False,
                     orders=None,
                     DEBUG_PLOTS=False, **kwargs):

    radvel = get_barycorr(spectrum)

    spectrum = open_or_coadd_frame(spectrum)
    flats = open_or_coadd_frame(flats)
    comps = open_or_coadd_frame(comps)
    biases = open_or_coadd_frame(biases)

    if orders is None:
        orders = calibrate_orders(flats, comps, biases,
                                  idcomp_dir=idcomp_dir,
                                  idcomp_offset=idcomp_offset,
                                  sampling=sampling,
                                  min_order_samples=min_order_samples,
                                  frame_for_slice=frame_for_slice,
                                  verbose=verbose, DEBUG_PLOTS=DEBUG_PLOTS)

#    print("NORDERS", len(orders))

    times_sigma = 2

    if verbose: print("- extracting orders")
    # Extract different spectra
    for o in orders:
        if verbose: print("- order", o.id, end="\r")
        o.extract_along_order(spectrum, "science", times_sigma=times_sigma)
        o.extract_along_order(flats, "flat", times_sigma=times_sigma)
        o.extract_along_order(biases, "bias", times_sigma=times_sigma)
        o.extract_along_order(comps, "comp", times_sigma=times_sigma)

        o.apply_corrections(comparison=True)

        # o.plot_frame_1d("science")
        # o.plot_frame_1d("flat")
        # o.plot_frame_1d("bias")
        # o.plot_frame_1d("comp")
        # o.plot_frame_1d("comp_orig")

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
            "orders": orders}

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
                            frame_for_slice=bp+"e202409020033.fit",
                            idcomp_dir=idcomp_dir,
                            verbose=verbose)

    plt.plot(spec["wave"], spec["flux"])
    plt.show()

