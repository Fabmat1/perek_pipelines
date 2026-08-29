"""Keep every order consistent with the grating equation.

Each order's dispersion relation is fitted on its own, so nothing stops one of
them drifting -- and the sparsest carry only about a dozen lines against a
four-parameter cubic, which is little constraint. But the spectrograph ties them
together: for an echelle `m * lambda` and `m^2 * dispersion` are both smooth,
low-order functions of the order number, holding to about one part in 10^4
across all 52 orders.

An order that disagrees with its neighbours by more than a few times that
scatter is not measuring the instrument, it is fitting its own noise. Its scale
and zero point are then replaced by what the other orders predict, keeping
whatever curvature its own fit found.
"""
import numpy as np



def fit_grating(order_numbers, centre_wavelengths, dispersions):
    """Model `m*lambda` and `m^2*dispersion` against order number.

    Both are nearly constant, so a quadratic in `m` absorbs the residual trend
    and extrapolates a few orders safely. Returns the two coefficient sets and
    the scatter of each, which set the tolerances used later.
    """
    m = np.asarray(order_numbers, float)
    wc = np.asarray(centre_wavelengths, float)
    disp = np.asarray(dispersions, float)

    p_wl = np.polyfit(m, m * wc, 2)
    r_wl = m * wc - np.polyval(p_wl, m)
    p_disp = np.polyfit(m, disp * m ** 2, 2)
    r_disp = disp * m ** 2 - np.polyval(p_disp, m)
    return dict(p_wl=p_wl, p_disp=p_disp,
                scatter_wl=float(np.std(r_wl)),
                scatter_disp=float(np.std(r_disp)),
                mean_wl=float(np.mean(m * wc)))


def order_numbers(centre_wavelengths, valid=None):
    """Physical order numbers, from the assignment that makes m*lambda flat.

    A single runaway order is enough to wreck this if it is allowed to vote: a
    fit that has diverged can report a central wavelength of 10^7 A, and the
    search then picks whatever `m0` makes *that* look least bad. So the scale
    is set by a robust statistic over the orders that agree, not by the mean
    over all of them.
    """
    wc = np.asarray(centre_wavelengths, float)
    idx = np.argsort(wc)[::-1]          # descending lambda == ascending m
    w = wc[idx]

    if valid is None:
        # an order whose centre is far from the run of its neighbours is not
        # measuring the instrument; keep it out of the scale determination
        med = np.median(w)
        mad = np.median(np.abs(w - med)) * 1.4826
        valid_sorted = np.isfinite(w) & (w > 0)
        if np.isfinite(mad) and mad > 0:
            valid_sorted &= np.abs(w - med) < 20 * mad
    else:
        valid_sorted = np.asarray(valid, bool)[idx]

    if valid_sorted.sum() < 5:
        valid_sorted = np.isfinite(w) & (w > 0)

    best = None
    for m0 in range(20, 200):
        m = m0 + np.arange(len(w))
        mw = (m * w)[valid_sorted]
        if not np.all(np.isfinite(mw)) or np.median(mw) <= 0:
            continue
        # median absolute deviation, so a survivor outlier cannot dominate
        s = np.median(np.abs(mw - np.median(mw))) / np.median(mw)
        if best is None or s < best[0]:
            best = (s, m)
    if best is None:
        return np.arange(len(w), dtype=float)
    _, m = best
    out = np.empty(len(w))
    out[idx] = m
    return out


def enforce_invariant(orders, thar=None, verbose=False, tol=5.0,
                      min_orders=12):
    """Check every solved order against the grating relation, and refit strays.

    Each order is fitted on its own, so nothing stops one of them drifting --
    and the sparsest carry only thirteen lines against a four-parameter cubic,
    which is little constraint. But the spectrograph ties them together:
    `m*lambda` and `m^2*dispersion` are both smooth, low-order functions of the
    order number, measured here to about one part in 10^4 across 52 orders.

    An order that disagrees with its neighbours by more than `tol` times that
    scatter is not measuring the instrument, it is fitting its own noise. Such
    an order has its wavelength scale replaced by the grating prediction,
    keeping whatever curvature its own fit found but restoring the scale and
    zero point the other fifty orders agree on.

    Returns the number of orders corrected.
    """
    good = [o for o in orders if getattr(o, "wl", None) is not None]
    if len(good) < min_orders:
        return 0

    wc = np.array([float(np.median(o.wl)) for o in good])
    disp = np.array([float(np.median(np.diff(o.wl))) for o in good])
    m = order_numbers(wc)

    # The model has to be built from the orders that agree with each other. A
    # diverged fit -- ten seed lines against a four-parameter cubic is enough
    # to allow one -- would otherwise set the very scale it is then tested
    # against, and the check would pass it.
    mw = m * wc
    med = np.median(mw)
    mad = np.median(np.abs(mw - med)) * 1.4826
    trusted = np.isfinite(mw) & (mw > 0)
    if np.isfinite(mad) and mad > 0:
        trusted &= np.abs(mw - med) < 10 * mad
    if trusted.sum() < min_orders:
        return 0

    model = fit_grating(m[trusted], wc[trusted], disp[trusted])

    resid = m * wc - np.polyval(model["p_wl"], m)
    # Floor the scale. `scatter_wl` says how well a quadratic describes the
    # orders it was fitted to, and on a good night that can be well under an
    # Angstrom-order -- which would make this check fire on differences far
    # below the accuracy of the wavelength solution itself. A real order is not
    # located better than a few tenths of an Angstrom, so do not pretend to
    # test it more tightly than that.
    scale = max(model["scatter_wl"], 0.2 * float(np.median(m)))
    if not np.isfinite(scale) or scale <= 0:
        return 0

    nfixed = 0
    for o, mm, r in zip(good, m, resid):
        if abs(r) <= tol * scale:
            continue
        wl = np.asarray(o.wl, float)
        npix = len(wl)
        x = np.arange(npix, dtype=float) - npix / 2.0
        # keep the shape this order measured, replace the scale and zero point
        own_disp = float(np.median(np.diff(wl)))
        shape = (wl - float(np.median(wl)) - own_disp * x) / own_disp
        pred_cen = np.polyval(model["p_wl"], mm) / mm
        pred_disp = np.polyval(model["p_disp"], mm) / mm ** 2
        o.wl = pred_cen + pred_disp * x + shape * pred_disp
        nfixed += 1
        if verbose:
            print("- order %s off the grating relation by %.1f sigma "
                  "(%.1f A): scale restored from the other orders"
                  % (o.id, abs(r) / scale, abs(r) / mm))

    if verbose and nfixed == 0:
        print("- all %d orders consistent with the grating relation "
              "(m*lambda = %.0f +- %.0f)"
              % (len(good), model["mean_wl"], scale))
    return nfixed
