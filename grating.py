"""Keep every order consistent with the grating equation."""
import numpy as np

#: order numbers an echelle can reach; the OES runs m = 40-93
M_MIN, M_MAX = 10.0, 300.0

#: how close K/lambda must land to an integer to place an order
M_TOLERANCE = 0.25

#: fewest orders the invariant can be measured from
MIN_FOR_INVARIANT = 4


def fit_grating(order_numbers, centre_wavelengths, dispersions):
    """Model `m*lambda` and `m^2*dispersion` against order number."""
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


def invariant(centre_wavelengths):
    """The echelle invariant `K = m*lambda`, from the order spacing alone."""
    lam = np.sort(np.asarray(centre_wavelengths, float))
    if lam.size < MIN_FOR_INVARIANT:
        return np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        m_local = lam[1:] / np.diff(lam)
    good = np.isfinite(m_local) & (m_local > M_MIN) & (m_local < M_MAX)
    if good.sum() < 3:
        return np.nan
    return float(np.median(m_local[good] * lam[:-1][good]))


def order_numbers(centre_wavelengths, valid=None, tol=M_TOLERANCE):
    """Physical order numbers, from the assignment that makes m*lambda flat."""
    wc = np.asarray(centre_wavelengths, float)
    out = np.full(wc.shape, np.nan)

    usable = np.isfinite(wc) & (wc > 0)
    sets_scale = usable
    if valid is not None:
        sets_scale = usable & np.asarray(valid, bool)
        if sets_scale.sum() < MIN_FOR_INVARIANT:
            sets_scale = usable

    K = invariant(wc[sets_scale])
    if not np.isfinite(K):
        return out

    with np.errstate(divide="ignore", invalid="ignore"):
        m = K / np.where(usable, wc, np.nan)
    placed = np.isfinite(m) & (np.abs(m - np.round(m)) <= tol) \
        & (np.round(m) >= 1)
    out[placed] = np.round(m[placed])
    return out


def enforce_invariant(orders, thar=None, verbose=False, tol=5.0,
                      min_orders=12, max_nonmonotonic=0.02):
    """Check every solved order against the grating relation, and refit strays.
"""
    good = [o for o in orders if getattr(o, "wl", None) is not None]
    if len(good) < min_orders:
        return 0

    wc = np.array([float(np.median(o.wl)) for o in good])
    disp = np.array([float(np.median(np.diff(o.wl))) for o in good])
    m = order_numbers(wc)

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

    nonmono = []
    for o in good:
        wl = np.asarray(getattr(o, "wl", None), float)
        if wl is None or wl.size < 3:
            continue
        dw = np.diff(wl)
        fin = np.isfinite(dw)
        if fin.sum() < 3:
            continue
        up, dn = int(np.sum(dw[fin] > 0)), int(np.sum(dw[fin] < 0))
        frac_wrong = min(up, dn) / float(up + dn)
        o.dispersion_ok = frac_wrong <= max_nonmonotonic
        if not o.dispersion_ok:
            nonmono.append((o.id, frac_wrong, float(np.nanmax(wl) - np.nanmin(wl))))
    if verbose and nonmono:
        for oid, fw, span in nonmono:
            print("- order %s dispersion is not monotonic (%.0f%% of steps run "
                  "the wrong way, span %.0f A): excluded" % (oid, 100 * fw, span))
    scale = max(model["scatter_wl"], 0.2 * float(np.nanmedian(m)))
    if not np.isfinite(scale) or scale <= 0:
        return 0

    nfixed = 0
    unplaced = []
    for o, mm, r in zip(good, m, resid):
        if not np.isfinite(mm):
            o.dispersion_ok = False
            unplaced.append(o.id)
            continue
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

    if verbose and unplaced:
        print("- order%s %s cannot be placed in the order block "
              "(centre wavelength is not one order's away from any other): "
              "excluded" % ("s" if len(unplaced) > 1 else "",
                            ", ".join(str(u) for u in unplaced)))
    if verbose and nfixed == 0 and not unplaced:
        print("- all %d orders consistent with the grating relation "
              "(m*lambda = %.0f +- %.0f)"
              % (len(good), model["mean_wl"], scale))
    return nfixed
