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

#: Physical order numbers reachable by an echelle: outside this range a
#: neighbour-spacing estimate is measuring something other than two adjacent
#: orders. The OES runs m = 40-93 on the two bundled reference lists.
M_MIN, M_MAX = 10.0, 300.0

#: `K/lambda` has to land this close to an integer for the order to be placed.
#: On the bundled reference lists the worst of the 103 orders misses by 0.02,
#: so this is a factor of ten of headroom -- and an order that misses by more
#: than this cannot be called one order rather than its neighbour anyway.
M_TOLERANCE = 0.25

#: fewest orders the invariant can be measured from
MIN_FOR_INVARIANT = 4


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


def invariant(centre_wavelengths):
    """The echelle invariant `K = m*lambda`, from the order spacing alone.

    For adjacent orders `m1*lam1 = (m1-1)*lam2`, so `m1 = lam2/(lam2 - lam1)`
    with no integer search and no assumption about where the block starts.
    Taken as a median over every adjacent pair, because a pair is only wrong
    if one of its two members is: a few diverged orders, or a few missing
    ones, cannot move it. Returns NaN if it cannot be measured.
    """
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
    """Physical order numbers, from the assignment that makes m*lambda flat.

    Returns a float array in input order, NaN for any order whose m could not
    be established -- which the caller must not guess at.

    `m` is read off each order's own wavelength as `m = K/lambda`, not from its
    rank in the sorted list. Rank is wrong in two situations that both turn up
    on ordinary nights. An order whose fit diverged takes a rank it does not
    own, shifting every order on one side of it by one; and an order that
    failed to solve is absent from the list altogether, so a consecutive run of
    m steps straight over the gap. Neither is visible afterwards, because the
    whole block shifts together and `m*lambda` stays as smooth as it ever was.

    A single runaway order is also enough to wreck the scale if it is allowed
    to vote -- a fit that has diverged can report a central wavelength of
    10^7 A -- so K comes from a median over pairs rather than a fit to all of
    them, and `valid` can exclude known-bad orders from setting it.
    """
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

    Every order also gets a `dispersion_ok` flag from a monotonicity test. A
    relation that reverses inside the detector has the wrong shape rather than
    the wrong scale, so rescaling cannot repair it; `max_nonmonotonic` is the
    tolerated fraction of steps running against the trend.

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

    # A cubic through a dozen lines can fold inside the detector. Such an
    # order still has a normal *median* dispersion, so the m^2*dispersion test
    # above passes it; only the sign changes reveal it. Rescaling cannot help,
    # so mark it and let the merge drop it.
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
    # Floor the scale. `scatter_wl` says how well a quadratic describes the
    # orders it was fitted to, and on a good night that can be well under an
    # Angstrom-order -- which would make this check fire on differences far
    # below the accuracy of the wavelength solution itself. A real order is not
    # located better than a few tenths of an Angstrom, so do not pretend to
    # test it more tightly than that.
    scale = max(model["scatter_wl"], 0.2 * float(np.nanmedian(m)))
    if not np.isfinite(scale) or scale <= 0:
        return 0

    nfixed = 0
    unplaced = []
    for o, mm, r in zip(good, m, resid):
        if not np.isfinite(mm):
            # Its centre does not sit where any order's does, so there is no
            # saying which order to restore it to. Guessing is what seeding an
            # unsolved order from its neighbours tried and lost on: one order
            # out is ~10 A at the red end and still looks entirely plausible.
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
