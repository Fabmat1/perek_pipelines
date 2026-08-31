"""Blaze model in grating phase, for orders whose flat is too faint to show one.

In the bluest OES orders the flat lamp holds ~1 count, so the flat records
almost no blaze of its own (peak/median 1.08 at order 1 against 1.65 by order
6). Dividing by it removes the pixel response but leaves the star's blaze in
the spectrum, and no low-order polynomial follows a blaze.

The blaze belongs to the grating, so in the phase coordinate

    X = m * (lambda / lambda_blaze - 1)

it is the same curve for every order -- the well-exposed flats collapse onto
one profile with 5.2% scatter over 49 orders, which is what licenses
transferring the shape into the faint ones. A fractional-position profile does
not work: order 1 covers 1.37 free spectral ranges, order 52 only 0.49.

`m` need not be assumed. For an echelle m*lambda is constant, so adjacent
orders give m = lam2/(lam2 - lam1), which comes out as clean integers (41-93).
On 20260829, m*lambda = 355981 A and m*lambda_blaze = 355782 +- 290 A.
"""

import numpy as np
from scipy.ndimage import median_filter

#: flats fainter than this (median counts) are assumed not to record a blaze
FAINT_FLAT = 8.0
#: only orders at least this bright are used to build the model
BRIGHT_FLAT = 8.0


def dead_pixels(flx, win=11, drop=0.7):
    """Isolated pixels far below their neighbours.

    The extraction leaves masked detector columns at exactly the bias level,
    so `science - bias` is 0.0 and the propagated error NaN -- an error-based
    cut never sees them. Cut on local contrast, so a genuinely faint order is
    not mistaken for a dead pixel.
    """
    flx = np.asarray(flx, float)
    fin = np.isfinite(flx)
    out = np.zeros(flx.shape, bool)
    if fin.sum() < win + 5:
        return out
    sm = np.full(flx.shape, np.nan)
    sm[fin] = median_filter(flx[fin], size=win, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = (flx - sm) / np.maximum(np.abs(sm), 1e-9)
    out[fin] = rel[fin] < -drop
    out |= (flx == 0) & fin
    return out


def order_numbers(wls, ids=None, skip=()):
    """Physical interference order m for each order, from the spacing alone.

    Returns (m_array, K) with K the fitted m*lambda invariant. `skip` lists
    order ids whose wavelength solution is not to be trusted -- a bad solution
    would corrupt K for every order.
    """
    lam = np.array([float(np.nanmedian(w)) for w in wls])
    ids = np.arange(len(wls)) if ids is None else np.asarray(ids)
    ok = np.array([i not in skip for i in ids]) & np.isfinite(lam)
    idx = np.argsort(np.where(ok, lam, np.inf))[:ok.sum()]
    ls = lam[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        m_local = ls[1:] / np.diff(ls)
    good = np.isfinite(m_local) & (m_local > 10) & (m_local < 300)
    if good.sum() < 3:
        return np.full(len(wls), np.nan), np.nan
    K = float(np.median(m_local[good] * ls[:-1][good]))
    with np.errstate(divide="ignore", invalid="ignore"):
        return K / lam, K


def blaze_peak_invariant(wls, flats, m, level_min=BRIGHT_FLAT):
    """m * lambda_blaze, measured on the flats that are bright enough."""
    vals = []
    for w, f, mi in zip(wls, flats, m):
        if f is None or not np.isfinite(mi) or np.nanmedian(f) < level_min:
            continue
        i = np.argsort(w)
        ws, fs = np.asarray(w)[i], np.asarray(f)[i]
        fin = np.isfinite(fs)
        if fin.sum() < 100:
            continue
        sm = median_filter(fs[fin], size=max(51, fin.sum() // 12),
                           mode="nearest")
        vals.append(mi * float(ws[fin][int(np.nanargmax(sm))]))
    return float(np.median(vals)) if vals else np.nan


def build_profile(wls, flats, m, Kb, xgrid=None, level_min=BRIGHT_FLAT):
    """Median normalised blaze in grating phase, from the well-exposed flats."""
    if xgrid is None:
        xgrid = np.linspace(-0.8, 0.8, 321)
    stack = []
    for w, f, mi in zip(wls, flats, m):
        if f is None or not np.isfinite(mi) or np.nanmedian(f) < level_min:
            continue
        X = phase(w, mi, Kb)
        y = np.asarray(f, float) / np.nanpercentile(f, 98)
        i = np.argsort(X)
        stack.append(np.interp(xgrid, X[i], y[i], left=np.nan, right=np.nan))
    if not stack:
        return xgrid, None, 0
    return xgrid, np.nanmedian(np.vstack(stack), axis=0), len(stack)


def phase(wl, m, Kb):
    """Grating phase X = m*(lambda/lambda_blaze - 1), with lambda_b = Kb/m."""
    return m * (np.asarray(wl, float) * m / Kb - 1.0)


def required_blaze(wls, flxs, m, Kb, tmpl_wl, tmpl_rect, rv=0.0,
                   xgrid=None, level_min=None, sci_min=8.0, clip=3.0,
                   niter=3, smooth=9, min_orders=5, skip=(), ids=None):
    """The blaze the data actually needs, measured directly on the star.

    For a star with a known model the required continuum is not something to
    be inferred from the flat at all -- it is

        C_true = flux / model

    and in grating phase every order should give the same curve. Stacking the
    orders in X and taking a robust median therefore measures the blaze the
    *instrument* imposes.

    **`flxs` must be the flux BEFORE flat division** (science - bias). With
    flat-fielded flux the well-exposed orders contribute a flat curve, since
    their blaze is already gone, and the stack peaks in the wrong place.

    Better conditioned than the flat-derived profile at the order edges, where
    `model / flat` divides two small noisy numbers but `flux / model` stays
    well determined as long as the star is bright.

    Robustness matters: one dead column at the bias level would pull the
    stacked median down. Hence dead-pixel rejection, deep template lines
    excluded, sigma-clipping, and a median rather than a mean. Orders below
    `sci_min` counts are noise and do not contribute, and phases reached by
    fewer than `min_orders` orders are returned as NaN -- past |X| ~ 0.75 only
    one or two orders reach, so the median there is a single order's opinion.
    """
    if xgrid is None:
        xgrid = np.linspace(-0.9, 0.9, 361)
    rows = []
    for w, f, mi, oid in zip(wls, flxs, m, ids if ids is not None
                             else range(len(wls))):
        if f is None or not np.isfinite(mi) or oid in skip:
            continue
        w = np.asarray(w, float)
        f = np.asarray(f, float)
        if np.nanmedian(f) < sci_min:
            continue
        t = np.interp(w, np.asarray(tmpl_wl, float) * (1 + rv / 299792.458),
                      tmpl_rect, left=np.nan, right=np.nan)
        good = (np.isfinite(f) & np.isfinite(t) & (t > 0.5) & (f > 0)
                & ~dead_pixels(f))
        if good.sum() < 100:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.where(good, f / t, np.nan)
        # each order carries its own throughput; normalise it away before
        # stacking, using a robust level rather than the peak
        lvl = np.nanpercentile(c, 90)
        if not np.isfinite(lvl) or lvl <= 0:
            continue
        X = phase(w, mi, Kb)
        i = np.argsort(X)
        rows.append(np.interp(xgrid, X[i], (c / lvl)[i],
                              left=np.nan, right=np.nan))
    if not rows:
        return xgrid, None, 0
    A = np.vstack(rows)
    # phases reached by too few orders are not measured, they are anecdote
    thin = np.sum(np.isfinite(A), axis=0) < min_orders
    A[:, thin] = np.nan
    prof = np.nanmedian(A, axis=0)
    for _ in range(niter):
        resid = A - prof
        s = 1.4826 * np.nanmedian(np.abs(resid - np.nanmedian(resid, axis=0)),
                                  axis=0)
        s = np.where(np.isfinite(s) & (s > 0), s, np.inf)
        A = np.where(np.abs(resid) > clip * s, np.nan, A)
        prof = np.nanmedian(A, axis=0)
    if smooth and smooth > 1:
        fin = np.isfinite(prof)
        if fin.sum() > smooth:
            sm = prof.copy()
            sm[fin] = median_filter(prof[fin], size=smooth, mode="nearest")
            prof = sm
    pk = np.nanpercentile(prof, 98)
    if np.isfinite(pk) and pk > 0:
        prof = prof / pk
    return xgrid, prof, len(rows)


def residual_blaze(wl, flat, m, Kb, xgrid, profile, floor=0.05):
    """The blaze the flat failed to record: model / (what the flat shows).

    Both are peak-normalised, so this is a pure shape correction. NaN outside
    the phase range the bright orders cover -- the profile is measured, not
    analytic, so extrapolating past its last constraint would be a guess.
    """
    X = phase(wl, m, Kb)
    fin = np.isfinite(profile)
    if fin.sum() < 5:
        return None
    lo, hi = float(xgrid[fin].min()), float(xgrid[fin].max())
    model = np.where((X >= lo) & (X <= hi),
                     np.interp(X, xgrid[fin], profile[fin]), np.nan)
    f = np.asarray(flat, float)
    fl = f / np.nanpercentile(f, 98)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = model / np.clip(fl, floor, None)
    r = np.where(np.isfinite(r), r, np.nan)
    pk = np.nanpercentile(r, 98)
    if not np.isfinite(pk) or pk <= 0:
        return None
    return r / pk


def blazes_for_orders(wls, flats, sciences, ids=None, skip=(),
                      faint_flat=FAINT_FLAT, sci_min=8.0, blue_limit=4200.0,
                      a=1.0444, x0=-0.0013, amp=1.0022):
    """Per-order blaze for `poly_normalization(blazes=...)`, or None per order.

    The default sinc^2 parameters were measured on Vega and serve both the
    2021 and 2026 epochs: the widths agree to 0.1 sigma and cross-applying one
    epoch's fit to the other costs under 0.003 in rms.

    An order is corrected only if its flat is too faint to have recorded a
    blaze (brighter flats already remove it), it carries at least `sci_min`
    counts (below ~8 the spectrum is noise and dividing by a small blaze
    amplifies it), and it lies blueward of `blue_limit`.

    Returns (blazes, applied_ids).
    """
    ids = list(range(len(wls))) if ids is None else list(ids)

    # Drop orders whose wavelength solution is visibly broken before they can
    # corrupt anything: left in, such an order both skews the m*lambda
    # invariant and receives a meaningless "blaze" of its own.
    skip = set(skip) | bad_orders(wls, ids)

    m, _ = order_numbers(wls, ids, skip=skip)
    Kb = blaze_peak_invariant(wls, flats, m)
    if not np.isfinite(Kb):
        return [None] * len(wls), []

    xg = np.linspace(-1.0, 1.0, 401)
    prof = amp * np.sinc(a * (xg - x0)) ** 2

    out, applied = [], []
    for oid, w, fl, sci, mi in zip(ids, wls, flats, sciences, m):
        lvl = np.nan if fl is None else float(np.nanmedian(fl))
        if (oid in skip or fl is None or not np.isfinite(lvl)
                or not np.isfinite(mi) or lvl >= faint_flat
                or float(np.nanmedian(w)) >= blue_limit
                or float(np.nanmedian(sci)) < sci_min):
            out.append(None)
            continue
        tot = np.interp(phase(w, mi, Kb), xg, prof, left=np.nan, right=np.nan)
        pk_fl = np.nanpercentile(fl, 98)
        if not np.isfinite(pk_fl) or pk_fl <= 0:
            out.append(None)
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            g = tot / np.clip(np.asarray(fl, float) / pk_fl, 0.05, None)
        pk = np.nanpercentile(g, 98)
        if not np.isfinite(pk) or pk <= 0:
            out.append(None)
            continue
        out.append(g / pk)
        applied.append(oid)
    return out, applied


def monotonic_fraction(wl):
    """Fraction of steps running against the trend. 0 = strictly monotonic."""
    dw = np.diff(np.asarray(wl, float))
    fin = np.isfinite(dw)
    if fin.sum() < 2:
        return 0.0
    up, dn = int(np.sum(dw[fin] > 0)), int(np.sum(dw[fin] < 0))
    if up + dn == 0:
        return 1.0
    return min(up, dn) / float(up + dn)


def bad_orders(wls, ids=None, max_nonmonotonic=0.02, span_factor=3.0):
    """Order ids whose wavelength solution cannot be trusted.

    Two symptoms, catching different failures: a dispersion relation that
    folds back inside the detector, and a span more than `span_factor` times
    the median or less than its reciprocal. `calibrate.fit_dispersion` now
    prevents folding at the source, but captures made before that still need
    detecting. Which order is broken varies by night, so never hard-code it.
    """
    ids = list(range(len(wls))) if ids is None else list(ids)
    spans = np.array([float(np.nanmax(w) - np.nanmin(w)) for w in wls])
    ref = float(np.nanmedian(spans))
    bad = set()
    for oid, w, sp in zip(ids, wls, spans):
        if not np.isfinite(sp) or ref <= 0:
            bad.add(oid)
            continue
        if sp > span_factor * ref or sp < ref / span_factor:
            bad.add(oid)
        elif monotonic_fraction(w) > max_nonmonotonic:
            bad.add(oid)
    return bad
