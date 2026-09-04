"""Blaze model in grating phase.

A faint flat records no blaze, so the order needs the whole one modelled. A
bright flat records the wrong one, leaving a ~30% decline per order that a
cubic in wavelength cannot follow; its fitting residual is the 1-2% dip at
every order centre.
"""

import numpy as np
from scipy.ndimage import median_filter

#: flats fainter than this (median counts) are assumed not to record a blaze
FAINT_FLAT = 8.0
#: only orders at least this bright are used to build the model
BRIGHT_FLAT = 8.0

#: Residual continuum vs grating phase: flux/Kurucz over 200 orders of seven
#: alp Lyr exposures, 2021-2026. Unmeasured phases are omitted, not
#: extrapolated. RESIDUAL_RATIO tilts it by wavelength (order groups differ by
#: 3-5%), interpolated in log lambda between RESIDUAL_LAM, 1.0 where a group
#: was unmeasured.
#:
#: Do not select orders for this by the level of flat-fielded flux: it falls
#: 300 -> 0.4 counts blue to red as the flat brightens, so such a cut keeps
#: only the blue half.
RESIDUAL_X0 = -0.66
RESIDUAL_DX = 0.02
RESIDUAL_C = np.array([
    1.0064, 1.0016, 0.9967, 0.9892, 0.9816, 0.9765, 0.9715, 0.9652,
    0.9589, 0.9528, 0.9467, 0.9378, 0.9288, 0.9215, 0.9142, 0.9069,
    0.8996, 0.8925, 0.8853, 0.8803, 0.8753, 0.8694, 0.8636, 0.8574,
    0.8512, 0.8455, 0.8399, 0.8342, 0.8286, 0.8227, 0.8169, 0.8108,
    0.8047, 0.7996, 0.7945, 0.7903, 0.7862, 0.7837, 0.7812, 0.7792,
    0.7773, 0.7762, 0.7752, 0.7731, 0.7710, 0.7688, 0.7666, 0.7637,
    0.7607, 0.7577, 0.7546, 0.7511, 0.7475, 0.7421, 0.7366, 0.7328,
    0.7291, 0.7287, 0.7284, 0.7297, 0.7310, 0.7431, 0.7552, 0.7571,
    0.7591, 0.7799, 0.8006,
])
RESIDUAL_LAM = np.array([4139., 4505., 4943., 5476., 6355.])
RESIDUAL_RATIO = np.array([
    [  # 4139 A
     0.9990, 0.9951, 0.9913, 0.9877, 0.9841, 0.9826, 0.9811, 0.9771,
     0.9731, 0.9727, 0.9727, 0.9727, 0.9763, 0.9785, 0.9792, 0.9800,
     0.9807, 0.9819, 0.9819, 0.9819, 0.9816, 0.9816, 0.9816, 0.9840,
     0.9865, 0.9865, 0.9865, 0.9885, 0.9905, 0.9935, 0.9966, 1.0003,
     1.0040, 1.0066, 1.0093, 1.0146, 1.0199, 1.0226, 1.0252, 1.0307,
     1.0360, 1.0361, 1.0362, 1.0402, 1.0444, 1.0491, 1.0539, 1.0577,
     1.0616, 1.0667, 1.0718, 1.0771, 1.0824, 1.0884, 1.0944, 1.0944,
     1.0944, 1.0923, 1.0901, 1.0781, 1.0662, 1.0330, 1.0008, 0.9986,
     0.9964, 0.9964, 0.9964,
    ],
    [  # 4505 A
     1.0000, 1.0051, 1.0051, 1.0051, 1.0013, 0.9958, 0.9902, 0.9884,
     0.9866, 0.9838, 0.9826, 0.9826, 0.9826, 0.9835, 0.9845, 0.9878,
     0.9912, 0.9912, 0.9912, 0.9910, 0.9910, 0.9910, 0.9911, 0.9916,
     0.9920, 0.9934, 0.9934, 0.9940, 0.9941, 0.9941, 0.9947, 0.9980,
     1.0014, 1.0017, 1.0020, 1.0051, 1.0082, 1.0114, 1.0146, 1.0178,
     1.0209, 1.0232, 1.0255, 1.0304, 1.0353, 1.0386, 1.0419, 1.0434,
     1.0449, 1.0498, 1.0547, 1.0605, 1.0664, 1.0688, 1.0713, 1.0726,
     1.0726, 1.0726, 1.0707, 1.0638, 1.0570, 1.0320, 1.0079, 1.0079,
     1.0079, 1.0079, 1.0000,
    ],
    [  # 4943 A
     1.0000, 1.0000, 1.0113, 1.0113, 1.0113, 1.0038, 0.9962, 0.9934,
     0.9906, 0.9868, 0.9847, 0.9839, 0.9832, 0.9832, 0.9817, 0.9811,
     0.9811, 0.9811, 0.9805, 0.9804, 0.9791, 0.9787, 0.9787, 0.9787,
     0.9787, 0.9787, 0.9787, 0.9787, 0.9783, 0.9783, 0.9783, 0.9795,
     0.9795, 0.9802, 0.9803, 0.9803, 0.9810, 0.9814, 0.9814, 0.9815,
     0.9819, 0.9822, 0.9835, 0.9843, 0.9851, 0.9865, 0.9879, 0.9891,
     0.9898, 0.9898, 0.9898, 0.9896, 0.9892, 0.9890, 0.9888, 0.9886,
     0.9884, 0.9811, 0.9737, 0.9639, 0.9639, 0.9639, 1.0000, 1.0000,
     1.0000, 1.0000, 1.0000,
    ],
    [  # 5476 A
     1.0000, 1.0000, 1.0000, 1.0000, 1.0238, 1.0241, 1.0241, 1.0241,
     1.0186, 1.0146, 1.0112, 1.0109, 1.0106, 1.0092, 1.0071, 1.0057,
     1.0057, 1.0053, 1.0049, 1.0049, 1.0035, 1.0030, 1.0025, 1.0020,
     1.0016, 1.0010, 1.0003, 0.9982, 0.9981, 0.9981, 0.9981, 0.9981,
     0.9981, 0.9961, 0.9937, 0.9920, 0.9903, 0.9892, 0.9881, 0.9876,
     0.9871, 0.9868, 0.9865, 0.9860, 0.9854, 0.9842, 0.9834, 0.9832,
     0.9830, 0.9807, 0.9780, 0.9752, 0.9724, 0.9718, 0.9713, 0.9676,
     0.9639, 0.9559, 0.9559, 0.9559, 1.0000, 1.0000, 1.0000, 1.0000,
     1.0000, 1.0000, 1.0000,
    ],
    [  # 6355 A
     1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
     1.0491, 1.0491, 1.0491, 1.0466, 1.0441, 1.0433, 1.0424, 1.0401,
     1.0377, 1.0368, 1.0358, 1.0334, 1.0309, 1.0307, 1.0304, 1.0299,
     1.0294, 1.0274, 1.0254, 1.0236, 1.0218, 1.0206, 1.0194, 1.0157,
     1.0123, 1.0121, 1.0120, 1.0096, 1.0068, 1.0054, 1.0040, 1.0033,
     1.0026, 1.0018, 1.0009, 0.9996, 0.9983, 0.9970, 0.9958, 0.9944,
     0.9930, 0.9899, 0.9867, 0.9850, 0.9833, 0.9764, 0.9764, 0.9764,
     1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
     1.0000, 1.0000, 1.0000,
    ],
])


def dead_pixels(flx, win=11, drop=0.7):
    """Isolated pixels far below their neighbours."""
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
    """Physical interference order m for each order, from the spacing alone."""
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


def residual_continuum(wl, m, Kb, min_pixels=50):
    """`RESIDUAL_C` at this order's phases, tilted for its wavelength. None if
    too little of the order falls inside the tabulated range."""
    if not np.isfinite(m) or not np.isfinite(Kb):
        return None
    xt = RESIDUAL_X0 + RESIDUAL_DX * np.arange(len(RESIDUAL_C))
    lam = float(np.nanmedian(wl))
    if not np.isfinite(lam):
        return None
    u = np.clip(np.log(lam), np.log(RESIDUAL_LAM[0]), np.log(RESIDUAL_LAM[-1]))
    ratio = np.array([np.interp(u, np.log(RESIDUAL_LAM), RESIDUAL_RATIO[:, j])
                      for j in range(RESIDUAL_RATIO.shape[1])])
    c = np.interp(phase(wl, m, Kb), xt, RESIDUAL_C * ratio,
                  left=np.nan, right=np.nan)
    return None if np.isfinite(c).sum() < min_pixels else c


def required_blaze(wls, flxs, m, Kb, tmpl_wl, tmpl_rect, rv=0.0,
                   xgrid=None, level_min=None, sci_min=8.0, clip=3.0,
                   niter=3, smooth=9, min_orders=5, skip=(), ids=None):
    """The blaze the data actually needs, measured directly on the star."""
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
    """The blaze the flat failed to record:"""
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
                      a=1.0444, x0=-0.0013, amp=1.0022, residual=True):
    """Per-order blaze for `poly_normalization(blazes=...)`, or None per order.

    A faint-flat order gets the modelled blaze, every other order the residual.
    ``residual=False`` restores the faint-flat-only behaviour.
    """
    ids = list(range(len(wls))) if ids is None else list(ids)

    skip = set(skip) | bad_orders(wls, ids)

    m, _ = order_numbers(wls, ids, skip=skip)
    Kb = blaze_peak_invariant(wls, flats, m)
    if not np.isfinite(Kb):
        return [None] * len(wls), []

    xg = np.linspace(-1.0, 1.0, 401)
    prof = amp * np.sinc(a * (xg - x0)) ** 2

    out, applied = [], []
    for oid, w, fl, sci, mi in zip(ids, wls, flats, sciences, m):
        b = None
        if oid not in skip and np.isfinite(mi):
            lvl = np.nan if fl is None else float(np.nanmedian(fl))
            if (fl is not None and np.isfinite(lvl) and lvl < faint_flat
                    and float(np.nanmedian(w)) < blue_limit
                    and float(np.nanmedian(sci)) >= sci_min):
                b = _faint_flat_blaze(w, fl, mi, Kb, xg, prof)
            if b is None and residual:
                b = residual_continuum(w, mi, Kb)
        out.append(b)
        if b is not None:
            applied.append(oid)
    return out, applied


def _faint_flat_blaze(wl, flat, m, Kb, xgrid, profile):
    """The whole blaze, for an order whose flat is too faint to show one."""
    tot = np.interp(phase(wl, m, Kb), xgrid, profile,
                    left=np.nan, right=np.nan)
    pk_fl = np.nanpercentile(flat, 98)
    if not np.isfinite(pk_fl) or pk_fl <= 0:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        g = tot / np.clip(np.asarray(flat, float) / pk_fl, 0.05, None)
    pk = np.nanpercentile(g, 98)
    if not np.isfinite(pk) or pk <= 0:
        return None
    return g / pk


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
    """Order ids whose wavelength solution cannot be trusted."""
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
