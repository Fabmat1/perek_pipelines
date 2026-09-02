"""Regression tests for the reduction pipeline."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from echelle_reduction import (normalize_single_order, _fit_continuum,
                               get_detector_noise, red_edge_keep)
from identify_orders import trace_windows, FALLBACK_WINDOWS
from orders import SpectralOrder, gaussian_pixel_weights_2d
from calibrate import (fit_dispersion, parse_idcomp,
                       solve_dispersion_shift, solve_dispersion_shifts)
import tools
from tools import shared, publish_shared, polynomial
import grating
from template import output_stem
from paths import select_idcomp_dir


def _norm_args(wl, flx, ignore_windows, neighbours=(), extrapolated_out=True):
    """The positional argument tuple `normalize_single_order` expects."""
    return (0, wl, flx, 3, ignore_windows, 31, 0.25,
            1.5, [(w, f, None) for w, f in neighbours], 3.0,
            extrapolated_out, 8.0, 6.0, 4,
            None, 0.20, False)


def _fake_order(npix=64, ncol=200, centre=40.0, width=4.0):
    """A SpectralOrder with a flat trace, enough to extract along."""
    o = SpectralOrder(1)
    o.solution = [0.0, 0.0, 0.0, centre]          # cubic -> constant row
    o.pixel_x = np.arange(ncol, dtype=float)
    o.order_width = np.full(ncol, width)
    o.generate_width_fcn()
    return o


def test_ignore_window_at_order_end_keeps_pixels_without_a_neighbour():
    """The O2 A-band case:"""
    wl = np.linspace(7505.0, 7631.0, 2000)
    flx = np.ones_like(wl) * 100.0
    flx[wl > 7590] *= 0.3                       # deep telluric absorption

    _, _, _, keep, _ = normalize_single_order(
        _norm_args(wl, flx, [(7590.0, 7660.0)]))

    # everything except the reddest `edge_width`, which is cut on purpose
    tail = (wl > 7598.0) & (wl < wl.max() - 1.5)
    assert keep[tail].all(), (
        "pixels past the start of the ignore window were dropped even though "
        "no neighbouring order covers them: lost %.1f A"
        % (wl[tail & ~keep].max() - wl[tail & ~keep].min()))
    assert not keep[-1], "the deliberate red-edge cut went missing"


def test_ignore_window_at_order_end_drops_pixels_when_a_neighbour_covers_them():
    """Where an adjacent order did measure those wavelengths, the extrapolated
    continuum is still discarded in its favour -- that was the point of the
    check and it has to keep working."""
    wl = np.linspace(7505.0, 7631.0, 2000)
    flx = np.ones_like(wl) * 100.0
    flx[wl > 7590] *= 0.3

    neighbour_wl = np.linspace(7560.0, 7700.0, 2000)
    neighbour_flx = np.ones_like(neighbour_wl) * 100.0

    _, _, _, keep, _ = normalize_single_order(
        _norm_args(wl, flx, [(7590.0, 7660.0)],
                   neighbours=[(neighbour_wl, neighbour_flx)]))

    assert not keep[wl > 7600.0].any(), \
        "a covered, extrapolated tail should be left to the neighbouring order"


def test_interior_ignore_window_never_drops_anything():
    """H-alpha sits in an ignore window in the middle of its order."""
    wl = np.linspace(6532.0, 6642.0, 2000)
    flx = np.ones_like(wl) * 100.0
    flx[(wl > 6555) & (wl < 6570)] *= 0.4

    _, _, _, keep, _ = normalize_single_order(
        _norm_args(wl, flx, [(6540.0, 6590.0)]))

    halpha = (wl > 6540.0) & (wl < 6590.0)
    assert keep[halpha].all(), "H-alpha was deleted from its own order"


def test_extracted_variance_matches_analytic_sum_of_squared_weights():
    gain, read_noise = 2.0, 10.0
    ny, nx = 80, 50
    rng = np.random.default_rng(0)
    bias = np.full((ny, nx), 110.0)
    raw = bias + rng.uniform(50.0, 5000.0, size=(ny, nx))

    o = _fake_order(ncol=nx)
    var = o.extract_variance_along_order(raw, bias, gain, read_noise)

    columns, y_pixels, weights = o.aperture(ny, nx)
    counts = raw[y_pixels, columns[:, None]] - bias[y_pixels, columns[:, None]]
    expected = np.einsum(
        "ij,ij->i",
        np.maximum(counts, 0.0) / gain + (read_noise / gain) ** 2,
        np.square(weights))

    assert np.allclose(var, expected)


def test_extraction_weights_sum_to_one_so_variance_needs_sum_of_squares():
    """The extracted value is a weighted mean, not a sum."""
    ny, nx = 80, 20
    o = _fake_order(ncol=nx)
    _, _, weights = o.aperture(ny, nx)

    assert np.allclose(weights.sum(axis=1), 1.0)
    sumw2 = np.square(weights).sum(axis=1)
    assert (sumw2 < 0.5).all(), \
        "sum(w^2) should be well below 1 for a multi-pixel aperture"


def test_variance_uses_gain_and_read_noise():
    """Doubling the gain halves the photon term; the read-noise floor is set by
    read_noise/gain in ADU."""
    ny, nx = 60, 10
    bias = np.zeros((ny, nx))
    bright = np.full((ny, nx), 10000.0)
    o = _fake_order(ncol=nx)

    v1 = o.extract_variance_along_order(bright, bias, 1.0, 0.0).copy()
    v2 = o.extract_variance_along_order(bright, bias, 2.0, 0.0).copy()
    assert np.allclose(v2, v1 / 2.0)

    dark = np.zeros((ny, nx))
    floor = o.extract_variance_along_order(dark, bias, 2.0, 10.0)
    _, _, weights = o.aperture(ny, nx)
    assert np.allclose(floor, (10.0 / 2.0) ** 2 * np.square(weights).sum(axis=1))


def test_variance_never_negative_below_bias():
    ny, nx = 60, 10
    bias = np.full((ny, nx), 110.0)
    below = np.full((ny, nx), 50.0)          # reads under bias
    o = _fake_order(ncol=nx)
    var = o.extract_variance_along_order(below, bias, 2.0, 10.0)
    assert (var > 0).all() and np.isfinite(var).all()


def test_get_detector_noise_reads_header_and_falls_back(tmpdir=None):
    from astropy.io import fits
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "frame.fit")
        hdu = fits.PrimaryHDU(np.zeros((4, 4), dtype=np.int16))
        hdu.header["GAIN"] = 2
        hdu.header["READNOIS"] = 10
        hdu.writeto(path)
        assert get_detector_noise(path) == (2.0, 10.0)

        path2 = os.path.join(d, "bare.fit")
        fits.PrimaryHDU(np.zeros((4, 4), dtype=np.int16)).writeto(path2)
        assert get_detector_noise(path2) == (1.0, 0.0)

    # an array rather than a path: nothing to read, so the defaults stand
    assert get_detector_noise(np.zeros((4, 4))) == (1.0, 0.0)


def test_fit_dispersion_mask_lines_up_with_unsorted_input():
    """`fit_dispersion` sorts internally; the mask it returns has to index the
    caller's unsorted arrays."""
    rng = np.random.default_rng(1)
    x = np.arange(50, 2000, 40, dtype=float)
    y = polynomial(x, 1e-9, -1e-6, 0.05, 4000.0)
    yerr = np.full_like(x, 0.01)

    # one obvious outlier, then shuffle so sorted != input order
    y[7] += 5.0
    perm = rng.permutation(len(x))
    x, y, yerr = x[perm], y[perm], yerr[perm]
    bad = int(np.where(perm == 7)[0][0])

    out = fit_dispersion(x, y, yerr)
    mask = out["mask_good"]
    assert not mask[bad], "the clipped line is not the one that was rejected"
    assert mask.sum() >= len(x) - 3


def _synthetic_profile(ny, rows, amps):
    p = np.zeros(ny)
    for r, a in zip(rows, amps):
        p += a * np.exp(-0.5 * ((np.arange(ny) - r) / 2.0) ** 2)
    return p


def test_trace_windows_locates_the_order_block_and_orients_on_the_flat():
    ny = 2000
    rows = np.arange(700, 1660, 20)
    # one frame bright in the blue, one bright in the red
    blue_amps = np.linspace(300, 20, len(rows))
    red_amps = np.linspace(20, 300, len(rows))
    profiles = [_synthetic_profile(ny, rows, blue_amps) + 1.0,
                _synthetic_profile(ny, rows, red_amps) + 1.0]
    # the flat is bright in the red, dark in the blue
    flat = _synthetic_profile(ny, rows, np.linspace(5, 5000, len(rows)))

    win, measured = trace_windows(profiles, flat_profile=flat)
    assert measured
    assert 650 < win["blue"][0] < 760, win
    assert 1550 < win["red"][1] < 1700, win
    assert win["blue"][1] < win["red"][0]
    # the noise window must not sit on the orders
    assert win["noise"][1] <= win["blue"][0]


def test_trace_windows_flips_when_the_flat_is_bright_at_low_rows():
    ny = 2000
    rows = np.arange(700, 1660, 20)
    profiles = [_synthetic_profile(ny, rows, np.full(len(rows), 100.0)) + 1.0]
    flat = _synthetic_profile(ny, rows, np.linspace(5000, 5, len(rows)))

    win, measured = trace_windows(profiles, flat_profile=flat)
    assert measured
    assert win["blue"][0] > win["red"][1], \
        "with a red-bright flat at low rows the blue end is the high one"


def test_trace_windows_falls_back_when_nothing_is_detectable():
    flat_noise = [np.zeros(2000)]
    win, measured = trace_windows(flat_noise)
    assert not measured
    assert win == FALLBACK_WINDOWS


def _echo_payload(key):
    """Pool worker: hand back what `shared()` sees for one key."""
    value = shared(key)
    if isinstance(value, np.ndarray):
        return value.shape, value.dtype.str, float(np.sum(value))
    return value


def test_a_payload_survives_the_trip_through_shared_memory():
    """Shapes, dtypes and nesting have to come back exactly as they went in.

    Payloads go to the workers as raw bytes in a shared block now, not as a
    pickle, so this is the round trip the whole pool rests on."""
    payload = {"zerod": np.array(3.5),
               "empty": np.zeros(0),
               "f32": np.arange(6, dtype=np.float32).reshape(2, 3),
               "bools": np.array([True, False]),
               "nested": {"pairs": [(1, 2), np.arange(3)]},
               "plain": "not an array"}
    shm = tools._publish_segment(payload)
    try:
        back = tools._attach_segment(shm.name)
        assert back["zerod"].shape == ()
        assert back["empty"].shape == (0,)
        assert back["f32"].dtype == np.float32
        np.testing.assert_array_equal(back["f32"], payload["f32"])
        np.testing.assert_array_equal(back["bools"], payload["bools"])
        np.testing.assert_array_equal(back["nested"]["pairs"][1], np.arange(3))
        assert back["nested"]["pairs"][0] == (1, 2)
        assert back["plain"] == "not an array"
        assert not back["f32"].flags.writeable   # the block is shared
    finally:
        tools._release_segments()
        shm.close()
        shm.unlink()


def test_the_pool_sees_each_payload_and_not_the_one_before_it():
    """The pool outlives a payload, so a stale block must not leak into it."""
    ncpu = tools.get_ncpu()
    tools.set_ncpu(2)
    try:
        with tools.shared_pool({"img": np.arange(10, dtype=float)}) as pool:
            first = pool.map(_echo_payload, ["img"] * 3)
        with tools.shared_pool({"img": np.arange(10, dtype=float) + 100}) as pool:
            second = list(pool.imap(_echo_payload, ["img"] * 3))
    finally:
        tools.set_ncpu(ncpu)
    assert first == [((10,), "<f8", 45.0)] * 3
    assert second == [((10,), "<f8", 1045.0)] * 3


def test_one_worker_runs_in_this_process_instead_of_opening_a_pool():
    ncpu = tools.get_ncpu()
    tools.set_ncpu(1)
    try:
        with tools.shared_pool({"img": np.arange(4, dtype=float)}) as pool:
            assert list(pool.imap(_echo_payload, ["img"])) == [((4,), "<f8", 6.0)]
    finally:
        tools.set_ncpu(ncpu)


def test_shared_raises_on_a_missing_key_but_honours_an_explicit_default():
    publish_shared({"present": 1})
    assert shared("present") == 1
    assert shared("absent", 42) == 42
    try:
        shared("absent")
    except KeyError:
        pass
    else:
        raise AssertionError("a missing key should raise without a default")


def test_fit_continuum_survives_an_order_with_almost_no_points():
    wl = np.linspace(4000.0, 4010.0, 4)
    flx = np.array([1.0, 1.1, 0.9, 1.0])
    cont = _fit_continuum(wl, flx, 3)
    assert np.isfinite(cont(wl)).all()


def test_normalize_marks_a_zero_continuum_instead_of_dividing_by_it():
    wl = np.linspace(4000.0, 4100.0, 500)
    flx = np.zeros_like(wl)
    _, norm, _, keep, _ = normalize_single_order(_norm_args(wl, flx, []))
    assert not np.isnan(norm[keep]).any(), \
        "pixels kept for the merge must have a usable normalisation"


def test_red_edge_keep_cuts_only_the_red_end():
    wl = np.linspace(5000.0, 5100.0, 1000)
    flx = np.ones_like(wl)
    keep = red_edge_keep(wl, flx, width=1.5)
    assert keep[0] and not keep[-1]
    assert wl[keep].max() < wl.max() - 1.4


def test_output_stem_strips_glob_characters_from_the_object_name():
    stem = output_stem("e202608290033.fit", "*_alp_xyz")
    assert "*" not in stem
    assert stem == "e202608290033_alp_xyz"


def test_output_stem_keeps_the_designation_of_a_variable_star():
    assert output_stem("e202608270038.fit", "V*_AB_Xyz") == "e202608270038_V_AB_Xyz"


def test_output_stem_leaves_ordinary_names_alone():
    assert output_stem("e202409010033.fit", "BD+00_0000") == "e202409010033_BD+00_0000"


def test_output_stem_only_drops_the_trailing_extension():
    # ".fit" also appears inside the name here
    assert output_stem("fit.fit", "XY_1") == "fit_XY_1"


_IDCOMP_RECORD = """begin\tidentify tzc01 - Ap 1
\taplow\t%(lo).2f
\taphigh\t%(hi).2f
\tfeatures\t%(n)d
%(rows)s\tfunction chebyshev
\torder 4
"""


def _idcomp_text(records):
    out = []
    for lo, hi, waves in records:
        rows = "".join("\t%10.2f %s  %s   4.0 1 1 \n" % (100.0 * (i + 1), w, w)
                       for i, w in enumerate(waves))
        out.append(_IDCOMP_RECORD % dict(lo=lo, hi=hi, n=len(waves), rows=rows))
    return "".join(out)


def test_parse_idcomp_keeps_only_the_last_record():
    import tempfile
    text = _idcomp_text([(853.04, 861.27, ["4027.0091", "4024.8025"]),
                         (853.04, 861.27, ["4027.0091", "4024.8025",
                                           "4022.0674"])])
    with tempfile.NamedTemporaryFile("w", suffix=".idcomp", delete=False) as fh:
        fh.write(text)
        path = fh.name
    try:
        aplow, aphigh, table = parse_idcomp(path)
    finally:
        os.unlink(path)
    assert (aplow, aphigh) == (853.04, 861.27)
    assert table.shape == (3, 6)


def test_parse_idcomp_returns_an_empty_table_for_a_file_that_is_not_a_list():
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write("import glob\nprint('not an idcomp file')\n")
        path = fh.name
    try:
        aplow, aphigh, table = parse_idcomp(path)
    finally:
        os.unlink(path)
    assert aplow is None and aphigh is None
    assert table.shape == (0, 6)


_CENTRES = (
    8678.625, 8473.728, 8276.686, 8088.606, 7908.878, 7736.961,
    7572.368, 7414.629, 7262.925, 7118.081, 6978.527, 6844.344,
    6715.224, 6590.885, 6471.071, 6355.534, 6244.051, 6136.411,
    6032.424, 5931.901, 5834.675, 5740.584, 5649.480, 5561.226,
    5475.686, 5392.738, 5312.268, 5234.176, 5158.323, 5084.651,
    5013.056, 4943.447, 4875.747, 4809.876, 4745.763, 4683.339,
    4622.534, 4563.289, 4505.544, 4449.244, 4394.334, 4340.763,
    4288.482, 4237.446, 4187.612, 4138.933, 4091.377, 4044.897,
    3999.462, 3955.223, 3911.616, 3869.126, 3827.221,
)
_DISPERSIONS = (
    -0.083525, -0.068963, -0.067354, -0.065838, -0.064361, -0.062970,
    -0.061621, -0.060339, -0.059886, -0.057920, -0.056778, -0.055689,
    -0.054635, -0.053626, -0.052647, -0.051701, -0.050797, -0.049919,
    -0.049069, -0.048256, -0.047461, -0.046695, -0.045955, -0.045233,
    -0.044535, -0.043861, -0.043207, -0.042586, -0.041949, -0.041350,
    -0.040769, -0.040196, -0.039642, -0.039103, -0.038582, -0.038072,
    -0.037571, -0.037088, -0.036608, -0.036151, -0.035702, -0.035263,
    -0.034834, -0.034412, -0.034000, -0.033602, -0.033219, -0.032845,
    -0.032472, -0.031946, -0.031736, -0.031367, -0.030575,
)
#: m of the first entry above; the block runs 93 down to 41
_M_REDDEST = 41.0


class _Order:
    """The three attributes `enforce_invariant` reads and writes."""

    def __init__(self, oid, wl):
        self.id = oid
        self.wl = wl
        self.dispersion_ok = True


def _real_orders(skip=(), npix=2048):
    """The reference block as orders, reddest first, `skip` left out."""
    x = np.arange(npix, dtype=float) - npix / 2.0
    curve = (x ** 2 - float(np.mean(x ** 2))) / npix ** 2
    out = []
    for i, (cen, disp) in enumerate(zip(_CENTRES, _DISPERSIONS)):
        if i in skip:
            continue
        out.append(_Order(i, cen + disp * x + 4.0 * disp * curve))
    return out


def _centres(orders):
    return np.array([float(np.median(o.wl)) for o in orders])


def test_invariant_recovers_m_lambda_from_the_bundled_reference_list():
    K = grating.invariant(_CENTRES)
    # m*lambda = 355981 A on 20260829; the 2023 list sits within a few hundred
    assert 355000 < K < 357000, K
    m = K / np.asarray(_CENTRES)
    assert np.max(np.abs(m - np.round(m))) < 0.05, \
        "K/lambda has to land near an integer or no order can be placed"


def test_invariant_is_unmoved_by_a_diverged_order():
    """A cubic through a dozen seed lines can report a centre of 10^7 A."""
    clean = grating.invariant(_CENTRES)
    with_runaway = grating.invariant(list(_CENTRES) + [1.0e7])
    assert abs(with_runaway - clean) / clean < 1e-6, \
        "one diverged order moved the invariant"


def test_invariant_is_unmoved_by_orders_missing_from_the_list():
    """Unsolved orders are dropped before `enforce_invariant` sees them, so the
    list arrives with holes in it."""
    kept = [c for i, c in enumerate(_CENTRES) if i not in (7, 8, 20, 33, 34)]
    assert abs(grating.invariant(kept) - grating.invariant(_CENTRES)) < 50.0


def test_invariant_returns_nan_when_it_cannot_be_measured():
    assert not np.isfinite(grating.invariant([5000.0, 5100.0]))       # too few
    assert not np.isfinite(grating.invariant([1.0, 2.0, 3.0, 4.0]))   # not orders
    assert not np.isfinite(grating.invariant([5000.0] * 10))          # no spacing


def test_order_numbers_are_consecutive_on_the_bundled_reference_list():
    m = grating.order_numbers(_CENTRES)
    assert np.isfinite(m).all(), "every reference order has to be placed"
    assert np.array_equal(m, np.arange(len(m)) + _M_REDDEST), \
        "the reference block is m = 41 (reddest) up to 93 (bluest), unbroken"


def test_order_numbers_survive_an_order_missing_from_the_middle():
    """The gap regression."""
    full = grating.order_numbers(_CENTRES)
    gap_at = 25
    kept = [c for i, c in enumerate(_CENTRES) if i != gap_at]
    m = grating.order_numbers(kept)
    expected = np.delete(full, gap_at)
    assert np.array_equal(m, expected), \
        "m shifted by %s past the gap" % np.unique(m - expected)


def test_order_numbers_of_healthy_orders_are_unmoved_by_a_diverged_neighbour():
    """The runaway regression."""
    full = grating.order_numbers(_CENTRES)
    hurt = list(_CENTRES)
    hurt[10] = 1.0e7
    m = grating.order_numbers(hurt)
    healthy = np.arange(len(hurt)) != 10
    assert np.array_equal(m[healthy], full[healthy]), \
        "a diverged order moved its neighbours by %s" \
        % np.unique(m[healthy] - full[healthy])


def test_order_numbers_refuses_to_place_a_diverged_order():
    """No integer is better than a wrong one:"""
    hurt = list(_CENTRES)
    hurt[10] = 1.0e7
    hurt[20] = 0.5 * (_CENTRES[20] + _CENTRES[21])   # halfway to its neighbour
    m = grating.order_numbers(hurt)
    assert not np.isfinite(m[10]), "a 10^7 A centre was given an order number"
    assert not np.isfinite(m[20]), \
        "an order halfway between two orders was placed on one of them"


def test_order_numbers_follow_the_input_order_not_the_sorted_one():
    """The result indexes the caller's list, which is in detector order, not
    wavelength order."""
    rng = np.random.default_rng(3)
    perm = rng.permutation(len(_CENTRES))
    shuffled = np.asarray(_CENTRES)[perm]
    m = grating.order_numbers(shuffled)
    assert np.array_equal(m, grating.order_numbers(_CENTRES)[perm])


def test_order_numbers_honours_the_valid_mask_for_the_scale():
    """`valid` keeps an order from setting K."""
    hurt = list(_CENTRES)
    hurt[3] = 6.0e6
    valid = np.ones(len(hurt), bool)
    valid[3] = False
    m = grating.order_numbers(hurt, valid=valid)
    healthy = np.arange(len(hurt)) != 3
    assert np.array_equal(m[healthy], grating.order_numbers(_CENTRES)[healthy])


def test_order_numbers_returns_all_nan_on_degenerate_input():
    for bad in ([], [np.nan] * 20, [0.0] * 20, [-5000.0] * 20):
        m = grating.order_numbers(bad)
        assert len(m) == len(bad)
        assert not np.isfinite(m).any(), bad


def test_fit_grating_recovers_a_quadratic_in_m_exactly():
    m = np.arange(41.0, 94.0)
    wc = (355981.0 + 0.5 * m - 0.002 * m ** 2) / m
    disp = (-140.0 - 0.1 * m) / m ** 2
    model = grating.fit_grating(m, wc, disp)
    assert np.allclose(np.polyval(model["p_wl"], m), m * wc)
    assert np.allclose(np.polyval(model["p_disp"], m), disp * m ** 2)
    assert model["scatter_wl"] < 1e-6 and model["scatter_disp"] < 1e-6


def test_fit_grating_scatter_tracks_the_noise_it_was_given():
    """`scatter_wl` sets the tolerance every order is then judged against, so
    it has to measure the spread of the orders rather than anything else."""
    m = np.arange(41.0, 94.0)
    rng = np.random.default_rng(5)
    wc = 355981.0 / m
    for sigma in (1.0, 10.0):
        noisy = (355981.0 + rng.normal(0.0, sigma, m.size)) / m
        got = grating.fit_grating(m, noisy, -140.0 / m ** 2)["scatter_wl"]
        assert 0.4 * sigma < got < 2.5 * sigma, (sigma, got)
    assert grating.fit_grating(m, wc, -140.0 / m ** 2)["scatter_wl"] < 1e-6


def test_enforce_invariant_leaves_a_consistent_night_alone():
    orders = _real_orders()
    before = [o.wl.copy() for o in orders]
    assert grating.enforce_invariant(orders) == 0
    assert all(np.array_equal(o.wl, b) for o, b in zip(orders, before))
    assert all(o.dispersion_ok for o in orders)


def test_enforce_invariant_restores_a_stray_zero_point():
    """The one thing the module is for:"""
    orders = _real_orders()
    stray = 20
    truth = orders[stray].wl.copy()
    orders[stray].wl = orders[stray].wl + 3.0

    assert grating.enforce_invariant(orders) == 1
    err = float(np.max(np.abs(orders[stray].wl - truth)))
    assert err < 1.0, "the stray is still %.2f A out after the refit" % err


def test_enforce_invariant_touches_only_the_stray():
    """The regression that mattered:"""
    orders = _real_orders()
    orders[10].wl = orders[10].wl * 30.0
    before = {o.id: o.wl.copy() for o in orders}

    grating.enforce_invariant(orders)

    rewritten = [o.id for o in orders if not np.array_equal(o.wl, before[o.id])]
    assert rewritten == [], "healthy orders rewritten: %s" % rewritten


def test_enforce_invariant_drops_a_diverged_order_instead_of_replacing_its_scale():
    """Its centre is not one order's spacing from anything, so there is no
    telling which order to restore it to."""
    orders = _real_orders()
    orders[10].wl = orders[10].wl * 30.0
    before = orders[10].wl.copy()

    grating.enforce_invariant(orders)

    assert not orders[10].dispersion_ok, "a diverged order was left usable"
    assert np.array_equal(orders[10].wl, before), \
        "a diverged order was refit to a guessed position"
    assert all(o.dispersion_ok for o in orders if o.id != 10)


def test_enforce_invariant_flags_a_non_monotonic_order():
    """A relation that folds inside the detector has the wrong shape, not the
    wrong scale, so no rescaling repairs it."""
    orders = _real_orders()
    folded = 30
    x = np.arange(len(orders[folded].wl), dtype=float)
    disp = _DISPERSIONS[folded]
    period = len(x) / 8.0
    wobble = 2.0 * abs(disp) * period / (2 * np.pi)
    orders[folded].wl = orders[folded].wl + wobble * np.sin(2 * np.pi * x / period)

    grating.enforce_invariant(orders)

    assert not orders[folded].dispersion_ok, "a folded order was left usable"
    assert all(o.dispersion_ok for o in orders if o.id != folded)


def _run_verbose(orders, **kw):
    """`enforce_invariant`'s report, which is all the user sees of it."""
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        n = grating.enforce_invariant(orders, verbose=True, **kw)
    return n, buf.getvalue()


def test_enforce_invariant_reports_a_consistent_night():
    n, out = _run_verbose(_real_orders())
    assert n == 0
    assert "all 53 orders consistent" in out, out
    assert "m*lambda = 3559" in out, out


def test_enforce_invariant_reports_the_order_it_corrected():
    orders = _real_orders()
    orders[20].wl = orders[20].wl + 3.0
    n, out = _run_verbose(orders)
    assert n == 1
    assert "order 20 off the grating relation" in out, out
    assert "scale restored" in out, out
    assert "consistent" not in out, "it claimed the night was clean anyway"


def test_enforce_invariant_reports_an_order_it_could_not_place():
    orders = _real_orders()
    orders[10].wl = orders[10].wl * 30.0
    n, out = _run_verbose(orders)
    assert "order 10 cannot be placed" in out, out
    assert "consistent" not in out, \
        "an unplaceable order must not be reported as a clean night"


def test_enforce_invariant_reports_a_non_monotonic_order():
    orders = _real_orders()
    x = np.arange(len(orders[30].wl), dtype=float)
    period = len(x) / 8.0
    wobble = 2.0 * abs(_DISPERSIONS[30]) * period / (2 * np.pi)
    orders[30].wl = orders[30].wl + wobble * np.sin(2 * np.pi * x / period)
    n, out = _run_verbose(orders)
    assert "order 30 dispersion is not monotonic" in out, out


def test_order_numbers_ignores_a_valid_mask_that_leaves_too_little():
    """A mask that excludes almost everything cannot set the scale, so it is
    dropped rather than obeyed into a meaningless answer."""
    valid = np.zeros(len(_CENTRES), bool)
    valid[:2] = True
    m = grating.order_numbers(_CENTRES, valid=valid)
    assert np.array_equal(m, grating.order_numbers(_CENTRES))


def test_enforce_invariant_does_nothing_with_too_few_orders():
    """Below `min_orders` there is nothing to measure the relation from, and a
    handful of orders must not be talked into agreeing with each other."""
    orders = _real_orders()[:6]
    before = [o.wl.copy() for o in orders]
    assert grating.enforce_invariant(orders) == 0
    assert all(np.array_equal(o.wl, b) for o, b in zip(orders, before))


def test_enforce_invariant_is_not_confused_by_a_gap_in_the_order_list():
    """Unsolved orders never reach here, so holes are the normal case."""
    orders = _real_orders(skip=(0, 17, 18, 40))
    before = [o.wl.copy() for o in orders]
    assert grating.enforce_invariant(orders) == 0
    assert all(np.array_equal(o.wl, b) for o, b in zip(orders, before))
    assert all(o.dispersion_ok for o in orders)


def _synthetic_arc(line_px, npix=2048, sigma=1.6):
    """An arc with a Gaussian line at each position, on a low pedestal."""
    x = np.arange(npix) + 1.0
    arc = np.full(npix, 0.02)
    for c in line_px:
        arc += np.exp(-0.5 * ((x - c) / sigma) ** 2)
    return arc


def test_idcomp_set_is_chosen_by_the_night_and_not_by_the_default():
    assert select_idcomp_dir("2024-09-01").endswith("idcomp_2307")
    assert select_idcomp_dir("2025-09-03").endswith("idcomp_2307")
    assert select_idcomp_dir("2026-08-29").endswith("idcomp_2026")
    # a full timestamp works as well as a bare date
    assert select_idcomp_dir("2026-08-29T22:19:09").endswith("idcomp_2026")
    # and the boundary is where it says it is
    assert select_idcomp_dir("2025-12-31").endswith("idcomp_2307")
    assert select_idcomp_dir("2026-01-01").endswith("idcomp_2026")


def test_dispersion_shift_recovers_a_drift_larger_than_the_fit_window():
    rng = np.random.default_rng(0)
    true_px = np.sort(rng.uniform(60, 1990, 22))
    arc = _synthetic_arc(true_px)
    seeds = true_px - 14.0
    shift, residual, quality = solve_dispersion_shift(seeds, arc)
    assert abs(shift - 14.0) < 0.3
    assert residual < 0.5
    assert quality > 2


def test_dispersion_shift_is_zero_when_the_seeds_already_match():
    rng = np.random.default_rng(1)
    true_px = np.sort(rng.uniform(60, 1990, 22))
    arc = _synthetic_arc(true_px)
    shift, residual, _ = solve_dispersion_shift(true_px, arc)
    assert abs(shift) < 0.2
    assert residual < 0.5


def test_dispersion_shift_gives_up_rather_than_guessing():
    # too few seeds to place anything: return no shift, not a random one
    arc = _synthetic_arc([100.0, 400.0, 900.0])
    shift, residual, _ = solve_dispersion_shift(np.array([90.0, 390.0]), arc)
    assert shift == 0.0
    assert np.isnan(residual)


class _FakeOrder:
    def __init__(self, comparison, pixel_y_cen):
        self.comparison = comparison
        self.pixel_y_cen = pixel_y_cen
        self.id = int(pixel_y_cen)


def test_dispersion_shifts_replace_an_order_that_disagrees_with_the_detector():
    rng = np.random.default_rng(2)
    orders, linelists, avg_aps, pairs = [], {}, [], []
    for i in range(8):
        true_px = np.sort(rng.uniform(60, 1990, 20))
        drift = 10.0 + 0.5 * i                     # a smooth trend
        arc = _synthetic_arc(true_px)
        orders.append(_FakeOrder(arc, 800.0 + 15.0 * i))
        table = np.zeros((len(true_px), 6))
        table[:, 0] = true_px - drift
        avg_aps.append(800.0 + 15.0 * i)
        linelists[avg_aps[i]] = table
        pairs.append((i, i))
    shifts = solve_dispersion_shifts(pairs, orders, linelists,
                                     np.array(avg_aps))
    assert len(shifts) == 8
    for i in range(8):
        assert abs(shifts[i] - (10.0 + 0.5 * i)) < 0.4


def test_dispersion_shifts_returns_nothing_when_no_order_is_measurable():
    orders = [_FakeOrder(np.full(2048, 0.02), 900.0)]
    table = np.zeros((3, 6))
    table[:, 0] = [100.0, 400.0, 900.0]
    shifts = solve_dispersion_shifts([(0, 0)], orders, {900.0: table},
                                     np.array([900.0]))
    assert shifts == {}


# --------------------------------------------------------------------------

def _main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:
            failed += 1
            print("FAIL %s: %s" % (name, exc))
        else:
            print("ok   %s" % name)
    print("\n%d/%d passed" % (len(tests) - failed, len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
