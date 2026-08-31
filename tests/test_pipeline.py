"""Regression tests for the reduction pipeline.

Runs under pytest if it is installed, and as a plain script if it is not:

    python tests/test_pipeline.py

Everything here is synthetic -- no frames are read -- so the whole file runs in
well under a second and can be run before every commit.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from echelle_reduction import (normalize_single_order, _fit_continuum,
                               get_detector_noise, red_edge_keep)
from identify_orders import trace_windows, FALLBACK_WINDOWS
from orders import SpectralOrder, gaussian_pixel_weights_2d
from calibrate import fit_dispersion, parse_idcomp
from tools import shared, publish_shared, polynomial
from template import output_stem


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Fix 1: an ignore window on an order's end must not delete the order's tail
# --------------------------------------------------------------------------

def test_ignore_window_at_order_end_keeps_pixels_without_a_neighbour():
    """The O2 A-band case: window (7590, 7660) covers the red end of an order
    running to 7631, and no other order reaches there. Those 33 A were being
    dropped outright, leaving a hole in the merged spectrum."""
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
    """H-alpha sits in an ignore window in the middle of its order. Masking it
    from the continuum fit must not remove it from the spectrum."""
    wl = np.linspace(6532.0, 6642.0, 2000)
    flx = np.ones_like(wl) * 100.0
    flx[(wl > 6555) & (wl < 6570)] *= 0.4

    _, _, _, keep, _ = normalize_single_order(
        _norm_args(wl, flx, [(6540.0, 6590.0)]))

    halpha = (wl > 6540.0) & (wl < 6590.0)
    assert keep[halpha].all(), "H-alpha was deleted from its own order"


# --------------------------------------------------------------------------
# Fix 2: photon noise
# --------------------------------------------------------------------------

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
    """The extracted value is a weighted mean, not a sum. Treating it as a raw
    pixel count -- as `var = |science| + ...` did -- overstates the noise by
    1/sum(w^2), which for this aperture is several-fold."""
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


# --------------------------------------------------------------------------
# the sorted-mask fix in fit_dispersion
# --------------------------------------------------------------------------

def test_fit_dispersion_mask_lines_up_with_unsorted_input():
    """`fit_dispersion` sorts internally; the mask it returns has to index the
    caller's unsorted arrays. The ThAr refinement appends predicted lines with
    vstack, so the input really is unsorted by then."""
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


# --------------------------------------------------------------------------
# trace-frame window measurement
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# smaller guards
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# output filenames
# --------------------------------------------------------------------------

def test_output_stem_strips_glob_characters_from_the_object_name():
    # "* psi cyg" is a real OBJECT value; an asterisk in the filename is a
    # wildcard to every shell and tool that reads the spectra back
    stem = output_stem("e202608290033.fit", "*_psi_cyg")
    assert "*" not in stem
    assert stem == "e202608290033_psi_cyg"


def test_output_stem_keeps_the_designation_of_a_variable_star():
    assert output_stem("e202608270038.fit", "V*_BP_Boo") == "e202608270038_V_BP_Boo"


def test_output_stem_leaves_ordinary_names_alone():
    assert output_stem("e202409010033.fit", "BD+26_2766") == "e202409010033_BD+26_2766"


def test_output_stem_only_drops_the_trailing_extension():
    # ".fit" also appears inside the name here
    assert output_stem("fit.fit", "HD_1") == "fit_HD_1"


# --------------------------------------------------------------------------
# idcomp line lists
# --------------------------------------------------------------------------

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
    # IRAF appends a record per save; the 2026 lists ship with four or five of
    # them, and concatenating them would feed the fit the same line repeatedly
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
