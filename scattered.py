"""Remove the scattered light that fills the gaps between the orders.

The grating throws a diffuse halo across the detector. In the flat it reads
~10 counts where no order falls, rising to ~30 just below the reddest orders
and decaying outwards with a ~170 px scale length. It is scattered light, not
dark current: it points away from the brightest orders, and the 25 s flat
carries more of it than a 300 s science exposure.

It is 2-3% of a red order peak and can be ignored there, but in the blue the
lamp is so faint that the halo is comparable to the signal -- the bluest orders
peak at 13 counts over a floor of 10. Flat division then divides by mostly
halo, leaving the blaze in and the order ends drooping against their
neighbours.

The halo is measured between the orders, so it must be subtracted before
extraction collapses those pixels away. Below ~4000 A this rescues nothing --
the flat's order edges hold no lamp signal to recover -- but it fixes the
region just above, where real signal is buried under the halo.
"""
import numpy as np
from scipy.ndimage import median_filter


def _order_centres(orders, column, nrow, min_separation=3.0):
    """Order centres in one column, sorted, near-duplicates dropped.

    Two traces that have converged would otherwise contribute a "midpoint"
    lying on an order, which would then be read as background.
    """
    y = []
    for o in orders:
        if o.solution is None:
            continue
        v = float(np.polyval(np.asarray(o.solution), column)) \
            if not callable(getattr(o, "evaluate", None)) else float(o.evaluate(column))
        if 0 <= v < nrow:
            y.append(v)
    if not y:
        return np.empty(0)
    y = np.sort(np.asarray(y, float))
    keep = [y[0]]
    for v in y[1:]:
        if v - keep[-1] > min_separation:
            keep.append(v)
    return np.asarray(keep)


def background(image, orders, colstep=32, halfwin=3, pad=3, tail=4,
               smooth_columns=7, quantile=25.0):
    """Model the scattered-light halo under `image`.

    Sampled between adjacent orders, at the 25th percentile of each gap: the
    mean sits too high because two bright neighbours' wings meet in the middle,
    but the outright minimum is biased low by ~1.35 sigma at any level (~30
    counts under the red orders, and negative on a frame whose background is
    genuinely near zero).

    Beyond the outermost orders it is sampled at `tail` increasing distances to
    follow the decay; holding it flat at the last value leaves tens of counts
    of error under the reddest orders.

    Every `colstep`-th column is measured and the rest interpolated -- the halo
    is a shallow dome (~6 counts at the detector edge against ~11 in the
    middle), so sampling every column would only add noise. Profiles are
    median-filtered across columns to reject cosmics landing in a gap.
    """
    image = np.asarray(image, dtype=float)
    nrow, ncol = image.shape

    columns = np.arange(0, ncol, colstep)
    if columns[-1] != ncol - 1:
        columns = np.append(columns, ncol - 1)

    rows = np.arange(nrow)
    profiles = np.empty((len(columns), nrow))

    for j, c in enumerate(columns):
        centres = _order_centres(orders, c, nrow)
        if len(centres) < 2:
            profiles[j] = np.median(image[:, c])
            continue

        spacing = float(np.median(np.diff(centres)))
        samples = list(0.5 * (centres[:-1] + centres[1:]))
        for k in range(1, tail + 1):
            below = centres[0] - pad * spacing * k
            above = centres[-1] + pad * spacing * k
            if below > 2:
                samples.append(below)
            if above < nrow - 3:
                samples.append(above)
        # anchor both detector edges so the interpolation does not run flat
        samples = np.concatenate(([2.0], np.array(sorted(samples)),
                                  [nrow - 3.0]))

        values = np.empty(len(samples))
        for k, m in enumerate(samples):
            m = int(round(m))
            lo = max(0, m - halfwin)
            hi = min(nrow, m + halfwin + 1)
            values[k] = np.percentile(image[lo:hi, c], quantile)

        profiles[j] = np.interp(rows, samples, values)

    if smooth_columns > 1 and len(columns) >= smooth_columns:
        profiles = median_filter(profiles, size=(smooth_columns, 1),
                                 mode="nearest")

    bg = np.empty((nrow, ncol))
    for r in range(nrow):
        bg[r] = np.interp(np.arange(ncol), columns, profiles[:, r])

    # a floor may not exceed the frame, or subtraction invents negative flux
    return np.minimum(bg, image)


def remove_background(image, orders, verbose=False, label="", **kwargs):
    """`image` with its scattered-light halo subtracted."""
    image = np.asarray(image, dtype=float)
    bg = background(image, orders, **kwargs)
    if verbose:
        print("- scattered light removed from %s: %.1f counts median between "
              "the orders (%.1f max)" % (label or "frame", np.median(bg),
                                         np.max(bg)))
    return image - bg
