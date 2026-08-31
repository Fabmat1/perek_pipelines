# Perek echelle reduction pipeline

Data reduction for echelle spectra from the Perek 2m telescope.

## Installation

The pipeline is pure Python and runs on Linux, macOS and Windows.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) once:

```
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then:

```
git clone <repository-url>
cd perek_pipelines
uv sync
```

`uv sync` downloads a suitable Python and all dependencies into a local
`.venv`. You do not need to install Python yourself or activate anything.

## Running

Reduce the example night bundled with the repository:

```
uv run python template.py
```

Reduce your own data:

```
uv run python template.py /path/to/20250903
uv run python template.py 20250903 --science e202509030022 --plot
```

## Order identification

Each order has to be matched to the reference line list (`idcomp_2307`) that
gives it its wavelength solution. The match is made by position on the
detector, so it depends on how far the spectrograph has moved in the
cross-dispersion direction since the reference was taken. That shift is
measured from the data for every night and reported as it runs:

```
- idcomp offset = -26.03 px (residual 0.14 px, 22.6x better than next candidate)
```

The residual is how closely the orders land on the reference apertures, and
the ratio says how much better this alignment is than the next best one. A
large residual, or a ratio near 1, means the identification is not trustworthy
and the pipeline says so. This matters because the order spacing is only about
15 px: if the shift is wrong by one order, every order gets its neighbour's
line list and the reduction still finishes, but the wavelengths are wrong.

To see how the shift was chosen, run with `--debug-plots`. The figure shows
the scan over candidate shifts (the correct one sits in a deep, narrow well)
and how the orders line up under it, next to the off-by-one alignments that
were rejected. Those alternatives agree near the middle of the detector and
fan apart towards the edges, which is what makes the correct shift
identifiable at all.

Pass `--idcomp-offset <px>` to force a particular shift instead of measuring
it.

## Which frame the orders are traced on

The orders have to be traced on something before they can be identified, and
that choice decides how far into the blue the reduction reaches. The flat lamp
has almost no output there -- on the 2026 detector it peaks near 20 counts in
the bluest orders against 20000 in the red -- so tracing on the flat alone
loses them. Science frames have the signal, but only some of them: a red star
through a long exposure can be the brightest frame of the night and still leave
the bluest orders invisible.

By default the pipeline measures this instead of guessing. Every science frame
is scored by how far its bluest orders rise above the noise, the orders are
traced on a median stack of the best few, and the choice is reported:

```
- tracing orders on 4 frame(s):
    bluest: e202608260044.fit    blue    187 sigma, red     43 sigma
            e202608260036.fit    blue      8 sigma, red     29 sigma
```

The first frame is the one with the most blue signal and sets the blue cutoff;
the rest are chosen for the red end, because the bluest-strong frames are hot
blue stars that are faint in the reddest orders. On the 2026 night this reaches
3829 A, against 4009 A when tracing on the flat.

Use `--trace-stack N` to change how many frames are stacked, or
`--frame-for-slice` to override the choice entirely: `flat` for the flat alone,
`science` for all science frames, or a path to one FITS file. If no science
frame can be scored -- a calibration-only directory, say -- the pipeline falls
back to the flat.

## Sky background

The OES cannot measure the sky, which is dispersed into the orders along with
the star and so is never subtracted. It is negligible for a bright target, but
in the faintest blue orders it adds a floor that fills in absorption lines, so
their depths are unreliable below a few tens of counts.

## ThAr line lists for the red orders

`--thar-list` refines the wavelength solution against a ThAr atlas after the
`idcomp` lists have seeded it. The bundled Lovis & Pepe (2007) list stops at
6912 A, so the reddest orders have nothing to refine against; the Murphy et al.
(2007) atlas reaches 10506 A. Pass `lovis`, `murphy`, `both`, or a
comma-separated list of your own files.

What this buys, measured on 20260826 in the orders past 7300 A, is coverage
across the detector rather than a smaller residual:

```
                lines/order   fraction of the order extrapolated
without ThAr        6-12                  5-18%
with Murphy        25-43                  2-7%
```

The fit residual goes *up* (0.007 -> 0.15 A in the best case), and so does the
reported resolution. Neither means the solution got worse: a cubic through
seven points sits close to those seven points and then extrapolates blind over
the rest of the order, and R is measured from the widths of whatever lines were
used, so a sample of six bright narrow ones flatters the instrument. The
merged spectra from the two settings agree to 0.04-0.05 A in the red, well
inside one resolution element (0.27 A at 7600 A).

Orders that fail to solve are still dropped rather than guessed at. Seeding
them from their solved neighbours was tried and removed: the grating relation
predicts an unsolved order's wavelengths to a few Angstrom, but the order
spacing at the red end is only ~10 A, so the prediction can be a whole order
out. Checking the refit against the arc does not settle it either -- a ThAr arc
has ~150 lines across 2048 pixels, so a wrong-by-one-order solution still puts
catalogue lines on top of real peaks about as often as the right one does. On
four test orders that test picked the wrong order twice.

Reduced spectra are written to `done/` as both FITS tables and plain text
(`wavelength  flux  error  resolution`). See all options with:

```
uv run python template.py --help
```

If you prefer a plain virtual environment over uv:

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python template.py
```

On Windows use `.venv\Scripts\pip` and `.venv\Scripts\python` instead.

## Optional: faster resampling

Resampling uses the `spectres` package by default. A Fortran implementation is
roughly an order of magnitude faster; it is optional and requires a Fortran
compiler (`gfortran`). To build it in the repository root:

```
uv run python -m numpy.f2py -c -m pyresample_spectres resample_spectres.f90
```

It is picked up automatically once built. Both backends produce the same
results. `template.py` reports which one is in use.
