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

Each order has to be matched to a reference line list (`idcomp_2307` by
default, but see below) that gives it its wavelength solution. Any directory
of IRAF `identify` records will do; the filenames are not read, so the
observatory's own naming works as it arrives. The match is made by position on the
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

## Which line list to use, and why it depends on the year

Two reference lists are bundled:

```
idcomp_2307   53 orders, 1615 lines, 3811-8772 A   taken 2023-07
idcomp_2026   50 orders,  881 lines, 3967-8962 A   taken 2026-08
```

They are not interchangeable, and the difference is much larger than the
line counts suggest. Between September 2025 and August 2026 the order block
did not just move across the detector, it changed scale: matched order for
order, the two lists differ by 1 px at the blue end and 13 px at the red one,
which no single `--idcomp-offset` can absorb.

What that costs was measured by scoring the wavelength solution against the
full Lovis & Pepe + Murphy ThAr atlas -- thousands of lines, against the
fourteen to thirty a list seeds an order with -- on one frame from each night
we hold raw data for. Per order, the median offset of the arc lines from their
catalogue wavelengths:

```
                    idcomp_2307        idcomp_2026
20240901             0.0013 A           0.074 A
20250903             0.0019 A           0.121 A
20260829             0.081  A           0.0014 A
```

So each list is right for its own epoch and wrong by fifty to a hundred times
that for the other, which in velocity is the difference between 0.3 km/s and
10 km/s of scatter within an order. Nothing in the run warns about it: the
wrong list still identifies every order, still passes the grating-relation
check, and reports an idcomp residual of 0.16 px against 0.07 px for the right
one -- comfortably inside the threshold that would complain.

`--thar-list` does not rescue it. Refining against half the atlas and scoring
against the half held out, the 2023 list on the 2026 night goes from 0.081 A to
0.059 A: the seeds are wrong by enough that the refinement locks onto the wrong
catalogue line about as often as the right one.

**Use `--idcomp-dir idcomp_2026` for data from 2026 onwards.** The default
stays `idcomp_2307` because it is what the bundled example night needs, and
because it is the right list for everything in the archive up to at least
September 2025. Exactly when the spectrograph moved is not pinned down -- we
have no raw frames between 20250903 and 20260826 -- so a night from that gap
is worth reducing both ways before it is trusted.

The 2026 list is also 150 A shorter in the blue: it starts at 3963 A against
3811 A, which is four orders, and takes Ca II K (3934 A), H8 and H9 with it. It
gains one order in the red instead, out to 8966 A. If the blue matters, the
list wants extending blueward at the telescope; seeding those orders from the
2023 list is not an option, because their apertures sit 25 px away from where
the 2026 list puts the rest of the block.

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
(2007) atlas reaches 10506 A. It always takes a value -- `lovis`, `murphy`,
`both`, or a comma-separated list of your own files -- because an optional
argument would be ambiguous against the positional data directory. Without the
flag the refinement is off.

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

## Worker processes

Every pool uses one worker per core. `--ncpu N` turns that down, which is worth
doing on a shared machine; it has no effect unless you pass it. `PEREK_NCPU`
does the same through the environment.

## Uncertainties

The pipeline propagates photon noise from the raw frame, reading `GAIN`
(e-/ADU) and `READNOIS` (e-) out of the frame header rather than assuming
them, and combines the aperture pixels with the square of the extraction
weights -- the extracted value is a weighted mean, not a sum. The scattered
light is counted before it is subtracted, because its shot noise stays in the
spectrum after the halo has gone.

On a bright star the result is not what sets the error bars. The science is
divided by a *median-filtered* flat, so whatever structure the flat has on
scales shorter than the filter is never corrected, and that residual dominates
the photon term about thirty to one. Measured against the scatter of a
line-free continuum on alp Cyg the whole budget still comes out roughly a
factor two small, so the relative weighting is sound but the absolute scale
wants calibrating on a clean-continuum star before it is quoted.

## Tests

```
python tests/test_pipeline.py     # or: pytest tests/
```

Synthetic only -- no frames are read -- so it runs in under a second.
