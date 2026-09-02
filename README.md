# Perek echelle reduction pipeline

Data reduction for echelle spectra from the Perek 2m telescope.

## Install

```
git clone <repository-url>
cd perek_pipelines
uv sync
```

`uv sync` fetches Python and all dependencies into `.venv`. Without uv:

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

On Windows use `.venv\Scripts\pip` and `.venv\Scripts\python`.

## Run

```
uv run python template.py                          # bundled example night
uv run python template.py /path/to/20250903        # one night
uv run python template.py 20250903 --science e202509030022 --plot
```

Input is a directory of `.fit` frames; the `OBJECT` keyword sorts them into
`zero`, `flat`, `comp` and science. Output goes to `done/` as FITS tables and
text (`wavelength  flux  error  resolution`).

`uv run python template.py --help` lists all options. The ones that matter:

| option | effect |
| --- | --- |
| `-o, --outdir DIR` | where to write (default `done`) |
| `-s, --science NAME` | only frames whose filename contains NAME |
| `--idcomp-dir DIR` | line-list set; `auto` picks it by date |
| `--idcomp-offset PX` | force the cross-dispersion shift |
| `--frame-for-slice` | `auto`, `flat`, `science`, or a FITS path |
| `--trace-stack N` | frames stacked to trace on (default 4) |
| `--thar-list LIST` | refine against `lovis`, `murphy`, `both`, or paths |
| `--ncpu N` | worker processes (default: every core) |
| `--no-normalize` | skip continuum normalisation |
| `--debug-plots` | diagnostic plots from every step |

## Line lists

Two IRAF `identify` reference sets are bundled. Filenames are not read, so any
directory of `identify` records works.

```
idcomp_2307   53 orders, 1615 lines, 3811-8772 A   taken 2023-07
idcomp_2026   50 orders,  881 lines, 3967-8962 A   taken 2026-08
```

They are not interchangeable: the wrong one shifts the whole wavelength scale
by tens of km/s without failing or warning. `--idcomp-dir auto` (the default)
picks by the night's `DATE-OBS` and reports its choice. Epoch boundaries live
in `IDCOMP_SETS` in `paths.py`.

The run measures two shifts against the chosen set and prints both:

```
- idcomp offset = -26.03 px (residual 0.14 px, 22.6x better than next candidate)
- dispersion shift = -14.4 px (range -17.8..-12.8 over 53 orders, 1 replaced by the trend)
```

The first is across the orders and decides which list each order gets; a
residual near the order spacing (~15 px) or a ratio near 1 means the
identification is untrustworthy and the pipeline says so. The second is along
the orders and is removed before fitting. Both are ~0 when the set matches the
epoch. `--debug-plots` shows how the first was chosen.

## Optional: faster resampling

```
uv run python -m numpy.f2py -c -m pyresample_spectres resample_spectres.f90
```

Needs `gfortran`. Picked up automatically once built, ~10x faster than
`spectres`, same results. `template.py` reports which backend is in use.

## Tests

```
pytest tests/          # or: python tests/test_pipeline.py
```

Synthetic only, runs in under a second.

## Caveats

- **Sky is not subtracted.** The OES cannot measure it. Negligible on bright
  targets; in the faintest blue orders it fills in absorption lines, so their
  depths are unreliable below a few tens of counts.
- **Error bars are a factor ~2 small.** Photon noise is propagated from the raw
  frame using `GAIN` and `READNOIS`, but division by a median-filtered flat
  leaves residual structure that dominates it ~30:1. Relative weighting is
  sound; the absolute scale wants calibrating on a clean-continuum star.
- **`idcomp_2026` starts at 3963 A**, 150 A redder than `idcomp_2307`, losing
  four orders including Ca II K, H8 and H9. It gains one order out to 8966 A.
  Seeding the blue from the 2023 list does not work: those apertures sit 25 px
  from where the 2026 list puts the rest of the block.
- **`--thar-list` buys coverage, not precision.** Past 7300 A it raises
  lines/order from 6-12 to 25-43 and cuts the extrapolated fraction from
  5-18% to 2-7%. The residual and reported R both go up; neither means the
  solution got worse.
- **Orders that fail to solve are dropped**, not interpolated from neighbours:
  at the red end the order spacing is ~10 A, so the grating relation can be a
  whole order out, and a ThAr arc is dense enough that a wrong-by-one solution
  still lands on real peaks.
- **The 2025/2026 boundary is not pinned down** — no raw frames exist between
  20250903 and 20260826. Reduce a night from that gap both ways before
  trusting it.
