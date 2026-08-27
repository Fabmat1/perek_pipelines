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
