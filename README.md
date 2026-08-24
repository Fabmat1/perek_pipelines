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
