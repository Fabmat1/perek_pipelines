"""Selects the spectral resampling backend.

The compiled Fortran extension ``pyresample_spectres`` is used when it is
available, since it is considerably faster. It is optional: building it needs a
Fortran compiler, so the pure-Python ``spectres`` package is a hard dependency
and is used otherwise.

Both backends are exposed through the same signature, with the same ``fill``
default, so that results do not depend on which one is installed.
"""

try:
    from resample_spectres import resample
    BACKEND = "pyresample_spectres (Fortran)"
except ModuleNotFoundError:
    from spectres import spectres as _spectres
    BACKEND = "spectres (pure Python)"

    def resample(wave_out, wave_in, flux_in, fill=0.0, verbose=False):
        # spectres defaults to fill=None, which yields NaN outside the input
        # range; the Fortran backend fills with 0.0. Keep them identical.
        return _spectres(wave_out, wave_in, flux_in, fill=fill, verbose=verbose)
