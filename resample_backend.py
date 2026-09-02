"""Selects the spectral resampling backend."""

import numpy as np

try:
    from resample_spectres import resample
    BACKEND = "pyresample_spectres (Fortran)"
except ModuleNotFoundError:
    from spectres import spectres as _spectres
    BACKEND = "spectres (pure Python)"

    def resample(wave_out, wave_in, flux_in, fill=0.0, verbose=False):
        if len(wave_out) == 0 or len(wave_in) == 0:
            return np.zeros(len(wave_out), dtype=float)
        return _spectres(wave_out, wave_in, flux_in, fill=fill, verbose=verbose)
