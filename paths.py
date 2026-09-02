"""Locations of the data files shipped with the pipeline."""
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# IRAF-style ``idcomp`` line identifications used to seed the wavelength solution
DEFAULT_IDCOMP_DIR = os.path.join(PACKAGE_DIR, "idcomp_2307")

# cleaned ThAr line list from 2007A&A...468.1115L
DEFAULT_THAR_LIST = os.path.join(PACKAGE_DIR, "thar_lovis_pepe_clean.csv")

# reaches 10506 A; Lovis & Pepe stops at 6912 A and leaves the reddest orders bare
MURPHY_THAR_LIST = os.path.join(PACKAGE_DIR,
                                "Murphy2007_mnras0378-0221-SD1.txt")

# example night bundled with the repository
DEFAULT_DATA_DIR = os.path.join(PACKAGE_DIR, "20240901")

# Kurucz model of Vega: wavelength (nm, vacuum), flux, continuum flux
DEFAULT_TEMPLATE = os.path.join(PACKAGE_DIR, "vegallpr25.20000resam13")


# (first date the set is valid for, directory); the last matching row wins
IDCOMP_SETS = [
    ("2000-01-01", os.path.join(PACKAGE_DIR, "idcomp_2307")),
    ("2026-01-01", os.path.join(PACKAGE_DIR, "idcomp_2026")),
]


def select_idcomp_dir(date_obs, sets=None):
    """Pick the idcomp set that describes the spectrograph on ``date_obs``."""
    if sets is None:
        sets = IDCOMP_SETS
    day = str(date_obs).strip()[:10]
    chosen = sets[0][1]
    for start, path in sets:
        if day >= start:
            chosen = path
    return chosen
