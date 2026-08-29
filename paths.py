"""Locations of the data files shipped with the pipeline.

These are resolved relative to this file rather than the current working
directory, so the pipeline can be run from anywhere.
"""
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# IRAF-style ``idcomp`` line identifications used to seed the wavelength solution
DEFAULT_IDCOMP_DIR = os.path.join(PACKAGE_DIR, "idcomp_2307")

# cleaned ThAr line list from 2007A&A...468.1115L
DEFAULT_THAR_LIST = os.path.join(PACKAGE_DIR, "thar_lovis_pepe_clean.csv")

# ThAr atlas from 2007MNRAS.378..221M. Lovis & Pepe stops at 6912 A, which
# leaves the reddest OES orders with no lines at all; Murphy reaches 10506 A.
MURPHY_THAR_LIST = os.path.join(PACKAGE_DIR,
                                "Murphy2007_mnras0378-0221-SD1.txt")

# example night bundled with the repository
DEFAULT_DATA_DIR = os.path.join(PACKAGE_DIR, "20240901")
