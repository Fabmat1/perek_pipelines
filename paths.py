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

# example night bundled with the repository
DEFAULT_DATA_DIR = os.path.join(PACKAGE_DIR, "20240901")
