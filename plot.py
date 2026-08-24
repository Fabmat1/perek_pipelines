import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

def group_close_stars_with_bjd(fps, max_sep_arcsec=30):
    """
    Read RA/Dec and BJD from a list of FITS files and group stars closer than max_sep_arcsec.
    
    Parameters
    ----------
    fps : list of str
        List of FITS file paths.
    max_sep_arcsec : float
        Maximum separation in arcseconds to consider stars as a group.
    
    Returns
    -------
    groups : list of dict
        Each dict has keys:
            'files' : list of FITS file paths in the group
            'bjds'  : list of BJD values corresponding to the files
    coords : SkyCoord
        SkyCoord object of all stars.
    """
    coords_list = []
    files_list = []
    bjds_list = []

    # Collect RA/Dec and BJD for all FITS files
    for fp in fps:
        if "fits" in fp:
            header, bjd = read_fits_header(fp)
            if "RA" in header and "DEC" in header:
                ra_deg = header["RA"]
                dec_deg = header["DEC"]
                coords_list.append((ra_deg, dec_deg))
                files_list.append(fp)
                bjds_list.append(bjd)

    if not coords_list:
        return [], None

    # Convert to SkyCoord array
    coords = SkyCoord(
        ra=[c[0] for c in coords_list]*u.deg,
        dec=[c[1] for c in coords_list]*u.deg
    )

    # Search for pairs within max_sep
    max_sep = max_sep_arcsec * u.arcsec
    idx1, idx2, _, _ = coords.search_around_sky(coords, max_sep)

    # Group stars (avoid self-matches and duplicates)
    groups = []
    used = set()
    for i1, i2 in zip(idx1, idx2):
        if i1 == i2:
            continue  # skip self
        if i1 in used or i2 in used:
            continue
        group_files = [files_list[i1], files_list[i2]]
        group_bjds = [bjds_list[i1], bjds_list[i2]]
        groups.append({'files': group_files, 'bjds': group_bjds})
        used.update([i1, i2])
    groups = [pd.DataFrame(g) for g in groups]

    return groups, coords

def read_fits_header(filepath):
    """Read FITS header and convert RA/DEC to decimal degrees if present."""
    with fits.open(filepath) as hdul:
        header = hdul[0].header
        ra = header.get('RA', None)
        dec = header.get('DEC', None)

        if ra is not None and dec is not None:
            # Convert RA/DEC to decimal degrees
            coord = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg))
            header["RA"] = np.float64(coord.ra.deg)
            header["DEC"] = np.float64(coord.dec.deg)
        else:
            print("RA or DEC not found in header of %s." % filepath)

        bjd = header.get('BJD', None)
        return header, bjd

def read_fits_data(filepath):
    """Read FITS 'SCIENCE' table data and convert to pandas DataFrame."""
    with fits.open(filepath) as hdul:
        table_data = hdul['SCIENCE'].data
        data_dict = {}

        for col in table_data.columns:
            col_data = np.asarray(table_data[col.name])
            # FITS tables are big-endian; pandas wants native byte order
            if col_data.dtype.byteorder not in ("=", "|"):
                col_data = col_data.astype(col_data.dtype.newbyteorder("="))
            data_dict[col.name] = col_data

        df = pd.DataFrame(data_dict)
        return df

def read_spectrum_file(fp):
    df = pd.read_csv(fp, sep="\\s+", header=None)
    wave = df.iloc[:,0]
    flux = df.iloc[:,1]
    err = df.iloc[:,2]
    return wave, flux, err

def plot_spectrum(wave, flux, err, title):
    plt.title(title)
    plt.plot(wave, flux, color="black")
    plt.plot(wave, err, color="tab:gray")
    plt.xlabel("Wavelength / Angstrom")
    plt.ylabel("Normalised flux")
    ymin, ymax = plt.gca().get_ylim()
    plt.gca().set_ylim(max(0,ymin), max(ymax, 1.2))
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", help="File pattern or list of files")
    parser.add_argument("--no-plots", action="store_true", help="Disable plotting")
    args = parser.parse_args()

    fps = args.files
    show_plots = not args.no_plots
    if "*" in fps:
        fps = list(glob(fps))
    else:
        fps = [fps]
    print(fps)

#    groups, coords = group_close_stars_with_bjd(fps, max_sep_arcsec=30)
#    for g in groups: print(g)

    for fp in fps:
        if "fits" in fp:
            header, bjd = read_fits_header(fp)
            df = read_fits_data(fp)
            wave = df["wave"]
            flux = df["flux"]
            err = df["error"]
        else:
            wave, flux, err = read_spectrum_file(fp)
        snr = flux/err
        median_snr = np.nanmedian(snr)
        print("%33s %.1f" % (fp, median_snr))
        if show_plots:
            plot_spectrum(wave, flux, err, fp)

if __name__ == "__main__":
    main()
