import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob
from astropy.io import fits

def read_fits_file(filepath):
    with fits.open(filepath) as hdul:
        header = hdul[0].header
        table_data = hdul['SCIENCE'].data
        bjd = header.get('BJD', None)
        data_dict = {}
        if bjd is not None:
            data_dict['bjd'] = bjd
        for col in table_data.columns:
            col_data = table_data[col.name]
            if hasattr(col_data, 'byteswap'):
                col_data = col_data.byteswap()
                if hasattr(col_data, 'newbyteorder'):
                    col_data = col_data.newbyteorder()
                else:
                    col_data = col_data.view(col_data.dtype.newbyteorder())
            data_dict[col.name] = col_data
        df = pd.DataFrame(data_dict)
        return df, header

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

    for fp in fps:
        if "fits" in fp:
            data, header = read_fits_file(fp)
            wave = data["wave"]
            flux = data["flux"]
            err = data["error"]
        else:
            wave, flux, err = read_spectrum_file(fp)
        snr = flux/err
        median_snr = np.nanmedian(snr)
        print(fp, "%.1f" % median_snr)
        if show_plots:
            plot_spectrum(wave, flux, err, fp)

if __name__ == "__main__":
    main()
