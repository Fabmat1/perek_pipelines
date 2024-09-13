import re
import os
import sys
import time
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from echelle_reduction import extract_spectrum

def main():
    dir = "20240901"
    idcomp_dir = "idcomp_2307/"
    fn_science = None
    #fn_science = "e202408300035.fit"
    verbose = True
    save_as_fits = True
    save_as_ascii = True
    plot_spectra = False
    frame_for_slice = "20240901/e202409020033.fit"
    if not os.path.exists(frame_for_slice):
        frame_for_slice = None

    reduce_night(dir, idcomp_dir,
                 fn_science=fn_science,
                 frame_for_slice=frame_for_slice,
                 verbose=verbose,
                 save_as_fits=save_as_fits,
                 save_as_ascii=save_as_ascii,
                 plot_spectra=plot_spectra)

def reduce_night(dir, idcomp_dir, fn_science=None,
                 frame_for_slice=None,
                 verbose=True,
                 save_as_fits=True,
                 save_as_ascii=True,
                 plot_spectra=False):

    flats = []
    biases = []
    comps = []

    science = []
    scname = []

    for file in sorted(os.listdir(dir)):
        if file.endswith(".fit"):
            hdul = fits.open(os.path.join(dir, file))
            header = dict(hdul[0].header)
            ftype = header["OBJECT"]
            if ftype == "zero":
                biases.append(hdul[0].data)
            elif ftype == "flat":
                flats.append(hdul[0].data)
            elif ftype == "comp":
                comps.append(hdul[0].data)
            else:
                if (fn_science is not None) and (fn_science in file):
                    science.append(file)
                    scname.append(ftype.strip().replace(" ", "_"))
                elif fn_science is None:
                    science.append(file)
                    scname.append(ftype.strip().replace(" ", "_"))

    if len(science) == 0:
        if fn_science is not None:
            sys.exit("did not find %s in %s" % (fn_science, dir))

    for i in range(len(science)):
        fp = science[i]
        name = scname[i]

        # replace non-alphanumeric characters, except "_", ".", "+", "-"
        fp_save = re.sub(r'[^\w_.+-]', '_', fp)

        fp_save_fits = fp_save.replace(".fit", "_" + name + ".fits")
        fp_save_ascii = fp_save.replace(".fit", "_" + name + ".dat")
        if (save_as_fits and (not os.path.exists(fp_save_fits))) or \
           (save_as_ascii and (not os.path.exists(fp_save_ascii))):
            print("> reducing %s (%s)" % (fp, name))
            tstart = time.time()
            s = extract_spectrum(dir+"/"+fp, flats, comps, biases,
                                 frame_for_slice=frame_for_slice,
                                 idcomp_dir=idcomp_dir, verbose=verbose)
            tstop = time.time()
            print("> done in %.1f s" % (tstop-tstart))

            mask_good = s["error"] > 0
            SNR = np.nanmedian(s["flux"][mask_good]/s["error"][mask_good])
            print("> median SNR = %.1f" % SNR)

            if save_as_fits:
                # steal the original header
                with fits.open(dir+"/"+fp) as hdul:
                    header = hdul[0].header
                primary_hdu = fits.PrimaryHDU(header=header)
                fits_cols = [fits.Column(name=key, array=s[key], format='D') for key in s]
                # the name is always in captials
                table_hdu = fits.BinTableHDU.from_columns(fits_cols, name="SCIENCE")
                hdul = fits.HDUList([primary_hdu, table_hdu])
                hdul.writeto(fp_save_fits, overwrite=True)
                print("> saved to", fp_save_fits)
            if save_as_ascii:
                d_save = np.vstack([s["wave"], s["flux"], s["error"], s["res"]]).T
                np.savetxt(fp_save_ascii, d_save, fmt="%1.6f")
                print("> saved to", fp_save_ascii)

            if plot_spectra:
                plt.plot(s["wave"], s["flux"], linewidth=1, color="black")
                plt.plot(s["wave"], s["error"], linewidth=1, color="gray")
                plt.ylim((0, 2))
                plt.legend()
                plt.tight_layout()
                plt.show()
        else:
            print("> skipped %s (%s)" % (fp, name))


if __name__ == "__main__":
    main()

