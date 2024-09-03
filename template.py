import os
import sys
import time
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from echelle_reduction import extract_spectrum

#spec = "e202408300035.fit"
spec = None
idcomp_dir = "idcomp_2307/"
dir = "20240901"
plot_spectra = True
verbose = True

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
            if (spec is not None) and (spec in file):
                science.append(file)
                scname.append(ftype.strip().replace(" ", "_"))
            elif spec is None:
                science.append(file)
                scname.append(ftype.strip().replace(" ", "_"))

if len(science) == 0:
    if spec is not None:
        sys.exit("did not find %s in %s" % (spec, dir))

for i in range(len(science)):
    fp = science[i]
    name = scname[i]
    fp_save = fp.replace(".fit", "_" + name + ".dat")

    if not os.path.exists(fp_save):
        print("> reducing %s (%s)" % (fp, name))
        tstart = time.time()
        s = extract_spectrum(dir+"/"+fp, flats, comps, biases,
                             idcomp_dir=idcomp_dir, verbose=verbose)
        tstop = time.time()
        print("> done in %.1f s" % (tstop-tstart))

        mask_good = s["error"] > 0
        SNR = np.nanmedian(s["flux"][mask_good]/s["error"][mask_good])
        print("> median SNR = %.1f" % SNR)

        d_save = np.vstack([s["wave"], s["flux"], s["error"]]).T
        np.savetxt(fp_save, d_save, fmt="%1.6f")
        print("> saved to", fp_save)

        if plot_spectra:
            plt.plot(s["wave"], s["flux"], linewidth=1, color="black")
            plt.plot(s["wave"], s["error"], linewidth=1, color="gray")
            plt.ylim((0, 2))
            plt.legend()
            plt.tight_layout()
            plt.show()
    else:
        print("> skipped %s (%s)" % (fp, name))

