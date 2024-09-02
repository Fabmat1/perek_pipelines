import os
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from echelle_reduction import extract_spectrum

#spec = "e202408300035.fit"
spec = None
idcomp_dir = "idcomp_2307/"
dir = "20240901"

flats = []
biases = []
comps = []

science = []
scname = []

for file in os.listdir(dir):
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
        print("did not find %s in %s" % (spec, dir))

for i in range(len(science)):
    fp = science[i]
    name = scname[i]
    print("> reducing", fp)
    wl, flx = extract_spectrum(dir+"/"+fp, flats, comps, biases,
                               idcomp_dir=idcomp_dir)

    fp_save = fp.replace(".fit", "_" + name + ".dat")
    np.savetxt(fp_save, np.vstack([wl, flx]).T, fmt="%1.6f")

    plot = False
    if plot:
        plt.plot(wl, flx, linewidth=1, color="black")
        plt.ylim((0, 4))
        plt.legend()
        plt.tight_layout()
        plt.show()
