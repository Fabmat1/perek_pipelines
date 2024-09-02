import os
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from echelle_reduction import extract_spectrum

dir = "20240830"
fname = "e202408300035.fit"

flats = []
biases = []
comps = []

data2 = fits.open("~/Downloads/DCN-TYC1531-595-1_test0830.fit")[0]
dh = dict(data2.header)
wls2 = np.arange(len(data2.data))*dh["CDELT1"]+dh["CRVAL1"]
flx2 = data2.data

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

wl, flx = extract_spectrum(dir+"/"+fname, flats, comps, biases)


plt.plot(wls2, flx2, linewidth=1, label="Icky bad IRAF :(", color="darkred")
plt.plot(wl, flx, linewidth=1, color="black", label="Good and cool Pipeline")
plt.ylim((0, 4))
plt.legend()
plt.tight_layout()
plt.show()

np.savetxt(fname.replace(".fit", ".dat"), np.vstack([wl, flx]).T, fmt="%1.6f")
