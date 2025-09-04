import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from glob import glob

#try:
show_plots = True

if True:
    fps = sys.argv[-1]
    if "*" in fps:
        fps = list(glob(fps))
    else:
        fps = [fps]
    print(fps)
    for fp in fps:
        df = pd.read_csv(fp, sep="\\s+", header=None)
        wave = df.iloc[:,0]
        flux = df.iloc[:,1]
        err = df.iloc[:,2]
        snr = flux/err
        print(fp, "%.1f" % np.nanmedian(snr))
        if show_plots:
            plt.title(fp)
            plt.plot(wave, flux, color="black")
            plt.plot(wave, err, color="tab:gray")
            plt.xlabel("Wavelength / Angstrom")
            plt.ylabel("Normalised flux")
            ymin, ymax = plt.gca().get_ylim()
            plt.gca().set_ylim(max(0,ymin), max(ymax, 1.2))
            plt.tight_layout()
            plt.show()
#except:
#    print("provide a .dat spectrum as last argument")
