"""Reduce a night of Perek echelle data.

By default this reduces the example night bundled with the repository, so it
can be run without any arguments:

    python template.py

To reduce your own data, point it at the directory holding the FITS frames:

    python template.py /path/to/20250903
    python template.py 20250903 --science e202509030022 --plot
"""
import os

# Pin the linear-algebra libraries to a single thread. This has to happen
# before numpy is imported: OpenBLAS, MKL and Accelerate read these variables
# once, at load time, and otherwise each starts a thread per core. That nests
# underneath our own process pool -- ncpu workers times one thread per core
# each -- and the threads then contend rather than compute. The names cover the
# backends numpy is built against on Linux, macOS and Windows; setting one that
# does not apply is harmless.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import re
import sys
import time
import argparse
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib import pyplot as plt
from echelle_reduction import extract_spectrum
from identify_orders import select_trace_frames
from resample_backend import BACKEND
from calibrate import load_thar_list
from tools import set_ncpu, get_ncpu
from paths import (DEFAULT_IDCOMP_DIR, DEFAULT_THAR_LIST, DEFAULT_DATA_DIR,
                   MURPHY_THAR_LIST)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_dir", nargs="?", default=DEFAULT_DATA_DIR,
                   help="directory with the .fit frames of one night "
                        "(default: the bundled example night)")
    p.add_argument("-o", "--outdir", default="done",
                   help="where to write reduced spectra (default: %(default)s)")
    p.add_argument("-s", "--science", default=None, metavar="NAME",
                   help="only reduce science frames whose filename contains "
                        "NAME (default: all of them)")
    p.add_argument("--idcomp-dir", default=DEFAULT_IDCOMP_DIR,
                   help="directory with idcomp line identifications "
                        "(default: the bundled idcomp_2307)")
    p.add_argument("--idcomp-offset", default="auto", metavar="PX",
                   help="cross-dispersion shift in pixels between the idcomp "
                        "reference and this night; \"auto\" measures it from "
                        "the data (default: %(default)s)")
    p.add_argument("--frame-for-slice", default="auto", metavar="FRAME",
                   help='frame used to trace the orders: "auto" to stack the '
                        'science frames with the most signal in the bluest '
                        'orders, "science" to use all of them, "flat" to use '
                        'only the flat, or a path to a FITS file '
                        '(default: %(default)s)')
    p.add_argument("--trace-stack", type=int, default=4, metavar="N",
                   help="how many science frames \"auto\" stacks to trace on; "
                        "the blue cutoff is set by the best frame, the rest "
                        "guard against cosmics and a bad pick "
                        "(default: %(default)s)")
    # takes a value always, and a comma-separated one for several lists:
    # an optional-argument form (nargs="?"/"*") is ambiguous against the
    # positional data_dir, which argparse resolves by eating the directory
    p.add_argument("--thar-list", default=None, metavar="LIST",
                   help='refine the wavelength solution against ThAr line '
                        'lists. Requires a value: "both" (or "lovis,murphy") '
                        'to merge the two bundled lists, "lovis", "murphy", '
                        'or a comma-separated list of paths. Lovis & Pepe '
                        '(2007) ends at 6912 A, so Murphy (2007) is what '
                        'covers the reddest orders (default: disabled)')
    p.add_argument("--no-normalize", dest="normalize", action="store_false",
                   help="skip continuum normalisation")
    p.add_argument("--no-scattered", dest="remove_scattered",
                   action="store_false",
                   help="keep the grating's scattered-light halo. It is only "
                        "2-3%% of a red order but comparable to the signal in "
                        "the blue, where leaving it in stops flat division "
                        "removing the blaze and makes the order ends droop")
    p.add_argument("--no-fits", dest="save_as_fits", action="store_false",
                   help="do not write the .fits output")
    p.add_argument("--no-ascii", dest="save_as_ascii", action="store_false",
                   help="do not write the .dat output")
    p.add_argument("--plot", dest="plot_spectra", action="store_true",
                   help="show the merged spectrum for each frame")
    p.add_argument("--debug-plots", dest="DEBUG_PLOTS", action="store_true",
                   help="show diagnostic plots from every reduction step")
    p.add_argument("--ncpu", type=int, default=None, metavar="N",
                   help="worker processes to use (default: every core). Pass "
                        "this to turn the worker count down on a shared "
                        "machine; asking for more workers than you have free "
                        "cores makes them contend rather than run")
    p.add_argument("-q", "--quiet", dest="verbose", action="store_false",
                   help="less output")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not os.path.isdir(args.data_dir):
        sys.exit("data directory does not exist: %s" % args.data_dir)
    if not os.path.isdir(args.idcomp_dir):
        sys.exit("idcomp directory does not exist: %s" % args.idcomp_dir)

    frame_for_slice = args.frame_for_slice
    if frame_for_slice not in ("auto", "science", "flat") \
            and not os.path.exists(frame_for_slice):
        print(frame_for_slice + " does not exist")
        frame_for_slice = None

    thar_list = None
    if args.thar_list is not None:
        names = {"lovis": DEFAULT_THAR_LIST, "murphy": MURPHY_THAR_LIST}
        spec = "lovis,murphy" if args.thar_list == "both" else args.thar_list
        wanted = [w.strip() for w in spec.split(",") if w.strip()]
        paths = [names.get(w, w) for w in wanted]
        for p in paths:
            if not os.path.exists(p):
                sys.exit("ThAr line list does not exist: %s" % p)
        thar_list = load_thar_list(*paths)
        if args.verbose:
            print("> ThAr lines: %d covering %.0f-%.0f A"
                  % (len(thar_list), thar_list["wave_air"].min(),
                     thar_list["wave_air"].max()))

    # only override the default (every core) when --ncpu was actually given
    ncpu = set_ncpu(args.ncpu) if args.ncpu is not None else get_ncpu()
    if args.verbose:
        print("> resampling backend: %s" % BACKEND)
        print("> using %d worker process%s"
              % (ncpu, "" if ncpu == 1 else "es"))

    idcomp_offset = args.idcomp_offset
    if idcomp_offset != "auto":
        try:
            idcomp_offset = float(idcomp_offset)
        except ValueError:
            sys.exit('--idcomp-offset must be a number or "auto"')

    reduce_night(args.data_dir, args.idcomp_dir,
                 fn_science=args.science,
                 idcomp_offset=idcomp_offset,
                 frame_for_slice=frame_for_slice,
                 trace_stack=args.trace_stack,
                 outdir=args.outdir,
                 normalize=args.normalize,
                 remove_scattered=args.remove_scattered,
                 verbose=args.verbose,
                 save_as_fits=args.save_as_fits,
                 save_as_ascii=args.save_as_ascii,
                 thar_list=thar_list,
                 plot_spectra=args.plot_spectra,
                 DEBUG_PLOTS=args.DEBUG_PLOTS)


def reduce_night(dir, idcomp_dir, fn_science=None,
                 idcomp_offset="auto",
                 frame_for_slice=None,
                 trace_stack=4,
                 outdir="done",
                 verbose=True,
                 normalize=True,
                 remove_scattered=True,
                 thar_list=None,
                 save_as_fits=True,
                 save_as_ascii=True,
                 plot_spectra=False,
                 DEBUG_PLOTS=False):

    flats = []
    biases = []
    comps = []

    science = []
    scname = []
    # every science frame of the night, whether or not --science selected it:
    # the orders are traced on the frames with the most blue signal, and that
    # is a property of the night, not of the frame being reduced
    all_science = []

    for file in sorted(os.listdir(dir)):
        if file.endswith(".fit"):
            # read eagerly and close: astropy memory-maps by default, and a full
            # night of frames would otherwise keep every file handle open
            with fits.open(os.path.join(dir, file)) as hdul:
                header = dict(hdul[0].header)
                ftype = header["OBJECT"]
                data = np.asarray(hdul[0].data) if ftype in ("zero", "flat", "comp") else None
            if ftype == "zero":
                biases.append(data)
            elif ftype == "flat":
                flats.append(data)
            elif ftype == "comp":
                comps.append(data)
            else:
                all_science.append(file)
                if (fn_science is None) or (fn_science in file):
                    science.append(file)
                    scname.append(ftype.strip().replace(" ", "_"))

    if len(science) == 0:
        if fn_science is not None:
            sys.exit("did not find %s in %s" % (fn_science, dir))
        sys.exit("no science frames found in %s" % dir)

    os.makedirs(outdir, exist_ok=True)

    # Resolve "auto" once: the trace frames describe the night, so choosing
    # them per science frame would only repeat the same measurement.
    trace_frames = None
    if frame_for_slice == "auto":
        bias_ref = np.median(biases, axis=0) if len(biases) else None
        # the flat tells blue from red: it falls away towards the blue, so the
        # faint end of the order block is the one that sets the blue cutoff
        flat_ref = np.median(flats, axis=0) if len(flats) else None
        trace_frames = select_trace_frames(
            [os.path.join(dir, sc) for sc in all_science],
            bias=bias_ref, nstack=trace_stack, verbose=verbose,
            flat=flat_ref)
        if not trace_frames:
            # calibration-only night, or nothing scorable: the flat is the
            # only thing left to trace on, which is what find_orders uses
            # when it is handed nothing
            if verbose:
                print("- no usable science frame to trace on; using the flat")
            trace_frames = None

    orders = None
    for i in range(len(science)):
        fp = science[i]
        name = scname[i]

        # replace non-alphanumeric characters, except "_", ".", "+", "-"
        fp_save = re.sub(r'[^\w_.+-]', '_', fp)
        fp_save = os.path.join(outdir, fp_save)
        fp_save_fits = fp_save.replace(".fit", "_" + name + ".fits")
        fp_save_ascii = fp_save.replace(".fit", "_" + name + ".dat")
        if (save_as_fits and (not os.path.exists(fp_save_fits))) or \
           (save_as_ascii and (not os.path.exists(fp_save_ascii))):
            print("> reducing %s (%s)" % (fp, name))
            tstart = time.time()
            if frame_for_slice == "auto":
                frame_for_slice_i = trace_frames
            elif frame_for_slice == "science":
                frame_for_slice_i = [os.path.join(dir, sc) for sc in science]
            elif frame_for_slice == "flat":
                frame_for_slice_i = None
            else:
                frame_for_slice_i = frame_for_slice
            s = extract_spectrum(os.path.join(dir, fp), flats, comps, biases,
                                 idcomp_offset=idcomp_offset,
                                 frame_for_slice=frame_for_slice_i,
                                 orders=orders,
                                 normalize=normalize,
                                 remove_scattered=remove_scattered,
                                 idcomp_dir=idcomp_dir,
                                 thar_list=thar_list,
                                 verbose=verbose,
                                 DEBUG_PLOTS=DEBUG_PLOTS)
            tstop = time.time()
            print("> done in %.1f s" % (tstop-tstart))
            orders = s["orders"]

            mask_good = np.isfinite(s["error"]) & (s["error"] > 0)
            if np.any(mask_good):
                SNR = float(np.nanmedian(s["flux"][mask_good]/s["error"][mask_good]))
            else:
                SNR = float("nan")
            print("> median SNR = %.1f" % SNR)

            if save_as_fits:
                # steal the original header
                with fits.open(os.path.join(dir, fp)) as hdul:
                    header = hdul[0].header
                if s["bjd"] is not None:
                    header["BJD"] = s["bjd"]
                # FITS cannot store a NaN card value
                if np.isfinite(SNR):
                    header["SNR"] = SNR
                primary_hdu = fits.PrimaryHDU(header=header)
                fits_cols = [fits.Column(name=key, array=s[key], format='D') for key in s if type(s[key]) == np.ndarray]
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
                plt.title("Merged spectrum")
                plt.xlabel("Wavelength / Angstrom")
                plt.ylabel("Normalised flux")
                plt.tight_layout()
                plt.show()
        else:
            print("> skipped %s (%s)" % (fp, name))


if __name__ == "__main__":
    main()
