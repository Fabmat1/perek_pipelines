import numpy as np
import pyresample_spectres

# python3 -m numpy.f2py -c -m pyresample_spectres resample_spectres.f90

def resample(wave_out, wave_in, flux_in, fill=0.0, verbose=False):
    nwave_in = len(wave_in)
    nwave_out = len(wave_out)
    # no need to resample if output array is much longer than input
    if nwave_in >= int(nwave_out / 1.2):
        wave_in = np.asfortranarray(wave_in, dtype='d')
        flux_in = np.asfortranarray(flux_in, dtype='d')

        wave_out = np.asfortranarray(wave_out, dtype='d')
        flux_out = np.asfortranarray(np.zeros(nwave_out, dtype='d'))

        pyresample_spectres.resample(wave_in, flux_in, nwave_in,
                                    wave_out, flux_out, nwave_out,
                                    fill)
    else:
        flux_out = np.interp(wave_out, wave_in, flux_in)
        if verbose:
            print("%d < %d: using np.interp", nwave_in, int(nwave_out / 1.2))

    return flux_out

def performance(number=1000):
    import timeit
    setup = '''
import numpy as np
from resample_spectres import resample
nx = 100000
nxx = 1000
x = np.linspace(0, 3*np.pi, nx)
y = np.sin(x)
xx = np.linspace(0, 3*np.pi, nxx)
'''

    a = timeit.repeat("yy = resample(xx, x, y, fill=0.0)",
                  setup=setup,
                  number=number)
    a = np.array(a) / number

    print("resample_spectres: %.2f +- %.2f us" % (np.mean(a)*1e6, np.std(a)*1e6))

def main():
    performance()

if __name__ == "__main__":
    main()
