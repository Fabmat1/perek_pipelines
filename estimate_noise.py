import numpy as np
from scipy.special import binom

def estimate_noise_scalar(flux, **kwargs):
    '''
    mostly based on https://stdatu.stsci.edu/vodocs/der_snr.pdf
    works only if there are no gaps in spectrum
    '''
    n = len(flux)
    # For spectra shorter than this, no value can be returned
    order = 3
    npixmin = 6
    if 'order' in kwargs:
        order = kwargs['order']
    if order == 4:
        npixmin = 10
    if (n > npixmin):
        signal = np.median(flux)
        if order == 4:
            # using 4th order of median (tracks spectral lines better, may overestimate SNR)
            # testing with gaussian noise indicates this is more reliable at least in UV and SNR=30
            f4 = 1.482602 / np.sqrt(binom(3, 0)**2 + binom(3, 1)**2 + binom(3, 2)**2 + binom(3, 3)**2)
            noise = f4 * np.median(np.abs(3.0 * flux[4:n-4] - flux[2:n-6] - 3.0 * flux[6:n-2] + flux[8:n]))
        elif order == 3:
            # using 3rd order of median (linear slopes not counted as noise, may underestimate SNR)
            #f3 = 1.482602 / np.sqrt(binom(2, 0)**2 + binom(2, 1)**2 + binom(2, 2)**2)
            f3 = 0.6052697319
            noise = f3 * np.median(np.abs(2.0 * flux[2:n-2] - flux[0:n-4] - flux[4:n]))
        else:
            return float('NaN')
        return noise
    else:
        return float('NaN')

def fill_nan(y):
    '''replace nan values in 1-array by interpolated values'''

    y = np.array(y)
    nans = np.isnan(y)
    x = lambda z: z.nonzero()[0]
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])

    return y

def estimate_noise(wave, flux, winsize_aa='guess'):
    '''
    Estimate the uncertainty for each wavelength
    by applying "estimate_noise_scalar" in a window around each pixel
    winsize_aa: window size in Angstrom (max: 10% of wave array
                                         min: 20 pixel)
    if winsize_aa == "guess":
        winsize_aa = 150 * (np.max(wave) - np.min(wave)) / len(wave)
        -> widow width is set to 1/150 of the spectral range
    '''

    flux = np.array(flux)

    if winsize_aa == 'guess':
        winsize_aa = 150 * (np.max(wave) - np.min(wave)) / len(wave)

    # for symmetric window, otherwise noise column is shifted
    winsize_aa_half = winsize_aa / 2

    # approx. translate Angstrom to pixel using average spacing
    avspacing = np.median([wave[i+1] - wave[i] for i in range(len(wave)-1)])

    winsize_pix_av = int(winsize_aa_half / avspacing)
    winsize_min = 20
    winsize_max = int(len(wave)/10)
    winsize = max(min(winsize_pix_av, winsize_max), winsize_min)

    keyw = {'order':3}
    noise = [estimate_noise_scalar(flux[i-winsize:i+winsize], **keyw) for i in range(len(flux))]

    # replace nan values by interpolated values
    noise = fill_nan(noise)

    return noise

