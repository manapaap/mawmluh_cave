# -*- coding: utf-8 -*-
"""
Analyze frequencies present in MAW-3 (and eventually CH1) through
fourier and wavelet transform

@author: Aakas
"""

import pywt
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
from scipy.interpolate import interp1d
import pandas as pd
from os import chdir


# chdir('C:/Users/Aakas/Documents/School/Oster_lab/programs')
chdir('C:/Users/Aakas/Documents/School/Oster_lab/')


def load_data(clean=True, filter_year='40000'):
    """
    Loats MAW-3 record (and eventually CH1 too!)
    """
    maw_3_proxy = pd.read_csv('internal_excel_sheets/filled_seb_runs/' +
                              'MAW-3-filled-AGES.csv')
    if clean:
        maw_3_proxy = maw_3_proxy.query(f'age_BP <= {filter_year}')
        maw_3_proxy = maw_3_proxy.dropna(subset=['age_BP', 'd18O', 'd13C'])

    return maw_3_proxy


def downsample(df, resolution, kind='slinear'):
    """
    Returns a downsampled version of our proxy record in dataframe form
    """
    func_d18O = interp1d(df['age_BP'], df['d18O'], kind=kind)
    func_d13C = interp1d(df['age_BP'], df['d13C'], kind=kind)
    new_ages = np.arange(df['age_BP'].min(), df['age_BP'].max(), resolution)
    new_df = pd.DataFrame({'d18O': func_d18O(new_ages),
                           'd13C': func_d13C(new_ages),
                           'age_BP': new_ages})
    return new_df


def sine_wave(input_x, wavelen, y_shift=0):
    """
    Outputs sine wave of given wavelength across extent of the input array
    """
    return np.sin(2 * np.pi * input_x / wavelen) + y_shift


def im_plot_wavelet(coef, freq):
    """
    Plots the scalogram using a different plotting method so I can compare them
    """
    plt.imshow(abs(coef), interpolation='bilinear', aspect='auto',
               vmax=abs(coef).max(), vmin=abs(coef).min())
    plt.gca().invert_yaxis()
    plt.show()


def fourier(df, fs=20, proxy='d18O', fig=1):
    """
    Plots a power spectral density of our downscaled dataframe-
    again to observe any dominant frequencies
    """
    pxx, freq = plt.psd(df[proxy], Fs=1 / fs)
    plt.title(f'Power Spectral Density of {proxy}')
    plt.show()


def main():
    down_period = 20

    maw_3_proxy = load_data(clean=True, filter_year='40000')
    maw_3_down = downsample(maw_3_proxy, down_period)

    # Let's plot a power spectral density
    fourier(maw_3_down, proxy='d18O', fig=1, fs=down_period)
    fourier(maw_3_down, proxy='d13C', fig=1, fs=down_period)

    # Pandas autocorrelation plots- d13C has no interesting signal
    # but d18O does
    pd.plotting.autocorrelation_plot(maw_3_down['d18O'])
    plt.title('Autocorrelation of d18O')

    plt.figure(2)
    pd.plotting.autocorrelation_plot(maw_3_down['d13C'])
    plt.title('Autocorrelation of d13C')

    # Let's try to find the actual peak
    corr_range = [maw_3_down['d18O'].autocorr(lag) for lag in range(100, 200)]
    max_corr = down_period * (corr_range.index(max(corr_range)) + 100)
    print(f'Max autocorrelation at {max_corr} year period')

    # Downsampled for input to matlab- this will be out actual CWT test
    # as the defauly python ones are lacking
    maw_3_down.to_csv('internal_excel_sheets/filled_seb_runs/' +
                      'MAW-3-downsample.csv', index=False)


if __name__ == '__main__':
    main()
