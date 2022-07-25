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
from scipy.signal import welch
from scipy.interpolate import interp1d
import pandas as pd
from os import chdir
from ssqueezepy import ssq_cwt, visuals


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


def im_plot_wavelet(coef, freq):
    """
    Plots the scalogram using a different plotting method so I can compare them
    """
    plt.imshow(abs(coef), interpolation='bilinear', aspect='auto',
               vmax=abs(coef).max(), vmin=abs(coef).min())
    plt.gca().invert_yaxis()
    plt.show()


def other_cwt(df, proxy='d18O'):
    Tx, Wx, freqs, scales, *_ = ssq_cwt(np.array(df[proxy]),
                                        wavelet='cmhat')
    visuals.imshow(Wx, abs=1, yticks=1/freqs)


def fourier(df, fs=20, proxy='d18O', fig=1):
    """
    Plots a power spectral density of our downscaled dataframe-
    again to observe any dominant frequencies
    """
    pxx, freq = plt.psd(df[proxy], Fs=1 / fs)
    plt.show()


def main():
    maw_3_proxy = load_data(clean=True, filter_year='40000')
    maw_3_down = downsample(maw_3_proxy, 20)

    # coefs_O, freq_O = pywt.cwt(maw_3_down['d18O'],
    #                            np.arange(1, 2048, 1),
    #                            'morl', 20)
    # coefs_C, freq_C = pywt.cwt(maw_3_down['d13C'],
    #                           np.arange(1, 2048, 1),
    #                           'morl', 20)

    # im_plot_wavelet(coefs_O, freq_O)
    # im_plot_wavelet(coefs_C, freq_C)

    other_cwt(maw_3_down, 'd18O')
    other_cwt(maw_3_down, 'd13C')

    fourier(maw_3_down, proxy='d18O', fig=1, fs=20)

    # Downsampled for input to matlab- this will be out actual CWT test
    # as the defauly python ones are lacking
    maw_3_down.to_csv('internal_excel_sheets/filled_seb_runs/' +
                      'MAW-3-downsample.csv', index=False)


if __name__ == '__main__':
    main()
