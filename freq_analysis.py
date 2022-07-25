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
import pandas as pd
from os import chdir


# chdir('C:/Users/Aakas/Documents/School/Oster_lab/programs')
chdir('C:/Users/Aakas/Documents/School/Oster_lab/')


def load_data(clean=True, filter_year='42500'):
    """
    Loats MAW-3 record (and eventually CH1 too!)
    """
    maw_3_proxy = pd.read_csv('internal_excel_sheets/filled_seb_runs/' +
                              'MAW-3-filled-AGES.csv')
    if clean:
        maw_3_proxy = maw_3_proxy.query(f'age_BP <= {filter_year}')
        maw_3_proxy = maw_3_proxy.dropna(subset=['age_BP', 'd18O', 'd13C'])

    return maw_3_proxy


def wavelet(time, signal, scales, wavelet='mexh'):
    """
    Performs our wavelet transform
    """
    coefs, freq = pywt.cwt(signal, scales, wavelet)
    return coefs, freq


def plot_wavelet(coefficients, frequencies, time, proxy='d18O'):
    """
    Plots wavelets
    """
    power = abs(coefficients)
    period = 1. / frequencies
    levels = [0.0625 * 2**n for n in range(10)]
    contourlevels = np.log2(levels)

    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.contourf(time, np.log2(period), np.log2(power),
                     contourlevels, extend='both')

    ax.set_title('Wavelet Transform (Power Spectrum) of ' + proxy,
                 fontsize=20)
    ax.set_ylabel('Period (years)', fontsize=18)
    ax.set_xlabel('Time', fontsize=18)

    yticks = 2**np.arange(np.ceil(np.log2(period.min())),
                          np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], -1)

    cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    plt.grid()
    plt.show()


def im_plot_wavelet(coef, freq):
    """
    Plots the scalogram using a different plotting method so I can compare them
    """
    plt.imshow(abs(coef), interpolation='bilinear', aspect='auto',
               vmax=abs(coef).max(), vmin=abs(coef).min())
    plt.gca().invert_yaxis()
    plt.show()


def main():
    global freq_O, coefs_O
    maw_3_proxy = load_data()
    coefs_O, freq_O = pywt.cwt(maw_3_proxy['d18O'], np.arange(1, 4096, 1),
                               'morl', maw_3_proxy['time_resolution'].mean())
    coefs_C, freq_C = pywt.cwt(maw_3_proxy['d13C'], np.arange(1, 2048, 1),
                               'morl', maw_3_proxy['time_resolution'].mean())

    plot_wavelet(coefs_O, freq_O, maw_3_proxy['age_BP'])
    im_plot_wavelet(coefs_O, freq_O)
    plot_wavelet(coefs_C, freq_C, maw_3_proxy['age_BP'], proxy='d13C')


if __name__ == '__main__':
    main()
