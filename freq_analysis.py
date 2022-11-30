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
from matplotlib import ticker


# chdir('C:/Users/Aakas/Documents/School/Oster_lab/programs')
chdir('C:/Users/Aakas/Documents/School/Oster_lab/')


def load_data(clean=True, filter_year='40000'):
    """
    Loats MAW-3 record (and eventually CH1 too!)
    """
    maw_3_proxy = pd.read_csv('internal_excel_sheets/filled_seb_runs/' +
                              'MAW-3-filled-AGES.csv')

    # Greenland NGRIP Data to compare trends
    # https://www.iceandclimate.nbi.ku.dk/data/
    ngrip_data = pd.read_excel('external_excel_sheets/NGRIP_d18O.xls',
                               sheet_name='NGRIP-2 d18O and Dust',
                               names=['depth_m', 'd18O', 'dust',
                                      'age_BP', 'age_error'])

    if clean:
        maw_3_clean = maw_3_proxy.query(f'age_BP <= {filter_year}')
        maw_3_clean = maw_3_clean.dropna(subset=['age_BP', 'd18O', 'd13C'])

    records = {'maw_3_clean': maw_3_clean,
               'maw_3_proxy': maw_3_proxy,
               'ngrip': ngrip_data}

    return records


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


def compare_records(cave_record, ngrip_record, since=40000, how='layer'):
    """
    Plots the two records side by side to allow for comparisions
    """
    cave_record = cave_record.query(f'age_BP < {since}')

    max_year = cave_record['age_BP'].iloc[-1]
    min_year = cave_record['age_BP'].iloc[1]

    ngrip_small = ngrip_record.query(f'age_BP <= {max_year} & ' +
                                     f'age_BP >= {min_year}')

    if how == 'layer':
        layered_plot(cave_record, ngrip_small)
    else:
        staggered_plot(cave_record, ngrip_small)


def layered_plot(cave_record, ngrip_small):
    """
    Plots the records overlaid with each other
    """
    fig, ax = plt.subplots()
    twin1 = ax.twinx()
    twin2 = ax.twinx()
    twin2.spines['right'].set_position(("axes", 1.15))

    color1 = plt.cm.viridis(0)
    color2 = plt.cm.viridis(0.5)
    color3 = plt.cm.viridis(.9)

    d18O_green, = ax.plot(ngrip_small['age_BP'], ngrip_small['d18O'],
                          label='NGRIP d18O', color=color3)
    d18O, = twin1.plot(cave_record['age_BP'], cave_record['d18O'],
                       label='MAW-3 d18O', color=color1)
    d13C, = twin2.plot(cave_record['age_BP'], cave_record['d13C'],
                       label='MAW-3 d13C', color=color2)

    ax.set_xlim()
    ax.grid()
    ax.set_ylabel('NGRIP d18O (‰)')
    twin1.set_ylabel('MAW-3 d18O (‰)')
    twin2.set_ylabel('MAW-3 d13C (‰)')
    ax.set_xlabel('Age (Years BP)')
    ax.set_title('MAW-3 Comparision with NGRIP Record')

    ax.set_ylim(-55, -30)
    twin1.set_ylim(-12, 0)
    twin2.set_ylim(-5, 4)

    ax.legend(handles=[d18O_green, d13C, d18O], loc='best')

    plt.show()


def staggered_plot(cave_record, ngrip_small):
    """
    Plots the records but in a staggered fashion
    """
    fig, ax = plt.subplots(3, 1, sharex=True)
    fig.set_size_inches(10, 5)
    fig.subplots_adjust(hspace=0)

    color1 = plt.cm.viridis(0)
    color2 = plt.cm.viridis(0.5)
    color3 = plt.cm.viridis(.9)

    ax[0].plot(cave_record['age_BP'], cave_record['d18O'],
               label='MAW-3 d18O', color=color1)
    ax[0].set_ylim(-8, -0.5)
    ax[0].grid()
    ax[0].set_ylabel('MAW-3 d18O ‰')
    ax[0].set_yticks(np.arange(-7, 0, 2))

    ax[1].plot(cave_record['age_BP'], cave_record['d13C'],
               label='MAW-3 d13C', color=color2)
    ax[1].set_ylim(-5, 4)
    ax[1].grid()
    ax[1].set_ylabel('MAW-3 d13C ‰')
    ax[1].set_yticks(np.arange(-4, 4, 2))

    ax[2].plot(ngrip_small['age_BP'], ngrip_small['d18O'],
               label='NGRIP d18O', color=color3)
    ax[2].set_ylim(-52, -33)
    ax[2].grid()
    ax[2].set_ylabel('NGRIP d18O ‰')
    ax[2].set_xlabel('Age (Years BP)')
    ax[2].set_yticks(np.arange(-48, -32, 4))

    ax[0].set_title('Comparison of Speleothem Proxies with NGRIP d18O')
    plt.show()


def main():
    down_period = 20

    records = load_data(clean=True, filter_year='40000')
    records['maw_3_down'] = downsample(records['maw_3_clean'], down_period)

    # Let's plot a power spectral density
    fourier(records['maw_3_down'], proxy='d18O', fig=1, fs=down_period)
    fourier(records['maw_3_down'], proxy='d13C', fig=1, fs=down_period)

    # Pandas autocorrelation plots- d13C has no interesting signal
    # but d18O does
    pd.plotting.autocorrelation_plot(records['maw_3_down']['d18O'])
    plt.title('Autocorrelation of d18O')

    plt.figure(2)
    pd.plotting.autocorrelation_plot(records['maw_3_down']['d13C'])
    plt.title('Autocorrelation of d13C')

    # Let's try to find the actual peak
    corr_range = [records['maw_3_down']['d18O'].autocorr(lag) for lag in range(100, 200)]
    max_corr = down_period * (corr_range.index(max(corr_range)) + 100)
    print(f'Max autocorrelation at {max_corr} year period')

    # Comapre to NGRIP
    compare_records(records['maw_3_proxy'], records['ngrip'], how='stagger')

    # Downsampled for input to matlab- this will be out actual CWT test
    # as the defauly python ones are lacking
    records['maw_3_down'].to_csv('internal_excel_sheets/filled_seb_runs/' +
                                 'MAW-3-downsample.csv', index=False)


if __name__ == '__main__':
    main()
