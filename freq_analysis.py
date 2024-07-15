# -*- coding: utf-8 -*-
"""
Analyze frequencies present in MAW-3 (and eventually CH1) through
fourier and wavelet transform

@author: Aakas
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from os import chdir
from scipy.signal import find_peaks
import pywt


chdir('C:/Users/aakas/Documents/Oster_lab/programs')
from shared_funcs import combine_mawmluh, load_data
chdir('C:/Users/aakas/Documents/Oster_lab/')


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


def compare_records(cave_record, ice_record, cave_2, since, how='north', prox='d18O'):
    """
    Plots the two records side by side to allow for comparisions
    """
    cave_record = cave_record.query(f'age_BP < {since}')

    max_year = cave_record['age_BP'].iloc[-1]
    min_year = cave_record['age_BP'].iloc[1]

    ice_small = ice_record.query(f'age_BP <= {max_year} & ' +
                                     f'age_BP >= {min_year}')
    cave_2_small = cave_2.query(f'age_BP <= {max_year} & ' +
                            f'age_BP >= {min_year}')

    if how == 'north':
        staggered_plot(cave_record, ice_small, cave_2_small)
    else:
        staggered_plot_south(cave_record, ice_small, cave_2_small, prox)


def staggered_plot(cave_record, ngrip_small, hulu_small):
    """
    Plots the records but in a staggered fashion
    """
    fig, ax = plt.subplots(4, 1, sharex=True)
    plt.subplots_adjust(top=0.7)
    fig.set_size_inches(10, 8)
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0)

    color1 = plt.cm.viridis(0.5)
    color2 = plt.cm.viridis(0)
    color3 = plt.cm.viridis(0.7)
    color4 = plt.cm.viridis(0.95)

    ax[1].plot(cave_record['age_BP'], cave_record['d18O'],
               label='MAW-3 d18O', color=color1)
    ax[1].set_ylim(-8, -0.5)
    ax[1].invert_yaxis()
    ax[1].grid()
    ax[1].set_ylabel('MAW-3 δ¹⁸O ‰')
    ax[1].set_yticks(np.arange(-7, 0, 2))
    ax[1].spines[['top']].set_visible(False)
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")

    ax[0].plot(cave_record['age_BP'], cave_record['d13C'],
               label=' MAW-3 δ¹³C', color=color2)
    ax[0].set_ylim(-5, 4)
    ax[0].invert_yaxis()
    ax[0].grid()
    ax[0].set_ylabel('MAW-3 δ¹³C')
    ax[0].set_yticks(np.arange(-4, 4, 2))

    ax[2].plot(hulu_small['age_BP'], hulu_small['d18O'],
               label='NGRIP d18O', color=color3)
    ax[2].grid()
    ax[2].invert_yaxis()
    ax[2].set_ylabel('Hulu δ¹⁸O ‰ ')
    ax[2].set_yticks(np.arange(-8, -6, 1))
    ax[2].spines[['top']].set_visible(False)

    ax[3].plot(ngrip_small['age_BP'], ngrip_small['d18O'],
               label='NGRIP d18O', color=color4)
    ax[3].set_ylim(-52, -33)
    ax[3].grid()
    ax[3].set_ylabel('NGRIP δ¹⁸O ‰')
    ax[3].set_xlabel('Age (Years BP)')
    ax[3].set_yticks(np.arange(-48, -32, 4))
    ax[3].spines[['top']].set_visible(False)
    ax[3].yaxis.tick_right()
    ax[3].yaxis.set_label_position("right")
    

    ax[0].set_title('Proxy Stack')
    plt.show()


def staggered_plot_south(cave_record, vostok_small, ball_small, prox):
    """
    Plots the speleothem record, comparing it to records from the
    southern hemisphere
    """
    fig, ax = plt.subplots(4, 1, sharex=True)
    fig.set_size_inches(10, 8)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)

    color1 = plt.cm.viridis(0.5)
    color2 = plt.cm.viridis(0)
    color3 = plt.cm.viridis(0.7)
    color4 = plt.cm.viridis(0.95)

    ax[1].plot(cave_record['age_BP'], cave_record['d18O'],
               label='MAW-3 d18O', color=color1)
    ax[1].set_ylim(-8, -0.5)
    ax[1].invert_yaxis()
    ax[1].grid()
    ax[1].set_ylabel('MAW-3 δ¹⁸O ‰')
    ax[1].set_yticks(np.arange(-7, 0, 2))

    ax[0].plot(cave_record['age_BP'], cave_record['d13C'],
               label=' MAW-3 δ¹³C', color=color2)
    ax[0].set_ylim(-5, 4)
    ax[0].invert_yaxis()
    ax[0].grid()
    ax[0].set_ylabel('MAW-3 δ¹³C')
    ax[0].set_yticks(np.arange(-4, 4, 2))

    ax[2].plot(ball_small['age_BP'], ball_small[prox],
               label='NGRIP d18O', color=color3)
    ax[2].grid()
    ax[2].invert_yaxis()
    ax[2].set_ylabel('Arabia Sed. Refl.')
    # ax[2].set_yticks(np.arange(-8, -6, 1))

    ax[3].plot(vostok_small['age_BP'], vostok_small['d18O'],
               label='WAIS d18O', color=color4)
    # ax[3].set_ylim(-52, -33)
    ax[3].grid()
    ax[3].set_ylabel('WAIS d18O')
    ax[3].set_xlabel('Age (Years BP)')
    # ax[3].set_yticks(np.arange(-48, -32, 4))

    ax[0].set_title('Comparison of Speleothem Proxies with WAIS d18O')
    plt.show()


def merge_records(maw_3_record, ch1_record):
    """
    Merges the two individual records for one collective larger record

    Colbined record will only contain minimum needed information
    """
    maw_3_strip = maw_3_record[['age_BP', 'd18O', 'd13C', 'stal', 'segment']]
    ch1_strip = ch1_record[['age_BP', 'd18O', 'd13C', 'stal', 'segment']]

    full_record = pd.concat([maw_3_strip, ch1_strip])
    full_record.sort_values(by='age_BP', inplace=True)

    return full_record


def main():
    global records
    down_period = 15

    records = load_data(filter_year='46000')
    records['maw_comb'] = combine_mawmluh(records, cutoff=39500)

    records['maw_3_down'] = downsample(records['maw_comb'], down_period)
    
    pd.plotting.autocorrelation_plot(records['maw_3_down']['d18O'])
    plt.title('Autocorrelation of d18O')
    # Let's try to find the actual peak
    corr_range = [records['maw_3_down']['d18O'].autocorr(lag) for lag in range(100, 600)]
    peaks, _ = find_peaks(corr_range, distance=100)
    max_corr = down_period * (corr_range.index(max(corr_range[:200])) + 100)
    print(f'Max autocorrelation at {max_corr} year period')


    # Downsampled for input to matlab- this will be out actual CWT test
    # as the defauly python ones are lacking
    records['maw_3_down'].to_csv('internal_excel_sheets/filled_seb_runs/' +
                                  'MAW-3-downsample.csv', index=False)
    
    # Let's try to get the CWT in python!
    # It doesn't work :(
    # widths = np.geomspace(128, 4096, num=100)
    coefs, freqs = pywt.cwt(records['maw_3_down']['d18O'],
                            scales=np.geomspace(1, 20, num=2000),
                            wavelet="mexh", sampling_period=down_period)
    cwtmatr = np.abs(coefs[:-1, :-1])
    fig = plt.figure()
    ax = plt.gca()
    pcm = ax.pcolormesh(records['maw_3_down']['age_BP'], 1/freqs, cwtmatr)
    # ax.set_yscale("log")
    ax.set_xlabel("Age BP")
    ax.set_ylabel("Period (Years)")
    ax.set_title("Continuous Wavelet Transform (Scaleogram)")
    fig.colorbar(pcm, ax=ax)

if __name__ == '__main__':
    main()
