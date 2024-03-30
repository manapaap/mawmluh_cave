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


# chdir('C:/Users/Aakas/Documents/School/Oster_lab/programs')
chdir('C:/Users/Aakas/Documents/School/Oster_lab/')


def load_data(filter_year='46000'):
    """
    Loats MAW-3 record (and eventually CH1 too!)

    Time to add in  CH1!
    """
    maw_3_proxy = pd.read_csv('internal_excel_sheets/filled_seb_runs/' +
                              'MAW-3-filled-AGES.csv')
    ch1_proxy = pd.read_csv('internal_excel_sheets/filled_seb_runs/' +
                            'CH1-filled-AGES.csv')

    # Greenland NGRIP Data to compare trends
    # https://www.iceandclimate.nbi.ku.dk/data/
    ngrip_data = pd.read_excel('external_excel_sheets/NGRIP_d18O.xls',
                               sheet_name='NGRIP-2 d18O and Dust',
                               names=['depth_mm', 'd18O', 'dust',
                                      'age_BP', 'age_error'])

    vostok_data = pd.read_excel('external_excel_sheets/totalvosdata.xls',
                                sheet_name='Vostok',
                                skiprows=1,
                                names=['depth_m', 'ice_age_ka', 'd18O', 
                                       'dust', 'gas_age_ka', 'CO2', 'CH4'])
    vostok_data['age_BP'] = vostok_data['ice_age_ka'] * 1000

    ball_gown = pd.read_csv('external_excel_sheets/' +
                            'ball_gown_cave.txt',
                             skiprows=145,
                             names=['stalagmite', 'depth_mm', 'age_BP',
                                    'd18O', 'd13C'],
                             sep='	')
    ball_gown.sort_values(by='age_BP', inplace=True)
    ball_gown.reset_index(inplace=True, drop=True)

    dome_fuji = pd.read_excel('external_excel_sheets/df2012isotope-temperature.xls',
                               sheet_name='DF1 Isotopes',
                               skiprows=11,
                               names=['ID', 'top_m', 'bottom_m', 'center_m',
                                      'age_ka', 'd18O', 'dD', 'd_excess'])
    dome_fuji['age_BP'] = dome_fuji['age_ka'] * 1000
    dome_fuji['age_BP'] = dome_fuji['age_BP'] - 50

    wais = pd.read_excel('external_excel_sheets/wais_data.xls',
                         sheet_name='WDC d18O',
                                skiprows=48,
                                names=['tube_num', 'depth_m', 'depth_b_m',
                                       'd18O', 'age_top', 'age_bottom'])
    # Remove some weird outliers
    wais = wais.query('d18O < 1000')
    wais['age_BP'] = 1000 * ((wais['age_top'] + wais['age_bottom']) / 2)

    epica_t = pd.read_csv('external_excel_sheets/buizert2021edc-temp.txt',
                          skiprows=122, sep='	',
                          names=['age_BP', 't_anom', 't_high', 't_low'])

    # Old hulu data- let's try to link the unpublished record with this too
    # hulu_old = pd.DataFrame()
    # # Need to do bullshit for hulu cave as the file is badly formatted
    # age = []
    # d18O = []
    # with open('external_excel_sheets/hulu_cave_d18O.txt') as file:
    #     for count, line in enumerate(file.readlines()):
    #         if count == 0:
    #             continue
    #         age.append(line[15:20].strip())
    #         d18O.append(line[25:].strip())

    # hulu_old = pd.DataFrame({'age_BP': age,
    #                           'd18O': d18O})
    # hulu_old['d18O'] = pd.to_numeric(hulu_old['d18O'])
    # hulu_old['age_BP'] = pd.to_numeric(hulu_old['age_BP'])
    # hulu_old.sort_values(by='age_BP', inplace=True)
    
    # Hulu has multiple speleothems that must be combined together
    hulu_msl = pd.read_excel('external_excel_sheets/hulu_unpublished.xlsx', 
                             skiprows=1, usecols='B,F', 
                             names=['age_BP', 'd18O'], sheet_name='MSL')
    hulu_msd = pd.read_excel('external_excel_sheets/hulu_unpublished.xlsx', 
                             skiprows=1, usecols='B,F', 
                             names=['age_BP', 'd18O'], sheet_name='MSD')
    hulu_h82 = pd.read_excel('external_excel_sheets/hulu_unpublished.xlsx', 
                             skiprows=1, usecols='B,F', 
                             names=['age_BP', 'd18O'], sheet_name='H82')
    hulu_data = pd.concat([hulu_msl, hulu_msd,
                           hulu_h82]).sort_values(by='age_BP')
    
    # Arabian sediment record
    arabia = pd.read_csv('external_excel_sheets/arabian_sediment.txt',
                         skiprows=111,
                         names=['depth_m', 'age_2000', 'refl'], sep='\t')
    arabia['age_BP'] = arabia['age_2000'] - 50

    # maw_3_clean = maw_3_proxy.query(f'age_BP <= {filter_year}')
    # maw_3_clean = maw_3_clean.dropna(subset=['age_BP', 'd18O', 'd13C'])
    
    
    # Let's insert the new MAW-3 data here- need to combine d18O and d13C sheets
    maw_3_d18O = pd.read_excel('internal_excel_sheets/filled_seb_runs/' +
                              'MAW-3_am10_copra.xlsx', usecols='B:C,K', 
                              names=['top_dist_mm', 'age_BP', 'd18O'], 
                              sheet_name='d18O final',
                              skiprows=68)
    maw_3_d13C = pd.read_excel('internal_excel_sheets/filled_seb_runs/' +
                              'MAW-3_am10_copra.xlsx', usecols='B:C,K', 
                              names=['top_dist_mm', 'age_BP', 'd13C'], 
                              sheet_name='d13C final',
                              skiprows=68)
    maw_3_clean = maw_3_d18O.merge(maw_3_d13C, on='age_BP', how='left')
    
    # "tidy" this and focus on good data region
    maw_3_clean = maw_3_clean.query(f'age_BP <= {filter_year}')
    # Remove the duplicates it is finding
    # maw_3_clean.drop_duplicates(subset='age_BP', inplace=True)
    
    # Include the Jaglan et. al paper from 2021
    maw_jag = pd.read_excel('external_excel_sheets/maw_jagalan.xlsx', 
                            names=['top_dist_mm', 'age_BP', 'd18O', 'd13C'],
                            sheet_name='Depth, Age, O & C isotope data')

    records = {'maw_3_clean': maw_3_clean,
               'ch1_proxy': ch1_proxy,
               'maw_jag': maw_jag,
               'ngrip': ngrip_data,
               'hulu': hulu_data,
               'vostok': vostok_data,
               'ball_gown': ball_gown,
               'dome_fuji': dome_fuji,
               'wais': wais,
               'arabia': arabia,
               'epica_t': epica_t}

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


def compare_stals(records_clean):
    """
    Plots the data from CH1 and MAW-3 side by side to allow for comparision
    """
    ch1_data = records_clean.query('stal == "CH1"')
    ch1_max = ch1_data['age_BP'].max()
    ch1_min = ch1_data['age_BP'].min()

    maw_3_data = records_clean.query('stal == "MAW-3"')
    maw_3_data = maw_3_data.query(f'age_BP > {ch1_min} & age_BP < {ch1_max}')

    # Use ax1 for d18O and ax2 for d13C
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(maw_3_data['age_BP'], maw_3_data['d18O'], color='blue',
             label='MAW-3', alpha=0.5)
    ax1.scatter(ch1_data['age_BP'], ch1_data['d18O'], color='red',
                label='CH1', s=1)
    ax1.legend(loc='upper right')

    ax2.plot(maw_3_data['age_BP'], maw_3_data['d13C'], color='green',
             label='MAW-3', alpha=0.5)
    ax2.scatter(ch1_data['age_BP'], ch1_data['d13C'], color='red',
                label='CH1', s=1)
    ax2.legend(loc='upper right')

    ax1.set_ylabel('δ¹⁸O (‰)')
    ax2.set_ylabel('δ¹³C (‰)')
    ax1.set_title('Stable Isotope Information from Different Speleothems')
    ax2.set_xlabel('Age BP')
    plt.grid()
    ax1.grid()


def compare_stals_2(records_clean):
    """
    Plots the data from CH1 and MAW-3 in a staaggered manner for comparision
    """
    ch1_data = records_clean.query('stal == "CH1"')
    fig, ax = plt.subplots(4, 1, sharex=True)
    fig.subplots_adjust(hspace=0)
    plt.tight_layout()

    ch1_max = ch1_data['age_BP'].max()
    ch1_min = ch1_data['age_BP'].min()

    maw_3_data = records_clean.query('stal == "MAW-3"')
    maw_3_data = maw_3_data.query(f'age_BP > {ch1_min} & age_BP < {ch1_max}')

    color1 = plt.cm.viridis(0.5)
    color2 = plt.cm.viridis(0)
    color3 = plt.cm.viridis(0.7)
    color4 = plt.cm.viridis(0.95)

    ax[0].plot(maw_3_data['age_BP'], maw_3_data['d18O'],
               label='MAW-3 d18O', color=color1)
    ax[0].set_ylim(-4, 0)
    ax[0].grid()
    ax[0].set_ylabel('MAW-3 δ¹⁸O')
    ax[0].set_yticks(np.arange(-4, 0, 1))

    ax[1].plot(ch1_data['age_BP'], ch1_data['d18O'],
               label='CH1 d18O', color=color2)
    ax[1].grid()
    ax[1].set_ylim(-4, 0)
    ax[1].set_ylabel('CH1 δ¹⁸O')
    ax[1].set_yticks(np.arange(-4, 0, 1))

    ax[2].plot(maw_3_data['age_BP'], maw_3_data['d13C'],
               label='CH1 d18O', color=color3)
    ax[2].grid()
    ax[2].set_ylabel('MAW-3 δ¹³C')
    ax[2].set_ylim(-1, 6)
    ax[2].set_yticks(np.arange(-1, 6, 2))

    ax[3].plot(ch1_data['age_BP'], ch1_data['d13C'],
               label='CH1 d18O', color=color4)
    ax[3].grid()
    ax[3].set_ylabel('CH1 δ¹³C')
    ax[3].set_xlabel('Age (Years BP)')
    ax[3].set_ylim(-1, 6)
    ax[3].set_yticks(np.arange(-1, 6, 2))

    ax[0].set_title('Comparison of CH1 and MAW-3')
    plt.show()


def main():
    global records
    down_period = 20

    records = load_data(filter_year='46000')

    
    # Remove this damn outlier (handled by seb already)
    # records['maw_3_clean'].drop(5651, inplace=True)
    records['maw_3_down'] = downsample(records['maw_3_clean'], down_period)
    # records['maw_3_down']['d18O'] = signal.detrend(records['maw_3_down']['d18O'])

    # Let's plot a power spectral density
    # fourier(records['maw_3_down'], proxy='d18O', fig=1, fs=down_period)
    # fourier(records['maw_3_down'], proxy='d13C', fig=1, fs=down_period)

    # Pandas autocorrelation plots- d13C has no interesting signal
    # but d18O does
    pd.plotting.autocorrelation_plot(records['maw_3_down']['d18O'])
    plt.title('Autocorrelation of d18O')

    # plt.figure(2)
    # pd.plotting.autocorrelation_plot(records['maw_3_down']['d13C'])
    # plt.title('Autocorrelation of d13C')

    # Let's try to find the actual peak
    corr_range = [records['maw_3_down']['d18O'].autocorr(lag) for lag in range(100, 600)]
    peaks, _ = find_peaks(corr_range, distance=100)
    max_corr = down_period * (corr_range.index(max(corr_range[:200])) + 100)
    print(f'Max autocorrelation at {max_corr} year period')

    # Comapre to NGRIP
    compare_records(records['maw_3_clean'], records['ngrip'], records['hulu'],
                    how='north', since=46000)

    # Compare to South Pole
    compare_records(records['maw_3_clean'], records['wais'],
                    records['arabia'], how='south', since=46000,
                    prox='refl')

    # Downsampled for input to matlab- this will be out actual CWT test
    # as the defauly python ones are lacking
    records['maw_3_down'].to_csv('internal_excel_sheets/filled_seb_runs/' +
                                  'MAW-3-downsample.csv', index=False)
    
    records['maw_3_clean'].to_csv('internal_excel_sheets/filled_seb_runs/' +
                                  'MAW-3-clean.csv', index=False)


if __name__ == '__main__':
    main()
