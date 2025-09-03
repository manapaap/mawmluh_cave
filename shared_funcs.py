# -*- coding: utf-8 -*-
"""
Shared FUnction files
"""

import pandas as pd
import warnings
from pandas.core.generic import SettingWithCopyWarning
from scipy.optimize import curve_fit
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def load_data(filter_year='46000'):
    """
    Loats MAW-3 record (and eventually CH1 too!)

    Time to add in  CH1!
    """
    ch1_proxy = pd.read_csv('internal_excel_sheets/filled_seb_runs/' +
                            'CH1-filled-AGES.csv')

    # Greenland NGRIP Data to compare trends
    # https://www.iceandclimate.nbi.ku.dk/data/
    ngrip_data = pd.read_excel('external_excel_sheets/NGRIP_d18O.xls',
                               sheet_name='NGRIP-2 d18O and Dust',
                               names=['depth_mm', 'd18O', 'dust',
                                      'age_BP', 'age_error'])

    # vostok_data = pd.read_excel('external_excel_sheets/totalvosdata.xls',
    #                             sheet_name='Vostok',
    #                             skiprows=1,
    #                             names=['depth_m', 'ice_age_ka', 'd18O', 
    #                                    'dust', 'gas_age_ka', 'CO2', 'CH4'])
    # vostok_data['age_BP'] = vostok_data['ice_age_ka'] * 1000

    # ball_gown = pd.read_csv('external_excel_sheets/' +
    #                         'ball_gown_cave.txt',
    #                          skiprows=145,
    #                          names=['stalagmite', 'depth_mm', 'age_BP',
    #                                 'd18O', 'd13C'],
    #                          sep='	')
    # ball_gown.sort_values(by='age_BP', inplace=True)
    # ball_gown.reset_index(inplace=True, drop=True)

    # dome_fuji = pd.read_excel('external_excel_sheets/df2012isotope-temperature.xls',
    #                            sheet_name='DF1 Isotopes',
    #                            skiprows=11,
    #                            names=['ID', 'top_m', 'bottom_m', 'center_m',
    #                                   'age_ka', 'd18O', 'dD', 'd_excess'])
    # dome_fuji['age_BP'] = dome_fuji['age_ka'] * 1000
    # dome_fuji['age_BP'] = dome_fuji['age_BP'] - 50

    wais = pd.read_excel('external_excel_sheets/wais_data.xls',
                         sheet_name='WDC d18O',
                                skiprows=48,
                                names=['tube_num', 'depth_m', 'depth_b_m',
                                       'd18O', 'age_top', 'age_bottom'])
    # Remove some weird outliers
    wais = wais.query('d18O < 1000')
    
    with warnings.catch_warnings():
        # Throwing up a BS error
        warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
        wais['age_BP'] = 1000 * ((wais['age_top'] + wais['age_bottom']) / 2)

    # epica_t = pd.read_csv('external_excel_sheets/buizert2021edc-temp.txt',
    #                       skiprows=122, sep='	',
    #                       names=['age_BP', 't_anom', 't_high', 't_low'])

    # Old hulu data- let's try to link the unpublished record with this too
    hulu_old = pd.DataFrame()
    # Need to do bullshit for hulu cave as the file is badly formatted
    age = []
    d18O = []
    with open('external_excel_sheets/hulu_cave_d18O.txt') as file:
        for count, line in enumerate(file.readlines()):
            if count == 0:
                continue
            age.append(line[15:20].strip())
            d18O.append(line[25:].strip())

    hulu_old = pd.DataFrame({'age_BP': age,
                              'd18O': d18O})
    hulu_old['d18O'] = pd.to_numeric(hulu_old['d18O'])
    hulu_old['age_BP'] = pd.to_numeric(hulu_old['age_BP'])
    hulu_old.sort_values(by='age_BP', inplace=True)
    
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
    # Remove NaN values
    hulu_data = hulu_data[~hulu_data['d18O'].isnull()].reset_index()
    
    # Arabian sediment record
    arabia = pd.read_csv('external_excel_sheets/arabian_sediment.txt',
                         skiprows=111,
                         names=['depth_m', 'age_2000', 'refl'], sep='\t')
    arabia['age_BP'] = arabia['age_2000'] - 50

    # maw_3_clean = maw_3_proxy.query(f'age_BP <= {filter_year}')
    # maw_3_clean = maw_3_clean.dropna(subset=['age_BP', 'd18O', 'd13C'])
    
    
    # Let's insert the new MAW-3 data here- need to combine d18O and d13C sheets
    maw_3_d18O = pd.read_excel('internal_excel_sheets/filled_seb_runs/' +
                              'MAW-3_am11_copra.xlsx', usecols='B:C,K', 
                              names=['top_dist_mm', 'age_BP', 'd18O'], 
                              sheet_name='d18O final',
                              skiprows=68)
    maw_3_d13C = pd.read_excel('internal_excel_sheets/filled_seb_runs/' +
                              'MAW-3_am11_copra.xlsx', usecols='B:C,F,K', 
                              names=['top_dist_mm', 'age_BP', 'growth_rate','d13C'], 
                              sheet_name='d13C final am11',
                              skiprows=68)
    maw_3_clean = maw_3_d18O.merge(maw_3_d13C, on='age_BP', how='left')
    maw_3_clean = maw_3_clean.rename(columns={'top_dist_mm_x': 'top_dist_mm'}).\
            drop(columns='top_dist_mm_y')
      
    # "tidy" this and focus on good data region
    maw_3_clean = maw_3_clean.query(f'age_BP <= {filter_year}')
    # Include the Jaglan et. al paper from 2021
    maw_jag_old = pd.read_excel('external_excel_sheets/maw_jagalan.xlsx', 
                            names=['top_dist_mm', 'age_BP', 'd18O', 'd13C'],
                            sheet_name='Depth, Age, O & C isotope data')
    
    maw_jag_d18O = pd.read_excel('internal_excel_sheets/Jaglan_Maw_am1_copra.xlsx', 
                            usecols='B:C,K',names=['top_dist_mm', 'age_BP', 'd18O'], 
                            sheet_name='d18O final',
                            skiprows=2)
    maw_jag_d13C = pd.read_excel('internal_excel_sheets/Jaglan_Maw_am1_copra.xlsx', 
                            usecols='B:C,K',names=['top_dist_mm', 'age_BP', 'd13C'], 
                            sheet_name='d13C final',
                            skiprows=2)
    maw_jag = maw_jag_d18O.merge(maw_jag_d13C, on='age_BP', how='left')
    maw_jag = maw_jag.rename(columns={'top_dist_mm_x': 'top_dist_mm'}).\
        drop(columns='top_dist_mm_y')
    
    # Yemen, 2003
    # socotra = pd.read_excel('external_excel_sheets/moomi2003.xls', 
    #                         names=['top_dist_mm', 'd18O', 'age_BP'],
    #                         sheet_name='M1-2 d18O', skiprows=6)
    # Remove NaN values
    #socotra = socotra[~socotra['d18O'].isnull()].reset_index()
    
    # Turkey, 2024 {conversion from kyr to yr}
    sofular = pd.read_excel('external_excel_sheets/turkey_nature_2024.xlsx',
                            sheet_name='Fig 3a', skiprows=4, usecols='E:F',
                            names=['age_BP', 'd13C']).dropna()
    sofular['age_BP'] *= 1000
    
    # WAIS CO2 https://www.pnas.org/doi/10.1073/pnas.2319652121#supplementary-materials
    # for H4 plot
    wais_co2_new = pd.read_excel('external_excel_sheets/wais_co2_1.xlsx',
                             skiprows=2, usecols='D,F',
                             names=['age_BP', 'CO2'], 
                             sheet_name='Sheet1').dropna(how='any')
    wais_co2_stack = pd.read_excel('external_excel_sheets/wais_co2_2.xlsx', 
                               skiprows=2, usecols='B:C',
                               names=['depth', 'age_BP', 'CO2'], 
                               sheet_name='WDC').dropna(how='any')
    wais_co2 = pd.concat([wais_co2_new, wais_co2_stack]).\
        sort_values(by='age_BP').reset_index(drop=True)
    
    # Brazil H4 https://www.sciencedirect.com/science/article/pii/S0012821X18307404#se0140
    brazil_sheets = ['barri', 'boa_4', 'boa_6']
    for n, sheet in enumerate(brazil_sheets):
        brazil_sheets[n] = pd.read_excel('internal_excel_sheets/brazil_stals_clean.xlsx',
                                         sheet_name=sheet)
    brazil_stack = pd.concat(brazil_sheets).dropna(how='any').\
        sort_values(by='age_BP').reset_index(drop=True)[1:]
    
    records = {'maw_3_clean': maw_3_clean,
               'ch1_proxy': ch1_proxy,
               'maw_jag': maw_jag,
               'maw_jag_old': maw_jag_old,
               'ngrip': ngrip_data,
               'hulu': hulu_data,
               'wais': wais,
               'arabia': arabia,
               'hulu_old': hulu_old,
               'sofular': sofular,
               'wais_co2': wais_co2,
               'ne_brazil': brazil_stack}

    return records


def combine_mawmluh(records, cutoff=40000):
    """
    Combines the MAW-3 record with the Jaglan record at a defined cutoff year
    """
    maw_sliced = records['maw_3_clean'].query(f'age_BP < {cutoff}')
    jag_sliced = records['maw_jag'].query(f'age_BP > {cutoff}')
    
    combined = pd.concat([maw_sliced, jag_sliced]).drop(['top_dist_mm'], axis=1)
    return combined.sort_values('age_BP').reset_index(drop=True)


def add_hiatus(dataarray, tol=50, varz=['d18O', 'd13C']):
    """
    Adds hiatus periods to d18O and d13C, setting values to np.nan 
    as defined by tolerance. Changes this for variables in varz
    """
    for n, (age_1, age_2) in enumerate(zip(dataarray.age_BP, dataarray.age_BP[1:])):
        diff = abs(age_2 - age_1)
        if diff > tol:
            for var in varz:
                dataarray[var][n] = np.nan
    return dataarray  


def red_power(f, autocorr, A):
    """
    Function for red noise spectrum. Will be fit to PSD. 
    """ 
    rs = A * (1.0 - autocorr**2) /(1. -(2.0 * autocorr * np.cos(f *2.0 * np.pi)) + autocorr**2)
    return rs


def plot_psd(array, nperseg=128, sig=0.95, cutoff=0.005, 
             period=20, nfft=256,name='Mawmluh'):
    """
    Plots the psd and red noise null hypothesis to check for significant peaks
    under "cutoff"
    
    Returns relevant parameters to reconstruct the figure
    """    
    f, Pxx = signal.welch(array , fs=1/period, nperseg=nperseg, nfft=nfft)
    Pxx /= Pxx.mean()
    # cut off the high frequency bs
    Pxx = Pxx[f < cutoff]
    f = f[f < cutoff]
    # Get lag-1 autocorr to fit the red noise
    corr = signal.correlate(array, array, mode='full')[array.shape[0]:] /\
        (np.var(array) * len(array))
    corr /= np.max(corr)
    corr_1 = float(corr[1])
    # Fit the red noise function
    red_params, red_covar = curve_fit(red_power, f, Pxx, p0=(corr_1, 1))
    red_fitted = red_power(f, *red_params)
    # Calculate f-value
    n_df = 2 * len(array) / nperseg
    m_df = len(array) / 2
    f_stat = stats.f.ppf(sig, n_df, m_df)
    # Plot the PSD
    # This way to extract the peak assumes only one sig peak but that is OK
    # Reworkng this
    plt.figure()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        num_peaks = len(f[Pxx > (f_stat * red_fitted)])
        peaks = f[Pxx > (f_stat * red_fitted)]
    if num_peaks == 0:
        plt.plot(f, Pxx, label=f'{name}, Not Significant')
        plt.text(0.21, np.max(red_fitted), 'No significant freqs')
    else:
        plt.plot(f, Pxx, label=f'{name}')
        plt.text(0.21, np.max(red_fitted), 'Significant freqs:')
        for n, peak in enumerate(peaks):
            plt.text(0.22, np.max(red_fitted) - 0.5 * (n + 1), 
                     f'{1/peak:.3f} years')
    plt.plot(f, red_fitted, label='Fitted Red Noise')
    plt.plot(f, f_stat * red_fitted, label=f'{sig} Significance')
    plt.grid()
    plt.ylabel('Power')
    plt.xlabel('Frequency (Cycles / year)')
    plt.legend()
    plt.title(f'Power Spectrum of {name}')
    
    spectral_params = {'f': f,
                       'Pxx': Pxx,
                       'red_fit': red_fitted,
                       'f_stat': f_stat}
    return spectral_params


def d_o_dates(min_date=27000, max_date=60000):
    """
    Source: 
        https://cp.copernicus.org/articles/18/2021/2022/#section5
    
    Uses the webplotdigitizer dates to return dictionaries containing the 
    d_o event dates. REturns a 3-tuple:
    
    d_o_all: every single d_o evnet, 1 through 20
    
    d_o_events: every d_o event within maw-3, 3 thtough 11

    d_o_ng: eveny d_o event within ngrip: 1 through 17
    """

    # Raw output from webplotdigntizer
    d_o_all = {17.2: 59457,
               17.1: 59073,
               16.2: 58284,
               16.1: 58044, 
               15.2: 55810, 
               15.1: 55001,
               14: 54223, 
               13: 49305, 
               12: 46885,
               11: 43359, 
               10: 41462,
               9: 40139, 
               8: 38223, 
               7: 35500, 
               6: 33740, 
               5.2: 32223, 
               5.1: 30782,
               4: 28908, 
               3: 27782, 
               2: 23364, 
               1: 14689}

    # Round to nearest ten because this is uncertain + literal plot digitizer
    d_o_all = {event: round(year, -1) for event, year in d_o_all.items()}

    # Capture events corredponding to maw-3 (4-11), reverse order
    d_o_events = list(range(11, 2, -1))
    d_o_events = {event: year for event, year in d_o_all.items() if 4 <= event <= 12}

    # do this to get scotra as well
    d_o_ng = {event: year for event, year in zip(range(20, 0, -1),
                                                 d_o_all.values()) if
              year > 42000 and year < 53000}

    return d_o_all, d_o_events, d_o_ng


def heinrich_dates():
    """
    Returns approximate dates for H3 H4 so they can be plotted
    """
    return {3: 29500, 4: 39000}
    
    
def resolution(time_axis, min_age=28351.3258, max_age=47359.5933):
    """
    Prints statistics of record resolution during some specified min/max age
    range
    
    Default min/max range is for MAW_COMB age range
    """
    analysis = time_axis.query(f'{min_age} < age_BP < {max_age}')
    resolution_array = np.full(len(analysis) - 1, np.nan)
    
    resolution = -(min_age - max_age) / len(analysis)
    
    for n, (x, y) in enumerate(zip(analysis['age_BP'], analysis['age_BP'][1:])):
        # Drop Hiatuses
        res = y - x
        if res > 100:
            continue
        resolution_array[n] = res
        
    resolution_array = resolution_array[~np.isnan(resolution_array)]
    print(f'Mean Resolution: {resolution:.2f}')
    print(f'Loop Resolution: {resolution_array.mean():.2f}')
    print(f'Resolution stdev: {resolution_array.std():.2f}')
    print(f'Median Resolution: {np.median(resolution_array):.2f}')
    print(f'Min Resolution: {np.min(resolution_array):.2f}')
    print(f'Max Resolution: {np.max(resolution_array):.2f}')
    
    
    
    
    
    
    
    
    
    
    