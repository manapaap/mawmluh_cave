# -*- coding: utf-8 -*-
"""
testing cross correlation between MAW-3 and other records
"""

from scipy.interpolate import interp1d
import numpy as np
import scipy.signal as signal
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from os import chdir
from scipy.optimize import curve_fit

chdir('C:/Users/Aakas/Documents/School/Oster_lab/programs')

from freq_analysis import load_data

chdir('C:/Users/Aakas/Documents/School/Oster_lab/')


def d_o_dates(min_date=27000, max_date=60000):
    """
    Source: 
        https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2000pa000571
    
    Uses the webplotdigitizer dates to return dictionaries containing the 
    d_o event dates. REturns a 3-tuple:
    
    d_o_all: every single d_o evnet, 1 through 20
    
    d_o_events: every d_o event within maw-3, 3 thtough 11

    d_o_ng: eveny d_o event within ngrip: 1 through 17
    """

    # Raw output from webplotdigntizer
    d_o_all = np.array([72893.83561643836, -36.59294117647053,
                        68570.20547945207, -36.338823529411705,
                        61806.506849315076, -38.004705882352894,
                        57696.91780821918, -36.70588235294112,
                        56498.28767123288, -37.355294117647006,
                        53672.94520547946, -37.52470588235289,
                        51703.767123287675, -36.367058823529355,
                        46909.24657534247, -38.371764705882306,
                        45418.53424657534, -36.621176470588175,
                        43597.57534246575, -37.52470588235289,
                        41130.13698630137, -37.77882352941172,
                        40145.54794520548, -38.56941176470584,
                         38176.3698630137, -36.9317647058823,
                         35366.17808219178, -38.032941176470544,
                         33690.08219178082, -37.4682352941176,
                         32367.219178082192, -37.55294117647054,
                         28886.986301369863, -37.14352941176465,
                         27773.972602739726, -36.88941176470583,
                         23364.72602739726, -38.20235294117643,
                         14546.232876712327, -34.87058823529404])

    # Extract the year events
    d_o_all = d_o_all[d_o_all > 0]
    # Round to nearest hundred because this is uncertain + literal plot digitizer
    d_o_all = (d_o_all / 100).astype(int) * 100

    # Capture events corredponding to maw-3 (3-11), reverse order
    d_o_events = list(range(11, 2, -1))
    # Capture the corredponding years (shifted due to zero-incexing, 
    # flip for order, flip back to start from 11)
    d_o_years = d_o_all[::-1][list(range(2, 11))][::-1]
    d_o_events = {event: year for event, year in zip(d_o_events, d_o_years)}

    # But the full events into a dict form too
    d_o_all = {event: year for event, year in zip(range(20, 0, -1), d_o_all)}

    # do this to get ngrip as well
    d_o_ng = {event: year for event, year in zip(range(20, 0, -1),
                                                 d_o_all.values()) if
              year > min_date and year < max_date}

    return d_o_all, d_o_events, d_o_ng


def smooth_full_ice(ngrip, wais, arab, min_age, max_age, period=25, fil_freq=1/500):
    """
    Interpolates wais and ngrip to period and puts through a low pass filter 
    to remove noise

    returns pandas df containing the full arrays. The periods of intrest
    to put min/max age are:

    - all ngrip era as wais spans a longer time span (but has lower fs)
    - youngest-2000 until oldest ngrip record (d-o events continue past ngrip)
    - only during MIS 3 (25,000 to end of ngrip)
    """
    # Flip NGRIP for phase consistency with wais
    ngrip = deepcopy(ngrip)
    mean = ngrip['d18O'].mean()
    ngrip['d18O'] *= -1
    ngrip['d18O'] += 2 * mean

    # Basically copying code from the 3-array version of this
    func_ngrip = interp1d(ngrip['age_BP'], ngrip['d18O'])
    func_wais = interp1d(wais['age_BP'], wais['d18O'])
    func_arab = interp1d(arab['age_BP'], arab['refl'])

    ages = np.arange(min_age, max_age, period)

    ngrip = ngrip.query(f'{min_age} < age_BP < {max_age}')
    wais = wais.query(f'{min_age} < age_BP < {max_age}')
    arab = arab.query(f'{min_age} < age_BP < {max_age}')

    new_ngrip = func_ngrip(ages)
    new_wais = func_wais(ages)
    new_arab = func_arab(ages)

    # Pass through a low pass filter now
    new_ngrip = butter_lowpass_filter(new_ngrip, fil_freq, 1/period, 2)
    new_wais = butter_lowpass_filter(new_wais, fil_freq, 1/period, 2)
    new_arab = butter_lowpass_filter(new_arab, fil_freq, 1/period, 2)

    # Compare to the original record
    fix, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    plt.suptitle('Original vs Interpolated Ice Records')

    ax1.set_ylabel('NGRIP (Inverted)')
    ax1.plot(ngrip['age_BP'], ngrip['d18O'], color='lightgreen')
    ax1.plot(ages, new_ngrip, color='blue', alpha=0.5)
    ax1.set_xlim(min_age, max_age)
    ax1.grid()

    ax2.set_ylabel('WAIS')
    ax2.plot(wais['age_BP'], wais['d18O'], color='lightgreen')
    ax2.plot(ages, new_wais, color='blue', alpha=0.5)
    ax2.set_xlim(min_age, max_age)
    ax2.grid()
    
    ax3.set_ylabel('Arabia')
    ax3.set_xlabel('Age BP')
    ax3.plot(arab['age_BP'], arab['refl'], color='lightgreen')
    ax3.plot(ages, new_arab, color='blue', alpha=0.5)
    ax3.set_xlim(min_age, max_age)
    ax3.grid()

    # Normalize our arrays
    new_ngrip = (new_ngrip - np.mean(new_ngrip)) / np.std(new_ngrip)
    new_wais = (new_wais - np.mean(new_wais)) / np.std(new_wais)
    new_arab = (new_arab - np.mean(new_arab)) / np.std(new_arab)

    # Check if detrend in a second- will be more complicated
    new_ngrip = signal.detrend(new_ngrip)
    new_wais = signal.detrend(new_wais)
    new_arab = signal.detrend(new_arab)

    data = pd.DataFrame({'age_BP': ages,
                         'ngrip': new_ngrip,
                         'wais': new_wais,
                         'arab': new_arab})

    return data
    

def plot_cross_corr(arrs, title='', period=25, filter_freq=1/500, plot=True):
    """
    Interpolates to even spacing and puts three signals through a low-pass
    filter to remove noise
    
    plots the original and interpolated signals for continuity
    """
    arr1, arr2, arr3, arr4 = arrs
    # Flip NGRIP for consistency in phases
    arr2 = deepcopy(arr2)
    mean = arr2['d18O'].mean()
    arr2['d18O'] *= -1
    arr2['d18O'] += 2 * mean

    func_arr1 = interp1d(arr1['age_BP'], arr1['d18O'])
    func_arr2 = interp1d(arr2['age_BP'], arr2['d18O'])
    func_arr3 = interp1d(arr3['age_BP'], arr3['d18O'])
    func_arr4 = interp1d(arr4['age_BP'], arr4['refl'])

    min_age, max_age = arr1['age_BP'].min(), arr1['age_BP'].max()
    ages = np.arange(min_age, max_age, period)
    min_age2, max_age2 = arr2['age_BP'].min(), arr2['age_BP'].max()
    ages_2 = np.arange(min_age2, max_age2, period)

    arr2 = arr2.query(f'{min_age} <= age_BP <= {max_age}')
    arr3 = arr3.query(f'{min_age} <= age_BP <= {max_age}')
    arr4 = arr4.query(f'{min_age} <= age_BP <= {max_age}')

    new_arr1 = func_arr1(ages)
    new_arr2 = func_arr2(ages)
    # new_arr2_full = func_arr2(ages_2) I don't actually use this?
    new_arr3 = func_arr3(ages)
    new_arr4 = func_arr4(ages)

    # Let's pass this through a low pass filter to remove noise
    new_arr1 = butter_lowpass_filter(new_arr1, filter_freq, 1/period, 8)
    new_arr2 = butter_lowpass_filter(new_arr2, filter_freq, 1/period, 8)
    new_arr3 = butter_lowpass_filter(new_arr3, filter_freq, 1/period, 8)
    new_arr4 = butter_lowpass_filter(new_arr4, filter_freq, 1/period, 8)

    # First, compare the interpolation
    if plot:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
        fig.suptitle('Original vs. Interpolated d18O Record')
    
        ax1.set_ylabel('MAW-3')
        ax1.plot(arr1['age_BP'], arr1['d18O'], color='lightgreen')
        ax1.plot(ages, new_arr1, color='blue', alpha=0.5)
        ax1.grid()
    
        ax2.set_ylabel('NGRIP (Flip)')
        ax2.plot(arr2['age_BP'], arr2['d18O'], color='lightgreen')
        ax2.plot(ages, new_arr2, color='blue', alpha=0.5)
        ax2.set_xlim(min_age, max_age)
        ax2.grid()
    
        ax3.set_ylabel('WAIS')
        # ax3.set_xlabel('Age BP')
        ax3.plot(arr3['age_BP'], arr3['d18O'], color='lightgreen')
        ax3.plot(ages, new_arr3, color='blue', alpha=0.5)
        ax3.set_xlim(min_age, max_age)
        ax3.grid()
        
        ax4.set_ylabel('Arab. Sed.')
        ax4.set_xlabel('Age BP')
        ax4.plot(arr4['age_BP'], arr4['refl'], color='lightgreen')
        ax4.plot(ages, new_arr4, color='blue', alpha=0.5)
        ax4.set_xlim(min_age, max_age)
        ax4.grid()

    # Normalize our arrays
    new_arr1 = (new_arr1 - np.mean(new_arr1)) / np.std(new_arr1)
    new_arr2 = (new_arr2 - np.mean(new_arr2)) / np.std(new_arr2)
    new_arr3 = (new_arr3 - np.mean(new_arr3)) / np.std(new_arr3)
    new_arr4 = (new_arr4 - np.mean(new_arr4)) / np.std(new_arr4)

    # Detrend our data
    new_arr1 = signal.detrend(new_arr1)
    new_arr2 = signal.detrend(new_arr2)
    new_arr3 = signal.detrend(new_arr3)
    new_arr4 = signal.detrend(new_arr4)

    data = pd.DataFrame({'age_BP': ages,
                         'maw': new_arr1,
                         'ngrip': new_arr2,
                         'wais': new_arr3,
                         'arabia': new_arr4})

    return data


def butter_lowpass_filter(data, cutoff, fs, order):
    """
    Lowpass filter
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y


def find_nearest(array, value):
    """
    Finds closest value in array to provided value, returns index
    """
    idx = (np.abs(array - value)).argmin()
    return idx


def lag_finder(y1, y2, period, title='', plot=True):
    """
    Finds the lag between two signals of equal sample rate
    """
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    n = y1.size
    corr = signal.correlate(y1, y2, mode='same')
    corr /= np.sqrt(signal.correlate(y1, y1, mode='same')[int(n/2)] *\
                    signal.correlate(y2, y2, mode='same')[int(n/2)])

    # Let's focus this on the 500 max lag period
    lags = period * signal.correlation_lags(n, n, mode='same')
    low_idx = find_nearest(lags, -500)
    high_idx = find_nearest(lags, 500)

    lags_cop = lags[low_idx:high_idx]
    corr_cop = corr[low_idx:high_idx]

    max_lag = corr.argmax()
    while max_lag < low_idx or max_lag > high_idx:
        corr[max_lag] = -10000
        max_lag = corr.argmax()
        
    lag_len = lags[max_lag]

    if plot:
        plt.figure(np.random.randint(0, 100000))
        plt.plot(lags_cop, corr_cop)
        plt.title(title)
        plt.xlabel('Years of Lag')
        plt.ylabel('Correlation')
        plt.grid()
        plt.ylim(min(corr_cop) - 0.05, 1)

        plt.vlines(lag_len, min(corr_cop) - 0.05, 1, color='red', linestyle='dashed')
        # print(f'Maximum correlation at {lag_len:.2f} years')
        plt.text(-900, 0.85, f'Max correlation \nat {lag_len:.2f} years')

    else:
        return lag_len


def lag_finder_full(y1, y2, period, plot=True):
    """
    Finds the lag between two signals of equal sample rate
    """
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    n = y1.size
    corr = signal.correlate(y1, y2, mode='same')
    lags = np.linspace(-0.5*n*period, 0.5*n*period, n)
    corr /= np.sqrt(signal.correlate(y1, y1, mode='same')[int(n/2)] *\
                    signal.correlate(y2, y2, mode='same')[int(n/2)])

    max_lag = corr.argmax()
    lag_len = lags[max_lag]

    if plot:
        plt.figure(np.random.randint(0, 100000))
        plt.plot(lags, corr)
        plt.title('Correlation between Signals')
        plt.xlabel('Years of Lag')
        plt.ylabel('Correlation')
        plt.grid()
        plt.ylim(min(corr) - 0.05, 1)

        plt.vlines(lag_len, min(corr) - 0.05, 1, color='red', linestyle='dashed')
        print(f'Maximum correlation at {lag_len:.2f} years')
        plt.text(-1500, 0.85, f'Max correlation \nat {lag_len:.2f} years')

    else:
        return lag_len


def epoch_anal(prox_data, d_o_events, period, overlap):
    """
    Does an epoch analysis of the proxy records by finding correlation 
    over the course of each D-O event
    """
    lag_df = pd.DataFrame(columns=['MAW-NGRIP', 'MAW-WAIS', 'NGRIP-WAIS'])

    for event, year in d_o_events.items():
        # Get starting year and handle exception for starting event
        if event + 1 in list(d_o_events.keys()):
            # 500 year overlap
            start = d_o_events[event + 1] + overlap
        else:
            start = prox_data['age_BP'].iloc[-1]
        # Get ending year and handle exception for ending event
        if event == min(list(d_o_events.keys())):
            end = prox_data['age_BP'].iloc[0]
        else:
            # Again, 500 year overlap when possible
            end = year - overlap

        # Get the actual indices we need
        idx_start = find_nearest(prox_data['age_BP'], start)
        idx_end = find_nearest(prox_data['age_BP'], end)

        rel_data = prox_data.iloc[idx_end: idx_start]
        # Seems to work to slice our indices as needed!
        # Let's calculate the lags
        lag_dict = {'MAW-NGRIP': lag_finder_full(rel_data.maw, rel_data.ngrip, 
                                            period, False),
                    'MAW-WAIS': lag_finder_full(rel_data.maw, rel_data.wais,
                                           period, False),
                    'NGRIP-WAIS': lag_finder_full(rel_data.ngrip, rel_data.wais,
                                             period, False)}
        lag_dict = pd.DataFrame(lag_dict, index=[event])
        lag_df = pd.concat([lag_df, lag_dict])

    return lag_df


def epoch_chunks(prox_data, d_o_events, overlap=500):
    """
    Just returns the chunked df rather than the actual lag stats
    """
    chunks = {event: None for event in d_o_events.keys()}

    for event, year in d_o_events.items():
        # Get starting year and handle exception for starting event
        if event + 1 in list(d_o_events.keys()):
            # 500 year overlap
            start = d_o_events[event + 1] + overlap
        else:
            start = prox_data['age_BP'].iloc[-1]
        # Get ending year and handle exception for ending event
        if event == min(list(d_o_events.keys())):
            end = prox_data['age_BP'].iloc[0]
        else:
            # Again, 500 year overlap when possible
            end = year - overlap

        # Get the actual indices we need
        idx_start = find_nearest(prox_data['age_BP'], start)
        idx_end = find_nearest(prox_data['age_BP'], end)

        rel_data = prox_data.iloc[idx_end: idx_start]

        chunks[event] = rel_data

    return chunks


def epoch_ice(ice_data, d_o_events, period, overlap):
    """
    Does the epoch analysis for the much simpler ice system where
    there's only two lags to compare between
    """
    lag_df = pd.DataFrame(columns=['NGRIP-WAIS', 'NGRIP-ARABIA', 'ARABIA-WAIS'])

    for event, year in d_o_events.items():
        # Get starting year and handle exception for starting event
        if event + 1 in list(d_o_events.keys()):
            # 500 year overlap
            start = d_o_events[event + 1] + overlap
        else:
            start = ice_data['age_BP'].iloc[-1]
        # Get ending year and handle exception for ending event
        if event == min(list(d_o_events.keys())):
            end = ice_data['age_BP'].iloc[0]
        else:
            # Again, 500 year overlap when possible
            end = year - overlap

        # Get the actual indices we need
        idx_start = find_nearest(ice_data['age_BP'], start)
        idx_end = find_nearest(ice_data['age_BP'], end)

        rel_data = ice_data.iloc[idx_end:idx_start]
        # Seems to work to slice our indices as needed!
        # Let's calculate the lags
        lag_dict = {'NGRIP-WAIS': lag_finder_full(rel_data.ngrip, 
                                                  rel_data.wais,
                                                  period, False),
                    'NGRIP-ARABIA': lag_finder_full(rel_data.ngrip, 
                                                    rel_data.arab,
                                                    period, False),
                    'ARABIA-WAIS': lag_finder_full(rel_data.arab, 
                                                    rel_data.wais,
                                                    period, False)}
        lag_dict = pd.DataFrame(lag_dict, index=[event])
        lag_df = pd.concat([lag_df, lag_dict])

    return lag_df


def print_stats(lag_df, title, perc=0.975, mode='95CI', ice=False):
    """
    Prints 95% CI for the various lags in the given data
    """
    means = lag_df.mean()
    if mode == '95CI':
        st_err = stats.t.ppf(perc, len(lag_df)) * lag_df.std() /\
            np.sqrt(len(lag_df) - 1)
    elif mode =='stdev':
        st_err = lag_df.std()
    else:
        lag_df_np = np.array(lag_df)
        means = lag_df[lag_df_np < 750].mean()
        st_err = lag_df[lag_df_np < 750].std()
    print('\n' + title)
    if not ice:
        print(f'MAW-NGRIP lag: {means["MAW-NGRIP"]:.0f} ± {st_err["MAW-NGRIP"]:.0f}')
        print(f'MAW-WAIS lag: {means["MAW-WAIS"]:.0f} ± {st_err["MAW-WAIS"]:.0f}')
    elif ice:
        print(f'NGRIP-ARABIA lag: {means["NGRIP-ARABIA"]:.0f} ± {st_err["NGRIP-ARABIA"]:.0f}')
        print(f'ARABIA-WAIS lag: {means["ARABIA-WAIS"]:.0f} ± {st_err["ARABIA-WAIS"]:.0f}')
    print(f'NGRIP-WAIS lag: {means["NGRIP-WAIS"]:.0f} ± {st_err["NGRIP-WAIS"]:.0f}')


def plot_chunk(data, title=''):
    """
    Plots the chunk of data to observe trends
    """
    years = np.linspace(-128 * 20 / 2, 128 * 20 / 2, 128)
    
    plt.figure()
    for x in data.columns[1:]:
        plt.plot(years, -data[x], label=x)
    plt.grid()
    plt.legend()
    plt.xlabel('Years From Dansgaard-Oeschger Event')
    plt.ylabel('Normalized Signal')
    plt.title(title)


def composite(d_o_events, chunks):
    """
    Creates a composite D-O event from all the individual D-O events
    """
    # Let's aggregate these chunks into one array
    for n, (event, chunk) in enumerate(chunks.items()):
        if n == 0:
            aggregate = chunk.copy()
        else:
            for col in aggregate.columns[1:]:
                aggregate[col] += chunk[col]
    aggregate['age_BP'] = np.arange(0, 20 * len(chunk), 20)
    
    return aggregate


def chunked_d_o(d_o_events, prox_data):
    """
    Organizes the continous D-O event record into a series of chunks, 
    index by the dictionary
    """
    # Let's create "chunks" of data around each D-O event
    chunks = {event: None for event in d_o_events.keys()}

    for event, year in d_o_events.items():
        start = year - 128 * 20 / 2
        end = year + 128 * 20 / 2
        
        chunk = prox_data.query(f'{start} < age_BP < {end}').reset_index(drop=True)
        chunks[event] = chunk
    # Not enough data...
    chunks.pop(4)
    chunks.pop(3)
    
    return chunks
    

def lag_corr_mat(data, period):
    """
    Loops over the data to create a confustion-esque matrix
    of time lags between various records
    """
    comp_mat = pd.DataFrame(index=data.columns[1:],
                            columns=data.columns[1:])
    for loc in comp_mat.columns:
        for loc2 in comp_mat.columns:
            comp_mat[loc][loc2] = lag_finder(data[loc], data[loc2],
                                                  period=period, plot=False)
    return comp_mat


def averaged_lag_mats(chunks, period):
    """
    Loops over the chunks of data and creates an "average" lag matrix along
    with associated standard error at 95% CI
    """
    # Let's now do this for every piece of the chunked analysis...
    lag_dict = {event: None for event in chunks.keys()}
    
    for n, (event, chunk) in enumerate(chunks.items()):
        lag_dict[event] = lag_corr_mat(chunks[event], period)
    
    # Let's assemble a numpy array for the purposes of this aggregation...
    stacked_arr = np.stack([mat.to_numpy(dtype=float) for mat in lag_dict.values()])
    mean_lag = stacked_arr.mean(axis=0)
    std_lag = stacked_arr.std(axis=0)
    se_lag = stats.t.ppf(0.975, stacked_arr.shape[0]) * std_lag / np.sqrt(stacked_arr.shape[0])
    # Get these back into pandas dataframes
    # Should be fine since shape and order are preserved when doing this
    mean_lag = pd.DataFrame(mean_lag, columns=chunks[10].columns[1:], 
                            index=chunks[10].columns[1:])
    se_lag = pd.DataFrame(se_lag, columns=chunks[10].columns[1:], 
                            index=chunks[10].columns[1:])

    return mean_lag, se_lag


def clean_lag_corr(comp_mean, comp_se):
    """
    Returns a lag-correlation matrix in the string format
    mean ± se for nicer printing
    """
    # Round off to nearest year
    comp_mean = comp_mean.round(0).astype(int).astype(str)
    comp_se = comp_se.round(0).astype(int).astype(str)
    
    for column in comp_mean.columns:
        comp_mean[column] = comp_mean[column] + ' ± ' + comp_se[column]
    
    return comp_mean


def red_power(f, autocorr, A):
    """
    Function for red noise spectrum. Will be fit to PSD. 
    """ 
    rs = A * (1.0 - autocorr**2) /(1. -(2.0 * autocorr * np.cos(f *2.0 * np.pi)) + autocorr**2)
    return rs


def plot_psd(data, record, nperseg=128, sig=0.95, cutoff=0.005, period=20, name='Mawmluh'):
    """
    Plots the psd and red noise null hypothesis to check for significant peaks
    under "cutoff"
    """
    global f, Pxx
    array = data[record]
    
    f, Pxx = signal.welch(array , fs=1/period, nperseg=256, nfft=256)
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
    max_sig = f[Pxx.argmax()]
    plt.figure()
    plt.plot(f, Pxx, label=f'{name}, max at {1/max_sig:.0f} years')
    plt.plot(f, red_fitted, label='Fitted Red Noise')
    plt.plot(f, f_stat * red_fitted, label=f'{sig} Significance')
    plt.grid()
    plt.ylabel('Power')
    plt.xlabel('Frequency (Cycles / year)')
    plt.legend()
    plt.title(f'Power Spectrum of {name}')


def main():
    global prox_data, prox_data_high
    records = load_data(filter_year='46000')
    records['maw_3_clean'] = pd.read_csv('internal_excel_sheets/' +
                                         'filled_seb_runs/MAW-3-clean.csv')

    period = 20
    filt_freq = 1/100
    min_ice_date = 27000
    max_ice_date = records['ngrip']['age_BP'].max()

    _, d_o_events, _ = d_o_dates(min_date=min_ice_date, 
                                 max_date=max_ice_date)
    
    prox_data = plot_cross_corr([records['maw_3_clean'], records['ngrip'],
                                records['wais'], records['arabia']],
                    title='Cross Correlation Between MAW-3 and NGRIP', 
                    period=period, filter_freq=filt_freq)

    # Let's now do our chunk-based analysis of D-O events
    chunks = chunked_d_o(d_o_events, prox_data)
    # Use this to create a composite, and plot
    aggregate = composite(d_o_events, chunks)
    plot_chunk(aggregate, 'Composite D-O Event')
    
    # Lag-correlation matrices
    comp_mat_full = lag_corr_mat(prox_data, 20)
    comp_mat_agg = lag_corr_mat(aggregate, 20)
    comp_mat_chunk, comp_mat_se = averaged_lag_mats(chunks, 20)
    # Clean this up and print
    comp_mat_chunk = clean_lag_corr(comp_mat_chunk, comp_mat_se)
    
    print('\n\nFull Record')
    print(comp_mat_full)
    print('Composite')
    print(comp_mat_agg)
    print('Each event')
    print(comp_mat_chunk)
    
    # Great! This seems to work. Let's try the red fit for maw    
    plot_psd(prox_data, 'maw', nperseg=128, period=period,
             sig=0.95, cutoff=filt_freq, name='Mawmluh Cave')
        
    
if __name__ == '__main__':
    main()
