# -*- coding: utf-8 -*-
"""
testing cross correlation between MAW-3 and other records
"""

from scipy.interpolate import interp1d
import numpy as np
import scipy.signal as signal
from copy import deepcopy
import pandas as pd


# Approximate timeline from webplotdigitizer
# and 10.1029/2000PA000571
d_o_events = {11: 42400,
              10: 41100,
              9: 40100,
              8: 38200,
              7: 35100,
              6: 33500,
              5: 32200,
              4: 28800,
              3: 0}

d_o_young = {event: year for event, year in d_o_events.items() if year < 41000}


def plot_cross_corr(arr1, arr2, arr3, title, period=25, filter_freq=1/500):
    """
    Plots the cross correlation between two arrays
    
    must be padded before passing into this func 
    
    arr1 is indended to be our maw-3 record
    """
    # Flip NGRIP for consistency in phases
    arr2 = deepcopy(arr2)
    mean = arr2['d18O'].mean()
    arr2['d18O'] *= -1
    arr2['d18O'] += 2 * mean

    func_arr1 = interp1d(arr1['age_BP'], arr1['d18O'])
    func_arr2 = interp1d(arr2['age_BP'], arr2['d18O'])
    func_arr3 = interp1d(arr3['age_BP'], arr3['d18O'])

    min_age, max_age = arr1['age_BP'].min(), arr1['age_BP'].max()
    ages = np.arange(min_age, max_age, period)
    min_age2, max_age2 = arr2['age_BP'].min(), arr2['age_BP'].max()
    ages_2 = np.arange(min_age2, max_age2, period)

    arr2 = arr2.query(f'{min_age} <= age_BP <= {max_age}')
    arr3 = arr3.query(f'{min_age} <= age_BP <= {max_age}')

    new_arr1 = func_arr1(ages)
    new_arr2 = func_arr2(ages)
    # new_arr2_full = func_arr2(ages_2) I don't actually use this?
    new_arr3 = func_arr3(ages)

    # Let's pass this through a low pass filter to remove noise
    new_arr1 = butter_lowpass_filter(new_arr1, filter_freq, 1/period, 2)
    new_arr2 = butter_lowpass_filter(new_arr2, filter_freq, 1/period, 2)
    new_arr3 = butter_lowpass_filter(new_arr3, filter_freq, 1/period, 2)

    # First, compare the interpolation
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    fig.suptitle('Original vs. Interpolated d18O Record')

    ax1.set_ylabel('MAW-3')
    ax1.plot(arr1['age_BP'], arr1['d18O'], color='lightgreen')
    ax1.plot(ages, new_arr1, color='blue', alpha=0.5)
    ax1.grid()

    ax2.set_ylabel('NGRIP')
    ax2.plot(arr2['age_BP'], arr2['d18O'], color='lightgreen')
    ax2.plot(ages, new_arr2, color='blue', alpha=0.5)
    ax2.set_xlim(min_age, max_age)
    ax2.grid()

    ax3.set_ylabel('WAIS')
    ax3.set_xlabel('Age BP')
    ax3.plot(arr3['age_BP'], arr3['d18O'], color='lightgreen')
    ax3.plot(ages, new_arr3, color='blue', alpha=0.5)
    ax3.set_xlim(min_age, max_age)
    ax3.grid()

    # Normalize our arrays
    new_arr1 = (new_arr1 - np.mean(new_arr1)) / np.std(new_arr1)
    new_arr2 = (new_arr2 - np.mean(new_arr2)) / np.std(new_arr2)
    new_arr3 = (new_arr3 - np.mean(new_arr3)) / np.std(new_arr3)

    # Detrend our data
    new_arr1 = signal.detrend(new_arr1)
    new_arr2 = signal.detrend(new_arr2)
    new_arr3 = signal.detrend(new_arr3)

    data = pd.DataFrame({'age_BP': ages,
                         'maw': new_arr1,
                         'ngrip': new_arr2,
                         'wais': new_arr3})

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


def lag_finder(y1, y2, period, plot=True):
    """
    Finds the lag between two signals of equal sample rate
    """
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    n = len(y1)
    corr = signal.correlate(y1, y2, mode='same')
    corr /= np.sqrt(signal.correlate(y1, y1, mode='same')[int(n/2)] *\
                    signal.correlate(y2, y2, mode='same')[int(n/2)])

    # Let's focus this on the 1000 max lag period
    lags = np.linspace(-0.5*n*period, 0.5*n*period, n)
    low_idx = find_nearest(lags, -1000)
    high_idx = find_nearest(lags, 1000)

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
        plt.title('Correlation between Signals')
        plt.xlabel('Years of Lag')
        plt.ylabel('Correlation')
        plt.grid()
        plt.ylim(min(corr_cop) - 0.05, 1)

        plt.vlines(lag_len, min(corr_cop) - 0.05, 1, color='red', linestyle='dashed')
        print(f'Maximum correlation at {lag_len:.2f} years')
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


def main():
    period = 25
    filt_freq = 1/400
    overlap = 400
    
    prox_data = plot_cross_corr(records['maw_3_clean'], records['ngrip'],
                                                     records['wais'],
                    title='Cross Correlation Between MAW-3 and NGRIP', 
                    period=period, filter_freq=filt_freq)
    
    lag_df = epoch_anal(prox_data, d_o_events, period, overlap)
    lag_df_good = epoch_anal(prox_data, d_o_events, period, overlap).iloc[1:-2]
    
    print('Events ~3-11')
    print(lag_df)
    print()
    print(lag_df.mean())
    print()
    print(lag_df.std() / np.sqrt(len(lag_df) - 1))
    print()
    
    print('Events 5-10')
    print(lag_df_good)
    print()
    print(lag_df_good.mean())
    print()
    print(lag_df_good.std() / np.sqrt(len(lag_df_good) - 1))


if __name__ == '__main__':
    main()
