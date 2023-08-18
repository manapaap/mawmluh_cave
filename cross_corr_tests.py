# -*- coding: utf-8 -*-
"""
testing cross correlation between MAW-3 and other records
"""

from scipy.interpolate import interp1d
import numpy as np
import scipy.signal as signal
from copy import deepcopy


def plot_cross_corr(arr1, arr2, title, flip=False, period=25, prox='d18O'):
    """
    Plots the cross correlation between two arrays
    
    must be padded before passing into this func 
    
    arr1 is indended to be our maw-3 record
    """
    # Testing for anticorrelation
    if flip:
        arr2 = deepcopy(arr2)
        mean = arr2[prox].mean()
        arr2[prox] *= -1
        arr2[prox] += 2 * mean

    func_arr1 = interp1d(arr1['age_BP'], arr1['d18O'])
    func_arr2 = interp1d(arr2['age_BP'], arr2[prox])

    min_age, max_age = arr1['age_BP'].min(), arr1['age_BP'].max()
    ages = np.arange(min_age, max_age, period)

    arr2 = arr2.query(f'{min_age} <= age_BP <= {max_age}')

    new_arr1 = func_arr1(ages)
    new_arr2 = func_arr2(ages)

    # Let's pass this through a low pass filter to remove noise
    new_arr1 = butter_lowpass_filter(new_arr1, 1/100, 1/period, 2)
    new_arr2 = butter_lowpass_filter(new_arr2, 1/100, 1/period, 2)

    # First, compare the interpolation
    fix, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.set_title('Original vs. Interpolated')
    ax1.plot(arr1['age_BP'], arr1['d18O'], color='lightgreen')
    ax1.plot(ages, new_arr1, color='blue', alpha=0.5)
    ax1.grid()

    ax2.set_xlabel('Age BP')
    ax2.plot(arr2['age_BP'], arr2[prox], color='lightgreen')
    ax2.plot(ages, new_arr2, color='blue', alpha=0.5)
    ax2.set_xlim(min_age, max_age)
    ax2.grid()

    # Normalize our arrays
    new_arr1 = (new_arr1 - np.mean(new_arr1)) / np.std(new_arr1)
    new_arr2 = (new_arr2 - np.mean(new_arr2)) / np.std(new_arr2)

    return new_arr1, new_arr2, ages


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


def lag_finder(y1, y2, period):
    """
    Finds the lag between two signals of equal sample rate
    """
    corr = signal.correlate(y1, y2)
    lags = signal.correlation_lags(len(y1), len(y2))
    corr /= np.max(corr)

    # Let's focus this on the 1000 max lag period
    lags *= period
    low_idx = find_nearest(lags, -1000)
    high_idx = find_nearest(lags, 1000)

    lags_cop = lags[low_idx:high_idx]
    corr_cop = corr[low_idx:high_idx]

    plt.figure(np.random.randint(0, 100000))
    plt.plot(lags_cop, corr_cop)
    plt.title('Correlation between Signals')
    plt.xlabel('Years of Lag')
    plt.ylabel('Correlation')
    plt.grid()

    max_lag = corr.argmax()
    while max_lag < low_idx or max_lag > high_idx:
        corr[max_lag] = -10000
        max_lag = corr.argmax()

    lag_len = lags[max_lag]

    plt.vlines(lag_len, min(corr_cop) - 0.2, 1, color='red', linestyle='dashed')
    print(f'Maximum correlation at {lag_len} years')
    plt.text(-900, 0.8, f'Max correlation \nat {lag_len} years')


maw_smooth, ngrip_smooth, ages = plot_cross_corr(records['maw_3_clean'], records['ngrip'],
                title='Cross Correlation Between MAW-3 and NGRIP', flip=True)

_, wais_smooth, _ = plot_cross_corr(records['maw_3_clean'], records['wais'],
                                      title='Cross Correlation Between MAW-3 and WAIS', flip=False)

ngrip_range = records['ngrip'].query(f"{records['maw_3_clean']['age_BP'].min()} <" +
                              f"age_BP < {records['maw_3_clean']['age_BP'].max()}")

plot_cross_corr(ngrip_range, records['wais'],
                title='Cross Correlation Between NGRIP and WAIS', flip=False)



# remove means from arrays
# inner product of arrays to obtain a scalar
# divide by stdev of the sample
