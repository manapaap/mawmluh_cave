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
from os import chdir
from itertools import cycle
import scipy.stats as stats


chdir('C:/Users/aakas/Documents/Oster_lab/programs')

from shared_funcs import combine_mawmluh, load_data, plot_psd, d_o_dates

chdir('C:/Users/aakas/Documents/Oster_lab/')



def smooth_data(time, data, new_time, period=25, filt_freq=1/500):
    """
    Interpolates, normalizes, and applies a low-pass filter through a provided
    signal
    
    Mostly to be used in the clean_arrays function
    """
    # Interpolate to standard sampling period
    func_data = interp1d(time, data)
    new_data = func_data(new_time)
    # Low pass filter
    new_data = butter_lowpass_filter(new_data, filt_freq, 1/period, 5)
    # normalize
    mean = np.mean(new_data)
    std = np.std(new_data)
    new_data = (new_data - mean) / std
    # linear detrend
    new_data = signal.detrend(new_data)
    
    return new_data


def clean_data(arrs, period=25, filt_freq=1/500):
    """
    Applies a data cleaning technique to each array in arrs
    """
    arr1, arr2, arr3, arr4, arr5, arr6 = arrs
    
    min_age, max_age = arr1['age_BP'].min(), arr1['age_BP'].max()
    new_time = np.arange(min_age, max_age, period)
    
    new_arr1 = smooth_data(arr1['age_BP'], arr1['d18O'], new_time,
                       period=period, filt_freq=filt_freq)
    new_arr15 = smooth_data(arr1['age_BP'], arr1['d13C'], new_time,
                       period=period, filt_freq=filt_freq)
    # Flip NGRIP to be in phase with rest
    new_arr2 = smooth_data(arr2['age_BP'], -arr2['d18O'], new_time,
                       period=period, filt_freq=filt_freq)
    new_arr3 = smooth_data(arr3['age_BP'], arr3['d18O'], new_time,
                       period=period, filt_freq=filt_freq)
    new_arr4 = smooth_data(arr4['age_BP'], arr4['refl'], new_time,
                       period=period, filt_freq=filt_freq)
    new_arr5 = smooth_data(arr5['age_BP'], arr5['d18O'], new_time,
                       period=period, filt_freq=filt_freq)
    new_arr6 = smooth_data(arr6['age_BP'], arr6['d13C'], new_time,
                       period=period, filt_freq=filt_freq)
    
    data = pd.DataFrame({'age_BP': new_time,
                         'MAW': new_arr1,
                         'NGRIP': new_arr2,
                         'Wais': new_arr3,
                         'OMZ': new_arr4,
                         'Hulu': new_arr5,
                         'SOF': new_arr6})

    # Drop carbon for now- but report to Jessica and Seb
    return data



def plot_cross_corr(arrs, title='', period=25, filter_freq=1/500, plot=True, label='maw'):
    """
    Interpolates to even spacing and puts three signals through a low-pass
    filter to remove noise
    
    plots the original and interpolated signals for continuity
    
    DEPRACATED
    """
    arr1, arr2, arr3, arr4, arr5 = arrs
    # Flip NGRIP for consistency in phases
    arr2 = deepcopy(arr2)
    mean = arr2['d18O'].mean()
    arr2['d18O'] *= -1
    arr2['d18O'] += 2 * mean

    func_arr1 = interp1d(arr1['age_BP'], arr1['d13C'])
    func_arr2 = interp1d(arr2['age_BP'], arr2['d18O'])
    func_arr3 = interp1d(arr3['age_BP'], arr3['d18O'])
    func_arr4 = interp1d(arr4['age_BP'], arr4['refl'])
    func_arr5 = interp1d(arr5['age_BP'], arr5['d18O'])

    min_age, max_age = arr1['age_BP'].min(), arr1['age_BP'].max()
    ages = np.arange(min_age, max_age, period)
    min_age2, max_age2 = arr2['age_BP'].min(), arr2['age_BP'].max()
    ages_2 = np.arange(min_age2, max_age2, period)

    arr2 = arr2.query(f'{min_age} <= age_BP <= {max_age}')
    arr3 = arr3.query(f'{min_age} <= age_BP <= {max_age}')
    arr4 = arr4.query(f'{min_age} <= age_BP <= {max_age}')
    arr5 = arr5.query(f'{min_age} <= age_BP <= {max_age}')

    new_arr1 = func_arr1(ages)
    new_arr2 = func_arr2(ages)
    # new_arr2_full = func_arr2(ages_2) I don't actually use this?
    new_arr3 = func_arr3(ages)
    new_arr4 = func_arr4(ages)
    new_arr5 = func_arr5(ages)

    # Let's pass this through a low pass filter to remove noise
    new_arr1 = butter_lowpass_filter(new_arr1, filter_freq, 1/period, 8)
    new_arr2 = butter_lowpass_filter(new_arr2, filter_freq, 1/period, 8)
    new_arr3 = butter_lowpass_filter(new_arr3, filter_freq, 1/period, 8)
    new_arr4 = butter_lowpass_filter(new_arr4, filter_freq, 1/period, 8)
    new_arr5 = butter_lowpass_filter(new_arr5, filter_freq, 1/period, 8)

    # First, compare the interpolation
    if plot:
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True)
        fig.suptitle('Original vs. Interpolated d18O Record')
    
        ax1.set_ylabel(label)
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
        
        ax5.set_ylabel('Hulu')
        ax5.set_xlabel('Age BP')
        ax5.plot(arr5['age_BP'], arr5['d18O'], color='lightgreen')
        ax5.plot(ages, new_arr5, color='blue', alpha=0.5)
        ax5.set_xlim(min_age, max_age)
        ax5.grid()

    # Normalize our arrays
    new_arr1 = (new_arr1 - np.mean(new_arr1)) / np.std(new_arr1)
    new_arr2 = (new_arr2 - np.mean(new_arr2)) / np.std(new_arr2)
    new_arr3 = (new_arr3 - np.mean(new_arr3)) / np.std(new_arr3)
    new_arr4 = (new_arr4 - np.mean(new_arr4)) / np.std(new_arr4)
    new_arr5 = (new_arr5 - np.mean(new_arr5)) / np.std(new_arr5)

    # Detrend our data
    new_arr1 = signal.detrend(new_arr1)
    new_arr2 = signal.detrend(new_arr2)
    new_arr3 = signal.detrend(new_arr3)
    new_arr4 = signal.detrend(new_arr4)
    new_arr5 = signal.detrend(new_arr5)

    data = pd.DataFrame({'age_BP': ages,
                         label: new_arr1,
                         'ngrip': new_arr2,
                         'wais': new_arr3,
                         'arabia': new_arr4,
                         'hulu': new_arr5})

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


def plot_chunk(data, title='', period=20, anal_len=128):
    """
    Plots the chunk of data to observe trends
    
    Highlight NGRIP/MAW data
    """
    years = np.linspace(-anal_len * period / 2, anal_len * period / 2, 
                        anal_len)
    # Cycle through linestyles for clarity?
    lines = ["--","-."]
    linecycler = cycle(lines)
    
    plt.figure()
    for n, x in enumerate(data.columns[1:]):
        if x=='MAW' or x=='NGRIP':
            plt.plot(years, -data[x], color='black', linewidth=2.5, alpha=0.8)
            plt.plot(years, -data[x], label=x, linewidth=1, alpha=0.9)
        else:
            plt.plot(years, -data[x], label=x, alpha=0.8, 
                     linestyle=next(linecycler))
    plt.grid()
    plt.legend()
    plt.xlabel('Years From Dansgaard-Oeschger Event')
    plt.ylabel('Normalized Signal')
    plt.title(title)


def composite(d_o_events, chunks, period):
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
    aggregate['age_BP'] = np.arange(0, period * len(chunk), period)
    
    return aggregate


def chunked_d_o(d_o_events, prox_data, period, anal_len=128):
    """
    Organizes the continous D-O event record into a series of chunks, 
    index by the dictionary
    """
    # Let's create "chunks" of data around each D-O event
    chunks = {event: None for event in d_o_events.keys()}

    for event, year in d_o_events.items():
        start = year - anal_len * period / 2
        end = year + anal_len * period / 2
        
        chunk = prox_data.query(f'{start} < age_BP < {end}').reset_index(drop=True)
        chunks[event] = chunk
    
    # We need to remove those records which are too short
    chunk_len = np.max([len(chunk) for chunk in chunks.values()])
    
    chunks = {event: value for event, value in chunks.items() \
              if len(value) == chunk_len}
    
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


def averaged_lag_mats(chunks, period, sig=0.95):
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
    se_lag = stats.t.ppf((1 - (1 - sig)/2), stacked_arr.shape[0]) * std_lag /\
        np.sqrt(stacked_arr.shape[0] - 1)
    # Get these back into pandas dataframes
    # Should be fine since shape and order are preserved when doing this
    colnames = chunks[next(iter(chunks))].columns[1:]
    mean_lag = pd.DataFrame(mean_lag, columns=colnames, 
                            index=colnames)
    se_lag = pd.DataFrame(se_lag, columns=colnames, 
                            index=colnames)

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


def clean_data_maw(arrs, period=25, filt_freq=1/500):
    """
    Applies a data cleaning technique to each array in arrs
    """
    arr1 = arrs
    
    min_age, max_age = arr1['age_BP'].min(), arr1['age_BP'].max()
    new_time = np.arange(min_age, max_age, period)
    
    new_arr1 = smooth_data(arr1['age_BP'], arr1['d18O'], new_time,
                       period=period, filt_freq=filt_freq)
    new_arr15 = smooth_data(arr1['age_BP'], arr1['d13C'], new_time,
                       period=period, filt_freq=filt_freq)
    
    data = pd.DataFrame({'age_BP': new_time,
                         'MAW-d18O': new_arr1,
                         'MAW-d13C': new_arr15})

    # Drop carbon for now- but report to Jessica and Seb
    return data


def main():
    global records, prox_data
    records = load_data(filter_year='46000')

    period = 30
    filt_freq = 1/100
    min_ice_date = 27000
    anal_len = 128
    max_ice_date = records['ngrip']['age_BP'].max()

    _, d_o_events, _ = d_o_dates(min_date=min_ice_date, 
                                 max_date=max_ice_date)
    
    # Combine the Mawmluh cave records
    records['maw_comb'] = combine_mawmluh(records, cutoff=39000)
    
    prox_data = clean_data([records['maw_comb'], records['ngrip'],
                                records['wais'], records['arabia'],
                                records['hulu'], records['sofular']],
                    period=period, filt_freq=filt_freq)

    # Let's now do our chunk-based analysis of D-O events
    chunks = chunked_d_o(d_o_events, prox_data, period)
    # Use this to create a composite, and plot
    
    aggregate = composite(d_o_events, chunks, period)
    plot_chunk(aggregate, 'Composite D-O Event', period)
    
    # Lag-correlation matrices
    comp_mat_full = lag_corr_mat(prox_data, period)
    comp_mat_agg = lag_corr_mat(aggregate, period)
    comp_mat_chunk, comp_mat_se = averaged_lag_mats(chunks, period, sig=0.99)
    # Clean this up and print
    comp_mat_chunk = clean_lag_corr(comp_mat_chunk, comp_mat_se)
    
    print('\n\nFull Record')
    print(comp_mat_full)
    print('\nComposite')
    print(comp_mat_agg)
    print('\nEach event')
    print(comp_mat_chunk)
    
    # Great! This seems to work. Let's try the red fit for maw    
    plot_psd(prox_data['MAW'], nperseg=anal_len, period=period, nfft=256,
             sig=0.95, cutoff=0.0025, name='Mawmluh Cave')
    
    # Let's test lag between just mawmluh stable isotopes
    short_period = 3
    anal_len = 128 * 6
    prox_maw = clean_data_maw(records['maw_comb'], period=short_period, 
                              filt_freq=filt_freq)
    chunks_maw = chunked_d_o(d_o_events, prox_maw, short_period, anal_len)
    aggregate_maw = composite(d_o_events, chunks_maw, short_period)
    plot_chunk(aggregate_maw, 'Composite D-O Event', short_period, anal_len)
    comp_mat_agg_maw = lag_corr_mat(aggregate_maw, short_period)
    comp_mat_maw, comp_maw_se = averaged_lag_mats(chunks_maw, short_period,
                                                    sig=0.95)
    comp_mat_maw = clean_lag_corr(comp_mat_maw, comp_maw_se)
    
    print('\nMawmluh Cave Isotopes')
    print(comp_mat_maw)
    print('\nComposite')
    print(comp_mat_agg_maw)
    
    
if __name__ == '__main__':
    main()
