# -*- coding: utf-8 -*-
"""
Testing for ENSO Variability in Wet/Dry Phases

Goal: Isolate the wet and dry phases from MAW and assess for ENSO
Variability in the Speleothems 
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import chdir
from scipy import signal
from scipy.interpolate import interp1d


chdir('C:/Users/Aakas/Documents/School/Oster_lab/programs')
from freq_analysis import load_data
from cross_corr_tests import combine_mawmluh, plot_psd
chdir('C:/Users/Aakas/Documents/School/Oster_lab/')


def separate_high_freq_data(records, cutoff_len, cutoff_period):
    """
    Returns a dict of high resolution data as defined by having period less 
    than cutoff period and having length creater than length cutoff
    """
    dist = np.full(len(records['maw_comb']), False)
    for n, (prev, follow) in enumerate(zip(records['maw_comb']['age_BP'], records['maw_comb']['age_BP'][1:])):
        if (follow - prev) <= cutoff_period:
            dist[n] = True
        else:
            dist[n] = False
    high_res_maw = records['maw_comb'][dist]
    # We now need to split this up into little chunks of data which are continous
    # at high resolution
    cts_chunks = dict()
    start = high_res_maw['age_BP'].iloc[0]
    n = 0
    for prev, follow in zip(high_res_maw['age_BP'], high_res_maw['age_BP'][1:]):
        if (follow - prev) > cutoff_period:
            # Isolate the continous chunks one at a time
            cts_chunks[n] = high_res_maw.query(f'{start} <= age_BP <= {prev}')
            start = follow
            n += 1
            
    # Prune the chunks that are too short
    cts_chunks = [value for key, value in cts_chunks.items()\
                  if len(value) >= cutoff_len]

    return cts_chunks

    
def plot_chunk(chunk):
    """
    Plots a single chunk of high resolution data
    """
    plt.figure()
    plt.plot(chunk['age_BP'], chunk['d18O'])
    plt.grid()
    plt.xlabel('Age BP')
    plt.ylabel('d18O')


def plot_chunks(cts_chunks):
    """
    Plots all the chunks to show their relative positions and lengths, along
    with mean frequency
    """
    for value in cts_chunks:
        mean_freq = (value['age_BP'].iloc[-1] - value['age_BP'].iloc[0]) /\
            len(value)
        plt.plot(value['age_BP'], value['d18O'], label=f'{mean_freq:.1f}')
        
    plt.grid()
    plt.xlabel(f'Age BP {len(cts_chunks)} Samples')
    plt.ylabel('d18O')
    plt.legend()
    

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def downsample(df, period, kind='slinear'):
    """
    Returns a downsampled version of our proxy record in dataframe form
    """
    func_d18O = interp1d(df['age_BP'], df['d18O'], kind=kind)
    func_d13C = interp1d(df['age_BP'], df['d13C'], kind=kind)
    new_ages = np.arange(df['age_BP'].min(), df['age_BP'].max(), period)
    new_df = pd.DataFrame({'d18O': func_d18O(new_ages),
                           'd13C': func_d13C(new_ages),
                           'age_BP': new_ages})
    return new_df


def detrend_data(df, cutoff_period, filt_freq):
    """
    Linear detrend and normalizes data. ALso a high-pass filter
    to remove low-frequency stuff
    """
    df['d18O'] = signal.detrend(df['d18O'])
    df['d13C'] = signal.detrend(df['d13C'])
    
    df['d18O'] = (df['d18O'] - np.mean(df['d18O'])) / np.std(df['d18O'])
    df['d13C'] = (df['d13C'] - np.mean(df['d13C'])) / np.std(df['d13C'])
    
    # df['d18O'] = butter_highpass_filter(df['d18O'], filt_freq, 1/cutoff_period)
    # df['d13C'] = butter_highpass_filter(df['d13C'], filt_freq, 1/cutoff_period)
    
    return df
    

def main():
    global records
    records = load_data(filter_year='46000')
    records['maw_comb'] = combine_mawmluh(records, cutoff=39500)
            
    # Let's select chunks of time where we have high sampling rates
    cutoff_period = 1 # Min sampling frequency we want
    cutoff_len = 64 # Min length of record to be useful
    
    cts_chunks = separate_high_freq_data(records, 
                                         cutoff_len, cutoff_period)
    
    plot_chunks(cts_chunks)
    print('Lengths of high-res chunks: ', [len(x) for x in cts_chunks])
    # Let's grab this high resolution regieme and play around
    seasonal = [x for x in cts_chunks if len(x) > 256][0]
    
    plot_chunk(seasonal)
    
    seasonal = downsample(seasonal, cutoff_period)
    seasonal = detrend_data(seasonal, cutoff_period, 1/32)
    plot_psd(seasonal['d18O'], nperseg=32, sig=0.95, cutoff=1, period=cutoff_period,
             nfft=64, name='Seasonal')
    plot_psd(seasonal['d13C'], nperseg=32, sig=0.95, cutoff=1, period=cutoff_period,
             nfft=64, name='Seasonal')

# Let's now standardize these to the cutoff freqnency to ensure that
# Anaysis can be consistant

# We should now put these through a high-pass filter to get rid
# of anything other than sub-50 year periods

if __name__ == '__main__':
    main()
