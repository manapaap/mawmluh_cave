# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 18:10:31 2022

@author: Aakas
"""

from os import chdir, listdir
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


# chdir('C:/Users/Aakas/Documents/School/Oster_lab/programs')
lab_files = 'C:/Users/Aakas/Documents/School/Oster_lab/'
chdir(lab_files)


def load_settings(filepath):
    """
    Loads the dataframe with the specific useful settings
    """
    df = pd.read_excel(filepath,
                       usecols='B, D:H',
                       names=['ID', 'mass_mg', 'd13C_im',
                              'stdev_d13C_im',
                              'd18O_im', 'stdev_d18O_im'])
    return df


def intermediate_data_load(stal_segment):
    """
    Loads the intermediate runs for a given stalagmite segment and
    returns a single Pandas dataframe
    """
    im_runs = listdir('internal_excel_sheets/d18O_d18C_data/' + stal_segment)
    for run, num in zip(im_runs, range(len(im_runs))):
        im_runs[num] = load_settings('internal_excel_sheets/d18O_d18C_data/' +
                                     stal_segment + '/' + run)
    im_runs = pd.concat(im_runs)
    return im_runs.reset_index(drop=True)


def load_data():
    """
    Loads in seb data alongside age dates from our lab

    Returns:
        Pandas dataframe containing Seb's MAW 3_5 d18O run

        Pandas dataframe containing our MAW 3_5 runs from various ranges
    """
    full_runs = []
    seb_raw_data_3_4 = pd.read_excel('external_excel_sheets/' +
                                     'MAW-3_record_all.xlsx',
                                     sheet_name='MAW-3_4',
                                     skiprows=3,
                                     na_values=['MAX d18O', 'MIN d18O'],
                                     usecols='A, C:E, I:J, Q:R',
                                     names=['segment', 'ID',
                                            'dist_mm', 'top_dist_mm',
                                            'd13C', 'd18O', 'd13C_stdev',
                                            'd18O_stdev'])
    full_runs.append(seb_raw_data_3_4)
    seb_raw_data_3_5 = pd.read_excel('external_excel_sheets/' +
                                     'MAW-3_record_all.xlsx',
                                     sheet_name='MAW-3_5',
                                     skiprows=3,
                                     na_values=['MAX d18O', 'MIN d18O'],
                                     usecols='A,C:E, I:J, Q:R',
                                     names=['segment', 'ID',
                                            'dist_mm', 'top_dist_mm',
                                            'd13C', 'd18O', 'd13C_stdev',
                                            'd18O_stdev'])
    full_runs.append(seb_raw_data_3_5)
    seb_raw_data_3_6 = pd.read_excel('external_excel_sheets/' +
                                     'MAW-3_record_all.xlsx',
                                     sheet_name='MAW-3_6',
                                     skiprows=3,
                                     na_values=['MAX d18O', 'MIN d18O'],
                                     usecols='A, C:E, I:J, Q:R',
                                     names=['segment', 'ID',
                                            'dist_mm', 'top_dist_mm',
                                            'd13C', 'd18O', 'd13C_stdev',
                                            'd18O_stdev'])
    full_runs.append(seb_raw_data_3_6)
    new_runs_3_4 = intermediate_data_load('MAW-3_4')
    new_runs_3_5 = intermediate_data_load('MAW-3_5')
    new_runs_3_6 = intermediate_data_load('MAW-3_6')

    return full_runs, [new_runs_3_4, new_runs_3_5, new_runs_3_6]


def load_full_data():
    """
    Loads the other (non incomplete) runs from Seb's data
    """
    runs = []
    seg_run_3_1 = pd.read_excel('external_excel_sheets/' +
                                'MAW-3_record_all.xlsx',
                                sheet_name='MAW-3_1_hr',
                                skiprows=3,
                                na_values=['MAX d18O', 'MIN d18O'],
                                usecols='A, C:D, H:I, T:U',
                                names=['segment', 'ID',
                                       'dist_mm',
                                       'd13C', 'd18O', 'd13C_stdev',
                                       'd18O_stdev'])
    seg_run_3_1['top_dist_mm'] = seg_run_3_1.dist_mm
    runs.append(seg_run_3_1)
    seg_run_3_2 = pd.read_excel('external_excel_sheets/' +
                                'MAW-3_record_all.xlsx',
                                sheet_name='MAW-3_2_hr',
                                skiprows=3,
                                na_values=['MAX d18O', 'MIN d18O'],
                                usecols='A, C:D, F, O:P, W:X',
                                names=['segment', 'ID',
                                       'dist_mm', 'top_dist_mm',
                                       'd13C', 'd18O', 'd13C_stdev',
                                       'd18O_stdev'])
    runs.append(seg_run_3_2)
    seg_run_3_3 = pd.read_excel('external_excel_sheets/' +
                                'MAW-3_record_all.xlsx',
                                sheet_name='MAW-3_3',
                                skiprows=3,
                                na_values=['MAX d18O', 'MIN d18O'],
                                usecols='A, C:E, I:J, Q:R',
                                names=['segment', 'ID',
                                       'dist_mm', 'top_dist_mm',
                                       'd13C', 'd18O', 'd13C_stdev',
                                       'd18O_stdev'])
    runs.append(seg_run_3_3)
    seg_run_3_7 = pd.read_excel('external_excel_sheets/' +
                                'MAW-3_record_all.xlsx',
                                sheet_name='MAW-3_7',
                                skiprows=3,
                                na_values=['MAX d18O', 'MIN d18O'],
                                usecols='A, C:E, I:J, Q:R',
                                names=['segment', 'ID',
                                       'dist_mm', 'top_dist_mm',
                                       'd13C', 'd18O', 'd13C_stdev',
                                       'd18O_stdev'])
    runs.append(seg_run_3_7)
    for seg_run in runs:
        seg_run.drop_duplicates('ID', inplace=True)
        seg_run.reset_index(inplace=True, drop=True)
        seg_run.dropna(how='all', inplace=True)

    return runs


def iterative_merge(final_seb, row_one, row_two):
    """
    Does an annoying merge that combines the newly bound data from our lab
    to Seb's existing data allowing for NaN to be summed but not removing
    0.0 values
    Computationally inefficient but i dont see an easy alternative
    plus its only like 1000 data points so whatever
    """
    row_one_copy = deepcopy(final_seb[row_one])
    row_two_copy = deepcopy(final_seb[row_two])
    for val_one, val_two, num in zip(row_one_copy,
                                     row_two_copy,
                                     range(final_seb['ID'].size)):
        if not(np.isnan(val_one)):
            continue
        elif np.isnan(val_one) and np.isnan(val_two):
            continue
        elif np.isnan(val_one) and not(np.isnan(val_two)):
            row_one_copy[num] = val_two
        else:
            row_one_copy[num] = np.mean((val_one, val_two))
    final_seb[row_one] = row_one_copy
    return final_seb


def join_to_data(seb_raw_data, in_between_run):
    """
    Does the annoying job of binding to the existing d18O, d13C, stedv columns
    of the Seb datasheet from an existing in-between run
    """
    bad_cols = ~pd.to_numeric(in_between_run.ID, errors='coerce').isna()

    num_in_bet = in_between_run[bad_cols].drop(['mass_mg'], axis=1)
    num_in_bet.ID = num_in_bet.ID.astype(int)
    num_in_bet.reset_index(inplace=True, drop=True)

    final_seb = seb_raw_data.merge(num_in_bet, on='ID', how='left')

    problem_cols = ['d13C_im', 'd18O_im', 'stdev_d13C_im', 'stdev_d18O_im']
    good_cols = ['d13C', 'd18O', 'd13C_stdev', 'd18O_stdev']

    for g_col, p_col in zip(good_cols, problem_cols):
        final_seb = iterative_merge(final_seb, g_col, p_col)
        final_seb.drop([p_col], inplace=True, axis=1)

    return final_seb


def bind_rows(seb_raw_data, in_between_runs):
    """
    Fixes the jankyness of seb's data, merges in tht good data

    Returns:
        final_seb:
            Merged seb data with new points
        seb_neat_data:
            Seb's data with the annoying repeated rows removed
    """
    seb_raw_data.drop_duplicates('ID', inplace=True)
    seb_raw_data.reset_index(inplace=True, drop=True)
    seb_raw_data.dropna(how='all', inplace=True)

    final_seb = deepcopy(seb_raw_data)
    final_seb = join_to_data(final_seb, in_between_runs)

    return final_seb, seb_raw_data


def plot_comp_data(new_seb_data, old_seb_data, max_val, data='d18O', fig=1):
    """
    Dot plots of depth vs. d18O/ d13C.
    Speficically to compare old/new results
    """
    new_seb_data_plot = new_seb_data.loc[lambda df: df['ID'] <= max_val]
    old_seb_data_plot = old_seb_data.loc[lambda df: df['ID'] <= max_val]
    plt.figure(fig)

    plt.scatter(new_seb_data_plot['dist_mm'], new_seb_data_plot[data],
                label='New Data')
    plt.scatter(old_seb_data_plot['dist_mm'], old_seb_data_plot[data],
                label='Old Data')
    plt.xlabel('Distance from top (mm)')
    plt.ylabel(data + ' value')
    plt.title('Variation of ' + data + ' with Sample Depth')
    plt.legend(title='Data source: ', loc='best')
    plt.grid()


def plot_comp_smooth(new_seb_data, old_seb_data, data, max_val=True, fig=1):
    """
    Smooth line of septh vs d18O/d13C
    Specifically to compare old/new results
    """
    if max_val:
        max_val = len(new_seb_data)
    new_seb_data_plot = new_seb_data.loc[lambda df: df.index <= max_val]
    old_seb_data_plot = old_seb_data.loc[lambda df: df.index <= max_val]
    # Removing stdev of d18O and d13C as some values have NA from Seb data
    stdevs = ['d13C_stdev', 'd18O_stdev']
    new_clean_data = new_seb_data_plot.drop(stdevs, axis=1).dropna()
    old_clean_data = old_seb_data_plot.drop(stdevs, axis=1).dropna()

    plt.figure(fig)
    plt.plot(new_clean_data['top_dist_mm'].dropna(),
             new_clean_data[data].dropna(),
             label='New Data')
    plt.plot(old_clean_data['top_dist_mm'], old_clean_data[data],
             label='Old Data')
    plt.xlabel('Distance from top (mm)')
    plt.ylabel(data + ' value')
    plt.title('Variation of ' + data + ' with Sample Depth')
    plt.legend(title='Data source: ', loc='best')
    plt.grid()


if __name__ == '__main__':
    full_runs = load_full_data()
    seb_raw_data, in_bet_data = load_data()
    final_seb, seb_neat_data = [], []
    for full_run, slice_run in zip(seb_raw_data, in_bet_data):
        final_seb_mid, seb_neat_data_mid = bind_rows(full_run, slice_run)
        final_seb.append(final_seb_mid)
        seb_neat_data.append(seb_neat_data_mid)

    all_data = pd.concat(full_runs + final_seb)
    all_seb_data = pd.concat(full_runs + seb_neat_data)

    plot_comp_smooth(all_data, all_seb_data, 'd18O')
    plot_comp_smooth(all_data, all_seb_data, 'd13C', fig=2)

    # all_data.to_csv('internal_excel_sheets/filled_seb_runs/' +
    #                  'MAW-3-filled-ALL.csv')
