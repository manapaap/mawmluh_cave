# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 18:10:31 2022

@author: Aakas
"""

from os import chdir
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


# chdir('C:/Users/Aakas/Documents/School/Oster_lab/programs')
chdir('C:/Users/Aakas/Documents/School/Oster_lab/')


def load_data():
    """
    Loads in seb data alongside age dates from our lab

    Returns:
        Pandas dataframe containing Seb's MAW 3_5 d18O run

        Pandas dataframe containing our MAW 3_5 runs from various ranges
    """
    new_runs = []
    seb_raw_data = pd.read_excel('external_excel_sheets/MAW-3_record_all.xlsx',
                                 sheet_name='MAW-3_5',
                                 skiprows=3,
                                 na_values=['MAX d18O', 'MIN d18O'],
                                 usecols='C:E, I:J, Q:R',
                                 names=['ID', 'dist_mm', 'top_dist_mm',
                                        'd13C', 'd18O', 'd13C_stdev',
                                        'd18O_stdev'])
    data_1_120 = pd.read_excel('internal_excel_sheets/d18O_d18C_data' +
                               '/MAW_5-3_1-120.xlsx',
                               usecols='B, D:H',
                               names=['ID', 'mass_mg', 'd13C_im',
                                      'stdev_d13C_im',
                                      'd18O_im', 'stdev_d18O_im'])
    new_runs.append(data_1_120)
    data_120_200 = pd.read_excel('internal_excel_sheets/d18O_d18C_data' +
                                 '/MAW_5-3_120-200.xlsx',
                                 usecols='B, D:H',
                                 names=['ID', 'mass_mg', 'd13C_im',
                                        'stdev_d13C_im',
                                        'd18O_im', 'stdev_d18O_im'])
    new_runs.append(data_120_200)
    data_200_300 = pd.read_excel('internal_excel_sheets/d18O_d18C_data' +
                                 '/MAW_5-3_200-300.xlsx',
                                 usecols='B, D:H',
                                 names=['ID', 'mass_mg', 'd13C_im',
                                        'stdev_d13C_im',
                                        'd18O_im', 'stdev_d18O_im'])
    new_runs.append(data_200_300)
    data_300_400 = pd.read_excel('internal_excel_sheets/d18O_d18C_data' +
                                 '/MAW_5-3_300-400.xlsx',
                                 usecols='B, D:H',
                                 names=['ID', 'mass_mg', 'd13C_im',
                                        'stdev_d13C_im',
                                        'd18O_im', 'stdev_d18O_im'])
    new_runs.append(data_300_400)
    data_400_500 = pd.read_excel('internal_excel_sheets/d18O_d18C_data' +
                                 '/MAW_5-3_400-500.xlsx',
                                 usecols='B, D:H',
                                 names=['ID', 'mass_mg', 'd13C_im',
                                        'stdev_d13C_im',
                                        'd18O_im', 'stdev_d18O_im'])
    new_runs.append(data_400_500)
    new_runs = pd.concat(new_runs)

    return seb_raw_data, new_runs


def join_to_data(seb_raw_data, in_between_run):
    """
    Does the annoying job of binding to the existing d18O, d13C, stedv columns
    of the Seb datasheet from an existing in-between run
    """
    bad_cols = ~pd.to_numeric(in_between_run.ID, errors='coerce').isna()

    num_in_bet = in_between_run[bad_cols].drop(['mass_mg'], axis=1)
    num_in_bet.ID = num_in_bet.ID.astype(int)
    num_in_bet.reset_index(inplace=True)

    final_seb = seb_raw_data.merge(num_in_bet, on='ID', how='left')

    problem_cols = ['d13C_im', 'd18O_im', 'stdev_d13C_im', 'stdev_d18O_im']
    good_cols = ['d13C', 'd18O', 'd13C_stdev', 'd18O_stdev']

    for g_col, p_col in zip(good_cols, problem_cols):
        final_seb[g_col] = final_seb[[g_col, p_col]].sum(axis=1)
        final_seb[g_col].replace(0, np.nan, inplace=True)
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
    seb_raw_data.reset_index(inplace=True)
    seb_raw_data.drop(['index'], inplace=True, axis=1)
    seb_raw_data.dropna(how='all', inplace=True)

    final_seb = deepcopy(seb_raw_data)
    final_seb = join_to_data(final_seb, in_between_runs)

    final_seb.drop(['index'], inplace=True, axis=1)

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


def plot_comp_smooth(new_seb_data, old_seb_data, max_val, data, fig=1):
    """
    Smooth line of septh vs d18O/d13C
    Specifically to compare old/new results
    """
    new_seb_data_plot = new_seb_data.loc[lambda df: df['ID'] <= max_val]
    old_seb_data_plot = old_seb_data.loc[lambda df: df['ID'] <= max_val]
    # Removing stdev of d18O and d13C as some values have NA from Seb data
    stdevs = ['d13C_stdev', 'd18O_stdev']
    new_clean_data = new_seb_data_plot.drop(stdevs, axis=1).dropna()
    old_clean_data = old_seb_data_plot.drop(stdevs, axis=1).dropna()

    plt.figure(fig)
    plt.plot(new_clean_data['dist_mm'].dropna(),
             new_clean_data[data].dropna(),
             label='New Data')
    plt.plot(old_clean_data['dist_mm'], old_clean_data[data],
             label='Old Data')
    plt.xlabel('Distance from top (mm)')
    plt.ylabel(data + ' value')
    plt.title('Variation of ' + data + ' with Sample Depth')
    plt.legend(title='Data source: ', loc='best')
    plt.grid()


if __name__ == '__main__':
    seb_raw_data, in_bet_data = load_data()
    final_seb, seb_neat_data = bind_rows(seb_raw_data, in_bet_data)

    plot_comp_smooth(final_seb, seb_neat_data, 500, 'd18O')
    plot_comp_smooth(final_seb, seb_neat_data, 500, 'd13C', fig=2)

    final_seb.to_csv('internal_excel_sheets/filled_seb_runs/' +
                     'MAW-3_5-filled.csv')
