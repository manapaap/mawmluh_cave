# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 11:48:05 2022

@author: Aakas
"""

import pandas as pd
import numpy as np
from os import chdir
import matplotlib.pyplot as plt


# chdir('C:/Users/Aakas/Documents/School/Oster_lab/programs')
chdir('C:/Users/Aakas/Documents/School/Oster_lab/')


def scam_proxy(start, stop, resolution, name):
    """
    Creates a fake proxy CSV for the sake of COPRA to run but allows me
    to do the actual assignment in python
    """
    num = int((stop - start) / resolution)
    proxy = pd.DataFrame({'depth': np.linspace(start, stop, num=num),
                         'proxy': np.random.rand(num)})

    proxy.to_csv('internal_excel_sheets/COPRA/' + name + '.csv',
                 header=False, index=False)


def load_data_MAW_3():
    """
    Loads the CSV containing the proxy information, and the COPRA output
    with the age model information
    """
    maw_3_data = pd.read_csv('internal_excel_sheets/filled_seb_runs/' +
                             'MAW-3-filled-ALL.csv')
    copra_data = pd.read_csv('internal_excel_sheets/COPRA/MAW-3-copra.txt',
                             names=['age_BP', 'dummy0', '2.5_quatile_age',
                                    '97.5_quantile_age', 'dummy', 'dummy2',
                                    'top_dist_mm'],
                             skiprows=1,
                             sep='\t',
                             index_col=False)

    return maw_3_data, copra_data


def find_nearest(array, value, array2):
    """
    Returns nearest value from an array, modified from:
    https://stackoverflow.com/questions/2566412/
    find-nearest-value-in-numpy-array

    Will allow for date assignment using the copra output
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array2[idx]


def assign_dates(speleothem_data, copra_dates):
    """
    Assigns a date column to the speleothem data
    """
    ages = np.zeros(len(speleothem_data))

    for num, dist in zip(range(len(ages)), speleothem_data.top_dist_mm):
        ages[num] = find_nearest(copra_maw_3.top_dist_mm,
                                 dist,
                                 copra_maw_3.age_BP)
    speleothem_data['age_BP'] = ages
    speleothem_data['age'] = 1950 - speleothem_data['age_BP']

    return speleothem_data


def plot_proxies(dataframe, isotope='d18O', fig=1, age='age_BP'):
    """
    Plots our proxy record!
    """
    plt.figure(fig)
    plt.plot(dataframe[age], dataframe[isotope])

    if age == 'age_BP':
        plt.xlabel('Age (Years BP)')
    else:
        plt.xlabel('Age')
    plt.ylabel(isotope + 'variation')
    plt.title('Variation of ' + isotope + ' with Age')
    plt.grid()


if __name__ == '__main__':
    maw_3_proxy, copra_maw_3 = load_data_MAW_3()
    maw_3_with_ages = assign_dates(maw_3_proxy, copra_maw_3)


    young_proxies = maw_3_with_ages.query('age_BP < 40000')
    plot_proxies(young_proxies)
    plot_proxies(young_proxies, 'd13C', 2)
