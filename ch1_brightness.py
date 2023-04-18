# -*- coding: utf-8 -*-
"""
Look at depths vs brightness of CH1 to tease information about minerology

@author: Aakas
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os import chdir


chdir('C:/Users/Aakas/Documents/School/Oster_lab/programs')
from merge_ch1 import load_copra
from add_ages import assign_dates


chdir('C:/Users/Aakas/Documents/School/Oster_lab/')


def load_data():
    """
    Loads data, returning a dictionary containing CH1 run information and
    pixel brightness information
    """
    ch1_proxy = pd.read_csv('internal_excel_sheets/filled_seb_runs/' +
                            'CH1-filled-AGES.csv')

    ch1_A_bright = pd.read_csv('internal_excel_sheets/CH1_A_pixel_bright.csv')
    ch1_B_bright = pd.read_csv('internal_excel_sheets/CH1_B_pixel_bright.csv')

    ch1_info = {'proxy': ch1_proxy,
                'A_bright': ch1_A_bright,
                'B_bright': ch1_B_bright}

    return ch1_info


def pixel_depth(ch1_info):
    """
    Adds depth information to the pixel files
    """
    # Let's start with A
    max_depth_a = ch1_info['proxy'].query('segment == "A"')['depth'].iloc[-1]
    depths = np.linspace(0, max_depth_a,
                         num=ch1_info['A_bright']['Gray_Value'].size)
    ch1_info['A_bright']['top_dist_mm'] = depths

    # Same process for B
    max_depth_b = ch1_info['proxy'].query('segment == "B"')['depth'].iloc[-1]
    depths = np.linspace(max_depth_a, max_depth_b + max_depth_a,
                         num=ch1_info['B_bright']['Gray_Value'].size)
    ch1_info['B_bright']['top_dist_mm'] = depths

    return pd.concat([ch1_info['A_bright'], ch1_info['B_bright']])


def bright_depth_plot(ch1_info, min_d=0, max_d=400):
    """
    Plots pixel brightness versus depth of the speleothem
    """
    fig, axs = plt.subplots(2, 1, sharex=True)

    # Pixel brightness
    axs[0].plot(ch1_info['all']['top_dist_mm'], ch1_info['all']['Gray_Value'])
    axs[0].set_ylabel('Pixel Grey Value')
    axs[0].grid()
    axs[0].set_title('Pixel Darkness Vs. Depth')
    axs[0].set_xlim(min_d, max_d)

    # d18O values
    axs[1].scatter(ch1_info['proxy']['top_dist_mm'], ch1_info['proxy']['d18O'],
                   c='red', s=1)
    axs[1].set_ylabel('δ¹⁸O')
    axs[1].set_xlabel('Depth from Top')
    axs[1].grid()
    axs[1].set_xlim(min_d, max_d)


def bright_age_plot(ch1_info, min_a=28000, max_a=35000):
    """
    Plots pixel brightness versus depth of the speleothem
    """
    fig, axs = plt.subplots(2, 1, sharex=True)

    # Pixel brightness
    axs[0].plot(ch1_info['dated']['age_BP'], ch1_info['dated']['Gray_Value'])
    axs[0].set_ylabel('Pixel Grey Value')
    axs[0].grid()
    axs[0].set_title('Pixel Brightness vs. Age')
    axs[0].set_xlim(min_a, max_a)

    # d18O values
    axs[1].scatter(ch1_info['proxy']['age_BP'], ch1_info['proxy']['d18O'],
                   c='red', s=1)
    axs[1].set_ylabel('δ¹⁸O')
    axs[1].set_xlabel('Age (Years BP)')
    axs[1].grid()
    axs[1].set_xlim(min_a, max_a)


if __name__ == '__main__':
    ch1_info = load_data()
    copra_data = load_copra()

    ch1_info['all'] = pixel_depth(ch1_info)

    bright_depth_plot(ch1_info)
    # Let's zoom in on the area of intrest
    bright_depth_plot(ch1_info, min_d=80, max_d=270)

    # Assign dates based off what we know
    ch1_info['dated'] = assign_dates(ch1_info['all'], copra_data)
    # Remove the duplicates and round down to the age precision we have
    ch1_info['dated'] = ch1_info['dated'].groupby('age_BP',
                                                  as_index=False).mean()

    bright_age_plot(ch1_info, min_a=28000, max_a=32000)
