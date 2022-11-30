"""
Merge the CH1 irms data together to allow for plotting

As CH1 is much more simple than MAW-3, we can also add the ages
within this single file rather than breaking it up
"""


import numpy as np
import matplotlib.pyplot as plt
from os import chdir
import pandas as pd
chdir('C:/Users/Aakas/Documents/School/Oster_lab/programs')
# Use some old functions again!
from merge_seb_data import intermediate_data_load


# chdir('C:/Users/Aakas/Documents/School/Oster_lab/programs')
chdir('C:/Users/Aakas/Documents/School/Oster_lab/')


def load_data():
    """
    Loads all the data from CH1- returns a three-tuple containing:

    CH1_all:
        dict containing the original drilling planning excel file

    ch1_A_data:
        dataframe of IRMS runs for segment A

    ch1_B_data:
        dataframe of IRMS runs for segment B
    """
    CH1_all = {
        'A': pd.read_excel('internal_excel_sheets/CH1_drilling.xlsx',
                           sheet_name='CH1-A', usecols='A:C'),
        'B': pd.read_excel('internal_excel_sheets/CH1_drilling.xlsx',
                           sheet_name='CH1-B', usecols='A:C')
        }

    ch1_A_data = intermediate_data_load('CH1-A')
    ch1_B_data = intermediate_data_load('CH1-B')

    return CH1_all, ch1_A_data, ch1_B_data


def clean_irms_data(df):
    """
    Cleans the IRMS data, removing non-numeric values and unused columns
    """
    df = df[pd.to_numeric(df['ID'], errors='coerce').notnull()]
    df.drop(['mass_mg', 'stdev_d13C_im', 'stdev_d18O_im'],
            axis=1, inplace=True)
    df.rename(columns={'d13C_im': 'd13C', 'd18O_im': 'd18O'}, inplace=True)
    df.ID = pd.to_numeric(df.ID)

    return df


if __name__ == '__main__':
    CH1_all, ch1_A_data, ch1_B_data = load_data()

    ch1_A_data = clean_irms_data(ch1_A_data)
    ch1_B_data = clean_irms_data(ch1_B_data)

    # Merge in the irms data
    CH1_all['A'] = CH1_all['A'].merge(ch1_A_data, how='left', on='ID')
    CH1_all['B'] = CH1_all['B'].merge(ch1_B_data, how='left', on='ID')