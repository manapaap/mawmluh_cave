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
from add_ages import assign_dates


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


def load_copra():
    """
    Loads the age data output from copra
    """
    copra_data = pd.read_csv('internal_excel_sheets/COPRA/CH1-copra.txt',
                             names=['age_BP', 'dummy0', '2.5_quatile_age',
                                    '97.5_quantile_age', 'dummy', 'dummy2',
                                    'top_dist_mm'],
                             skiprows=1,
                             sep='\t',
                             index_col=False)

    return copra_data


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


def plot_irms(df, info='depth'):
    """
    Plots the IRMS isotope data by depth
    """
    df = df.dropna(subset=['d18O', 'd13C'])
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    color1 = plt.cm.viridis(0)
    color2 = plt.cm.viridis(0.5)

    ax1.scatter(df[info], df['d18O'], color=color1, s=0.9)
    ax2.scatter(df[info], df['d13C'], color=color2, s=0.9)

    ax1.set_ylabel('CH1 d18O (‰)')
    ax2.set_ylabel('CH1 d13C (‰)')
    if info == 'depth':
        ax1.set_xlabel('Depth (mm)')
    elif info == 'age_BP':
        ax1.set_xlabel('Age BP')
        ax1.set_title('Stable Isotope Variation by Age in CH1')
    else:
        ax1.set_xlabel(info)

    plt.grid()
    ax1.grid()
    plt.show()


def main():
    CH1_all, ch1_A_data, ch1_B_data = load_data()
    copra_data = load_copra()

    ch1_A_data = clean_irms_data(ch1_A_data)
    ch1_B_data = clean_irms_data(ch1_B_data)

    # Merge in the irms data
    CH1_all['A'] = CH1_all['A'].merge(ch1_A_data, how='left', on='ID')
    CH1_all['B'] = CH1_all['B'].merge(ch1_B_data, how='left', on='ID')

    # Add a column indicator for later analysis
    CH1_all['A']['segment'] = 'A'
    CH1_all['B']['segment'] = 'B'

    # Fix depth
    CH1_all['B']['depth'] += CH1_all['A']['depth'].iloc[-1]

    # Do the merge!
    CH1_all = pd.concat(list(CH1_all.values()))

    # How does the data look?
    plot_irms(CH1_all)

    # Add our ages in! need to change axis labels since python is being annoy
    CH1_all['top_dist_mm'] = CH1_all['depth']
    CH1_all = assign_dates(CH1_all, copra_data)

    # How does it look now?
    plot_irms(CH1_all, info='age_BP')

    # Create a CSV with our new creation, adding in some useful info
    CH1_all['stal'] = 'CH1'
    CH1_all.to_csv('internal_excel_sheets/filled_seb_runs/' +
                   'CH1-filled-AGES.csv', index=False)


if __name__ == '__main__':
    main()
