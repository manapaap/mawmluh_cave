# -*- coding: utf-8 -*-
"""
Plotting code for speleothe project: publication-level plots

Basically steal relevant data loading functions from previous files
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import chdir
import cartopy.crs as ccrs

chdir('C:/Users/Aakas/Documents/Oster_lab/programs')
from shared_funcs import combine_mawmluh, load_data, d_o_dates, heinrich_dates
chdir('C:/Users/Aakas/Documents/Oster_lab/')


def proxy_stack(records, d_o_dates, age_data):
    """
    Plots the proxy stack containing Our d18O and d13C, Hulu d18O, NGRIP d18O,
    WAIS d18O, arabian sea refl
    
    Labels all D-O events and Heinrich Events
    
    Puts scatter points showing error in dating
    """
    fig, ax = plt.subplots(7, 1, sharex=True)
    plt.subplots_adjust(top=0.6)
    fig.set_size_inches(10, 15)
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    min_age = records['maw_3_clean']['age_BP'].min()
    max_age = records['maw_3_clean']['age_BP'].max()
    
    color1 = plt.cm.viridis(0.0)
    color1_5 = plt.cm.viridis(0.65)
    color2 = plt.cm.viridis(0.2)
    color2_5 = plt.cm.viridis(0.95)
    color3 = plt.cm.viridis(0.4)
    color4 = plt.cm.viridis(0.6)
    color5 = plt.cm.viridis(0.8)
    color6 = plt.cm.viridis(.99)
    color7 = plt.cm.viridis(0.7)
    
    # First, plot the records
    ax[0].plot(records['maw_3_clean'].age_BP, records['maw_3_clean'].d13C, 
                   label='MAW-3 d18O', color=color1)
    ax[0].plot(records['maw_jag'].age_BP, records['maw_jag'].d13C,
               label='Jaglan 2021', color=color2_5, alpha=0.7)
    ax[0].set_ylim(-5, 4)
    ax[0].invert_yaxis()
    ax[0].grid()
    ax[0].set_ylabel('MAW-3 δ¹³C ‰')
    ax[0].set_yticks(np.arange(-4, 4, 2))
    ax[0].set_xlim(min_age, max_age)
    
    ax[1].plot(records['maw_3_clean'].age_BP, records['maw_3_clean'].d18O,
               label='MAW-3 d18O', color=color2)
    ax[1].plot(records['maw_jag'].age_BP, records['maw_jag'].d18O,
               label='Jaglan 2021', color=color1_5, alpha=0.6)
    ax[1].set_ylim(-8, -0.5)
    ax[1].invert_yaxis()
    ax[1].grid()
    ax[1].set_ylabel('MAW-3 δ¹⁸O ‰')
    ax[1].set_yticks(np.arange(-7, 0, 2))
    ax[1].spines[['top']].set_visible(False)
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")
    ax[1].set_ylim(-1, -5.5)
    # ax[1].legend()

    ax[2].plot(records['hulu'].age_BP, records['hulu'].d18O,
               label='NGRIP d18O', color=color3)
    ax[2].grid()
    ax[2].invert_yaxis()
    ax[2].set_ylabel('Hulu δ¹⁸O ‰ ')
    ax[2].set_yticks(np.arange(-8, -5, 1.5))
    ax[2].spines[['top']].set_visible(False)
    ax[2].set_ylim(-6, -9)
    
    ax[3].plot(records['ngrip'].age_BP, records['ngrip'].d18O,
               label='NGRIP d18O', color=color4)
    ax[3].set_ylim(-52, -33)
    ax[3].grid()
    ax[3].set_ylabel('NGRIP δ¹⁸O ‰')
    ax[3].set_xlabel('Age (Years BP)')
    ax[3].set_yticks(np.arange(-48, -32, 6))
    ax[3].spines[['top']].set_visible(False)
    ax[3].yaxis.tick_right()
    ax[3].yaxis.set_label_position("right")
    ax[3].set_ylim(-48, -34)
    
    ax[4].plot(records['wais'].age_BP, records['wais'].d18O,
               label='WAIS d18O', color=color5)
    ax[4].grid()
    ax[4].set_ylabel('Wais δ¹⁸O ‰')
    ax[4].spines[['top']].set_visible(False)
    # ax[4].set_yticks(np.arange(-43, -36, 2))
    ax[4].set_ylim(-42, -37)
    ax[4].invert_yaxis()
    
    ax[5].plot(records['arabia'].age_BP, records['arabia'].refl,
               label='WAIS d18O', color=color6)
    ax[5].grid()
    ax[5].set_ylabel('Arabian Sed. Refl.')
    # ax[5].set_xlabel('Age (Years BP)')
    ax[5].spines[['top']].set_visible(False)
    ax[5].yaxis.set_label_position("right")
    ax[5].invert_yaxis()
    ax[5].yaxis.tick_right()
    ax[5].set_yticks(np.arange(90, 50, -20))
    ax[5].set_ylim(95, 50)
    
    ax[6].plot(records['sofular'].age_BP, records['sofular'].d13C,
               label='Sofular d13C', color=color7)
    ax[6].grid()
    ax[6].set_ylabel('Sofular δ¹³C ‰')
    ax[6].set_xlabel('Age (Years BP)')
    ax[6].spines[['top']].set_visible(False)
    ax[6].yaxis.set_label_position("left")
    ax[6].set_ylim(-10, -6)
    ax[6].invert_yaxis()
    
    ax[0].set_title('Proxy Stack')
    
    # Add vlines for d-o events
    for event, year in d_o_dates.items():
        # Don't print DO #12 here
        if event > 11:
            continue
        ax[3].text(year + 170, -36, f'{event}', c='red', alpha=0.9)
        for axis in ax:
            axis.vlines(year, -1000, 1000, colors='red', 
                        linestyle='dashed', alpha=0.6)
            
    # Add dating ages and error bar for it
    age_data = age_data.query(f'age < {max_age}')
    age_data = age_data.query(f'age > {min_age}')
    age_data.reset_index(inplace=True)
    
    for n in range(len(age_data)):
        data = age_data.iloc[n]
        ax[0].errorbar(data['age'], 2, xerr=2 * data['error'], fmt='o',
                       color='orange', capsize=0.1)
    
    plt.show()


def proxy_stack_comb(records, d_o_dates, age_data, hein_dates):
    """
    Plots the proxy stack containing Our d18O and d13C, Hulu d18O, NGRIP d18O,
    WAIS d18O, arabian sea refl
    
    Labels all D-O events and Heinrich Events
    
    Puts scatter points showing error in dating
    """
    fig, ax = plt.subplots(7, 1, sharex=True)
    plt.subplots_adjust(top=0.6)
    fig.set_size_inches(10, 15)
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    min_age = records['maw_comb']['age_BP'].min()
    max_age = records['maw_comb']['age_BP'].max()
    
    color1 = plt.cm.viridis(0.0)
    color1_5 = plt.cm.viridis(0.65)
    color2 = plt.cm.viridis(0.2)
    color2_5 = plt.cm.viridis(0.95)
    color3 = plt.cm.viridis(0.4)
    color4 = plt.cm.viridis(0.6)
    color5 = plt.cm.viridis(0.8)
    color6 = plt.cm.viridis(.99)
    color7 = plt.cm.viridis(0.7)
    
    # First, plot the records
    ax[0].plot(records['maw_comb'].age_BP, records['maw_comb'].d13C, 
                   label='MAW-3 d13C', color=color1)
    ax[0].set_ylim(-5, 4)
    ax[0].invert_yaxis()
    ax[0].grid()
    ax[0].set_ylabel('MAW-3 δ¹³C ‰')
    ax[0].set_yticks(np.arange(-4, 4, 2))
    ax[0].set_xlim(min_age, max_age)
    
    ax[1].plot(records['maw_comb'].age_BP, records['maw_comb'].d18O,
               label='MAW-3 d18O', color=color2)
    ax[1].set_ylim(-8, -0.5)
    ax[1].invert_yaxis()
    ax[1].grid()
    ax[1].set_ylabel('MAW-3 δ¹⁸O ‰')
    ax[1].set_yticks(np.arange(-7, 0, 2))
    ax[1].spines[['top']].set_visible(False)
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")
    ax[1].set_ylim(-1, -5.5)
    # ax[1].legend()

    ax[2].plot(records['hulu'].age_BP, records['hulu'].d18O,
               label='NGRIP d18O', color=color3)
    ax[2].grid()
    ax[2].invert_yaxis()
    ax[2].set_ylabel('Hulu δ¹⁸O ‰ ')
    ax[2].set_yticks(np.arange(-8, -5, 1.5))
    ax[2].spines[['top']].set_visible(False)
    ax[2].set_ylim(-5, -9)
    
    ax[3].plot(records['ngrip'].age_BP, records['ngrip'].d18O,
               label='NGRIP d18O', color=color4)
    ax[3].set_ylim(-52, -33)
    ax[3].grid()
    ax[3].set_ylabel('NGRIP δ¹⁸O ‰')
    ax[3].set_xlabel('Age (Years BP)')
    ax[3].set_yticks(np.arange(-48, -32, 6))
    ax[3].spines[['top']].set_visible(False)
    ax[3].yaxis.tick_right()
    ax[3].yaxis.set_label_position("right")
    ax[3].set_ylim(-48, -34)
    
    ax[4].plot(records['wais'].age_BP, records['wais'].d18O,
               label='WAIS d18O', color=color5)
    ax[4].grid()
    ax[4].set_ylabel('Wais δ¹⁸O ‰')
    ax[4].spines[['top']].set_visible(False)
    # ax[4].set_yticks(np.arange(-43, -36, 2))
    ax[4].set_ylim(-42, -37)
    ax[4].invert_yaxis()
    
    ax[5].plot(records['arabia'].age_BP, records['arabia'].refl,
               label='WAIS d18O', color=color6)
    # ax[3].set_ylim(-52, -33)
    ax[5].grid()
    ax[5].set_ylabel('Arabian Sed. Refl.')
    ax[5].spines[['top']].set_visible(False)
    ax[5].yaxis.set_label_position("right")
    ax[5].invert_yaxis()
    ax[5].yaxis.tick_right()
    ax[5].set_yticks(np.arange(90, 50, -20))
    ax[5].set_ylim(95, 50)
    
    ax[6].plot(records['sofular'].age_BP, records['sofular'].d13C,
               label='Sofular d13C', color=color7)
    ax[6].grid()
    ax[6].set_ylabel('Sofular δ¹³C ‰')
    ax[6].set_xlabel('Age (Years BP)')
    ax[6].spines[['top']].set_visible(False)
    ax[6].yaxis.set_label_position("left")
    ax[6].set_ylim(-10, -6)
    ax[6].invert_yaxis()
    
    ax[0].set_title('Proxy Stack')
    
    # Add vlines for d-o events
    # d_o_dates.pop(3)
    for event, year in d_o_dates.items():
        # Hack different plot for #12 due to location on the edge
        if event == 12:
            ax[3].text(year - 600, -32, f'{event}', c='red', alpha=0.9, 
                       size='large')
        else:
            ax[3].text(year + 200, -32, f'{event}', c='red', alpha=0.9,
                       size='large')
        for axis in ax:
            axis.vlines(year, -1000, 1000, colors='red', 
                        linestyle='dashed', alpha=0.6)
            
    for event, year in hein_dates.items():
        ax[4].text(year + 400, -37.25, f'H{event}', c='orange', alpha=0.9,
                   size='large')
        for axis in ax:
            axis.vlines(year, -1000, 1000, colors='orange', 
                        linestyle='dashdot', alpha=0.6)
            
    plt.show()


def plot_map():
    """
    Plots a map with markers representing the locations from which the proxy
    stack has been derived
    """
    # Pulled from papers I'm citing anyway
    loc_mawmluh = [25.25888889, 91.71250000][::-1]
    loc_hulu = [32.50000000, 119.16666669][::-1]
    loc_ngrip = [75.1, -42.32][::-1]
    loc_wais = [-79.468, 112.086][::-1]
    loc_arab = [23.12, 66.497][::-1]
    loc_turk = [41.5, 32][::-1]
    
    # Same colors as proxy stack
    color2 = plt.cm.viridis(0.0)
    color3 = plt.cm.viridis(0.2)
    color4 = plt.cm.viridis(0.4)
    color5 = plt.cm.viridis(0.6)
    color6 = plt.cm.viridis(0.8)
    color7 = plt.cm.viridis(0.7)
    
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    
    size = 90
    ax.scatter(*loc_mawmluh, label='Mawmluh Cave Speleothem', c=color2, s=size,
               marker='X', edgecolor='red', linewidths=0.25)
    ax.scatter(*loc_hulu, label='Hulu Cave Speleothem', c=color3, s=size,
               marker='X', edgecolor='red', linewidths=0.25)
    ax.scatter(*loc_ngrip, label='NGRIP Ice Core', c=color4, s=size,
               marker='X', edgecolor='red', linewidths=0.25)
    ax.scatter(*loc_wais, label='WAIS Divide Ice Core', c=color5, s=size,
               marker='X', edgecolor='red', linewidths=0.25)
    ax.scatter(*loc_arab, label='Arabian Sea Sediment Core', c=color6, s=size,
               marker='X', edgecolor='red', linewidths=0.25)
    ax.scatter(*loc_turk, label='Sofular Cave Speleotuem', c=color7, s=size,
               marker='X', edgecolor='red', linewidths=0.25)
    legend = ax.legend(borderpad=0.5)
    for leg in legend.get_texts():
        leg.set_fontsize('small')
    ax.set_xlim(-180, 180)
    ax.set_title('Proxy Stack Locations')

    plt.show()
  
    
def plot_heinrich(maw_data, hein_dates):
    """
    Plots H3 and H4 in a subplot for visuslization
    """
    fig, axs = plt.subplots(2, 2, sharex=False)
    plt.subplots_adjust(hspace=0, wspace=0.05) # Tighten space between records
    # fig.suptitle('Heinrich Events 3 and 4 in MAW-3')
    fig.text(0.5, 0.04, 'Age (kyr BP)', ha='center', va='center')
    
    # Define the trapezoid H4 using web plot digizer
    trapz_age = (39800, 39375, 38257, 37980)
    trapz_isotope = (-3.32, -1.56, -2.82, -4.6)
    
    color1 = plt.cm.viridis(0.0)
    color2 = plt.cm.viridis(0.6)
    
    h3_plot = maw_data.query(f'{hein_dates[3] - 2000} < age_BP < {hein_dates[3] + 2000}')
    h4_plot = maw_data.query(f'{hein_dates[4] - 2000} < age_BP < {hein_dates[4] + 2000}')
    
    # Carbon on top row and oxygen on bottom
    axs[0, 0].plot(h3_plot['age_BP'], h3_plot['d13C'], color=color1)
    axs[0, 0].grid()
    axs[0, 0].set_title('H3')
    axs[0, 0].set_ylabel('δ¹³C ‰')
    axs[0, 0].tick_params(bottom=False, labelbottom=False)
    axs[0, 0].invert_yaxis()
    axs[0, 1].plot(h4_plot['age_BP'], h4_plot['d13C'], color=color1)
    axs[0, 1].grid()
    axs[0, 1].set_title('H4')
    axs[0, 1].yaxis.tick_right()
    axs[0, 1].tick_params(bottom=False, labelbottom=False)
    axs[0, 1].invert_yaxis()
    
    axs[1, 0].plot(h3_plot['age_BP'], h3_plot['d18O'], color=color2,
                   zorder=1)
    axs[1, 0].grid()
    axs[1, 0].set_ylabel('δ¹⁸O ‰')
    axs[1, 0].invert_yaxis()
    axs[1, 0].set_xticks(np.arange(30000, 34000, 1000))
    axs[1, 1].plot(h4_plot['age_BP'], h4_plot.d18O, color=color2)
    axs[1, 1].grid()
    axs[1, 1].yaxis.tick_right()
    axs[1, 1].set_xticks(np.arange(37500, 41000, 1000))
    # axs[1, 1].vlines(trapz_age, ymin=-5, ymax=-1, color='red')
    axs[1, 1].plot(trapz_age, trapz_isotope, color='black', linestyle='dashed',
                   zorder=2)
    axs[1, 1].scatter(trapz_age, trapz_isotope, color='red', zorder=3)
    axs[1, 1].invert_yaxis()
    
  

def main():
    records = load_data(filter_year='46000')
    
    age_data = pd.read_excel('internal_excel_sheets/filled_seb_runs/' +
                              'MAW-3_am10_copra.xlsx', usecols='A:C', 
                              names=['depth', 'age', 'error'],
                              sheet_name='dating_input')
    
    min_ice_date = 27000
    max_ice_date = records['ngrip']['age_BP'].max()

    _, d_o_events, _ = d_o_dates(min_date=min_ice_date, 
                                 max_date=max_ice_date)
    hein_dates = heinrich_dates()
    
    # Plot out nonsense here
    proxy_stack(records, d_o_events, age_data)
    plot_map()
    
    # Plot the same stack but with the combined record
    records['maw_comb'] = combine_mawmluh(records, cutoff=39500)
    proxy_stack_comb(records, d_o_events, age_data, hein_dates)
    
    # Plot heinrich events
    plot_heinrich(records['maw_3_clean'], hein_dates)
    

if __name__ == '__main__':
    main()
