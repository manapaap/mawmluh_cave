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
import xarray as xr
import cartopy.feature as cfeature

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
    plt.subplots_adjust(top=0.4)
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
    # ax[0].grid()
    ax[0].set_ylabel('MAW-3 δ¹³C ‰')
    ax[0].set_yticks(np.arange(-4, 4, 2))
    ax[0].set_xlim(min_age, max_age)
    ax[0].spines['bottom'].set_visible(False)
    
    ax[1].plot(records['maw_comb'].age_BP, records['maw_comb'].d18O,
               label='MAW-3 d18O', color=color2)
    ax[1].set_ylim(-8, -0.5)
    ax[1].invert_yaxis()
    # ax[1].grid()
    ax[1].set_ylabel('MAW-3 δ¹⁸O ‰')
    ax[1].set_yticks(np.arange(-7, 0, 2))
    ax[1].spines[['top']].set_visible(False)
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")
    ax[1].set_ylim(-1, -5.5)
    # ax[1].legend()
    ax[1].spines['bottom'].set_visible(False)

    ax[2].plot(records['hulu'].age_BP, records['hulu'].d18O,
               label='NGRIP d18O', color=color3)
    # ax[2].grid()
    ax[2].invert_yaxis()
    ax[2].set_ylabel('Hulu δ¹⁸O ‰ ')
    ax[2].set_yticks(np.arange(-8, -5, 1.5))
    ax[2].spines[['top']].set_visible(False)
    ax[2].set_ylim(-5, -9)
    ax[2].spines['bottom'].set_visible(False)
    
    ax[3].plot(records['ngrip'].age_BP, records['ngrip'].d18O,
               label='NGRIP d18O', color=color4)
    ax[3].set_ylim(-52, -33)
    # ax[3].grid()
    ax[3].set_ylabel('NGRIP δ¹⁸O ‰')
    ax[3].set_xlabel('Age (Years BP)')
    ax[3].set_yticks(np.arange(-48, -32, 6))
    ax[3].spines[['top']].set_visible(False)
    ax[3].yaxis.tick_right()
    ax[3].yaxis.set_label_position("right")
    ax[3].set_ylim(-48, -34)
    ax[3].spines['bottom'].set_visible(False)
    
    ax[4].plot(records['wais'].age_BP, records['wais'].d18O,
               label='WAIS d18O', color=color5)
    # ax[4].grid()
    ax[4].set_ylabel('Wais δ¹⁸O ‰')
    ax[4].spines[['top']].set_visible(False)
    # ax[4].set_yticks(np.arange(-43, -36, 2))
    ax[4].set_ylim(-42, -37)
    ax[4].invert_yaxis()
    ax[4].spines['bottom'].set_visible(False)
    
    ax[5].plot(records['arabia'].age_BP, records['arabia'].refl,
               label='WAIS d18O', color=color6)
    # ax[3].set_ylim(-52, -33)
    # ax[5].grid()
    ax[5].set_ylabel('Arabian Sed. Refl.')
    ax[5].spines[['top']].set_visible(False)
    ax[5].yaxis.set_label_position("right")
    ax[5].invert_yaxis()
    ax[5].yaxis.tick_right()
    ax[5].set_yticks(np.arange(90, 40, -15))
    ax[5].set_ylim(95, 50)
    ax[5].spines['bottom'].set_visible(False)
    
    ax[6].plot(records['sofular'].age_BP, records['sofular'].d13C,
               label='Sofular d13C', color=color7)
    # ax[6].grid()
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
            ax[3].text(year + 200, -33, f'{event}', c='red', alpha=0.9,
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
    ax.scatter(*loc_mawmluh, label='Mawmluh Cave Speleothem', color=color2, s=size,
               marker='X', edgecolor='red', linewidths=0.25)
    ax.scatter(*loc_hulu, label='Hulu Cave Speleothem', color=color3, s=size,
               marker='X', edgecolor='red', linewidths=0.25)
    ax.scatter(*loc_ngrip, label='NGRIP Ice Core', color=color4, s=size,
               marker='X', edgecolor='red', linewidths=0.25)
    ax.scatter(*loc_wais, label='WAIS Divide Ice Core', color=color5, s=size,
               marker='X', edgecolor='red', linewidths=0.25)
    ax.scatter(*loc_arab, label='Arabian Sea Sediment Core', color=color6, s=size,
               marker='X', edgecolor='red', linewidths=0.25)
    ax.scatter(*loc_turk, label='Sofular Cave Speleotuem', color=color7, s=size,
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
    

def assign_season(time):
    month = time.dt.month
    seasons = {'DJF': (12, 1, 2), 'MAM': (3, 4, 5), 
               'JJA': (6, 7, 8), 'SON': (9, 10, 11)}
    for season, months in seasons.items():
        if month in months:
            return season


def seasonalize(era5):
    """
    Creates the seasonal averages from the regular era5 data.
    
    Loops over dataset to avoid creating large intermediate arrays
    """
    time_axis = np.asarray(era5.date, dtype=str)
    # Composite variables
    djf, mam, jja, son = [None] * 4
    n_djf, n_mam, n_son, n_jja = [0] * 4
    
    for time in time_axis:
        month = int(time[4:6])
        time = int(time)
        
        if month in [12, 1, 2]:
            if djf is None:
                djf = era5.sel(date=time)
            else:
                djf += era5.sel(date=time)
            n_djf += 1
        elif month in [3, 4, 5]:
            if mam is None:
                mam = era5.sel(date=time)
            else:
                mam += era5.sel(date=time)
            n_mam += 1
        elif month in [6, 7, 8]:
            if jja is None:
                jja = era5.sel(date=time)
            else:
                jja += era5.sel(date=time)
            n_jja += 1
        elif month in [9, 10, 11]:
            if son is None:
                son = era5.sel(date=time)
            else:
                son += era5.sel(date=time)
            n_son += 1
        else:
            print('MONTH IDENTIFICATION ERROR')
    
    # Calculate seasonal averages'
    # drop the dumb date coordinate variable
    data_dict = {
        'DJF': djf.drop_vars('expver') / n_djf,
        'MAM': mam.drop_vars('expver') / n_mam,
        'JJA': jja.drop_vars('expver') / n_jja,
        'SON': son.drop_vars('expver') / n_son
    }

    return data_dict


def plot_seasonal(era5_seasonal, era5_clm, pres=950, every=10, method='stream'):
    """
    Creates subplots of 850 hpa winds over india to show monsoons
    
    Includes location of Mawmluh cave speleothem
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 11.75), 
                             subplot_kw={'projection': ccrs.PlateCarree()},
                             layout='compressed')
    
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    titles = ['Winter (DJF)', 'Spring (MAM)', 'Summer (JJA)', 'Fall (SON)']
    
    for i, season in enumerate(seasons):
        ax = axes.flat[i]
        data = era5_seasonal[season].sel(pressure_level=pres)
        
        # Get the wind components and grid information
        u_sub = data.u[::every, ::every]
        v_sub = data.v[::every, ::every]
        lats = data.latitude[::every]
        lons = data.longitude[::every]
        
        # Set up the map
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        # Focus on India region
        ax.set_extent([60, 100, 0, 40], crs=ccrs.PlateCarree())  
        mag = np.asarray(np.hypot(u_sub, v_sub))
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        lon_grid_f, lat_grid_f = np.meshgrid(era5_clm[season].longitude, 
                                             era5_clm[season].latitude)
        
        # Plot the wind vectors
        if method == "stream":
            strm = ax.streamplot(lon_grid, lat_grid, u_sub, v_sub, 
                                 color=mag, linewidth=1, cmap='viridis', 
                                 density=0.75, arrowsize=2,
                                 transform=ccrs.PlateCarree(),
                                 zorder=9)
        else:
            strm = ax.quiver(lon_grid, lat_grid, u_sub, v_sub, 
                             mag, cmap='viridis', transform=ccrs.PlateCarree())
        # Contours for MSLP
        cs = ax.contourf(lon_grid_f, lat_grid_f, era5_clm[season].msl / 100,
                        transform=ccrs.PlateCarree(),
                        zorder=0, alpha=0.8, cmap='RdBu')
        ax.clabel(cs, fmt='%d hPa', fontsize=10)
        ax.set_title(titles[i], fontsize=12)
        # Plotting Mawmluh location
        loc_mawmluh = [25.25888889, 91.71250000][::-1]
        ax.scatter(*loc_mawmluh, label='Mawmluh Cave Speleothem', color='red',
                   marker='X', edgecolor='black', linewidths=0.25, s=350)
        if i == 0:
            ax.text(91.7, 29, 'Mawmluh\nCave', color='red',
                    zorder=10, weight='bold', horizontalalignment='center',
                    verticalalignment='center', bbox=dict(facecolor='white', 
                                                          edgecolor='black',
                                                          alpha=1))
        gl = ax.gridlines(draw_labels=True)
        # Suppress gridlines
        # Control for each subplot's gridline
        if season == 'DJF':  # Example for Winter
            gl.top_labels = True
            gl.right_labels = False
            gl.left_labels = True  # Only show on left side
            gl.bottom_labels = False  # Show at the bottom
            gl.xlines = True  # Show x-axis gridlines if desired
            gl.ylines = True  # Show y-axis gridlines if desired
        elif season == 'MAM':  # Example for Spring
            gl.top_labels = True
            gl.right_labels = True
            gl.left_labels = False  # Hide labels on left side
            gl.bottom_labels = False
            gl.xlines = False  # Turn off x-axis gridlines
            gl.ylines = True  # Keep y-axis gridlines
        elif season == 'JJA':  # Example for Summer
            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = True
            gl.bottom_labels = True  # Hide bottom labels
            gl.xlines = True  # Keep x-axis gridlines
            gl.ylines = False  # Turn off y-axis gridlines
        elif season == 'SON':  # Example for Fall
            gl.top_labels = False
            gl.right_labels = True
            gl.left_labels = False
            gl.bottom_labels = True
            gl.xlines = False  # No x-axis gridlines
            gl.ylines = False  # No y-axis gridlines
    
    cbar = fig.colorbar(strm.lines, ax=axes.ravel().tolist(), 
                        orientation='horizontal', pad=0.01,
                        shrink=0.6)
    cbar.set_label('Wind Speed (m/s)')
    cbar.ax.tick_params(labelsize=10, pad=10)
    
    cbar_ax_mslp = fig.add_axes([0.15, 0.001, 0.7, 0.02])
    cbar_mslp = fig.colorbar(cs, cax=cbar_ax_mslp,
                             orientation='horizontal', shrink=0.6)
    cbar_mslp.set_label('Mean Sea Level Pressure (hPa)')
    cbar_mslp.ax.tick_params(labelsize=10, pad=5)
    
    # Adjust layout and show the plot
    fig.suptitle('Climatological 900 hPa Winds over India')
    # plt.tight_layout()
    plt.show()    


def add_hiatus(dataarray, tol=50, varz=['d18O', 'd13C']):
    """
    Adds hiatus periods to d18O and d13C, setting values to np.nan 
    as defined by tolerance. Changes this for variables in varz
    """
    for n, (age_1, age_2) in enumerate(zip(dataarray.age_BP, dataarray.age_BP[1:])):
        diff = abs(age_2 - age_1)
        if diff > tol:
            for var in varz:
                dataarray[var][n] = np.nan
            print(n, age_1)
    return dataarray    


def main():
    global records, erera5_sea_clim
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
    # Create combined record
    records['maw_comb'] = combine_mawmluh(records, cutoff=39500) 
    # Add hiatuses
    records['maw_3_clean'] = add_hiatus(records['maw_3_clean'], 100)
    
    era5_india = xr.load_dataset('external_excel_sheets/era5_india_winds.nc')
    era5_clim = xr.load_dataset('external_excel_sheets/era5_india_clim.nc')
    era5_seasonal = seasonalize(era5_india)
    era5_sea_clim = seasonalize(era5_clim)
    plot_seasonal(era5_seasonal, era5_sea_clim)
    
    # Plot out nonsense here
    proxy_stack(records, d_o_events, age_data)
    plot_map()
    
    # Add hiatuses
    records['maw_comb'] = add_hiatus(records['maw_comb'], 100)
    proxy_stack_comb(records, d_o_events, age_data, hein_dates)
    
    # Plot heinrich events
    plot_heinrich(records['maw_3_clean'], hein_dates)
    

if __name__ == '__main__':
    main()
