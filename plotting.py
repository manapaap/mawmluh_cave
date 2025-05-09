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
from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec


chdir('C:/Users/Aakas/Documents/Oster_lab/programs')
from shared_funcs import combine_mawmluh, load_data, d_o_dates, heinrich_dates, add_hiatus
chdir('C:/Users/Aakas/Documents/Oster_lab/')


def proxy_stack(records, d_o_dates, age_data):
    """
    Plots the proxy stack containing Our d18O and d13C, Hulu d18O, NGRIP d18O,
    WAIS d18O, arabian sea refl
    
    Labels all D-O events and Heinrich Events
    
    Puts scatter points showing error in dating
    """
    fig, ax = plt.subplots(7, 1, sharex=True)
    plt.subplots_adjust(top=0.5)
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
                   label='MAW-3 d18O', color=color1, alpha=0.8)
    ax[0].plot(records['maw_jag'].age_BP, records['maw_jag'].d13C,
               label='Jaglan 2021', color='limegreen', alpha=1, zorder=100)
    ax[0].set_ylim(-5, 4)
    ax[0].invert_yaxis()
    # ax[0].grid()
    ax[0].set_ylabel('Mawmluh δ¹³C   \n[‰ VPDB]')
    ax[0].set_yticks(np.arange(-4, 4, 2))
    ax[0].set_xlim(min_age, max_age)
    ax[0].spines['bottom'].set_visible(False)
    
    ax[1].plot(records['maw_3_clean'].age_BP, records['maw_3_clean'].d18O,
               label='MAW-3 d18O', color=color2, alpha=0.8)
    ax[1].plot(records['maw_jag'].age_BP, records['maw_jag'].d18O,
               label='Jaglan 2021', color='springgreen', alpha=1, zorder=100)
    ax[1].set_ylim(-10, 0)
    ax[1].invert_yaxis()
    # ax[1].grid()
    ax[1].set_ylabel('Mawmluh δ¹⁸O\n[‰ VPDB]')
    ax[1].set_yticks(np.arange(-8, 0, 2))
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
    ax[2].set_ylabel('Hulu δ¹⁸O ‰\n[‰ VPDB]')
    ax[2].set_yticks(np.arange(-8, -5, 1.5))
    ax[2].spines[['top']].set_visible(False)
    ax[2].set_ylim(-5, -9)
    ax[2].spines['bottom'].set_visible(False)
    
    ax[5].plot(records['ngrip'].age_BP, records['ngrip'].d18O,
               label='NGRIP d18O', color=color4)
    ax[5].set_ylim(-52, -33)
    # ax[5].grid()
    ax[5].set_ylabel('NGRIP δ¹⁸O\n[‰ VSMOW]')
    ax[5].set_xlabel('Age (Years BP)')
    ax[5].set_yticks(np.arange(-50, -32, 6))
    ax[5].spines[['top']].set_visible(False)
    ax[5].yaxis.tick_right()
    ax[5].yaxis.set_label_position("right")
    ax[5].set_ylim(-50, -34)
    ax[5].spines['bottom'].set_visible(False)
    
    ax[6].plot(records['wais'].age_BP, records['wais'].d18O,
               label='WAIS d18O', color=color5)
    # ax[6].grid()
    ax[6].set_ylabel('Wais δ¹⁸O\n[‰ VSMOW]')
    ax[6].spines[['top']].set_visible(False)
    # ax[4].set_yticks(np.arange(-43, -36, 2))
    ax[6].set_ylim(-42, -37)
    ax[6].invert_yaxis()
    
    ax[3].plot(records['arabia'].age_BP, records['arabia'].refl,
               label='WAIS d18O', color=color6)
    # ax[3].grid()
    ax[3].set_ylabel('Arabian Sed.\nReflectance')
    # ax[5].set_xlabel('Age (Years BP)')
    ax[3].spines[['top']].set_visible(False)
    ax[3].yaxis.set_label_position("right")
    ax[3].invert_yaxis()
    ax[3].yaxis.tick_right()
    ax[3].set_yticks(np.arange(90, 40, -15))
    ax[3].set_ylim(100, 50)
    ax[3].spines['bottom'].set_visible(False)
    
    ax[4].plot(records['sofular'].age_BP, records['sofular'].d13C,
               label='Sofular d13C', color=color7)
    # ax[4].grid()
    ax[4].set_ylabel('Sofular δ¹³C\n[‰ VPDB]')
    ax[4].set_xlabel('Age (Years BP)')
    ax[4].spines[['top']].set_visible(False)
    ax[4].yaxis.set_label_position("left")
    ax[4].set_ylim(-10, -6)
    ax[4].invert_yaxis()
    ax[4].spines['bottom'].set_visible(False)
    
    # Add vlines for d-o events
    for event, year in d_o_dates.items():
        # Don't print DO #12 here
        if event > 11:
            continue
        ax[5].text(year + 170, -37, f'{event}', c='red', alpha=0.9)
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
    plt.subplots_adjust(top=0.5)
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
    ax[0].set_ylabel('MAW-3 δ¹³C\n[‰ VPDB]')
    ax[0].set_yticks(np.arange(-4, 4, 2))
    ax[0].set_xlim(min_age, max_age)
    ax[0].spines['bottom'].set_visible(False)
    
    ax[1].plot(records['maw_comb'].age_BP, records['maw_comb'].d18O,
               label='MAW-3 d18O', color=color2)
    ax[1].set_ylim(-8, -0.5)
    ax[1].invert_yaxis()
    # ax[1].grid()
    ax[1].set_ylabel('MAW-3 δ¹⁸O\n[‰ VPDB]')
    ax[1].set_yticks(np.arange(-7, 0, 2))
    ax[1].spines[['top']].set_visible(False)
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")
    ax[1].set_ylim(-0.5, -5.5)
    # ax[1].legend()
    ax[1].spines['bottom'].set_visible(False)

    ax[2].plot(records['hulu'].age_BP, records['hulu'].d18O,
               label='NGRIP d18O', color=color3)
    # ax[2].grid()
    ax[2].invert_yaxis()
    ax[2].set_ylabel('Hulu δ¹⁸O\n[‰ VPDB]')
    ax[2].set_yticks((-5, -7, -9))
    ax[2].spines[['top']].set_visible(False)
    ax[2].set_ylim(-5, -9)
    ax[2].spines['bottom'].set_visible(False)
    
    ax[5].plot(records['ngrip'].age_BP, records['ngrip'].d18O,
               label='NGRIP d18O', color=color4)
    ax[5].set_ylim(-52, -33)
    # ax[3].grid()
    ax[5].set_ylabel('NGRIP δ¹⁸O\n[‰ VSMOW]')
    ax[5].set_xlabel('Age (Years BP)')
    ax[5].set_yticks(np.arange(-48, -32, 6))
    ax[5].spines[['top']].set_visible(False)
    ax[5].yaxis.tick_right()
    ax[5].yaxis.set_label_position("right")
    ax[5].set_ylim(-50, -34)
    ax[5].spines['bottom'].set_visible(False)
    
    ax[6].plot(records['wais'].age_BP, records['wais'].d18O,
               label='WAIS d18O', color=color5)
    # ax[4].grid()
    ax[6].set_ylabel('Wais δ¹⁸O\n[‰ VSMOW]')
    ax[6].spines[['top']].set_visible(False)
    # ax[4].set_yticks(np.arange(-43, -36, 2))
    ax[6].set_ylim(-42, -37)
    ax[6].invert_yaxis()
    ax[6].set_yticks([-38, -40, -42])
    ax[6].set_xlabel('Age (Years BP)')
    
    ax[3].plot(records['arabia'].age_BP, records['arabia'].refl,
               label='WAIS d18O', color=color6) 
    # ax[3].set_ylim(-52, -33)
    # ax[5].grid()
    ax[3].set_ylabel('Arabian Sed.\nReflectance')
    ax[3].spines[['top']].set_visible(False)
    ax[3].yaxis.set_label_position("right")
    ax[3].invert_yaxis()
    ax[3].yaxis.tick_right()
    ax[3].set_yticks((50, 75, 100))
    ax[3].set_ylim(100, 50)
    ax[3].spines['bottom'].set_visible(False)
    
    ax[4].plot(records['sofular'].age_BP, records['sofular'].d13C,
               label='Sofular d13C', color=color7)
    # ax[6].grid()
    ax[4].set_ylabel('Sofular δ¹³C\n[‰ VPDB]')
    ax[4].spines[['top']].set_visible(False)
    ax[4].yaxis.set_label_position("left")
    ax[4].set_ylim(-10, -6)
    ax[4].invert_yaxis()
    ax[4].spines['bottom'].set_visible(False)
        
    # Add vlines for d-o events
    # d_o_dates.pop(3)
    for event, year in d_o_dates.items():
        # Hack different plot for #12 due to location on the edge
        if event == 12:
            ax[0].text(year - 600, -2.5, f'{event}', c='red', alpha=0.9, 
                       size='large')
        else:
            ax[0].text(year + 200, -2.5, f'{event}', c='red', alpha=0.9,
                       size='large')
        for axis in ax:
            axis.vlines(year, -1000, 1000, colors='red', 
                        linestyle='dashed', alpha=0.6)
            
    for event, year in hein_dates.items():
        for axis in ax:
            # Draw the vertical lines for Heinrich events
            # axis.vlines(year, -1000, 1000, colors='orange', linestyle='dashdot', alpha=0.6)
            axis.axvspan(year - 1000, year + 1000  , color='orange', alpha=0.2)
        ax[1].text(year + 400, -4.5, f'H{event}', c='darkorange', alpha=1,
                   size='large')
            
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
    loc_braz = [-10.1602, -40.8605][::-1]
    
    num = 7
    color2 = plt.cm.Set1(0 * 1 / num)
    color3 = plt.cm.Set1(1 * 1 / num)
    color4 = plt.cm.Set1(2 * 1 / num)
    color5 = plt.cm.Set1(3 * 1 / num)
    color6 = plt.cm.Set1(4 * 1 / num)
    color7 = plt.cm.Set1(5 * 1 / num)
    color8 = plt.cm.Set1(6 * 1 / num)
    
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(alpha=0.9, linestyle='dashdot', linewidth=0.4)
    fig = plt.gcf()
    fig.set_size_inches(10, 3.5)
    
    size = 90
    border_col = 'black'
    width = 0.5
    marker = '*'
    ax.scatter(*loc_mawmluh, label='Mawmluh Cave Speleothem', color=color2, s=size,
               marker=marker, edgecolor=border_col, linewidths=width, zorder=10)
    ax.scatter(*loc_hulu, label='Hulu Cave Speleothem', color=color3, s=size,
               marker=marker, edgecolor=border_col, linewidths=width, zorder=10)
    ax.scatter(*loc_ngrip, label='NGRIP Ice Core', color=color4, s=size,
               marker=marker, edgecolor=border_col, linewidths=width, zorder=10)
    ax.scatter(*loc_wais, label='WAIS Divide Ice Core', color=color5, s=size,
               marker=marker, edgecolor=border_col, linewidths=width, zorder=10)
    ax.scatter(*loc_arab, label='Arabian Sea Sediment Core', color=color6, s=size,
               marker=marker, edgecolor=border_col, linewidths=width, zorder=10)
    ax.scatter(*loc_turk, label='Sofular Cave Speleothem', color=color7, s=size,
               marker=marker, edgecolor=border_col, linewidths=width, zorder=10)
    ax.scatter(*loc_braz, label='NE Brazil Speleothems', color=color8, s=size,
               marker=marker, edgecolor=border_col, linewidths=width, zorder=10)
    ax.stock_img()
    # Terrain features
    ax.add_feature(cfeature.LAKES, edgecolor='black', linewidth=0.2,
                   linestyle='dashdot')
    # Add a thicker, semi-transparent line as a border
    ax.add_feature(cfeature.RIVERS.with_scale('110m'), linewidth=0.4, 
                   edgecolor='black', alpha=0.9, zorder=2, linestyle='dashdot')
    # Add the actual river line on top
    ax.add_feature(cfeature.RIVERS.with_scale('110m'), 
                   linewidth=0.2, edgecolor='lightblue', zorder=3)
    
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                 linestyle='dashed', alpha=0.6, color='dimgrey', linewidth=0.65,)
    gl.right_labels = False
    legend = ax.legend(bbox_to_anchor=(1.025, 0.97), loc='upper left',
                       borderaxespad=0., frameon=False, fontsize='large',
                       markerscale=1.6, labelspacing=1.2)
    ax.set_xlim(-120, 180)

    plt.show()
  

def combined_figure(records, d_o_dates, age_data, hein_dates):
    fig = plt.figure(figsize=(10, 18))
    gs = GridSpec(2, 1, height_ratios=[8, 2], hspace=0.15, figure=fig)

    # --- Panel A: proxy stack (7 rows) ---
    gs_top = gs[0].subgridspec(7, 1, hspace=0.0)
    axs = []
    for i in range(7):
        if i == 0:
            ax = fig.add_subplot(gs_top[i])
        else:
            ax = fig.add_subplot(gs_top[i], sharex=axs[0])
        axs.append(ax)

    # define colors
    cmap = plt.cm.viridis
    color1   = cmap(0.0)
    color1_5 = cmap(0.65)
    color2   = cmap(0.2)
    color2_5 = cmap(0.95)
    color3   = cmap(0.4)
    color4   = cmap(0.6)
    color5   = cmap(0.8)
    color6   = cmap(.99)
    color7   = cmap(0.7)

    min_age = records['maw_comb']['age_BP'].min()
    max_age = records['maw_comb']['age_BP'].max()

    # 0: MAW-3 d13C
    ax = axs[0]
    ax.plot(records['maw_comb'].age_BP, records['maw_comb'].d13C,
            color=color1)
    ax.set_ylim(-5, 4)
    ax.invert_yaxis()
    ax.set_ylabel('MAW-3 δ¹³C\n[‰ VPDB]')
    ax.set_yticks(np.arange(-4, 4, 2))
    ax.set_xlim(min_age, max_age)
    ax.spines['bottom'].set_visible(False)

    # 1: MAW-3 d18O
    ax = axs[1]
    ax.plot(records['maw_comb'].age_BP, records['maw_comb'].d18O,
            color=color2)
    ax.set_ylim(-0.5, -5.5)
    ax.invert_yaxis()
    ax.set_ylabel('MAW-3 δ¹⁸O\n[‰ VPDB]')
    ax.set_yticks(np.arange(-7, 0, 2))
    ax.spines['top'].set_visible(False)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.spines['bottom'].set_visible(False)

    # 2: Hulu δ18O
    ax = axs[2]
    ax.plot(records['hulu'].age_BP, records['hulu'].d18O,
            color=color3)
    ax.set_ylim(-5, -9)
    ax.invert_yaxis()
    ax.set_ylabel('Hulu δ¹⁸O\n[‰ VPDB]')
    ax.set_yticks([-5, -7, -9])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # 3: Arabian Sea Reflectance
    ax = axs[3]
    ax.plot(records['arabia'].age_BP, records['arabia'].refl,
            color=color6)
    ax.set_ylabel('Arabian Sed.\nReflectance')
    ax.invert_yaxis()
    ax.set_yticks([50, 75, 100])
    ax.set_ylim(100, 50)
    ax.spines['top'].set_visible(False)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.spines['bottom'].set_visible(False)

    # 4: Sofular δ13C
    ax = axs[4]
    ax.plot(records['sofular'].age_BP, records['sofular'].d13C,
            color=color7)
    ax.set_ylabel('Sofular δ¹³C\n[‰ VPDB]')
    ax.set_ylim(-10, -6)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # 5: NGRIP δ18O
    ax = axs[5]
    ax.plot(records['ngrip'].age_BP, records['ngrip'].d18O,
            color=color4)
    ax.set_ylim(-50, -34)
    ax.set_ylabel('NGRIP δ¹⁸O\n[‰ VSMOW]')
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xlabel('Age (Years BP)')

    # 6: WAIS δ18O
    ax = axs[6]
    ax.plot(records['wais'].age_BP, records['wais'].d18O,
            color=color5)
    ax.set_ylabel('WAIS δ¹⁸O\n[‰ VSMOW]')
    ax.set_ylim(-42, -37)
    ax.invert_yaxis()
    ax.set_yticks([-38, -40, -42])
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Age (Years BP)')

    # shared vlines for D-O events
    for event, year in d_o_dates.items():
        # label on the first panel
        if event == 12:
            axs[0].text(year - 600, -2.5, str(event),
                        color='red', size='large', alpha=0.9)
        else:
            axs[0].text(year + 200, -2.5, str(event),
                        color='red', size='large', alpha=0.9)
        for ax in axs:
            ax.vlines(year, -1000, 1000,
                      colors='red', linestyle='dashed', alpha=0.6)

    # Heinrich spans + labels on panel 1
    for event, year in hein_dates.items():
        for ax in axs:
            ax.axvspan(year - 1000, year + 1000,
                       color='orange', alpha=0.2)
        axs[1].text(year + 400, -4.5, f'H{event}',
                    color='darkorange', size='large')

    # clean up lower x-labels
    for ax in axs[:-1]:
        ax.tick_params(labelbottom=False)

    # --- Panel B: map ---
    ax_map = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())
    ax_map.coastlines(alpha=0.9, linestyle='dashdot', linewidth=0.4)
    ax_map.stock_img()
    ax_map.add_feature(cfeature.LAKES, edgecolor='black',
                       linewidth=0.2, linestyle='dashdot')
    ax_map.add_feature(cfeature.RIVERS.with_scale('110m'),
                       linewidth=0.4, edgecolor='black',
                       alpha=0.9, linestyle='dashdot')
    ax_map.add_feature(cfeature.RIVERS.with_scale('110m'),
                       linewidth=0.2, edgecolor='lightblue', zorder=3)

    # cave/core locations
    locs = {
        'Mawmluh Cave Speleothem': [25.2589, 91.7125],
        'Hulu Cave Speleothem':     [32.5000, 119.1667],
        'NGRIP Ice Core':           [75.1,   -42.32],
        'WAIS Divide Ice Core':     [-79.468,112.086],
        'Arabian Sea Sediment Core':[23.12,  66.497],
        'Sofular Cave Speleothem':  [41.5,   32],
        'NE Brazil Speleothems':    [-10.1602,-40.8605],
    }
    cmap2 = plt.cm.Set1
    for i,(label, (lat,lon)) in enumerate(locs.items()):
        ax_map.scatter(lon, lat,
                       s=90, marker='*',
                       color=cmap2(i/7),
                       edgecolor='black', linewidth=0.5,
                       transform=ccrs.PlateCarree(),
                       label=label, zorder=10)

    gl = ax_map.gridlines(draw_labels=True, dms=True,
                          x_inline=False, y_inline=False,
                          linestyle='dashed', alpha=0.6,
                          color='dimgrey', linewidth=0.65)
    gl.right_labels = False

    ax_map.set_xlim(-120, 180)
    legend = ax_map.legend(bbox_to_anchor=(1.025, 0.97),
                           loc='upper left',
                           frameon=False,
                           fontsize='large',
                           markerscale=1.2,
                           labelspacing=1.1)

    # panel labels (flush left in figure coords)
    fig.text(0.01, 0.98, '(A)', fontsize=16,
             fontweight='bold', va='top')
    fig.text(0.01, 0.46, '(B)', fontsize=16,
             fontweight='bold', va='top')

    fig.align_ylabels(axs)
    plt.tight_layout()
    plt.show()    
  
    
def plot_heinrich(maw_data, hein_dates):
    """
    Plots H3 and H4 in a subplot for visuslization
    """
    fig, axs = plt.subplots(2, 2, sharex=False)
    plt.subplots_adjust(hspace=0, top=0.2) # Tighten space between records
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
    # axs[0, 0].plot(h3_plot['age_BP'], h3_plot['d13C'], color=color1)
    axs[0, 0].grid()
    axs[0, 0].set_title('H3')
    axs[0, 0].set_ylabel('δ¹³C ‰')
    axs[0, 0].tick_params(bottom=False, labelbottom=False)
    axs[0, 0].invert_yaxis()
    # axs[0, 1].plot(h4_plot['age_BP'], h4_plot['d13C'], color=color1)
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
    

def plot_heinrich_trapz(records, hein_date, trapz=True):
    """
    Plots H4/AIM8 data for d18O only, identifying the trapezoid shape
    as put forward in Liang et al. 
    
    NEW: Brazil stalagmite data and WAIS CO2
    """
    if hein_date==29500:
        fig, axs = plt.subplots(3, 1, sharex=True, sharey=False)
        fig.set_size_inches(3, 5)
        ngrip=2
    else:
        fig, axs = plt.subplots(5, 1, sharex=True, sharey=False)
        fig.set_size_inches(3, 7)
        ngrip=4
    plt.subplots_adjust(hspace=0.0, top=0.5)
    fig.text(0.5, 0.04, 'Age (kyr BP)', ha='center', va='center')    
    
    # Define the trapezoid H4 using web plot digizer
    trapz_age_maw = (39900, 39375, 38257, 38050)
    trapz_isotope_maw = (-3.6, -1.56, -2.82, -4.3)
    
    trapz_age_ng = (38140, 38334, 39869, 40044)
    trapz_iso_ng = (-37.87, -42.45, -43.69, -40.15)
    
    trapz_age_hulu = (38123, 38318, 39539, 39947)
    trapz_iso_hulu = (-8.356, -6.94, -5.93, -7.71)
    
    # Same colors as proxy stack
    color_maw = plt.cm.viridis(0.0)
    color_hulu = plt.cm.viridis(0.2)
    color_ng = plt.cm.viridis(0.4)
    color_braz = plt.cm.viridis(0.3)
    color_wais = plt.cm.viridis(0.8)
    
    axs[0].set_xlim((hein_date - 1500, hein_date + 1500))
    
    axs[0].plot(records['maw_3_clean'].age_BP, records['maw_3_clean'].d18O,
                color=color_maw)
    if trapz:
        axs[0].plot(trapz_age_maw, trapz_isotope_maw, color='black', 
                    linestyle='dashed', zorder=2, alpha=0.6)
        axs[0].scatter(trapz_age_maw, trapz_isotope_maw, color='red', 
                       zorder=3, alpha=0.9)
    axs[0].set_ylabel('MAW-3 δ¹⁸O    \n[‰ VPDB]   ')
    axs[0].spines['bottom'].set_visible(False)
    axs[0].set_ylim((-5.2, -0.6))
    axs[0].invert_yaxis()
    
    axs[ngrip].plot(records['ngrip'].age_BP, records['ngrip'].d18O, 
                color=color_ng)
    if False:
        axs[ngrip].plot(trapz_age_ng, trapz_iso_ng, color='black', linestyle='dashed',
                    zorder=2, alpha=0.6)
        axs[ngrip].scatter(trapz_age_ng, trapz_iso_ng, color='red', zorder=3,
                       alpha=0.9)
    axs[ngrip].set_ylabel('NGRIP δ¹⁸O\n[‰ VSMOW]')
    axs[ngrip].spines[['top']].set_visible(False)
    axs[ngrip].set_ylim((-48, -34))
    
    axs[1].plot(records['hulu'].age_BP, records['hulu'].d18O, color=color_hulu)
    if trapz:
        axs[1].plot(trapz_age_hulu, trapz_iso_hulu, color='black', 
                    linestyle='dashed', zorder=2, alpha=0.6)
        axs[1].scatter(trapz_age_hulu, trapz_iso_hulu, color='red', zorder=3,
                       alpha=0.9)
    axs[1].yaxis.set_label_position("right")
    axs[1].set_ylabel('Hulu δ¹⁸O ‰\n[‰ VPDB]')
    axs[1].spines[['top']].set_visible(False)   
    axs[1].spines['bottom'].set_visible(False)
    axs[1].yaxis.tick_right()
    axs[1].set_ylim((-8.7, -5.7))    
    axs[1].invert_yaxis()
    
    if hein_date==39000:
        axs[2].plot(records['ne_brazil'].age_BP, records['ne_brazil'].d18O,
                    color=color_braz)
        axs[2].set_ylabel('NE Brazil δ¹⁸O\n[‰ VPDB]')
        axs[2].spines[['top']].set_visible(False)   
        axs[2].spines['bottom'].set_visible(False)
        axs[2].invert_yaxis()
        
        axs[3].plot(records['wais_co2'].age_BP, records['wais_co2'].CO2,
                    color=color_wais)
        axs[3].set_ylim((195, 225))
        axs[3].set_yticks((200, 210, 220))
        axs[3].set_ylabel('WAIS CO₂\n[ppm]')
        axs[3].spines[['top']].set_visible(False)   
        axs[3].spines['bottom'].set_visible(False)
        axs[3].yaxis.set_label_position("right")
        axs[3].yaxis.tick_right()
    plt.show()   
    plt.tight_layout()


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


def plot_seasonal(era5_seasonal, era5_clm, pres=950, every=10, method='stream',
                  mslp=True):
    """
    Creates subplots of 850 hpa winds over india to show monsoons
    
    Includes location of Mawmluh cave speleothem
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 11.75), 
                             subplot_kw={'projection': ccrs.PlateCarree()},
                             layout='compressed')
    
    seasons = ['MAM', 'SON', 'DJF', 'JJA'][::-1]
    titles = ['Spring (MAM)', 'Fall (SON)', 'Winter (DJF)','Summer (JJA)'][::-1]
    
    for i, season in enumerate(seasons):
        ax = axes.flat[i]
        data = era5_seasonal[season].sel(pressure_level=pres)
        
        # Get the wind components and grid information
        u_sub = data.u[::every, ::every]
        v_sub = data.v[::every, ::every]
        lats = data.latitude[::every]
        lons = data.longitude[::every]
        
        # Set up the map
        ax.coastlines(linestyle='dashdot', linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.7)
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
        if mslp:
            cs = ax.contourf(lon_grid_f, lat_grid_f, era5_clm[season].msl / 100,
                            transform=ccrs.PlateCarree(),
                            zorder=0, alpha=0.8, cmap='RdBu')
            ax.clabel(cs, inline=False)
            
        else:
            ax.stock_img()
            ax.add_feature(cfeature.RIVERS)
        ax.set_title(titles[i], fontsize=12)
        # Plotting Mawmluh location
        marker_color = 'navy'
        loc_mawmluh = [25.25888889, 91.71250000][::-1]
        ax.scatter(*loc_mawmluh, label='Mawmluh Cave Speleothem', color=marker_color,
                   marker='*', edgecolor='black', linewidths=0.5, s=450, zorder=10)
        if i == 0:
            ax.text(91.7, 29, 'Mawmluh\nCave', color=marker_color,
                    zorder=10, weight='bold', horizontalalignment='center',
                    verticalalignment='center', bbox=dict(facecolor='white', 
                                                          edgecolor='black',
                                                          alpha=1))
        gl = ax.gridlines(linestyle='dashed', alpha=0.8,
                          color='dimgrey', linewidth=0.8)
        # Suppress gridlines
        if season == 'DJF':
            gl.top_labels = True
            gl.right_labels = True
            gl.left_labels = False 
            gl.bottom_labels = False
        elif season == 'MAM':  # Example for Spring
            gl.top_labels = False
            gl.right_labels = True
            gl.left_labels = False
            gl.bottom_labels = True
        elif season == 'JJA':  # Example for Summer
            gl.top_labels = True
            gl.right_labels = False
            gl.left_labels = True  # Only show on left side
            gl.bottom_labels = False # Hide bottom labels
        elif season == 'SON':  # Example for Fall
            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = True
            gl.bottom_labels = True     
    cbar = fig.colorbar(strm.lines, ax=axes.ravel().tolist(), 
                        orientation='horizontal', pad=0.01,
                        shrink=0.6)
    cbar.set_label('Wind Speed (m/s)')
    # cbar.ax.tick_params(labelsize=10, pad=10)
    if mslp:
        cbar_ax_mslp = fig.add_axes([0.15, 0.001, 0.7, 0.02])
        cbar_mslp = fig.colorbar(cs, cax=cbar_ax_mslp,
                                 orientation='horizontal', shrink=0.6)
        cbar_mslp.set_label('Mean Sea Level Pressure (hPa)')
    # cbar_mslp.ax.tick_params(labelsize=10, pad=5)
    
    # Adjust layout and show the plot
    # fig.suptitle('Indian Subcontinent Climatology')
    # plt.tight_layout()
    plt.show() 


def plot_monsoons(era5_seasonal, era5_clm, pres=950, every=10,
                  var2='tcw', name='Total Column Water', unit='kg/m²'):
    """
    Creates subplots of 950 hpa winds over India to show monsoons,
    along with contours of total precipitation over land.
    Only plotting for winter and summer.
    Includes location of Mawmluh cave speleothem.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 11.75), 
                             subplot_kw={'projection': ccrs.PlateCarree()},
                             layout='compressed')

    seasons = ['DJF', 'JJA'][::-1]
    titles = [f'Winter {pres} hPa Winds', f'Summer {pres} hPa Winds'][::-1]
    titles2 = [f'Winter {name}', f'Summer {name}'][::-1]
    labels = ['A', 'B', 'C', 'D']

    # Storage for handles to use for shared colorbars
    strm_handles = []
    cs_handles = []

    for i, season in enumerate(seasons):
        ax = axes.flat[i]
        ax2 = axes.flat[i + 2]
        data = era5_seasonal[season].sel(pressure_level=pres)
        
        ax.text(0.05, 1.03, f'{labels[i]})', transform=ax.transAxes,
                fontsize=14,  va='center', ha='right')

        ax2.text(0.05, 1.03, f'{labels[i + 2]})', transform=ax2.transAxes,
                 fontsize=14,  va='center', ha='right')

        # Get the wind components and grid info
        u_sub = data.u[::every, ::every]
        v_sub = data.v[::every, ::every]
        lats = data.latitude[::every]
        lons = data.longitude[::every]

        ax.coastlines(linestyle='dashdot', linewidth=0.8)
        ax2.coastlines(linestyle='dashdot', linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.7)
        ax2.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.7)
        ax.set_extent([60, 100, 0, 40], crs=ccrs.PlateCarree())

        mag = np.asarray(np.hypot(u_sub, v_sub))
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        lon_grid_f, lat_grid_f = np.meshgrid(era5_clm[season].longitude,
                                             era5_clm[season].latitude)

        # Plot wind
        ax.stock_img()
        ax.add_feature(cfeature.RIVERS)

        strm = ax.streamplot(lon_grid, lat_grid, u_sub, v_sub,
                             color=mag, linewidth=1, cmap='viridis',
                             density=0.75, arrowsize=2,
                             transform=ccrs.PlateCarree(),
                             zorder=9)
        strm_handles.append(strm)

        # Plot TCW
        cs = ax2.contourf(lon_grid_f, lat_grid_f, era5_clm[season][var2],
                          transform=ccrs.PlateCarree(),
                          zorder=0, alpha=0.8, cmap='RdBu')
        ax2.clabel(cs, inline=False)
        cs_handles.append(cs)

        ax.set_title(titles[i], fontsize=12)
        ax2.set_title(titles2[i], fontsize=12)

        # Mawmluh location
        marker_color = 'navy'
        loc_mawmluh = [25.25888889, 91.71250000][::-1]
        for ax_ in [ax, ax2]:
            ax_.scatter(*loc_mawmluh, label='Mawmluh Cave Speleothem', color=marker_color,
                        marker='*', edgecolor='black', linewidths=0.5, s=450,
                        zorder=99)

        if i == 1:
            ax2.text(91.7, 29.5, 'Mawmluh\nCave', color='black',
                    zorder=100, weight='bold', horizontalalignment='center',
                    verticalalignment='center', alpha=1,
                    bbox=dict(facecolor='white', 
                            edgecolor='black',
                            alpha=0.6))

        # Gridlines
        gl = ax.gridlines(linestyle='dashed', alpha=0.7,
                          color='dimgrey', linewidth=0.8)
        gl2 = ax2.gridlines(linestyle='dashed', alpha=0.7,
                            color='dimgrey', linewidth=0.8, zorder=21)
        
        gl.xlocator = MultipleLocator(10)
        gl.ylocator = MultipleLocator(10)
        gl2.xlocator = MultipleLocator(10)
        gl2.ylocator = MultipleLocator(10)

        if season == 'DJF': 
            gl.top_labels = False; gl2.right_labels = False
            gl.left_labels = False; gl2.bottom_labels = True
            gl.bottom_labels = False; gl2.left_labels = False
            gl.right_labels = False; gl2.top_labels = False
        elif season == 'JJA':  
            gl.top_labels = False; gl2.right_labels = False
            gl.left_labels = True; gl2.bottom_labels = True
            gl.bottom_labels = False; gl2.left_labels = True
            gl.right_labels = False; gl2.top_labels = False

    # Fix layout and shared colorbars
    fig.subplots_adjust(right=0.87)  # Space for colorbars

    # Wind colorbar
    cbar_wind = fig.colorbar(strm_handles[0].lines, ax=[axes[0, 0], axes[0, 1]],
                             orientation='vertical', shrink=0.8, pad=0.02)
    cbar_wind.set_label('Wind Speed (m/s)', fontsize=10)

    # TCW colorbar
    cbar_tcw = fig.colorbar(cs_handles[0], ax=[axes[1, 0], axes[1, 1]],
                            orientation='vertical', shrink=0.8, pad=0.02)
    cbar_tcw.set_label(f'{name} ({unit})', fontsize=10)

    plt.show()  


def main():
    global records
    records = load_data(filter_year='46000')
    # For this file
    plt.rcParams['figure.dpi'] = 600
    
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
    # Because I hate SI apparently
    era5_clim['msl'] /= 100
     
    era5_seasonal = seasonalize(era5_india)
    era5_sea_clim = seasonalize(era5_clim)
    # plot_seasonal(era5_seasonal, era5_sea_clim)
    # plot_seasonal(era5_seasonal, era5_sea_clim, mslp=False)
    
    plot_monsoons(era5_seasonal, era5_sea_clim)
    #plot_monsoons(era5_seasonal, era5_sea_clim, var2='msl',
    #              name='Mean Sea Level Pressure', unit='hPa')
    
    # Plot out nonsense here
    proxy_stack(records, d_o_events, age_data)
    plot_map()
    
    # COmbined plot?
    # combined_figure(records, d_o_events, age_data, hein_dates)
    
    # Add hiatuses
    records['maw_comb'] = add_hiatus(records['maw_comb'], 100)
    proxy_stack_comb(records, d_o_events, age_data, hein_dates)
    
    # Plot heinrich events
    # plot_heinrich(records['maw_3_clean'], hein_dates)
    # This one was no good...let's compare it to other records
    plot_heinrich_trapz(records, hein_dates[4])
    plot_heinrich_trapz(records, hein_dates[3], False)
    

if __name__ == '__main__':
    main()
