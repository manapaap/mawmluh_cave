# -*- coding: utf-8 -*-
"""
Testing clustering of the MAW-3 d13C and d18O data
"""


from os import chdir
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


chdir('C:/Users/aakas/Documents/Oster_lab/')


from programs.shared_funcs import load_data


def normalize_all(pd_df):
    """
    Normalizes all columns in a pandas dataframe by subtracting mean
    and dividing by stdev
    """
    for col in pd_df.columns:
        mean = np.mean(pd_df[col])
        std = np.std(pd_df[col])
        
        pd_df[col] -= mean
        pd_df[col] /= std
    
    return pd_df


def elbow_plot(data, clus_range=(1, 10)):
    """
    creates an elbow plot to identify the optimal number of clusters
    """
    inertia = []
    
    for i in range(*clus_range):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
        
    plt.plot(range(*clus_range), inertia, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia?')
    plt.grid()
    plt.title('Elbow')
    plt.show()


def fit_cluster(data, n_clusters=2):
    """
    Fits a kmeans object to the provided data
    """
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data)

    return kmeans    


def plot_clusters(data, clust):
    """
    Creates two subplots for the clustered data on a d18O-d13C space
    and in d18O-time space
    """
    fig, axs = plt.subplots(2, 1)
    plt.subplots_adjust(hspace=0.5)
    
    axs[0].scatter(data.d18O, data.d13C, c=clust.labels_)
    axs[0].grid()
    axs[0].set_xlabel('d18O')
    axs[0].set_ylabel('d13C')
    
    axs[1].scatter(data.age_BP, data.d18O, c=clust.labels_)
    axs[1].grid()
    axs[1].set_xlabel('Age BP')
    axs[1].set_ylabel('d18O')
    
    plt.show()


def linreg_pprint(linreg_obj, title):
    """
    Linear regression pretty print, thanks chat
    """
    slope, intercept, r_value, p_value, std_err = linreg_obj
    print(f"\nLinear Regression Results for {title}:\n"
          f"Slope: {slope:.4f}\n"
          f"Intercept: {intercept:.4f}\n"
          f"R-squared: {r_value**2:.4f}\n"
          f"P-value: {p_value:.4e}\n"
          f"Standard Error: {std_err:.4f}\n")


def stad_intr_slope(maw_stad, maw_intr, linreg_stad, linreg_intr, linreg_all):
    """
    Plots the best fit lines of each cluster on top of the cluster
    to make the trends evident
    """
    plt.figure()
    # Construct our best fit lines
    range_stad = maw_stad.d18O.min(), maw_stad.d18O.max()
    range_intr = maw_intr.d18O.min(), maw_intr.d18O.max()
    range_all = [min((maw_stad.d18O.min(), maw_intr.d18O.min())),
                 max((maw_stad.d18O.max(), maw_intr.d18O.max()))]
    range_stad = np.linspace(*range_stad, num=100)
    range_intr = np.linspace(*range_intr, num=100)
    range_all = np.linspace(*range_all, num=100)
    # Output
    out_stad = linreg_stad.slope * range_stad + linreg_stad.intercept
    out_intr = linreg_intr.slope * range_intr + linreg_intr.intercept
    out_all = linreg_all.slope * range_all + linreg_all.intercept
    
    plt.scatter(maw_stad.d18O, maw_stad.d13C, label='Stadial Data',
                color='royalblue', alpha=0.8)
    plt.scatter(maw_intr.d18O, maw_intr.d13C, label='Interstadial Data',
                color='gold', alpha=0.8)
    plt.plot(range_stad, out_stad, label='Stadial Best Fit' +\
             f'\nR²={linreg_stad.rvalue**2:.3f}', color='red', 
             linestyle='dashed')
    plt.plot(range_intr, out_intr, label='Interstadial Best Fit' +\
             f'\nR²={linreg_intr.rvalue**2:.3f}', color='blueviolet', 
             linestyle='dashed')
    plt.plot(range_all, out_all, label='Full Record Best Fit' +\
             f'\nR²={linreg_all.rvalue**2:.3f}', color='black', 
             linestyle='dashed')
    plt.xlabel('δ¹⁸O [‰ VSMOW]')
    plt.ylabel('δ¹³C [‰ VPDB]')
    
    # Arrange legend
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
               fancybox=True, ncol=3)
    plt.grid()
    plt.show()


def main():
    global maw_stad, maw_intr
    records = load_data()
    maw = records['maw_3_clean']
    maw_norm = normalize_all(maw.copy())
    
    # Testing cluster values
    # elbow_plot(maw_norm[['d18O', 'd13C', 'age_BP']])
    elbow_plot(maw_norm[['d18O', 'd13C']]) 
    # Doesn't look like there's a clear "elbow" if we include age. 
    # Let's try n=2 and just the isotope data
    clust = fit_cluster(maw_norm[['d18O', 'd13C']])
    
    plot_clusters(maw, clust)
    
    maw_stad = maw.iloc[clust.labels_ == 1]
    maw_intr = maw.iloc[clust.labels_ == 0]
    
    # Do these have different properties?
    linreg_all = linregress(maw.d18O, maw.d13C)
    linreg_stad = linregress(maw_stad.d18O, maw_stad.d13C)
    linreg_intr = linregress(maw_intr.d18O, maw_intr.d13C)
    linreg_pprint(linreg_all, 'Full Record')
    linreg_pprint(linreg_stad, 'Stadial Periods')
    linreg_pprint(linreg_intr, 'Interstadial Periods')
    
    # Pretty plot summarizing this
    stad_intr_slope(maw_stad, maw_intr, linreg_stad, linreg_intr, linreg_all)
    
    
if __name__ == '__main__':
    main()
