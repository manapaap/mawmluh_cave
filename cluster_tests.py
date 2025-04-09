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
from scipy.odr import Model, Data, ODR


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
    kmeans = KMeans(n_clusters=2, n_init=10)
    kmeans.fit(data)

    return kmeans    


def plot_clusters(data, clust):
    """
    Creates two subplots for the clustered data on a d18O-d13C space
    and in d18O-time space
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.5)
    
    axs[0].scatter(data.d18O, data.d13C, c=clust.labels_)
    axs[0].grid()
    axs[0].set_xlabel('d18O')
    axs[0].set_ylabel('d13C')
    
    axs[1].scatter(data.age_BP, data.d18O, c=clust.labels_)
    axs[1].grid()
    axs[1].set_ylabel('d18O')
    
    axs[2].scatter(data.age_BP, data.d13C, c=clust.labels_)
    axs[2].grid()
    axs[2].set_xlabel('Age BP')
    axs[2].set_ylabel('d13C')
    
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
   
    
def plot_combined_clusters(data, maw_stad, maw_intr, linreg_stad, linreg_intr, linreg_all):
    """
    Generates two subplots:
    - Top: Clustered δ¹⁸O-δ¹³C space with linear regression overlays for 
    stadial and interstadial data
    - Bottom: δ¹⁸O vs. Age BP with consistent cluster coloring
    
    Thanks chat for combining the plots. 
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), 
                             gridspec_kw={'height_ratios': [3, 1.5]})
    plt.subplots_adjust(hspace=0.4)

    # Top subplot: δ¹⁸O-δ¹³C space with regression overlays
    range_stad = np.linspace(maw_stad.d18O.min(), maw_stad.d18O.max(), num=100)
    range_intr = np.linspace(maw_intr.d18O.min(), maw_intr.d18O.max(), num=100)
    range_all = np.linspace(min(data.d18O), max(data.d18O), num=100)

    out_stad = linreg_stad.slope * range_stad + linreg_stad.intercept
    out_intr = linreg_intr.slope * range_intr + linreg_intr.intercept
    out_all = linreg_all.slope * range_all + linreg_all.intercept

    axs[0].scatter(maw_stad.d18O, maw_stad.d13C, label='Stadial Data',
                   color='royalblue', alpha=0.8)
    axs[0].scatter(maw_intr.d18O, maw_intr.d13C, label='Interstadial Data',
                   color='gold', alpha=0.8)
    axs[0].plot(range_stad, out_stad, label='Stadial Best Fit' +
                f'\nR²={linreg_stad.rvalue**2:.3f}', color='red',
                linestyle='dashed')
    axs[0].plot(range_intr, out_intr, label='Interstadial Best Fit' +
                f'\nR²={linreg_intr.rvalue**2:.3f}', color='blueviolet',
                linestyle='dashed')
    axs[0].plot(range_all, out_all, label='Full Record Best Fit' +
                f'\nR²={linreg_all.rvalue**2:.3f}', color='black',
                linestyle='dashed')
    axs[0].set_xlabel('δ¹⁸O [‰ VSMOW]')
    axs[0].set_ylabel('δ¹³C [‰ VPDB]')
    axs[0].grid()
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                  fancybox=True, ncol=3)

    # Bottom subplot: δ¹⁸O vs. Age BP
    axs[1].scatter(maw_stad.age_BP, maw_stad.d18O, label='Stadial Data',
                   color='royalblue', alpha=0.8)
    axs[1].scatter(maw_intr.age_BP, maw_intr.d18O, label='Interstadial Data',
                   color='gold', alpha=0.8)
    axs[1].set_xlabel('Age BP')
    axs[1].set_ylabel('δ¹⁸O [‰ VSMOW]')
    axs[1].grid()
    # axs[1].legend(loc='best')

    plt.show()


def orthogonal_least_squares(x, y):
    """
    Perform Orthogonal Distance Regression (ODR) to fit a line to the data.
    """
    # Define the linear model
    def linear_func(params, x):
        slope, intercept = params
        return slope * x + intercept

    # Create the data and model objects
    data = Data(x, y)
    model = Model(linear_func)
    
    # Set initial guess for slope and intercept
    odr = ODR(data, model, beta0=[1.0, 0.0])
    output = odr.run()
    
    return output


def plot_combined_clusters_odr(data, maw_stad, maw_intr, odr_stad, odr_intr, odr_all):
    """
    Generates two subplots:
    - Top: Clustered δ¹⁸O-δ¹³C space with ODR regression overlays for 
    stadial and interstadial data
    - Bottom: δ¹⁸O vs. Age BP with consistent cluster coloring
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), 
                             gridspec_kw={'height_ratios': [3, 1.5]})
    plt.subplots_adjust(hspace=0.4)

    # Top subplot: δ¹⁸O-δ¹³C space with regression overlays
    range_stad = np.linspace(maw_stad.d18O.min(), maw_stad.d18O.max(), num=100)
    range_intr = np.linspace(maw_intr.d18O.min(), maw_intr.d18O.max(), num=100)
    range_all = np.linspace(min(data.d18O), max(data.d18O), num=100)

    out_stad = odr_stad.beta[0] * range_stad + odr_stad.beta[1]
    out_intr = odr_intr.beta[0] * range_intr + odr_intr.beta[1]
    out_all = odr_all.beta[0] * range_all + odr_all.beta[1]

    axs[0].scatter(maw_stad.d18O, maw_stad.d13C, label='Stadial Data',
                   color='royalblue', alpha=0.8)
    axs[0].scatter(maw_intr.d18O, maw_intr.d13C, label='Interstadial Data',
                   color='gold', alpha=0.8)
    axs[0].plot(range_stad, out_stad, label='Stadial ODR Fit' +
                f'\nSlope={odr_stad.beta[0]:.3f}', color='red',
                linestyle='dashed')
    axs[0].plot(range_intr, out_intr, label='Interstadial ODR Fit' +
                f'\nSlope={odr_intr.beta[0]:.3f}', color='blueviolet',
                linestyle='dashed')
    axs[0].plot(range_all, out_all, label='Full Record ODR Fit' +
                f'\nSlope={odr_all.beta[0]:.3f}', color='black',
                linestyle='dashed')
    axs[0].set_xlabel('δ¹⁸O [‰ VSMOW]')
    axs[0].set_ylabel('δ¹³C [‰ VPDB]')
    axs[0].grid()
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                  fancybox=True, ncol=3)

    # Bottom subplot: δ¹⁸O vs. Age BP
    axs[1].scatter(maw_stad.age_BP, maw_stad.d18O, label='Stadial Data',
                   color='royalblue', alpha=0.8)
    axs[1].scatter(maw_intr.age_BP, maw_intr.d18O, label='Interstadial Data',
                   color='gold', alpha=0.8)
    axs[1].set_xlabel('Age BP')
    axs[1].set_ylabel('δ¹⁸O [‰ VSMOW]')
    axs[1].grid()

    plt.show()


def main():
    global maw_norm
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

    plot_combined_clusters(maw, maw_stad, maw_intr, linreg_stad,
                           linreg_intr, linreg_all)
    
    
if __name__ == '__main__':
    main()
