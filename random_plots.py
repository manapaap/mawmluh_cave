# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 17:21:41 2023

@author: Aakas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Plot 1: Heinrich event 4

maw_3_h4 = records['maw_3_clean'].query('37000 < age_BP < 41000')
# maw_3_h4 = records['ngrip'].query('37000 < age_BP < 41000')

plt.figure(1)
plt.plot(maw_3_h4['age_BP'], maw_3_h4['d18O'])
plt.title('Heinrich Event 4')
plt.xlabel('Age (Years BP)')
plt.ylabel('δ¹⁸O (‰)')
plt.grid()
plt.ylim(-5.5, -0.5)
plt.gca().invert_yaxis()


# Plot 2: Heinrich event 3

maw_3_h3 = records['maw_3_clean'].query('28500 < age_BP < 31500')

plt.figure(2)
plt.plot(maw_3_h3['age_BP'], maw_3_h3['d13C'])
plt.title('Heinrich Event 3')
plt.xlabel('Age (Years BP)')
plt.ylabel('δ¹⁸O (‰)')
plt.grid()
# plt.ylim(-5.5, -0.5)
plt.gca().invert_yaxis()


# Plot 3: Just the MAW-3 oxygen record to compare to wavelets

plt.figure(3, figsize=(6, 2.5))
plt.plot(records['maw_3_clean']['age_BP'], records['maw_3_clean']['d18O'])
plt.title('MAW-3 Oxygen Record')
plt.ylabel('δ¹⁸O (‰)')
plt.xlabel('Age (Years BP)')
plt.ylim(-6, 0)
plt.xlim(28000, 45000)
plt.gca().invert_yaxis()
plt.grid()





