# Mawmluh Cave Paleoclimate Reconstruction

Hello everyone! This are all the scripts I used to process and analyze the data from "A stalagmite from NE India reveals atmospheric teleconnections to monsoons during Dansgaard Oeschger Events". The work began during my undergrad at Vanderbilt as a part of my honors thesis in chemistry. Since then, I've gradually worked a thesis into a manuscript. These scripts are not the most organized, but do contain everything necessary to replicate my plots and results, once the raw data has been obtained. I will describe what each file does to help anyone who wants to try that. 

### cluster_tests.py

This file does the clistering analysis that showed that d18O and d13C are more strongly correlated during warm D-O stadials than the cold interstadials. 

### cross_corr_tests.py

This file does the cross-correlation analysis that allows for teleconnections to be inferred from the lags between records. 

### enso_testing.py

This file calculates PSDs for the high-resolution segments of MAW-3, from which we inferred ENSO activity/teleconnections on the monsoon. 

### freq_analysis.m

This file calculates the wavelet scalogram for the MAW-3 record, showing the 1490-odd periodicity corresponding to D-O events. Unlike the other files, this is in matlab, as I failed to get it to work correctly in Python (see the attempt in freq_analysis.py). 

### plotting.py

This file contains the main plotting functions that I use for the paper. The record stacks and maps in particular. 

All other files are deprecated and haven't been used for a while. I am retaining them here for completeness. 


