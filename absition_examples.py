import pandas as pd # package handling data structures 
from matplotlib import pyplot as plt # plotting 
import numpy as np # math
from os import listdir 
import os
from pylab import convolve # convolution function
from analysis import intervalsBelow

plt.ion()

# parameters 
data_files = ['Sham-CHI_7DPI_2cmL.csv', 'CHI_7DPI_2cmL.csv','Sham-CHI_18DPI_2cmL.csv', 'CHI_18DPI_2cmL.csv']
base_vals = [758, 758, 660, 660]
pathname = 'data/'
pixelFactor = 1
var_name = 'LHfoot'
Fs = 60 

fig = plt.figure()

for filename, base, ind in zip(data_files, base_vals, range(len(base_vals))):
    ## load data 
    csv_file = open(pathname + filename, 'rb') 
    df = pd.read_csv(csv_file, header=[1,2], index_col=0) # read csv file
    csv_file.close()

    if filename.split('_')[0].startswith('Sham'):
        clr = 'black'
        ttl = 'Sham-CHI ' + filename.split('_')[1].split('D')[0] + ' Days Post Injury'
    else:
        clr = 'red'
        ttl = 'CHI ' + filename.split('_')[1].split('D')[0] + ' Days Post Injury'


    ### smoothing 
    binsz = 5 # N points for smoothing
    fblur = np.array([1.0/binsz for i in range(binsz)]) # boxcar 
    lh_y_smooth = convolve(df[var_name]['y'], fblur, 'same') # convolve y-pos w/ boxcar
    lh_y_smooth = (lh_y_smooth - base) * -1 # correct for y-axis being flipped 
    lh_y_smooth = np.multiply(lh_y_smooth, pixelFactor)

    time = np.linspace(0, len(df[var_name]['y'])/Fs, len(df[var_name]['y']))
    
    ints = intervalsBelow(lh_y_smooth) 

    plt.subplot(2,2,ind+1)
    plt.plot(time, lh_y_smooth, 'k')
    plt.plot([0,10], [0,0], 'k--')

    for val in ints: 
        x = time[val[0]:val[1]] 
        y = lh_y_smooth[val[0]:val[1]] 
        plt.fill_between(x, y, alpha=0.3, color=clr) 
    plt.ylim(-210,210)
    plt.xticks(fontsize=12)
    plt.yticks([-200,-100,0,100,200], fontsize=12)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Pixels', fontsize=14)
    plt.title(ttl, fontsize=16)

plt.subplot(2,2,1)
plt.xlim(0,3.3)
plt.xlabel('')
plt.subplot(2,2,2)
plt.xlim(0,10)
plt.ylabel('')
plt.subplot(2,2,3)
plt.xlim(0,9)
plt.subplot(2,2,4)
plt.xlim(0, 9.9)
plt.ylabel('')