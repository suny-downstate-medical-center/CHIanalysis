import pandas as pd # package handling data structures 
from matplotlib import pyplot as plt # plotting 
import numpy as np # math
from os import listdir 
from pylab import convolve # convolution function
import argparse
import os

# functions for analyzing leg positions 
def intervalsBelow(y):
    ## find the intervals where leg is below the beam (y < 0)
    ind = 0 # index variable 
    ints = [] # list of intervals
    while ind < len(y): # loop through lh_y_smooth
        if y[ind] < 0: # check if value at ind is less than 0 
            ind2 = ind + 1 # make second index
            while y[ind2] < 0:
                ind2 = ind2 + 1 # increment ind2 when less than zero
            ints.append((ind,ind2)) # store the interval 
            ind = ind2 + 1 # set move on to the next index
        else:
            ind = ind + 1

    return ints 
 
def negAbsement(y):
    ## compute the total negative displacement of a foot below the beam during an experiment 
    ints = intervalsBelow(y)

    ## compute the integral of each interval
    grals = [] # list of integrals 
    for interval in ints: # loop through intervals 
        grals.append(np.trapz(y[interval[0]:interval[1]])) # store trapezoidal apprx of integral 
    
    return np.sum(grals) # return sum of the integrals 

def fallMagnitudes(y):
    ## compute average fall magnitude for a trial 
    ints = intervalsBelow(y)

    mags = []
    for interval in ints:
        mags.append(np.min(y[interval[0]:interval[1]]))

    return mags 

def slipFrequency(t, y):
    ## compute the frequency of foot slips below the beam 
    ints = intervalsBelow(y)

    return (len(ints) / (t[-1]-t[0])) # return foot slip frequency in hz 

def slipNumber(y):
    ## compute the frequency of foot slips below the beam 
    ints = intervalsBelow(y)

    return len(ints)

def totalTimeBelow(t, y):
    ## compute the intervals where foot falls below the beam 
    ints = intervalsBelow(y)

    ## compute total time below the beam 
    total = 0
    for interval in ints:
        total = total + (t[interval[1]]-t[interval[0]])

    return total # in seconds 

def analyzeFile(pathname, filename, var_name='LHfoot', Fs=60, base=374, pixelFactor=1, smoothBin=5):
    ## code for checking single file's output - load data file 
    csv_file = open(pathname + filename, 'rb') # renamed your data file eggs.csv
    df = pd.read_csv(csv_file, header=[1,2], index_col=0) # read csv file
    csv_file.close()

    ### smoothing 
    binsz = smoothBin # N points for smoothing
    fblur = np.array([1.0/binsz for i in range(binsz)]) # boxcar 
    lh_y_smooth = convolve(df[var_name]['y'], fblur, 'same') # convolve y-pos w/ boxcar
    lh_y_smooth = (lh_y_smooth - base) * -1 # correct for y-axis being flipped 
    lh_y_smooth = np.multiply(lh_y_smooth, pixelFactor)

    time = np.linspace(0, len(df[var_name]['y'])/Fs, len(df[var_name]['y']))

    ints = intervalsBelow(lh_y_smooth)
    if len(ints):
        ## gen figures for slip magnitude, slip duration, total negative abasement of each slip  
        mags = fallMagnitudes(lh_y_smooth)
        durs = [] 
        total_abs = []
        slip_times = [] 
        for interval in ints:
            durs.append(time[interval[1]]-time[interval[0]])
            total_abs.append(np.trapz(lh_y_smooth[interval[0]:interval[1]]))
            slip_times.append(interval[0])
        
        plt.figure(figsize=(9,12))
        plt.subplot(3,1,1)
        plt.scatter(slip_times, mags)
        plt.xlabel('Slip Times', fontsize=16)
        plt.ylabel('Slip Magnitude', fontsize=16)
        plt.title(filename[:-4], fontsize=18)
        plt.subplot(3,1,2)
        plt.scatter(slip_times, durs)
        plt.xlabel('Slip Times', fontsize=16)
        plt.ylabel('Slip Duration', fontsize=16)
        plt.subplot(3,1,3)
        plt.scatter(slip_times, total_abs)
        plt.ylabel('Slip Absition', fontsize=16)
        plt.xlabel('Slip Times', fontsize=16)
        plt.tight_layout()
        plt.savefig(filename[:-4]+'.png')
        plt.close()

        plt.figure()
        plt.rcParams['font.size'] = '20'
        plt.plot(time, lh_y_smooth,  color='black')
        plt.ylim([-2.5, 4])
        plt.xlim([0.1, 8])
        plt.yticks([-2, -1, 0, 1, 2, 3, 4])
        plt.xlabel('Time (seconds)', fontsize=20)
        plt.ylabel('Absition', fontsize=20)
        plt.tight_layout()
        plt.savefig(filename[:-4]+'_traces.png')
        plt.close()


def analyzeFolder(pathname, var_name='LHfoot', Fs=60, base=374, pixelFactor=1, smoothBin=5):
    ## measures of interest 
    file_name = []
    total_time_below = []
    slip_frequency = []
    negative_abasement = [] 
    avg_slip_magnitude = []
    slip_number = []

    ## list data files
    file_names = listdir(pathname)

    for file_name in file_names:
        ### plot figure for single trial 
        analyzeFile(pathname, file_name, var_name, Fs, base, pixelFactor, smoothBin)

        ### load data file 
        csv_file = open(pathname + file_name, 'rb') # renamed your data file eggs.csv
        df = pd.read_csv(csv_file, header=[1,2], index_col=0) # read csv file
        csv_file.close()

        ### smoothing 
        binsz = smoothBin # N points for smoothing
        fblur = np.array([1.0/binsz for i in range(binsz)]) # boxcar 
        lh_y_smooth = convolve(df[var_name]['y'], fblur, 'same') # convolve y-pos w/ boxcar
        lh_y_smooth = (lh_y_smooth - base) * -1 # correct for y-axis being flipped 
        lh_y_smooth = np.multiply(lh_y_smooth, pixelFactor)

        time = np.linspace(0, len(df[var_name]['y'])/Fs, len(df[var_name]['y']))

        ints = intervalsBelow(lh_y_smooth)
        if len(ints):
            ### compute good shit 
            #file_name.append(file_name)
            negative_abasement.append(negAbsement(lh_y_smooth))
            total_time_below.append(totalTimeBelow(time, lh_y_smooth))
            slip_frequency.append(slipFrequency(time, lh_y_smooth))
            avg_slip_magnitude.append(np.mean(fallMagnitudes(lh_y_smooth)))
            slip_number.append(slipNumber(lh_y_smooth))

    ## collect output
    out = {'file_name' : file_names,
           'negative_abasement' : negative_abasement,
	       'total_time_below' : total_time_below,
           'slip_frequency' : slip_frequency, 
           'avg_slip_magnitude' : avg_slip_magnitude, 
           'slip_number' : slip_number}

    return out 

# user defined inputs 
try:
    parser = argparse.ArgumentParser(description = 'Running banged up rodent analysis')
    
    parser.add_argument('--variable', nargs='?', type=str, default='LHfoot',
                        help='variable of interest, defaults to LHfoot')
    parser.add_argument('--pixelFactor', nargs='?', type=float, default=1.0,
                        help='Factor converting pixels to sensible spatial unit, defaults to 1.0 (no conversion)')
    parser.add_argument('--Fs', nargs='?', type=int, default=60,
                        help='sampling frequency, defaults to 60 fps')
    parser.add_argument('--basePixel', nargs='?', type=int, default=660,
                        help='pixel corresponding to the top of the beam, defaults to 660')
    parser.add_argument('--binSize', nargs='?', type=int, default=5,
                        help='bin size for filtering, defaults to 5')
    parser.add_argument('pathname', metavar='dir', type=str,
                        help='the directory containing the data')
    args = parser.parse_args()
except:
    os._exit(1)

# main code 
out = analyzeFolder(args.pathname, var_name=args.variable, Fs=args.Fs, base=args.basePixel, pixelFactor=args.pixelFactor, smoothBin=args.binSize)
df = pd.DataFrame.from_dict(out)
df.to_csv('output.csv')