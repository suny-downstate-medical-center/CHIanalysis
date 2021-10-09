# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:27:25 2021

@author: SIOLA
"""
import pandas as pd # package handling data structures 
from matplotlib import pyplot as plt # plotting 
import numpy as np # math
from os import listdir 
from pylab import convolve # convolution function
import argparse

#args = vars(ap.parse_args())
###import pandas as pd # package handling data structures 
#from matplotlib import pyplot as plt # plotting 
###import numpy as np # math
##plt.ion() # plots pop up as you make them (no plt.show())

import math
from itertools import combinations


csv_file = open('D:/CSV3/Simple_wheel_day_6/318509/7DLC_resnet50_SCWheelSep8shuffle1_423700.csv', 'rb') # renamed your data file eggs.csv
df = pd.read_csv(csv_file, header=[1,2], index_col=0) # r6ad csv file6
Fs = 240 # sampling frequency (fps)
#time = np.linspace(0, len(df['wheel1']['x'])/Fs, len(df['wheel1']['x'])) # time array in seconds

#base = 620.194 # this is my best guess for y-pixel for top of the beam 

# smoothing - this is a boxcar / moving average filter, simplest thing going
# especially if there's going to be missing data (not sure how the tracking
# handles when foot's out of frame), you will want to switch to something fancier.

#binsz = 5 # N points for smoothing
#fblur = np.array([1.0/binsz for i in range(binsz)]) # boxcar 
#lh_y_smooth = convolve(df['wheel1']['x'], fblur, 'same') # convolve y-pos w/ boxcar

#lh_y_smooth = (lh_y_smooth - base) * -1 # correct for y-axis being flipped 


#y100 = (df['wheel2']['y'])
    
x1 = (df['wheel1']['x'])
x2 = (df['wheel2']['x'])
y1 = (df['wheel1']['y'])
y2 = (df['wheel2']['y'])

x3 = (df['left_forearm2']['x'])
x4 = (df['right_forearm2']['x'])
y3 = (df['left_forearm2']['y'])
y4 = (df['right_forearm2']['y'])

x5 = (df['left_hindleg2']['x'])
x6 = (df['right_hindleg2']['x'])
y5 = (df['left_hindleg2']['y'])
y6 = (df['right_hindleg2']['y'])

x11 = (df['left_forearm1']['x'])
x12 = (df['right_hindleg1']['x'])
y11 = (df['left_forearm1']['y'])
y12 = (df['right_hindleg1']['y'])

x13 = (df['right_forearm1']['x'])
x14 = (df['left_hindleg1']['x'])
y13 = (df['right_forearm1']['y'])
y14 = (df['left_hindleg1']['y'])

x15 = (df['left_forearm1']['x'])
x16 = (df['left_hindleg1']['x'])
y15 = (df['left_forearm1']['y'])
y16 = (df['left_hindleg1']['y'])

x17 = (df['right_forearm1']['x'])
x18 = (df['right_hindleg1']['x'])
y17 = (df['right_forearm1']['y'])
y18 = (df['right_hindleg1']['y'])

x19 = (df['nose1']['x'])
x20 = (df['basetail1']['x'])
y19 = (df['nose1']['y'])
y20 = (df['basetail1']['y'])

y100 = (df['wheel2']['y'])

def distance(p1, p2):   
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

dists0 = [distance((xa, ya), (xb, yb)) for xa, ya, xb, yb in zip(x1, y1, x2, y2)]

dists1 = [distance((xa, ya), (xb, yb)) for xa, ya, xb, yb in zip(x3, y3, x4, y4)]

dists2 = [distance((xa, ya), (xb, yb)) for xa, ya, xb, yb in zip(x5, y5, x6, y6)]

dists5 = [distance((xa, ya), (xb, yb)) for xa, ya, xb, yb in zip(x11, y11, x12, y12)]

dists6 = [distance((xa, ya), (xb, yb)) for xa, ya, xb, yb in zip(x13, y13, x14, y14)]

dists7 = [distance((xa, ya), (xb, yb)) for xa, ya, xb, yb in zip(x15, y15, x16, y16)]

dists8 = [distance((xa, ya), (xb, yb)) for xa, ya, xb, yb in zip(x17, y17, x18, y18)]

dists9 = [distance((xa, ya), (xb, yb)) for xa, ya, xb, yb in zip(x19, y19, x20, y20)]

print(np.max(y100))

eq = np.mean(dists0)/6.2
print(eq)
eq1 = eq*2.54
eq2 = 2.54/eq1

print(eq2) # centimeter value of 1 pixel
#print(np.max(y0))

#######################################
## --------------Upper limb absition in cm*seconds and pixels*seconds------------
#######################################
Fs = 240 # sampling frequency (fps)
time2 = np.linspace(0, len(df['left_forearm1']['y'])/Fs, len(df['left_forearm1']['y'])) # time array in seconds
#time3 = np.linspace(0, len((df['right_forearm1']['y']) * eq2) /Fs, len(df['right_forearm1']['y'])) # time array in seconds

base = max(y100) # this is my best guess for y-pixel for top of the beam 

# smoothing - this is a boxcar / moving average filter, simplest thing going
# especially if there's going to be missing data (not sure how the tracking
# handles when foot's out of frame), you will want to switch to something fancier.
binsz = 5 # N points for smoothing
fblur = np.array([1.0/binsz for i in range(binsz)]) # boxcar 

lh_y_smooth2 = convolve(df['left_forearm1']['y'], fblur, 'same') # convolve y-pos w/ boxcar

## centimeter conversion 
lh_y_smooth2 = ((lh_y_smooth2 - base) * -1) * eq2 # correct for y-axis being flipped 

# plotting position vs. time 
#plt.figure()
#plt.plot(time2, lh_y_smooth2)
#plt.xlabel('Time (s)')
#plt.ylabel('Y Position (centimeters)')

def intervalsBelow2(y21):
    ## find the intervals where leg is below the beam (y < 0)
    ind = 0 # index variable 
    ints2 = [] # list of intervals
    while ind < len(y21): # loop through lh_y_smooth
        if y21[ind] < 0: # check if value at ind is less than 0 
            ind2 = ind + 1 # make second index
            while y21[ind2] < 0:
                ind2 = ind2 + 1 # increment ind2 when less than zero
            ints2.append((ind,ind2)) # store the interval 
            ind = ind2 + 1 # set move on to the next index
        else:
            ind = ind + 1

    return ints2 

# computing "negative absement" - time integral of y position under the beam 
def negAbsement2(y21):

    ints2 = intervalsBelow2(y21)

    ## compute the integral of each interval
    grals = [] # list of integrals 
    for interval in ints2: # loop through intervals 
        grals.append(np.trapz(y21[interval[0]:interval[1]])) # store trapezoidal apprx of integral 
    
    return np.sum(grals) # return sum of the integrals 

totalBelow2 = negAbsement2(lh_y_smooth2) # call to above function
#print('Left forearm absition: ' + str(np.round(totalBelow2,2)) + ' (cm*seconds)')
print(str(np.round(totalBelow2,2)))
##########################################
#######################################
Fs = 240 # sampling frequency (fps)
time3 = np.linspace(0, len(df['left_forearm1']['y'])/Fs, len(df['left_forearm1']['y'])) # time array in seconds
#time3 = np.linspace(0, len((df['right_forearm1']['y']) * eq2) /Fs, len(df['right_forearm1']['y'])) # time array in seconds

#base = max(y1) # this is my best guess for y-pixel for top of the beam 

# smoothing - this is a boxcar / moving average filter, simplest thing going
# especially if there's going to be missing data (not sure how the tracking
# handles when foot's out of frame), you will want to switch to something fancier.
binsz = 5 # N points for smoothing
fblur = np.array([1.0/binsz for i in range(binsz)]) # boxcar 

lh_y_smooth3 = convolve(df['left_forearm1']['y'], fblur, 'same') # convolve y-pos w/ boxcar

## centimeter conversion 
lh_y_smooth3 = (lh_y_smooth3 - base) * -1 # correct for y-axis being flipped 

# plotting position vs. time 
#plt.figure()
#plt.plot(time3, lh_y_smooth3)
#plt.xlabel('Time (s)')
#plt.ylabel('Y Position (pixels)')

def intervalsBelow3(y30):
    ## find the intervals where leg is below the beam (y < 0)
    ind = 0 # index variable 
    ints3 = [] # list of intervals
    while ind < len(y30): # loop through lh_y_smooth
        if y30[ind] < 0: # check if value at ind is less than 0 
            ind2 = ind + 1 # make second index
            while y30[ind2] < 0:
                ind2 = ind2 + 1 # increment ind2 when less than zero
            ints3.append((ind,ind2)) # store the interval 
            ind = ind2 + 1 # set move on to the next index
        else:
            ind = ind + 1

    return ints3 

# computing "negative absement" - time integral of y position under the beam 
def negAbsement3(y30):

    ints3 = intervalsBelow3(y30)

    ## compute the integral of each interval
    grals = [] # list of integrals 
    for interval in ints3: # loop through intervals 
        grals.append(np.trapz(y30[interval[0]:interval[1]])) # store trapezoidal apprx of integral 
    
    return np.sum(grals) # return sum of the integrals 

totalBelow3 = negAbsement3(lh_y_smooth3) # call to above function
#print('Left forearm absition: ' + str(np.round(totalBelow3,2)) + ' (pixels*seconds)')
print(str(np.round(totalBelow3,2)))
##########################################

Fs = 240 # sampling frequency (fps)
time4 = np.linspace(0, len(df['right_forearm1']['y'])/Fs, len(df['right_forearm1']['y'])) # time array in seconds
#time3 = np.linspace(0, len((df['right_forearm1']['y']) * eq2) /Fs, len(df['right_forearm1']['y'])) # time array in seconds

#base = max(y1) # this is my best guess for y-pixel for top of the beam 

# smoothing - this is a boxcar / moving average filter, simplest thing going
# especially if there's going to be missing data (not sure how the tracking
# handles when foot's out of frame), you will want to switch to something fancier.
binsz = 5 # N points for smoothing
fblur = np.array([1.0/binsz for i in range(binsz)]) # boxcar 

lh_y_smooth4 = convolve(df['right_forearm1']['y'], fblur, 'same') # convolve y-pos w/ boxcar

lh_y_smooth4 = ((lh_y_smooth4 - base) * -1) * eq2 # correct for y-axis being flipped 

# plotting position vs. time 
#plt.figure()
#plt.plot(time4, lh_y_smooth4)
#plt.xlabel('Time (s)')
#plt.ylabel('Y Position (centimeters)')

def intervalsBelow4(y40):
    ## find the intervals where leg is below the beam (y < 0)
    ind = 0 # index variable 
    ints4 = [] # list of intervals
    while ind < len(y40): # loop through lh_y_smooth
        if y40[ind] < 0: # check if value at ind is less than 0 
            ind2 = ind + 1 # make second index
            while y40[ind2] < 0:
                ind2 = ind2 + 1 # increment ind2 when less than zero
            ints4.append((ind,ind2)) # store the interval 
            ind = ind2 + 1 # set move on to the next index
        else:
            ind = ind + 1

    return ints4 

# computing "negative absement" - time integral of y position under the beam 
def negAbsement4(y40):

    ints4 = intervalsBelow4(y40)

    ## compute the integral of each interval
    grals = [] # list of integrals 
    for interval in ints4: # loop through intervals 
        grals.append(np.trapz(y40[interval[0]:interval[1]])) # store trapezoidal apprx of integral 
    
    return np.sum(grals) # return sum of the integrals 

totalBelow4 = negAbsement4(lh_y_smooth4) # call to above function
#print('Right forearm absition: ' + str(np.round(totalBelow4,2)) + ' (cm*seconds)')
print(str(np.round(totalBelow4,2)))

#######################################

Fs = 240 # sampling frequency (fps)
time5 = np.linspace(0, len(df['right_forearm1']['y'])/Fs, len(df['right_forearm1']['y'])) # time array in seconds

#base = max(y1) # this is my best guess for y-pixel for top of the beam 

# smoothing - this is a boxcar / moving average filter, simplest thing going
# especially if there's going to be missing data (not sure how the tracking
# handles when foot's out of frame), you will want to switch to something fancier.
binsz = 5 # N points for smoothing
fblur = np.array([1.0/binsz for i in range(binsz)]) # boxcar 

lh_y_smooth5 = convolve(df['right_forearm1']['y'], fblur, 'same') # convolve y-pos w/ boxcar

lh_y_smooth5 = (lh_y_smooth5 - base) * -1 # correct for y-axis being flipped 

# plotting position vs. time 
#plt.figure()
#plt.plot(time5, lh_y_smooth5)
#plt.xlabel('Time (s)')
#plt.ylabel('Y Position (pixels)')

def intervalsBelow5(y50):
    ## find the intervals where leg is below the beam (y < 0)
    ind = 0 # index variable 
    ints5 = [] # list of intervals
    while ind < len(y50): # loop through lh_y_smooth
        if y50[ind] < 0: # check if value at ind is less than 0 
            ind2 = ind + 1 # make second index
            while y50[ind2] < 0:
                ind2 = ind2 + 1 # increment ind2 when less than zero
            ints5.append((ind,ind2)) # store the interval 
            ind = ind2 + 1 # set move on to the next index
        else:
            ind = ind + 1

    return ints5 
# computing "negative absement" - time integral of y position under the beam 
def negAbsement5(y50):

    ints5 = intervalsBelow5(y50)

    ## compute the integral of each interval
    grals = [] # list of integrals 
    for interval in ints5: # loop through intervals 
        grals.append(np.trapz(y50[interval[0]:interval[1]])) # store trapezoidal apprx of integral 
    
    return np.sum(grals) # return sum of the integrals 

totalBelow5 = negAbsement5(lh_y_smooth5) # call to above function
#print('Right forearm absition: ' + str(np.round(totalBelow5,2)) + ' (pixels*seconds)')
print(str(np.round(totalBelow5,2)))
#######################################################################
#---------------Hindlimb absition in 
#######################################################################

Fs = 240 # sampling frequency (fps)
time6 = np.linspace(0, len(df['left_hindleg1']['y'])/Fs, len(df['left_hindleg1']['y'])) # time array in seconds
binsz = 5 # N points for smoothing
fblur = np.array([1.0/binsz for i in range(binsz)]) # boxcar 

lh_y_smooth6 = convolve(df['left_hindleg1']['y'], fblur, 'same') # convolve y-pos w/ boxcar

## centimeter conversion 
lh_y_smooth6 = ((lh_y_smooth6 - base) * -1) * eq2 # correct for y-axis being flipped 


def intervalsBelow6(y60):
    ## find the intervals where leg is below the beam (y < 0)
    ind = 0 # index variable 
    ints6 = [] # list of intervals
    while ind < len(y60): # loop through lh_y_smooth
        if y60[ind] < 0: # check if value at ind is less than 0 
            ind2 = ind + 1 # make second index
            while y60[ind2] < 0:
                ind2 = ind2 + 1 # increment ind2 when less than zero
            ints6.append((ind,ind2)) # store the interval 
            ind = ind2 + 1 # set move on to the next index
        else:
            ind = ind + 1

    return ints6 

# computing "negative absement" - time integral of y position under the beam 
def negAbsement6(y60):

    ints6 = intervalsBelow6(y60)

    ## compute the integral of each interval
    grals = [] # list of integrals 
    for interval in ints6: # loop through intervals 
        grals.append(np.trapz(y60[interval[0]:interval[1]])) # store trapezoidal apprx of integral 
    
    return np.sum(grals) # return sum of the integrals 

totalBelow6 = negAbsement6(lh_y_smooth6) # call to above function
#print('Left hindleg absition: ' + str(np.round(totalBelow6,2)) + ' (cm*seconds)')
print(str(np.round(totalBelow6,2)))
##########################################
#######################################
Fs = 240 # sampling frequency (fps)
time7 = np.linspace(0, len(df['left_hindleg1']['y'])/Fs, len(df['left_hindleg1']['y'])) # time array in seconds
#time3 = np.linspace(0, len((df['right_forearm1']['y']) * eq2) /Fs, len(df['right_forearm1']['y'])) # time array in seconds

#base = max(y1) # this is my best guess for y-pixel for top of the beam 

# smoothing - this is a boxcar / moving average filter, simplest thing going
# especially if there's going to be missing data (not sure how the tracking
# handles when foot's out of frame), you will want to switch to something fancier.
binsz = 5 # N points for smoothing
fblur = np.array([1.0/binsz for i in range(binsz)]) # boxcar 

lh_y_smooth7 = convolve(df['left_hindleg1']['y'], fblur, 'same') # convolve y-pos w/ boxcar

## centimeter conversion 
lh_y_smooth7 = (lh_y_smooth7 - base) * -1 # correct for y-axis being flipped 

# plotting position vs. time 
#plt.figure()
#plt.plot(time7, lh_y_smooth7)
#plt.xlabel('Time (s)')
#plt.ylabel('Y Position (pixels)')

def intervalsBelow7(y70):
    ## find the intervals where leg is below the beam (y < 0)
    ind = 0 # index variable 
    ints7 = [] # list of intervals
    while ind < len(y70): # loop through lh_y_smooth
        if y70[ind] < 0: # check if value at ind is less than 0 
            ind2 = ind + 1 # make second index
            while y70[ind2] < 0:
                ind2 = ind2 + 1 # increment ind2 when less than zero
            ints7.append((ind,ind2)) # store the interval 
            ind = ind2 + 1 # set move on to the next index
        else:
            ind = ind + 1

    return ints7 

# computing "negative absement" - time integral of y position under the beam 
def negAbsement7(y70):

    ints7 = intervalsBelow7(y70)

    ## compute the integral of each interval
    grals = [] # list of integrals 
    for interval in ints7: # loop through intervals 
        grals.append(np.trapz(y70[interval[0]:interval[1]])) # store trapezoidal apprx of integral 
    
    return np.sum(grals) # return sum of the integrals 

totalBelow7 = negAbsement7(lh_y_smooth7) # call to above function
#print('Left hindleg absition: ' + str(np.round(totalBelow7,2)) + ' (pixels*seconds)')
print(str(np.round(totalBelow7,2)))
##########################################

Fs = 240 # sampling frequency (fps)
time8 = np.linspace(0, len(df['right_hindleg1']['y'])/Fs, len(df['right_hindleg1']['y'])) # time array in seconds
#time3 = np.linspace(0, len((df['right_forearm1']['y']) * eq2) /Fs, len(df['right_forearm1']['y'])) # time array in seconds

#base = max(y1) # this is my best guess for y-pixel for top of the beam 

# smoothing - this is a boxcar / moving average filter, simplest thing going
# especially if there's going to be missing data (not sure how the tracking
# handles when foot's out of frame), you will want to switch to something fancier.
binsz = 5 # N points for smoothing
fblur = np.array([1.0/binsz for i in range(binsz)]) # boxcar 

lh_y_smooth8 = convolve(df['right_hindleg1']['y'], fblur, 'same') # convolve y-pos w/ boxcar

lh_y_smooth8 = ((lh_y_smooth8 - base) * -1) * eq2 # correct for y-axis being flipped 

# plotting position vs. time 
#plt.figure()
#plt.plot(time8, lh_y_smooth8)
#plt.xlabel('Time (s)')
#plt.ylabel('Y Position (centimeters)')

def intervalsBelow8(y80):
    ## find the intervals where leg is below the beam (y < 0)
    ind = 0 # index variable 
    ints8 = [] # list of intervals
    while ind < len(y80): # loop through lh_y_smooth
        if y80[ind] < 0: # check if value at ind is less than 0 
            ind2 = ind + 1 # make second index
            while y80[ind2] < 0:
                ind2 = ind2 + 1 # increment ind2 when less than zero
            ints8.append((ind,ind2)) # store the interval 
            ind = ind2 + 1 # set move on to the next index
        else:
            ind = ind + 1

    return ints8 

# computing "negative absement" - time integral of y position under the beam 
def negAbsement8(y80):

    ints8 = intervalsBelow8(y80)

    ## compute the integral of each interval
    grals = [] # list of integrals 
    for interval in ints8: # loop through intervals 
        grals.append(np.trapz(y80[interval[0]:interval[1]])) # store trapezoidal apprx of integral 
    
    return np.sum(grals) # return sum of the integrals 

totalBelow8 = negAbsement8(lh_y_smooth8) # call to above function
#print('Right hindleg absition: ' + str(np.round(totalBelow8,2)) + ' (cm*seconds)')
print(str(np.round(totalBelow8,2)))
#######################################
eq = np.mean(dists0)/6.2
eq1 = eq*2.54
eq2 = 2.54/eq1

Fs = 240 # sampling frequency (fps)
time9 = np.linspace(0, len(df['right_hindleg1']['y'])/Fs, len(df['right_hindleg1']['y'])) # time array in seconds

#base = max(y1) # this is my best guess for y-pixel for top of the beam 

# smoothing - this is a boxcar / moving average filter, simplest thing going
# especially if there's going to be missing data (not sure how the tracking
# handles when foot's out of frame), you will want to switch to something fancier.
binsz = 5 # N points for smoothing
fblur = np.array([1.0/binsz for i in range(binsz)]) # boxcar 

lh_y_smooth9 = convolve(df['right_hindleg1']['y'], fblur, 'same') # convolve y-pos w/ boxcar

lh_y_smooth9 = (lh_y_smooth9 - base) * -1 # correct for y-axis being flipped 

# plotting position vs. time 
#plt.figure()
#plt.plot(time9, lh_y_smooth9)
#plt.xlabel('Time (s)')
#plt.ylabel('Y Position (pixels)')

def intervalsBelow9(y90):
    ## find the intervals where leg is below the beam (y < 0)
    ind = 0 # index variable 
    ints9 = [] # list of intervals
    while ind < len(y90): # loop through lh_y_smooth
        if y90[ind] < 0: # check if value at ind is less than 0 
            ind2 = ind + 1 # make second index
            while y90[ind2] < 0:
                ind2 = ind2 + 1 # increment ind2 when less than zero
            ints9.append((ind,ind2)) # store the interval 
            ind = ind2 + 1 # set move on to the next index
        else:
            ind = ind + 1

    return ints9 
# computing "negative absement" - time integral of y position under the beam 
def negAbsement9(y90):

    ints9 = intervalsBelow9(y90)

    ## compute the integral of each interval
    grals = [] # list of integrals 
    for interval in ints9: # loop through intervals 
        grals.append(np.trapz(y90[interval[0]:interval[1]])) # store trapezoidal apprx of integral 
    
    return np.sum(grals) # return sum of the integrals 

totalBelow9 = negAbsement9(lh_y_smooth9) # call to above function
#print('Right hindleg absition: ' + str(np.round(totalBelow9,2)) + ' (pixels*seconds)')
print(str(np.round(totalBelow9,2)))



print('dist wheel')
print(np.mean(dists0)) #calculateDistance()
print(np.min(dists0))
print(np.max(dists0))
print(np.std(dists0))
#print(np.median(dist))


print('dist forearms2')
print(np.mean(dists1)) #calculateDistance()
print(np.min(dists1))
print(np.max(dists1))
print(np.std(dists1))

print('dist hindlegs2')
print(np.mean(dists2)) #calculateDistance()
print(np.min(dists2))
print(np.max(dists2))
print(np.std(dists2))

print('dist LF1 RHL1')
print(np.mean(dists5)) #calculateDistance()
print(np.min(dists5))
print(np.max(dists5))
print(np.std(dists5))

print('dist RF1 LH1')
print(np.mean(dists6)) #calculateDistance()
print(np.min(dists6))
print(np.max(dists6))
print(np.std(dists6))

print('dist L F+H')
print(np.mean(dists7)) #calculateDistance()
print(np.min(dists7))
print(np.max(dists7))
print(np.std(dists7))

print('dist R F+H')
print(np.mean(dists8)) #calculateDistance()
print(np.min(dists8))
print(np.max(dists8))
print(np.std(dists8))

print('dist nose+tail')
print(np.mean(dists9)) #calculateDistance()
print(np.min(dists9))
print(np.max(dists9))
print(np.std(dists9))



