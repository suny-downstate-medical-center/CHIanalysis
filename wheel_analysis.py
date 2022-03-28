from matplotlib.collections import PathCollection
import pandas as pd # package handling data structures 
from matplotlib import pyplot as plt # plotting 
import numpy as np # math
import imageio
from pylab import convolve, fft # convolution function

from scipy.signal import decimate, find_peaks

varnames = ['left_forearm1', 'right_forearm1', 'left_hindleg1', 
            'right_hindleg1']
# varnames = ['left_forearm1', 'right_hindleg1']

def loadCSV(pathname, filename):
    with open(pathname + filename, 'rb') as fileObj: 
        df = pd.read_csv(fileObj, header=[1,2], index_col=0) # read csv file
    return df
    
def smooth(var, binsz):
    fblur = np.array([1.0/binsz for i in range(binsz)]) # boxcar 
    return convolve(var, fblur, 'same')

def trim(var, tcut, time):
    var = [w for t, w in zip(time, var) if tcut < t < time[-1]-tcut]
    return var

def relativeFoot(foot, tail, nose, time, tcut=15, binsz=72):
    center = np.divide(np.subtract(tail, nose),2)
    relative_foot = np.subtract(center, foot)
    relative_foot = smooth(relative_foot, binsz)
    # import IPython; IPython.embed()
    # relative_foot = trim(relative_foot, tcut, time)
    relative_foot = np.multiply(relative_foot, -1)
    relative_foot = np.subtract(relative_foot, np.min(relative_foot))
    return relative_foot

def plotNoseTail(pathname, filename, binsz=24, Fs=240):
    df = loadCSV(pathname, filename)
    nose_x = smooth(df['nose1']['x'], binsz)
    nose_y = smooth(df['nose1']['y'], binsz)
    tail_x = smooth(df['basetail1']['x'],binsz)
    tail_y = smooth(df['basetail1']['y'],binsz)
    time = np.linspace(0, len(nose_x)/Fs, len(nose_x))
    fig, plts = plt.subplots(nrows=3, ncols=1, sharex=True)
    plts[0].plot(time, nose_x, label='Nose', color='blue')
    plts[0].plot(time, tail_x, label='Tail', color='orange')
    plts[0].set_ylabel('X-Position (pixels)', fontsize=16)
    plts[0].legend(fontsize=14)
    plts[1].plot(time, nose_y, label='Nose', color='red')
    plts[1].plot(time, tail_y, label='Tail', color='green')
    plts[1].set_ylabel('Y-Position (pixels)', fontsize=16)
    plts[2].plot(time[1:], np.diff(nose_x), label='Nose X', color='blue')
    plts[2].plot(time[1:], np.diff(tail_x), label='Tail X', color='orange')
    plts[2].plot(time[1:], np.diff(nose_y), label='Nose Y', color='red')
    plts[2].plot(time[1:], np.diff(tail_y), label='Tail Y', color='green')
    plts[2].legend(fontsize=14)
    plts[2].set_xlabel('Time (s)', fontsize=16)
    plts[2].set_ylabel('d/dt (pixels/frame)', fontsize=16)

def plotVarX(pathname, filename, varname='left_forearm1', binsz=48, Fs=240):
    df = loadCSV(pathname, filename)
    var = smooth(df[varname]['x'], binsz)
    # var = decimate(df[varname]['x'], 10)
    time = np.linspace(0, len(var)/(Fs/10), len(var))
    dvardt = smooth(np.diff(var), binsz)
    fig, plts = plt.subplots(nrows=2, ncols=1, sharex=True)
    plts[0].plot(time, var)
    plts[0].set_ylabel('X-Position (pixels)', fontsize=16)
    plts[1].plot(time[1:], np.diff(var))
    plts[1].set_ylabel('dX/dt (pixels/frame)')
    plts[1].set_xlabel('Time (s)', fontsize=16)

def findSteps(foot, tail, nose, wheel_x, wheel_y, wheel_steady, binsz=72, Fs=240, tcut=15):
    time = np.linspace(0, len(foot)/Fs, len(foot))
    relative_foot = relativeFoot(foot, tail, nose, time)
    # trim by tcut
    wheel_x = trim(wheel_x, tcut, time)
    wheel_y = trim(wheel_y, tcut, time)
    time = trim(time, tcut, time)
    pks, props = find_peaks(relative_foot, height=80, width=int(0.3*Fs))
    intervals = []
    for start, stop in zip(pks[:-1], pks[1:]):
        trough = np.argmin(relative_foot[start:stop])
        intervals.append((start, start + trough))

    for val in intervals:
        plt.figure()
        plt.subplot(121)
        plt.plot(time[val[0]:val[1]], list(relative_foot[val[0]:val[1]]))
        plt.subplot(122)
        plt.scatter(wheel_x[val[0]+1:val[1]-1], wheel_y[val[0]+1:val[1]-1], color='orange')
        plt.plot(wheel_x[val[0]], wheel_y[val[0]], '*',color='green', markersize=12)
        plt.plot(wheel_x[val[1]], wheel_y[val[1]], '*', color='red', markersize=12)
        plt.plot(wheel_steady[0], wheel_steady[1], '*')
    # return intervals
    
def foot2center(pathname, filename, varname='left_forearm1', binsz=10, Fs=240, tcut=15):
    df = loadCSV(pathname, filename)
    foot = df[varname]['x']
    tail = df['basetail1']['x']
    nose = df['nose1']['x']
    center = np.divide(np.subtract(tail, nose),2)
    relative_foot = np.subtract(center, foot)
    relative_foot = smooth(relative_foot, binsz)
    time = np.linspace(0, len(relative_foot)/Fs, len(relative_foot))
    # trim by tcut
    relative_foot = [r for t, r in zip(time, relative_foot) if tcut < t < time[-1]-tcut]
    time = [t for t in time if tcut < t < time[-1]-tcut]
    # fix to put body at zero, steps to the left as positive
    # relative_foot = np.add(relative_foot, np.max(relative_foot))
    relative_foot = np.multiply(relative_foot, -1)
    relative_foot = np.subtract(relative_foot, np.min(relative_foot))
    pks, props = find_peaks(relative_foot, height=40, width=int(0.3*Fs))
    fig, plts = plt.subplots(nrows=1, ncols=1, sharex=True)
    plts.plot(time, relative_foot)
    plts.set_ylabel('Relative Foot X position (pixels)', fontsize=16)
    plts.set_xlabel('Time (s)', fontsize=16)
    plts.scatter([time[ind] for ind in pks], props['peak_heights'], color='red')

def addOrientation(df):
    orientation = []
    for nose, tail in zip(df['nose1']['x'], df['basetail1']['x']):
        if nose < tail:
            orientation.append('left')
        else:
            orientation.append('right')
    df['orientation'] = orientation
    return df

def addCenter(df):
    df['center', 'x'] = np.divide(np.subtract(df['basetail1']['x'], 
        df['nose1']['x']),2)  
    df['center', 'y'] = np.divide(np.subtract(df['basetail1']['y'], 
        df['nose1']['y']),2)  
    return df

def addTime(df, Fs):
    df['time'] = np.linspace(0, len(df['nose1']['x'])/Fs, len(df['nose1']['x']))
    return df

def addRelativeFootDist(df):
    for varname in varnames:
        dist = []
        for x, y, center_x, center_y in zip(df[varname]['x'],
            df[varname]['y'], df['center']['x'], df['center']['y']):
            dist.append(((x-center_x)**2+(y-center_y)**2)**(1/2))
        df[varname, 'relative_dist'] = dist
        df[varname, 'relative_x'] = df[varname]['x'] - df['center']['x']
    return df

def smoothVar(df, binsz, var='relative_dist_fix'):
    for varname in varnames:
        df[varname, var] = smooth(df[varname, var], binsz)
    return df

def plotVar(df, var='relative_x_fix'):
    for varname in varnames:
        plt.plot(df['time'], np.subtract(df[varname][var], df[varname][var][0]), label=varname)
    plt.legend()

def nanInterp(df, dxThresh=15, method='spline', order=3):
    ## spline interpolation
    for varname in varnames:
        for var in ['relative_dist', 'relative_x']:
            newvals = [df[varname][var][0]]
            dxdt = np.diff(df[varname][var])
            for ind, dx in enumerate(dxdt):
                if np.abs(dx) > dxThresh:
                    newvals.append(np.nan)
                else:
                    newvals.append(df[varname][var][ind+1])
            df[varname, var+'_fix'] = pd.Series(newvals).interpolate(method=method, order=order)
    return df

def plotVidFrames(vidfile, frames, df, nrows=2, xlims=None, points=None, likelihood=False):
    vid = imageio.get_reader(vidfile, 'ffmpeg')
    fig, sbplts = plt.subplots(int(nrows), int(len(frames)/nrows), sharex=True, sharey=True)
    for frame, ax in zip(frames, sbplts.reshape(-1)):
        image = vid.get_data(frame)
        ax.imshow(image)
        if isinstance(points,list):
            for point in points:
                if likelihood:
                    l = point + ': ' + str(round(df[point]['likelihood'][frame],3))
                    ax.plot(df[point]['x'][frame], df[point]['y'][frame], '*', label=l)
                else:
                    ax.plot(df[point]['x'][frame], df[point]['y'][frame], '*', label=point)
        ax.set_title('Frame #:' + str(frame), fontsize=12)
        ax.legend()
    if isinstance(xlims, list):
        ax.set_xlim(xlims[0], xlims[1])
    if isinstance(points, list) and not likelihood:
        ax.legend()

if __name__ == '__main__':
    # pathname = './'
    # filename = 'wheel_example.csv'
    # vidfile = pathname + 'wheel_movie.mp4'
    # pathname = 'MN12_318093/'
    # filename = 'MN12_3.csv'
    # vidfile = pathname + 'MN12_3_labeled.mp4'
    # pathname = './'
    pathname = 'Files/'
    # filename = 'IMG_5314DLC_resnet50_SCWheelSep8shuffle1_392800.csv'
    # vidfile = pathname + 'IMG_5314DLC_resnet50_SCWheelSep8shuffle1_392800_labeled.mp4'
    # filename = 'IMG_5308DLC_resnet50_SCWheelSep8shuffle1_392800.csv'
    # vidfile = pathname + 'IMG_5308DLC_resnet50_SCWheelSep8shuffle1_392800_labeled.mp4'
    filename = 'IMG_6626_1DLC_resnet50_SCWheelSep8shuffle1_646500.csv'
    # filename = '15_1DLC_resnet50_SCWheelSep8shuffle1_646500.csv'

    plt.ion()
    df = loadCSV(pathname, filename)
    df = addCenter(df)
    df = addTime(df, 240)
    df = addRelativeFootDist(df)
    df = nanInterp(df)

    for ind, var in enumerate(['relative_dist', 'relative_x']):
        fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
        plt.subplot(311)
        plt.title(var)
        plotVar(df, var=var)
        plt.subplot(312)
        plt.title(var+'_fix')
        plotVar(df, var=var+'_fix')
        plt.subplot(313)
        df = smoothVar(df, 20, var=var+'_fix')
        plotVar(df, var=var+'_fix')
        plt.title(var+'_fix smooth 20')

    ## fft 
    var = 'relative_dist'
    fig2, axs2 = plt.subplots(nrows=4, ncols=1, sharex=True)
    for ind, varname in enumerate(varnames):
        # y = df[varname][var+'_fix'] - np.mean(df[varname][var+'_fix'])
        y = [val for val, t in zip(df[varname][var+'_fix'].values, df['time'].values) if 6.5 < t < 10.0]
        y = np.array(y) - np.mean(y)
        Y = (fft(y)/len(y))[0:int(len(y)/2)]
        Freq = np.linspace(0.0, 240/2.0, len(Y))
        axs2[ind].plot(Freq, np.abs(Y))
        axs2[ind].set_title(varname)
        plt.xlim(0,20)

    # plotRelativeDist(df)

