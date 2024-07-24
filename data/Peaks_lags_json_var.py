from matplotlib.collections import PathCollection
import pandas as pd # package handling data structures 
from matplotlib import pyplot as plt # plotting 
import numpy as np # math
import imageio
from pylab import convolve, fft # convolution function
from scipy.signal import find_peaks, correlate, correlation_lags

varnames = ['left_forearm1', 'right_forearm1', 'left_hindleg1', 
            'right_hindleg1']

limb_pairs = [('left_forearm1', 'left_hindleg1'), ('right_forearm1', 'right_hindleg1'), 
            ('left_forearm1', 'right_hindleg1'), ('right_forearm1', 'left_hindleg1')]
rl_pair = limb_pairs[3]
lr_pair = limb_pairs[2]
# varnames = ['left_forearm1', 'right_hindleg1']

def loadCSV(pathname, filename):
    with open(pathname + filename, 'rb') as fileObj: 
        df = pd.read_csv(fileObj, header=[1,2], index_col=0) # read csv file
    return df

def loadEpochs(pathname, filename):
    with open(pathname + filename, 'rb') as fileObj:
        df = pd.read_csv(fileObj, header=[0], index_col=0)
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

def allRunningEpochs(data_dir, folders, epoch_file='epoch_data.csv'):
    # basic statistics on running epochs 
    out = {'avg' : [], 'std' : [], 'N' : [], 'err' : []}
    for folder in folders:
        pathname = data_dir + folder
        epoch_data = loadEpochs(pathname, epoch_file)
        all_epochs = []
        for ind, filename in enumerate(epoch_data['filename']):
            exec('global running_epochs; running_epochs = ' + epoch_data['running_epochs'][ind])
            all_epochs.extend(running_epochs)
        durs = [ep[1]-ep[0] for ep in all_epochs]
        out['avg'].append(np.mean(durs))
        out['err'].append(np.std(durs) / len(durs))
        out['N'].append(len(durs))
        out['std'].append(np.std(durs))
    return out 

def longestRunningEpoch(pathname, epoch_file, fps=240, var='relative_dist'):
    ## load epoch file 
    epoch_data = loadEpochs(pathname, epoch_file)
    ## get all epochs and associated files
    all_epochs = []
    which_file = []
    for ind, filename in enumerate(epoch_data['filename']):
        exec('global running_epochs; running_epochs = ' + epoch_data['running_epochs'][ind])
        all_epochs.extend(running_epochs)
        which_file.extend(filename for i in running_epochs)
    ## find the longest running epoch 
    epoch_durs = [ep[1]-ep[0] for ep in all_epochs]
    lngst_ind = len(epoch_data) - 1 #np.argmax(epoch_durs) #now last epoch
    filename = which_file[lngst_ind]
    epoch = all_epochs[lngst_ind]
    T = epoch[1] - epoch[0]
    ## preprocessing 
    df = loadCSV(pathname, filename)
    df = addCenter(df)
    df = addTime(df, fps)
    df = addRelativeFootDist(df)
    df = nanInterp(df)
    df = smoothVar(df, 20, var=var+'_fix')
    ## correlations of each limb pair for epoch 
    corr_peaks = {}
    corr_peak_lags = {}
    for pairind, pair in enumerate(limb_pairs):
        y0 = [val for val, t in zip(df[pair[0]][var+'_fix'].values, df['time'].values) if epoch[0] < t < epoch[1]]
        y0 = y0 - np.mean(y0)
        y0 = y0 / np.max(y0)
        y1 = [val for val, t in zip(df[pair[1]][var+'_fix'].values, df['time'].values) if epoch[0] < t < epoch[1]]
        y1 = y1 - np.mean(y1)
        y1 = y1 / np.max(y1)
        corr = correlate(y0, y1)
        lags = correlation_lags(len(y0), len(y1))
        lags = [l/fps for l in lags]
        corr_peaks[pair] = np.max(corr) / T
        corr_peak_lags[pair] = lags[np.argmax(corr)]
        # plt.figure()
        # plt.plot(lags,corr)
        # plt.title(str(pair))
    return corr_peaks, corr_peak_lags 

def plotAllEpochs(pathname, epoch_file, fps=240, var='relative_dist'):
    epoch_data = loadEpochs(pathname, epoch_file)
    for ind, filename in enumerate(epoch_data['filename']):
        df = loadCSV(pathname, filename)
        df = addCenter(df)
        df = addTime(df, fps)
        df = addRelativeFootDist(df)
        df = nanInterp(df)
        df = smoothVar(df, 20, var=var+'_fix')
        exec('global running_epochs; running_epochs = ' + epoch_data['running_epochs'][ind])
        for epoch in running_epochs:
            plt.figure()
            plt.subplot(212)
            # for idx, foot in enumerate(varnames):
            #     plt.subplot(220+idx+1)
            #     y0 = [val for val, t in zip(df[foot][var+'_fix'].values, df['time'].values) if epoch[0] < t < epoch[1]]
            #     plt.plot(y0)

def analyzeEpochedFolder(pathname, epoch_file, fps=240, var='relative_dist'):
    epoch_data = loadEpochs(pathname, epoch_file)
    out = {}
    for ind, filename in enumerate(epoch_data['filename']):
        print('processing file: ' + filename)
        out[filename] = {'corr_peaks' : [], 
                        'corr_peak_lags': [], 
                        'step_freqs' : [],
                        'epoch_durs' : []}
        ## preprocessing 
        df = loadCSV(pathname, filename)
        df = addCenter(df)
        df = addTime(df, fps)
        df = addRelativeFootDist(df)
        df = nanInterp(df)
        df = smoothVar(df, 20, var=var+'_fix')
        exec('global running_epochs; running_epochs = ' + epoch_data['running_epochs'][ind])
        for epoch in running_epochs:
            ## epoch duration 
            print('epoch: ' + str(epoch))
            out[filename]['epoch_durs'].append(epoch[1]-epoch[0])
            ## correlation analysis 
            corr_peaks = {}
            corr_peak_lags = {}
            T = epoch[1] - epoch[0]
            for pairind, pair in enumerate(limb_pairs):
                y0 = [val for val, t in zip(df[pair[0]][var+'_fix'].values, df['time'].values) if epoch[0] < t < epoch[1]]
                y0 = y0 - np.mean(y0)
                y0 = y0 / np.max(y0)
                y1 = [val for val, t in zip(df[pair[1]][var+'_fix'].values, df['time'].values) if epoch[0] < t < epoch[1]]
                y1 = y1 - np.mean(y1)
                y1 = y1 / np.max(y1)
                corr = correlate(y0, y1)
                lags = correlation_lags(len(y0), len(y1))
                lags = [l/fps for l in lags]
                # plt.figure()
                # plt.subplot(311)
                # plt.title(str(pair) + ' ' + str(epoch))
                # plt.plot(y0)
                # plt.subplot(312)
                # plt.plot(y1)
                # plt.subplot(313)
                # plt.plot(lags, corr)
                corr_peaks[pair] = np.max(corr) / T
                corr_peak_lags[pair] = lags[np.argmax(corr)]
            out[filename]['corr_peaks'].append(corr_peaks)
            out[filename]['corr_peak_lags'].append(corr_peak_lags)
            ## FFT / step frequency 
            step_freqs = {}
            ### step variability
            step_var = {}
            for ind, varname in enumerate(varnames):
                y0 = [val for val, t in zip(df[varname][var+'_fix'].values, df['time'].values) if epoch[0] < t < epoch[1]]
                y0 = y0 - np.mean(y0)
                y0 = y0 / np.max(y0)
                Y = (fft(y0)/len(y0))[0:int(len(y0)/2)]
               
                Freq = np.linspace(0.0, fps/2.0, len(Y))
                #import IPython; IPython.embed()
                step_freqs[varname] = Freq[np.argmax(np.abs(Y))]
                Y = Y[Freq <= 10]
                step_var[varname] = np.std(np.abs(Y))
            out[filename]['step_freqs'].append(step_freqs)
            out[filename]['step_var'].append(step_var)
                        
            #for ind, varname in enumerate(varnames):
                #y0 = [val for val, t in zip(df[varname][var+'_fix'].values, df['time'].values) if epoch[0] < t < epoch[1]]
               # y0 = y0 - np.mean(y0)
               # y0 = y0 / np.max(y0)
               # y0 = y0 - np.std(y0)
               # Y = (fft(y0)/len(y0))[0:int(len(y0)/2)]
               # Var = np.linspace(0.0, fps/2.0, len(Y))
               # step_var[varname] = Var[np.argmax(Y)]
            #out[filename]['step_var'].append(step_var)
        # plt.show()
    return out

def corrLongestEpoch(data_dir, folders, epoch_file='epoch_data.csv'):
    print('Analyzing correlations for longest running epochs of each animal:')
    out = {rl_pair : {'peaks' : [], 'lags' : []}, 
           lr_pair : {'peaks' : [], 'lags' : []}}
    for folder in folders:
        print(folder)
        pathname = data_dir + folder
        pks, lags = longestRunningEpoch(pathname, epoch_file)
        out[rl_pair]['peaks'].append(pks[rl_pair])
        out[lr_pair]['peaks'].append(pks[lr_pair])
        out[rl_pair]['lags'].append(lags[rl_pair])
        out[lr_pair]['lags'].append(lags[lr_pair])
    return out 

def allAnimalPeakLagDifs(data):
    peak_lag_difs = []
    for filename in data.keys():
        for trial in data[filename]['corr_peak_lags']:
            rl = trial[rl_pair]
            lr = trial[lr_pair]
            peak_lag_difs.append(lr-rl)
    return peak_lag_difs

def extractPhaseMetrics(data):
    out = {'lag_difs' : [],
            'rl_lags' : [],
            'lr_lags' : [],
            'rl_peaks' : [],
            'lr_peaks' : [],
            'peak_difs' : []}
    for f in data.keys():
        for ep in data[f]['corr_peak_lags']:
            rl = ep[rl_pair]
            lr = ep[lr_pair]
            out['rl_lags'].append(rl)
            out['lr_lags'].append(lr)
            out['lag_difs'].append(lr-rl)
        for ep in data[f]['corr_peaks']:
            rl = ep[rl_pair]
            lr = ep[lr_pair]
            out['rl_peaks'].append(rl)
            out['lr_peaks'].append(lr)
            out['peak_difs'].append(lr-rl)
    return out

# if __name__ == '__main__':
#     epoch_file = 'epoch_data.csv' # 'epoch_data_chi.csv'
#     cmplx_sham_dir = 'wheel_data/complex_wheel_day_3/Sham/'
#     cmplx_sham_folders = ['09/', '10/', '11/', '12/']
#     cmplx_chi_dir = 'wheel_data/complex_wheel_day_3/CHI/'
#     cmplx_chi_folders = ['86/', '87/', '97/', '98/']
#     folder = cmplx_chi_folders[1]
#     pathname = cmplx_chi_dir + folder 
#     data = analyzeEpochedFolder(pathname, epoch_file)

if __name__ == '__main__':
    # initial setup 
    import time as systime
    start_time = systime.time()
    import argparse
    from os import listdir, mkdir  
    import json
    ## parse user input 
    parser = argparse.ArgumentParser(description = '''Analyze running sessions for animals of a particular condition on a particular day''')
    parser.add_argument('--OutputFolder', nargs='?', type=str, default='./')
    parser.add_argument('--Folder', nargs='?', type=str, default='data/SW_D3/Sham/')
    args = parser.parse_args()
    epoch_file = 'epoch_data.csv' # 'epoch_data_chi.csv'
    ## data and output folders 
    base_folder = args.Folder
    animal_folders = listdir(base_folder)
    animal_folders = [fd+'/' for fd in animal_folders]
    output_folder = args.OutputFolder
    try:
        mkdir(output_folder)
    except:
        pass 

    # basic running epoch stats 
    epoch_stats = allRunningEpochs(base_folder, animal_folders)
    with open(output_folder + 'epochStats.json', 'w') as fileObj:
        json.dump(epoch_stats, fileObj)
    print('Epoch statistics execution time: %.2f seconds' % (systime.time() - start_time))

    # correlation analysis of longest running epochs for each animal 
    corrtime = systime.time()
    longest_epoch_corr = corrLongestEpoch(base_folder, animal_folders)
    corr_output = {'rl' : longest_epoch_corr[('right_forearm1', 'left_hindleg1')], 
                    'lr' : longest_epoch_corr[('left_forearm1', 'right_hindleg1')]}
    # for key in corr_output:
    #     corr_output[key]['lags'] = [int(l) for l in corr_output[key]['lags']]
    with open(output_folder + 'longestRunnigEpoch.json', 'w') as fileObj:
        json.dump(corr_output, fileObj)
    print('Longest running epoch correlations execution time: %.2f seconds' % (systime.time() - corrtime))

    # coorelation analysis of all running epochs 
    alltime = systime.time()
    sham_rl_pks = []
    sham_rl_lags = []
    sham_lr_pks = []
    sham_lr_lags = []
    sham_lf_step_freq = []
    sham_lh_step_freq = []
    sham_rf_step_freq = []
    sham_rh_step_freq = []
    sham_lf_step_var = []
    sham_lh_step_var = []
    sham_rf_step_var = []
    sham_rh_step_var = []
    for ind, folder in enumerate(animal_folders):
        pathname = base_folder + folder 
        data = analyzeEpochedFolder(pathname, epoch_file)
        #import IPython; IPython.embed()
        rl_lags = []
        lr_lags = []
        rl_peaks = []
        lr_peaks = []
        lf_step_freq = []
        lh_step_freq = []
        rf_step_freq = []
        rh_step_freq = []
        lf_step_var = []
        lh_step_var = []
        rf_step_var = []
        rh_step_var = []
        for f in data.keys():
            for ep in data[f]['corr_peak_lags']:
                rl = ep[rl_pair]
                lr = ep[lr_pair]
                rl_lags.append(rl)
                lr_lags.append(lr)
            for ep in data[f]['corr_peaks']:
                rl = ep[rl_pair]
                lr = ep[lr_pair]
                rl_peaks.append(rl)
                lr_peaks.append(lr)
            for ep in data[f]['step_freqs']:
                lf_step_freq.append(ep[varnames[0]])
                lh_step_freq.append(ep[varnames[2]])
                rf_step_freq.append(ep[varnames[1]])
                rh_step_freq.append(ep[varnames[3]])
            for ep in data[f]['step_var']:
                lf_step_var.append(ep[varnames[0]])
                lh_step_var.append(ep[varnames[2]])
                rf_step_var.append(ep[varnames[1]])
                rh_step_var.append(ep[varnames[3]])
        sham_rl_pks.append(rl_peaks)
        sham_rl_lags.append(rl_lags)
        sham_lr_pks.append(lr_peaks)
        sham_lr_lags.append(lr_lags)
        sham_lf_step_freq.append(lf_step_freq)
        sham_lh_step_freq.append(lh_step_freq)
        sham_rf_step_freq.append(rf_step_freq)
        sham_rh_step_freq.append(rh_step_freq)
        sham_lf_step_var.append(lf_step_var)
        sham_lh_step_var.append(lh_step_var)
        sham_rf_step_var.append(rf_step_var)
        sham_rh_step_var.append(rh_step_var)
    out = {'rl_pks' : sham_rl_pks,
        'rl_lags' : sham_rl_lags,
        'lr_pks' : sham_lr_pks,
        'lr_lags' : sham_lr_lags,
        'lf_step_freq' : sham_lf_step_freq,
        'lh_step_freq' : sham_lh_step_freq,
        'rf_step_freq' : sham_rf_step_freq,
        'rh_step_freq' : sham_rh_step_freq,
        'lf_step_var' : sham_lf_step_var,
        'lh_step_var' : sham_lh_step_var,
        'rf_step_var' : sham_rf_step_var,
        'rh_step_var' : sham_rh_step_var,
        }
    with open(output_folder + 'allRunningEpochs.json', 'w') as fileObj:
        json.dump(out, fileObj)
    print('All running epoch correlations execution time: %.2f seconds' % (systime.time() - alltime))

    # cmplx_sham_dir = 'wheel_data/complex_wheel_day_3/Sham/'
    # cmplx_sham_folders = ['09/', '10/', '11/', '12/']
    # cmplx_chi_dir = 'wheel_data/complex_wheel_day_3/CHI/'
    # cmplx_chi_folders = ['86/', '87/', '97/', '98/']
    # day = 'D6'
    # cmplx_sham_dir = 'data/SW_D6/'
    # # cmplx_sham_folders = ['318087/',
    # #                     '318510/',
    # #                     '318512/',
    # #                     '318511/',
    # #                     '318509/',
    # #                     '318098/',
    # #                     '318097/',
    # #                     '318086/']
    # cmplx_sham_folders = ['318086/',
    #                     '318087/',
    #                     '318097/',
    #                     '318098/']
    # # cmplx_chi_dir = 'data/SW_D6/'
    # cmplx_chi_dir = cmplx_sham_dir
    # cmplx_chi_folders = ['318509/',
    #                     '318510/',
    #                     '318511/',
    #                     '318512/']
    # # cmplx_chi_folders = ['318098/',
    # #                     '318086/',
    # #                     '318087/',
    # #                     '318512/',
    # #                     '318097/',
    # #                     '318509/',
    # #                     '318511/',
    # #                     '318510/']
    # # basic running epoch stats 
    # ep_fig, ep_axs = plt.subplots(1,2)
    # cmplx_sham_ep_data = allRunningEpochs(cmplx_sham_dir, cmplx_sham_folders)
    # cmplx_chi_ep_data = allRunningEpochs(cmplx_chi_dir, cmplx_chi_folders)
    # ep_axs[0].errorbar([i for i in range(len(cmplx_sham_folders))], cmplx_sham_ep_data['avg'], yerr=cmplx_sham_ep_data['err'], color='blue', label='Sham')
    # ep_axs[0].errorbar([i for i in range(len(cmplx_sham_folders)+1, len(cmplx_sham_folders)+1+len(cmplx_chi_folders))], cmplx_chi_ep_data['avg'], yerr=cmplx_chi_ep_data['err'], color='red', label='CHI')
    # ep_axs[0].set_ylabel('Running Epoch Duration')
    # ep_axs[0].set_xlabel('Animal ID')
    # ep_axs[1].plot([i for i in range(len(cmplx_sham_folders))], cmplx_sham_ep_data['N'], color='blue', label='Sham')
    # ep_axs[1].plot([i for i in range(len(cmplx_sham_folders)+1, len(cmplx_sham_folders)+1+len(cmplx_chi_folders))], cmplx_chi_ep_data['N'], color='red', label='CHI')
    # ep_axs[1].set_ylabel('Number of Running Epochs')
    # ep_axs[1].set_xlabel('Animal ID')
    # ep_axs[1].legend()
    # ep_fig.savefig('figures/epoch_stats_' + day + '.png')
    # plt.tight_layout()
    # # correlation data from longest running epoch 
    # ## sham data
    # cmplx_sham_corrs = corrLongestEpoch(cmplx_sham_dir, cmplx_sham_folders)
    # ## CHI data
    # cmplx_chi_corrs = corrLongestEpoch(cmplx_chi_dir, cmplx_chi_folders)
    # lngcr_fig, lngcr_axs = plt.subplots(2, 2)
    # lngcr_axs[0][0].plot([i for i in range(len(cmplx_sham_folders))], cmplx_sham_corrs[rl_pair]['peaks'], '*-', label='Sham')
    # lngcr_axs[0][0].plot([i for i in range(len(cmplx_sham_folders)+1, len(cmplx_sham_folders)+1+len(cmplx_chi_folders))], cmplx_chi_corrs[rl_pair]['peaks'], '*-', label='CHI')
    # lngcr_axs[0][0].set_title(str(rl_pair))
    # lngcr_axs[0][0].set_ylabel('peak correlation')
    # lngcr_axs[1][0].plot([i for i in range(len(cmplx_sham_folders))], cmplx_sham_corrs[rl_pair]['lags'], '*-', label='Sham')
    # lngcr_axs[1][0].plot([i for i in range(len(cmplx_sham_folders)+1, len(cmplx_sham_folders)+1+len(cmplx_chi_folders))], cmplx_chi_corrs[rl_pair]['lags'],  '*-', label='CHI')
    # lngcr_axs[1][0].set_ylabel('peak lag')
    # lngcr_axs[1][0].set_xlabel('animal id')
    # lngcr_axs[0][1].plot([i for i in range(len(cmplx_sham_folders))], cmplx_sham_corrs[lr_pair]['peaks'], '*-', label='Sham')
    # lngcr_axs[0][1].plot([i for i in range(len(cmplx_sham_folders)+1, len(cmplx_sham_folders)+1+len(cmplx_chi_folders))], cmplx_chi_corrs[lr_pair]['peaks'],  '*-', label='CHI')
    # lngcr_axs[0][1].set_title(str(lr_pair))
    # lngcr_axs[1][1].plot([i for i in range(len(cmplx_sham_folders))], cmplx_sham_corrs[lr_pair]['lags'], '*-', label='Sham')
    # lngcr_axs[1][1].plot([i for i in range(len(cmplx_sham_folders)+1, len(cmplx_sham_folders)+1+len(cmplx_chi_folders))], cmplx_chi_corrs[lr_pair]['lags'],  '*-', label='CHI')
    # lngcr_axs[1][1].set_xlabel('animal id')
    # lngcr_fig.savefig('figures/longest_epoch_' + day + '_corr.png')
    # plt.tight_layout()
    # # difference betwen left forearm - right hindleg peak lag and right forearm - left hindleg peak lag 
    # cmplx_sham_lag_diff = [lr-rl for lr, rl in zip(cmplx_sham_corrs[lr_pair]['lags'], cmplx_sham_corrs[rl_pair]['lags'])]
    # cmplx_chi_lag_diff = [lr-rl for lr, rl in zip(cmplx_chi_corrs[lr_pair]['lags'], cmplx_chi_corrs[rl_pair]['lags'])]
    # lagdif_fig, lagdif_axs = plt.subplots(1,1)
    # lagdif_axs.plot([i for i in range(len(cmplx_sham_folders))], cmplx_sham_lag_diff)
    # lagdif_axs.plot([i for i in range(len(cmplx_sham_folders)+1, len(cmplx_sham_folders)+1+len(cmplx_chi_folders))], cmplx_chi_lag_diff)
    # lagdif_axs.set_ylabel('Difference in peak lag')
    # lagdif_fig.savefig('figures/diffInPeaklag_longestEpoch' + day + '.png')
    # # looking at all running epochs 
    # ## sham
    # fig, axs = plt.subplots(2,3)
    # sham_lag_difs = []
    # sham_peak_difs = []
    # sham_rl_pks = []
    # sham_rl_lags = []
    # sham_lr_pks = []
    # sham_lr_lags = []
    # for ind, folder in enumerate(cmplx_sham_folders):
    #     pathname = cmplx_sham_dir + folder 
    #     data = analyzeEpochedFolder(pathname, epoch_file)
    #     lag_difs = []
    #     rl_lags = []
    #     lr_lags = []
    #     rl_peaks = []
    #     lr_peaks = []
    #     peak_difs = []
    #     for f in data.keys():
    #         for ep in data[f]['corr_peak_lags']:
    #             rl = ep[rl_pair]
    #             lr = ep[lr_pair]
    #             rl_lags.append(rl)
    #             lr_lags.append(lr)
    #             lag_difs.append(lr-rl)
    #         for ep in data[f]['corr_peaks']:
    #             rl = ep[rl_pair]
    #             lr = ep[lr_pair]
    #             rl_peaks.append(rl)
    #             lr_peaks.append(lr)
    #             peak_difs.append(lr-rl)
    #     axs[0][0].plot([ind for i in lag_difs], rl_lags, '*')
    #     axs[0][1].plot([ind for i in lag_difs], lr_lags, '*')
    #     axs[0][2].plot([ind for i in lag_difs], np.abs(lag_difs), '*')
    #     axs[1][0].plot([ind for i in lag_difs], rl_peaks, '*')
    #     axs[1][1].plot([ind for i in lag_difs], lr_peaks, '*')
    #     axs[1][2].plot([ind for i in lag_difs], np.abs(peak_difs), '*')
    #     axs[0][0].errorbar([ind], [np.mean(rl_lags)], yerr=[np.std(rl_lags)], color='k')
    #     axs[0][1].errorbar([ind], [np.mean(lr_lags)], yerr=[np.std(lr_lags)], color='k')
    #     axs[0][2].errorbar([ind], [np.mean(np.abs(lag_difs))], yerr=[np.std(np.abs(lag_difs))/len(lag_difs)], color='k',marker='*')
    #     axs[1][0].errorbar([ind], [np.mean(rl_peaks)], yerr=[np.std(rl_peaks)], color='k')
    #     axs[1][1].errorbar([ind], [np.mean(lr_peaks)], yerr=[np.std(lr_peaks)], color='k')
    #     axs[1][2].errorbar([ind], [np.mean(np.abs(peak_difs))], yerr=[np.std(np.abs(peak_difs))/len(lag_difs)], color='k', marker='*')
    #     sham_lag_difs.append(lag_difs)
    #     sham_peak_difs.append(peak_difs)
    #     sham_rl_pks.append(rl_peaks)
    #     sham_rl_lags.append(rl_lags)
    #     sham_lr_pks.append(lr_peaks)
    #     sham_lr_lags.append(lr_lags)
    # ## chi
    # chi_lag_difs = []
    # chi_peak_difs = []
    # chi_rl_pks = []
    # chi_rl_lags = []
    # chi_lr_pks = []
    # chi_lr_lags = []
    # for ind, folder in enumerate(cmplx_chi_folders):
    #     pathname = cmplx_chi_dir + folder 
    #     data = analyzeEpochedFolder(pathname, epoch_file)
    #     lag_difs = []
    #     rl_lags = []
    #     lr_lags = []
    #     rl_peaks = []
    #     lr_peaks = []
    #     peak_difs = []
    #     for f in data.keys():
    #         for ep in data[f]['corr_peak_lags']:
    #             rl = ep[rl_pair]
    #             lr = ep[lr_pair]
    #             rl_lags.append(rl)
    #             lr_lags.append(lr)
    #             lag_difs.append(lr-rl)
    #         for ep in data[f]['corr_peaks']:
    #             rl = ep[rl_pair]
    #             lr = ep[lr_pair]
    #             rl_peaks.append(rl)
    #             lr_peaks.append(lr)
    #             peak_difs.append(lr-rl)
    #     axs[0][0].plot([ind + len(cmplx_sham_folders) + 1 for i in lag_difs], rl_lags, 'o')
    #     axs[0][1].plot([ind + len(cmplx_sham_folders) + 1 for i in lag_difs], lr_lags, 'o')
    #     axs[0][2].plot([ind + len(cmplx_sham_folders) + 1 for i in lag_difs], np.abs(lag_difs), 'o')
    #     axs[1][0].plot([ind + len(cmplx_sham_folders) + 1 for i in lag_difs], rl_peaks, 'o')
    #     axs[1][1].plot([ind + len(cmplx_sham_folders) + 1 for i in lag_difs], lr_peaks, 'o')
    #     axs[1][2].plot([ind + len(cmplx_sham_folders) + 1 for i in lag_difs], np.abs(peak_difs), 'o')
    #     axs[0][0].errorbar([ind + len(cmplx_sham_folders) + 1], [np.mean(rl_lags)], yerr=[np.std(rl_lags)], color='k')
    #     axs[0][1].errorbar([ind + len(cmplx_sham_folders) + 1], [np.mean(lr_lags)], yerr=[np.std(lr_lags)], color='k')
    #     axs[0][2].errorbar([ind + len(cmplx_sham_folders) + 1], [np.mean(np.abs(lag_difs))], yerr=[np.std(np.abs(lag_difs))/len(lag_difs)], color='k', marker='*')
    #     axs[1][0].errorbar([ind + len(cmplx_sham_folders) + 1], [np.mean(rl_peaks)], yerr=[np.std(rl_peaks)], color='k')
    #     axs[1][1].errorbar([ind + len(cmplx_sham_folders) + 1], [np.mean(lr_peaks)], yerr=[np.std(lr_peaks)], color='k')
    #     axs[1][2].errorbar([ind + len(cmplx_sham_folders) + 1], [np.mean(np.abs(peak_difs))], yerr=[np.std(np.abs(peak_difs))/len(lag_difs)], color='k', marker='*')
    #     chi_lag_difs.append(lag_difs)
    #     chi_peak_difs.append(peak_difs)
    #     chi_rl_pks.append(rl_peaks)
    #     chi_rl_lags.append(rl_lags)
    #     chi_lr_pks.append(lr_peaks)
    #     chi_lr_lags.append(lr_lags)
    # animal_labels = cmplx_sham_folders
    # animal_labels.extend([''])
    # animal_labels.extend(cmplx_chi_folders)
    # for r in axs:
    #     for ax in r:
    #         ax.set_xticklabels(animal_labels)
    # axs[0][0].set_title(str(rl_pair))
    # axs[0][1].set_title(str(lr_pair))
    # axs[0][2].set_title('Difference')
    # axs[0][0].set_ylabel('Lag of Peak Correlation')
    # axs[1][0].set_ylabel('Peak Correlation Magnitude')
    # axs[1][1].set_xlabel('Animal ID')
    # fig.savefig('figures/allRunningEpochs_' + day + '.png')
    # plt.ion()
    # plt.show()


# if __name__ == '__main__':
#     # plt.ion()
#     # pathname = 'wheel_data/simple_wheel_day_3/Sham/'
#     # pathname = 'wheel_data/complex_wheel_day_3/CHI/86/'
#     # pathname = 'wheel_data/complex_wheel_day_3/Sham/12/'
#     epoch_file = 'epoch_data.csv' # 'epoch_data_chi.csv'
#     data_dir = 'wheel_data/complex_wheel_day_3/Sham/'
#     corfig, coraxs = plt.subplots(2,4, sharex=True)
#     stepfig, stepaxs = plt.subplots(1,4,sharex=True)
#     epochfig = plt.figure()
#     epochax = epochfig.add_subplot(111)
#     for folder in ['09/', '10/', '11/', '12/']:
#         pathname = data_dir + folder
#         data = analyzeEpochedFolder(pathname, epoch_file)
#         ## create figs 
#         ## plot correlation data
#         for pairind, pair in enumerate(limb_pairs):
#             trials = []
#             corr_peaks = []
#             peak_lags = []
#             for ind, filename in enumerate(data.keys()):
#                 trial = [ind for i in range(len(data[filename]['corr_peaks']))]
#                 corrs = [peak[pair] for peak in data[filename]['corr_peaks']]
#                 lags = [peak[pair] for peak in data[filename]['corr_peak_lags']]
#                 trials.extend(trial)
#                 corr_peaks.extend(corrs)
#                 peak_lags.extend(lags)
#             coraxs[0][pairind].set_title(pair)
#             coraxs[0][pairind].scatter(trials, corr_peaks, marker='x', label='Sham ' + folder)
#             coraxs[1][pairind].scatter(trials, peak_lags, marker='x', label='Sham ' + folder)
#         ## plot step frequency 
#         for varind, varname in enumerate(varnames):
#             trials = []
#             freqs = []
#             for ind, filename in enumerate(data.keys()):
#                 trial = [ind for i in range(len(data[filename]['step_freqs']))]
#                 freq = [f[varname] for f in data[filename]['step_freqs']]
#                 trials.extend(trial)
#                 freqs.extend(freq)
#             stepaxs[varind].set_title(varname)
#             stepaxs[varind].scatter(trials, freqs, marker='x', label='Sham ' + folder)
#         ## plot ecoch durations 
#         trials = []
#         durs = []
#         for ind, filename in enumerate(data.keys()):
#             trial = [ind for i in range(len(data[filename]['epoch_durs']))]
#             dur = [d for d in data[filename]['epoch_durs']]
#             durs.extend(dur)
#             trials.extend(trial)
#         epochax.scatter(trials, durs, marker='x', label='Sham ' + folder)
        

#     data_dir = 'wheel_data/complex_wheel_day_3/CHI/'
#     for folder in ['86/', '87/', '97/', '98/']:
#         pathname = data_dir + folder
#         data = analyzeEpochedFolder(pathname, epoch_file)
#         ## plot correlation data
#         for pairind, pair in enumerate(limb_pairs):
#             trials = []
#             corr_peaks = []
#             peak_lags = []
#             for ind, filename in enumerate(data.keys()):
#                 trial = [ind for i in range(len(data[filename]['corr_peaks']))]
#                 corrs = [peak[pair] for peak in data[filename]['corr_peaks']]
#                 lags = [peak[pair] for peak in data[filename]['corr_peak_lags']]
#                 trials.extend(trial)
#                 corr_peaks.extend(corrs)
#                 peak_lags.extend(lags)
#             coraxs[0][pairind].set_title(pair)
#             coraxs[0][pairind].scatter(trials, corr_peaks, marker='+', label='CHI ' + folder)
#             coraxs[1][pairind].scatter(trials, peak_lags, marker='+', label='CHI ' + folder)
#         ## plot step frequency 
#         for varind, varname in enumerate(varnames):
#             trials = []
#             freqs = []
#             for ind, filename in enumerate(data.keys()):
#                 trial = [ind for i in range(len(data[filename]['step_freqs']))]
#                 freq = [f[varname] for f in data[filename]['step_freqs']]
#                 trials.extend(trial)
#                 freqs.extend(freq)
#             stepaxs[varind].set_title(varname)
#             stepaxs[varind].scatter(trials, freqs, marker='+', label='CHI ' + folder)
#         ## plot ecoch durations 
#         trials = []
#         durs = []
#         for ind, filename in enumerate(data.keys()):
#             trial = [ind for i in range(len(data[filename]['epoch_durs']))]
#             dur = [d for d in data[filename]['epoch_durs']]
#             durs.extend(dur)
#             trials.extend(trial)
#         epochax.scatter(trials, durs, marker='+', label='CHI ' + folder)
#     coraxs[0][0].legend()
#     stepaxs[0].legend()
#     epochax.legend()
#     plt.ion()
#     plt.show()

#######################################################################################
# epoch_data = loadEpochs(pathname, epoch_file)
# fps = 240
# # filename = epoch_data['filename'][1]
# # ind = 1
# for ind, filename in enumerate(epoch_data['filename']):
#     df = loadCSV(pathname, filename)
#     df = addCenter(df)
#     df = addTime(df, fps)
#     df = addRelativeFootDist(df)
#     df = nanInterp(df)
#     var = 'relative_dist'
#     df = smoothVar(df, 20, var=var+'_fix')
#     exec('running_epochs = ' + epoch_data['running_epochs'][ind])
#     # df = smoothVar(df, 20, var=var+'_fix')
#     Nepochs = len(running_epochs)
#     for epoch in running_epochs:
#         # fig2, axs2 = plt.subplots(nrows=4, ncols=1, sharex=True)
#         # for varind, varname in enumerate(varnames):
#         #     y = [val for val, t in zip(df[varname][var+'_fix'].values, df['time'].values) if epoch[0] < t < epoch[1]]
#         #     time = [t for t in df['time'].values if epoch[0] < t < epoch[1]]
#         #     axs2[varind].plot(time, y)
#         #     axs2[varind].set_title(varname)
#         # axs2[0].set_title(filename + ' ' + varnames[0] + '; Epoch ' + str(epoch))
#         plt.figure()
#         for pairind, pair in enumerate(limb_pairs):
#             y0 = [val for val, t in zip(df[pair[0]][var+'_fix'].values, df['time'].values) if epoch[0] < t < epoch[1]]
#             y0 = y0 - np.mean(y0)
#             y0 = y0 / np.max(y0)
#             y1 = [val for val, t in zip(df[pair[1]][var+'_fix'].values, df['time'].values) if epoch[0] < t < epoch[1]]
#             y1 = y1 - np.mean(y1)
#             y1 = y1 / np.max(y1)
#             time = [t for t in df['time'].values if epoch[0] < t < epoch[1]]
#             corr = correlate(y0, y1)
#             lags = correlation_lags(len(y0), len(y1))
#             # plt.figure()
#             plt.subplot(320 + pairind + 1)
#             plt.plot(time, y0)
#             plt.title(filename + ': ' + pair[0] + '; Epoch ' + str(epoch))
#             plt.subplot(320 + pairind + 3)
#             plt.plot(time, y1)
#             plt.title(pair[1])
#             plt.subplot(320 + pairind + 5)
#             plt.plot(lags / fps, corr)
#             plt.title('Correlation: ' + pair[0] + '-' + pair[1])
#         plt.show()

# # pathname = './'
# # filename = 'wheel_example.csv'
# # vidfile = pathname + 'wheel_movie.mp4'
# # pathname = 'MN12_318093/'
# # filename = 'MN12_3.csv'
# # vidfile = pathname + 'MN12_3_labeled.mp4'
# # pathname = './'
# # pathname = 'Files/'
# # filename = 'IMG_5314DLC_resnet50_SCWheelSep8shuffle1_392800.csv'
# # vidfile = pathname + 'IMG_5314DLC_resnet50_SCWheelSep8shuffle1_392800_labeled.mp4'
# # filename = 'IMG_5308DLC_resnet50_SCWheelSep8shuffle1_392800.csv'
# # vidfile = pathname + 'IMG_5308DLC_resnet50_SCWheelSep8shuffle1_392800_labeled.mp4'
# # filename = 'IMG_6626_1DLC_resnet50_SCWheelSep8shuffle1_646500.csv'
# # filename = '15_1DLC_resnet50_SCWheelSep8shuffle1_646500.csv'

# # plt.ion()
# # df = loadCSV(pathname, filename)
# # df = addCenter(df)
# # df = addTime(df, 240)
# # df = addRelativeFootDist(df)
# # df = nanInterp(df)

# # for ind, var in enumerate(['relative_dist', 'relative_x']):
# #     fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
# #     plt.subplot(311)
# #     plt.title(var)
# #     plotVar(df, var=var)
# #     plt.subplot(312)
# #     plt.title(var+'_fix')
# #     plotVar(df, var=var+'_fix')
# #     plt.subplot(313)
# #     df = smoothVar(df, 20, var=var+'_fix')
# #     plotVar(df, var=var+'_fix')
# #     plt.title(var+'_fix smooth 20')

# # ## fft 
# # var = 'relative_dist'
# # fig2, axs2 = plt.subplots(nrows=4, ncols=1, sharex=True)
# # for ind, varname in enumerate(varnames):
# #     # y = df[varname][var+'_fix'] - np.mean(df[varname][var+'_fix'])
# #     y = [val for val, t in zip(df[varname][var+'_fix'].values, df['time'].values) if 6.5 < t < 10.0]
# #     y = np.array(y) - np.mean(y)
# #     Y = (fft(y)/len(df = smoothVar(df, 20, var=var+'_fix')y))[0:int(len(y)/2)]
# #     Freq = np.linspace(0.0, 240/2.0, len(Y))
# #     axs2[ind].plot(Freq, np.abs(Y))
# #     axs2[ind].set_title(varname)
# #     plt.xlim(0,20)

# # plotRelativeDist(df)
