from os import listdir 
from wheel_analysis import *
from matplotlib import pyplot as plt
import pandas as pd

plt.ion()

pathname = input('Enter relative path to data: ')
filenames = listdir(pathname)

out = {'filename' : [], 'running_epochs' : [], 'active_epochs' : []}
print('Enter epochs as lists of tuples')
print('No epochs: Enter running epoch(s): []')
print('Single epoch example: Enter running epoch(s): [(1.1, 5.6)]')
print('Multi-epoch example: Enter running epoch(s): [(1.1, 5.6), (7.6, 12.15), (13.0, 20.0)]\n')

for filename in filenames:
    print('File: %s' % (filename))
    df = loadCSV(pathname, filename)
    df = addCenter(df)
    df = addTime(df, 240)
    df = addRelativeFootDist(df)
    df = nanInterp(df)
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    for ind, var in enumerate(['relative_dist', 'relative_x']):
        df = smoothVar(df, 20, var=var+'_fix')
        plt.subplot(2,1,ind+1)
        plotVar(df, var=var+'_fix')
        plt.ylabel(var)
    plt.subplot(211)
    plt.title(filename)
    plt.subplot(212)
    plt.xlabel('Time (s)')
    running_epochs = input('Enter running epoch(s): ')
    active_epochs = input('Enter active epoch(s): ')
    out['filename'].append(filename)
    eval("out['running_epochs'].append(%s)" % (running_epochs))
    eval("out['active_epochs'].append(%s)" % (active_epochs))
    plt.close()

metadata = pd.DataFrame.from_dict(out)
metadata.to_csv(pathname + 'epoch_data.csv')