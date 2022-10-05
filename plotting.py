from matplotlib import pyplot as plt 
import json 
import numpy as np 

###################################################################
#                           Simple Wheel                          #
###################################################################
# all running epoch 
with open('analyzed_data/simple_wheel/d3sham/allRunningEpochs.json', 'rb') as fileObj:
    d3sham = json.load(fileObj)
with open('analyzed_data/simple_wheel/d3chi/allRunningEpochs.json', 'rb') as fileObj:
    d3chi = json.load(fileObj)
with open('analyzed_data/simple_wheel/d6sham/allRunningEpochs.json', 'rb') as fileObj:
    d6sham = json.load(fileObj)
with open('analyzed_data/simple_wheel/d6chi/allRunningEpochs.json', 'rb') as fileObj:
    d6chi = json.load(fileObj)

# correlation peaks 
allRunEpochCorrFig, allRunEpochCorrAxs = plt.subplots(2,2)
## right-left 
d3sham_rl_pks_mean = [np.mean(n) for n in d3sham['rl_pks']]
d3chi_rl_pks_mean = [np.mean(n) for n in d3chi['rl_pks']]
d3sham_rl_pks_err = [np.std(n) for n in d3sham['rl_pks']]
d3chi_rl_pks_err = [np.std(n) for n in d3chi['rl_pks']]
d6sham_rl_pks_mean = [np.mean(n) for n in d6sham['rl_pks']]
d6chi_rl_pks_mean = [np.mean(n) for n in d6chi['rl_pks']]
d6sham_rl_pks_err = [np.std(n) for n in d6sham['rl_pks']]
d6chi_rl_pks_err = [np.std(n) for n in d6chi['rl_pks']]
sham_rl_pks_mean = [[a, b] for a, b in zip(d3sham_rl_pks_mean, d6sham_rl_pks_mean)]
chi_rl_pks_mean = [[a, b] for a, b in zip(d3chi_rl_pks_mean, d6chi_rl_pks_mean)]
sham_rl_pks_err = [[a, b] for a, b in zip(d3sham_rl_pks_err, d6sham_rl_pks_err)]
chi_rl_pks_err = [[a, b] for a, b in zip(d3chi_rl_pks_err, d6chi_rl_pks_err)]
for ind, vals in enumerate(zip(sham_rl_pks_mean, sham_rl_pks_err)):
    allRunEpochCorrAxs[0][1].errorbar([ind/4, ind/4 + 6], vals[0], yerr=vals[1], color='b', label='Sham', fmt='*-')
for ind, vals in enumerate(zip(chi_rl_pks_mean, chi_rl_pks_err)):
    allRunEpochCorrAxs[0][1].errorbar([ind/4+2, ind/4 + 8], vals[0], yerr=vals[1], color='r', label='Chi', fmt='*-')
allRunEpochCorrAxs[0][1].set_ylabel('Peak Correlation (a.u.)', fontsize=14)
allRunEpochCorrAxs[0][1].set_xticks([0.5, 2.5, 6.5, 8.5])
allRunEpochCorrAxs[0][1].set_xticklabels(['Sham Day 3', 'Chi Day 3', 'Sham Day 6', 'Chi Day 6'], fontsize=12)
allRunEpochCorrAxs[0][1].set_title('Right Frontleg - Left Hindleg', fontsize=16)
## left-right
d3sham_lr_pks_mean = [np.mean(n) for n in d3sham['lr_pks']]
d3chi_lr_pks_mean = [np.mean(n) for n in d3chi['lr_pks']]
d3sham_lr_pks_err = [np.std(n) for n in d3sham['lr_pks']]
d3chi_lr_pks_err = [np.std(n) for n in d3chi['lr_pks']]
d6sham_lr_pks_mean = [np.mean(n) for n in d6sham['lr_pks']]
d6chi_lr_pks_mean = [np.mean(n) for n in d6chi['lr_pks']]
d6sham_lr_pks_err = [np.std(n) for n in d6sham['lr_pks']]
d6chi_lr_pks_err = [np.std(n) for n in d6chi['lr_pks']]
sham_lr_pks_mean = [[a, b] for a, b in zip(d3sham_lr_pks_mean, d6sham_lr_pks_mean)]
chi_lr_pks_mean = [[a, b] for a, b in zip(d3chi_lr_pks_mean, d6chi_lr_pks_mean)]
sham_lr_pks_err = [[a, b] for a, b in zip(d3sham_lr_pks_err, d6sham_lr_pks_err)]
chi_lr_pks_err = [[a, b] for a, b in zip(d3chi_lr_pks_err, d6chi_lr_pks_err)]
for ind, vals in enumerate(zip(sham_lr_pks_mean, sham_lr_pks_err)):
    allRunEpochCorrAxs[0][0].errorbar([ind/4, ind/4 + 6], vals[0], yerr=vals[1], color='b', label='Sham', fmt='*-')
for ind, vals in enumerate(zip(chi_lr_pks_mean, chi_lr_pks_err)):
    allRunEpochCorrAxs[0][0].errorbar([ind/4+2, ind/4 + 8], vals[0], yerr=vals[1], color='r', label='Chi', fmt='*-')
allRunEpochCorrAxs[0][0].set_ylabel('Peak Correlation (a.u.)', fontsize=14)
allRunEpochCorrAxs[0][0].set_xticks([0.5, 2.5, 6.5, 8.5])
allRunEpochCorrAxs[0][0].set_xticklabels(['Sham Day 3', 'Chi Day 3', 'Sham Day 6', 'Chi Day 6'], fontsize=12)
allRunEpochCorrAxs[0][0].set_title('Left Frontleg - Right Hindleg', fontsize=16)
# correlation lags
## right-left
d3sham_rl_lags_mean = [np.mean(n) for n in d3sham['rl_lags']]
d3chi_rl_lags_mean = [np.mean(n) for n in d3chi['rl_lags']]
d3sham_rl_lags_err = [np.std(n) for n in d3sham['rl_lags']]
d3chi_rl_lags_err = [np.std(n) for n in d3chi['rl_lags']]
d6sham_rl_lags_mean = [np.mean(n) for n in d6sham['rl_lags']]
d6chi_rl_lags_mean = [np.mean(n) for n in d6chi['rl_lags']]
d6sham_rl_lags_err = [np.std(n) for n in d6sham['rl_lags']]
d6chi_rl_lags_err = [np.std(n) for n in d6chi['rl_lags']]
sham_rl_lags_mean = [[a, b] for a, b in zip(d3sham_rl_lags_mean, d6sham_rl_lags_mean)]
chi_rl_lags_mean = [[a, b] for a, b in zip(d3chi_rl_lags_mean, d6chi_rl_lags_mean)]
sham_rl_lags_err = [[a, b] for a, b in zip(d3sham_rl_lags_err, d6sham_rl_lags_err)]
chi_rl_lags_err = [[a, b] for a, b in zip(d3chi_rl_lags_err, d6chi_rl_lags_err)]
for ind, vals in enumerate(zip(sham_rl_lags_mean, sham_rl_lags_err)):
    allRunEpochCorrAxs[1][1].errorbar([ind/4, ind/4 + 6], vals[0], yerr=vals[1], color='b', label='Sham', fmt='*-')
for ind, vals in enumerate(zip(chi_rl_lags_mean, chi_rl_lags_err)):
    allRunEpochCorrAxs[1][1].errorbar([ind/4+2, ind/4 + 8], vals[0], yerr=vals[1], color='r', label='Chi', fmt='*-')
allRunEpochCorrAxs[1][1].set_ylabel('Lag of Peak Correlation (s)', fontsize=14)
allRunEpochCorrAxs[1][1].set_xticks([0.5, 2.5, 6.5, 8.5])
allRunEpochCorrAxs[1][1].set_xticklabels(['Sham Day 3', 'Chi Day 3', 'Sham Day 6', 'Chi Day 6'], fontsize=12)
## left-right
d3sham_lr_lags_mean = [np.mean(n) for n in d3sham['lr_lags']]
d3chi_lr_lags_mean = [np.mean(n) for n in d3chi['lr_lags']]
d3sham_lr_lags_err = [np.std(n) for n in d3sham['lr_lags']]
d3chi_lr_lags_err = [np.std(n) for n in d3chi['lr_lags']]
d6sham_lr_lags_mean = [np.mean(n) for n in d6sham['lr_lags']]
d6chi_lr_lags_mean = [np.mean(n) for n in d6chi['lr_lags']]
d6sham_lr_lags_err = [np.std(n) for n in d6sham['lr_lags']]
d6chi_lr_lags_err = [np.std(n) for n in d6chi['lr_lags']]
sham_lr_lags_mean = [[a, b] for a, b in zip(d3sham_lr_lags_mean, d6sham_lr_lags_mean)]
chi_lr_lags_mean = [[a, b] for a, b in zip(d3chi_lr_lags_mean, d6chi_lr_lags_mean)]
sham_lr_lags_err = [[a, b] for a, b in zip(d3sham_lr_lags_err, d6sham_lr_lags_err)]
chi_lr_lags_err = [[a, b] for a, b in zip(d3chi_lr_lags_err, d6chi_lr_lags_err)]
for ind, vals in enumerate(zip(sham_lr_lags_mean, sham_lr_lags_err)):
    allRunEpochCorrAxs[1][0].errorbar([ind/4, ind/4 + 6], vals[0], yerr=vals[1], color='b', label='Sham', fmt='*-')
for ind, vals in enumerate(zip(chi_lr_lags_mean, chi_lr_lags_err)):
    allRunEpochCorrAxs[1][0].errorbar([ind/4+2, ind/4 + 8], vals[0], yerr=vals[1], color='r', label='Chi', fmt='*-')
allRunEpochCorrAxs[1][0].set_ylabel('Lag of Peak Correlation (s)', fontsize=14)
allRunEpochCorrAxs[1][0].set_xticks([0.5, 2.5, 6.5, 8.5])
allRunEpochCorrAxs[1][0].set_xticklabels(['Sham Day 3', 'Chi Day 3', 'Sham Day 6', 'Chi Day 6'], fontsize=14)


# step frequency all running epochs 
stepfig, stepaxs = plt.subplots(2,2)
d3sham_rf_stepFreq_mean = [np.mean(n) for n in d3sham['rf_step_freq']]
d3sham_lf_stepFreq_mean = [np.mean(n) for n in d3sham['lf_step_freq']]
d3sham_rh_stepFreq_mean = [np.mean(n) for n in d3sham['rh_step_freq']]
d3sham_lh_stepFreq_mean = [np.mean(n) for n in d3sham['lh_step_freq']]
d3sham_rf_stepFreq_err = [np.std(n) for n in d3sham['rf_step_freq']]
d3sham_lf_stepFreq_err = [np.std(n) for n in d3sham['lf_step_freq']]
d3sham_rh_stepFreq_err = [np.std(n) for n in d3sham['rh_step_freq']]
d3sham_lh_stepFreq_err = [np.std(n) for n in d3sham['lh_step_freq']]
d6sham_rf_stepFreq_mean = [np.mean(n) for n in d6sham['rf_step_freq']]
d6sham_lf_stepFreq_mean = [np.mean(n) for n in d6sham['lf_step_freq']]
d6sham_rh_stepFreq_mean = [np.mean(n) for n in d6sham['rh_step_freq']]
d6sham_lh_stepFreq_mean = [np.mean(n) for n in d6sham['lh_step_freq']]
d6sham_rf_stepFreq_err = [np.std(n) for n in d6sham['rf_step_freq']]
d6sham_lf_stepFreq_err = [np.std(n) for n in d6sham['lf_step_freq']]
d6sham_rh_stepFreq_err = [np.std(n) for n in d6sham['rh_step_freq']]
d6sham_lh_stepFreq_err = [np.std(n) for n in d6sham['lh_step_freq']]
d3chi_rf_stepFreq_mean = [np.mean(n) for n in d3chi['rf_step_freq']]
d3chi_lf_stepFreq_mean = [np.mean(n) for n in d3chi['lf_step_freq']]
d3chi_rh_stepFreq_mean = [np.mean(n) for n in d3chi['rh_step_freq']]
d3chi_lh_stepFreq_mean = [np.mean(n) for n in d3chi['lh_step_freq']]
d3chi_rf_stepFreq_err = [np.std(n) for n in d3chi['rf_step_freq']]
d3chi_lf_stepFreq_err = [np.std(n) for n in d3chi['lf_step_freq']]
d3chi_rh_stepFreq_err = [np.std(n) for n in d3chi['rh_step_freq']]
d3chi_lh_stepFreq_err = [np.std(n) for n in d3chi['lh_step_freq']]
d6chi_rf_stepFreq_mean = [np.mean(n) for n in d6chi['rf_step_freq']]
d6chi_lf_stepFreq_mean = [np.mean(n) for n in d6chi['lf_step_freq']]
d6chi_rh_stepFreq_mean = [np.mean(n) for n in d6chi['rh_step_freq']]
d6chi_lh_stepFreq_mean = [np.mean(n) for n in d6chi['lh_step_freq']]
d6chi_rf_stepFreq_err = [np.std(n) for n in d6chi['rf_step_freq']]
d6chi_lf_stepFreq_err = [np.std(n) for n in d6chi['lf_step_freq']]
d6chi_rh_stepFreq_err = [np.std(n) for n in d6chi['rh_step_freq']]
d6chi_lh_stepFreq_err = [np.std(n) for n in d6chi['lh_step_freq']]
sham_rh_stepFreq_mean = [[a, b] for a, b in zip(d3sham_rh_stepFreq_mean, d6sham_rh_stepFreq_mean)]
sham_lh_stepFreq_mean = [[a, b] for a, b in zip(d3sham_lh_stepFreq_mean, d6sham_lh_stepFreq_mean)]
sham_rf_stepFreq_mean = [[a, b] for a, b in zip(d3sham_rf_stepFreq_mean, d6sham_rf_stepFreq_mean)]
sham_lf_stepFreq_mean = [[a, b] for a, b in zip(d3sham_lf_stepFreq_mean, d6sham_lf_stepFreq_mean)]
sham_rh_stepFreq_err = [[a, b] for a, b in zip(d3sham_rh_stepFreq_err, d6sham_rh_stepFreq_err)]
sham_lh_stepFreq_err = [[a, b] for a, b in zip(d3sham_lh_stepFreq_err, d6sham_lh_stepFreq_err)]
sham_rf_stepFreq_err = [[a, b] for a, b in zip(d3sham_rf_stepFreq_err, d6sham_rf_stepFreq_err)]
sham_lf_stepFreq_err = [[a, b] for a, b in zip(d3sham_lf_stepFreq_err, d6sham_lf_stepFreq_err)]
chi_rh_stepFreq_mean = [[a, b] for a, b in zip(d3chi_rh_stepFreq_mean, d6chi_rh_stepFreq_mean)]
chi_lh_stepFreq_mean = [[a, b] for a, b in zip(d3chi_lh_stepFreq_mean, d6chi_lh_stepFreq_mean)]
chi_rf_stepFreq_mean = [[a, b] for a, b in zip(d3chi_rf_stepFreq_mean, d6chi_rf_stepFreq_mean)]
chi_lf_stepFreq_mean = [[a, b] for a, b in zip(d3chi_lf_stepFreq_mean, d6chi_lf_stepFreq_mean)]
chi_rh_stepFreq_err = [[a, b] for a, b in zip(d3chi_rh_stepFreq_err, d6chi_rh_stepFreq_err)]
chi_lh_stepFreq_err = [[a, b] for a, b in zip(d3chi_lh_stepFreq_err, d6chi_lh_stepFreq_err)]
chi_rf_stepFreq_err = [[a, b] for a, b in zip(d3chi_rf_stepFreq_err, d6chi_rf_stepFreq_err)]
chi_lf_stepFreq_err = [[a, b] for a, b in zip(d3chi_lf_stepFreq_err, d6chi_lf_stepFreq_err)]
for ind, vals in enumerate(zip(sham_lf_stepFreq_mean, sham_lf_stepFreq_err)):
    stepaxs[0][0].errorbar([ind/4, ind/4 + 6], vals[0], yerr=vals[1], color='b', label='Sham', fmt='*-')
for ind, vals in enumerate(zip(chi_lf_stepFreq_mean, chi_lf_stepFreq_err)):
    stepaxs[0][0].errorbar([ind/4+2, ind/4 + 8], vals[0], yerr=vals[1], color='r', label='Chi', fmt='*-')
for ind, vals in enumerate(zip(sham_rf_stepFreq_mean, sham_rf_stepFreq_err)):
    stepaxs[0][1].errorbar([ind/4, ind/4 + 6], vals[0], yerr=vals[1], color='b', label='Sham', fmt='*-')
for ind, vals in enumerate(zip(chi_rf_stepFreq_mean, chi_rf_stepFreq_err)):
    stepaxs[0][1].errorbar([ind/4+2, ind/4 + 8], vals[0], yerr=vals[1], color='r', label='Chi', fmt='*-')
for ind, vals in enumerate(zip(sham_lh_stepFreq_mean, sham_lh_stepFreq_err)):
    stepaxs[1][0].errorbar([ind/4, ind/4 + 6], vals[0], yerr=vals[1], color='b', label='Sham', fmt='*-')
for ind, vals in enumerate(zip(chi_lh_stepFreq_mean, chi_lh_stepFreq_err)):
    stepaxs[1][0].errorbar([ind/4+2, ind/4 + 8], vals[0], yerr=vals[1], color='r', label='Chi', fmt='*-')
for ind, vals in enumerate(zip(sham_rh_stepFreq_mean, sham_rh_stepFreq_err)):
    stepaxs[1][1].errorbar([ind/4, ind/4 + 6], vals[0], yerr=vals[1], color='b', label='Sham', fmt='*-')
for ind, vals in enumerate(zip(chi_rh_stepFreq_mean, chi_rh_stepFreq_err)):
    stepaxs[1][1].errorbar([ind/4+2, ind/4 + 8], vals[0], yerr=vals[1], color='r', label='Chi', fmt='*-')

stepaxs[0][0].set_title('Left Foreleg', fontsize=16)
stepaxs[0][1].set_title('Right Foreleg', fontsize=16)
stepaxs[1][0].set_title('Left Hindleg', fontsize=16)
stepaxs[1][1].set_title('Right Hindleg', fontsize=16)
stepaxs[0][0].set_ylabel('Step Frequency (Hz)', fontsize=14)
stepaxs[1][0].set_ylabel('Step Frequency (Hz)', fontsize=14)

for row in stepaxs:
    for ax in row:
        ax.set_xticks([0.5, 2.5, 6.5, 8.5])
        ax.set_xticklabels(['Sham Day 3', 'Chi Day 3', 'Sham Day 6', 'Chi Day 6'], fontsize=15)

# longest running epoch 
# all running epoch 
with open('analyzed_data/simple_wheel/d3sham/longestRunnigEpoch.json', 'rb') as fileObj:
    d3sham = json.load(fileObj)
with open('analyzed_data/simple_wheel/d3chi/longestRunnigEpoch.json', 'rb') as fileObj:
    d3chi = json.load(fileObj)
with open('analyzed_data/simple_wheel/d6sham/longestRunnigEpoch.json', 'rb') as fileObj:
    d6sham = json.load(fileObj)
with open('analyzed_data/simple_wheel/d6chi/longestRunnigEpoch.json', 'rb') as fileObj:
    d6chi = json.load(fileObj)

lngst_fig, lngst_axs = plt.subplots(2,2)
sham_rl_pks = [[a,b] for a, b in zip(d3sham['rl']['peaks'], d6sham['rl']['peaks'])]
sham_rl_lags = [[a,b] for a, b in zip(d3sham['rl']['lags'], d6sham['rl']['lags'])]
chi_rl_pks = [[a,b] for a, b in zip(d3chi['rl']['peaks'], d6chi['rl']['peaks'])]
chi_rl_lags = [[a,b] for a, b in zip(d3chi['rl']['lags'], d6chi['rl']['lags'])]
sham_lr_pks = [[a,b] for a, b in zip(d3sham['lr']['peaks'], d6sham['lr']['peaks'])]
sham_lr_lags = [[a,b] for a, b in zip(d3sham['lr']['lags'], d6sham['lr']['lags'])]
chi_lr_pks = [[a,b] for a, b in zip(d3chi['lr']['peaks'], d6chi['lr']['peaks'])]
chi_lr_lags = [[a,b] for a, b in zip(d3chi['lr']['lags'], d6chi['lr']['lags'])]
for ind, val in enumerate(sham_lr_pks):
    lngst_axs[0][0].plot([ind/4, ind/4 + 6], val, 'b*-')
for ind, val in enumerate(chi_lr_pks):
    lngst_axs[0][0].plot([ind/4+2, ind/4 + 8], val, 'r*-')
for ind, val in enumerate(sham_rl_pks):
    lngst_axs[0][1].plot([ind/4, ind/4 + 6], val, 'b*-')
for ind, val in enumerate(chi_rl_pks):
    lngst_axs[0][1].plot([ind/4+2, ind/4 + 8], val, 'r*-')
for ind, val in enumerate(sham_lr_lags):
    lngst_axs[1][0].plot([ind/4, ind/4 + 6], val, 'b*-')
for ind, val in enumerate(chi_lr_lags):
    lngst_axs[1][0].plot([ind/4+2, ind/4 + 8], val, 'r*-')
for ind, val in enumerate(sham_rl_lags):
    lngst_axs[1][1].plot([ind/4, ind/4 + 6], val, 'b*-')
for ind, val in enumerate(chi_rl_lags):
    lngst_axs[1][1].plot([ind/4+2, ind/4 + 8], val, 'r*-')
lngst_axs[0][0].set_ylabel('Peak Correlation (a.u.)', fontsize=14)
lngst_axs[1][0].set_ylabel('Lag of Peak Correlation (s)', fontsize=14)
lngst_axs[0][0].set_title('Left Forelimb - Right Hindleg', fontsize=16)
lngst_axs[0][1].set_title('Right Forelimb - Left Hindleg', fontsize=16)
for row in lngst_axs:
    for ax in row:
        ax.set_xticks([0.5, 2.5, 6.5, 8.5])
        ax.set_xticklabels(['Sham Day 3', 'Chi Day 3', 'Sham Day 6', 'Chi Day 6'], fontsize=15)

###################################################################
#                           Complex Wheel                         #
###################################################################
# all running epoch 
with open('analyzed_data/complex_wheel/d3sham/allRunningEpochs.json', 'rb') as fileObj:
    d3sham = json.load(fileObj)
with open('analyzed_data/complex_wheel/d3chi/allRunningEpochs.json', 'rb') as fileObj:
    d3chi = json.load(fileObj)

# correlation peaks 
d3sham_rl_pks_mean = [np.mean(n) for n in d3sham['rl_pks']]
d3chi_rl_pks_mean = [np.mean(n) for n in d3chi['rl_pks']]
d3sham_rl_pks_err = [np.std(n) for n in d3sham['rl_pks']]
d3chi_rl_pks_err = [np.std(n) for n in d3chi['rl_pks']]
d3sham_lr_pks_mean = [np.mean(n) for n in d3sham['lr_pks']]
d3chi_lr_pks_mean = [np.mean(n) for n in d3chi['lr_pks']]
d3sham_lr_pks_err = [np.std(n) for n in d3sham['lr_pks']]
d3chi_lr_pks_err = [np.std(n) for n in d3chi['lr_pks']]
d3sham_rl_lags_mean = [np.mean(n) for n in d3sham['rl_lags']]
d3chi_rl_lags_mean = [np.mean(n) for n in d3chi['rl_lags']]
d3sham_rl_lags_err = [np.std(n) for n in d3sham['rl_lags']]
d3chi_rl_lags_err = [np.std(n) for n in d3chi['rl_lags']]
d3sham_lr_lags_mean = [np.mean(n) for n in d3sham['lr_lags']]
d3chi_lr_lags_mean = [np.mean(n) for n in d3chi['lr_lags']]
d3sham_lr_lags_err = [np.std(n) for n in d3sham['lr_lags']]
d3chi_lr_lags_err = [np.std(n) for n in d3chi['lr_lags']]

smpl_corr_fig, smpl_corr_axs = plt.subplots(2,2)
smpl_corr_axs[0][0].errorbar([i/4 for i in range(4)], d3sham_lr_pks_mean, yerr=d3sham_lr_pks_err, fmt='*', color='b')
smpl_corr_axs[0][0].errorbar([i/4 + 2 for i in range(4)], d3chi_lr_pks_mean, yerr=d3chi_lr_pks_err, fmt='*', color='r')
smpl_corr_axs[0][1].errorbar([i/4 for i in range(4)], d3sham_rl_pks_mean, yerr=d3sham_rl_pks_err, fmt='*', color='b')
smpl_corr_axs[0][1].errorbar([i/4 + 2 for i in range(4)], d3chi_rl_pks_mean, yerr=d3chi_rl_pks_err, fmt='*', color='r')
smpl_corr_axs[1][0].errorbar([i/4 for i in range(4)], d3sham_lr_lags_mean, yerr=d3sham_lr_lags_err, fmt='*', color='b')
smpl_corr_axs[1][0].errorbar([i/4 + 2 for i in range(4)], d3chi_lr_lags_mean, yerr=d3chi_lr_lags_err, fmt='*', color='r')
smpl_corr_axs[1][1].errorbar([i/4 for i in range(4)], d3sham_rl_lags_mean, yerr=d3sham_rl_lags_err, fmt='*', color='b')
smpl_corr_axs[1][1].errorbar([i/4 + 2 for i in range(4)], d3chi_rl_lags_mean, yerr=d3chi_rl_lags_err, fmt='*', color='r')
for row in smpl_corr_axs:
    for ax in row:
        ax.set_xticks([0.3, 2.3])
        ax.set_xticklabels(['Sham Day 3', 'Chi Day 3'], fontsize=15)
smpl_corr_axs[0][0].set_ylabel('Peak Correlation (a.u.)', fontsize=14)
smpl_corr_axs[1][0].set_ylabel('Lag of Peak Correlation (s)', fontsize=14)
smpl_corr_axs[0][0].set_title('Left Forelimb - Right Hindleg', fontsize=16)
smpl_corr_axs[0][1].set_title('Right Forelimb - Left Hindleg', fontsize=16)

# step frequency
smpl_step_fig, smpl_step_axs = plt.subplots(2,2)
d3sham_rf_stepFreq_mean = [np.mean(n) for n in d3sham['rf_step_freq']]
d3sham_lf_stepFreq_mean = [np.mean(n) for n in d3sham['lf_step_freq']]
d3sham_rh_stepFreq_mean = [np.mean(n) for n in d3sham['rh_step_freq']]
d3sham_lh_stepFreq_mean = [np.mean(n) for n in d3sham['lh_step_freq']]
d3sham_rf_stepFreq_err = [np.std(n) for n in d3sham['rf_step_freq']]
d3sham_lf_stepFreq_err = [np.std(n) for n in d3sham['lf_step_freq']]
d3sham_rh_stepFreq_err = [np.std(n) for n in d3sham['rh_step_freq']]
d3sham_lh_stepFreq_err = [np.std(n) for n in d3sham['lh_step_freq']]
d3chi_rf_stepFreq_mean = [np.mean(n) for n in d3chi['rf_step_freq']]
d3chi_lf_stepFreq_mean = [np.mean(n) for n in d3chi['lf_step_freq']]
d3chi_rh_stepFreq_mean = [np.mean(n) for n in d3chi['rh_step_freq']]
d3chi_lh_stepFreq_mean = [np.mean(n) for n in d3chi['lh_step_freq']]
d3chi_rf_stepFreq_err = [np.std(n) for n in d3chi['rf_step_freq']]
d3chi_lf_stepFreq_err = [np.std(n) for n in d3chi['lf_step_freq']]
d3chi_rh_stepFreq_err = [np.std(n) for n in d3chi['rh_step_freq']]
d3chi_lh_stepFreq_err = [np.std(n) for n in d3chi['lh_step_freq']]

smpl_step_axs[0][0].errorbar([i/4 for i in range(4)], d3sham_lf_stepFreq_mean, yerr=d3sham_lf_stepFreq_err, fmt='*', color='b')
smpl_step_axs[0][0].errorbar([i/4 + 2 for i in range(4)], d3chi_lf_stepFreq_mean, yerr=d3chi_lf_stepFreq_err, fmt='*', color='r')
smpl_step_axs[0][1].errorbar([i/4 for i in range(4)], d3sham_rf_stepFreq_mean, yerr=d3sham_rf_stepFreq_err, fmt='*', color='b')
smpl_step_axs[0][1].errorbar([i/4 + 2 for i in range(4)], d3chi_rf_stepFreq_mean, yerr=d3chi_rf_stepFreq_err, fmt='*', color='r')
smpl_step_axs[1][0].errorbar([i/4 for i in range(4)], d3sham_lh_stepFreq_mean, yerr=d3sham_lh_stepFreq_err, fmt='*', color='b')
smpl_step_axs[1][0].errorbar([i/4 + 2 for i in range(4)], d3chi_lh_stepFreq_mean, yerr=d3chi_lh_stepFreq_err, fmt='*', color='r')
smpl_step_axs[1][1].errorbar([i/4 for i in range(4)], d3sham_rh_stepFreq_mean, yerr=d3sham_rh_stepFreq_err, fmt='*', color='b')
smpl_step_axs[1][1].errorbar([i/4 + 2 for i in range(4)], d3chi_rh_stepFreq_mean, yerr=d3chi_rh_stepFreq_err, fmt='*', color='r')

smpl_step_axs[0][0].set_title('Left Foreleg', fontsize=16)
smpl_step_axs[0][1].set_title('Right Foreleg', fontsize=16)
smpl_step_axs[1][0].set_title('Left Hindleg', fontsize=16)
smpl_step_axs[1][1].set_title('Right Hindleg', fontsize=16)
smpl_step_axs[0][0].set_ylabel('Step Frequency (Hz)', fontsize=14)
smpl_step_axs[1][0].set_ylabel('Step Frequency (Hz)', fontsize=14)
for row in smpl_step_axs:
    for ax in row:
        ax.set_xticks([0.3, 2.3])
        ax.set_xticklabels(['Sham Day 3', 'Chi Day 3'], fontsize=15)

# longest running epoch 
with open('analyzed_data/complex_wheel/d3sham/longestRunnigEpoch.json', 'rb') as fileObj:
    d3sham = json.load(fileObj)
with open('analyzed_data/complex_wheel/d3chi/longestRunnigEpoch.json', 'rb') as fileObj:
    d3chi = json.load(fileObj)

fig, axs = plt.subplots(2,2)
axs[0][0].plot([i/4 for i in range(4)], d3sham['rl']['peaks'], 'b*')
axs[0][0].plot([i/4 + 2 for i in range(4)], d3chi['rl']['peaks'], 'r*')
axs[0][1].plot([i/4 for i in range(4)], d3sham['lr']['peaks'], 'b*')
axs[0][1].plot([i/4 + 2 for i in range(4)], d3chi['lr']['peaks'], 'r*')
axs[1][0].plot([i/4 for i in range(4)], d3sham['rl']['lags'], 'b*')
axs[1][0].plot([i/4 + 2 for i in range(4)], d3chi['rl']['lags'], 'r*')
axs[1][1].plot([i/4 for i in range(4)], d3sham['lr']['lags'], 'b*')
axs[1][1].plot([i/4 + 2 for i in range(4)], d3chi['lr']['lags'], 'r*')

axs[0][0].set_ylabel('Peak Correlation (a.u.)', fontsize=14)
axs[1][0].set_ylabel('Lag of Peak Correlation (s)', fontsize=14)
axs[0][0].set_title('Left Forelimb - Right Hindleg', fontsize=16)
axs[0][1].set_title('Right Forelimb - Left Hindleg', fontsize=16)
for row in axs:
    for ax in row:
        ax.set_xticks([0.3, 2.3])
        ax.set_xticklabels(['Sham Day 3', 'Chi Day 3'], fontsize=15)
plt.ion()
plt.show()