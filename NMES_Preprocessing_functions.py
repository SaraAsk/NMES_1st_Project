#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 17:42:57 2022

@author: sara
"""

import mne
import json
import pyxdf
import numpy as np
from scipy import stats
from scipy.stats import  zscore
from mne.channels import make_standard_montage


# =============================================================================
def XDF_correct_time_stamp_reject_pulses(f):
    
    
    marker = pyxdf.load_xdf(f, select_streams=[{'name': 'reiz_marker_sa'}])[0][0]
    brainvision = pyxdf.load_xdf(f, select_streams=[{'name': 'BrainVision RDA'}])[0][0]
    edcix = [i for i,v in enumerate(brainvision['info']['desc'][0]['channels'][0]['channel']) if v['label'][0] == 'EDC_R'][0]
    edcdat = brainvision['time_series'][:,edcix]
    out = {'pulse_BV':[], 'drop_idx_list': []}
    
    # pulses creates a list of the indices of the marker timestamps for the stimulation condition trials only
    # i.e., excluding the vigilance task trials
    pulses = [i for i,m in enumerate(marker['time_series']) if "\"stim_type\": \"TMS\"" in m[0]]

    # pulseinfo contains a list of the stim.condition time stamps and descriptions
    # each item in the list contains a list with the size 2: pulseinfo[i][0] is the timestamp corresponding with the index i from pulses,
    # pulseinfo[i][1] contains the corresponding stimulus description (i.e., stim phase and freq, etc.)
    pulseinfo = [[np.searchsorted(brainvision['time_stamps'], marker['time_stamps'][p]), marker['time_series'][p]] for p in pulses]
    n=0
    
    for i,p in enumerate(pulseinfo):
        pulse_idx = pulses[pulseinfo.index(p)]
        sample = p[0]

        # For the NMES study, we use the ECD_R data to identify the artifact
        # and we use a time window around the onset of the original reizmarker_timestamp: [sample-1500:sample+1500]
        onset = sample-1500
        offset = sample+1500
        edcep = edcdat[onset:offset]
        dmy= np.abs(stats.zscore(edcep))
        tartifact = np.argmax(dmy)
        
        # edcep contains 3000 timepoints or samples (-1500 to +1500 samples around the original rm_marker)
        # so, if tartifact is < 1500, the new marker is in the period before the original marker
        # if tartifact is >1500, the new marker is in the period after the original marker      
        corrected_timestamp = sample - 1500 + tartifact
 
        #print('the original marker ts was: ' + str(sample)+' and the corrected ts is: '+str(corrected_timestamp))
        
        # the section below is to check for trials where no clear stimulation artifact is present
        # a list of indices is created and saved in out['drop_idx_list'], to be used to reject 
        # these epochs when the preprocessing in MNE is started
        if max(dmy) < 3:
            n+=1
            out['drop_idx_list'].append(pulse_idx)
        out['pulse_BV'].append(corrected_timestamp)
    _, _, pulses_ind_drop = np.intersect1d(out['drop_idx_list'], pulses, return_indices=True)

# =============================================================================
#     pulses_ind_drop_filename = 'pulses_ind_drop_'+ str(f.parts[-3])+'_'+str(f.parts[-1][-8:-4])+'.p'
#     with open(str(save_folder_pickle) +pulses_ind_drop_filename, 'wb') as fp:
#         pickle.dump(pulses_ind_drop, fp, protocol=pickle.HIGHEST_PROTOCOL)
# =============================================================================
        
        
    """        
    Next, replace the original timestamps in the marker stream with the new ones.   
    - the original markers are stored in marker['time_stamps']
    - the new time stamp values are based on the brainvision['time_stamps'] values that 
    correspond with the brainvision['time_stamps'] index as stored in out['pulse_BV']
        E.g., 
        corrected_timestamp = 50961
        In [9]: brainvision['time_stamps'][50961]
        Out[9]: 374680.57453827135
        
    IMPORTANT:the values in corrected_timestamp (and pulse info) refer to the index of the timestamp, not
    the actual time value, of a timestamp in brainvision
    """
    
    marker_corrected = marker
    
    for i in range(len(pulses)):
        # for the stim.condition time stamps (corresponding to the indices stored in pulses)
        # replace original reizmarker (rm) timestamp value with the corrected timestamp value based on the EDC artifact (corrected_timestamp)
        rm_timestamp_idx = pulses[i]
        brainvision_idx = out['pulse_BV'][i]
        rm_timestamp_new_value = brainvision['time_stamps'][brainvision_idx] 
                
        #print('old value: '+str(marker['time_stamps'][pulses[i]]))
        # replace original stimulus onset time stamp with the new timestamp value
        marker_corrected['time_stamps'][rm_timestamp_idx] = rm_timestamp_new_value
        #print('new value: '+str(marker['time_stamps'][pulses[i]]))

        

    #### convert brainvision and corrected marker stream into a fif file that can be read by MNE ###    

    #marker_corrected = marker    #pyxdf.load_xdf(f, select_streams=[{'name': 'reiz_marker_sa'}])[0][0]
    data = brainvision   #pyxdf.load_xdf(f, select_streams=[{'name': 'BrainVision RDA'}])[0][0]
    marker_corrected['time_stamps'] -= data['time_stamps'][0] #remove clock offset
    
    channel_names = [c['label'][0] for c in data['info']['desc'][0]['channels'][0]['channel'] ]
    sfreq = int(data['info']['nominal_srate'][0])
    types = ['eeg']*64
    types.extend(['emg']*(len(channel_names)-64)) #64 EEG chans, rest is EMG/EKG
    info = mne.create_info(ch_names = channel_names, sfreq = sfreq, ch_types = types)
    raw = mne.io.RawArray(data = data['time_series'].T, info = info)
    
    if len(marker_corrected['time_stamps']) > 1:
        descs = [msg[0] for msg in marker_corrected['time_series']]
        ts = marker_corrected['time_stamps']
        
        sel = [i for i,v in enumerate(descs) if "TMS" in v]
        descs = [descs[i] for i in sel]
        
        ts = [ts[i] for i in sel]
        
        shortdescs = [json.loads(msg)['freq'] + 'Hz_' + json.loads(msg)['phase'] for msg in descs]

        anno = mne.Annotations(onset = ts, duration = 0, description = shortdescs)
        raw = raw.set_annotations(anno)
        
    ts_new = np.delete(ts, pulses_ind_drop)
    shortdescs_new = np.delete(shortdescs, pulses_ind_drop)
    anno = mne.Annotations(onset = ts_new, duration = 0, description = shortdescs_new)
    raw = raw.set_annotations(anno)      
    #print(len(ts), len(ts_new))
    #print(str(f.parts[-3]))
   
          


    return raw, len(ts) , len(ts_new)                  




from sklearn.decomposition import PCA
from statsmodels.regression.linear_model import OLS

def regress_out_pupils(raw, ocular_channels = ['Fpz', 'Fp1', 'Fp2', 'AF7', 'AF8'], method = 'PCA'):
    
    """
    raw: Continuous raw data in MNE format
    ocular_channels: can be labels of EOG channels or EEG channels close to the
        eyes if no EOG was recorded
    method: how to combine the ocular channels. Can be 'PCA', 'mean', or 'median'.
    """
    
    raw_data = raw.get_data(picks = 'eeg')
    ocular_data = raw.get_data(picks = ocular_channels)
    
    if method == 'PCA':
        pca = PCA()
        comps = pca.fit_transform(ocular_data.T)
        ocular_chan = comps[:,0]
    elif method == 'mean':
        ocular_chan = np.mean(ocular_data, axis = 0)
    elif method == 'median':
        ocular_chan = np.median(ocular_data, axis = 0)
    
    for ch in range(raw_data.shape[0]):
        m = OLS(raw_data[ch,:], ocular_chan)
        raw_data[ch,:] -= m.fit().predict()
    raw._data[:raw_data.shape[0],:] = raw_data
    return raw





def mark_bad_channels_interpolate(f, raw):

    """
    Detects channels above a certain threshold when looking at zscored data.
    Plots time series with pre-detected channels marked in red.
    Enables user to mark bad channels interactively and saves selection in raw object.

    Args:
        raw : MNE raw object with EEG data
        ch_names (list): list of strings with channel names
        threshold (float, int): threshold based on which to detect outlier channels 
                                (maximal zscored absolute standard deviation). Defaults to 1.5.

    Returns:
        MNE raw object: raw, with bad channel selection (bads) updated
    """


    ch_names = raw.info['ch_names']


    # plotting of channel variance
    vars = np.var(raw._data.T, axis=0)
    badchans_threshold = np.where(np.abs(zscore(vars)) > 1.5)
    #badchans = visual_inspection(vars)
    raw.info['bads'] = [ch_names[i] for i in list(badchans_threshold[0])]

    # filtering and plotting of raw data with marked bad channels
    # bandpass and bandstop filter data
# =============================================================================
#     raw.filter(0.5, 49, method='iir', verbose=0)
#     raw._data = mne.filter.notch_filter(raw._data, raw.info['sfreq'], 50, notch_widths=2,
#                                         picks=[i for i, ch in enumerate(ch_names) if ch not in raw.info['bads']], 
#                                         phase='zero', verbose=0)
# =============================================================================

    montage = make_standard_montage('standard_1005')
    raw.set_montage(montage)
    ch_names = raw.info['ch_names']
    badchans_threshold  = raw.info['bads']
    raw_eeg_interp = raw.interpolate_bads(reset_bads=True)
    
    

    return raw_eeg_interp, badchans_threshold




def MNE_raw_format(eegdata, ch_names, sfreq):

    """
    Helps get EEG data into MNE raw object format by using some default values.
    Channel types all defined as EEG and default montage used.

    Args:
        eegdata (numpy array): EEG data array, timepoints*channels
        ch_names (list): list of strings with channel names
        sfreq (float, int): sampling frequency

    Returns:
        raw : instance of MNE raw
    """
    
    import mne
    
    ch_types = ['eeg']*len(ch_names)
    
    info = mne.create_info(ch_names=ch_names, 
                           sfreq=sfreq, 
                           ch_types=ch_types,
                           verbose=0)

    raw = mne.io.RawArray(np.transpose(eegdata), info, verbose=0)

    raw.set_montage(mne.channels.make_standard_montage('standard_1005'), verbose=0)

    return raw




def visual_inspection(x, indexmode = 'exclude'):
    """
    Allows you to visually inspect and exclude elements from an array.
    The array x typically contains summary statistics, e.g., the signal
    variance for each trial.
    """

    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector
    import numpy as np

    x = np.array(x)
    x = x.flatten()
    nanix = np.zeros(len(x))
    
    
    
    
    def line_select_callback(eclick, erelease):
        """
        Callback for line selection.
    
        *eclick* and *erelease* are the press and release events.
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        print(" The button you used were: %s %s" % (eclick.button, erelease.button))
    

    fig, current_ax = plt.subplots()                 # make a new plotting range
    # plt.plot(np.arange(len(x)), x, lw=1, c='b', alpha=.7)  # plot something
    current_ax.plot(x, 'b.', alpha=.7)  # plot something

    print("\n      click  -->  release")

    # drawtype is 'box' or 'line' or 'none'
    RS = RectangleSelector(current_ax, line_select_callback,
                                    drawtype='box', useblit=True,
                                    button=[1],  # don't use middle button
                                    minspanx=5, minspany=5,
                                    spancoords='pixels',
                                    interactive=True)
    RSinv = RectangleSelector(current_ax, line_select_callback,
                                drawtype='box', useblit=True,
                                button=[3],  # don't use middle button
                                minspanx=5, minspany=5,
                                spancoords='pixels',
                                interactive=True)
    plt.connect('key_press_event', (RS, RSinv))

    while plt.fignum_exists(1):
        plt.cla()
        current_ax.set_ylim([np.min(x[np.where(nanix == 0)[0]]), 1.1*np.max(x[np.where(nanix == 0)[0]])])
        current_ax.plot(x, 'b.', alpha=.7)  # plot something
        if np.sum(nanix) > 0:
            current_ax.plot(np.squeeze(np.where(nanix == 1)), x[np.where(nanix == 1)], 'w.', alpha=.7)  # plot something

        fig.show()
        plt.pause(.1)
        if plt.fignum_exists(1):
            plt.waitforbuttonpress(timeout = 2)
            
            if (RS.geometry[1][1] > 1):
                exclix = np.where((x > min(RS.geometry[0])) & (x < max(RS.geometry[0])))[0]
                exclix = exclix[np.where((exclix > min(RS.geometry[1])) & (exclix < max(RS.geometry[1])))]
                nanix[exclix] = 1
            if (RSinv.geometry[1][1] > 1):
                exclix = np.where((x > min(RSinv.geometry[0])) & (x < max(RSinv.geometry[0])))[0]
                exclix = exclix[np.where((exclix > min(RSinv.geometry[1])) & (exclix < max(RSinv.geometry[1])))]
                nanix[exclix] = 0
            if not plt.fignum_exists(1):
                break
            else:
                plt.pause(.1)
        else:
            plt.pause(.1)
            break
    if indexmode == 'exclude':
    	return np.where(nanix == 1)[0]
    elif indexmode == 'keep':
    	return np.where(nanix == 0)[0]
    else:
    	raise ValueError
        
        
#%% ICA

import mne
from pathlib import Path

import deepdish as dd
from autoreject import AutoReject, get_rejection_threshold
import collections

# from .eeg_utils import *


# https://github.com/HemuManju/human-effort-classification-eeg/blob/223e320e7201f6c93cbe8f9728e401d7199453a2/src/data/clean_eeg_dataset.py
def autoreject_repair_epochs(epochs, reject_plot=False):
    """Rejects the bad epochs with AutoReject algorithm
    Parameter
    ----------
    epochs : Epoched, filtered eeg data
    Returns
    ----------
    epochs : Epoched data after rejection of bad epochs
    """
    # Cleaning with autoreject
    picks = mne.pick_types(epochs.info, eeg=True)  # Pick EEG channels
    ar = AutoReject(n_interpolate=[1, 4, 32],
                    n_jobs=6,
                    picks=picks,
                    thresh_func='bayesian_optimization',
                    cv=10,
                    random_state=42,
                    verbose=False)

    cleaned_epochs, reject_log = ar.fit_transform(epochs, return_log=True)

    if reject_plot:
        reject_log.plot_epochs(epochs, scalings=dict(eeg=40e-6))

    return cleaned_epochs


def append_eog_index(epochs, ica):
    """Detects the eye blink aritifact indices and adds that information to ICA
    Parameter
    ----------
    epochs : Epoched, filtered, and autorejected eeg data
    ica    : ica object from mne
    Returns
    ----------
    ICA : ICA object with eog indices appended
    """
    # Find bad EOG artifact (eye blinks) by correlating with Fp1
    eog_inds, scores_eog = ica.find_bads_eog(epochs,
                                             ch_name='Fp1',
                                             verbose=False)
    eog_inds.sort()
    # Append only when the correlation is high
    id_eog = [i for i, n in enumerate(scores_eog.tolist()) if abs(n) >= 0.65]
    ica.exclude += id_eog

    # Find bad EOG artifact (eye blinks) by correlation with Fp2
    eog_inds, scores_eog = ica.find_bads_eog(epochs,
                                             ch_name='Fp2',
                                             verbose=False)
    eog_inds.sort()
    # Append only when the correlation is high
    id_eog = [i for i, n in enumerate(scores_eog.tolist()) if abs(n) >= 0.75]
    ica.exclude += id_eog

    return ica


def clean_with_ica(epochs, reject_log, show_ica=False):
    """Clean epochs with ICA.
    Parameter
    ----------
    epochs : Epoched, filtered, and autorejected eeg data
    Returns
    ----------
    ica     : ICA object from mne
    epochs  : ICA cleaned epochs
    """

    picks = mne.pick_types(epochs.info,
                           meg=False,
                           eeg=True,
                           eog=False,
                           stim=False,
                           exclude='bads')
    ica = mne.preprocessing.ICA(n_components=None,
                                method="fastica",
                                verbose=False)
    # Get the rejection threshold using autoreject
# =============================================================================
#     reject_threshold = get_rejection_threshold(epochs)
#     ica.fit(epochs, picks=picks, reject=reject_threshold)
# =============================================================================
    ica.fit(epochs[~reject_log.bad_epochs])
    
    ica = append_eog_index(epochs, ica)  # Append the eog index to ICA
    if show_ica:
        ica.plot_components(inst=epochs)
    clean_epochs = ica.apply(epochs.copy())  # Remove selected components from the signal.

    return clean_epochs, ica


# =============================================================================
# def clean_dataset(epochs):
#     """Create cleaned dataset (by running autoreject and ICA)
#     with each subject data in a dictionary.
#     Parameter
#     ----------
#     subject : string of subject ID e.g. 7707
#     trial   : HighFine, HighGross, LowFine, LowGross
#     Returns
#     ----------
#     clean_eeg_dataset : dataset of all the subjects with different conditions
#     """
#     data  = {}
#     ica_epochs, ica = clean_with_ica(epochs)
#     repaired_eeg = autoreject_repair_epochs(ica_epochs)
#     data['eeg'] = ica_epochs
#     data['ica'] = ica
# 
# 
#     return data
#         
# =============================================================================

def clean_dataset(epochs, reject_log):
    """Create cleaned dataset (by running autoreject and ICA)
    with each subject data in a dictionary.
    Parameter
    ----------
    subject : string of subject ID e.g. 7707
    trial   : HighFine, HighGross, LowFine, LowGross
    Returns
    ----------
    clean_eeg_dataset : dataset of all the subjects with different conditions
    """
    data  = {}
    ica_epochs, ica = clean_with_ica(epochs, reject_log)
    repaired_eeg = autoreject_repair_epochs(ica_epochs)
    data['eeg'] = repaired_eeg
    data['ica'] = ica


    return data
        
