#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 14:24:51 2023

@author: sara
"""



import mne
import math
import json
import pickle
import pycircstat

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import  zscore
import matplotlib.pyplot as plt

import seaborn as sns; sns.set_theme()

import lmfit
import itertools  
import mne.stats
from tqdm import tqdm
from pathlib import Path
from scipy import stats
from multiprocessing import Pool
from scipy.interpolate import interp1d
from mne.stats import permutation_cluster_test
import phase_analysis_function_Jan as ph_analysis


lst = []
save_folder_pickle = "/home/sara/NMES/analyzed data/"

def extract_phase(filtered_data, fs, freq_band, cycles, angletype = 'degree'):
    ''' Extract the phase in degree (default) of a specific frequency band (alpha/mu or beta) with FFT
    Phase is extracted dynamically, dependending on frequencies ->
    X-cycles of each frequencies before TMS pulse are taken to get phase
    
    Args:
        filtered_data(complex):
            filtered data in epoch form 
            complex, because it's based on FFT filtered data
            epochs * times * channels   
        fs(int):
            sampling frequency   
        freq_band(string):
            name of frequency band to extract
            either 'alpha'or 'mu', or 'beta'
        cycles(int):
            based on how many cycles of frequencies the phase should be extracted      
        angletype(str):
            type of angle measurement, either 'radiant' or 'degree'
            default is to 'degree'
    
    Returns:
        float:
            eeg_epoch_filt_phase: phase of filtered epoch at sepecific frequency band
        list:
            freq_band: list of frequency-values in frequency band
                
    call: extract_phase(eeg_epoch_filtered, fs, 'beta', 3)
    '''
    
        
        
    if freq_band == '4Hz step':
        freq_band =  list(range(4, 41, 4))
    elif freq_band == '1Hz step':
        freq_band = list(range(4, 41))
    else:
        raise ValueError("Frequency band must be defined as 'alpha', 'mu' or 'beta'")  
        
    
    
    
    
    num_epoch = filtered_data.shape[0]
    eeg_epoch_filt_phase = np.nan*np.zeros((len(freq_band),num_epoch))

    for idx_epoch in range(0,num_epoch):
        for idx, value in enumerate(freq_band):
            #print(idx, value)
            respective_tw = math.ceil((1/value)*cycles*fs) + 5  # 3 cycles of frequency and 5 extra sample points
            #math.ceil: rounds a number up to the next largest integer
            cycle_tp_start = len(filtered_data[2]) - respective_tw 
            signal = filtered_data[idx_epoch, cycle_tp_start:]
            
            # put a 500ms window ?
            signal = filtered_data[idx_epoch, 0:500]
            
            
            N = len(signal) 
            fx = np.fft.rfft((signal), N)  # Extract FFT from real data. discretes Fourier Transform for real input
            T = 1.0/ fs # Timesteps
            freq = np.fft.fftfreq(N, d=T)  # Return the Discrete Fourier Transform sample frequencies.
            #idx_freq, = np.where(np.isclose(freq, value, atol=cycles+(T*N)))
            # not as straight forward
            
            # Mara's solution for indexing
            # more robust, a little bit slower
            diff = [np.abs(f-value) for f in freq]
            idx_freq = diff.index(np.min(diff))
            
            # or: is it possible to index based on cycles I look at?
            # phase = np.rad2deg(np.angle(fx)[cycles])      

# Not sure if i use the correct index here:
        # Extract phase 
        # degree -> *(np.pi/180) -> radiant
        # rad -> *(180/np.pi) -> degree
            if angletype == 'radiant':
                phase = np.angle(fx)[idx_freq] 
                eeg_epoch_filt_phase[idx,idx_epoch] = phase
                print(idx)
            
            elif angletype == 'degree':
               phase = np.rad2deg(np.angle(fx)[idx_freq])
               eeg_epoch_filt_phase[idx,idx_epoch] = phase
               
            else:
                raise ValueError("Specify angle type with either 'radiant' or 'degree'")    
          
    return eeg_epoch_filt_phase, freq_band


def assign_bin_class(epoched_phases, bin_num):
    ''' Assign each phase to the corresponding bin of phases on the unit circle
    
    Args:
        epoched_phases(float):
            phase of filtered epoch at specific frequency band
            frequencies * epochs ( * channels not yet) 
        bin_num(int):
            number of bins on unit circle that epoch phases will be assigned to
    Returns:
        float:
           bin_class: array with numbers from 0 to bin_num, corresponding to
           epoched_phases array
               frequencies * epochs
                
    call: assign_bin_class(eeg_epoch_filt_phase, bin_num = 16)
    '''    
    
# for phase as degree
    bin_anticlockwise = np.linspace(0,360,int(bin_num+1))  # cover half of the circle -> with half of bin_num
    bin_clockwise = np.linspace(-360,0,int(bin_num+1)) 


    bin_class = np.nan*np.zeros(epoched_phases.shape)

    for [row,col], phases in np.ndenumerate(epoched_phases):  
    # numbers correspond to the anti-clockwise unit circle eg. bin = 1 -> equals 22.5 deg phase for 16 bins
        if phases > 0:
                idx, = np.where(np.isclose(epoched_phases[row,col], bin_anticlockwise[:], atol=360/(bin_num*2)))
                # Returns a boolean array where two arrays are element-wise equal within a tolerance.
    # atol -> absolute tolerance level -> bin margins defined by 360° devided by twice the bin_num      
    # problem: rarely exactly between 2 bins -> insert nan
                if len(idx) > 1:
                    idx = np.nan
                bin_class[row,col] = idx
    
        elif phases < 0:
                idx, = np.where(np.isclose(epoched_phases[row,col], bin_clockwise[:], atol=360/(bin_num*2)))  
                if len(idx) > 1:
                    idx = np.nan      
                bin_class[row,col] = idx
                
                
                
                
    # bin_anticlockwise = np.linspace(0,180,int(bin_num/2+1))  # cover half of the circle -> with half of bin_num
    # bin_clockwise = np.linspace(-180,0,int(bin_num/2+1)) 
    #    # bin_clockwise = np.flip(np.linspace(-180,0,int(bin_num/2+1)))


    # bin_class = np.nan*np.zeros(epoched_phases.shape)

    # for [row,col], phases in np.ndenumerate(epoched_phases):  
    # # numbers correspond to the anti-clockwise unit circle eg. bin = 1 -> equals 22.5 deg phase for 16 bins
    #     if phases > 0:
    #             idx, = np.where(np.isclose(epoched_phases[row,col], bin_anticlockwise[:], atol=180/(bin_num)))
    # # atol -> absolute tolerance level -> bin margins defined by 360° devided by twice the bin_num      
    # # problem: rarely exactly between 2 bins -> insert nan
    #             if len(idx) > 1:
    #                 idx = np.nan
    #             bin_class[row,col] = idx
    
    #     elif phases < 0:
    #             idx, = np.where(np.isclose(epoched_phases[row,col], bin_clockwise[:], atol=180/(bin_num)))  
    #             if len(idx) > 1:
    #                 idx = np.nan      
    #             bin_class[row,col] = idx*2
    # PROBLEM -> 0 and -180 get the same bin class of 0
            
            
    return bin_class

def get_phase_bin_mean(epoched_phases, bin_class, bin_num, eeg_anticlockwise_phase = True):
    ''' take the values for each bin and average them to get the mean of each phase-bin
    
    Args:
        epoched_phases(float):
            phase of filtered epoch at sepecific frequency band
            frequencies * epochs ( * channels not yet)
        bin_class (float):
            array with values of bin numbers, corresponding to the phases in epoched_phases
            frequencies * epochs   
        bin_num(int):
            number of bins on unit circle that epoch phases will be assigned to     
        eeg_anticlockwise_phase(bool):
            if True: returns also an array with anti-clockwise(only positive) 
            values of phases; default is False  
        
    Returns:
        list:
            phase_bin_means: list of means of each trial in corresponding bin
    Optional:
        float:
            eeg_phase: only positive values for phase 
            (for an anticlockwise unit circel)
    
    call: get_phase_bin_mean(eeg_epoch_filt_phase, bin_class, bin_num = 16)
  
    '''
    # get mean of bins with vector in complex space
    # x = np.randmom.random(10)  * 2* np.pi
    # np.angle(np.exp(1j * x).mean())

    # make surethat same bins (esp. 0/360°) are projected on same bin)
    eeg_phase  =  epoched_phases.copy()
    for [r,c], value in np.ndenumerate(eeg_phase):
        if value < 0 :
            eeg_phase[r,c] = eeg_phase[r,c] + 360

     
    phase_bin_means = list(range(0,bin_num)) 

    # get radiants of values
    phase_rad = np.deg2rad(eeg_phase)
    
    # take mean for every phase bin
    # check where which values correspond to the bin_class -> take mean of all those values
    # change phase into complex number values -> np.exp(1j*phase)
    # take the mean of the vectors in complex space
    for value in list(range(0,bin_num)): 
        phase_bin_means[value] = np.angle(np.exp(1j * phase_rad[np.where(bin_class[:,:] == value)]).mean())
   
    # bin 0 and last bin have to be combined, both sit at same side 0°/360°
        if value == 0:
             bin_0 = phase_rad[np.where(bin_class[:,:] == value)]
             bin_0_360 = np.append(bin_0, phase_rad[np.where(bin_class[:,:] == bin_num)]) 
             phase_bin_means[value] = np.angle(np.exp(1j * bin_0_360).mean())
    
    # get phase value back in degrees
    phase_bin_means = np.rad2deg(phase_bin_means)
   
    return phase_bin_means










def get_phase_bin_mean_each_freq(target_freq, epoched_phases, bin_class, bin_num, eeg_anticlockwise_phase = True):
    ''' take the values for each bin and average them to get the mean of each phase-bin in each freq
    
    Args:
        epoched_phases(float):
            phase of filtered epoch at sepecific frequency band
            frequencies * epochs ( * channels not yet)
        bin_class (float):
            array with values of bin numbers, corresponding to the phases in epoched_phases
            frequencies * epochs   
        bin_num(int):
            number of bins on unit circle that epoch phases will be assigned to     
        eeg_anticlockwise_phase(bool):
            if True: returns also an array with anti-clockwise(only positive) 
            values of phases; default is False  
        
    Returns:
        list:
            phase_bin_means: list of means of each trial in corresponding bin
    Optional:
        float:
            eeg_phase: only positive values for phase 
            (for an anticlockwise unit circel)
    
    call: get_phase_bin_mean(eeg_epoch_filt_phase, bin_class, bin_num = 16)
  
    '''
    # get mean of bins with vector in complex space
    # x = np.randmom.random(10)  * 2* np.pi
    # np.angle(np.exp(1j * x).mean())

    # make surethat same bins (esp. 0/360°) are projected on same bin)
    eeg_phase  =  epoched_phases.copy()
    for [r,c], value in np.ndenumerate(eeg_phase):
        if value < 0 :
            eeg_phase[r,c] = eeg_phase[r,c] + 360

     
    phase_bin_means = np.zeros([len(target_freq), bin_num])

    # get radiants of values
    phase_rad = np.deg2rad(eeg_phase)
    
    # take mean for every phase bin
    # check where which values correspond to the bin_class -> take mean of all those values
    # change phase into complex number values -> np.exp(1j*phase)
    # take the mean of the vectors in complex space
    for i,freq in enumerate(target_freq):
        print(i,freq)
    
        for value in list(range(0,bin_num)): 
            phase_bin_means[i, value] = np.angle(np.exp(1j * phase_rad[i,:][np.where(bin_class[i,:] == value)]).mean())
       
        # bin 0 and last bin have to be combined, both sit at same side 0°/360°
            if value == 0:
                 bin_0 = phase_rad[i,:][np.where(bin_class[i,:] == value)]
                 bin_0_360 = np.append(bin_0, phase_rad[i,:][np.where(bin_class[i,:] == bin_num)]) 
                 phase_bin_means[i, value] = np.angle(np.exp(1j * bin_0_360).mean())
        
        # get phase value back in degrees
        phase_bin_means = np.rad2deg(phase_bin_means)
   
    return phase_bin_means






def plot_phase_bins(bin_means, bin_class, bin_num, scaling_proportion):
    ''' plots the mean of phase-bins and also plots the bins with the height 
    representing the number of trials within the bin (in proportion)
    
    
    Args:
        bin_means(list):
            list of means of each trial in corresponding bin
        bin_class (float):
            array with values of bin numbers, corresponding to the phases in epoched_phases
            frequencies * epochs
        bin_num(int):
            number of bins on unit circle that epoch phases will be assigned to
        scaling_proportion(int):
            an integer that scales the height of bins in order to adjust to 
            unit circle with radius = 1
            e.g. amount of trials in bin 5 = 230 -> scale by 300: 230/300 = 0.7667
            -> this bin will have a height of 0.7667 on unit circle when plotted
        
    Returns:
        plotted phases within corresponding bin
        also: height of displayed bin shows number of trials within bin (higher -> more trials)
    
    call: plot_phase_bins(bin_means, bin_class, bin_num, scaling_proportion = 300)
    '''
    theta = np.linspace(0.0, 2 * np.pi, bin_num, endpoint=False)
    rad_height = np.nan*np.zeros(bin_num+1)  # +1 because bin 0 and 16 are not combined yet
    unique, counts = np.unique(bin_class, return_counts=True) 
    # how many cases are there for each unique bin? -> dictionary of unique values of array bin_class
 
    for idx in range(unique.size):
        if unique[idx].is_integer() is True:  # Only for intger values (so, no nan due to phase in the middle of 2 bins)
               rad_height[idx] = counts[idx]/scaling_proportion  # Set height in proportion
               if unique[idx] == 0:  # For bin 0 and last bin, combine values (0/360°)
                   rad_height[0] = (counts[0]+counts[-1])/scaling_proportion
        
    # Exclude bins with nan value and the last bin (already combined with first bin)    
    nan_array = ~(np.isnan(rad_height))
    rad_height = rad_height[nan_array]
    rad_height = rad_height[:-1] 
    
    R1 = [0,1]  # Defined as UNIT circle (radius = 1)
    bin_phase_rad = np.deg2rad(bin_means)
    
    plt.figure()
    plt.polar([bin_phase_rad, bin_phase_rad], R1, lw=2, color = 'navy')
    width = (2*np.pi) / bin_num
    ax = plt.subplot(111, projection='polar')
    ax.bar(theta, rad_height, width=width, bottom=0.0, color = 'lightgrey' , edgecolor = 'grey')

    return (plt)










# =============================================================================

import pyxdf


def XDF_correct_time_stamp_reject_pulses_bip(f):
    
    
    marker = pyxdf.load_xdf(f, select_streams=[{'name': 'reiz_marker_sa'}])[0][0]
    brainvision = pyxdf.load_xdf(f, select_streams=[{'name': 'BrainVision RDA'}])[0][0]
    edcix = [i for i,v in enumerate(brainvision['info']['desc'][0]['channels'][0]['channel']) if v['label'][0] == 'EDC_R'][0]
    edcdat = brainvision['time_series'][:,edcix]
    out = {'pulse_BV':[], 'drop_idx_list': []}
    # bipolar signal near to C3
    bipolar = pyxdf.load_xdf(f, select_streams=[{'name': 'Spongebob-Data'}])[0][0]
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

    pulses_ind_drop_filename = 'pulses_ind_drop_'+ str(f.parts[-3])+'_'+str(f.parts[-1][-8:-4])+'.p'
    with open(str(save_folder_pickle) +pulses_ind_drop_filename, 'wb') as fp:
        pickle.dump(pulses_ind_drop, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
        
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
    info_bi =  mne.create_info(13, sfreq=info['sfreq'])
    raw_bip = mne.io.RawArray(bipolar['time_series'].T, info = info_bi).set_annotations(anno)     
    # Index of bipolar channel
    #raw_bip = raw_bip.pick_channels(['1'])
    #print(len(ts), len(ts_new))
    #print(str(f.parts[-3]))
    cols = ['','', 'Pulses', 'Pulses Corrected']
    lst.append(['', '',  len(ts) , len(ts_new)])
    df1 = pd.DataFrame(lst, columns=cols)     
    df1.to_csv ('/home/sara/NMES/analyzed data/' +'Subject_ pulse.csv', index = None, header=True)    
    return  raw_bip, f.parts                  








def get_ERP_1st_2nd(exdir_epoch, save_folder, plot =True):
    import matplotlib.pyplot as plt
    plt.style.use('default')
    import numpy as np
    import mne
    import os
    evoked_all_condition_data = os.path.join(exdir_epoch,'all_subs_ave.fif')
    Evoked_GrandAv= mne.read_evokeds(evoked_all_condition_data, verbose=False)
    Evoked_GrandAv = Evoked_GrandAv[0]    
 
   
    if plot:
        
        Evoked_GrandAv.info['bads'] = ['TP9', 'FT9']
        Evoked_GrandAv = Evoked_GrandAv.interpolate_bads(reset_bads=True, mode='accurate',  origin=(0., 0., 1.))
        evoked_peaks = Evoked_GrandAv.plot_joint(times = [0.03, 0.060, 0.120, 0.190], topomap_args = { 'sphere':(0.00, 0.00, 0.00, 0.11)})
        evoked_peaks.savefig(save_folder + 'components.svg')
        
        
           
        all_times = np.arange(0, 0.3, 0.02)
        a  = Evoked_GrandAv.plot_topomap(all_times, ch_type='eeg', time_unit='s', ncols=5, nrows='auto')
        a.savefig(save_folder+ 'topo_steps.svg')

        Evoked_GrandAv.plot(gfp=True, spatial_colors=True)        


        #all_times = np.arange(-0.2, 0.5, 0.01)
        #topo_plots = Evoked_GrandAv.plot_topomap(all_times, ch_type='eeg', time_unit='s', ncols=8, nrows='auto')
    return(Evoked_GrandAv.ch_names)



# =============================================================================







def cosinus(x, amp, phi):
    return amp * np.cos(x + phi)

def unif(x, offset):
    return offset

def do_one_perm(model, params, y,x):
    resultperm = model.fit(y, params, x=x)
    return resultperm.best_values['amp']


def do_cosine_fit(erp_amplitude, phase_bin_means, freq_band, labels, perm = True):

    
    cosinefit = {}
    amplitudes_cosine = np.zeros([len(freq_band), len(labels)*2])
    pvalues_cosine = np.zeros([len(freq_band), len(labels)*2])

    
    
    x = np.radians(np.array([0, 45, 90, 135, 180, 225, 270, 315]))
    #x = phase_bin_means
    
    for i in range(len(erp_amplitude)):
        cosinefit[str(i)] = {}
        for jf, f in enumerate(freq_band):    
            print('cosine fits for frequency {}'.format(f))
            y = zscore(np.array(list(erp_amplitude[str(i)][f].values())))
            cosinefit[str(i)][str(f)] = []
            fits = []
            for phase_start in [-np.pi/2, 0, np.pi/2]:      
        
                amp_start = np.sqrt(np.mean(y**2))
                model = lmfit.Model(cosinus)
                params = model.make_params()
        
                params["amp"].set(value=amp_start, min=0, max=np.ptp(y)/2)
                params["phi"].set(value=phase_start, min=-np.pi, max=np.pi)
                data = {ph: np.mean(y[x == ph]) for ph in np.unique(x)}
                fits.append(model.fit(y, params, x=x))
                
            result = fits[np.argmin([f.aic for f in fits])]
            
            if perm:
                model = lmfit.Model(cosinus)
                params = result.params
                dataperm = []
            
                # use all possible combinations of the 8 phase bins to determine p.
                # Take out the first combination because it is the original
                all_perms = list(itertools.permutations(x))
                del all_perms[0]
            
                for iper in tqdm(range(len(all_perms))):
                    x_shuffled = all_perms[iper]
                    dataperm.append([model,params, y, x_shuffled])
            
                with Pool(4) as p:
                    surrogate = p.starmap(do_one_perm, dataperm)
            else: 
                surrogate = [np.nan]
                
            
            nullmodel = lmfit.Model(unif)
            params = nullmodel.make_params()
            params["offset"].set(value=np.mean(y), min=min(y), max=max(y))
            nullfit = nullmodel.fit(y, params, x=x)
            surrogate = np.array(surrogate)
            surrogate = surrogate[np.invert(np.isnan(surrogate))]
            
            cosinefit[str(i)][str(f)].append( { 'Model': 'cosinus', 
                                    'Frequency': f, 
                                    'Fit': result,
                                    'data': data, 
                                    'amp': result.best_values['amp'], 
                                    'surrogate': surrogate, 
                                    'p':[np.nan if perm == False else sum(np.abs(surrogate) >= np.abs(result.best_values['amp']))/len(surrogate)][0], 
                                    'std':[np.nan if perm == False else np.std(surrogate)][0], 
                                    'nullmodel':nullfit,
                                    })
            
            amplitudes_cosine[jf, i] = result.best_values['amp']
            pvalues_cosine[jf, i] = [np.nan if perm == False else sum(np.abs(surrogate) >= np.abs(result.best_values['amp']))/len(surrogate)][0] 
    
    
    
    return cosinefit, amplitudes_cosine, pvalues_cosine



def do_cosine_fit_ll(erp_amplitude, phase_bin_means, freq_band, labels, subjects, perm = True):

    



    """ 
    Inputs: 
        
    erp_amplitude: is a dictionary file of two ERPs of each target frequency and phase.
                   This variable is calculated by first averaging over the channels within 
                   the chosen cluster and then averaging over epochs in the main script. 
                   Z scoring happens inside this function. I have one value for each ERP,
                   target freq and target phase and I do z scoring for each ERP, target freq
                   within the phases.
                                        ____________                  
                                       \  ____0°    \      
                    ______ 4Hz ________\ |    .     \       
        ______ ERP1|______ 8Hz         \ |    .     \     
       |           |         .         \ |____315°  \   
       |           |         .         \____________\                     
ERPs               |______40Hz               \
       |                                     \                                                       
       |                                     \
       |                                     \                             
       |______ ERP2                          \
                                         z scoring
                                             \
                                             \
                                             \
                                       cosine fitting"""     


    
    cosinefit = {}
    amplitudes_cosine = np.zeros([len(freq_band), len(labels)])
    pvalues_cosine = np.zeros([len(freq_band), len(labels)])

    
    
    x = np.radians(np.array([0, 45, 90, 135, 180, 225, 270, 315]))
    #x = phase_bin_means
    
    for i in range(len(erp_amplitude)):
        cosinefit[str(i)] = {}
        for jf, f in enumerate(freq_band):    
            print('cosine fits for frequency {}'.format(f))
            if subjects == 'individual':
                y = zscore(list(erp_amplitude[str(i)][str(f)].values()))
            else:
                y = erp_amplitude[str(i)][str(f)]
# =============================================================================
#             if(math.isnan(((erp_amplitude[str(0)][str(8)].values()))[0]) == True):
#                 break
#            
# =============================================================================
            cosinefit[str(i)][str(f)] = []
            fits = []
            for phase_start in [-np.pi/2, 0, np.pi/2]:   
        
                amp_start = np.sqrt(np.mean(y**2))
                model = lmfit.Model(cosinus)
                params = model.make_params()
        
                params["amp"].set(value=amp_start, min=0, max=np.ptp(y)/2)
                params["phi"].set(value=phase_start, min=-np.pi, max=np.pi)
                data = {ph: np.mean(y[x == ph]) for ph in np.unique(x)}
                #data = y
                fits.append(model.fit(y, params, x=x))
                
            result = fits[np.argmin([f.aic for f in fits])]
            
            if perm:
                model = lmfit.Model(cosinus)
                params = result.params
                dataperm = []
            
                # use all possible combinations of the 8 phase bins to determine p.
                # Take out the first combination because it is the original
                all_perms = list(itertools.permutations(x))
                del all_perms[0]
            
                for iper in tqdm(range(len(all_perms))):
                    x_shuffled = all_perms[iper]
                    dataperm.append([model,params, y, x_shuffled])
            
                with Pool(4) as p:
                    surrogate = p.starmap(do_one_perm, dataperm)
            else: 
                surrogate = [np.nan]
                
            
            nullmodel = lmfit.Model(unif)
            params = nullmodel.make_params()
            params["offset"].set(value=np.mean(y), min=min(y), max=max(y))
            nullfit = nullmodel.fit(y, params, x=x)
            surrogate = np.array(surrogate)
            surrogate = surrogate[np.invert(np.isnan(surrogate))]
            
            cosinefit[str(i)][str(f)].append( { 'Model': 'cosinus', 
                                    'Frequency': f, 
                                    'Fit': result,
                                    'data': data, 
                                    'amp': result.best_values['amp'], 
                                    'surrogate': surrogate, 
                                    'p':[np.nan if perm == False else sum(np.abs(surrogate) >= np.abs(result.best_values['amp']))/len(surrogate)][0], 
                                    'std':[np.nan if perm == False else np.std(surrogate)][0], 
                                    'nullmodel':nullfit,
                                    })
            
            amplitudes_cosine[jf, i] = result.best_values['amp']
            pvalues_cosine[jf, i] = [np.nan if perm == False else sum(np.abs(surrogate) >= np.abs(result.best_values['amp']))/len(surrogate)][0] 
    
    
    
    return cosinefit, amplitudes_cosine, pvalues_cosine



def fig_2a_plot(erp_amplitude, freq_band , subject_info, freq_step_i, save_folder, vmin , vmax):
    # Plotting the heatmap for each ERP Fig2.a Torrecillos 2020
    
    from scipy import ndimage 
    from matplotlib.patches import Rectangle

    def get_largest_component(image):
        """
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html
        get the largest component from 2D image
        image: 2d array
        """
        
        # Generate a structuring element that will consider features connected even 
        #s = [[1,0,1],[0,1,0],[1,0,1]] 
        s = ndimage.generate_binary_structure(2,2)
        labeled_array, numpatches = ndimage.label(image, s)
        sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
        max_label = np.where(sizes == sizes.max())[0] + 1
        output = np.asarray(labeled_array == max_label, np.uint8)
        return  output 
        
    

        
    
    
    
        
    for erp in range(len(erp_amplitude)):
        data_erp = {}       
    
        
        for jf, freq in enumerate(freq_band):   
            data_erp[str(freq)] = erp_amplitude[str(erp)][freq]   
        data_erp_df = pd.DataFrame(data_erp) 
        data_erp_df = data_erp_df.T
        data_erp_arr = zscore(data_erp_df.to_numpy())
        fig = plt.figure()
        data_erp_df = pd.DataFrame(data_erp) 
        data_erp_df = data_erp_df.T
        
        
        # Plotting the biggest cluster
        # Setting a threshold to equal to standard deviation of ERP amplitude
        arr = data_erp_arr > np.std(data_erp_arr)
        arr_largest_cluster = get_largest_component(arr)
        
        
        data_erp_df = pd.DataFrame(data_erp) 
        data_erp_df = data_erp_df.T
        data_erp_df_rename = data_erp_df.rename(columns={0:'0', 1:'45', 2:'90', 3:'135', 4:'180', 5:'225', 6:'270', 7:'315'})
        data_erp_df_ph_reorder = data_erp_df_rename.reindex(columns = ["0", "45", "90", "135","180" , "225","270","315"])
        ax = sns.heatmap(data_erp_df_ph_reorder,   cmap ='viridis', vmin = vmin, vmax = vmax)
        arr_largest_cluster_ind = np.argwhere(arr_largest_cluster == 1)
        for i in range(len(arr_largest_cluster_ind)):
            ax.add_patch(Rectangle((arr_largest_cluster_ind[i][1], arr_largest_cluster_ind[i][0]),  1, 1,  ec = 'cyan', fc = 'none', lw=2, hatch='//'))
        
        # swap the axes
        ax.invert_yaxis()
        plt.xlabel("Phases", fontsize=16)
        plt.ylabel("Frequencies", fontsize=16)
        
        if erp==0 and str(subject_info == 'Group Average'):
             print(0)
             plt.title(f'1st ERP, Subject: {subject_info, freq_step_i}')
             fig.savefig(save_folder + '/cluster_freq/' + 'fig_2c_biggest_cluster' + '_' + str(subject_info) + '_'+ '1st ERP' +'_' + str(freq_step_i) + '.png')
        elif erp==0 and str(subject_info[-2] == 'Experiment'): 
             plt.title(f'1st ERP, Subject: {subject_info[-3], freq_step_i}')
             fig.savefig(save_folder +  'fig_2c_biggest_cluster' + '_' + str(subject_info[-3]) + '_'+ '1st ERP' +'_' + str(freq_step_i) + '.png')
             
        elif erp==1 and str(subject_info == 'Group Average'):
             plt.title(f'2nd ERP, Subject: {subject_info, freq_step_i}')
             fig.savefig(save_folder + '/cluster_freq/' + 'fig_2c_biggest_cluster' + '_' + str(subject_info) + '_'+ '2nd ERP' +'_' + str(freq_step_i) + '.png')
        elif erp==1 and str(subject_info[-2] == 'Experiment'):  
             plt.title(f'2nd ERP, Subject: {subject_info[-3], freq_step_i}')
             fig.savefig(save_folder + 'fig_2c_biggest_cluster' + '_' + str(subject_info[-3]) + '_'+ '2nd ERP' +'_' + str(freq_step_i) + '.png')
         
            
         

        plt.figure()
        data_erp_df_rename = data_erp_df.rename(columns={0:'0', 1:'45', 2:'90', 3:'135', 4:'180', 5:'225', 6:'270', 7:'315'})
        data_erp_df_ph_reorder = data_erp_df_rename.reindex(columns = ["0", "45", "90", "135", "180" , "225","270","315"])
        ax = sns.heatmap(data_erp_df_ph_reorder,   cmap ='viridis', vmin = vmin, vmax = vmax)
        arr_all_cluster_ind = np.argwhere(arr == True)
        for i in range(len(arr_all_cluster_ind)):
            ax.add_patch(Rectangle((arr_all_cluster_ind[i][1], arr_all_cluster_ind[i][0]),  1, 1,  ec = 'cyan', fc = 'none', lw=2, hatch='//'))
        
        # swap the axes
        ax.invert_yaxis()
        plt.xlabel("Phases", fontsize=16)
        plt.ylabel("Frequencies", fontsize=16)
        
        if erp==0 and str(subject_info == 'Group Average'):
             plt.title(f'1st ERP, Subject: {subject_info, freq_step_i}')
             fig.savefig(save_folder + '/cluster_freq/' + 'fig_2c_all_cluster' + '_' + str(subject_info) + '_'+ '1st ERP' +'_' + str(freq_step_i) + '.png')
        elif erp==0 and str(subject_info[-2] == 'Experiment' ):
             plt.title(f'1st ERP, Subject: {subject_info[-3], freq_step_i}')
             fig.savefig(save_folder + 'fig_2c_all_cluster' + '_' + str(subject_info[-3]) + '_'+ '1st ERP' +'_' + str(freq_step_i) + '.png')
             
        elif erp==1 and str(subject_info == 'Group Average'):
            plt.title(f'2nd ERP, Subject: {subject_info, freq_step_i}')
            fig.savefig(save_folder + '/cluster_freq/' + 'fig_2c_all_cluster' + '_' + str(subject_info) + '_'+ '2nd ERP' +'_' + str(freq_step_i) + '.png')
        elif erp==1 and str(subject_info[-2] == 'Experiment'): 
            plt.title(f'2nd ERP, Subject: {subject_info[-3], freq_step_i}')
            fig.savefig(save_folder + 'fig_2c_all_cluster' + '_' + str(subject_info[-3]) + '_'+ '2nd ERP' +'_' + str(freq_step_i) + '.png')
            
    return  fig     


def fig_2c_plot(erp_amplitude, freq_band, cosinefit, subject_info, freq_step_i, save_folder):
    # Fig 2.c
    for i in range(len(erp_amplitude)):
        mod_depth = {}
        #data_z = {}
        #data_z_min = {}
        #data_z_max = {}
        best_fit = {}
        best_fit_min = {}
        best_fit_max = {}
        surr_min = {}
        surr_max = {}
        for jf, freq in enumerate(freq_band):   
           mod_depth[str(freq)] = cosinefit[str(i)][str(freq)][0]['amp']
           #data_z[str(freq)] =  np.array(list(cosinefit[str(i)][str(freq)][0]['data'].values()))
           #data_z_min[str(freq)] = min(data_z[str(freq)])
           #data_z_max[str(freq)] = max(data_z[str(freq)])
           best_fit[str(freq)] = cosinefit[str(i)][str(freq)][0]['Fit'].best_fit
           best_fit_min[str(freq)]  = min(cosinefit[str(i)][str(freq)][0]['Fit'].best_fit)
           best_fit_max [str(freq)] = max(cosinefit[str(i)][str(freq)][0]['Fit'].best_fit)
           surr_min[str(freq)] = min(cosinefit[str(i)][str(freq)][0]['surrogate'])
           surr_max[str(freq)] = max(cosinefit[str(i)][str(freq)][0]['surrogate'])
           # Fig2.C
        fig = plt.figure()
        plt.plot(np.array(freq_band), (np.array(list(best_fit_max.values())) + np.array(list(best_fit_min.values())))/1, 'k')
        #plt.fill_between(np.array(freq_band), np.array(list(data_z_min.values())), np.array(list(data_z_max.values())), color = '0.8')
        plt.fill_between(np.array(freq_band), np.array(list(best_fit_min.values())), np.array(list(best_fit_max.values())), color = '0.8')
        #plt.fill_between(np.array(freq_band), np.array(list(surr_min.values())), np.array(list(surr_max.values()))/2, color = 'r')
        #plt.plot(np.array(list(best_fit.values())), 'r')
        plt.xlabel("Frequecies")
        plt.ylabel("Strength of Mod")

            
        if i==0 and str(subject_info == 'Group Average'):
             plt.title(f'1st ERP, Subject: {subject_info, freq_step_i}')
             fig.savefig(save_folder + '/cluster_freq/' + 'fig_2c_all_cluster' + '_' + str(subject_info) + '_'+ '1st ERP' +'_' + str(freq_step_i) + '.png')
        elif i==0 and str(subject_info[-2] == 'Experiment' ):
             plt.title(f'1st ERP, Subject: {subject_info[-3], freq_step_i}')
             fig.savefig(save_folder + 'fig_2c_all_cluster' + '_' + str(subject_info[-3]) + '_'+ '1st ERP' +'_' + str(freq_step_i) + '.png')
             
        elif i==1 and str(subject_info == 'Group Average'):
             plt.title(f'2nd ERP, Subject: {subject_info, freq_step_i}')
             fig.savefig(save_folder + '/cluster_freq/' + 'fig_2c_all_cluster' + '_' + str(subject_info) + '_'+ '2nd ERP' +'_' + str(freq_step_i) + '.png')
        elif i==1 and str(subject_info[-2] == 'Experiment'): 
             plt.title(f'2nd ERP, Subject: {subject_info[-3], freq_step_i}')
             fig.savefig(save_folder + 'fig_2c_all_cluster' + '_' + str(subject_info[-3]) + '_'+ '2nd ERP' +'_' + str(freq_step_i) + '.png')
    
    return fig




def Select_Epochs_frequency(epochs, freq):
   
    
    index_list = []
    events_array = epochs.events
    event_id_dict = epochs.event_id
    # example o event description for acute NMES study: “freq”: “4”, “phase”: “0”
    freq_to_select = str(freq) 

    
    
    for i in range(len(events_array)):
        event_code = events_array[i,2]
        event_id_key = list(event_id_dict.keys())[list(event_id_dict.values()).index(event_code)]
        
        if freq >= 0 and freq <= 40:

            #if (freq_to_select in str(event_id_key[:(event_id_key.find('_') -2)])) == True and (phase_to_select in str(event_id_key[event_id_key.find('_') + 1:])) == True:
            if (freq_to_select == str(event_id_key[:(event_id_key.find('_') -2)]))   :    
                index_list.append(i)      
         
    return index_list



def Select_Epochs(epochs, freq, phase):
    """ 
    this is a function that will identify epochs based on their key (a string) in event_id, 
    which describes the stimulation condition
        
    selection depends on the frequency and the phase of interest
        
    the function returns a list of event indices, that only includes the indices of epochs that contained 
    stimulation at the desired frequency and phase
        
        
    data: epochs data in MNE format
    freq: an integer number, this can be any number between 0 and 40 and depends on the frequencies
    that were stimulated in your study (and thus described in your event description (a string) in event_id)
    phase: an integer number, this can be any number between 0 and 360 and depends on the phases
    that were stimulated in your study (and thus described in your event description in event_id)
    """
    
    index_list = []
    events_array = epochs.events
    event_id_dict = epochs.event_id
    # example o event description for acute NMES study: “freq”: “4”, “phase”: “0”
    freq_to_select = str(freq) 
    phase_to_select = str(phase) 
    
    
    for i in range(len(events_array)):
        event_code = events_array[i,2]
        event_id_key = list(event_id_dict.keys())[list(event_id_dict.values()).index(event_code)]
        
        if freq >= 0 and freq <= 40:
            if phase >= 0 and phase <=360:
                #if (freq_to_select in str(event_id_key[:(event_id_key.find('_') -2)])) == True and (phase_to_select in str(event_id_key[event_id_key.find('_') + 1:])) == True:
                if (freq_to_select == str(event_id_key[:(event_id_key.find('_') -2)]))  and (phase_to_select == str(event_id_key[event_id_key.find('_') + 1:])) :    
                    index_list.append(i)      
                else:
                    continue
            else:
                print("the specified phase is not within the range of 0 to 360 degrees")
        else:
            print("the specified freq is not within the range of 0 to 40 Hz")
    return index_list
    










def clustering_channels(win_erps, exdir_epoch, thresholds, labels, save_folder):
    
    
    files = Path(exdir_epoch).glob('*epo.fif*')
    plt.close('all')
    #idx =263
    
    unique_phases = np.arange(0, 360, 45 )
    unique_freqs = np.arange(4, 44, 4)    

    peaks = np.zeros([20, 64, len(labels), len(unique_phases), len(unique_freqs)])     
    mask_com = np.zeros([64, len(labels)-1])
    a = np.zeros([64, len(labels)])



    
    for ifiles, f in enumerate(files):
        epochs = mne.read_epochs(f, preload=True).copy().pick_types( eeg=True)
        # removing the effect of phase amp according to Granö et al. 2022.
        # amp after stim - amp before stim     
        epochs_amp_mod = epochs._data[:,:,1001:] - epochs._data[:,:,0:1000]
        # making mne epoch structure
        epochs = mne.EpochsArray(data = epochs_amp_mod,  info = epochs.info, events = epochs.events, event_id = epochs.event_id, on_missing='ignore')
     
        
        epochs_byfreqandphase = {} 
        erp_byfreqandphase = {} 
        peaks_byfreqandphase = {}
        peaks_byfreqandphase_std = {}
        evoked_zscored  = {}
        
        for ifreq, freq in enumerate(unique_freqs):
            epochs_byfreqandphase[str(freq)] = {}
            erp_byfreqandphase[str(freq)] = {} 
            peaks_byfreqandphase[str(freq)] = {} 
            peaks_byfreqandphase_std[str(freq)] = {}
            evoked_zscored[str(freq)] = {}
            for iphase, phase in enumerate(unique_phases):
                sel_idx = ph_analysis.Select_Epochs(epochs, freq, phase)
                epochs_byfreqandphase[str(freq)][str(phase)] = epochs[sel_idx]
                erp_byfreqandphase[str(freq)][str(phase)]  = epochs_byfreqandphase[str(freq)][str(phase)].average() 
                        
                for ipeak, peak in enumerate(labels):
                    #print(ipeak, peak)
                    
                    if ipeak == 0:    #P45
                        peaks_byfreqandphase[str(freq)][str(phase)] = np.mean((erp_byfreqandphase[str(freq)][str(phase)]._data[:,win_erps[0,0]:win_erps[0,1]]),1)
                    
                    elif  ipeak == 1: #N60
                        peaks_byfreqandphase[str(freq)][str(phase)] = np.mean((erp_byfreqandphase[str(freq)][str(phase)]._data[:,win_erps[1,0]:win_erps[1,1]]),1)
                    
                    elif  ipeak == 2: 
                        peaks_byfreqandphase[str(freq)][str(phase)] = np.mean((erp_byfreqandphase[str(freq)][str(phase)]._data[:,win_erps[2,0]:win_erps[2,1]]),1)

                    elif  ipeak == 3: 
                        peaks_byfreqandphase[str(freq)][str(phase)] = np.mean((erp_byfreqandphase[str(freq)][str(phase)]._data[:,win_erps[3,0]:win_erps[3,1]]),1)
                   

                    if str(erp_byfreqandphase[str(freq)][str(phase)].comment) == str(''):    # To remove none arrays after selecting epochs
                        peaks_byfreqandphase[str(freq)][str(phase)] = np.zeros(64) 
                       
                    else:
                        peaks[ifiles, :, ipeak, iphase, ifreq] = peaks_byfreqandphase[str(freq)][str(phase)] 
             
    
               
                
    unique_phases = np.arange(0, 360, 45 )
    unique_freqs = np.arange(4, 44, 4)
    adjacency_mat,_ = mne.channels.find_ch_adjacency(epochs_byfreqandphase[str(freq)][str(phase)].info , 'eeg')
    clusters, mask, pvals = ph_analysis.permutation_cluster(peaks, adjacency_mat, thresholds)         
    #mask_com[:,[0,1,3]] = mask[:, [0, 1,4]] #combine labels 2 and 3, they are the same component. just different cluster 
    #mask_com[:,2] = np.logical_or(mask[:,2], mask[:,3])
    nsubj, nchans, npeaks, nphas, nfreqs = np.shape(peaks)    
    allclusters = np.zeros([nchans, npeaks])
    # get the t values for each of the peaks for plotting the topoplots
    for p in range(len(clusters)):
        allclusters[:,p] = clusters[p][0]
    # set all other t values to 0 to focus on clusters
    allclusters[mask==False] = 0
    ch_names = epochs.ch_names
    # this is putting the 5-dim data structure in the right format for performing the sine fits
    
    for p in range(len(clusters)):
        a[:,p] = clusters[p][0]
        
    #combine labels 2 and 3, they are the same component. just different cluster    
    #a_com = a[:,[0, 1, 2, 4]]
    fig = ph_analysis.plot_topomap_peaks_second_v(a, mask, ch_names, pvals,[-6,6], epochs.info)
    
    
    
    # Name and indices of the EEG electrodes that are in the biggest cluster
    all_ch_names_biggets_cluster =  []
    all_ch_ind_biggets_cluster =  []
    
    for p in range(len(clusters)):
        # indices
        all_ch_ind_biggets_cluster.append(np.where(mask[:,p] == 1))
        # channel names
        all_ch_names_biggets_cluster.append([ch_names[i] for i in np.where(mask[:,p] == 1)[0]])
        

    fig.savefig(save_folder + 'clusters_all_conditions_vs_0'+ '.svg')    

    return all_ch_names_biggets_cluster, pvals







def permutation_cluster(peaks, adjacency_mat, thresholds):
    
    # in this function, peaks is a 5 dim matrix with dims, nsubj, nchans, npeaks, nphas, nfreq
    import mne.stats
    # reduce dimensions by averaging over target frequencies and phases
    mean_peaks = np.mean(peaks, (-2, -1))
    # get matrix dimensions
    nsubj, nchans, npeaks = np.shape(mean_peaks)
    mask = np.zeros([nchans, npeaks])
    pvals = np.zeros([npeaks])
    clusters = []
    # get the original cluster size during the first loop
    # perform 1000 random permutations (sign flipping) and each time determine the size of the biggest cluster

    for p in range(npeaks):
        cluster = mne.stats.permutation_cluster_1samp_test(mean_peaks[:,:,p], out_type='mask',
                                                           adjacency=adjacency_mat, threshold=thresholds[p],
                                                           n_permutations=1000)
        t_sum = np.zeros([len(cluster[1])])
        # get the sum of the tvalues for each of the 
        # clusters to choose the main cluster 
        # (take magnitude to treat negative and positive cluster equally)
        for c in range(len(cluster[1])):
            t_sum[c] = np.abs(sum(cluster[0][cluster[1][c]]))
    
        # store the maximal cluster size for each iteration 
        # to later calculate p value
        # if no cluster was found, put in 0
        
        
        
        if len(t_sum) > 0:
            # components 2 and 3 have almost the same t value, so we are considering both
# =============================================================================
#             if  p==2:
#                 mask[:,p] = cluster[1][0]
#                 pvals[p] = cluster[2][0]
#             elif p==3:
#                 mask[:,p] = cluster[1][1]
#                 pvals[p] = cluster[2][1]
#                 
#                 
#             else:
#                     
# =============================================================================
                mask[:,p] = cluster[1][np.argmax(t_sum)]
                pvals[p] = min(cluster[2])
            
 
                
        clusters.append(cluster)         
        

    

    return clusters, mask, pvals





def plot_topomap_peaks_second_v(peaks, mask, ch_names, pvals, clim, pos):

    import matplotlib.pyplot as plt

    nplots =1 
    nchans, npeaks = np.shape(peaks)

    maskparam = dict(marker='.', markerfacecolor='k', markeredgecolor='k',
                linewidth=0, markersize=5)

    fig, sps = plt.subplots(nrows=nplots, ncols=npeaks, figsize=(10,6))
    plt.style.use('default')
    
    for iplot in range(nplots):
        for ipeak in range(npeaks):

            # if mask is None:
            #     psig = None
            # else:
            #     psig = np.where(mask[iplot, :, ipeak] < 0.01, True, False)

            # sps[ipeak, iplot].set_aspect('equal')

            if mask is not None:
                imask=mask[:,ipeak]
            else:
                imask = None

            im = topoplot_2d(ch_names, peaks[ :, ipeak], pos,
                                clim=clim, axes=sps[ipeak], 
                                mask=imask, maskparam=maskparam)

    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    cb = plt.colorbar(im[0],  ax = sps, fraction=0.01, pad=0.04)
    cb.ax.tick_params(labelsize=12)
    
    fig.suptitle('All Frequencies and Peak and trough Phases Vs zero', fontsize = 14)
    #fig.suptitle('All Frequencies and all phases', fontsize = 14)
# =============================================================================
#     sps[0].title.set_text(f' \n\n ERP 1\n\n TH = {thresholds[0]} \n\n  cluster_pv = {pvals[0]}')
#     sps[1].title.set_text(f' \n\n ERP 2\n\n TH = {thresholds[1]} \n\n  cluster_pv = {pvals[1]}')
# =============================================================================
    sps[0].set_title('\n\n P30' , fontsize=14, fontweight ='bold')
    sps[1].set_title('\n\n N60' , fontsize=14, fontweight ='bold')
    sps[2].set_title('\n\n N120', fontsize=14, fontweight ='bold')
    sps[3].set_title('\n\n P190', fontsize=14, fontweight ='bold')
    fig.text(0.15, 0.2, f' P = {pvals[0]} ',  ha='left')
    fig.text(0.35, 0.2, f' P = {pvals[1]} ',  ha='left')
    fig.text(0.55, 0.2, f' P = {pvals[2]} ',  ha='left')
    fig.text(0.75, 0.2, f' P = {pvals[3]} ',  ha='left')
    
    #fig.text(0, 0. ,f' \n\n  TH = {thresholds[0]} \n\n  cluster_pv = {pvals_all[str(0)]}\n\n {all_ch_names_biggets_cluster[str(0)][str(0)]}\n\n {all_ch_names_biggets_cluster[str(0)][str(1)]}\n\n  ',  ha='left')
    #fig.text(0.5, 0 ,f' \n\n  TH = {thresholds[1]} \n\n  cluster_pv = {pvals_all[str(1)]}\n\n {all_ch_names_biggets_cluster[str(1)][str(0)]}\n\n {all_ch_names_biggets_cluster[str(1)][str(1)]}\n\n  ',  ha='left')
    cb.set_label('t-value', rotation = 90)

    

    plt.show()

    return fig





def topoplot_2d (ch_names, ch_attribute, pos, clim=None, axes=None, mask=None, maskparam=None):
    
    """
    Function to plot the EEG channels in a 2d topographical plot by color coding 
    a certain attribute of the channels (such as PSD, channel specific r-squared).
    Draws headplot and color fields.
    Parameters
    ----------
    ch_names : String of channel names to plot.
    ch_attribute : vector of values to be color coded, with the same length as the channel, numerical.
    clim : 2-element sequence with minimal and maximal value of the color range.
           The default is None.
           
    Returns
    -------
    None.
    This function is a modified version of viz.py (mkeute, github)
    """    

    import mne
    # get standard layout with over 300 channels
    #layout = mne.channels.read_layout('EEG1005')
    
    # select the channel positions with the specified channel names
    # channel positions need to be transposed so that they fit into the headplot
# =============================================================================
#     pos = (np.asanyarray([layout.pos[layout.names.index(ch)] for ch in ch_names])
#            [:, 0:2] - 0.5) / 5
#     
# =============================================================================
    if maskparam == None:
        maskparam = dict(marker='o', markerfacecolor='k', markeredgecolor='k',
                    linewidth=0, markersize=3) #default in mne
    if clim == None:
        im = mne.viz.plot_topomap(ch_attribute, 
                                  pos, 
                                  ch_type='eeg',
                                  sensors=True,
                                  contours=6,
                                  cmap = 'RdBu_r',
                                  axes=axes,
                                  outlines = "head", 
                                  mask=mask,
                                  mask_params=maskparam,
                                  vlim = (clim[0], clim[1]),
                                  sphere=(0.00, 0.00, 0.00, 0.11),
                                  extrapolate = 'head')
    else:
        im = mne.viz.plot_topomap(ch_attribute, 
                                  pos, 
                                  ch_type='eeg',
                                  sensors=True,
                                  contours=6,
                                  cmap = 'RdBu_r',
                                  axes=axes,
                                  outlines = "head", 
                                  mask=mask,
                                  mask_params=maskparam,
                                  vlim = (clim[0], clim[1]),
                                  sphere=(0.00, 0.00, 0.00, 0.11),
                                  extrapolate = 'head')
    return im










def do_cosine_fit_phase_freq_extracted(erp_amplitude, erp_amplitude_sem, phase_bin_means, freq_band, labels, perm = True):
    '''Args:
           erp_amplitude:
               dict variable that contains amplitude of first and second peak
           freq_band:
               we fit the cosine to freq_band = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40] (freq_step_i == '4Hz step')
               to be comparable with Lucky loop labels.
           labels: 
               time points of first and second erp peaks estimated by GFP     
               
           
        '''
        
        
        
    
    cosinefit = {}
    amplitudes_cosine = np.zeros([len(freq_band), len(labels)*2 ])
    pvalues_cosine = np.zeros([len(freq_band), len(labels)*2])



    x = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    #x = phase_bin_means
    
    for i in range(len(erp_amplitude)):
        fig, sps = plt.subplots(nrows=2, ncols=int(len(freq_band)/2))
        if i == 0:
            fig.suptitle('ERP 1')
        else:
            fig.suptitle('ERP 2')
        
        
        cosinefit[str(i)] = {}
        for jf, f in enumerate(freq_band):    
            print('cosine fits for frequency {}'.format(f))
            y = zscore(np.array(list(erp_amplitude[str(i)][f].values())))  
            y_sem = np.array(list(erp_amplitude_sem[str(i)][f].values()))
            cosinefit[str(i)][str(f)] = []
            fits = []
            for phase_start in [-np.pi/2, 0, np.pi/2]:      
        
                amp_start = np.sqrt(np.mean(y**2))
                model = lmfit.Model(cosinus)
                params = model.make_params()
        
                params["amp"].set(value=amp_start, min=0, max=np.ptp(y)/2)
                params["phi"].set(value=phase_start, min=-np.pi, max=np.pi)
                data = {ph: np.mean(y[x == ph]) for ph in np.unique(x)}
                fits.append(model.fit(y, params, x=x))
                
            result = fits[np.argmin([f.aic for f in fits])]
            
            if perm:
                model = lmfit.Model(cosinus)
                params = result.params
                dataperm = []
            
                # use all possible combinations of the 8 phase bins to determine p.
                # Take out the first combination because it is the original
                all_perms = list(itertools.permutations(x))
                del all_perms[0]
            
                for iper in tqdm(range(len(all_perms))):
                    x_shuffled = all_perms[iper]
                    dataperm.append([model,params, y, x_shuffled])
            
                with Pool(4) as p:
                    surrogate = p.starmap(do_one_perm, dataperm)
            else: 
                surrogate = [np.nan]
                
            
            nullmodel = lmfit.Model(unif)
            params = nullmodel.make_params()
            params["offset"].set(value=np.mean(y), min=min(y), max=max(y))
            nullfit = nullmodel.fit(y, params, x=x)
            surrogate = np.array(surrogate)
            surrogate = surrogate[np.invert(np.isnan(surrogate))]
            
            cosinefit[str(i)][str(f)].append( { 'Model': 'cosinus', 
                                    'Frequency': f, 
                                    'Fit': result,
                                    'data': data, 
                                    'amp': result.best_values['amp'], 
                                    'surrogate': surrogate, 
                                    'p':[np.nan if perm == False else sum(np.abs(surrogate) >= np.abs(result.best_values['amp']))/len(surrogate)][0], 
                                    'std':[np.nan if perm == False else np.std(surrogate)][0], 
                                    'nullmodel':nullfit,
                                    })
            
            amplitudes_cosine[jf, i] = result.best_values['amp']
            pvalues_cosine[jf, i] = [np.nan if perm == False else sum(np.abs(surrogate) >= np.abs(result.best_values['amp']))/len(surrogate)][0] 
    
            if jf == 0:
                row = 0; col=0
            elif jf == 1:
                row = 0; col=1
            elif jf == 2:
                row = 0; col=2   
            elif jf == 3:
                row = 0; col=3
            elif jf == 4:
                row = 0; col=4 
            elif jf == 5:
                row = 1; col=0
            elif jf == 6:
                row = 1; col=1   
            elif jf == 7:
                row = 1; col=2
            elif jf == 8:
                row = 1; col=3 
            elif jf == 9:
                row = 1; col=4            
                
                 
            
            sps[row, col].errorbar(x, y, fmt='.k')
            sps[row, col].errorbar(x, y, yerr = y_sem, fmt='.k')
            sps[row, col].set_title(f'{int(f)} Hz')
            cosin = cosinus(x, result.values['amp'], result.values['phi'])
            #sps[row, col].plot(unique_phases, cosin, 'r')
            xnew = np.linspace(x[0], x[-1], num=41, endpoint=True)
            f2 = interp1d(x, cosin, kind='cubic')
            sps[row, col].plot(xnew, f2(xnew), 'r', label = 'Fitted Cosine')
            sps[row, col].plot([0,np.max(x)], [0,0], 'k', label = 'Data')
            sps[row, col].set_ylim([-5,5])
            sps[row, col].set_xlim([-5,320])
            sps[row, col].grid()
            sps[row, col].legend()
        
    
            if pvalues_cosine[jf, i]<0.05:
                sps[row, col].text(0.5,0.5, str(round(float(pvalues_cosine[jf, i]),4)), c='r')
            else:
                sps[row, col].text(0.5,0.5, str(round(float(pvalues_cosine[jf, i]),4)), c='k')
    
            plt.show()

    
    return cosinefit, amplitudes_cosine, pvalues_cosine









def epoch_concat_and_mod_dict_bip(files_GA):
    
        
    dict_origin_labels  = {'12Hz_0': 1, '12Hz_135': 2, '12Hz_180': 3, '12Hz_225': 4, '12Hz_270': 5,
                           '12Hz_315': 6, '12Hz_45': 7, '12Hz_90': 8, '16Hz_0': 9, '16Hz_135': 10, 
                           '16Hz_180': 11, '16Hz_225': 12, '16Hz_270': 13, '16Hz_315': 14, '16Hz_45': 15, 
                           '16Hz_90': 16, '20Hz_0': 17, '20Hz_135': 18, '20Hz_180': 19, '20Hz_225': 20, 
                           '20Hz_270': 21, '20Hz_315': 22, '20Hz_45': 23, '20Hz_90': 24, '24Hz_0': 25,
                           '24Hz_135': 26, '24Hz_180': 27, '24Hz_225': 28, '24Hz_270': 29, '24Hz_315': 30, 
                           '24Hz_45': 31, '24Hz_90': 32, '28Hz_0': 33, '28Hz_135': 34, '28Hz_180': 35, 
                           '28Hz_225': 36, '28Hz_270': 37, '28Hz_315': 38, '28Hz_45': 39, '28Hz_90': 40, 
                           '32Hz_0': 41, '32Hz_135': 42, '32Hz_180': 43, '32Hz_225': 44, '32Hz_270': 45, 
                           '32Hz_315': 46, '32Hz_45': 47, '32Hz_90': 48, '36Hz_0': 49, '36Hz_135': 50, 
                           '36Hz_180': 51, '36Hz_225': 52, '36Hz_270': 53, '36Hz_315': 54, '36Hz_45': 55,
                           '36Hz_90': 56, '40Hz_0': 57, '40Hz_135': 58, '40Hz_180': 59, '40Hz_225': 60,
                           '40Hz_270': 61, '40Hz_315': 62, '40Hz_45': 63, '40Hz_90': 64, '4Hz_0': 65,
                           '4Hz_135': 66, '4Hz_180': 67, '4Hz_225': 68, '4Hz_270': 69, '4Hz_315': 70, 
                           '4Hz_45': 71, '4Hz_90': 72, '8Hz_0': 73, '8Hz_135': 74, '8Hz_180': 75,
                           '8Hz_225': 76, '8Hz_270': 77, '8Hz_315': 78, '8Hz_45': 79, '8Hz_90': 80}
    
    
    
    
    
    
    
       
    mod = {}
    all_epochs_list = []
    all_epochs_events = []
    all_names = []
    
    for f_GA in files_GA:
        epochs_eeg = mne.read_epochs(f_GA, preload=True)
        # So basically the problem was mne creats a dict of all stimulation conditions in our case 80. For some epochs data with a small
        # size all these 80 conditions are not present. It can be 76 so the dict will start from zero to 76 and event_id keys and value will be 
        # different for each condition in different subjects and there will be a problem during concatinating.
        # I created a diffault dict, based on 80 condition and forced it to be the same for other epoch files even for the one with less
        # than 80 conditions.
        if len(epochs_eeg.event_id) < 80:
            mod_vals = np.zeros([len(epochs_eeg.events[:, 2]), 2])
            #shared_keys = set(epochs_eeg.event_id.keys()).intersection(set(dict_origin_labels))
            for i in epochs_eeg.event_id.keys():
                #print(i)
                mod[i] = [i, epochs_eeg.event_id[str(i)], dict_origin_labels[str(i)]]
            mod_arr = np.array(list(mod.values())) 
            
            for i in range(len(epochs_eeg.events[:, 2])):
                for j in range(len(mod_arr)):
                    if epochs_eeg.events[:, 2][i] == int(mod_arr[j,1]):
                        mod_vals[i] = [mod_arr[j,1],mod_arr[j,2] ]
            epochs_eeg.events[:, 2] = mod_vals[:,1]
            epochs_eeg.event_id = dict_origin_labels
    
        # channels based on clustered channels. Only using those because the size of this variable will be very large. 
        all_epochs_list.append(epochs_eeg)
        all_epochs_events.append(epochs_eeg.event_id)
        all_names.append(f_GA.parts[-1][0:9])
    
    all_epochs_concat = mne.concatenate_epochs(all_epochs_list)
    #all_epochs_concat.save(save_folder + '/epochs all subjects/' +'epochs_all_sub_epo.fif'  , overwrite = True, split_size='2GB')
    #all_epochs_concat.save(save_folder + '/epochs all subjects/' +'bip epochs_all_sub_epo.fif'  , overwrite = True, split_size='2GB')
    return all_epochs_concat











# =============================================================================
# 
# 
# 
# 
# def epoch_concat_and_mod_dict_bip(files_GA):
#     
#     save_folder = "/home/sara/NMES/analyzed data/phase_analysis/"
#     
#     
#     
#     
#     
#     
#     
#     
#        
#     mod = {}
#     all_epochs_list = []
#     all_epochs_events = []
#     all_names = []
#     
#     for f_GA in files_GA:
#         epochs_eeg = mne.read_epochs(f_GA, preload=True)
# 
#     
#         # channels based on clustered channels. Only using those because the size of this variable will be very large. 
#         all_epochs_list.append(epochs_eeg)
#         all_epochs_events.append(epochs_eeg.event_id)
#         all_names.append(f_GA.parts[-1][0:9])
#     
#     all_epochs_concat = mne.concatenate_epochs(all_epochs_list)
#     #all_epochs_concat.save(save_folder + '/epochs all subjects/' +'epochs_all_sub_epo.fif'  , overwrite = True, split_size='2GB')
#     #all_epochs_concat.save(save_folder + '/epochs all subjects/' +'bip epochs_all_sub_epo.fif'  , overwrite = True, split_size='2GB')
#     return all_epochs_concat
# 
# 
# 
# 
# =============================================================================































def epoch_concat_subs_mutltiple_files(files_GA):
    

    dict_origin_labels  = {'12Hz_0': 1, '12Hz_135': 2, '12Hz_180': 3, '12Hz_225': 4, '12Hz_270': 5,
                           '12Hz_315': 6, '12Hz_45': 7, '12Hz_90': 8, '16Hz_0': 9, '16Hz_135': 10, 
                           '16Hz_180': 11, '16Hz_225': 12, '16Hz_270': 13, '16Hz_315': 14, '16Hz_45': 15, 
                           '16Hz_90': 16, '20Hz_0': 17, '20Hz_135': 18, '20Hz_180': 19, '20Hz_225': 20, 
                           '20Hz_270': 21, '20Hz_315': 22, '20Hz_45': 23, '20Hz_90': 24, '24Hz_0': 25,
                           '24Hz_135': 26, '24Hz_180': 27, '24Hz_225': 28, '24Hz_270': 29, '24Hz_315': 30, 
                           '24Hz_45': 31, '24Hz_90': 32, '28Hz_0': 33, '28Hz_135': 34, '28Hz_180': 35, 
                           '28Hz_225': 36, '28Hz_270': 37, '28Hz_315': 38, '28Hz_45': 39, '28Hz_90': 40, 
                           '32Hz_0': 41, '32Hz_135': 42, '32Hz_180': 43, '32Hz_225': 44, '32Hz_270': 45, 
                           '32Hz_315': 46, '32Hz_45': 47, '32Hz_90': 48, '36Hz_0': 49, '36Hz_135': 50, 
                           '36Hz_180': 51, '36Hz_225': 52, '36Hz_270': 53, '36Hz_315': 54, '36Hz_45': 55,
                           '36Hz_90': 56, '40Hz_0': 57, '40Hz_135': 58, '40Hz_180': 59, '40Hz_225': 60,
                           '40Hz_270': 61, '40Hz_315': 62, '40Hz_45': 63, '40Hz_90': 64, '4Hz_0': 65,
                           '4Hz_135': 66, '4Hz_180': 67, '4Hz_225': 68, '4Hz_270': 69, '4Hz_315': 70, 
                           '4Hz_45': 71, '4Hz_90': 72, '8Hz_0': 73, '8Hz_135': 74, '8Hz_180': 75,
                           '8Hz_225': 76, '8Hz_270': 77, '8Hz_315': 78, '8Hz_45': 79, '8Hz_90': 80}
    
    
    
    # These lines go to the permutation cluster function and select the channels that will be appended 
    # in the epoch list.
    

    mod = {}
    all_epochs_list = []
    all_epochs_events = []
    all_names = []
    
    for f_GA in files_GA:
        epochs_eeg = mne.read_epochs(f_GA, preload=True)

        # So basically the problem was mne creats a dict of all stimulation conditions in our case 80. For some epochs data with a small
        # size all these 80 conditions are not present. It can be 76 so the dict will start from zero to 76 and event_id keys and value will be 
        # different for each condition in different subjects and there will be a problem during concatinating.
        # I created a diffault dict, based on 80 condition and forced it to be the same for other epoch files even for the one with less
        # than 80 conditions.
        if len(epochs_eeg.event_id) < 80:

            mod_vals = np.zeros([len(epochs_eeg.events[:, 2]), 2])
            #shared_keys = set(epochs_eeg.event_id.keys()).intersection(set(dict_origin_labels))
            for i in epochs_eeg.event_id.keys():
                #print(i)
                mod[i] = [i, epochs_eeg.event_id[str(i)], dict_origin_labels[str(i)]]
            mod_arr = np.array(list(mod.values())) 
            
            for i in range(len(epochs_eeg.events[:, 2])):
                for j in range(len(mod_arr)):
                    if epochs_eeg.events[:, 2][i] == int(mod_arr[j,1]):
                        mod_vals[i] = [mod_arr[j,1],mod_arr[j,2] ]
            epochs_eeg.events[:, 2] = mod_vals[:,1]
            epochs_eeg.event_id = dict_origin_labels
    
        # channels based on clustered channels. Only using those because the size of this variable will be very large. 
        all_epochs_list.append(epochs_eeg)
        all_epochs_events.append(epochs_eeg.event_id)
        all_names.append(f_GA.parts[-1][0:9])
    
    all_epochs_concat = mne.concatenate_epochs(all_epochs_list)
    return all_epochs_concat, f_GA.parts














def phase_optimal_distribution(ax, mag_df_array, title):
    
    bins_number = 8 
    bins = np.linspace(0.0, 2 * np.pi, bins_number + 1)
    bin_phases= np.array([0, 45, 90, 135, 180, 225, 270, 315])

    width = 2 * np.pi / bins_number
    ax.bar(bins[:bins_number], abs(mag_df_array), zorder=1, align='edge', width=width,  edgecolor='C0', fill=False, linewidth=1)
    mag_all = []
    for j, j_mag in enumerate(mag_df_array):
        #print(bin_phases[j], abs(j_mag))
        mag_all.append( P2R(abs(j_mag), bin_phases[j]))
    
    r, theta = R2P(np.mean(mag_all))
    ax.plot([0, np.degrees((theta))], [0, r],  lw=3, color = 'red')    
    #ax.set_ylim([0,0.6])
    ax.set_title(title,fontweight="bold")  

def P2R(radii, angles):
    return radii * np.exp(1j*angles)

def R2P(x):
    return abs(x), np.angle(x)


#%%
def Circ_corr(cosinefit_ll, labels):
    fig, ax =  plt.subplots(1,len(labels))
    
    phi_dict = {}
    freq_band = np.arange(4,44,4)
    phi_array_deg = np.zeros([len(freq_band), len(labels)])
    amp_array_all = np.zeros([len(freq_band), len(labels)])
    for i in range(len(labels)):
        amp = {}
        phi = {}
        for jf, freq in enumerate(freq_band):   
                
            amp[str(freq)] = cosinefit_ll[str(i)][str(freq)][0]['amp']  
            phi[str(freq)] = cosinefit_ll[str(i)][str(freq)][0]['Fit'].best_values['phi']
        
    
        
                
        amp_df = pd.DataFrame({'4Hz': amp[str(4)], '8Hz': amp[str(8)], '12Hz': amp[str(12)], '16Hz': amp[str(16)], \
                                    '20Hz': amp[str(20)], '24Hz': amp[str(24)], '28Hz': amp[str(28)], '32Hz': amp[str(32)], \
                                    '36Hz': amp[str(36)],'40Hz': amp[str(40)]}, index=['amp'])   
        amp_df_array = amp_df.to_numpy()
    
        phi_df = pd.DataFrame({'4Hz': phi[str(4)], '8Hz': phi[str(8)], '12Hz': phi[str(12)], '16Hz': phi[str(16)], \
                                    '20Hz': phi[str(20)], '24Hz': phi[str(24)], '28Hz': phi[str(28)], '32Hz': phi[str(32)], \
                                    '36Hz': phi[str(36)], '40Hz': phi[str(40)]}, index=['phi']) 
        phi_array = phi_df.to_numpy()
    
        phi_dict[str(i)] = phi_array
        amp_array_all[:,i] =  amp_df_array
        phi_array_deg[:,i] =  np.degrees(phi_array)
        for j, j2 in enumerate(freq_band):

    
            phi_array_deg[j,i] =  phi_array_deg[j,i] 
            
         
         
    
        
    
        cor, ci = pycircstat.corrcc(np.array(freq_band), phi_array_deg[:,i], ci=True)
        cor = np.abs(cor)
        rval = str(np.round(cor,4))
        tval = (cor*(np.sqrt(len(np.array(freq_band)-2)))/(np.sqrt(1-cor**2)))
        pval = str(np.round(1-stats.t.cdf(np.abs(tval),len(np.array(freq_band))-1),3))
        # plot scatter
     
    
        im = ax[i].scatter(phi_array_deg[:,i], freq_band, c= amp_df_array)    
        if i==0:
            erp_num = 'First'
        else:
            erp_num = 'Second'
        ax[i].title.set_text(f'{erp_num} ERP, r = {rval}, p = {pval}' )
        clb = fig.colorbar(im, ax=ax[i])    
        clb.ax.set_title('Strength of MD')
        fig.suptitle('Group Average, Real-Time')
        ax[i].set_xlim([0, 400])
        ax[i].set_xlabel('Optimal phases (deg)')
        ax[i].set_ylabel('Frequency (Hz)')
        ax[i].set_xlim(left=-10)
        plt.show()
    return(phi_array_deg)

















def amp_p_chi(cosinefit_ll, labels, phi_array_deg, save_folder):
    freq_band = np.arange(4,44,4)
    mod_depth = {}
    p_val = {}
    red_chi = {}
    for i in range(len(labels)):
        mod_depth[str(i)] = {} 
        p_val[str(i)] = {}
        red_chi[str(i)] = {}
        for jf, freq in enumerate(freq_band):   
           mod_depth[str(i)][str(freq)] = cosinefit_ll[str(i)][str(freq)][0]['amp']
           p_val[str(i)][str(freq)] = cosinefit_ll[str(i)][str(freq)][0]['p']
           red_chi[str(i)][str(freq)]  = cosinefit_ll[str(i)][str(freq)][0]['Fit'].redchi
           
    p_val_df = pd.DataFrame(p_val)   
    amp_df = pd.DataFrame(mod_depth)
    red_chi_df = pd.DataFrame(red_chi)
    amp_df_rename = amp_df.rename(columns={'0': 'P30', '1': 'N60', '2': 'N120', '3': 'P190'})
    p_val_df_rename = p_val_df.rename(columns={'0': 'P30', '1': 'N60', '2': 'N120', '3': 'P190'})
    red_chi_df_rename  = red_chi_df.rename(columns={'0': 'P30', '1': 'N60', '2': 'N120', '3': 'P190'})
        
        
    
    for i in np.arange(len(amp_df)):
        amp_df_rename = amp_df_rename.rename(index = {f'{amp_df.index[i]}' : f'{amp_df.index[i]} hz'})
    amp_df_r = amp_df_rename.T
    
    
    plt.style.use('default')
    
    # Plot modulation depth in bar format
    fig, ax = plt.subplots(1, 1)
    amp_df_r.plot(kind="bar", alpha=0.75, rot=0,  colormap = 'viridis_r', title = 'Strength of Modulation Depth,  Group Average', ax= ax).legend(loc = 'lower right')
    ax.set_ylim(bottom=-.010, top=0.5)
    
    
    
    
    # plot p values in a table format
    fig, ax = plt.subplots(figsize=(5,7))
    df = p_val_df_rename
    df.index.name = "Freq"
    table = ax.table(cellText=np.round(df.values, 3), rowLabels=df.index + 'Hz', cellLoc='center',
                     colColours=['gainsboro'] * len(df), colLabels=df.columns, loc='center',
                     colWidths= [0.15]*(len(df.columns)))
    
    w, h = table[0,1].get_width(), table[0,1].get_height()
    table.add_cell(0, -1, w,h, text=df.index.name)
    ax.grid(False)
    table.scale(1,1.2)
    plt.title('p values')
    plt.show()
    fig.savefig(save_folder +'P_value.svg')
    
    
    
    # plot reduced chi squared in atable format
    fig, ax = plt.subplots(figsize=(5,7))
    df = red_chi_df_rename
    df.index.name = "Freq"
    table = ax.table(cellText=np.round(df.values, 3), rowLabels=df.index + 'Hz', cellLoc='center',
                     colColours=['gainsboro'] * len(df), colLabels=df.columns, loc='center',
                     colWidths= [0.15]*(len(df.columns)))
    
    w, h = table[0,1].get_width(), table[0,1].get_height()
    table.add_cell(0, -1, w,h, text=df.index.name)
    ax.grid(False)
    table.scale(1,1.2)
    plt.title('Reduced chi-squared')
    plt.show()
    


    
    # Plot optimal phases
    # Here we need phase lag, not phase lead. So if a value is positive we need to diffrentiate is from 360. 
    
    phi_array_deg_correct = np.zeros([len(freq_band), len(labels)])

    
    for i_erp in np.arange((phi_array_deg.shape[1])):
        for i in np.arange(len(phi_array_deg)):

            # It's about phase lead and lag compared to the cosine with no phase shift
            # the phi that I get from fitted cosine, actually shows me how many degrees it leads or lags a cosine with no phase shift
            # look at "best_fit_plot" to understand it better
            if phi_array_deg[i][i_erp] < 0:
               #phi_array_deg_correct[i][i_erp] = 360 + phi_array_deg[i][i_erp]
               phi_array_deg_correct[i][i_erp] = abs(phi_array_deg[i][i_erp])
            else:

                 phi_array_deg_correct[i][i_erp] =  360 - phi_array_deg[i][i_erp]
                
    
    # It's about phase lead and lag compared to the cosine with no phase shift
    df = pd.DataFrame(phi_array_deg_correct)
    df = df.rename(columns={0: 'P30', 1: 'N60', 2: 'N120', 3: 'P190'})
    
    for i in np.arange(len(phi_array_deg_correct)):
        df = df.rename(index = {df.index[i] : f'{((df.index[i] + 1)*4)} '})
    fig, ax = plt.subplots(figsize=(7,10))    
    df.index.name = "Freq"
    table = ax.table(cellText=np.round(df.values, 3), rowLabels=df.index + 'Hz', cellLoc='center',
                     colColours=['gainsboro'] * len(df), colLabels=df.columns, loc='center',
                     colWidths= [0.15]*(len(df.columns)))
    
    w, h = table[0,1].get_width(), table[0,1].get_height()
    table.add_cell(0, -1, w,h, text=df.index.name)
    ax.grid(False)
    table.scale(1,1.3)
    plt.title('Optimal Phases')
    plt.show()
    fig.savefig(save_folder +'Optimal_phases.svg')

    return(pd.DataFrame.to_numpy(p_val_df_rename), phi_array_deg_correct)






def best_fit_plot(cosinefit_ll, labels, phi_array_deg_correct, save_folder, title, y_limit):
    
    X= np.array([0, 45, 90, 135, 180, 225, 270, 315])
    freq_band = np.arange(4,44,4)
    # Plotting data and interpolated fitted cosine 
    Y = {}
    Y_shifted ={}
    best_fit = {}
    best_fit_shifted = {}
    for i in range(len(labels)):
        Y[str(i)] = {}
        Y_shifted[str(i)] ={}
        best_fit[str(i)] ={}
        best_fit_shifted[str(i)] = {}
        for jf, freq in enumerate(freq_band):   
            Y[str(i)][str(jf)] = (np.array(list(cosinefit_ll[str(i)][str(freq)][0]['data'].values())))
            #Y[str(i)][str(jf)] = Y[str(i)][str(jf)] - np.mean(Y[str(i)][str(jf)])        
            best_fit[str(i)][str(jf)] = cosinefit_ll[str(i)][str(freq)][0]['Fit'].best_fit
            
            #find the maximum value (corresponsing to optimal phase), shift it as optimal phase value
            max_best_fit = np.argmax(best_fit[str(i)][str(jf)])
            max_Y =  np.argmax(Y[str(i)][str(jf)])
            Y_shifted[str(i)][str(jf)] = np.roll(Y[str(i)][str(jf)], max_Y)
            best_fit_shifted[str(i)][str(jf)] =  np.roll(best_fit[str(i)][str(jf)], max_best_fit)
    
    plt.style.use('default')  
    # Plotting data and interpolated fitted cosine 
    xi = range(len(X))
    x_new = np.linspace(0, 7, num=40)
    cols = [format(col) for col in np.arange(0,10,1)]
    

    for i_labels, name_labels in enumerate(labels):
         
        
    
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20,8))
        fig.suptitle(f'{name_labels}', fontsize=16, fontweight ='bold')
        
        for ax, col in zip(axes[0], cols):
            ax.set_title(f'{(int(col)+1)*4}Hz', fontsize=14, fontweight ='bold')
            ax.scatter(xi, Y[str(i_labels)][col], color='k')
            ax.set_ylim(y_limit)
            ax.set_xlim(-1, 8)
            f2 = interpolate.interp1d(xi, best_fit[str(i_labels)][col], kind='cubic')
            ax.plot(x_new,  f2(x_new), 'r')
            ax.set_xticks(xi, [round(i) for i in np.arange(0,360, 45)], fontsize=8)
        
        
        
        for ax, col in zip(axes[1], cols):
            ax.set_title(f'{(int(col)+6)*4}Hz', fontsize=14, fontweight ='bold')
            col = f'{int(col) +5}'
            ax.scatter(xi, Y[str(i_labels)][col], color='k')
            ax.set_ylim(y_limit)
            ax.set_xlim(-1, 8)
            f2 = interpolate.interp1d(xi, best_fit[str(i_labels)][col], kind='cubic')
            ax.plot(x_new,  f2(x_new), 'r')
            ax.set_xticks(xi, [round(i) for i in np.arange(0,360, 45)], fontsize=8)
        
        plt.show()
        fig.savefig(save_folder + f'{title}_' + f'{name_labels}'+'_sine_fit' +'.svg')

    return 




def fig_2c_scatter_plot(cosinefit_ll, labels, p_val,  save_folder):
    plt.style.use('default')
    freq_band = np.arange(4,44,4)
    xi = range(len(np.array(freq_band)))
    # Fig 2.c
    for i_labels, name_labels in enumerate(labels):
         mod_depth = {}
         surr = {}
         for jf, freq in enumerate(freq_band): 
             
            mod_depth[str(freq)] = cosinefit_ll[str(i_labels)][str(freq)][0]['amp']
            surr[str(freq)] = np.mean(cosinefit_ll[str(i_labels)][str(freq)][0]['surrogate'])
    
    
         fig = plt.figure(figsize=(7, 6))
         if p_val[jf, i_labels] <= 0.04 :
             plt.plot(3,  0.5, '*', color = 'k')
             plt.text(2,  0.55, f'P-value = {np.round(p_val[jf, i_labels], 3)}' , weight='bold')
         plt.scatter(xi, (np.array(list(mod_depth.values()))),  c = 'k', label='Real data')
         plt.plot(xi, (np.array(list(mod_depth.values()))),  c = 'k', alpha=0.1)

         plt.scatter(xi, ((np.array(list(surr.values())))),  c = 'r', label='Surrogate')
         plt.plot(xi, ((np.array(list(surr.values())))),  c = 'r', alpha= 0.1)
         plt.xticks(xi, np.array(freq_band))
         plt.xlabel("Frequecies (Hz)", weight='bold')
         plt.ylabel("Modulation depth", weight='bold')
         plt.legend(loc='upper right')
         plt.ylim(0, 0.6)
         plt.title(f'{name_labels} Group Cosine Models', weight='bold')
         plt.show()
         fig.savefig(save_folder+f'{name_labels}.svg')
         
    return


def fig_2c_scatter_sub_plot(cosinefit_ll, labels, p_val, save_folder):
    plt.style.use('default')
    freq_band = np.arange(4,44,4)
    xi = range(len(np.array(freq_band)))
    fig = fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    # Fig 2.c
    for i_labels, name_labels in enumerate(labels):
         mod_depth = {}
         surr = {}
         for jf, freq in enumerate(freq_band): 
             
            mod_depth[str(freq)] = cosinefit_ll[str(i_labels)][str(freq)][0]['amp']
            surr[str(freq)] = np.mean(cosinefit_ll[str(i_labels)][str(freq)][0]['surrogate'])
    
    
    
    
            
         ax[i_labels].scatter(xi, (np.array(list(mod_depth.values()))),  c = 'k', label='Real data')
         ax[i_labels].plot(xi, (np.array(list(mod_depth.values()))),  c = 'k', alpha=0.1)
    
         ax[i_labels].scatter(xi, ((np.array(list(surr.values())))),  c = 'r', label='Surrogate')
         ax[i_labels].plot(xi, ((np.array(list(surr.values())))),  c = 'r', alpha= 0.1)
         ax[i_labels].set_xticks(xi, np.array(freq_band))
         ax[i_labels].set_xlabel("Frequecies (Hz)", weight='bold')
         ax[i_labels].set_ylabel("Modulation depth", weight='bold')
         ax[i_labels].legend(loc='upper right')
         ax[i_labels].set_ylim(0, 0.6)
         ax[i_labels].set_title(f'{name_labels} Group Cosine Models', weight='bold')
         ax[3].plot(3,  0.5, '*', color = 'k')
         plt.show()
         fig.savefig(save_folder+'all_labels.svg')
    return fig
    


def reading_cosine_function_parameters(x, labels):
    
    p = {}
    chi = {}
    phi = {}
    mag = {}
    chi_red = {}
    mod_depth = {}
    surrogate = {}
    freq_band = np.arange(4, 44, 4)
    
    for num_sub in range(len(x)):
        p[str(num_sub)] = {}
        chi[str(num_sub)] = {}
        phi[str(num_sub)] = {}
        mag[str(num_sub)] = {}
        chi_red[str(num_sub)] = {}
        mod_depth[str(num_sub)] = {}
        surrogate[str(num_sub)] = {}
    
        
    
            
        for i in range(len(labels)): 
            p[str(num_sub)][str(i)]  = {}
            chi[str(num_sub)][str(i)]  = {}
            phi[str(num_sub)][str(i)]  = {}
            mag[str(num_sub)][str(i)]  = {}
            chi_red[str(num_sub)][str(i)]  = {}
            mod_depth[str(num_sub)][str(i)] = {}
            surrogate[str(num_sub)][str(i)]  = {}
    
            
    
            for jf, freq in enumerate(freq_band):  
                mod_depth[str(num_sub)][str(i)][str(freq)] = x[num_sub][str(i)][str(freq)][0]['amp']
                surrogate[str(num_sub)][str(i)][str(freq)] = x[num_sub][str(i)][str(freq)][0]['surrogate']
                phi[str(num_sub)][str(i)][str(freq)] = x[num_sub][str(i)][str(freq)][0]['Fit'].best_values['phi']
                phi[str(num_sub)][str(i)][str(freq)] =  np.degrees(x[num_sub][str(i)][str(freq)][0]['Fit'].best_values['phi'])
                p[str(num_sub)][str(i)][str(freq)]  = x[num_sub][str(i)][str(freq)][0]['p']
                chi[str(num_sub)][str(i)][str(freq)]  = x[num_sub][str(i)][str(freq)][0]['Fit'].chisqr
                chi_red[str(num_sub)][str(i)][str(freq)]  =  x[num_sub][str(i)][str(freq)][0]['Fit'].redchi
                mag[str(num_sub)][str(i)][str(freq)] = x[num_sub][str(i)][str(freq)][0]['Fit'].best_fit
                
    
    
    amp = {}
    phase = {}
    amp_erp_all = np.zeros([len(freq_band), len(x)])
    phase_erp_all = np.zeros([len(freq_band), len(x)])
    
    
    for i in range(len(labels)):
        amp[str(i)] = {}
        phase[str(i)] = {}
    
        
        for num_sub in range(len(x)):   
            amp_erp_all[:, num_sub]  = np.array(list(mod_depth[str(num_sub)][str(i)].values()))
            phase_erp_all[:, num_sub]  = np.array(list(phi[str(num_sub)][str(i)].values()))
        amp_array = np.mean(amp_erp_all, axis = 1)
        phase_array = np.mean(phase_erp_all, axis = 1)
        amp[str(i)] = amp_array
        phase[str(i)] = phase_array

    return(mod_depth, surrogate, phi)




def plot_torrecillos_2c_errorbar(mod_depth, surrogate, i, name_labels, save_folder):
    plt.style.use('default')
    freq_band = range(len(np.arange(4, 44, 4)))
       
    
    amp_erp_all = [] 
    surrogate_erp_all = []  
    for num_sub in range(len(mod_depth)):  
        print(num_sub)
        amp_erp_all.append(np.array(list(mod_depth[str(num_sub)][str(i)].values())))
        surrogate_erp_all.append(np.mean(np.array(list(surrogate[str(num_sub)][str(i)].values())), axis =1))
        
    amp_erp_all_arr = np.array(amp_erp_all)
    surrogate_erp_all_arr = np.array(surrogate_erp_all)
    
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(freq_band, np.mean(amp_erp_all_arr, axis = 0 ), color = 'k',  alpha=0.1)
    ax.plot(freq_band, np.mean(surrogate_erp_all_arr, axis = 0 ), color = 'r', alpha=0.1)
    e_mod= np.std(amp_erp_all_arr, axis = 0 )
    plt.errorbar(freq_band, np.mean(amp_erp_all_arr, axis = 0 ), e_mod, color = 'k',  linestyle='None', marker='o',   label = 'Real data')
    e_mod= np.std(surrogate_erp_all_arr, axis = 0 )
    plt.errorbar(freq_band, np.mean(surrogate_erp_all_arr, axis = 0 ), e_mod, color = 'r',  linestyle='None', marker='o',   label = 'Surrogate')
    plt.title(f'{name_labels}, Individual Sine Models', weight = 'bold')    
    ax.set_ylabel('Modulation depth', weight = 'bold') 
    ax.set_xlabel('Frequencies (Hz)', weight = 'bold')    
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=0.2, top=1.5)
    threshold = 6
    
    
    
    
    # cluster permutation test
    T_obs, clusters, cluster_p_values, H0 = \
    permutation_cluster_test([amp_erp_all_arr, surrogate_erp_all_arr], n_permutations=1000, threshold=threshold, tail=1, n_jobs=1, out_type='mask')



    for i_c, c in enumerate(clusters):
        c = c[0]
        if cluster_p_values[i_c] <= 0.05:
            plt.plot(freq_band[c.start],  1.15, '*', color = 'k')
            ax.text( freq_band[c.start]-1, 1.2, f'P-value = {cluster_p_values[i_c]}', color='k', weight = 'bold') 



    plt.xticks(freq_band, np.arange(4, 44, 4))
    plt.show()   
    fig.savefig(save_folder+f'plot_torrecillos_2c_errorbar_{name_labels}.svg')
    return fig



def subplot_torrecillos_2c_errorbar(mod_depth, surrogate, labels, save_folder): 
    plt.style.use('default')
    fig, ax = plt.subplots(1, 4, figsize=(20,4))
    for i_labels, name_labels in enumerate(labels):
        plt.style.use('default')
        freq_band = range(len(np.arange(4, 44, 4)))
           
        
        amp_erp_all = [] 
        surrogate_erp_all = []  
        for num_sub in range(len(mod_depth)):  
    
            amp_erp_all.append(np.array(list(mod_depth[str(num_sub)][str(i_labels)].values())))
            surrogate_erp_all.append(np.mean(np.array(list(surrogate[str(num_sub)][str(i_labels)].values())), axis =1))
            
        amp_erp_all_arr = np.array(amp_erp_all)
        surrogate_erp_all_arr = np.array(surrogate_erp_all)
        
        
    
        ax[i_labels].plot(freq_band, np.mean(amp_erp_all_arr, axis = 0 ), color = 'k',  alpha=0.1)
        ax[i_labels].plot(freq_band, np.mean(surrogate_erp_all_arr, axis = 0 ), color = 'r', alpha=0.1)
        e_mod= np.std(amp_erp_all_arr, axis = 0 )
        ax[i_labels].errorbar(freq_band, np.mean(amp_erp_all_arr, axis = 0 ), e_mod, color = 'k',  linestyle='None', marker='o',   label = 'Real data')
        e_mod= np.std(surrogate_erp_all_arr, axis = 0 )
        ax[i_labels].errorbar(freq_band, np.mean(surrogate_erp_all_arr, axis = 0 ), e_mod, color = 'r',  linestyle='None', marker='o',   label = 'Surrogate')
        ax[i_labels].set_title(f'{name_labels}, Individual Sine Models', weight = 'bold')    
        ax[i_labels].set_ylabel('Modulation depth', weight = 'bold') 
        ax[i_labels].set_xlabel('Frequencies (Hz)', weight = 'bold')    
        ax[i_labels].legend(loc='upper right')
        ax[i_labels].set_ylim(bottom=0.2, top=1.5)
        threshold = 6.5
        
        
        
        
        # cluster permutation test
        T_obs, clusters, cluster_p_values, H0 = \
        permutation_cluster_test([amp_erp_all_arr, surrogate_erp_all_arr], n_permutations=1000, threshold=threshold, tail=1, n_jobs=1, out_type='mask')
    
    
    
        for i_c, c in enumerate(clusters):
            c = c[0]
            if cluster_p_values[i_c] <= 0.05:
                ax[i_labels].plot(freq_band[c.start],  1.15, '*', color = 'k')
                ax[i_labels].plot(freq_band[c.stop - 1],  1.15, '*', color = 'k')
                ax[i_labels].text( freq_band[c.start]-2, 1.65, f'P-value = {cluster_p_values[i_c]}', color='k', weight = 'bold') 
    
    
    
        ax[i_labels].set_xticks(freq_band, np.arange(4, 44, 4))
        plt.show()   
    fig.savefig(save_folder+f'plot_torrecillos_2c_errorbar_subplot.svg')
    return fig



def subplot_torrecillos_2c_errorbar_ttest(mod_depth, surrogate, labels, save_folder): 
    plt.style.use('default')
    fig, ax = plt.subplots(1, 4, figsize=(20,4))
    for i_labels, name_labels in enumerate(labels):
        plt.style.use('default')
        freq_band = range(len(np.arange(4, 44, 4)))
           
        
        amp_erp_all = [] 
        surrogate_erp_all = []  
        for num_sub in range(len(mod_depth)):  
    
            amp_erp_all.append(np.array(list(mod_depth[str(num_sub)][str(i_labels)].values())))
            surrogate_erp_all.append(np.mean(np.array(list(surrogate[str(num_sub)][str(i_labels)].values())), axis =1))
            
        amp_erp_all_arr = np.array(amp_erp_all)
        surrogate_erp_all_arr = np.array(surrogate_erp_all)
        
        
    
        ax[i_labels].plot(freq_band, np.mean(amp_erp_all_arr, axis = 0 ), color = 'k',  alpha=0.1)
        ax[i_labels].plot(freq_band, np.mean(surrogate_erp_all_arr, axis = 0 ), color = 'r', alpha=0.1)
        e_mod= np.std(amp_erp_all_arr, axis = 0 )
        ax[i_labels].errorbar(freq_band, np.mean(amp_erp_all_arr, axis = 0 ), e_mod, color = 'k',  linestyle='None', marker='o',   label = 'Real data')
        e_mod= np.std(surrogate_erp_all_arr, axis = 0 )
        ax[i_labels].errorbar(freq_band, np.mean(surrogate_erp_all_arr, axis = 0 ), e_mod, color = 'r',  linestyle='None', marker='o',   label = 'Surrogate')
        ax[i_labels].set_title(f'{name_labels}, Individual Cosine Models', weight = 'bold')    
        ax[i_labels].set_ylabel('Modulation depth', weight = 'bold') 
        ax[i_labels].set_xlabel('Frequencies (Hz)', weight = 'bold')    
        ax[i_labels].legend(loc='upper right')
        ax[i_labels].set_ylim(bottom=0.2, top=1.5)
     
        
        
        
        # ttest
        #array([0.03187757, 0.38432387, 0.56625237, 0.01421358, 0.03051903,0.62597897, 0.0574029 , 0.04038249, 0.08356334, 0.17049245])
        t_val, Pval =stats.ttest_rel(amp_erp_all_arr, surrogate_erp_all_arr)
    
        for i_freq, _ in enumerate(freq_band):

            if Pval[i_freq] <= 0.05:
                ax[i_labels].plot(freq_band[i_freq],  1.2, '*', color = 'k')
                ax[i_labels].text( (i_freq/2)*3, 1.65, f'{np.round(Pval[i_freq],3)}', color='k', weight = 'bold') 
    
    
    
        ax[i_labels].set_xticks(freq_band, np.arange(4, 44, 4))
        plt.show()   
    fig.savefig(save_folder+f'plot_torrecillos_2c_errorbar_subplot_ttest.svg')
    return fig






def phase_to_bin_class(x, phi, labels):
    
    bin_class_all = {}
    phi_tar_freq_all = {}
    
    for i in range(len(labels)):
        phi_tar_freq = np.zeros([10, len(x)])
        for i_freq, freq in enumerate(np.arange(4,44,4)):
    
            for num_sub in range(len(x)):   
                phi_tar_freq[i_freq, num_sub]  = phi[str(num_sub)][str(i)][str(freq)]
                if  phi[str(num_sub)][str(i)][str(freq)] < 0:
                    phi_tar_freq[i_freq, num_sub]  = abs(phi[str(num_sub)][str(i)][str(freq)]) 
                else:
                    phi_tar_freq[i_freq, num_sub]  =  360 - phi[str(num_sub)][str(i)][str(freq)]
                    
    

        
    
    
    
    
    
    
    
        
        
        
        bin_num = 8
        bin_anticlockwise = np.linspace(0,360,int(bin_num+1))  # cover half of the circle -> with half of bin_num
        bin_clockwise = np.linspace(-360,0,int(bin_num+1)) 
        
        
        bin_class = np.nan*np.zeros(phi_tar_freq.shape)
        phi_idx = np.nan*np.zeros(phi_tar_freq.shape)
        
        for [row,col], phases in np.ndenumerate(phi_tar_freq):
        # numbers correspond to the anti-clockwise unit circle eg. bin = 1 -> equals 22.5 deg phase for 16 bins
            if phases > 0:
                    idx = np.where(np.isclose(math.ceil(phi_tar_freq[row,col]), bin_anticlockwise[:], atol=360/(bin_num*2)))
                    phi_idx[row,col] = bin_anticlockwise[idx]
                    # Returns a boolean array where two arrays are element-wise equal within a tolerance.
        # atol -> absolute tolerance level -> bin margins defined by 360° devided by twice the bin_num      
        # problem: rarely exactly between 2 bins -> insert nan
                    if len(idx) > 1:
                        idx = np.nan
                    bin_class[row,col] = idx[0]
        
            elif phases < 0:
                    idx, = np.where(np.isclose(math(phi_tar_freq[row,col]), bin_clockwise[:], atol=360/(bin_num*2)))  
                    phi_idx[row,col] = bin_clockwise[idx]
                    if len(idx) > 1:
                        idx = np.nan      
                    bin_class[row,col] = idx[0]
                    
                    
                    
        
        # combine 360  and 0 together because they are basically the same
        bin_class[bin_class == 8] = 0
        bin_class_all[str(i)] = bin_class
        phi_tar_freq_all[str(i)] = phi_idx
    return(bin_class_all, phi_tar_freq_all)











def P2R(radii, angles):
    return radii * np.exp(1j*angles)

def R2P(x):
    return abs(x), np.angle(x)


def phase_optimal_per_sub(ax, phase_class, phase_val, phase_val_g,  title):
    plt.style.use('default')
    
    plasma_colormap = plt.cm.get_cmap("plasma")
    COLORS = [plasma_colormap(x) for x in np.linspace(0.8, 0.15, num=4)]
  
 
    bin_num =8
    unique, counts = np.unique(phase_class, return_counts=True)
    phase_class_zero = np.zeros(8) 
    # adding zero
    for i_1, i_2 in enumerate(unique):
        phase_class_zero[int(i_2)] = counts[i_1]
    
    # number of equal bins
    bins = np.linspace(0.0, 2 * np.pi, bin_num + 1)
    
    #n, _, _ = plt.hist(bin_class[4, :], bins)
    # Why does plt.hist give me different values compared to when I do the bin class
    
    width = 2 * np.pi / bin_num
    #ax.plot(1, 1, 1, projection='polar')
    ax.bar(bins[:bin_num], phase_class_zero,  width=width, bottom=0.0)
    
    r, theta =R2P(np.sum(P2R(1,  np.radians(phase_val))))
    ax.plot([0,(theta)], [0, r ],  lw=3, color = 'red') 
    #ax.arrow(0,  0, theta, r , linewidth = 1.5,  head_width = 0.3, head_length = 1, fc='maroon', ec='maroon') 
    r_g, theta_g =R2P(np.sum(P2R(1,  np.radians(phase_val_g))))
    ax.plot([0,(theta_g)], [0, r_g ],  lw=2, color = 'k') 
    ax.set_ylim([0,11])
    ax.set_ylim([0,11])


    if np.degrees((theta))> 0:
         #ax.set_title(r"$\bf{" f'{title}' "}$" '\n'  f'{( np.degrees((theta)),2)}'   u'\N{DEGREE SIGN}'  '\n' f'{np.round( np.degrees((theta_g)),2)}' u'\N{DEGREE SIGN}' , color = {'red', 'k'}) 
       ax.text(1.75, 26,   f'{title}', weight = 'bold')
       ax.text(1.8, 22, f'{np.round( np.degrees((theta)),2)}'u'\N{DEGREE SIGN}', color = 'maroon')
       ax.text(1.84, 19, f'{np.round( np.degrees((theta_g)),2)}'u'\N{DEGREE SIGN}')
    else:
         # To show positive degrees 360 + np.degrees((theta))
         
         #ax.set_title(r"$\bf{" f'{title}' "}$" '\n' f'{np.round(360 + np.degrees((theta)),2)}' u'\N{DEGREE SIGN}' '\n' f'{np.round(360 + np.degrees((theta_g)),2)}' u'\N{DEGREE SIGN}' ) 
         ax.text(1.75, 26,  f'{title}', weight = 'bold')
         ax.text(1.8, 22, f'{np.round(360 + np.degrees((theta)),2)}' u'\N{DEGREE SIGN}', color = 'maroon' )
         ax.text(1.84, 19, f'{np.round(360 + np.degrees((theta_g)),2)}' u'\N{DEGREE SIGN}')

    plt.show()
    return(theta, theta_g)







def time_course(theta, theta_g):    
   
    plt.style.use('default')
    sr = 1000.0
    # sampling interval
    ts = 1/sr
    t = np.arange(0,400,1)
    # frequency of the signal
    freq = 0.016 
    
    y1 = 0.5*np.cos(2*np.pi*freq*t  )
    y2 = 0.5*np.cos(2*np.pi*freq*t -  theta_g)
    y3 = 0.5*np.cos(2*np.pi*freq*t - theta)

    d, ax = plt.subplots(figsize = (20, 6))
    ax.plot(t, y1, 'grey',  linestyle=':', label = '16 Hz wave ')
    ax.plot(t, y2, 'k', label = f' Group phase: {np.round(np.degrees(theta_g), 2)}\xb0 ' )
    ax.plot(t, y3, 'b', label = f' Individual phase: {np.round(np.degrees(theta))}\xb0 ' )
    ax.set_xticks(np.arange(min(t), max(t)+1, 20.0))
    #ax.plot(t, y4, 'g', label = ' Cluster-based \n permutation phase: 45\xb0 ' )
    ax.axvline(np.argmax(y2[170:210])+170, color='k')
    ax.axvline(np.argmax(y3[170:210])+170, color='b')
    #ax.axvline(np.argmax(y4[170:210])+170, color='g')
    
    #ax.axvline(t[np.argmax(y1)], color='grey', label = 'GFP peak for P190 channels' )
    ax.set_ylabel('Amplitude (z-scored)')
    ax.set_xlabel('Time (ms)')
    ax.legend(loc='upper right')
    plt.show()
    return d
    
   

def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter





def delay_degree(theta, theta_g):
    
    plt.style.use('default') 
    from collections import OrderedDict
    linestyles = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

    f,ax=plt.subplots(figsize=(10,6))
    x = np.linspace(-3/2*np.pi, 2*np.pi,500)
    y1 = 0.5*np.cos(x)
    y2 = 0.5*np.cos(x - theta_g)
    y3 = 0.5*np.cos(x - theta)
    y4 = 0.5*np.cos(x - np.radians(45))
    ax.plot(x, y1, 'grey',  linestyle = linestyles['loosely dashed'], label = '16 Hz wave ')
    ax.plot(x, y2, 'k', label = f' Group phase: {np.round(np.degrees(theta_g), 2)}\xb0 '  )
    ax.plot(x, y3, 'b', label = f' Individual phase: {np.round(np.degrees(theta), 2)}\xb0 ')
    ax.plot(x, y4, 'w',  linestyle = linestyles['loosely dotted'] )
    ax.axvline(x[np.argmax(y2)], color='k')
    ax.axvline(x[np.argmax(y3)], color='b')
    ax.scatter(x[np.argmax(y4)],y1[np.argmax(y4)], marker='x', color='r',linewidth=2)
    ax.scatter(x[np.argmin(y4[0:200])],y1[np.argmin(y4)], marker='x', color='r', linewidth=2)
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(ph_analysis.multiple_formatter()))
    ax.set_ylabel('Amplitude (z-scored)')
    ax.set_xlabel('Phases(\xb0)')
    ax.legend(loc='upper right')
    plt.show()
    return f

