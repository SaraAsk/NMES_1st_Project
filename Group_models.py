
"""
Created on Tue Oct 11 14:15:56 2022

@author: sara
"""




import mne
import pickle
import numpy as np
import pandas as pd
from scipy.stats import  zscore
import seaborn as sns; sns.set_theme()
from pathlib import Path
import matplotlib.pyplot as plt
# phase analysis function
import phase_analysis_function_Jan as ph_analysis






#%% Extracting phase and frequency from the bipolar channel around C3.

concat_epoch_sub = False
exdir_epoch ='/home/sara/NMES/analyzed data/Sep_repeat/Epochs_NMES_manually_rejected/'
save_folder = '/home/sara/NMES/analyzed data/Sep_repeat/Epochs_NMES_manually_rejected/'


"""files that have been used for this analysis: Subject UtLi and StLa was deleted completely
    BaJo_R001, BrTi_R001, EuSa_R001, HaSa_R001-HaSa_R002(concat), HeDo_R010, KoAl_R008, LaCa_R001, LiLu_R001
    MeAm_R001-MeAm_R002, MiAn_R001, Nema_R001, RaLu_R001, RuMa_R001, ScSe_R001, StLa_R001, VeVa_R001, WoIr_R001
    ZaHa_R001, ZhJi_R001-ZhJi_R002(concat), ZiAm_R001"""
    

# Subjects to concatenate the epochs: HaSa, MeAm, ZhJi
files = list(Path(exdir_epoch).glob('*ZhJi*'))
if concat_epoch_sub == True:

    epoch_concat_subs, info = ph_analysis.epoch_concat_subs_mutltiple_files(files)
    epoch_concat_subs.save(save_folder+ str(info[-1][0:4]) + '_' + 'concat_manually' + '_epo.fif', overwrite = True, split_size='2GB')
    




#%%






exdir_epoch = "/home/sara/NMES/analyzed data/Sep_repeat/Epochs_NMES_manually_rejected/"
save_folder =  '/home/sara/NMES/analyzed data/Sep_repeat/Group_Models/'
files = list(Path(exdir_epoch).glob('*epo.fif*'))
amplitudes_cosines_all_subjects = []
amplitudes_cosines_all_subjects_LL = []
all_subjects_names = []

cosine_fit_all_subjects = []
cosine_fit_all_subjects_LL = []



win_erp0 = [25, 40]   #P30
win_erp1 = [50, 75]   #N60
win_erp2 = [100, 130] #N120
win_erp3 = [170, 210] #P190
thresholds=[2.8, 2.8, 3.5, 3.2]
labels = ['P30', 'N60', 'N120', 'P190']
win_erps = np.array([win_erp0, win_erp1, win_erp2, win_erp3])


# Name of the clustered channels must be before looping though epochs of subject, so this process only happens one time. 
all_ch_names, pvals_all= ph_analysis.clustering_channels(win_erps, exdir_epoch,  thresholds, labels, save_folder)    


ch_names = ph_analysis.get_ERP_1st_2nd(exdir_epoch, save_folder)     



ERP_indexs = []
for i_ch, chs in enumerate(all_ch_names):
    _, _, ERP_index = np.intersect1d(chs ,ch_names, return_indices=True)
    ERP_indexs.append(np.sort(ERP_index).T)

#%%

all_sub_evoked = []
for i_sub, f in enumerate(files):


    
    # Extracting ERP amplitude for frequency and phases according to bipolar channel.
        
    # Subj_path is added to exdir, so the EEG epoch files and bipolar  channels are selected from the same subject. 
    epochs_eeg = mne.read_epochs(f, preload=True).copy().pick_types(eeg=True)
    # removing the effect of phase amp according to Granö et al. 2022.
    # amp after stim - amp before stim     
    epochs_eeg_amp_mod = epochs_eeg._data[:,:,1001:] - epochs_eeg._data[:,:,0:1000]
    # making mne epoch structure
    epochs_eeg = mne.EpochsArray(data = epochs_eeg_amp_mod,  info = epochs_eeg.info, events = epochs_eeg.events, event_id = epochs_eeg.event_id, on_missing='ignore')
 
    

    epochs_byfreqandphase = {} 
    erp_amplitude_ll = {}
    ERP_byfreqandphase = {}
    evoked = {}
    evoked_z = {}
    
    for i_ch, ch in enumerate(ERP_indexs):
        epochs_byfreqandphase[str(i_ch)] = {} 
        erp_amplitude_ll[str(i_ch)] = {}
        ERP_byfreqandphase[str(i_ch)] = {}
        evoked[str(i_ch)] = {}
        evoked_z[str(i_ch)] = {}
        for freq in np.arange(4,44,4):
            epochs_byfreqandphase[str(i_ch)][str(freq)] = {}
            ERP_byfreqandphase[str(i_ch)][str(freq)] = {}
            evoked[str(i_ch)][str(freq)] = {}
            evoked_z[str(i_ch)][str(freq)] = {}
            for phase in np.arange(0,360,45):
                sel_idx = ph_analysis.Select_Epochs(epochs_eeg, freq, phase) # Selecting lucky loop labels
                epochs_byfreqandphase[str(i_ch)][str(freq)][str(phase)] = epochs_eeg[sel_idx]
                
                if i_ch == 0:   #P30
                    ERP_byfreqandphase[str(i_ch)][str(freq)][str(phase)] = np.mean(epochs_byfreqandphase[str(i_ch)][str(freq)][str(phase)]._data[:, ch, win_erps[0,0]: win_erps[0,1]], axis=0)
                    evoked[str(i_ch)][str(freq)][str(phase)] = np.mean(np.max(ERP_byfreqandphase[str(i_ch)][str(freq)][str(phase)], axis = 1))
                
                elif i_ch == 1: #N60
                    ERP_byfreqandphase[str(i_ch)][str(freq)][str(phase)] = np.mean(epochs_byfreqandphase[str(i_ch)][str(freq)][str(phase)]._data[:, ch, win_erps[1,0]: win_erps[1,1]], axis=0)
                    evoked[str(i_ch)][str(freq)][str(phase)] = np.mean(np.min(ERP_byfreqandphase[str(i_ch)][str(freq)][str(phase)], axis = 1))
                 
                elif i_ch == 2: #N140
                    ERP_byfreqandphase[str(i_ch)][str(freq)][str(phase)] = np.mean(epochs_byfreqandphase[str(i_ch)][str(freq)][str(phase)]._data[:, ch, win_erps[2,0]: win_erps[2,1]], axis=0)
                    evoked[str(i_ch)][str(freq)][str(phase)] = np.mean(np.min(ERP_byfreqandphase[str(i_ch)][str(freq)][str(phase)], axis = 1))
       
                elif i_ch == 3: #P190
                    ERP_byfreqandphase[str(i_ch)][str(freq)][str(phase)] = np.mean(epochs_byfreqandphase[str(i_ch)][str(freq)][str(phase)]._data[:, ch, win_erps[3,0]: win_erps[3,1]], axis=0)
                    evoked[str(i_ch)][str(freq)][str(phase)] = np.mean(np.max(ERP_byfreqandphase[str(i_ch)][str(freq)][str(phase)], axis = 1))
                                 

                if str(evoked[str(i_ch)][str(freq)][str(phase)]) == 'nan':
                    evoked[str(i_ch)][str(freq)][str(phase)] = 0 # removing 'nan' objects
            evoked_z[str(i_ch)][str(freq)] = zscore(list(evoked[str(i_ch)][str(freq)].values()))        
    all_sub_evoked.append(evoked_z) 






# calculating z-scores per subject, and target frequency to render ERPs
all_sub_evoked_df = pd.DataFrame(all_sub_evoked)       
all_evoked_freq = {}
for i_erp, i in enumerate(ERP_indexs):
    all_evoked_freq[str(i_erp)] = {}
    for freq in np.arange(4, 44, 4):
        avg = []
        for i_sub in range(len(all_sub_evoked)):
            avg.append(all_sub_evoked_df[str(i_erp)][i_sub][str(freq)])
        all_evoked_freq[str(i_erp)][str(freq)] = np.mean(avg, axis = 0)

# I am fitiing a sine but I haven't change the name of the function
# I decided to fit a sine because of lucky-loop FFT algorithm (now it matches with the clustering results). I have change the model to def cosinus(x, amp, phi):
    #return amp * np.sin(x + phi) in the Phase_analysis_function to sine but didn't change the names. So everything is
    # named cosine but the model is sine. 
subject_info = 'Group'
cosinefit_ll, amplitudes_cosine_ll, pvalues_cosine_ll = ph_analysis.do_cosine_fit_ll(all_evoked_freq, np.arange(0,360,45), np.arange(4,44,4), labels, subjects = 'group' , perm = True)



 

#%%% Circular correlation


phi_array_deg = ph_analysis.Circ_corr(cosinefit_ll, labels)



#%% Modulation depth bar plot



# returns modelution depth bar plot, p value, optimal phase and reduced chi squared in data frame format and plots them in a table 
#  phi_array_deg_correct: positive phase values 
p_val, phi_array_deg_correct = ph_analysis.amp_p_chi(cosinefit_ll, labels, phi_array_deg, save_folder)



#%% Plotting best cosine fit and data
title = 'Group'


ph_analysis.best_fit_plot(cosinefit_ll, labels, phi_array_deg, save_folder, title, [-1,1])

#%%




# latency of the channels within the P190 cluster
#%% Modulation depth scatter plot

ph_analysis.fig_2c_scatter_plot(cosinefit_ll, labels, p_val, save_folder)
ph_analysis.fig_2c_scatter_sub_plot(cosinefit_ll, labels, p_val, save_folder)





#%%
# Saving the pickle files and plotting the strength of Mod by the average of subjects

names = 'Group_cosine'+ '.p'
with open(str(save_folder) + 'Group_cosine'+ '.p', 'wb') as fp:
    pickle.dump(cosinefit_ll, fp, protocol=pickle.HIGHEST_PROTOCOL)        
    
  



#%% If I fit a cosine wave
# The group phase is according to the cosine fit that I have fitted to all subs, opt_phase = -33 =360-33 =327




plt.style.use('default')
sr = 1000.0
# sampling interval
ts = 1/sr
t = np.arange(0,400,1)
# frequency of the signal
freq = 0.016

y1 = 0.5*np.cos(2*np.pi*freq*t  )
y2 = 0.5*np.cos(2*np.pi*freq*t -  np.radians(30))


d, ax = plt.subplots(figsize = (20, 6))
ax.plot(t, y1, 'grey',  linestyle='--', label = '16 Hz wave ')
ax.plot(t, y2, 'k', label = ' Group phase: 327\xb0 ' )
ax.set_xticks(np.arange(min(t), max(t)+1, 20.0))
#ax.plot(t, y4, 'g', label = ' Cluster-based \n permutation phase: 45\xb0 ' )
#ax.axvline(np.argmax(y2[170:210])+170, color='k')
#ax.axvline(np.argmax(y4[170:210])+170, color='g')

#ax.axvline(t[np.argmax(y1)], color='grey', label = 'GFP peak for P190 channels' )
ax.set_ylabel('Amplitude (z-scored)')
ax.set_xlabel('Time (ms)')
ax.legend(loc='upper right')
plt.show()


#%%

f,ax=plt.subplots(figsize=(10,6))
x = np.linspace(-3/2*np.pi, 2*np.pi,500)
y1 = 0.5*np.cos(x)
y2 = 0.5*np.cos(x +  np.radians(32))

ax.plot(x, y1, 'grey',  linestyle='--', label = '16 Hz wave ')
ax.plot(x, y2, 'k', label = ' Group phase: 327\xb0 ' )
ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.xaxis.set_major_formatter(plt.FuncFormatter(ph_analysis.multiple_formatter()))
ax.set_ylabel('Amplitude (z-scored)')
ax.set_xlabel('Phases(\xb0)')
ax.legend(loc='upper right')
plt.show()
#%%



f,ax=plt.subplots(figsize=(10,6))
x = np.linspace(-3/2*np.pi, 2*np.pi,500)
y1 = 0.5*np.cos(x)
y2 = 0.5*np.cos(x -  np.radians(32))
y3 = 0.5*np.sin(x)

ax.plot(x, y1, 'grey',  linestyle='--', label = 'cos ')
ax.plot(x, y2, 'k', label = ' data' )
ax.plot(x, y3,'b',  linestyle=':', label = 'sin ')
ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.xaxis.set_major_formatter(plt.FuncFormatter(ph_analysis.multiple_formatter()))
ax.set_ylabel('Amplitude (z-scored)')
ax.set_xlabel('Phases(\xb0)')
ax.legend(loc='upper right')
plt.show()