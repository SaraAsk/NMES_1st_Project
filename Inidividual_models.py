#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 13:52:39 2022

@author: sara
"""


import mne
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import phase_analysis_function_Jan as ph_analysis # phase analysis function



#%%
exdir_epoch = "/home/sara/NMES/analyzed data/Sep_repeat/Epochs_NMES_manually_rejected/"
save_folder =  "/home/sara/NMES/analyzed data/Sep_repeat/individual_models/"
files = list(Path(exdir_epoch).glob('**/*epo.fif'))



all_subjects_names = []
cosine_fit_all_subjects = []
cosine_fit_all_subjects_LL = []
amplitudes_cosines_all_subjects = []
amplitudes_cosines_all_subjects_LL = []


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

for i_sub, f in enumerate(files):

    subject_info = f.parts 
    all_subjects_names.append(str(subject_info[-1][0:9]))
    epochs_eeg = mne.read_epochs(f, preload=True).copy().pick_types(eeg=True)
    # removing the effect of phase amp according to Granö et al. 2022.
    # amp after stim - amp before stim     
    epochs_eeg_amp_mod = epochs_eeg._data[:,:,1001:] - epochs_eeg._data[:,:,0:1000]
    epochs_eeg = mne.EpochsArray(data = epochs_eeg_amp_mod,  info = epochs_eeg.info, events = epochs_eeg.events, event_id = epochs_eeg.event_id, on_missing='ignore')
    # 4 Hz step with lucky loop labels
    evoked = {}
    erp_amplitude_ll = {}
    ERP_byfreqandphase = {}
    epochs_byfreqandphase = {} 

    
    for i_ch, ch in enumerate(ERP_indexs):
        evoked[str(i_ch)] = {}
        erp_amplitude_ll[str(i_ch)] = {}
        ERP_byfreqandphase[str(i_ch)] = {}
        epochs_byfreqandphase[str(i_ch)] = {} 
        for freq in np.arange(4,44,4):
            epochs_byfreqandphase[str(i_ch)][str(freq)] = {}
            ERP_byfreqandphase[str(i_ch)][str(freq)] = {}
            evoked[str(i_ch)][str(freq)] = {}
            for phase in np.arange(0,360,45):
                sel_idx = ph_analysis.Select_Epochs(epochs_eeg, freq, phase)
                epochs_byfreqandphase[str(i_ch)][str(freq)][str(phase)] = epochs_eeg[sel_idx]
                
                if i_ch == 0: #P30
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
                    evoked[str(i_ch)][str(freq)][str(phase)] = 0
            
 
    cosinefit_ll, amplitudes_cosine_ll, pvalues_cosine_ll = ph_analysis.do_cosine_fit_ll(evoked, np.arange(0,360,45), np.arange(4,44,4), labels, subjects = 'individual' , perm = True)
    
    amplitudes_cosines_all_subjects_LL.append(amplitudes_cosine_ll)
    cosine_fit_all_subjects_LL.append(cosinefit_ll)
    
    if not (cosinefit_ll[str(0)] ):
        print(f'There are not enough epochs by freq and phase for Subject: {subject_info[-3]}')


        
#%% Saving the pickle files and plotting the strength of Mod by the average of subjects
names = 'all_subjects_names'+ '.p'
with open(str(save_folder) + names, 'wb') as fp:
    pickle.dump(all_subjects_names, fp, protocol=pickle.HIGHEST_PROTOCOL)
   
cosine_amp_LL = 'amplitudes_cosines_all_subjects_LL'+'.p'
with open(str(save_folder) + cosine_amp_LL, 'wb') as fp:
    pickle.dump(amplitudes_cosines_all_subjects_LL, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
cosine_fit_ll = 'cosine_fit_all_subjects_ll' + '.p'
with open(str(save_folder) + cosine_fit_ll, 'wb') as fp:
    pickle.dump(cosine_fit_all_subjects_LL, fp, protocol=pickle.HIGHEST_PROTOCOL)        
    






#%% reading pickle files
with open(save_folder +'cosine_fit_all_subjects_ll.p', 'rb') as f:
    cosine_fit_all_subjects_LL = pickle.load(f)
  
with open(save_folder+ 'all_subjects_names.p', 'rb') as f:
    subject_names =  pickle.load(f)     


#%%
#reading pickle file for group model    
with open("/home/sara/NMES/analyzed data/Sep_repeat/Group_Models/" +'Group_cosine.p', 'rb') as f:
    group_model = pickle.load(f)    
labels = ['P30', 'N60', 'N120', 'P190'] 
phi_array_deg = ph_analysis.Circ_corr(group_model, labels)    
p_val, phi_array_deg_correct = ph_analysis.amp_p_chi(group_model, labels, phi_array_deg, save_folder) 
    
#%%    


mod_depth, surrogate, phi = ph_analysis.reading_cosine_function_parameters(cosine_fit_all_subjects_LL, labels) 

#%%
title = 'Individual'
for count_sub, num_sub in enumerate(cosine_fit_all_subjects_LL):
    ph_analysis.best_fit_plot(cosine_fit_all_subjects_LL[count_sub], labels, save_folder + '/wave_fit_subs/', subject_names[count_sub], title, [-2, 2])
    
    
    



#%%

for i_labels, name_labels in enumerate(labels):
    ph_analysis.plot_torrecillos_2c_errorbar(mod_depth, surrogate, i_labels, name_labels, save_folder) 

#%%  





ph_analysis.subplot_torrecillos_2c_errorbar(mod_depth, surrogate, labels, save_folder)

ph_analysis.subplot_torrecillos_2c_errorbar_ttest(mod_depth, surrogate, labels, save_folder)
      
#%% 




bin_class_all, phi_tar_freq_all =  ph_analysis.phase_to_bin_class(cosine_fit_all_subjects_LL, phi, labels)

#%%

    
plt.style.use('default')  
titles = ['4 Hz', '8 Hz', '12 Hz', '16 Hz', '20 Hz', '24 Hz', '28 Hz', '32 Hz', '36 Hz', '40 Hz']   
fig = plt.figure(constrained_layout=True, figsize=(20,12))
#fig.suptitle('Optimal Phase distribution', fontweight="bold")
# create 3x1 subfigs
subfigs = fig.subfigures(nrows=len(labels), ncols=1)
theta = {}
theta_g = {}
for row, subfig in enumerate(subfigs):
    
    axs = subfig.subplots(nrows=1, ncols=10,subplot_kw=dict(projection='polar'))
    for freq, ax in enumerate(axs):
         axs[0].set_ylabel(f'{labels[row]}', rotation=0, size=14, labelpad = 50, fontweight="bold")
         theta[str(freq)], theta_g[str(freq)] = ph_analysis.phase_optimal_per_sub(ax,  bin_class_all[str(row)][freq,:],  phi_tar_freq_all[str(row)][freq,:], phi_array_deg_correct[freq,row], titles[freq])   
 
fig.savefig(save_folder+'optimal_phase_distribution.svg')         
   
    


#%%    



fig = ph_analysis.time_course(theta[str(3)], theta_g[str(3)])
fig.savefig(save_folder+'time_course_cosine.svg')    

#%%


fig = ph_analysis.delay_degree(theta[str(3)], theta_g[str(3)])
fig.savefig(save_folder+'degree_cosine.svg')  


