

#%%


import mne
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from autoreject import AutoReject
from mne.channels import make_standard_montage
import NMES_Preprocessing_functions as prepross

#%%


exdir = "/home/sara/NMES/NMES_Experimnet/"
files = list(Path(exdir).glob('**/*.xdf'))


lists_bad_channels = []
lists_pluses = []
for f in files:
    plt.close('all')
    print(str(f.parts[5]) + '_' + str(f.parts[-1][39:43])) # print sb's name
    
    # Step 1 : reading XDF files and changing their format to raw mne 
    raw, pulses, pulses_corrected = prepross.XDF_correct_time_stamp_reject_pulses(f)
    lists_pluses.append([str(f.parts[5]) +  str(f.parts[-1][39:43]), pulses, pulses_corrected])

    # Preprocessing EEG channels              
    # Selecting EEG channels
    raw_eeg = raw.drop_channels(['EDC_L', 'EDC_R', 'ECR_L', 'ECR_R', 'FCR_L', 'FCR_R', 'FDS_L', 'FDS_R', 'EKG'])
    print('Number of channels in raw_eeg:')
    print(len(raw_eeg.ch_names), end=' → drop nine → ')
    
    
    
    # step 2: apply regression for ocular artifacts
    print('starting regress out eye artifacts')
    #raw_regr = prepross.regress_out_pupils(raw_eeg)


    # Filtering the raw signal to avoid distortion
    raw_eeg._data = mne.filter.notch_filter(raw_eeg._data, raw_eeg.info['sfreq'], 50, notch_widths =2, phase='zero'  )
    raw_eeg.filter(1, 100, method='iir', phase='zero', verbose=0)
    
    
    # step 3 : Rejecting bad channels based on the visualization of variance of channels
    # plotting of channel variance

    eeg_regr_interp, badchans_threshold  = prepross.mark_bad_channels_interpolate(f.parts,raw_eeg)
    # Creating a list of subjects and bad channels that were rejected
    lists_bad_channels.append([str(f.parts[5]) + str(f.parts[-1][39:43]), badchans_threshold])
    
  
    # Step 4 : Creating epochs 
    # 4.1. Create events from the annotations present in the raw file
    # excluding non-unique events and time-stamps
    (events_from_annot, event_dict) = mne.events_from_annotations(raw_eeg)
    u, indices = np.unique(events_from_annot[:,0], return_index=True)
    events_from_annot_unique = events_from_annot[indices]
    event_unique, event_unique_ind  = np.unique(events_from_annot_unique[:,2], return_index=True)
    
    # 4.2. Create epochs based on the events, from -1 to 1s
    # Set the baseline to None, because mne suggests to do a baseline correction after ICA
    epochs = mne.Epochs(raw_eeg, events_from_annot_unique, event_id=event_dict,
                        tmin=-1, tmax=1, reject=None, preload=True,  baseline=None)
    
    
    # 4.3. filtering, 


    # Apply autoreject (local), apply ICA , and apply autoreject (local) again. https://mne.discourse.group/t/eeg-processing-pipeline-with-autoreject/3443/3
    # "first run autoreject only for detection of bad channels but without interpolation." https://www.sciencedirect.com/science/article/pii/S1053811917305013
    ar = AutoReject(n_interpolate=[0])
    epochs_ar, reject_log = ar.fit(epochs).transform(epochs, return_log = True)   
    
    
    # 4.4.  Applying ICA after filtering and before baseline correction 
    data_ica  = prepross.clean_dataset(epochs, reject_log)


    # 4.5. Applying baseline, this is based on what teasa toolbox suggested  
    epochs = data_ica['eeg']

    evokeds = epochs.average()
    all_times = np.arange(-0.8, 0.8, 0.1)
    fig_topo = evokeds.plot_topomap(all_times, ch_type='eeg', time_unit='s', ncols=8, nrows='auto')
    fig_erp = evokeds.plot(spatial_colors = True, gfp=True)

    

    
    # Save epoch files and figures
    save_folder = "/home/sara/NMES/analyzed data/Sep_repeat/epochs/"
    save_folder_figs = "/home/sara/NMES/analyzed data/Sep_repeat/epochs/figs/"
    epochs.save(save_folder+ str(f.parts[5]) + '_' + str(f.parts[-1][39:43]) + '_epo.fif', overwrite = True, split_size='2GB')
    fig_topo.savefig(save_folder_figs +  str(f.parts[5]) + '_' + str(f.parts[-1][39:43]) + '_' + '.png')
    fig_erp.savefig(save_folder_figs + str(f.parts[5]) + '_' + str(f.parts[-1][39:43]) + '_' + '.png')
    

cols_bad_chans = ['Subject', 'Bad Channels']
df_bad_chans = pd.DataFrame(lists_bad_channels, columns = cols_bad_chans)   
df_bad_chans.to_csv (save_folder +'Subject_ channel_rejected.csv', index = None, header=True)   

cols = ['','', 'Pulses', 'Pulses Corrected']
lists_pluses.append(['', '',  pulses , pulses_corrected])
df_pulses = pd.DataFrame(lists_pluses, columns=cols)     
df_pulses.to_csv (save_folder +'Subject_ pulse.csv', index = None, header=True) 
    
    
#%%




import mne
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mne.channels import make_standard_montage

evokeds_all = [] 
evokeds_pahse_effecft_all = []
epochs_data_dir = "/home/sara/NMES/analyzed data/Sep_repeat/epochs/"

epochs_files = Path(epochs_data_dir).glob('*epo.fif*')
save_folder_epochs = "/home/sara/NMES/analyzed data/Sep_repeat/Epochs_NMES_manually_rejected/"
Save_folder_fig = "/home/sara/NMES/analyzed data/Sep_repeat/Epochs_NMES_manually_rejected/Figs/"


dictionary_bad_channels = {'BaJo_R001': ['FT9', 'PO3', 'FT8', 'FT10', 'T8', 'TP10', 'F8', 'AF8', 'TP9'], 'EuSa_R001':['FT10', 'T8', 'TP9', 'FT9'], 
                           'HaSa_R001':['PO7', 'T8'], 'HaSa_R002':['PO4', 'PO7'], 'KoAl_R008':['FT9', 'FT7', 'TP9'], 
                           'LaCa_R001': ['FT7', 'Iz', 'PO7'], 'LiLu_R001':['T8', 'TP9', 'FT9'], 'MeAm_R001':['PO8', 'TP9', 'FT9'],  'MeAm_R002':['FT9', 'TP9'],'NeMa_R001':['FT9','FT10', 'TP9'], 
                           'RuMa_R001': ['Oz'], 'ScSe_R001':['F1', 'FC3', 'C3', 'FT7', 'TP9', 'PO8'], 'StLa':['TP9', 'FT10', 'FT9'], 'ZiAm' :['TP9', 'T7']}






for f in epochs_files:
    plt.close('all')
    
    subject_ID = f.parts[-1][0:9]
    epochs = mne.read_epochs(f, preload= True)
    epochs= epochs.apply_baseline(baseline=(-0.9, -0.1))  
    montage = make_standard_montage('standard_1005')
    epochs = epochs.set_montage(montage)
    epochs = epochs.set_eeg_reference(ref_channels='average')
    evokeds = epochs.average()
    
    
    all_times = np.arange(0, 0.5, 0.01)
    topo_plots = evokeds.plot_topomap(all_times, ch_type='eeg', time_unit='s', ncols=8, nrows='auto')
    ERP_plots = evokeds.plot(spatial_colors = True, gfp = True) 
    if subject_ID in dictionary_bad_channels:
       epochs.info['bads'] = dictionary_bad_channels[subject_ID]
    epochs_clean = epochs.interpolate_bads(reset_bads=True, mode='accurate')
    evokeds_clean = epochs_clean.average()
    all_times = np.arange(0, 0.5, 0.01)
    topo_plots = evokeds_clean.plot_topomap(all_times, ch_type='eeg', time_unit='s', ncols=8, nrows='auto')
    ERP_plots = evokeds_clean.plot(spatial_colors = True, gfp = True) 
    evokeds_all.append(evokeds_clean)
    #topo_plots.savefig(Save_folder_fig +  str(f.parts[-1][0:9]) +'_TOPO' ) 
    #ERP_plots.savefig(Save_folder_fig +  str(f.parts[-1][0:9]) + '_ERP') 
    #epochs_clean.save(save_folder_epochs + str(f.parts[-1][0:9]) + '_manually' + '_epo.fif', overwrite = True, split_size='2GB')


    # Compensating for phase effect
    # https://www.biorxiv.org/content/10.1101/2021.10.01.462795v1
    epochs_amp_mod = np.concatenate((epochs._data[:,:,0:1000], epochs._data[:,:,1001:] - epochs._data[:,:,0:1000]), axis=2)
    epochs_phase_effect = mne.EpochsArray(data = epochs_amp_mod,  info = epochs.info, events = epochs.events, event_id = epochs.event_id, tmin=-1, on_missing='ignore')
    evokeds_pahse_effecft = epochs_phase_effect.average()
    evokeds_pahse_effecft_all.append(evokeds_pahse_effecft)






Evoked_GrandAv = mne.grand_average(evokeds_all)    
Evoked_GrandAv.info['bads'] = ['TP9', 'FT9',  'FT7', 'T7', 'TP7',  'P7', 'F7', 'AF7', 'F5', 'PO7']
Evoked_GrandAv = Evoked_GrandAv.interpolate_bads(reset_bads=True, mode='accurate')
topo_steps = Evoked_GrandAv.plot_topomap(np.arange(0, 0.6, 0.01), ch_type='eeg', time_unit='s', ncols=8, nrows='auto') 
Evoked_GrandAv_crop = Evoked_GrandAv.crop(-0.1, 0.6)
Evoked_GrandAv_c = Evoked_GrandAv.plot_joint(times='peaks')
Evoked_GrandAv.plot(gfp=True, spatial_colors=True)

Evoked_GrandAv_c.savefig(Save_folder_fig +  'all_subs' + '_ERP') 
topo_steps.savefig(Save_folder_fig +  'all_subs' + '_ToPo') 
mne.evoked.write_evokeds(save_folder_epochs +'all_subs' + '_ave.fif', Evoked_GrandAv, overwrite=True) 

Evoked_GrandAv_phase = mne.grand_average(evokeds_pahse_effecft_all)    
Evoked_GrandAv_phase.info['bads'] = ['TP9', 'FT9', 'TP7', 'FT7', 'T7', 'F7']
Evoked_GrandAv_phase = Evoked_GrandAv_phase.interpolate_bads(reset_bads=True, mode='accurate')
topo_steps_ph = Evoked_GrandAv_phase.plot_topomap(np.arange(0, 0.6, 0.01), ch_type='eeg', time_unit='s', ncols=8, nrows='auto') 
Evoked_GrandAv_crop_ph = Evoked_GrandAv_phase.crop(-0.1, 0.6)
Evoked_GrandAv_phase.plot_joint(times='peaks')
Evoked_GrandAv_phase.plot(gfp=True, spatial_colors=True)



Evoked_GrandAv_crop.plot(gfp=False, spatial_colors=False)
Evoked_GrandAv_phase.plot(gfp=False, spatial_colors=False)

Evoked_GrandAv_c = Evoked_GrandAv.plot_joint(times='peaks')
Evoked_GrandAv_phase = Evoked_GrandAv_crop_ph.plot_joint(times='peaks')





Evoked_GrandAv_crop_ph.plot_joint(times='peaks', picks =  ['Fz', 'Cz', 'FC1', 'FC2', 'CP1', 'F1', 'C1', 'C2', 'CPz'])


#%%




import mne
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mne.channels import make_standard_montage
import phase_analysis_function_Jan as ph_analysis

evokeds_all = [] 
evokeds_pahse_effecft_all = []
epochs_data_dir = "/home/sara/NMES/analyzed data/Sep_repeat/Epochs_NMES_manually_rejected/"

epochs_files = Path(epochs_data_dir).glob('*epo.fif*')
save_folder_epochs = "/home/sara/NMES/analyzed data/Sep_repeat/Epochs_NMES_manually_rejected/"
Save_folder_fig = "/home/sara/NMES/analyzed data/Sep_repeat/Epochs_NMES_manually_rejected/Figs/"





for f in epochs_files:
    plt.close('all')
    
   
    epochs = mne.read_epochs(f, preload= True)

    sel_idx = ph_analysis.Select_Epochs_frequency(epochs, 16) # Selecting lucky loop labels
    epochs = epochs[sel_idx]

    montage = make_standard_montage('standard_1005')
    epochs = epochs.set_montage(montage)
    epochs = epochs.set_eeg_reference(ref_channels='average')
    evokeds = epochs.average()
    
    
    all_times = np.arange(0, 0.5, 0.01)
    topo_plots = evokeds.plot_topomap(all_times, ch_type='eeg', time_unit='s', ncols=8, nrows='auto')
    ERP_plots = evokeds.plot(spatial_colors = True, gfp = True) 
    evokeds_clean = epochs.average()
    all_times = np.arange(0, 0.5, 0.01)
    topo_plots = evokeds_clean.plot_topomap(all_times, ch_type='eeg', time_unit='s', ncols=8, nrows='auto')
    ERP_plots = evokeds_clean.plot(spatial_colors = True, gfp = True) 
    evokeds_all.append(evokeds_clean)
    #topo_plots.savefig(Save_folder_fig +  str(f.parts[-1][0:9]) +'_TOPO' ) 
    #ERP_plots.savefig(Save_folder_fig +  str(f.parts[-1][0:9]) + '_ERP') 
    #epochs_clean.save(save_folder_epochs + str(f.parts[-1][0:9]) + '_manually' + '_epo.fif', overwrite = True, split_size='2GB')


    # Compensating for phase effect
    # https://www.biorxiv.org/content/10.1101/2021.10.01.462795v1
    epochs_amp_mod = np.concatenate((epochs._data[:,:,0:1000], epochs._data[:,:,1001:] - epochs._data[:,:,0:1000]), axis=2)
    epochs_phase_effect = mne.EpochsArray(data = epochs_amp_mod,  info = epochs.info, events = epochs.events, event_id = epochs.event_id, tmin=-1, on_missing='ignore')
    evokeds_pahse_effecft = epochs_phase_effect.average()
    evokeds_pahse_effecft_all.append(evokeds_pahse_effecft)






Evoked_GrandAv = mne.grand_average(evokeds_all)    
Evoked_GrandAv.info['bads'] = ['TP9', 'FT9',  'FT7', 'T7', 'TP7',  'P7', 'F7', 'AF7', 'F5', 'PO7']
Evoked_GrandAv = Evoked_GrandAv.interpolate_bads(reset_bads=True, mode='accurate')
topo_steps = Evoked_GrandAv.plot_topomap(np.arange(0, 0.6, 0.01), ch_type='eeg', time_unit='s', ncols=8, nrows='auto') 
Evoked_GrandAv_crop = Evoked_GrandAv.crop(-0.1, 0.6)
Evoked_GrandAv_c = Evoked_GrandAv.plot_joint(times='peaks')
Evoked_GrandAv.plot(gfp=True, spatial_colors=True)

Evoked_GrandAv_c.savefig(Save_folder_fig +  'all_subs' + '_ERP') 
topo_steps.savefig(Save_folder_fig +  'all_subs' + '_ToPo') 
mne.evoked.write_evokeds(save_folder_epochs +'all_subs' + '_ave.fif', Evoked_GrandAv, overwrite=True) 

Evoked_GrandAv_phase = mne.grand_average(evokeds_pahse_effecft_all)    
Evoked_GrandAv_phase.info['bads'] = ['TP9', 'FT9', 'TP7', 'FT7', 'T7', 'F7']
Evoked_GrandAv_phase = Evoked_GrandAv_phase.interpolate_bads(reset_bads=True, mode='accurate')
topo_steps_ph = Evoked_GrandAv_phase.plot_topomap(np.arange(0, 0.6, 0.01), ch_type='eeg', time_unit='s', ncols=8, nrows='auto') 
Evoked_GrandAv_crop_ph = Evoked_GrandAv_phase.crop(-0.1, 0.6)
Evoked_GrandAv_phase.plot_joint(times='peaks')
Evoked_GrandAv_phase.plot(gfp=True, spatial_colors=True)



Evoked_GrandAv_crop.plot(gfp=False, spatial_colors=False)
Evoked_GrandAv_phase.crop(0, 0.4).plot(gfp=True, spatial_colors=False, picks =  ['Fz', 'Cz', 'FC1', 'FC2', 'CP1', 'F1', 'C1', 'C2', 'CPz'])



Evoked_GrandAv_c = Evoked_GrandAv.plot_joint(times='peaks')
Evoked_GrandAv_phase = Evoked_GrandAv_crop_ph.plot_joint(times='peaks')


Evoked_GrandAv_crop_ph.plot(gfp=True, spatial_colors=False)


Evoked_GrandAv_crop_ph.plot_joint(times='peaks', picks =  ['Fz', 'Cz', 'FC1', 'FC2', 'CP1', 'F1', 'C1', 'C2', 'CPz'])
Evoked_GrandAv_crop.plot_joint(times='peaks', picks =  ['Fz', 'Cz', 'FC1', 'FC2', 'CP1', 'F1', 'C1', 'C2', 'CPz'])



mne.viz.plot_compare_evokeds(Evoked_GrandAv_crop_ph, picks= ['Fz', 'Cz', 'FC1', 'FC2', 'CP1', 'F1', 'C1', 'C2', 'CPz'], combine='mean')











