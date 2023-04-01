#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:22:08 2022

@author: sara
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 11:41:37 2022

@author: sara
"""






def permutation_cluster_peak_vs_trough_new(peaks, adjacency_mat, thresholds, freq_band):
        # in this function, peaks is a 5 dim matrix with dims, nsubj, nchans, npeaks, nphas, nfreq
    import mne.stats
    
    # reduce dimensions by averaging over target frequencies and phases
    
    if freq_band == 'all':
    
        mean_peaks_freq = np.mean(peaks, ( -1))
        
    else:
        
        mean_peaks_freq = peaks
    
    #first row : peak - trough
    # According to mne "The first dimension should correspond to the difference between paired samples (observations) in two conditions. "
    mean_peaks_phase = mean_peaks_freq[:, :, :, 0] -  mean_peaks_freq[:, :, :, 1]
    
    mean_peaks_phase = np.mean(mean_peaks_freq, ( -1))
    # get matrix dimensions
    nsubj, nchans, npeaks, = np.shape(mean_peaks_phase)
    nperm = 100
    clusters = []
    mask = np.zeros([nchans, npeaks])
    max_cluster_size = np.zeros([nperm+1, npeaks])
    thresholds=thresholds
    # get the original cluster size during the first loop
    # perform 1000 random permutations (sign flipping) and each time determine the size of the biggest cluster

   
    for p in range(npeaks):
        mean_peaks_phase = (mean_peaks_freq[:, :, p, 0]) -  (mean_peaks_freq[:, :, p, 1])
        cluster = mne.stats.permutation_cluster_1samp_test((mean_peaks_phase), out_type='mask',
                                                           adjacency=adjacency_mat, threshold=thresholds[p],
                                                           n_permutations=1000)
        t_sum = np.zeros([len(cluster[1])])
        # get the sum of the tvalues for each of the 
        # clusters to choose the main cluster 
        # (take magnitude to treat negative and positive cluster equally)
        for c in range(len(cluster[1])):
            t_sum[c] = np.abs(sum(cluster[0][cluster[1][c]]))
            #t_sum[c] = sum(cluster[0][cluster[1][c]])

        # store the maximal cluster size for each iteration 
        # to later calculate p value
        # if no cluster was found, put in 0
        if len(t_sum) > 0:
            max_cluster_size[ p] = np.max(t_sum)
   
                # get the channels which are in the main cluster
            mask[:,p] = cluster[1][np.argmax(t_sum)]
        else:
            max_cluster_size[ p] = 0
            
            


        clusters.append(cluster)

    return clusters, mask










def permutation_cluster(peaks, adjacency_mat, thresholds):
        # in this function, peaks is a 5 dim matrix with dims, nsubj, nchans, npeaks, nphas, nfreq
    import mne.stats
    # reduce dimensions by averaging over target frequencies and phases
    mean_peaks = np.mean(peaks, (-2, -1))
    # get matrix dimensions
    nsubj, nchans, npeaks = np.shape(mean_peaks)
    clusters = []
    mask = np.zeros([nchans, npeaks])
    max_cluster_size = np.zeros([npeaks])
    thresholds=thresholds
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
            max_cluster_size[p] = np.max(t_sum)
            # save the original cluster information (1st iteration) 


                # get the channels which are in the main cluster
            mask[:,p] = cluster[1][np.argmax(t_sum)]
        else:
            max_cluster_size[p] = 0
            
            
        clusters.append(cluster)   


    
    return clusters, mask

def permutation_cluster_peak_and_trough_against_zero(peaks, adjacency_mat, thresholds):
    import mne.stats
    # reduce dimensions by averaging over target frequencies and phases
    mean_peaks = np.mean(peaks, (-1))
    # get matrix dimensions
    nsubj, nchans, npeaks = np.shape(mean_peaks)
    clusters = []
    mask = np.zeros([nchans, npeaks])
    max_cluster_size = np.zeros([npeaks])
    thresholds=thresholds
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
            max_cluster_size[p] = np.max(t_sum)
            # save the original cluster information (1st iteration) 


                # get the channels which are in the main cluster
            mask[:,p] = cluster[1][np.argmax(t_sum)]
        else:
            max_cluster_size[p] = 0
            
            
        clusters.append(cluster)   
    return clusters, mask   











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
                                  cmap = 'jet',
                                  axes=axes,
                                  outlines = "head",
                                  mask=mask,
                                  mask_params=maskparam,
                                  size=0.5,
                                  vmin = np.min(ch_attribute),
                                  vmax = np.max(ch_attribute))
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


def Select_Epochs_peak_trough(epochs, freq, phase):
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
    





def plot_topomap_peaks_second_v(peaks_tval, mask,  pos,ch_names,  clim):

    import matplotlib.pyplot as plt
    plt.style.use('default')
    nplots =1 
    nchans, npeaks = np.shape(peaks_tval)

    maskparam = dict(marker='.', markerfacecolor='k', markeredgecolor='k',
                linewidth=0, markersize=5)

    fig, sps = plt.subplots(nrows=nplots, ncols=npeaks, figsize=(8, 6))
    
    
    
    for iplot in range(nplots):
        for ipeak in range(npeaks):

            if mask is not None:
                imask=mask[:,ipeak]
            else:
                imask = None

            im = topoplot_2d (  ch_names, peaks_tval[ :, ipeak], pos,
                                clim=clim, axes=sps[ipeak], 
                                mask=imask, maskparam=maskparam)

    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    cb = plt.colorbar(im[0],  ax = sps, fraction=0.01, pad=0.04)
    cb.ax.tick_params(labelsize=12)
    
                 

    
    plt.show()
    return fig, sps, cb


def two_phases_each_freq(thresholds, peaks, phase_1, phase_2, labels):
    
    bin_sel_1 = int(phase_1/45)
    bin_sel_2 = int(phase_2/45)
    
    
    mask = {}
    clusters = {}
    cluster_pv_freq =  np.zeros([len(unique_freqs), len(labels)])
    for ifreq, freq in enumerate(unique_freqs):
    
        clusters[str(ifreq)], mask[str(ifreq)] = permutation_cluster_peak_vs_trough_new(peaks[:, :, :, [bin_sel_1,bin_sel_2], ifreq], adjacency_mat, thresholds , freq_band = 'f{freq}' )
    
        allclusters = np.zeros([nchans, npeaks])
        # get the t values for each of the peaks for plotting the topoplots
        for p in range(len(clusters[str(ifreq)])):
            allclusters[:,p] = clusters[str(ifreq)][p][0]
            
        # set all other t values to 0 to focus on clusters
        allclusters[mask[str(ifreq)]==False] = 0
        ch_names = epochs.ch_names
        cluster_pv = np.zeros([len(clusters[str(ifreq)])])
        for p in range(len(clusters[str(ifreq)])):
            peaks_tval[:,p] = clusters[str(ifreq)][p][0]
            if len(clusters[str(ifreq)][p][2]) >1:
                cluster_pv[p] = min(clusters[str(ifreq)][p][2])
            elif len(clusters[str(ifreq)][p][2]) ==1:
                cluster_pv[p] = clusters[str(ifreq)][p][2]
            else:
                cluster_pv[p] = 0
                
                
            #if  cluster_pv[p] < 0.05 :
            #    cluster_pv_freq[ifreq, p] = cluster_pv[p]
            cluster_pv_freq[ifreq, p] = cluster_pv[p]
            
    return(clusters, ch_names, mask,  cluster_pv_freq, phase_1, phase_2, labels)

            
     


def plot_two_phases_each_freq(thresholds, peaks,  phase_1, phase_2,  labels, ch_names):
    clusters, ch_names, mask,  cluster_pv_freq, phase_1, phase_2, labels = two_phases_each_freq(thresholds, peaks, phase_1, phase_2, labels)
    
    maskparam=None
    plt.style.use('default')   
    # https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
    fig, axes = plt.subplots(nrows=len(labels), ncols=10, figsize=(20,8))
    fig.suptitle(f'cluster {phase_1} vs {phase_2}', fontsize = 18, fontweight="bold")
    cols = [format(col) for col in np.arange(0,10,1)]
    rows = ['{}'.format(row) for row in [ 'P30\n\n','N60\n\n', 'N120\n\n', 'P190\n\n']]
    
         
    for i_row,_ in enumerate(labels):
    
        for ax, col in zip(axes[i_row], cols):
            ax.set_title(f'{np.multiply(int(col) +1, 4)} Hz', size =18, fontweight="bold")   
            im = topoplot_2d (ch_names, clusters[str(col)][i_row][0], pos, 
                                         clim=[-5,5], axes=ax, 
                                         mask=mask[str(col)][:,i_row], maskparam=maskparam)
        cb = plt.colorbar(im[0],  ax = ax, fraction=0.03, pad=0.04)
        cb.ax.tick_params(labelsize=12)
    
    
        for i_l, l in enumerate(np.where(cluster_pv_freq[:,i_row]>0)[0]):
        
            if l < 10:
                row = i_row; col =l
            axes[row, (col)].set_xlabel(f' P = {cluster_pv_freq[(col), i_row]}')
            if cluster_pv_freq[(col), i_row] < 0.050:
                axes[row, (col)].xaxis.label.set_color('red')
                
                
        for ax, row in zip(axes[:,0], rows):
           ax.set_ylabel(row, rotation=0, size=14, labelpad = 50, fontweight="bold")
        
    plt.tight_layout()
        
  
    return(fig)

    

def one_freq_all_phases(peaks, frequency, phase_to_choose,  labels):

    mask = {}
    clusters = {}
    phases = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    cluster_pv_phase =  np.zeros([len(unique_phases_cosine), len(labels)])
    
    ifrequency = int((frequency/4)-1 ) 
    i_ph_peak = int(phase_to_choose/45)
    
    for iphase, phase in enumerate(phases):
    
        clusters[str(iphase)], mask[str(iphase)] = permutation_cluster_peak_vs_trough_new(peaks[:, :, :, [i_ph_peak, iphase], ifrequency], adjacency_mat,  thresholds= thresholds, freq_band = frequency)
        allclusters = np.zeros([nchans, npeaks])
        # get the t values for each of the peaks for plotting the topoplots
        for p in range(len(clusters[str(iphase)])):
            allclusters[:,p] = clusters[str(iphase)][p][0]
            
        # set all other t values to 0 to focus on clusters
        allclusters[mask[str(iphase)]==False] = 0
        cluster_pv = np.zeros([len(clusters[str(iphase)])])
        for p in range(len(clusters[str(iphase)])):
            peaks_tval[:,p] = clusters[str(iphase)][p][0]
            if len(clusters[str(iphase)][p][2]) >1:
                cluster_pv[p] = min(clusters[str(iphase)][p][2])
            elif len(clusters[str(iphase)][p][2]) ==1:
                cluster_pv[p] = clusters[str(iphase)][p][2]
            else:
                cluster_pv[p] = 0

            cluster_pv_phase[iphase, p] = cluster_pv[p]
    
    return(clusters, mask,  cluster_pv_phase, frequency)





def plot_one_freq_all_phase(thresholds, peaks, frequency, phase_to_choose, labels, ch_names):
    
    
    clusters,  mask,  cluster_pv_phase, frequency = one_freq_all_phases(peaks, frequency, phase_to_choose,  labels)
    i_ph_peak = int(phase_to_choose/45)
    maskparam=None
    fig, axes = plt.subplots(nrows=len(labels), ncols=7,  constrained_layout = True, figsize=(20,8))
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    fig.suptitle(f'Real-time Analysis for {frequency} Hz', fontsize = 18, fontweight="bold")
    cols = [format(col) for col in np.delete( np.arange(0,8,1), i_ph_peak, 0)]
    rows = ['{}'.format(row) for row in [ 'P30\n\n','N60\n\n', 'N120\n\n', 'P190\n\n']]
    
    
    if phase_to_choose == 0:
         phases = np.array([0, 0, 45, 90, 135, 180, 225, 270, 315])
    elif phase_to_choose == 45:
         phases = np.array([45, 0, 45, 90, 135, 180, 225, 270, 315])
    elif phase_to_choose == 90:
         phases = np.array([90, 0, 45, 90, 135, 180, 225, 270, 315])
    elif phase_to_choose == 135:
         phases = np.array([135, 0, 45, 90, 135, 180, 225, 270, 315])
    elif phase_to_choose == 180:
         phases = np.array([180, 0, 45, 90, 135, 180, 225, 270, 315])
    elif phase_to_choose == 225:
         phases = np.array([225, 0, 45, 90, 135, 180, 225, 270, 315])   
    elif phase_to_choose == 270:
         phases = np.array([270, 0, 45, 90, 135, 180, 225, 270, 315])  
    elif phase_to_choose == 315:
         phases = np.array([315, 0, 45, 90, 135, 180, 225, 270, 315])
    
        
    for i_row,_ in enumerate(labels):
        for ax, col in zip(axes[i_row], cols):
            ax.set_title(f'{phases[0]} vs {phases[int(col) +1]} ', size=12, fontweight="bold") 
            im = topoplot_2d (ch_names, clusters[str(col)][i_row][0], pos = epochs.info, clim=[-5,+5], axes=ax, 
                                         mask=mask[str(col)][:,i_row], maskparam=maskparam)
        # colorbar
        cax = inset_axes(ax, width="8%",  height="100%",  loc='lower left', bbox_to_anchor=(1., -0.2, 1, 1),
                        bbox_transform=ax.transAxes, borderpad=+2)
        cb= fig.colorbar(im[0], cax=cax)    
        cb.set_label('t-value', weight='bold', rotation=90) 
    
    
        for i_l, l in enumerate(cols):
            row = i_row; col = i_l
            if cluster_pv_phase[int(l), i_row] != 0:
                axes[row, (i_l)].set_xlabel(f' P = {cluster_pv_phase[int(l), i_row]}', weight='bold')
            if cluster_pv_phase[int(l), i_row] < 0.050:
                axes[row, (i_l)].xaxis.label.set_color('red')
        
    for ax, row in zip(axes[:,0], rows):
        ax.set_ylabel(row, rotation=0, size=14, labelpad = 50, fontweight="bold")
        
    plt.tight_layout()
    return(fig)
    
    
  


def phase_phase180_mv(peaks, frequency, phase_to_choose, labels, ch_names, ERP_num):
    
    ifrequency = int((frequency/4)-1 ) 
    i_ph_peak = int(phase_to_choose/45)
    
    ##########
    
    
    phases = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    
    mean_peaks_sub = np.mean(peaks, (0))
    mean_peaks_peak = {}
    mean_peaks_trough = {}
    mean_peaks_diff = {}
    
    for iphase, phase in enumerate(phases):
        
        mean_peaks_peak[str(iphase)] = mean_peaks_sub[:, :,  i_ph_peak, ifrequency] 
        mean_peaks_trough[str(iphase)] = mean_peaks_sub[:, :,  iphase, ifrequency]
        mean_peaks_diff[str(iphase)] = mean_peaks_peak[str(iphase)] - mean_peaks_trough[str(iphase)]
   
    
    
    clim = [ 0, 6]; mask=None; maskparam=None
    clim_diff = [3, -3]
    fig, axes = plt.subplots(nrows=3, ncols=7, figsize=(20,8))
    fig.suptitle(f'Analysis for {frequency} Hz', fontsize = 18, fontweight="bold")
    cols = [format(col) for col in np.delete( np.arange(0,8,1), i_ph_peak, 0)]
    #rows = ['{}'.format(row) for row in [ 'Phase1\n','Phase2\n\n ', 'Phase1\n - \n phase2']]
    
    
    if phase_to_choose == 0:
         phases = np.array([0, 0, 45, 90, 135, 180, 225, 270, 315])
    elif phase_to_choose == 45:
         phases = np.array([45, 0, 45, 90, 135, 180, 225, 270, 315])
    elif phase_to_choose == 90:
         phases = np.array([90, 0, 45, 90, 135, 180, 225, 270, 315])
    elif phase_to_choose == 135:
         phases = np.array([135, 0, 45, 90, 135, 180, 225, 270, 315])
    elif phase_to_choose == 180:
         phases = np.array([180, 0, 45, 90, 135, 180, 225, 270, 315])
    elif phase_to_choose == 225:
         phases = np.array([225, 0, 45, 90, 135, 180, 225, 270, 315])   
    elif phase_to_choose == 270:
         phases = np.array([270, 0, 45, 90, 135, 180, 225, 270, 315])  
    elif phase_to_choose == 315:
         phases = np.array([315, 0, 45, 90, 135, 180, 225, 270, 315])
    
    
    for ax, col in zip(axes[0], cols):
        ax.set_title(f'{phases[0]} ', size =18, fontweight="bold") 
        im = topoplot_2d (ch_names, mean_peaks_peak[str(col)][:,ERP_num], pos, # ERP number
                                     clim=clim, axes=ax, 
                                     mask=mask, maskparam=maskparam)
    # colorbar
    cax = inset_axes(ax,
                 width="8%",  # width = 10% of parent_bbox width
                 height="100%",  # height : 50%
                 loc='lower left',
                 bbox_to_anchor=(1.05, -1.65, 1, 2),
                 bbox_transform=ax.transAxes,
                 borderpad=+2,
                 )
    cb= fig.colorbar(im[0], cax=cax)    
    cb.set_label('\u03bcV', weight='bold', rotation=90)    
    
    for ax, col in zip(axes[1], cols):
        ax.set_title(f'{phases[int(col) +1]} ', size =18, fontweight="bold")   
        im = topoplot_2d (ch_names, mean_peaks_trough[str(col)][:,ERP_num], pos,  # ERP number
                                     clim=clim, axes=ax, 
                                     mask=mask, maskparam=maskparam)
    
    
    for ax, col in zip(axes[2], cols):
        ax.set_title(f'{phases[0]} - {phases[int(col) +1]} ', size =18, fontweight="bold")   
        im = topoplot_2d (ch_names, mean_peaks_diff[str(col)][:,ERP_num], pos, # ERP number
                                     clim=clim_diff, axes=ax, 
                                     mask=mask, maskparam=maskparam)
    
    # colorbar
        ax = axes[2][-1]
        cax = inset_axes(ax,
                 width="8%",  # width = 10% of parent_bbox width
                 height="100%",  # height : 50%
                 loc='lower left',
                 bbox_to_anchor=(1.05, -0.2, 1, 1),
                 bbox_transform=ax.transAxes,
                 borderpad=+2,
                 )
        
        cb= fig.colorbar(im[0], cax=cax)    
        cb.set_label('\u03bcV', weight='bold', rotation=90)   
        
        
    return(fig)
    

def cluster_one_freq_phase_180(thresholds, peaks, frequency, labels, ch_names):
    mask = {}
    clusters = {}
    ifrequency = int((frequency/4) -1) 
    cluster_pv_phase =  np.zeros([len(unique_phases_cosine), len(labels)])
    phases = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    
    for iphase, phase in enumerate(phases[0:4]):
      
        clusters[str(iphase)], mask[str(iphase)] = permutation_cluster_peak_vs_trough_new(peaks[:, :, :, [iphase, iphase+4], ifrequency], adjacency_mat,  thresholds= thresholds, freq_band = frequency  )
    
       
        allclusters = np.zeros([nchans, npeaks])
        # get the t values for each of the peaks for plotting the topoplots
        for p in range(len(clusters[str(iphase)])):
            allclusters[:,p] = clusters[str(iphase)][p][0]
            
        # set all other t values to 0 to focus on clusters
        allclusters[mask[str(iphase)]==False] = 0
        ch_names = epochs.ch_names
        cluster_pv = np.zeros([len(clusters[str(iphase)])])
        for p in range(len(clusters[str(iphase)])):
            peaks_tval[:,p] = clusters[str(iphase)][p][0]
            if len(clusters[str(iphase)][p][2]) >1:
                cluster_pv[p] = min(clusters[str(iphase)][p][2])
            elif len(clusters[str(iphase)][p][2]) ==1:
                cluster_pv[p] = clusters[str(iphase)][p][2]
            else:
                cluster_pv[p] = 0
                
        
            cluster_pv_phase[iphase, p] = cluster_pv[p]
            
            
    maskparam=None
    fig, axes = plt.subplots(nrows=4, ncols=4,  constrained_layout = True, figsize=(20,8))
    #plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    #fig.suptitle(f'Real-time Analysis for {frequency} Hz', fontsize = 18, fontweight="bold")
    cols = [format(col) for col in np.delete( np.arange(0,5,1), 4, 0)]
    rows = ['{}'.format(row) for row in [ 'P30\n\n','N60\n\n', 'N120\n\n', 'P190\n\n']]
    
    phases = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    
    for i_row,_ in enumerate(labels):   
        
        for ax, col in zip(axes[i_row], cols):
            ax.set_title(f'{phases[int(col)]} vs {phases[int(col) +4]} ', size=16, fontweight="bold") 
            im = topoplot_2d (ch_names, clusters[str(col)][i_row][0], pos = epochs.info,
                                         clim=[-5,+5], axes=ax, 
                                         mask=mask[str(col)][:,i_row], maskparam=maskparam)
        # colorbar
        cax = inset_axes(ax,
                     width="8%",  # width = 10% of parent_bbox width
                     height="100%",  # height : 50%
                     loc='lower left',
                     bbox_to_anchor=(1.05, -0.2, 1, 1),
                     bbox_transform=ax.transAxes,
                     borderpad=+2,
                     )
        cb= fig.colorbar(im[0], cax=cax)    
        cb.set_label('t-value', weight='bold', rotation=90) 
        
        
   
            
            
        for i_l, l in enumerate(cols):

            row = i_row; col =i_l
            print(i_l, i_row, cols)
    
            if cluster_pv_phase[int(l), i_row] != 0:
                axes[row, (i_l)].set_xlabel(f' P = {cluster_pv_phase[ i_l, i_row]}', weight='bold')    
            if cluster_pv_phase[int(l), i_row] < 0.05:
                axes[row, (i_l)].xaxis.label.set_color('red')
    
                
    
    
        for ax, row in zip(axes[:,0], rows):
            ax.set_ylabel(row, rotation=0, size=18, labelpad = 50, fontweight="bold")
            
        #plt.tight_layout()
    return(fig, clusters, cluster_pv_phase, mask)
    






def figure_4(peaks, clusters, cluster_pv_phase, frequency,  labels, ch_names, ERP_num):
    ifrequency = int((frequency/4)-1)
    phases = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    mean_peaks_sub = np.mean(peaks, (0))
    mean_peaks_peak = {}
    mean_peaks_trough = {}
    mean_peaks_diff = {}
    mean_peaks_diff_inv = {}
    
    for iphase, phase in enumerate(phases[0:4]):
        mean_peaks_peak[str(iphase)] = mean_peaks_sub[:, :,  iphase, ifrequency] 
        for i, i1 in enumerate(np.argwhere(mean_peaks_peak[str(iphase)] > 4)):
            mean_peaks_peak[str(iphase)][i1[0], i1[1]] = 2

        
        mean_peaks_trough[str(iphase)] = mean_peaks_sub[:, :,  iphase+4, ifrequency]


        
        
        mean_peaks_diff[str(iphase)] = mean_peaks_peak[str(iphase)] -  mean_peaks_trough[str(iphase)]
        mean_peaks_diff_inv[str(iphase)] =  mean_peaks_trough[str(iphase)] - mean_peaks_peak[str(iphase)]
        
        # This is way of interpolating channels woth artifacts for oular channels, does not effect P190 significan channels
        for i, i1 in enumerate(np.argwhere(mean_peaks_peak[str(iphase)] > 3.7)):
            mean_peaks_peak[str(iphase)][i1[0], i1[1]] = 3
            
        for i, i1 in enumerate(np.argwhere(mean_peaks_trough[str(iphase)] > 3.7)):
            mean_peaks_trough[str(iphase)][i1[0], i1[1]] = 3
        
    clim=[0, 5]; maskparam=None

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20,8))
    fig.suptitle(f'Analysis for {frequency} Hz', fontsize = 18, fontweight="bold")
    cols = [format(col) for col in np.delete( np.arange(0,5,1), 4, 0)]
    
    
            
    
        
    for ax, col in zip(axes[2], cols):
        ax.set_title(f'{phases[int(col)]}\N{degree sign} vs {phases[int(col)+4 ]}\N{degree sign} ', size=16, fontweight="bold")   
        im = topoplot_2d (ch_names, clusters[str(col)][ERP_num][0],  pos = epochs.info,
                                     clim=[-5,+5], axes=ax, 
                                     mask=mask[str(col)][:,ERP_num], maskparam=maskparam)
    for i_l, l in enumerate(cols):
    
        row = 2 ; col =i_l
        if cluster_pv_phase[int(l), ERP_num] != 0:
            axes[row, (i_l)].set_xlabel(f' P = {cluster_pv_phase[int(l), ERP_num]}', weight='bold')    
        if cluster_pv_phase[int(l), ERP_num] < 0.05:
            axes[row, (i_l)].xaxis.label.set_color('red')
        
        
            # colorbar
        cax = inset_axes(ax,
                     width="8%",  # width = 10% of parent_bbox width
                     height="100%",  # height : 50%
                     loc='lower left',
                     bbox_to_anchor=(1.05, -0.2, 1, 1),
                     bbox_transform=ax.transAxes,
                     borderpad=+2,
                     )
        cb= fig.colorbar(im[0], cax=cax)    
        cb.set_label('t-value', weight='bold', rotation=90)    
    
            
    for ax, col in zip(axes[1], cols):
        ax.set_title(f'{phases[int(col) +4]}\N{degree sign}', size =18, fontweight="bold")   
        im = topoplot_2d (ch_names, mean_peaks_trough[str(col)][:,ERP_num], pos,  # ERP number
                                     clim=clim, axes=ax, 
                                     mask=None, maskparam=maskparam)
    
    
       
    
    for ax, col in zip(axes[0], cols):
        ax.set_title(f'{phases[int(col)]}\N{degree sign}   ', size =18, fontweight="bold") 
        im = topoplot_2d (ch_names, mean_peaks_peak[str(col)][:,ERP_num], pos, # ERP number
                                     clim=clim, axes=ax, 
                                     mask=None, maskparam=maskparam)
        
        
    cax = inset_axes(ax,
             width="8%",  # width = 10% of parent_bbox width
             height="100%",  # height : 50%
             loc='lower left',
             bbox_to_anchor=(1.05, -0.8, 1, 1),
             bbox_transform=ax.transAxes,
             borderpad=+2,
             )
    cb= fig.colorbar(im[0], cax=cax)    
    cb.set_label('\u03bcV', weight='bold', rotation=90)  
    plt.tight_layout()
    return(fig)














#%%
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import phase_analysis_function_Jan as ph_analysis
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import mne.stats
import mne



"""files that have been used for this analysis: Subject UtLi was deleted completely
    BaJo_R001, BrTi_R001, EuSa_R001, HaSa_R001-HaSa_R002(concat), HeDo_R010, KoAl_R008, LaCa_R001, LiLu_R001
    MeAm_R001-MeAm_R002, MiAn_R001, Nema_R001, RaLu_R001, RuMa_R001, ScSe_R001, StLa_R001, VeVa_R001, WoIr_R001
    ZaHa_R001, ZhJi_R001-ZhJi_R002(concat), ZiAm_R001"""
    


exdir = "/home/sara/NMES/analyzed data/Sep_repeat/Epochs_NMES_manually_rejected/"
save_folder_fig = '/home/sara/NMES/analyzed data/Sep_repeat/Clusters/'
files = Path(exdir).glob('*epo.fif*')
plt.close('all')




# Setting window length for each component
win_erp0 = [25, 40]   #P30
win_erp1 = [50, 75]   #N60
win_erp2 = [125, 160] #N120
win_erp3 = [170, 210] #P190
thresholds=[2.8, 2.8, 3.5, 3.2]
labels = ['P30', 'N60', 'N120', 'P190']# The components
win_erps = np.array([win_erp0, win_erp1, win_erp2, win_erp3])



peaks_tval = np.zeros([64,len(labels)])
unique_freqs = np.arange(4, 44, 4) 
unique_phases_cosine = np.arange(0, 360, 45) 
peaks = np.zeros([20, 64, len(labels), len(unique_phases_cosine), len(unique_freqs)]) #[subs, channels, components, phases, freqs]

   



for ifiles, f in enumerate(files):
    epochs = mne.read_epochs(f, preload=True).copy().pick_types( eeg=True)
    # removing the effect of phase amp according to Granö et al. 2022.
    # amp after stim - amp before stim     
    epochs_amp_mod = epochs._data[:,:,1001:] - epochs._data[:,:,0:1000]
    # channels based on clustered channels. Only using those because the size of this variable will be very large. 
    # making mne epoch structure
    epochs = mne.EpochsArray(data = epochs_amp_mod,  info = epochs.info, events = epochs.events, event_id = epochs.event_id, on_missing='ignore')
 
    
    

    erp_byfreqandphase = {} 
    peaks_byfreqandphase = {}
    epochs_byfreqandphase = {} 
    for ifreq, freq in enumerate(unique_freqs):
        erp_byfreqandphase[str(freq)]  = {}
        peaks_byfreqandphase[str(freq)] = {} 
        epochs_byfreqandphase[str(freq)] = {}
        for iphase, phase in enumerate(unique_phases_cosine):
            sel_idx = ph_analysis.Select_Epochs(epochs, freq, phase)
            epochs_byfreqandphase[str(freq)][str(phase)] = epochs[sel_idx]
            erp_byfreqandphase[str(freq)][str(phase)]  = epochs_byfreqandphase[str(freq)][str(phase)].average()          
            for ipeak, peak in enumerate(labels):
                if ipeak == 0:                                  #P30-->  max
                    peaks_byfreqandphase[str(freq)][str(phase)] = np.max((erp_byfreqandphase[str(freq)][str(phase)]._data[:,win_erps[0,0]:win_erps[0,1]]),1)
                elif ipeak == 1:                                #N60-->  min
                    peaks_byfreqandphase[str(freq)][str(phase)] = np.min((erp_byfreqandphase[str(freq)][str(phase)]._data[:,win_erps[1,0]:win_erps[1,1]]),1)
                elif ipeak == 2:                                #N120--> max
                    peaks_byfreqandphase[str(freq)][str(phase)] = np.max((erp_byfreqandphase[str(freq)][str(phase)]._data[:,win_erps[2,0]:win_erps[2,1]]),1)
                elif ipeak == 3:                                #P190--> max
                    peaks_byfreqandphase[str(freq)][str(phase)] = np.max((erp_byfreqandphase[str(freq)][str(phase)]._data[:,win_erps[3,0]:win_erps[3,1]]),1)
               # To remove none arrays after selecting epochs
                if str(erp_byfreqandphase[str(freq)][str(phase)].comment) == str(''):
                    peaks_byfreqandphase[str(freq)][str(phase)] = np.zeros(64) 
                else:
                    peaks[ifiles, :, ipeak, iphase, ifreq] = peaks_byfreqandphase[str(freq)][str(phase)] 
        
        
mask = {}
pvals = {}
clusters = {}
adjacency_mat,_ = mne.channels.find_ch_adjacency(epochs_byfreqandphase[str(freq)][str(phase)].info , 'eeg')
nsubj, nchans, npeaks, nphas, nfreqs = np.shape(peaks)   
pos = epochs.info

mne.EvokedArray(data= erp_byfreqandphase[str(16)][str(45)].data, info = epochs.info).plot(spatial_colors=True)
#%% This cell does a cluster-based permutation by averaging over all frequencies, phases can be chosen. 
thresholds= [2, 2, 2, 2]
clusters, mask = permutation_cluster_peak_vs_trough_new(peaks[:,:, :, [0,4],: ], adjacency_mat ,thresholds= thresholds, freq_band = 'all'   )

allclusters = np.zeros([nchans, npeaks])
# get the t values for each of the peaks for plotting the topoplots
for p in range(len(clusters)):
    allclusters[:,p] = clusters[p][0]    
# set all other t values to 0 to focus on clusters
allclusters[mask==False] = 0
ch_names = epochs.ch_names
cluster_pv = np.zeros([len(clusters)])
for p in range(len(clusters)):
    peaks_tval[:,p] = clusters[p][0]
    if len(clusters[p][2]) >1:
        cluster_pv[p] = min(clusters[p][2]) 
# min p value, because in the function maximum t_sum was chose
    elif len(clusters[p][2]) ==1:
        cluster_pv[p] = clusters[p][2]
    else:
        cluster_pv[p] = 0
        
        
                
fig, sps, cb = plot_topomap_peaks_second_v(peaks_tval, mask, pos,  ch_names, [-5,5])        
fig.suptitle('All Frequencies and Phase Peak Vs trough ', fontsize = 14)
sps[0].title.set_text(f' \n\n {labels[0]}\n\n TH = {thresholds[0]} \n\n pv = {cluster_pv[0]}')
sps[1].title.set_text(f' \n\n {labels[1]}\n\n TH = {thresholds[1]} \n\n  pv = {cluster_pv[1]}')
sps[2].title.set_text(f' \n\n {labels[2]}\n\n TH = {thresholds[2]} \n\n pv = {cluster_pv[2]}')
sps[3].title.set_text(f' \n\n {labels[3]}\n\n TH = {thresholds[3]} \n\n pv = {cluster_pv[3]}')
cb.set_label('t-value', rotation = 90)
fig.savefig(save_folder_fig + 'all_freq_0_vs_180.svg')


#%% This function gets two phases and compares them in all target frequencies

###########################################################################
          #f'cluster {phase_1} vs {phase_2}'          
       #4Hz  8Hz  12Hz  16Hz  20Hz  24Hz  28Hz  32Hz  36Hz  40Hz
# P30
# N60
# N120
# P190

############################################################################
thresholds= [2, 2, 2, 2]
phase_1 = 45
phase_2 = 225

fig = plot_two_phases_each_freq(thresholds, peaks, phase_1, phase_2, labels, ch_names)
fig.savefig(save_folder_fig + f'{phase_1}' + '_vs_' + f'{phase_2}' +'.svg')


#%% This function compares one phase to all the phases for one frequency


###########################################################################
          #f'cluster for{frequency}'          
       #f'{phase}' vs 0        f'{phase}' vs 45         f'{phase}' vs 90        f'{phase}' vs 135       f'{phase}' vs 180        f'{phase}' vs 225      f'{phase}' vs 270        f'{phase}' vs 315  
# P30
# N60
# N120
# P190

############################################################################
thresholds= [2, 2, 2, 2]
phase_to_choose = 45
frequency = 16

    
fig = plot_one_freq_all_phase(thresholds, peaks, frequency, phase_to_choose, labels, ch_names)
fig.savefig(save_folder_fig + f'{phase_to_choose}' + '_vs_all_' + f'{frequency}'+ 'Hz' +'.svg')


#%% This function diffrentiate one phase to all the phases for one frequency


###########################################################################
          # values are in micro volt          
       #f'{phase}'           f'{phase}'             f'{phase}'          f'{phase}'              f'{phase}'             f'{phase}'           f'{phase}'           f'{phase}' 
       #    0                   45                     90                   135                    180                    225                  270                 315 
       #f'{phase}'-0         f'{phase}'-45          f'{phase}'-90       f'{phase}'-135          f'{phase}'-180         f'{phase}'-225       f'{phase}'-270       f'{phase}'-315 
# P30
# N60
# N120
# P190
###########################################################################
frequency = 16
phase_to_choose = 45
ERP_num = labels.index("P190")
fig = phase_phase180_mv(peaks, frequency, phase_to_choose, labels, ch_names, ERP_num)
fig.savefig(save_folder_fig + f'{phase_to_choose}' + '_vs_' + str(int(f'{phase_to_choose}') +180 ) +'_' + f'{frequency}'+ 'Hz' +'_micro_volt' +'.svg')

#%% This function cluster between phase vs phase+180 for one frequency for all components[0, 45, 90, 135 vs (0, 45, 90, 135)+180] 

############################################################################
          #f'cluster for{frequency}'          
       #f'{phase}' vs 0        f'{phase}' vs 45         f'{phase}' vs 90        f'{phase}' vs 135       f'{phase}' vs 180        f'{phase}' vs 225      f'{phase}' vs 270        f'{phase}' vs 315  
# P30
# N60
# N120
# P190
##########################################################################

thresholds= [2, 2, 2, 2]
frequency = 16
fig, clusters, cluster_pv_phase, mask = cluster_one_freq_phase_180(thresholds, peaks, frequency,  labels, ch_names)
fig.savefig(save_folder_fig +'0+ 180' +'_' + f'{frequency}'+ 'Hz' +'cluster' +'.svg')



#%% combination of cluster and difference together
# https://physoc.onlinelibrary.wiley.com/doi/full/10.1113/JP278638, Figure 4


ERP_num = labels.index("P190")
frequency = 16

fig = figure_4(peaks, clusters, cluster_pv_phase, frequency,  labels, ch_names, ERP_num)
fig.savefig(save_folder_fig +'figure_4_Desideri' +'.svg')


#%%

erp2_16_ch = []
for i, i_ch in enumerate(np.where(mask[str(3)][:,1] == 1)[0]):
    erp2_16_ch.append(ch_names[i_ch])

ERP2_chan =['Fp1','Fp2','Fz','Cz','FC1','FC2','CP1','F1','C1','C2','AF3','AF7','Fpz','CPz']

# compare two lists
common_list = set(erp2_16_ch).intersection(ERP2_chan)
print(len(common_list), common_list)





























