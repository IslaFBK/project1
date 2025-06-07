#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 20:16:48 2021

@author: shni2598
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows, freqz


import matplotlib as mpl
mpl.use('Agg')
from scipy.stats import sem
#import load_data_dict
import mydata
import brian2.numpy_ as np
from brian2.only import *
#import post_analysis as psa
import firing_rate_analysis as fra
import frequency_analysis as fqa
import fano_mean_match
import connection as cn
import pickle
import sys
import os
import matplotlib.pyplot as plt
import shutil
#%%

data_dir = 'raw_data/'
#save_dir = 'mean_results/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
analy_type = 'state'
#datapath = data_dir
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/twoarea_hz/on_off/test_longstim2/'+data_dir

data = mydata.mydata()
loop_num = 0
data.load(datapath+'data%d.file'%loop_num)

#%%
data.a1.param.stim1.stim_on
#%%
'''spon'''
first_stim = 0; last_stim = 0
start_time = data.a1.param.stim1.stim_on[first_stim,0] - 2000
end_time = data.a1.param.stim1.stim_on[last_stim,0] 
#%%
start_time = 10000
end_time = 11000

data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 15)
data.a1.ge.get_centre_mass()
data.a1.ge.overlap_centreandspike()

data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 15)
data.a2.ge.get_centre_mass()
data.a2.ge.overlap_centreandspike()

#data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ni, window = 10)
#frames = int(end_time - start_time)
frames = data.a1.ge.spk_rate.spk_rate.shape[2]

# stim_on_off = data.a1.param.stim1.stim_on-start_time
# stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]

# #stim = [[[[31.5,31.5]], [stim_on_off], [[6]*stim_on_off.shape[0]]],None]
# stim = [[[[31.5,31.5],[63.5,-0.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]],None]

# adpt = None
# #adpt = [None, [[[31,31]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
# #adpt = [[[[31,31]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
title = ''
ani = fra.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, spkrate2=data.a2.ge.spk_rate.spk_rate, \
                                        frames = frames, start_time = start_time, interval_movie=15, anititle=title,stim=None, adpt=None)
#savetitle = title.replace('\n','')

moviefile = savetitle+'_spon_%d'%loop_num+'.mp4'

#ani.save(moviefile)

#%%
            value1.set_array(spkrate1[:,:,i])
            #value2.set_array(spk2[:,:,i])
            title.set_text(u"time: {} ms".format(time_title[i]))
            # if show_pattern_size:
            #     circle_pattern.set_center([pattern_ctr[i,1],pattern_ctr[i,0]])
            #     circle_pattern.set_radius(pattern_size[i])
            #     print('size')
            #     return value1, title,circle_pattern#, value2
            if show_pattern_size:
                circle_pattern.center = [pattern_ctr[i,1],pattern_ctr[i,0]]
                circle_pattern.radius = pattern_size[i]
                if stim is not None:
                    detect_event(i, 1, stim, loca_n_stim, all_done_stim, current_trial_stim, \
                             trial_state_stim, current_state_stim, circle_stim, max_trial_stim)                
                    # for ax_i in range(1):
                    #     if stim[ax_i] is not None:
                    #         #loca_n, max_trial, current_state, current_trial, trial_state, all_done, stim_circle = ax_init_loc_stim[ax_i]
                    #         for n in range(loca_n_stim[ax_i]):
                    #             if not all_done_stim[ax_i][n]:
                    #                 if i == stim[ax_i][1][n][current_trial_stim[ax_i][n],trial_state_stim[ax_i][n]]:
                    #                     if current_state_stim[ax_i][n] == False:
                    #                         #stim_circle[n].center = [pattern_ctr[i,1],pattern_ctr[i,0]]
                    #                         circle_stim[ax_i][n].radius = stim[ax_i][2][n][current_trial_stim[ax_i][n]]#pattern_size[i]
                    #                         trial_state_stim[ax_i][n] += 1
                    #                         current_state_stim[ax_i][n] = True
                    #                         print(ax_i,current_trial_stim[ax_i][n],trial_state_stim[ax_i][n],current_state_stim[ax_i][n],all_done_stim[ax_i][n])
                    #                     else:
                    #                         circle_stim[ax_i][n].radius = 0
                    #                         trial_state_stim[ax_i][n] -= 1
                    #                         current_trial_stim[ax_i][n] += 1
                    #                         current_state_stim[ax_i][n] = False
                    #                         if max_trial_stim[ax_i][n] == current_trial_stim[ax_i][n]:
                    #                             all_done_stim[ax_i][n] = True
                    #                             print(ax_i,n,'all done')
                    #                             print(ax_i,all_done_stim[ax_i][n])
                    #                         print(ax_i,current_trial_stim[ax_i][n],trial_state_stim[ax_i][n],current_state_stim[ax_i][n],all_done_stim[ax_i][n])
                    #                         #sleep(1)
                if adpt is not None:
                    detect_event(i, 1, adpt, loca_n_adpt, all_done_adpt, current_trial_adpt, \
                                 trial_state_adpt, current_state_adpt, circle_adpt, max_trial_adpt)                
                    # for ax_i in range(1):
                    #     if adpt[ax_i] is not None:
                    #         #loca_n, max_trial, current_state, current_trial, trial_state, all_done, stim_circle = ax_init_loc_stim[ax_i]
                    #         for n in range(loca_n_adpt[ax_i]):
                    #             if not all_done_adpt[ax_i][n]:
                    #                 if i == adpt[ax_i][1][n][current_trial_adpt[ax_i][n],trial_state_adpt[ax_i][n]]:
                    #                     if current_state_adpt[ax_i][n] == False:
                    #                         #stim_circle[n].center = [pattern_ctr[i,1],pattern_ctr[i,0]]
                    #                         circle_adpt[ax_i][n].radius = adpt[ax_i][2][n][current_trial_adpt[ax_i][n]]#pattern_size[i]
                    #                         trial_state_adpt[ax_i][n] += 1
                    #                         current_state_adpt[ax_i][n] = True
                    #                         print(ax_i,current_trial_adpt[ax_i][n],trial_state_adpt[ax_i][n],current_state_adpt[ax_i][n],all_done_adpt[ax_i][n])
                    #                     else:
                    #                         circle_adpt[ax_i][n].radius = 0
                    #                         trial_state_adpt[ax_i][n] -= 1
                    #                         current_trial_adpt[ax_i][n] += 1
                    #                         current_state_adpt[ax_i][n] = False
                    #                         if max_trial_adpt[ax_i][n] == current_trial_adpt[ax_i][n]:
                    #                             all_done_adpt[ax_i][n] = True
                    #                             print(ax_i,n,'all done')
                    #                             print(ax_i,all_done_adpt[ax_i][n])
                    #                         print(ax_i,current_trial_adpt[ax_i][n],trial_state_adpt[ax_i][n],current_state_adpt[ax_i][n],all_done_adpt[ax_i][n])
                if stim is not None or adpt is not None:
                    return (value1, *circle_all, circle_pattern, title) #, circle_adpt[ax_i][1]
                else: return value1, title, circle_pattern,
                
            else:
                if stim is not None:
                    detect_event(i, 1, stim, loca_n_stim, all_done_stim, current_trial_stim, \
                             trial_state_stim, current_state_stim, circle_stim, max_trial_stim)                
                if adpt is not None:                    
                    detect_event(i, 1, adpt, loca_n_adpt, all_done_adpt, current_trial_adpt, \
                                 trial_state_adpt, current_state_adpt, circle_adpt, max_trial_adpt)                
                if stim is not None or adpt is not None:
                    return (value1, *circle_all, title)
                else:
                    return value1, title,
        
        #value1.set_clim(vmin=0, vmax=6)
        # ax1.axis('off')
        #if stimu_onset >= 0: fig.suptitle('onset of stimulus:%dms'%(stimu_onset))
        fig.suptitle(anititle)
        ani=animation.FuncAnimation(fig, updatev,  frames=frames, interval=interval_movie, blit=True)    # frames=spk1.shape[2]
        return ani
    else:
        
#%%
#%%
start_time = 10500
end_time = 11000

data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 15)
data.a1.ge.get_centre_mass()
data.a1.ge.overlap_centreandspike()

data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 15)
data.a2.ge.get_centre_mass()

# data.a1.ge.get_centre_mass()
# data.a2.ge.get_centre_mass()
#%%
plt_fig_num = -2
def plot_pattern(data1, data2, centre_ind_1, centre_ind_2, plt_fig_num ):
    fig, ax = plt.subplots(1,2)
    #fig.suptitle('sensory to association strength: %.2f\nassociation to sensory strength: %.2f'
    #             %(scale_e_12[i],scale_e_21[j]))
    #ax1.set_title('sensory')
    #ax2.set_title('association')
    cmap_spk=plt.cm.get_cmap('Blues', 7) # viridis Blues
    camp_adapt = np.array([138/255,43/255,226/255,1.])
    cmap_c = np.array([1.,0.,0.,1.])
    cmap_stimulus = np.array([88/255,150/255.,0.,1.])
    cmap = np.vstack((camp_adapt,cmap_stimulus,cmap_c,cmap_spk(range(7))))
    cmap = mpl.colors.ListedColormap(cmap)
    #cmap.set_under('grey')
    bounds = np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]) - 0.5
    
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cbaxes = fig.add_axes([0.2, 0.15, 0.6, 0.03]) 
    
    cb = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap,
                                    norm=norm,
                                    boundaries=bounds,
                                    ticks=np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]),#+0.5,
                                    spacing='proportional',
                                    orientation='horizontal') #horizontal vertical
    cb.ax.set_xticklabels(['adpt','stim','ctr', 0, 1, 2, 3, 4, 5, 6])
    cb.set_label('number of spikes')
    
    titleaxes = fig.add_axes([0.3, 0.85, 0.4, 0.05])
    titleaxes.axis('off')
    title = titleaxes.text(0.5,0.05, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=titleaxes.transAxes, ha="center")
    #time_title = np.arange(data.a1.ge.spk_rate.spk_rate.shape[2]) + start_time
    
    value1=ax[0].matshow(data1, cmap=cb.cmap, norm=cb.norm)
    value2=ax[1].matshow(data2, cmap=cb.cmap, norm=cb.norm)
    ax[0].axis('off')
    ax[1].axis('off')
    
    centre_ind_1 = centre_ind_1.copy()
    centre_ind_1[:,0] = 64 - 1 - centre_ind_1[:,0]
    
    centre_ind_2 = centre_ind_2.copy()
    centre_ind_2[:,0] = 64 - 1 - centre_ind_2[:,0]
    
    ax[0] = plot_traj(ax[0], centre_ind_1[:plt_fig_num+1], corner = np.array([[-0.5,-0.5],[63.5, 63.5]]), coordinate_type_is_matrix=True)
    ax[1] = plot_traj(ax[1], centre_ind_2[:plt_fig_num+1], corner = np.array([[-0.5,-0.5],[63.5, 63.5]]), coordinate_type_is_matrix=True)
    
    ax[0].set_xlim([-0.5,63.5])
    ax[0].set_ylim([63.5,-0.5])
    ax[1].set_xlim([-0.5,63.5])
    ax[1].set_ylim([63.5,-0.5])
    ax[0].set_title('sens')
    ax[1].set_title('asso')
    return fig, ax 
#%%
#%%
start_time = 10100
end_time = 11000

data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 15)
data.a1.ge.get_centre_mass()
data.a1.ge.overlap_centreandspike()

data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 15)
data.a2.ge.get_centre_mass()

#%%
frames = data.a1.ge.spk_rate.spk_rate.shape[2]

#stim_on_off = data.a1.param.stim1.stim_on-start_time
#stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]

#stim = [[[[31.5,31.5],[-0.5,63.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]]]
#stim = [[[[31.5,31.5]], [stim_on_off], [[6]*stim_on_off.shape[0]]],None]
#stim = [[[[31.5,31.5]], [[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]], [[6]]],None]
stim = None

#adpt = [None, [[[31.5,31.5]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
adpt = None
title = '\nspontaneous activity'
ani = fra.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, spkrate2=data.a2.ge.spk_rate.spk_rate, \
                                        frames = frames, start_time = start_time, interval_movie=15, anititle=title,stim=stim, adpt=adpt)
savetitle = title.replace('\n','')

moviefile = 'spon.mp4'

# if loop_num%1 == 0:
ani.save(moviefile)
#%%
plt_fig_num = -2
fig, ax = plot_pattern(data.a1.ge.spk_rate.spk_rate[:,:,plt_fig_num], data.a2.ge.spk_rate.spk_rate[:,:,plt_fig_num], \
             data.a1.ge.centre_mass.centre_ind, data.a2.ge.centre_mass.centre_ind, plt_fig_num=plt_fig_num)    
    
fig.suptitle('\nspontaneous activity')
fig.savefig('spon.png')
#%%
data.a1.param.stim1.stim_on
#%%
start_time = 21000
end_time = 21900

data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 15)
data.a1.ge.get_centre_mass()
data.a1.ge.overlap_centreandspike()

data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 15)
data.a2.ge.get_centre_mass()
#%%
frames = data.a1.ge.spk_rate.spk_rate.shape[2]

#stim_on_off = data.a1.param.stim1.stim_on-start_time
#stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]

#stim = [[[[31.5,31.5],[-0.5,63.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]]]
#stim = [[[[31.5,31.5]], [stim_on_off], [[6]*stim_on_off.shape[0]]],None]
stim = [[[[31.5,31.5]], [[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]], [[6]]],None]
#stim = None

#adpt = [None, [[[31.5,31.5]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
adpt = None
title = '\nstimulus at centre of sensory area'
ani = fra.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, spkrate2=data.a2.ge.spk_rate.spk_rate, \
                                        frames = frames, start_time = start_time, interval_movie=15, anititle=title,stim=stim, adpt=adpt)
savetitle = title.replace('\n','')

moviefile = 'sti_noatt.mp4'

# if loop_num%1 == 0:
ani.save(moviefile)
#%%
plt_fig_num = -2
fig, ax = plot_pattern(data.a1.ge.spk_rate.spk_rate[:,:,plt_fig_num], data.a2.ge.spk_rate.spk_rate[:,:,plt_fig_num], \
             data.a1.ge.centre_mass.centre_ind, data.a2.ge.centre_mass.centre_ind, plt_fig_num=plt_fig_num)    
    
fig.suptitle('\nstimulus at centre of sensory area')

circle = plt.Circle([31.5,31.5],6, lw=1,color='g',fill=False,alpha=None)
ax[0].add_patch(circle)
fig.savefig('sti_noatt.png')
#%%
#%%
start_time = 324430
end_time = 325230

data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 15)
data.a1.ge.get_centre_mass()
data.a1.ge.overlap_centreandspike()

data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 15)
data.a2.ge.get_centre_mass()
#%%
frames = data.a1.ge.spk_rate.spk_rate.shape[2]

#stim_on_off = data.a1.param.stim1.stim_on-start_time
#stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]

#stim = [[[[31.5,31.5],[-0.5,63.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]]]
#stim = [[[[31.5,31.5]], [stim_on_off], [[6]*stim_on_off.shape[0]]],None]
#stim = [[[[31.5,31.5]], [[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]], [[6]]],None]
stim = None

adpt = [None, [[[31.5,31.5]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
#adpt = None
title = '\nattention at centre of association area'
ani = fra.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, spkrate2=data.a2.ge.spk_rate.spk_rate, \
                                        frames = frames, start_time = start_time, interval_movie=15, anititle=title,stim=stim, adpt=adpt)
savetitle = title.replace('\n','')

moviefile = 'nosti_att.mp4'

# if loop_num%1 == 0:
ani.save(moviefile)
#%%
plt_fig_num = -2
fig, ax = plot_pattern(data.a1.ge.spk_rate.spk_rate[:,:,plt_fig_num], data.a2.ge.spk_rate.spk_rate[:,:,plt_fig_num], \
             data.a1.ge.centre_mass.centre_ind, data.a2.ge.centre_mass.centre_ind, plt_fig_num=plt_fig_num)    
    
fig.suptitle('\nattention at centre of association area')

circle_stim = plt.Circle([31.5,31.5],6, lw=1,color='g',fill=False,alpha=None)
circle_adpat = plt.Circle([31.5,31.5],7, lw=1,color='tab:purple',fill=True,alpha=0.2) #'tab:purple'

ax[1].add_patch(circle_adpat)
fig.savefig('nosti_att.png')
#%%
#%%
start_time = 326450
end_time = 327350

data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 15)
data.a1.ge.get_centre_mass()
data.a1.ge.overlap_centreandspike()

data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 15)
data.a2.ge.get_centre_mass()
#%%
frames = data.a1.ge.spk_rate.spk_rate.shape[2]

#stim_on_off = data.a1.param.stim1.stim_on-start_time
#stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]

#stim = [[[[31.5,31.5],[-0.5,63.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]]]
#stim = [[[[31.5,31.5]], [stim_on_off], [[6]*stim_on_off.shape[0]]],None]
stim = [[[[31.5,31.5]], [[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]], [[6]]],None]
#stim = None

adpt = [None, [[[31.5,31.5]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
#adpt = None
title = 'stimulus at centre of sensory area\nattention at centre of association area'
ani = fra.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, spkrate2=data.a2.ge.spk_rate.spk_rate, \
                                        frames = frames, start_time = start_time, interval_movie=15, anititle=title,stim=stim, adpt=adpt)
savetitle = title.replace('\n','')

moviefile = 'sti_att.mp4'

# if loop_num%1 == 0:
ani.save(moviefile)
#%%
plt_fig_num = -2
fig, ax = plot_pattern(data.a1.ge.spk_rate.spk_rate[:,:,plt_fig_num], data.a2.ge.spk_rate.spk_rate[:,:,plt_fig_num], \
             data.a1.ge.centre_mass.centre_ind, data.a2.ge.centre_mass.centre_ind, plt_fig_num=plt_fig_num)    
    
fig.suptitle('stimulus at centre of sensory area\nattention at centre of association area')

circle_stim = plt.Circle([31.5,31.5],6, lw=1,color='g',fill=False,alpha=None)
circle_adpat = plt.Circle([31.5,31.5],7, lw=1,color='tab:purple',fill=True,alpha=0.2) #'tab:purple'

ax[0].add_patch(circle_stim)
ax[1].add_patch(circle_adpat)
fig.savefig('sti_att.png')
#%%
# stim = [[np.array([[31,31],[-0.5,-0.5]]), [np.array([[0,200],[300,400]]), np.array([0,200])], [[6,6],[6]]],\
#         [np.array([[31,31],[-0.5,-0.5]]), [np.array([[0,200],[300,400]]), np.array([0,200])], [[6,6],[6]]]]
# stim = [[np.array([[31,31],[-0.5,-0.5]]), [np.array([[0,200],[300,400]]), np.array([0,200])], [[6,6],[6]]],\
#         None]
# adpt = [None,\
#         [np.array([[31,31],[-0.5,-0.5]]), [np.array([[100,200],[400,500]]), np.array([100,200])], [[6,6],[6]]]]
#ax_init_loc_stim = [None]*2
if stim is not None:
    loca_n_stim, max_trial_stim, current_state_stim, current_trial_stim, trial_state_stim, all_done_stim, circle_stim = \
        [None]*2, [None]*2, [None]*2, [None]*2, [None]*2, [None]*2, [None]*2 
    for ax_i in range(2):
        #*ax_init_loc_stim[ax_i], ax[ax_i] = init_loca(ax[ax_i], stim[ax_i], 'g')
        loca_n_stim[ax_i], max_trial_stim[ax_i], current_state_stim[ax_i], current_trial_stim[ax_i], \
        trial_state_stim[ax_i], all_done_stim[ax_i], circle_stim[ax_i], ax[ax_i] = init_loca(ax[ax_i], stim[ax_i], 'g')
    circle_all = []
    for cir in circle_stim:
        if cir is not None:
            circle_all += cir
if adpt is not None:
    loca_n_adpt, max_trial_adpt, current_state_adpt, current_trial_adpt, trial_state_adpt, all_done_adpt, circle_adpt = \
        [None]*2, [None]*2, [None]*2, [None]*2, [None]*2, [None]*2, [None]*2 
    for ax_i in range(2):
        #*ax_init_loc_adpt[ax_i], ax[ax_i] = init_loca(ax[ax_i], stim[ax_i], 'g')
        loca_n_adpt[ax_i], max_trial_adpt[ax_i], current_state_adpt[ax_i], current_trial_adpt[ax_i], \
        trial_state_adpt[ax_i], all_done_adpt[ax_i], circle_adpt[ax_i], ax[ax_i] = init_loca(ax[ax_i], adpt[ax_i], 'tab:purple', lw=0, fill=True, alpha=0.2)
    if 'circle_all' not in locals():
        circle_all = []
    for cir in circle_adpt:
        if cir is not None:
            circle_all += cir
                    
                    
#%%
first_stim = 2*n_perStimAmp -1 + n_perStimAmp*n_StimAmp; last_stim = 2*n_perStimAmp + n_perStimAmp*n_StimAmp
start_time = data.a1.param.stim1.stim_on[first_stim,0] - 300
end_time = data.a1.param.stim1.stim_on[last_stim,0] + 500

#start_time = 
data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 15)
data.a1.ge.get_centre_mass()
data.a1.ge.overlap_centreandspike()

data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 15)
data.a2.ge.get_centre_mass()
data.a2.ge.overlap_centreandspike()

#data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ni, window = 10)
#frames = int(end_time - start_time)
frames = data.a1.ge.spk_rate.spk_rate.shape[2]

stim_on_off = data.a1.param.stim1.stim_on-start_time
stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]

#stim = [[[[31.5,31.5],[-0.5,63.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]]]
#stim = [[[[31.5,31.5]], [stim_on_off], [[6]*stim_on_off.shape[0]]],None]
stim = [[[[31.5,31.5],[63.5,-0.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]],None]

adpt = [None, [[[31.5,31.5]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
#adpt = None
ani = fra.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, spkrate2=data.a2.ge.spk_rate.spk_rate, \
                                        frames = frames, start_time = start_time, interval_movie=15, anititle=title,stim=stim, adpt=adpt)
savetitle = title.replace('\n','')

moviefile = savetitle+'_att_hipt_%d'%loop_num+'.mp4'

# if loop_num%1 == 0:
ani.save(moviefile)
#%%
clr = plt.rcParams['axes.prop_cycle'].by_key()['color']

#%%
def plot_traj(ax, centre_ind, corner = np.array([[-0.5,-0.5],[62.5, 62.5]]), coordinate_type_is_matrix=True):
    '''
    plot trajactory; periodic boundary is considered
    ax: input plot ax
    centre_ind: coordinates of points of trajactory; N*2 array
    corner: the down-left and upper-right corner of grid
    coordinate_type_is_matrix: if the format of coordinates uses the convention of matrix coordinate
    '''
    #wd = width; ht = height
    #corner = np.array([[-0.5,-0.5],[wd-0.5, ht-0.5]])
    
    centre_ind = np.copy(centre_ind) 
    # copy the centre_ind to avoid changing the value of original centre_ind out of the function 
    # if don't do that, the original centre_ind out of the function will be changed, not sure if it is a bug
    wd = corner[1,0]-corner[0,0]; ht = corner[1,1]-corner[0,1]

    if coordinate_type_is_matrix:
        centre_ind[:,0] = ht - 1 - centre_ind[:,0]
        centre_ind = np.flip(centre_ind, 1)
    else: pass

    #ax.plot(centre_ind[0,0], centre_ind[0,1], 'og', label='start')
    
    for i in range (len(centre_ind)-1):
        
        if abs(centre_ind[i,0]-centre_ind[i+1,0])>(wd/2) and abs(centre_ind[i,1]-centre_ind[i+1,1])>(ht/2):           
            
            if np.sign(centre_ind[i,1]-centre_ind[i+1,1]) == np.sign(centre_ind[i,0]-centre_ind[i+1,0]):        
                tr = int(centre_ind[i,0] > centre_ind[i+1,0])
                upright = tr*centre_ind[i] + (1-tr)*centre_ind[i+1] 
                downleft = (1-tr)*centre_ind[i] + tr*centre_ind[i+1]
                if (downleft[1] + ht - upright[1])/(downleft[0] + wd - upright[0]) > (corner[1,1] - upright[1])/(corner[1,0] - upright[0]):
                    x = [[upright[0], upright[0], upright[0]-wd],  
                         [downleft[0]+wd, downleft[0]+wd, downleft[0]]]
                    y = [[upright[1], upright[1]-ht, upright[1]-ht],
                         [downleft[1]+ht, downleft[1], downleft[1]]]
                else:
                    x = [[upright[0], upright[0]-wd, upright[0]-wd],  
                         [downleft[0]+wd, downleft[0], downleft[0]]]
                    y = [[upright[1], upright[1], upright[1]-ht],
                         [downleft[1]+ht, downleft[1]+ht, downleft[1]]]
                ax.plot(x,y,c=clr[3], lw=0.5)           
            else:   
                tr = int(centre_ind[i,0] < centre_ind[i+1,0])
                upleft = tr*centre_ind[i] + (1-tr)*centre_ind[i+1]
                downright = (1-tr)*centre_ind[i] + tr*centre_ind[i+1]
                if abs((downright[1] + ht -upleft[1])/(downright[0] - wd - upleft[0])) > abs((corner[1,1] - upleft[1])/(corner[0,0] - upleft[0])):
                    x = [[upleft[0], upleft[0], upleft[0]+wd],  
                         [downright[0]-wd, downright[0]-wd, downright[0]]]
                    y = [[upleft[1], upleft[1]-ht, upleft[1]-ht],
                         [downright[1]+ht, downright[1], downright[1]]]
                else:
                    x = [[upleft[0], upleft[0]+wd, upleft[0]+wd],  
                         [downright[0]-wd, downright[0], downright[0]]]
                    y = [[upleft[1], upleft[1], upleft[1]-ht],
                         [downright[1]+ht, downright[1]+ht, downright[1]]]                    
                ax.plot(x,y,c=clr[3], lw=0.5)
                
        elif abs(centre_ind[i,0]-centre_ind[i+1,0])>(wd/2):
            
            if centre_ind[i,0] > centre_ind[i+1,0]:
                ax.plot([centre_ind[i,0]-wd, centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]],c=clr[3], lw=0.5)
                ax.plot([centre_ind[i,0], centre_ind[i+1,0]+wd], [centre_ind[i,1], centre_ind[i+1,1]],c=clr[3], lw=0.5)
            else:
                ax.plot([centre_ind[i,0], centre_ind[i+1,0]-wd], [centre_ind[i,1], centre_ind[i+1,1]],c=clr[3], lw=0.5)
                ax.plot([centre_ind[i,0]+wd, centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]],c=clr[3], lw=0.5)
        
        elif abs(centre_ind[i,1]-centre_ind[i+1,1])>(ht/2):
            if centre_ind[i,1]-centre_ind[i+1,1]>0:
                ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1]-ht, centre_ind[i+1,1]],c=clr[3], lw=0.5)
                ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]+ht],c=clr[3], lw=0.5)
            else:
                ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]-ht],c=clr[3], lw=0.5)
                ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1]+ht, centre_ind[i+1,1]],c=clr[3], lw=0.5)
        else:
            ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]],c=clr[3], lw=0.5)
    ax.scatter(centre_ind[i+1,0], centre_ind[i+1,1], s=2, c=clr[3])#,label='end')
    #ax.legend()
    return ax