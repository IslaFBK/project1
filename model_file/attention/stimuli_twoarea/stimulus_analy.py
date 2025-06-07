#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 08:55:13 2020

@author: shni2598
"""
#%%
import scipy.io as sio
from brian2.only import *
import matplotlib.pyplot as plt
import brian2.numpy_ as np
import post_analysis as psa
#%%
file_num = 0
timepath = '/import/headnode1/shni2598/brian2/data/twoarea_stimuli/timev1%s.mat'%file_num
indexpath = '/import/headnode1/shni2598/brian2/data/twoarea_stimuli/indexv1%s.mat'%file_num
#%%
def importmatdata(timepath, indexpath, file_num, area):
    
    class spk_data:
        def __init__(self, spk_dict):
            for key in spk_dict:
                setattr(self, key, spk_dict[key])  
#    file_num = 0
    data = sio.loadmat(timepath%(area,file_num))
    time = data['time'][0]; del data
    time = time*0.1*ms
    data = sio.loadmat(indexpath%(area, file_num))
    index = data['ind'][0]; del data
    
    spk = spk_data({'t':time,'i':index})
    
    return spk
#%%
area = 1; file_num = 168; 
timepath = '/import/headnode1/shni2598/brian2/data/twoarea_stimuli3/timev%d%s.mat'#%(area,file_num)
indexpath = '/import/headnode1/shni2598/brian2/data/twoarea_stimuli3/indexv%d%s.mat'#%(area,file_num)
#area = 2
#timepath = '/import/headnode1/shni2598/brian2/data/twoarea_stimuli/timev%d%s.mat'%(area,file_num)
#indexpath = '/import/headnode1/shni2598/brian2/data/twoarea_stimuli/indexv%d%s.mat'%(area,file_num)

#%%
i = 2; j = 5
file_num = i*13+j
spkv1 = psa.importmatdata(timepath, indexpath, file_num=file_num, area=1)
spkv2 = psa.importmatdata(timepath, indexpath, file_num=file_num, area=2)

#%%
spkrate1 = psa.get_spike_rate(spkv1, start_time=500*ms, end_time=3000*ms, indiv_rate = True, popu_rate = False, \
                   sample_interval = 1*ms, n_neuron = 3969, window = 10*ms, dt = 0.1*ms)

spkrate2 = psa.get_spike_rate(spkv2, start_time=500*ms, end_time=3000*ms, indiv_rate = True, popu_rate = False, \
                   sample_interval = 1*ms, n_neuron = 3969, window = 10*ms, dt = 0.1*ms)


psa.show_pattern(spkrate1, spkrate2, area_num = 2)
#%%
i = 5; j = 3
file_num = i*13+j
spkv1 = psa.importmatdata(timepath, indexpath, file_num=file_num, area=1)
spkv2 = psa.importmatdata(timepath, indexpath, file_num=file_num, area=2)
#%%
spkrate1, centre_ind1, jump_size1, jump_dist1 = psa.get_rate_centre_jumpdist(spkv1, \
                             starttime=1000*ms, endtime=3200*ms, binforrate=10*ms, sample_interval=1*ms, \
                             slide_interval = 1, jump_interval = 15, dt=0.1*ms, \
                             show_trajectory=False)

spkrate2, centre_ind2, jump_size2, jump_dist2 = psa.get_rate_centre_jumpdist(spkv2, \
                             starttime=1000*ms, endtime=3200*ms, binforrate=10*ms, sample_interval=1*ms, \
                             slide_interval = 1, jump_interval = 15, dt=0.1*ms, \
                             show_trajectory=False)
#%%
ani=psa.show_pattern(spkrate1, spkrate2=spkrate2, area_num = 2, bottom_up=scale_e_12[i], top_down=scale_e_21[j])
#%%
plt.matshow(spkrate1[:,:,0])


#%%
scale_e_12 = np.concatenate([np.array([0]),np.arange(0.2,1.4,0.1)])
scale_e_21 = np.concatenate([np.array([0]),np.arange(0.2,1.4,0.1)])
#%%
fig,[ax1,ax2] = plt.subplots(1,2,figsize=[10,5])
psa.plot_traj(ax1, centre_ind1)
psa.plot_traj(ax2, centre_ind2)
ax1.set_xlim([-0.5,62.5])
ax1.set_ylim([-0.5,62.5])
ax2.set_xlim([-0.5,62.5])
ax2.set_ylim([-0.5,62.5])
fig.suptitle('bottom-up: %.1f*default top-down: %.1f*default'%(scale_e_12[i], scale_e_21[j]))
ax1.set_title('sensory'); ax2.set_title('association')
#%%
plt.savefig('%.1f_%.1f.png'%(scale_e_12[i], scale_e_21[j]))


