#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 17:40:45 2020

@author: shni2598
"""
import matplotlib as mpl
mpl.use('Agg')
#import load_data_dict
import mydata
import brian2.numpy_ as np
from brian2.only import *
#import post_analysis as psa
import firing_rate_analysis
import frequency_analysis as fa
import connection as cn
import pickle
import sys
import os
import matplotlib.pyplot as plt
import shutil
#%%
stim_loc = np.array([0,0])

dist = cn.coordination.lattice_dist(data.a1.param.e_lattice,62,stim_loc)
dist_bin = np.arange(0,31.5*2**0.5,2.5)
n_in_bin = [None]*(dist_bin.shape[0]-1)
neuron = np.arange(data.a1.param.Ne)

for i in range(len(dist_bin)-1):
    n_in_bin[i] = neuron[(dist >= dist_bin[i]) & (dist < dist_bin[i+1])]


simu_time_tot = 28000

N_stim = data.a1.param.stim.stim_amp_scale.shape[0]

data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])
#%%
fr_bin = np.array([150, 200, 250]) 

hz_loc = np.zeros([N_stim,len(dist_bin)-1,len(fr_bin)])
n_per_stim = 5

       
#%%
for b in range(len(fr_bin)):
    for i in range(N_stim):
        for j in range(len(n_in_bin)):
        #data.a1.ge.spk_matrix[]
    
            hz_loc[i, j, b] = data.a1.ge.spk_matrix[n_in_bin[j], data.a1.param.stim.stim_on[i,0]*10:(data.a1.param.stim.stim_on[i,0]+fr_bin[b])*10].sum()/n_in_bin[j].shape[0]
#%%
hz_spon = np.zeros([len(n_in_bin)])


for j in range(len(n_in_bin)):
    hz_spon[j] = data.a1.ge.spk_matrix[n_in_bin[j], 5000*10:10000*10].sum()/n_in_bin[j].shape[0]


#%%
#data_anly.hz_loc = hz_loc


for b in range(len(fr_bin)):
    hz_loc_tmp = hz_loc[:,:,b].reshape(6,5,-1)
    
    hz_loc_mean = hz_loc_tmp.mean(1)
    #plt.matshow(hz_loc_mean)#[:,0,:])
    
    stim_amp = np.unique(data.a1.param.stim.stim_amp_scale)*200
    
    plt.figure()
    for i in range(hz_loc_mean.shape[0]):
        plt.plot(hz_loc_mean[i], '-o', label = 'stim_amp: %.1f Hz'%(stim_amp[i]))
        plt.title('counts bins: %.1f ms'%fr_bin[b])
    plt.plot(hz_spon, '-o', label = 'stim_amp: 0 Hz')
    plt.legend()
#%%
spk_mua = data.a1.ge.spk_matrix[n_in_bin[0],:].sum(0).A[0]
#%%
spk_mua_mean = np.convolve(spk_mua, np.ones(100),'same')/0.01/len(n_in_bin[0])
#%%
stim_t = np.vstack((data.a1.param.stim.stim_on[:,0],data.a1.param.stim.stim_on[:,0]))

stim_t_2 =  np.vstack((np.ones(len(data.a1.param.stim.stim_on[:,0])), np.ones(len(data.a1.param.stim.stim_on[:,0]))*100))

#%%
plt.figure()
plt.plot(spk_mua_mean[::10])
plt.plot(stim_t, stim_t_2)
#%%
sua = data.a1.ge.spk_matrix[2000, :].A[0]
sua_mean = np.convolve(sua, np.ones(100),'same')/0.01
plt.figure()
plt.plot(sua_mean[::10])
plt.plot(stim_t, stim_t_2)
#%%
plt.figure()
plt.plot(sua.nonzero()[0]/10,np.ones(sua.nonzero()[0].shape),'|')
plt.plot(stim_t, stim_t_2)
#%%

dist_bin = np.arange(0, 31.5*2**0.5, 2.5)
hz_loc_all = np.zeros([N_stim,len(dist_bin)-1,len(fr_bin)])
hz_spon_all = np.zeros(len(dist_bin)-1)
#%%
analy_type = 'stim2'
#datapath = ''
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/spon_1area/'
#sys_argv = int(sys.argv[1])
for loop_num in range(81): #rep_ind*20 + ie_num
    data_anly = mydata.mydata()
    data_anly.load(datapath+'data_anly%d.file'%loop_num)
    hz_loc_all += data_anly.hz_loc
    hz_spon_all += data_anly.hz_spon

hz_loc_all /= 81
hz_spon_all /= 81
#%%
dist_bin = np.arange(0, 31.5*2**0.5, 2.5)
dist_bin_plot = (dist_bin+(dist_bin[1]-dist_bin[0])*0.5)[:-1]

for b in range(len(fr_bin)):
    hz_loc_tmp = hz_loc_all[:,:,b].reshape(6,5,-1)
    
    hz_loc_mean = hz_loc_tmp.mean(1)
    #plt.matshow(hz_loc_mean)#[:,0,:])
    
    stim_amp = np.unique(data.a1.param.stim.stim_amp_scale)*200
    
    plt.figure(figsize=[8,6])
    plt.plot(dist_bin_plot, hz_spon_all/5, '-o', label = 'stim_amp: 0 Hz')
    for i in range(hz_loc_mean.shape[0]):
        plt.plot(dist_bin_plot, hz_loc_mean[i]/(fr_bin[b]/1000), '-o', label = 'stim_amp: %.1f Hz'%(stim_amp[i]))
        plt.title('average firing rate versus distance to stimulus centre; no top-down\n Spike-count bin: %.1f ms\nie_ratio: default'%fr_bin[b])
        plt.xlim([dist_bin[0],dist_bin[-1]])
        plt.xlabel('distance')
        plt.ylabel('Hz')
    
    plt.legend()
#%%
num_per_bin = np.array([n.shape[0] for n in n_in_bin])

np.sum(hz_spon_all*(num_per_bin/num_per_bin.sum()))
#%%
'''firing rate - time  '''

#analy_type = 'stim2'
#datapath = ''
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/spon_1area/'
loop_num = 0 #rep_ind*20 + ie_num
#%%
data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)

#%%
dura = 400
hz_t = np.zeros([N_stim, 400])
mua_loca_1 = [0, 0]
mua_range_1 = 5
mua_neuron_1 = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca_1, mua_range_1, data.a1.param.width)
window = 5

for i in range(N_stim):
    
    start_time = data.a1.param.stim.stim_on[i,0]; end_time = data.a1.param.stim.stim_on[i,0] + dura
    data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = window)
    hz_t[i, :data.a1.ge.spk_rate.spk_rate.shape[-1]] = data.a1.ge.spk_rate.spk_rate.reshape(data.a1.param.Ne,-1)[mua_neuron_1].mean(0)/(window/1000)

#%%
plt.figure()
plt.plot(hz_t.reshape(6,5,-1).mean(1).T)
#%%
plt.figure()
plt.plot(hz_t[:5,:].T)
#%%
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/spon_1area_2/'
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/spon_1area/'

ie = 0

hz_t = np.zeros([N_stim, 400])

for loop_num in range(ie*80, (ie+1)*80+1): #rep_ind*20 + ie_num    
    data_anly = mydata.mydata()
    data_anly.load(datapath+'data_anly%d.file'%loop_num)
    hz_t += data_anly.hz_t 

stim_amp = np.arange(1,7)*200

hz_t /= 81
#%%
hz_t_mean = hz_t.reshape(6,5,-1).mean(1)
plt.figure()
for i in range(6):
    plt.plot(hz_t_mean[i], label = 'stim_amp: %.1f Hz'%(stim_amp[i]))

plt.legend()
plt.xlabel('ms')
plt.ylabel('Hz')
plt.title('average firing rate versus time (80 trials)\nie_ratio_e: %.3f'%(1.00))
#%%
start_time = data.a1.param.stim.stim_on[15,0] - 100
end_time = data.a1.param.stim.stim_on[16,0] + 400

data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 10)
#data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ni, window = 10)
#frames = int(end_time - start_time)
frames = data.a1.ge.spk_rate.spk_rate.shape[2]

stim_on_off = data.a1.param.stim.stim_on-(data.a1.param.stim.stim_on[15,0] - 100)
stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:2]
stim = [[[31,31]], [stim_on_off], [[6]*stim_on_off.shape[0]]]

ani = firing_rate_analysis.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, frames = frames, start_time = start_time, interval_movie=10, anititle='',stim=stim, adpt=None)
                                        

                                        

                                        

                                        

                                        

                                        