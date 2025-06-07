#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:11:28 2020

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
#%%
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/low_num_ie_chg_ds/chg_delay/'
#%%
delay = np.arange(2,5.1,0.5)
rate, pf_spon , pf_adapt , pf_mag_spon , pf_mag_adapt = np.zeros([5,7,7]), np.zeros([5,7,7]),np.zeros([5,7,7]),np.zeros([5,7,7]),np.zeros([5,7,7])

#%%
data_anly = mydata.mydata()

for loop_num in range(245):
    
    data_anly.load(datapath+'data_anly%d.file'%loop_num)

    d = loop_num%7
    e = loop_num//7%5
    i = loop_num//35%7
    
    rate[e,i,d] = data_anly.spon_rate
    pf_spon[e,i,d] = data_anly.spon_pf
    pf_adapt[e,i,d] = data_anly.adapt_pf
    pf_mag_spon[e,i,d] = data_anly.spon_pf_mag
    pf_mag_adapt[e,i,d] = data_anly.adapt_pf_mag

#%%

#for d in range(7):
for d in range(7):

    fig, ax = plt.subplots(3,2,figsize=[11,10])
    
    min_ = min(pf_spon[:,:,d].min(), pf_adapt[:,:,d].min())
    max_ = max(pf_spon[:,:,d].max(), pf_adapt[:,:,d].max())
    im = ax[0,0].matshow(pf_spon[:,:,d],aspect='auto',vmin=min_,vmax=max_);
    
    fig.colorbar(im,ax=ax[0,0])
    ax[0,0].axes.xaxis.set_ticks(np.arange(7))
    ax[0,0].axes.yaxis.set_ticks(np.arange(5))
    ax[0,0].axes.xaxis.set_ticklabels(np.round(np.arange(0.7,1.31,0.1)*4.444,2))
    ax[0,0].axes.yaxis.set_ticklabels(np.round(np.arange(0.7,1.11,0.1)*5.8,2))
    ax[0,0].set_xlabel('syn_decay_i')
    ax[0,0].set_ylabel('syn_decay_e')
    ax[0,0].set_title('peak_freq_spon',fontsize=10)
    
    im = ax[0,1].matshow(pf_adapt[:,:,d],aspect='auto',vmin=min_,vmax=max_);     ax[0,1].axes.xaxis.set_visible(False); ax[0,1].axes.yaxis.set_visible(False)
    fig.colorbar(im,ax=ax[0,1])
    ax[0,1].set_title('peak_freq_adapt',fontsize=10)
    
    min_ = min(pf_mag_spon[:,:,d].min(), pf_mag_adapt[:,:,d].min())
    max_ = max(pf_mag_spon[:,:,d].max(), pf_mag_adapt[:,:,d].max())
    
    im = ax[1,0].matshow(pf_mag_spon[:,:,d],aspect='auto',vmin=min_,vmax=max_);  ax[1,0].axes.xaxis.set_visible(False); ax[1,0].axes.yaxis.set_visible(False)
    fig.colorbar(im,ax=ax[1,0])
    ax[1,0].set_title('peak_freq_mag_spon',fontsize=10)
    
    im = ax[1,1].matshow(pf_mag_adapt[:,:,d],aspect='auto',vmin=min_,vmax=max_); ax[1,1].axes.xaxis.set_visible(False); ax[1,1].axes.yaxis.set_visible(False)
    fig.colorbar(im,ax=ax[1,1])
    ax[1,0].set_title('peak_freq_mag_adapt',fontsize=10)
    
    im = ax[2,0].matshow(rate[:,:,d],aspect='auto');                        ax[2,0].axes.xaxis.set_visible(False); ax[2,0].axes.yaxis.set_visible(False)
    fig.colorbar(im,ax=ax[2,0])
    ax[2,0].set_title('firing rate (Hz)',fontsize=10)
    
    #ax[2,1].matshow(rate[:,:,0]); ax[2,0].tick_params(axis='both', which='both', length=0)
    
    ax[2,1].set_visible(False)
    
    fig.suptitle('delay:%.2f ms'%delay[d])
    plt.savefig('effects_dse_dsi_d%.2f.png'%delay[d])

    
#%%
#e_lattice = cn.coordination.makelattice(int(np.sqrt(data.a1.param.Ne).round()),data.a1.param.width,[0,0])

chg_adapt_loca = data.a1.param.chg_adapt_loca #[0, 0]
chg_adapt_range = data.a1.param.chg_adapt_range #6 
chg_adapt_neuron = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, chg_adapt_loca, chg_adapt_range, data.a1.param.width)
#%%
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/low_num_ie_chg_ds/chg_ie/'
#%%
loop_num = 226
data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num) 
#%%
start_time = 0; end_time = 20e3
data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 5)
mua = data.a1.ge.spk_rate.spk_rate.reshape(data.a1.param.Ne,-1)[chg_adapt_neuron]
mua = mua.mean(0)/0.01
#%%
fig, [ax1,ax2] = plt.subplots(1,2)

fs = 200
data_fft = mua[4*fs:10*fs]
coef, freq = fa.myfft(data_fft, fs)

freq_max = 20
ind_len = freq[freq<freq_max].shape[0] # int(20/(fs/2)*(len(data_fft)/2)) + 1
ax1.plot(freq[1:ind_len], np.abs(coef[1:ind_len]))
ax1.plot([3,3],[0,np.max(np.abs(coef[1:ind_len]))],'r--')
ax1.plot([8,8],[0,np.max(np.abs(coef[1:ind_len]))],'r--')

ax1.set_title('spon')


data_fft = mua[10*fs:]
coef, freq = fa.myfft(data_fft, fs)

ind_len = freq[freq<freq_max].shape[0] # int(20/(fs/2)*(len(data_fft)/2)) + 1
ax2.plot(freq[1:ind_len], np.abs(coef[1:ind_len]))
ax2.plot([3,3],[0,np.max(np.abs(coef[1:ind_len]))],'r--')
ax2.plot([8,8],[0,np.max(np.abs(coef[1:ind_len]))],'r--')

ax2.set_title('adapt')

fig.suptitle(title)
    
#%%    
fig, [ax1,ax2] = plt.subplots(1,2)

fs = 1000
data_fft = mua[4*fs:10*fs]
coef, freq = fa.myfft(data_fft, fs)

freq_max = 100
ind_len = freq[freq<freq_max].shape[0] # int(20/(fs/2)*(len(data_fft)/2)) + 1
ax1.loglog(freq[1:ind_len], np.abs(coef[1:ind_len]))
ax1.plot([3,3],[0,np.max(np.abs(coef[1:ind_len]))],'r--')
ax1.plot([8,8],[0,np.max(np.abs(coef[1:ind_len]))],'r--')

ax1.set_title('spon')


data_fft = mua[10*fs:]
coef, freq = fa.myfft(data_fft, fs)

ind_len = freq[freq<freq_max].shape[0] # int(20/(fs/2)*(len(data_fft)/2)) + 1
ax2.loglog(freq[1:ind_len], np.abs(coef[1:ind_len]))
ax2.plot([3,3],[0,np.max(np.abs(coef[1:ind_len]))],'r--')
ax2.plot([8,8],[0,np.max(np.abs(coef[1:ind_len]))],'r--')

ax2.set_title('adapt')

fig.suptitle(title)
    






    




















