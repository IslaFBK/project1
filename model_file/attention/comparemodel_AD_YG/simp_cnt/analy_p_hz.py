#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 11:55:03 2020

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
from scipy.optimize import curve_fit
#%%
#datapath = '/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/'
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/chg_adpt_netsize/'
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/low_num_ie_chg_ds/chg_delay/'
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/bignet_hz_powerlaw/'
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/low_num_YG/'
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/t5_hz_ie/'
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/dp_num_ii/'
#datapath = ''
sys_argv = 0#int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num
#%%
rate = np.zeros(31)
for loop_num in range(31):
    data = mydata.mydata()
    data.load(datapath+'data_anly%d.file'%loop_num)
    rate[loop_num] = data.spon_rate
    
#%%
'''ii'''
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/dp_num_ii/'
'''ie'''
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/dp_num_ie/'
'''ee'''
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/dp_num_ee/'
'''ei'''
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/dp_num_ei/'

#%%
scale_d_p_ii = np.linspace(0.85,1.15,15)
num_ii = np.linspace(156, 297, 10,dtype=int)
scale_ie_1 = np.linspace(1.156-0.2,1.156+0.2,31)
#%%
'''ie'''
scale_d_p_ie = np.linspace(0.85,1.15,13)  
num_ie = np.linspace(93, 177, 10,dtype=int)
scale_ie_1 = np.linspace(1.156-0.15,1.156+0.15,25)
#%%
def get_data(scale_d_p, num_, scale_ie):
    rate = np.zeros([num_.shape[0], scale_ie_1.shape[0], scale_d_p.shape[0]])      
    rate_rev = np.zeros(rate.shape)
    pk_f = np.zeros(rate.shape)
    pk_f_rev = np.zeros(rate.shape)
    alpha = np.zeros(rate.shape)
    alpha_rev = np.zeros(rate.shape)
    loop_num = 0
    for ii in range(scale_d_p.shape[0]):
        for jj in range(num_.shape[0]):
            for kk in range(scale_ie.shape[0]):
                data_anly = mydata.mydata()
                data_anly.load(datapath+'data_anly%d.file'%loop_num)
                rate[jj,kk,ii] = data_anly.spon_rate
                rate_rev[jj,kk,ii] = data_anly.spon_rate
                try:pk_f[jj,kk,ii] = data_anly.peakF_spon;pk_f_rev[jj,kk,ii] = data_anly.peakF_spon
                except AttributeError: print(loop_num)
                
                try:alpha[jj,kk,ii] = data_anly.alpha_dist[0,0];alpha_rev[jj,kk,ii] = data_anly.alpha_dist[0,0]
                except AttributeError: print(loop_num)
                try: 
                    if data_anly.invalid_pie: 
                        rate_rev[jj,kk,ii] = np.nan
                        pk_f_rev[jj,kk,ii] = np.nan
                        alpha_rev[jj,kk,ii] = np.nan
                except AttributeError:
                    print(loop_num)
                loop_num += 1
    return rate, rate_rev, pk_f, pk_f_rev, alpha, alpha_rev
            
#%%
data_anly.save({'all':rate_2,'delete_invalid_pii':rate_2_rev},'effect_pii_data.file')
#%%
rate, rate_rev, pk_f, pk_f_rev, alpha, alpha_rev = \
    get_data(scale_d_p_ie, num_ie, scale_ie_1)


#%%
#plt.figure()
def plot_num_ieRatio(data, data_rev, num_,scale_ie_1, scale_d_p, scale_d_p_index):
    #decay_p = 2
    fig,[ax1, ax2] = plt.subplots(2,1, figsize=[10,7])
    im1 = ax1.imshow(data[:,:,scale_d_p_index],)
    plt.colorbar(im1,ax=ax1)
    im2 = ax2.imshow(data_rev[:,:,scale_d_p_index],)
    #im2 = ax1.imshow(data[:,:,decay_p],)
    plt.colorbar(im2,ax=ax2)
    ax1.set_yticks(np.arange(num_.shape[0]))
    ax1.set_yticklabels(num_)
    ax1.set_xticks(np.arange(scale_ie_1.shape[0])[::5])
    ax1.set_xticklabels(np.round(scale_ie_1,3)[::5])
    ax1.set_xlabel('ie_ratio')
    ax1.set_ylabel('num_ie')
    ax1.set_title('decay pie:%.2f'%scale_d_p[scale_d_p_index])
    
    ax2.set_title('delete invaild p_ie')
    fig.suptitle('effect of i-e connection probability decay and \nnum of connections(peak prob) on rate(hz)')
    return fig
#%%
figpf_ie = plot_num_ieRatio(pk_f, pk_f_rev, num_ie,scale_ie_1, scale_d_p_ie, scale_d_p_index=8)
#%%
def plot_num_decayP(data, data_rev, num_,scale_d_p, scale_ie, scale_ie_index):
    #ie_scale = 20
    fig,[ax1, ax2] = plt.subplots(2,1, figsize=[7,9])
    im1=ax1.imshow(data[:,scale_ie_index,:],)
    plt.colorbar(im1,ax=ax1)
    im2=ax2.imshow(data_rev[:,scale_ie_index,:],)
    plt.colorbar(im2,ax=ax2)
    ax1.set_yticks(np.arange(num_.shape[0]))
    ax1.set_yticklabels(num_)
    ax1.set_xticks(np.arange(scale_d_p.shape[0])[::3])
    ax1.set_xticklabels(np.round(scale_d_p,3)[::3])
    ax1.set_xlabel('scale_decay_p_ie')
    ax1.set_ylabel('num_ie')
    ax1.set_title('ie_ratio:%.2f'%scale_ie[scale_ie_index])
    
    ax2.set_title('delete invaild p_ie')
    fig.suptitle('effect of i-e connection probability decay and \nnum of connections(peak prob) on rate(hz)')
    return fig
#%%
figpff = plot_num_decayP(pk_f, pk_f_rev, num_ie,scale_d_p_ie, scale_ie_1, scale_ie_index=10)
figpf_dp.suptitle('effect of i-e connection probability decay and \nnum of connections(peak prob) on peak_freq(hz)')
#%%
tt = np.zeros([2,2])
tt[0,0] = np.nan
plt.imshow(tt)            

#%%
def get_data_dgk(dgk, tau_k, scale_ie):
    #rate = np.zeros([dgk.shape[0], tau_k.shape[0], scale_ie.shape[0]])      
    #rate_rev = np.zeros(rate.shape)
    pk_f = np.zeros([dgk.shape[0], tau_k.shape[0], scale_ie.shape[0]])           
    pk_f_adapt = np.zeros(pk_f.shape)
    #pk_f_rev = np.zeros(rate.shape)
    alpha = np.zeros(pk_f.shape)
    #alpha_rev = np.zeros(rate.shape)
    loop_num = 0
    for ii in range(scale_ie.shape[0]):
        for jj in range(dgk.shape[0]):
            for kk in range(tau_k.shape[0]):
                data_anly = mydata.mydata()
                data_anly.load(datapath+'data_anly%d.file'%loop_num)
                pk_f[jj,kk,ii] = data_anly.peakF_spon
                pk_f_adapt[jj,kk,ii] = data_anly.peakF_adapt
                # try:pk_f[jj,kk,ii] = data_anly.peakF_spon;#pk_f_rev[jj,kk,ii] = data_anly.peakF_spon
                
                # except AttributeError: print(loop_num)
                alpha[jj,kk,ii] = data_anly.alpha_dist
                # try:alpha[jj,kk,ii] = data_anly.alpha_dist[0,0];alpha_rev[jj,kk,ii] = data_anly.alpha_dist[0,0]
                # except AttributeError: print(loop_num)
                # try: 
                #     if data_anly.invalid_pie: 
                #         rate_rev[jj,kk,ii] = np.nan
                #         pk_f_rev[jj,kk,ii] = np.nan
                #         alpha_rev[jj,kk,ii] = np.nan
                # except AttributeError:
                #     print(loop_num)
                loop_num += 1
    return pk_f, pk_f_adapt, alpha#, pk_f_rev, alpha, alpha_rev
#%%
tau_k = np.linspace(8,16,9)
dgk = np.linspace(40,120, 9)
ie_scale = np.linspace(1.156-0.15,1.156+0.15,25)
#%%
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/adapt/'

#%%
pk_f, pk_f_adapt, alpha = get_data_dgk(dgk, tau_k, ie_scale)

#%%
scale_ie_index = 13
fig,[ax1, ax2] = plt.subplots(2,1, figsize=[6,8])
im1=ax1.imshow(pk_f[:,:,scale_ie_index])
plt.colorbar(im1,ax=ax1)
im2=ax2.imshow(pk_f_adapt[:,:,scale_ie_index])
plt.colorbar(im2,ax=ax2)
#%%
ax1.set_yticks(np.arange(num_.shape[0]))
ax1.set_yticklabels(num_)
ax1.set_xticks(np.arange(scale_d_p.shape[0])[::3])
ax1.set_xticklabels(np.round(scale_d_p,3)[::3])
ax1.set_xlabel('scale_decay_p_ie')
ax1.set_ylabel('num_ie')
ax1.set_title('ie_ratio:%.2f'%scale_ie[scale_ie_index])
