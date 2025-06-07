#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 11:19:59 2021

@author: shni2598
"""

import matplotlib as mpl
mpl.use('Agg')
import scipy.stats
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
save_dir = 'mean_results/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
#%%
clr = plt.rcParams['axes.prop_cycle'].by_key()['color']

# data = mydata.mydata()
# data.load(datapath+'data%d.file'%loop_num)
datapath = ''
data_anly = mydata.mydata()

hz_t_mean_1 = np.zeros([3,400,30])
hz_t_mean_1[:] = np.nan
hz_t_mean_2 = np.zeros([3,400,30])
hz_t_mean_2[:] = np.nan

spon_rate = np.zeros(30)
spon_rate[:] = np.nan
for loop_num in range(30):
    data_anly.load(datapath+'data_anly%d.file'%(loop_num))
    spon_rate[loop_num] = data_anly.spon_rate1
    for st in range(3):
        hz_t_mean_1[st, :, loop_num] = data_anly.hz_t_1[st*30:st*30+30].mean(0)
        hz_t_mean_2[st, :, loop_num] = data_anly.hz_t_2[st*30:st*30+30].mean(0)

hz_t_mean_reliz_1 = np.nanmean(hz_t_mean_1,2)
hz_t_mean_reliz_2 = np.nanmean(hz_t_mean_2,2)
hz_t_sem_reliz_1 = scipy.stats.sem(hz_t_mean_1,2, nan_policy='omit')
hz_t_sem_reliz_2 = scipy.stats.sem(hz_t_mean_2,2, nan_policy='omit')
spon_rate_mean = np.nanmean(spon_rate)

plt.figure(figsize=[8,6])
t_plot = np.arange(400) - 100

stim_amp_1 = 400
stim_amp_2 = np.array([200,400,600])
for st in range(3):
    plt.plot(t_plot, hz_t_mean_reliz_1[st], ls='-', c=clr[st], label = 'loc_1; stim_1: %.1f Hz; stim_2: %.1f Hz'%(stim_amp_1,stim_amp_2[st]))
    plt.fill_between(t_plot, hz_t_mean_reliz_1[st]-hz_t_sem_reliz_1[st], hz_t_mean_reliz_1[st]+hz_t_sem_reliz_1[st], \
                     ls='-', facecolor=clr[st], edgecolor=clr[st], alpha=0.2)
    plt.plot(t_plot, hz_t_mean_reliz_2[st], ls='--', c=clr[st], label = 'loc_2; stim_1: %.1f Hz; stim_2: %.1f Hz'%(stim_amp_1,stim_amp_2[st]))
    plt.fill_between(t_plot, hz_t_mean_reliz_2[st]-hz_t_sem_reliz_2[st], hz_t_mean_reliz_2[st]+hz_t_sem_reliz_2[st], \
                     ls='--', facecolor=clr[st], edgecolor=clr[st], alpha=0.2)
    
plt.xlabel('ms')
plt.ylabel('Hz')

plt.legend()

plt.title('firing rate-time; senssory')
savetitle = 'hz%.2f'%(spon_rate_mean)#title.replace('\n','')
tempfile = savetitle+'_temp'+'.png' #+'_%d'%loop_num+'.png'
plt.savefig(save_dir + tempfile)
plt.close()




