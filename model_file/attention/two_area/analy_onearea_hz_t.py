#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 15:29:34 2020

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
analy_type = 'one_stim'
datapath = ''
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/onearea/'
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/spon_1area/'
#sys_argv = int(sys.argv[1])
# loop_num = sys_argv #rep_ind*20 + ie_num
# good_dir = 'good/'
# goodsize_dir = 'good_size/'
save_dir = 'results/'
#%%
ie_r_e = np.arange(1.4,1.61,0.05) #2.76*6.5/5.8*
ie_r_i = np.arange(0.85,1.14,0.03) #2.450*6.5/5.8*

#%%
dist_bin = np.arange(0, 31.5*2**0.5, 2.5)
fr_bin = np.array([150, 200, 250]) 
N_stim = 30
hz_loc = np.zeros([N_stim,len(dist_bin)-1,len(fr_bin)])
spon_rate1 = 0
hz_t = np.zeros([N_stim, 400])
trial_per_ie = 10

stim_amp = np.arange(1,4)*400
dist_bin = np.arange(0, 31.5*2**0.5, 2.5)
dist_bin_plot = (dist_bin+(dist_bin[1]-dist_bin[0])*0.5)[:-1]

data_analy = mydata.mydata()


for ie_ind in range(50):
    hz_loc = np.zeros([N_stim,len(dist_bin)-1,len(fr_bin)])
    spon_rate1 = 0
    hz_t = np.zeros([N_stim, 400])

    for loop_num in range(ie_ind*trial_per_ie, ie_ind*trial_per_ie+trial_per_ie):
        #data_analy = mydata.mydata()
        data_analy.load(datapath+'data_anly%d.file'%loop_num)
        hz_loc += data_analy.hz_loc 
        spon_rate1 += data_analy.spon_rate1
        hz_t += data_analy.hz_t
        
        
    hz_loc /= trial_per_ie
    spon_rate1 /= trial_per_ie
    hz_t /= trial_per_ie
    
    for b in range(1,2):
        hz_loc_mean = hz_loc[:,:,b].reshape(3,10,-1).mean(1)
        plt.figure(figsize=[8,6])
        for i in range(3):
            
            plt.plot(dist_bin_plot, hz_loc_mean[i]/(fr_bin[b]/1000), '-o', label = 'stim_amp: %.1f Hz'%(stim_amp[i]))
        title = "eier:%.3f_iier:%.3f_1hz:%.2f"%(ie_r_e[ie_ind//10], ie_r_e[ie_ind//10]*ie_r_i[ie_ind%10], spon_rate1)
        plt.title('average firing rate versus distance to stimulus centre; no top-down\n Spike-count bin: %.1f ms\n'%fr_bin[b]+title)
        plt.xlim([dist_bin[0],dist_bin[-1]])
        plt.xlabel('distance')
        plt.ylabel('Hz')
        plt.legend()
        plt.savefig(save_dir+'stimei_'+title.replace(':','')+'_countbin%.0f'%fr_bin[b]+'_%d'%ie_ind+'.png')
    
    hz_t_mean = hz_t.reshape(3,10,-1).mean(1)
    plt.figure(figsize=[8,6])
    for i in range(3):
        plt.plot(hz_t_mean[i], label = 'stim_amp: %.1f Hz'%(stim_amp[i]))
    title = "eier:%.3f_iier:%.3f_1hz:%.2f"%(ie_r_e[ie_ind//10], ie_r_e[ie_ind//10]*ie_r_i[ie_ind%10], spon_rate1)
    plt.title(title)
    plt.legend()
    plt.savefig(save_dir+'stimei_'+title.replace(':','')+'_temp'+'_%d'%ie_ind+'.png')






