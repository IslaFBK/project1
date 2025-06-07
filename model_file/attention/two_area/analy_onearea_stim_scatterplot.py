#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 17:59:14 2020

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
# datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/onearea/stim_lowii/stim_to_e/'
# sys_argv = 455#int(sys.argv[1])
# loop_num = sys_argv #rep_ind*20 + ie_num
# good_dir = 'good/'
# goodsize_dir = 'good_size/'
save_dir = 'scatter_results/'
#%%
# data = mydata.mydata()
# data.load(datapath+'data%d.file'%loop_num)
# #%%
# data_anly = mydata.mydata()
# dt = 1/10000;
# end = int(10/dt); start = int(5/dt)
# spon_rate1 = np.sum((data.a1.ge.t < end) & (data.a1.ge.t > start))/5/data.a1.param.Ne
# #spon_rate2 = np.sum((data.a2.ge.t < end) & (data.a2.ge.t > start))/5/data.a2.param.Ne

# data_anly.spon_rate1 = spon_rate1
# #%%
# simu_time_tot = 28000

# N_stim = data.a1.param.stim.stim_amp_scale.shape[0]

# data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])

# #%%
# fr_bin = 200
# i = 0
# stim_n = 0
# row_ind = [np.array([]), np.array([]), np.array([])]
# col_ind = [np.array([]), np.array([]), np.array([])]
# for i in range(N_stim):
#     spk_ind = data.a1.ge.spk_matrix[int(3970/2)-1, data.a1.param.stim.stim_on[i,0]*10:(data.a1.param.stim.stim_on[i,0]+fr_bin)*10].nonzero()
#     if i%10 == 0 and i != 0:
#         stim_n += 1
#     row_ind[stim_n] = np.hstack((row_ind[stim_n], spk_ind[0]+i-stim_n*10))
#     col_ind[stim_n] = np.hstack((col_ind[stim_n], spk_ind[1]))
# #%%
# fig, ax = plt.subplots(3,1,figsize = [12,6])
# for i in range(3):
#     ax[i].plot(col_ind[i]/10, row_ind[i]/10, '|')
#     #ax[i].yaxis.set_visible(False)
#%%
#spk_ind[1].shape[0]
ie_r_e = np.arange(1.4,1.61,0.05) #2.76*6.5/5.8*
ie_r_i = np.arange(0.85,1.14,0.03) #2.450*6.5/5.8*

#%%
#dist_bin = np.arange(0, 31.5*2**0.5, 2.5)
#fr_bin = np.array([150, 200, 250]) 
N_stim = 30
#hz_loc = np.zeros([N_stim,len(dist_bin)-1,len(fr_bin)])
spon_rate1 = 0
#hz_t = np.zeros([N_stim, 400])
trial_per_ie = 10

stim_amp = np.arange(1,4)*400
#dist_bin = np.arange(0, 31.5*2**0.5, 2.5)
#dist_bin_plot = (dist_bin+(dist_bin[1]-dist_bin[0])*0.5)[:-1]
simu_time_tot = 28000

data_analy = mydata.mydata()
data = mydata.mydata()


for ie_ind in range(50):
    
    spon_rate1 = 0
    
    fr_bin = 200
    #i = 0
    
    row_ind = [np.array([]), np.array([]), np.array([])]
    col_ind = [np.array([]), np.array([]), np.array([])]

    for loop_num in range(ie_ind*trial_per_ie, ie_ind*trial_per_ie+trial_per_ie):
        #data_analy = mydata.mydata()
        data_analy.load(datapath+'data_anly%d.file'%loop_num)
        #hz_loc += data_analy.hz_loc 
        spon_rate1 += data_analy.spon_rate1
        #hz_t += data_analy.hz_t
        data.load(datapath+'data%d.file'%loop_num)
        data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])
        stim_n = 0
        for i in range(N_stim):
            spk_ind = data.a1.ge.spk_matrix[int(3970/2)-1, data.a1.param.stim.stim_on[i,0]*10:(data.a1.param.stim.stim_on[i,0]+fr_bin)*10].nonzero()
            if i%10 == 0 and i != 0:
                stim_n += 1
            row_ind[stim_n] = np.hstack((row_ind[stim_n], spk_ind[0]+i-stim_n*10+(loop_num-(ie_ind*trial_per_ie))*10))
            col_ind[stim_n] = np.hstack((col_ind[stim_n], spk_ind[1]))

        
    #hz_loc /= trial_per_ie
    spon_rate1 /= trial_per_ie
    #hz_t /= trial_per_ie
        
    #hz_t_mean = hz_t.reshape(3,10,-1).mean(1)
    
    fig, ax = plt.subplots(3,1,figsize = [8,12])
    for iax in range(3):
        ax[iax].plot(col_ind[iax]/10, row_ind[iax], '|', label = 'stim_amp: %.1f Hz'%(stim_amp[iax]))
        ax[iax].legend()
    title = "eier:%.3f_iier:%.3f_1hz:%.2f"%(ie_r_e[ie_ind//10], ie_r_e[ie_ind//10]*ie_r_i[ie_ind%10], spon_rate1)
    plt.suptitle('spike\n'+title)
    plt.savefig(save_dir+'stime_'+title.replace(':','')+'spk'+'_%d'%ie_ind+'.png')
    plt.close()


    # plt.figure(figsize=[8,6])
    # for i in range(3):
    #     plt.plot(hz_t_mean[i], label = 'stim_amp: %.1f Hz'%(stim_amp[i]))
    # title = "eier:%.3f_iier:%.3f_1hz:%.2f"%(ie_r_e[ie_ind//10], ie_r_e[ie_ind//10]*ie_r_i[ie_ind%10], spon_rate1)
    # plt.title(title)
    # plt.legend()
    # plt.savefig(save_dir+'stimei_'+title.replace(':','')+'_temp'+'_%d'%ie_ind+'.png')

