#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 23:19:03 2020

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
save_dir = 'fft_results/'
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
#hz_loc = np.zeros([N_stim,len(dist_bin)-1,len(fr_bin)])
spon_rate1 = 0
#hz_t = np.zeros([N_stim, 400])
trial_per_ie = 10


#dist_bin = np.arange(0, 31.5*2**0.5, 2.5)
#dist_bin_plot = (dist_bin+(dist_bin[1]-dist_bin[0])*0.5)[:-1]
start_time = 5000
end_time = 10000
simu_time_tot = 28000

data_analy = mydata.mydata()
data = mydata.mydata()

mua_loca_1 = [0, 0]
mua_range_1 = 5
# mua_neuron_1 = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca_1, mua_range_1, data.a1.param.width)
window_spk = 5
Fs = 1000
freq_max_plot = [20, 100]

for ie_ind in range(50):
    
    spon_rate1 = 0
    
    #fr_bin = 200
    #i = 0
    
    #row_ind = [np.array([]), np.array([]), np.array([])]
    #col_ind = [np.array([]), np.array([]), np.array([])]
    coef_mean = 0
    for loop_num in range(ie_ind*trial_per_ie, ie_ind*trial_per_ie+trial_per_ie):
        #data_analy = mydata.mydata()
        data_analy.load(datapath+'data_anly%d.file'%loop_num)
        #hz_loc += data_analy.hz_loc 
        spon_rate1 += data_analy.spon_rate1
        #hz_t += data_analy.hz_t
        data.load(datapath+'data%d.file'%loop_num)
        if loop_num == 0: mua_neuron_1 = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca_1, mua_range_1, data.a1.param.width)
        data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = window_spk)
        mua = data.a1.ge.spk_rate.spk_rate.reshape(data.a1.param.Ne,-1)[mua_neuron_1].mean(0)/(window_spk/1000)
        coef, freq = fa.myfft(mua, Fs, norm=True)
        coef_mean += np.abs(coef)
                
    #hz_loc /= trial_per_ie
    spon_rate1 /= trial_per_ie
    coef_mean /= trial_per_ie
    
    fig, ax = plt.subplots(2,2,figsize = [10,6])
    for i in range(2):
        for j in range(2):
            freq_plot = freq[freq < freq_max_plot[i]][1:]
            coef_plot = coef_mean[freq < freq_max_plot[i]][1:]
            if j==0:
                ax[i,j].plot(freq_plot, coef_plot)
            else:
                ax[i,j].loglog(freq_plot, coef_plot)

    title = "eier:%.3f_iier:%.3f_1hz:%.2f"%(ie_r_e[ie_ind//10], ie_r_e[ie_ind//10]*ie_r_i[ie_ind%10], spon_rate1)
    plt.suptitle('fft\n'+title)
    plt.savefig(save_dir+title.replace(':','')+'_fft'+'_%d'%ie_ind+'.png')
    plt.close()


    # plt.figure(figsize=[8,6])
    # for i in range(3):
    #     plt.plot(hz_t_mean[i], label = 'stim_amp: %.1f Hz'%(stim_amp[i]))
    # title = "eier:%.3f_iier:%.3f_1hz:%.2f"%(ie_r_e[ie_ind//10], ie_r_e[ie_ind//10]*ie_r_i[ie_ind%10], spon_rate1)
    # plt.title(title)
    # plt.legend()
    # plt.savefig(save_dir+'stimei_'+title.replace(':','')+'_temp'+'_%d'%ie_ind+'.png')

