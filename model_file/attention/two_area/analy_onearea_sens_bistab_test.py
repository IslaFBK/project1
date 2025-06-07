#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 22:55:48 2021

@author: shni2598
"""

'''bi-stable'''
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
fftplot = 1; getfano = 1; 
get_nscorr = 1; get_nscorr_t = 1
get_TunningCurve = 1; get_HzTemp = 1;

analy_type = 'fbrg'
savefile_name = 'data_anly' 
# save_dir = 'mean_results/'
# data_analy_file = 'data_anly'#data_anly' #data_anly_fano data_anly_fano10hz data_anly_nc data_anly_fano10hz data_anly_fanorange
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
    
clr = plt.rcParams['axes.prop_cycle'].by_key()['color']
   
data_anly = mydata.mydata()
data = mydata.mydata()

#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/twoarea_hz/fb/const_2ndstim/raw_data/'
datapath = 'raw_data/'

sys_argv = int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num

#data_anly.load(datapath+data_analy_file+'%d.file'%(loop_num))
data.load(datapath+'data%d.file'%(loop_num))
#%%
simu_time_tot = data.param.simutime#29000

data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10],mat_type='csc')

#ie_r_e1_list = np.arange(0.86, 1.141, 0.01)
ie_r_e1_list = data.param.ie_r_e1_list.copy()
rate = np.zeros(ie_r_e1_list.shape)
simu_time_init = 5000*int(round(1/data.dt))
simu_time_trial = 5000*int(round(1/data.dt))
for ii in range(len(ie_r_e1_list)):
    rate[ii] = (data.a1.ge.spk_matrix.indptr[simu_time_init+(ii+1)*simu_time_trial] - data.a1.ge.spk_matrix.indptr[simu_time_init+(ii)*simu_time_trial])/data.a1.param.Ne/5

data_anly.rate = rate
data_anly.ie_r_e1_list = ie_r_e1_list

#%%
len_ie_r_e1_list = int(round(ie_r_e1_list.shape[0]/2))
fig, ax = plt.subplots(1,2, figsize=[10,4])
ax[0].plot(ie_r_e1_list[:len_ie_r_e1_list], rate[:len_ie_r_e1_list],c=clr[0], label='increasing ie')
ax[0].plot(ie_r_e1_list[:len_ie_r_e1_list], rate[len_ie_r_e1_list:][::-1],c=clr[1], label='decreasing ie')
ax[0].legend()
ax[0].set_title('bi-stabel sens')
ax[1].loglog(ie_r_e1_list[:len_ie_r_e1_list], rate[:len_ie_r_e1_list],c=clr[0], label='increasing ie')
ax[1].loglog(ie_r_e1_list[:len_ie_r_e1_list], rate[len_ie_r_e1_list:][::-1],c=clr[1], label='decreasing ie')
ax[1].legend()

fig.savefig('bistable_sens_%d.png'%loop_num)

#%%
data_anly.save(data_anly.class2dict(), datapath+savefile_name+'%d.file'%loop_num)
