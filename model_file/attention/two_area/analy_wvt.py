#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 15:23:52 2021

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
data_dir = 'raw_data/'
analy_type = 'state'
datapath = data_dir
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/spon_1area/'
sys_argv = int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num
good_dir = 'good/'
goodsize_dir = 'good_size/'

savefile_name = 'data_anly' #'data_anly' data_anly_temp

fftplot = 1; get_wvt = 1
getfano = 1
get_nscorr = 1; get_nscorr_t = 1
get_TunningCurve = 1; get_HzTemp = 1
firing_rate_long = 1

if loop_num%4 == 0: save_img = 1
else: save_img = 0

if loop_num%10 ==0: get_ani = 1
else: get_ani = 0

#%%
clr = plt.rcParams['axes.prop_cycle'].by_key()['color']

data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
#data_anly = mydata.mydata()


if analy_type == 'state': # fbrg: feedback range
    title = 'hz_2irie%.2f_2ndgk%.1f_pk2e1e%.2f'%(data.param.ie_r_i2, data.param.new_delta_gk_2, \
                                               data.inter.param.peak_p_e2_e1)
        
        
n_StimAmp = data.a1.param.stim1.n_StimAmp
n_perStimAmp = data.a1.param.stim1.n_perStimAmp
stim_amp = 200*2**np.arange(n_StimAmp)
#%%
'''spontanous rate'''
dt = 1/10000;
end = int(20/dt); start = int(5/dt)
spon_rate1 = np.sum((data.a1.ge.t < end) & (data.a1.ge.t >= start))/15/data.a1.param.Ne
spon_rate2 = np.sum((data.a2.ge.t < end) & (data.a2.ge.t >= start))/15/data.a2.param.Ne

#data_anly.spon_rate1 = spon_rate1
#data_anly.spon_rate2 = spon_rate2
'''adapt rate'''
start = int(round((data.a1.param.stim1.stim_on[n_StimAmp*n_perStimAmp,0] - 2000)/1000/dt))
end = int(round((data.a1.param.stim1.stim_on[n_StimAmp*n_perStimAmp,0])/1000/dt))
adapt_rate1 = np.sum((data.a1.ge.t < end) & (data.a1.ge.t >= start))/2/data.a1.param.Ne
#data_anly.adapt_rate1 = adapt_rate1
adapt_rate2 = np.sum((data.a2.ge.t < end) & (data.a2.ge.t >= start))/2/data.a2.param.Ne
#data_anly.adapt_rate2 = adapt_rate2

title += '_1hz%.2f_1adphz%.2f_2hz%.2f_2adphz%.2f'%(spon_rate1,adapt_rate1,spon_rate2,adapt_rate2)

#%%
if get_wvt:
    mua_range_1 = 5
    mua_loca_1 = [0,0]
    mua_neuron_1 = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca_1, mua_range_1, data.a1.param.width)
    
    for st in range(n_StimAmp):
        
        start_time = data.a1.param.stim1.stim_on[st*n_perStimAmp,0] - 300
        end_time = data.a1.param.stim1.stim_on[st*n_perStimAmp+5,0] + 500 
            
        #start_time = 5e3; end_time = 20e3
        window = 5
        data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = window)
        mua = data.a1.ge.spk_rate.spk_rate.reshape(data.a1.param.Ne,-1)[mua_neuron_1]
        mua = mua.mean(0)/(window/1000)
        
        coef, freq = fqa.mycwt(mua, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)
    
        fig, ax, ax2 = fqa.plot_cwt(coef, freq, base = 10)
        plt.xlabel('ms')
        plt.ylabel('Hz')
        title_ = title + '\nstim%.1f_noatt'%(stim_amp[st])
        fig.suptitle(title_)
        savetitle = title_.replace('\n','')
        #nsfile = savetitle+'_nct_%d'%(loop_num)+'.png'
        fig.savefig(savetitle+'_wvt_st%.1f_%d'%(stim_amp[st],loop_num)+'.png')
        plt.close()
    
        start_time = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp,0] - 300
        end_time = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp+5,0] + 500 
            
        #start_time = 5e3; end_time = 20e3
        window = 5
        data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = window)
        mua = data.a1.ge.spk_rate.spk_rate.reshape(data.a1.param.Ne,-1)[mua_neuron_1]
        mua = mua.mean(0)/(window/1000)
        
        coef, freq = fqa.mycwt(mua, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)
    
        fig, ax, ax2 = fqa.plot_cwt(coef, freq, base = 10)
        plt.xlabel('ms')
        plt.ylabel('Hz')
        title_ = title + '\nstim%.1f_att'%(stim_amp[st])
        fig.suptitle(title_)
        savetitle = title_.replace('\n','')
        #nsfile = savetitle+'_nct_%d'%(loop_num)+'.png'
        fig.savefig(savetitle+'_wvt_st%.1f_%d'%(stim_amp[st],loop_num)+'.png')
        plt.close()    
#%%    
    
    
    
    
    