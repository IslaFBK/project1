#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 13:13:32 2021

@author: shni2598
"""
import mydata
import brian2.numpy_ as np
from brian2.only import *
#import post_analysis as psa
import firing_rate_analysis as fra
import frequency_analysis as fqa
import connection as cn
import pickle
import sys
import os
import matplotlib.pyplot as plt
import shutil
#%%
clr = plt.rcParams['axes.prop_cycle'].by_key()['color']

analy_type = 'var'
#datapath = ''
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/var/'
sys_argv = 0#int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num
good_dir = 'good/'
goodsize_dir = 'good_size/'
#%%
data_anly = mydata.mydata()
data_anly.load(datapath+'data_anly%d.file'%loop_num)
#%%
plot_mean = np.vstack((data_anly.mean_noatt, data_anly.mean_att))
plot_var = np.vstack((data_anly.var_noatt, data_anly.var_att))
slop = (plot_var[1,:] - plot_var[0,:])/(plot_mean[1,:] - plot_mean[0,:])
slop = slop[np.logical_not((slop == np.inf) | (slop == -np.inf))]
#%%
data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)

#%%
mua_loca_1 = [0, 0]
mua_range_1 = 5
mua_neuron_1 = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca_1, mua_range_1, data.a1.param.width)
window = 5
dura_onoff = data.a1.param.stim.stim_on.copy()
dura_onoff[:,0] -= 100
dura_onoff[:,1] += 100
hz_t = fra.firing_rate_time_multi(data.a1.ge, mua_neuron_1, dura_onoff, window=window)
#%%
stim_amp = 400
data_anly.hz_t = hz_t

hz_t_noatt_mean = hz_t[:100,:].mean(0)
hz_t_noatt_std = hz_t[:100,:].std(0)

hz_t_att_mean = hz_t[100:200,:].mean(0)
hz_t_att_std = hz_t[100:200,:].std(0)

plt.figure(figsize=[8,6])
#for i in range(n_amp_stim_att):
t_plot = np.arange(dura_onoff[0,1] - dura_onoff[0,0]) - 100
plt.plot(t_plot, hz_t_noatt_mean, ls='--', c=clr[0], label = 'stim_amp: %.1f Hz'%(stim_amp))
plt.fill_between(t_plot, hz_t_noatt_mean-hz_t_noatt_std,hz_t_noatt_mean+hz_t_noatt_std, \
                 ls='--', facecolor=clr[0], edgecolor=clr[0], alpha=0.2)
#for i in range(n_amp_stim_att,n_amp_stim):
plt.plot(t_plot, hz_t_att_mean, ls='-', c=clr[0], label = 'attention; stim_amp: %.1f Hz'%(stim_amp)) 
plt.fill_between(t_plot, hz_t_att_mean-hz_t_att_std,hz_t_att_mean+hz_t_att_std, \
                 ls='-', facecolor=clr[0], edgecolor=clr[0], alpha=0.2)

plt.legend()
#%%
data_anly = mydata.mydata()
hz_loc_noatt = np.zeros(17)
hz_loc_att = np.zeros(17)
hz_loc_spon = np.zeros([2,17])
for loop_num in range(10):
    data_anly.load(datapath+'data_anly%d.file'%loop_num)
    hz_loc_noatt += data_anly.hz_loc[:100].mean(0)
    hz_loc_att += data_anly.hz_loc[100:].mean(0)
    hz_loc_spon += data_anly.hz_loc_spon
#%%
hz_loc_noatt /= 10
hz_loc_att /=10
hz_loc_spon /=10
#%%
plt.figure(figsize=[8,6])
plt.plot(hz_loc_noatt)
plt.plot(hz_loc_att)
plt.plot(hz_loc_spon[0],'--')
plt.plot(hz_loc_spon[1])
#%%
loop_num = 1
data_anly.load(datapath+'data_anly%d.file'%loop_num)
#%%
data_anly.hz_t[:100,0].mean()
#%%
plt.figure();
plt.hist(data_anly.hz_t[:100,:100].reshape(-1),100)
plt.yscale('log')
plt.xscale('log')
#%%
plt.figure();
plt.plot(data_anly.hz_loc_spon[0,:])
plt.plot(data_anly.hz_loc_spon[1,:])


