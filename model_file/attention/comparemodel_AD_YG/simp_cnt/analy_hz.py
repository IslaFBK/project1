#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:25:37 2020

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
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/t5_hz_ie/'
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
#scale_ie_1 = np.arange(0.9025,1.2025,0.02)
#scale_ie_1 = np.linspace(0.7,1.3,20)*1.0225
#scale_ie_1 = np.linspace(1.14-0.2,1.14+0.2,31)
#scale_ie_1 = np.linspace(1.213-0.2,1.213+0.2,31)
scale_ie_1 = np.linspace(1.156-0.2,1.156+0.2,31)
#%%
#ie_ratio = 20*6.5/4/5.8*200/320/1.15*scale_ie_1
ie_ratio = 20/4*200/320/1.15*scale_ie_1

200 * 5/8 *20/ ((320 * 5/8)*4*1.15)
#%%
plt.figure()
#plt.plot(ie_ratio,rate,'*')
plt.plot(scale_ie_1,rate,'*')
plt.ylabel('Hz')
plt.xlabel('ie-ratio')
plt.title('dgk:14nS, tau_k:90ms')
#%%
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/bignet_hz_powerlaw/'
loop_num=1
data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
#%%
def power_law(x, a, b, c):
    return a*np.power(x, b) + c

pars1, cov1 = curve_fit(f=power_law, xdata=scale_ie_1[3:12], ydata=rate[3:12])
pars2, cov2 = curve_fit(f=power_law, xdata=scale_ie_1[12:], ydata=rate[12:])
#%%
rate1 = power_law(scale_ie_1[:12], *pars1)

rate2 = power_law(scale_ie_1[12:], *pars2)

#%%
plt.figure(figsize=[8,5])
plt.plot(scale_ie_1[3:],rate[3:],'*')
plt.plot(scale_ie_1[:12], rate1)
plt.plot(scale_ie_1[12:], rate2)
plt.xlabel('ie')
plt.ylabel('rate Hz')
#plt.title('ee1.20_ei1.30_ie*_ii1.03_dsi4.44_dse5.00\n/headnode1/shni2598/brian2/NeuroNet_brian/\nmodel_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp')
plt.title('ee1.20_ei1.27_ie*_ii1.00_dsi4.44_dse5.00\n/headnode1/shni2598/brian2/NeuroNet_brian/\nmodel_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/t5_hz_ie')
