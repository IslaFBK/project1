#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:43:40 2021

@author: shni2598
"""

import matplotlib as mpl
mpl.use('Agg')
#import load_data_dict
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
from cfc_analysis import cfc

from scipy.optimize import curve_fit
import scipy.io as sio
#%%
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/spon_1area/'
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/onearea_4096/try/sens/'
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/onearea_4096/verify/sens/'
#sys_argv = int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num
good_dir = 'good/'
goodsize_dir = 'good_size/'
#%%
data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
#%%
ie = 2.76*6.5/5.8*1.51*np.arange(0.6,1.15,0.02)
data_anly = mydata.mydata()
spon_rate = np.zeros(ie.shape)
repeat = 50
divd = repeat
        
for i in range(len(ie)):
    rate_tmp = 0
    divd = repeat
    for loop_num in range(i*repeat, i*repeat+repeat):
        try:data_anly.load(datapath+'data_anly%d.file'%(loop_num))
        except FileNotFoundError:
            divd -= 1
            continue
        
        #data_anly.load(datapath+'data_anly%d.file'%loop_num)
        rate_tmp += data_anly.spon_rate
    rate_tmp /= divd
    spon_rate[i] = rate_tmp

#%%
plt.figure(figsize = [8,6])
start_ind = 9
plt.plot(ie[start_ind:], spon_rate[start_ind:], 'o')
plt.yscale('log')
#%%
sio.savemat('rate_m_dgk16tau60.mat', {'rate_m': rate_m,'ie':ie})
#%%

def power_law1(x, a, b, c):
    return a*np.power(x, b) + c
def power_law2(x, a, b):
    return a*np.power(x, b) 
#%%
#rate_m = spon_rate[start_ind:]

index1 = np.s_[9:18]; index2 = np.s_[19:]; #index3 = np.s_[63:]; 

pars1, cov1 = curve_fit(f=power_law1, xdata=ie[index1], ydata=spon_rate[index1])  #,p0=[-130.9, 0.5404, 260.4 ])
pars2, cov2 = curve_fit(f=power_law1, xdata=ie[index2], ydata=spon_rate[index2],p0=[1.239e9, -17.18, 1.968])
#pars3, cov3 = curve_fit(f=power_law1, xdata=ie[index3], ydata=spon_rate[index3],p0=[8690, -6.901, 0.9625])

#%%
rate1 = power_law1(ie[index1], *pars1)

rate2 = power_law1(ie[index2], *pars2)
#rate3 = power_law1(ie[index3], *pars3)

#%%
plt.figure(figsize=[8,5])
plt.plot(ie[start_ind:],spon_rate[start_ind:],'*')
plt.plot(ie[index1], rate1, label='y=a*x^b+c, a={:.4e},b={:.3f}, c={:.3f}'.format(pars1[0],pars1[1],pars1[2]))
plt.plot(ie[index2], rate2, label='y=a*x^b+c, a={:.4e},b={:.3f}, c={:.3f}'.format(pars2[0],pars2[1],pars2[2]))
#plt.plot(ie[index3], rate3, label='y=a*x^b+c, a={:.4e},b={:.3f}, c={:.3f}'.format(pars3[0],pars3[1],pars3[2]))
# plt.xlabel('ie')
# plt.ylabel('rate Hz')
plt.legend()
plt.yscale('log')    
#plt.xscale('log')    
plt.ylabel('Hz')
plt.xlabel('ie_ratio')
#plt.title('ie,rate, t_ref:5ms,\nsecond part power law is not well fitted')
#plt.title('ie,rate, t_ref:5ms, gl_inhibitory_neuron_25*nS,\nsecond part power law is not well fitted')
#plt.title('ie,rate, t_ref:5ms, dgk:12nS, \nsecond part power law is not well fitted')
# plt.title('''/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/ie_hz_dgk16tau60/
#              t_ref:5ms, dgk:16nS, tau_k:60ms, adapt_range:6''')#,\nsecond part power law is not well fitted''')
plt.title('''/NeuroNet_brian/model_file/attention/two_area/data/onearea_4096/verify/sens
             t_ref:5ms, dgk:5nS, tau_k:60ms, delay:0.5-2.5ms, ''')#,\nsecond part power law is not well fitted''')

#plt.title('ee1.20_ei1.30_ie*_ii1.03_dsi4.44_dse5.00\n/headnode1/shni2598/brian2/NeuroNet_brian/\nmodel_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp')
#plt.title('ee1.20_ei1.27_ie*_ii1.00_dsi4.44_dse5.00\n/headnode1/shni2598/brian2/NeuroNet_brian/\nmodel_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/t5_hz_ie')

        
        