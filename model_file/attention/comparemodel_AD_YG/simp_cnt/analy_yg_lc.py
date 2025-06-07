#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:08:33 2020

@author: shni2598
"""


import matplotlib as mpl
#mpl.use('Agg')
#import load_data_dict
import mydata
import brian2.numpy_ as np
from brian2.only import *
#import post_analysis as psa
import firing_rate_analysis
import connection as cn
import pickle
import sys
import os
import matplotlib.pyplot as plt
#%%
loop_num = 2
datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_chg_adapt_highe_simp_verif/'#%%
data2 = mydata.mydata()
data2.load(datapath+'data%d.file'%loop_num)
#%%
e_lattice = cn.coordination.makelattice(63,62,[0,0])
chg_adapt_loca = [0, 0]
chg_adapt_range = 6
width = 62
chg_adapt_neuron = cn.findnearbyneuron.findnearbyneuron(e_lattice, chg_adapt_loca, chg_adapt_range, width)

#%%
data2.get_spike_rate(0,8000)

# data1.a1.ge.get_spike_rate(start_time=4000, end_time=14000,\
#                            sample_interval = 1, n_neuron = ijwd.Ne, window = 10, dt = 0.1, reshape_indiv_rate = True)

mua_adpt2 = data2.spk_rate.spk_rate.reshape(3969,-1)[chg_adapt_neuron].sum(0)
#%%
start_time = 1000; end_time = 4000
frames = end_time - start_time

ani = firing_rate_analysis.show_pattern(data2.spk_rate.spk_rate, data2.spk_rate.spk_rate, frames = frames-50, start_time = start_time, anititle='animation')
#%%
#%%
plt.figure()
plt.plot(mua_adpt2)



















