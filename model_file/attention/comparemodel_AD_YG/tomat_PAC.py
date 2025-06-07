#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 12:12:40 2020

@author: shni2598
"""

import matplotlib as mpl
mpl.use('Agg')
#import load_data_dict
import mydata
import brian2.numpy_ as np
from brian2.only import *
#import post_analysis as psa
import connection as cn
import firing_rate_analysis
import frequency_analysis
import pickle
import sys
import os
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import stats
from scipy import signal as spsignal
import scipy.io as sio
#%%
loop_num = int(sys.argv[1])

datapath = ''
data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
#%%
dt = data.dt/1000
Fs = 1/dt

data.a1.ge.get_sparse_spk_matrix(shape=[3969, int(np.round(data.a1.param.simutime/data.dt))])
#%%
lattice_ext = cn.coordination.makelattice(63,62,[0,0])
mua_loca = [0, 0]
mua_range = 5
mua_neuron = cn.findnearbyneuron.findnearbyneuron(lattice_ext, mua_loca, mua_range, 62)

mua = data.a1.ge.spk_matrix[mua_neuron,:].A.sum(0)
mua_ker = np.ones(20);
mua_rate = np.convolve(mua, mua_ker, mode='valid')
#%%
AD_data = {'mua':mua,'mua_rate':mua_rate,
           'LFP':{'lfp1': data.a1.LFP.lfp1,
           'lfp2': data.a1.LFP.lfp2,
           'lfp3': data.a1.LFP.lfp3}}

sio.savemat('AD_data%d.mat'%loop_num,{'AD_data':AD_data})

