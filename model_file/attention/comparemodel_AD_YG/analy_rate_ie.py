#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 13:57:03 2020

@author: shni2598
"""


import matplotlib as mpl
mpl.use('Agg')
#import load_data_dict
import mydata
import brian2.numpy_ as np
from brian2.only import *
import post_analysis as psa
import connection as cn
import pickle
import sys
import os
import matplotlib.pyplot as plt
from scipy.fftpack import fft
#%%
loop_num = int(sys.argv[1])

#datapath = '/headnode1/shni2598/brian2/data/attention/compare_AD_YG/YG/'
#datapath = '/headnode1/shni2598/brian2/data/attention/compare_AD_YG/AD/rk4/'
datapath = ''
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/gz_param_spkinput/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/gz_param_spkinput/squareinput/'
with open(datapath+'data%d.file'%loop_num, 'rb') as file:
    #data = pickle.load(file)
    spkdata = mydata.mydata(pickle.load(file))

spkdata.a1.ge.rate_overall = spkdata.a1.ge.i.size/10/3969
data_dict = spkdata.class2dict()

#%%
with open(datapath+'data%d.file'%loop_num, 'wb') as file:
    #data = pickle.load(file)
    pickle.dump(data_dict, file)


#%%



