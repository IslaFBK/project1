#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 15:22:18 2020

@author: shni2598
"""

# import matplotlib as mpl
# mpl.use('Agg')
#import load_data_dict
import mydata
# import brian2.numpy_ as np
# from brian2.only import *
#import post_analysis as psa
# import firing_rate_analysis
# import frequency_analysis as fa
# import connection as cn
# import pickle
import sys
import os
# import matplotlib.pyplot as plt
# import shutil


#%%
#analy_type = 'stim3'
datapath = ''
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/spon_1area/'
sys_argv = int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num
#good_dir = 'good/'
#goodsize_dir = 'good_size/'
#%%
print('start')
data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)

del data.a1.ge.v
del data.a1.gi.v

data.save(data.class2dict(), datapath+'data%d.file'%loop_num)
print('finished')
