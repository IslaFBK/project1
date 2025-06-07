#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:57:45 2020

@author: shni2598
"""

'''
remove i,j,w,d from data file to save disk space
'''
#import matplotlib as mpl
#mpl.use('Agg')
#import load_data_dict
import mydata
import numpy as np
# import brian2.numpy_ as np
# from brian2.only import *
#import post_analysis as psa
#import firing_rate_analysis
#import frequency_analysis as fa
#import connection as cn
import pickle
import sys
import os
#import matplotlib.pyplot as plt
import shutil
#%%
datapath = ''
sys_argv = int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num
#good_dir = 'good'
#%%
data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)

del data.a1.param.i_ee, data.a1.param.j_ee, data.a1.param.w_ee, data.a1.param.d_ee 
del data.a1.param.i_ei, data.a1.param.j_ei, data.a1.param.w_ei, data.a1.param.d_ei 
del data.a1.param.i_ie, data.a1.param.j_ie, data.a1.param.w_ie, data.a1.param.d_ie 
del data.a1.param.i_ii, data.a1.param.j_ii, data.a1.param.w_ii, data.a1.param.d_ii

data.save(data.class2dict(), datapath+'data%d.file'%loop_num)
 
