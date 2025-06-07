#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 19:35:14 2020

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
#%%
datapath = ''#'/headnode1/shni2598/brian2/data/attention/compare_AD_YG/AD/planewave/'
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
#if spkdata.a1.param.ie_ratio < 2.85 :
#if loop_num//26 == 0:

spkdata.a1.ge.t = spkdata.a1.ge.t*0.1*ms
#spkdata.a2.e.t = spkdata.a2.e.t*0.1*ms

starttime = 0*second; endtime = 10*second
binforrate=5*ms
spkrate1, centre_ind1, jump_size1, jump_dist1 = psa.get_rate_centre_jumpdist(spkdata.a1.ge, starttime=starttime, endtime=endtime, binforrate=binforrate, sample_interval=1*ms, slide_interval = 1, jump_interval = 15, dt=0.1*ms)

#plt.figure()
plt.matshow(spkrate1[:,:,1000])

#savename = "%.3f_%.2f_%d.png"%(spkdata.a1.param.ie_ratio, spkdata.a1.param.t_ref, spkdata.loop_num)
#%%
#savenum = loop_num//250
##os.chdir('ie%s'%savenum)
#savepath = os.path.abspath('pei%d'%(savenum+8))

#savename = savepath + "/%d_%.5f_%.5f_%.3f_%d.png"%(spkdata.a1.param.tau_p_ei, spkdata.a1.param.w_ei, spkdata.a1.param.w_ii, spkdata.a1.param.ie_ratio, spkdata.loop_num)
savename = "%.2f_%.3f_%d.png"%(spkdata.a1.param.cn_scale_wire, spkdata.a1.param.ie_ratio, spkdata.loop_num)

plt.savefig(savename)


