#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 11:33:02 2020

@author: shni2598
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 12:09:48 2020

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
import connection as cn
import pickle
import sys
import os
import matplotlib.pyplot as plt
#%%
#datapath = '/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/'
datapath = ''
sys_argv = int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num
#%%
data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
#%%
'''animation'''

start_time = 9e3; end_time = 11e3
data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, window = 10)
data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = 1024, window = 10)

data.a1.ge.get_MSD(start_time=1000, end_time=10000, window = 15, jump_interval=np.array([15]), fit_stableDist='pylevy')
#title = "ee:{:.3f}, ie:{:.3f}, decay_p_i:{:.2f}".format(data.a1.param.scale_ee_1,data.a1.param.scale_ie_1,data.a1.param.decay_p_ie)
title = "ee:{:.3f}, ie:{:.3f}, dgk:{:.1f}, alpha:{:.2f}".format(data.a1.param.scale_ee_1,data.a1.param.scale_ie_1,data.a1.param.delta_gk,data.a1.ge.MSD.stableDist_param[0,0])

frames = data.a1.ge.spk_rate.spk_rate.shape[2]
ani = firing_rate_analysis.show_pattern(data.a1.ge.spk_rate.spk_rate, data.a1.gi.spk_rate.spk_rate, frames = frames, start_time = start_time, anititle=title)
#data.a1.param.scale_ie_1

#savetitle = "ee{:.3f}_ie{:.3f}_dpi{:.2f}".format(data.a1.param.scale_ee_1,data.a1.param.scale_ie_1,data.a1.param.decay_p_ie)
savetitle = "ee{:.3f}_ie{:.3f}_dgk{:.1f}_alpha{:.2f}".format(data.a1.param.scale_ee_1,data.a1.param.scale_ie_1,data.a1.param.delta_gk,data.a1.ge.MSD.stableDist_param[0,0])

ani.save(savetitle+'_%d'%loop_num+'.mp4')
del data.a1.ge.spk_rate

#title = "ee:{:.3f}, ie:{:.3f}, decay_p_i:{:.2f}".format(data.a1.param.scale_ee_1,data.a1.param.scale_ie_1,data.a1.param.decay_p_ie)
#savetitle = "ee{:.3f}_ie{:.3f}_dpi{:.2f}".format(data.a1.param.scale_ee_1,data.a1.param.scale_ie_1,data.a1.param.decay_p_ie)

title = "ee:{:.3f}, ie:{:.3f}, dgk:{:.1f}, alpha:{:.2f}".format(data.a1.param.scale_ee_1,data.a1.param.scale_ie_1,data.a1.param.delta_gk,data.a1.ge.MSD.stableDist_param[0,0])
savetitle = "ee{:.3f}_ie{:.3f}_dgk{:.1f}_alpha{:.2f}".format(data.a1.param.scale_ee_1,data.a1.param.scale_ie_1,data.a1.param.delta_gk,data.a1.ge.MSD.stableDist_param[0,0])

#%%
'''rate'''
e_lattice = cn.coordination.makelattice(63,62,[0,0])

chg_adapt_loca = [0, 0]
chg_adapt_range = 6
chg_adapt_neuron = cn.findnearbyneuron.findnearbyneuron(e_lattice, chg_adapt_loca, chg_adapt_range, 62)
#%%
start_time = 5e3; end_time = 18e3
data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, window = 10)
mua = data.a1.ge.spk_rate.spk_rate.reshape(3969,-1)[chg_adapt_neuron]
mua = mua.mean(0)/0.01

fig, ax1 = plt.subplots(1,1)
ax1.plot(np.arange(len(mua))+start_time, mua)
ax1.set_title(title)

fig.savefig(savetitle+'_rate_%d'%loop_num+'.png')

del data.a1.ge.spk_rate
#%%
''' MSD '''
data.a1.ge.get_MSD(start_time=1000, end_time=10000, window = 15, jump_interval=np.arange(1,1000,5), fit_stableDist='Matlab')

#data.save(data.class2dict(), datapath+'data%d.file'%loop_num)
#%%
fig, ax1 = plt.subplots(1,1)
ax1.loglog(data.a1.ge.MSD.jump_interval, data.a1.ge.MSD.MSD)

ax2 = ax1.twinx()
ax2.set_ylim([0,2.5])
err_up = data.a1.ge.MSD.stableDist_param[:,2,0] - data.a1.ge.MSD.stableDist_param[:,0,0]
err_down = data.a1.ge.MSD.stableDist_param[:,0,0] - data.a1.ge.MSD.stableDist_param[:,1,0]

googfit = np.abs(err_up - err_down)/(err_up + err_down) < 0.01
ax2.errorbar(data.a1.ge.MSD.jump_interval[googfit], \
             data.a1.ge.MSD.stableDist_param[googfit,0,0], \
             yerr=data.a1.ge.MSD.stableDist_param[googfit,2,0] - data.a1.ge.MSD.stableDist_param[googfit,0,0], fmt='ro')
#ax2.errorbar(data.a1.ge.MSD.jump_interval, \
#             data.a1.ge.MSD.stableDist_param[:], yerr=0)

ax1.set_title(title)
fig.savefig(savetitle+'_msd_%d'%loop_num+'.png')

#%%
data.save(data.class2dict(), datapath+'data%d.file'%loop_num)
