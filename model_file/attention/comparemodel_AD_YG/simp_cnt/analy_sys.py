#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 21:24:46 2020

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
import frequency_analysis as fa
import connection as cn
import pickle
import sys
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#%%
scale_d_p_ii = np.linspace(0.85,1.15,15)
num_ii = linspace(156, 297, 10,dtype=int)
scale_ie_1 = np.linspace(1.156-0.2,1.156+0.2,31)
#%%
#file_n = 0
size = np.zeros([scale_d_p_ii.shape[0], num_ii.shape[0], scale_ie_1.shape[0]])
pkf_spon = np.zeros(size.shape)
pkf_adapt = np.zeros(size.shape)
rate_spon = np.zeros(size.shape)
jump_dist = np.zeros(size.shape)
#%%
data_anly = mydata.mydata()
file_n = 0
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/dp_num_ii/'
for ii in range(scale_d_p_ii.shape[0]):
    for jj in range(num_ii.shape[0]):
        for kk in range(scale_ie_1.shape[0]):
            
            data_anly.load(datapath+'data_anly%d.file'%file_n)
            size[ii,jj,kk] = data_anly.patterns_size_mean
            pkf_spon[ii,jj,kk] = data_anly.peakF_spon
            pkf_adapt[ii,jj,kk] = data_anly.peakF_adapt
            rate_spon[ii,jj,kk] = data_anly.spon_rate
            jump_dist[ii,jj,kk] = data_anly.jump_dist_mean
            
            file_n += 1
            
#%%
#plt.figure()
ie_num = 15
plt.matshow(size[:,:,ie_num])
plt.matshow(pkf_spon[:,:,ie_num])
plt.matshow(pkf_adapt[:,:,ie_num])
plt.matshow(rate_spon[:,:,ie_num])
#%%
# xtick = np.arange(num_ii.shape[0])[::5]
# xticklabel = num_ii[::5]
# ytick = np.arange(scale_d_p_ii.shape[0])[::5]
# yticklabel = np.round(scale_d_p_ii,3)[::5]

#ie_num = 15
xparam = num_ii
yparam = scale_d_p_ii
xlabel = 'num_ii'
ylabel = 'scale_d_p_ii'
#%%
ie_num = 15
data = size[:,:,ie_num]
title = '''effects on patternsize
ie_ratio%.3f'''%(scale_ie_1[ie_num])
fig1 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
    
data = pkf_spon[:,:,ie_num]
title = '''effects on pkFreq_spon
ie_ratio%.3f'''%(scale_ie_1[ie_num])
fig2 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)

data = pkf_adapt[:,:,ie_num]
title = '''effects on pkFreq_adapt
ie_ratio%.3f'''%(scale_ie_1[ie_num])
fig3 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)

data = rate_spon[:,:,ie_num]
title = '''effects on rate_spon
ie_ratio%.3f'''%(scale_ie_1[ie_num])
fig4 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)

data = jump_dist[:,:,ie_num]
title = '''effects on jump_dist
ie_ratio%.3f'''%(scale_ie_1[ie_num])
fig5 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
#%%
xparam =  scale_ie_1
yparam = scale_d_p_ii
xlabel =  'scale_ie_1'
ylabel = 'scale_d_p_ii'
#'scale_d_p_ii' 'num_ii' 'scale_ie_1'
num_ii_num = 4
#d_p_ii_num = 7
data = pkf_spon[:,num_ii_num,:]
#title = '''effects_on_pkFreq_spon_scale_d_p_ii%.3f'''%(scale_d_p_ii[d_p_ii_num])
title = '''effects_on_pkFreq_spon_num_ii%.3f'''%(num_ii[num_ii_num])

fig2 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
#%%
plt.savefig(title+'.png')
#%%
data_ie = mydata.mydata()

data_ie.scale_d_p = np.linspace(0.85,1.15,13)
data_ie.num = np.linspace(93, 177, 10,dtype=int)
data_ie.scale_ie = np.linspace(1.156-0.15,1.156+0.15,25)
#%%
#file_n = 0
data_ie.size = np.zeros([data_ie.scale_d_p.shape[0], data_ie.num.shape[0], \
                         data_ie.scale_ie.shape[0]])
data_ie.pkf_spon = np.zeros(data_ie.size.shape)
data_ie.pkf_adapt = np.zeros(data_ie.size.shape)
data_ie.rate_spon = np.zeros(data_ie.size.shape)
data_ie.jump_dist = np.zeros(data_ie.size.shape)
data_ie.alpha = np.zeros(data_ie.size.shape)
#%%
data_anly = mydata.mydata()
file_n = 0
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/dp_num_ie/'
for ii in range(data_ie.scale_d_p.shape[0]):
    for jj in range(data_ie.num.shape[0]):
        for kk in range(data_ie.scale_ie.shape[0]):
            
            data_anly.load(datapath+'data_anly%d.file'%file_n)
            data_ie.size[ii,jj,kk] = data_anly.patterns_size_mean
            data_ie.pkf_spon[ii,jj,kk] = data_anly.peakF_spon
            data_ie.pkf_adapt[ii,jj,kk] = data_anly.peakF_adapt
            data_ie.rate_spon[ii,jj,kk] = data_anly.spon_rate
            data_ie.jump_dist[ii,jj,kk] = data_anly.jump_dist_mean
            data_ie.alpha[ii,jj,kk] = data_anly.alpha_dist[0,0]
            file_n += 1

#%%
xparam = data_ie.num
yparam = data_ie.scale_d_p
xlabel = 'num_ie'
ylabel = 'scale_d_p_ie'
#%%
ie_num = 12
data = data_ie.size[:,:,ie_num]
title = '''effects_on_patternsize
ie_scale%.3f'''%(data_ie.scale_ie[ie_num])
fig1 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
plt.savefig(title.replace('\n','_')+'.png') 

data = data_ie.pkf_spon[:,:,ie_num]
title = '''effects_on_pkFreq_spon
ie_scale%.3f'''%(data_ie.scale_ie[ie_num])
fig2 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
plt.savefig(title.replace('\n','_')+'.png') 

data = data_ie.pkf_adapt[:,:,ie_num]
title = '''effects_on_pkFreq_adapt
ie_scale%.3f'''%(data_ie.scale_ie[ie_num])
fig3 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
plt.savefig(title.replace('\n','_')+'.png') 

data = data_ie.rate_spon[:,:,ie_num]
title = '''effects_on_rate_spon
ie_scale%.3f'''%(data_ie.scale_ie[ie_num])
fig4 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
plt.savefig(title.replace('\n','_')+'.png') 

data = data_ie.jump_dist[:,:,ie_num]
title = '''effects_on_jump_dist
ie_scale%.3f'''%(data_ie.scale_ie[ie_num])
fig5 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
plt.savefig(title.replace('\n','_')+'.png') 
    
data = data_ie.alpha[:,:,ie_num]
title = '''effects_on_alpha
ie_scale%.3f'''%(data_ie.scale_ie[ie_num])
fig5 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
plt.savefig(title.replace('\n','_')+'.png') 

#%%
xparam =  scale_ie_1
yparam = scale_d_p_ii
xlabel =  'scale_ie_1'
ylabel = 'scale_d_p_ii'
#'scale_d_p_ii' 'num_ii' 'scale_ie_1'
num_ii_num = 4
#d_p_ii_num = 7
data = pkf_spon[:,num_ii_num,:]
#title = '''effects_on_pkFreq_spon_scale_d_p_ii%.3f'''%(scale_d_p_ii[d_p_ii_num])
title = '''effects_on_pkFreq_spon_num_ii%.3f'''%(num_ii[num_ii_num])

fig2 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
#%%
plt.savefig(title+'.png')
#%%
data_ei = mydata.mydata()

data_ei.scale_d_p = np.linspace(0.85,1.15,13)
data_ei.num = np.linspace(204, 466, 10, dtype=int)
data_ei.scale_ie = np.linspace(1.156-0.15,1.156+0.15,25)
#%%
#file_n = 0
data_ei.size = np.zeros([data_ei.scale_d_p.shape[0], data_ei.num.shape[0], \
                         data_ei.scale_ie.shape[0]])
data_ei.pkf_spon = np.zeros(data_ei.size.shape)
data_ei.pkf_adapt = np.zeros(data_ei.size.shape)
data_ei.rate_spon = np.zeros(data_ei.size.shape)
data_ei.jump_dist = np.zeros(data_ei.size.shape)
data_ei.alpha = np.zeros(data_ei.size.shape)
#%%
data_anly = mydata.mydata()
file_n = 0
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/dp_num_ei/'
for ii in range(data_ei.scale_d_p.shape[0]):
    for jj in range(data_ei.num.shape[0]):
        for kk in range(data_ei.scale_ie.shape[0]):
            
            data_anly.load(datapath+'data_anly%d.file'%file_n)
            data_ei.size[ii,jj,kk] = data_anly.patterns_size_mean
            data_ei.pkf_spon[ii,jj,kk] = data_anly.peakF_spon
            data_ei.pkf_adapt[ii,jj,kk] = data_anly.peakF_adapt
            data_ei.rate_spon[ii,jj,kk] = data_anly.spon_rate
            data_ei.jump_dist[ii,jj,kk] = data_anly.jump_dist_mean
            data_ei.alpha[ii,jj,kk] = data_anly.alpha_dist[0,0]
            file_n += 1

#%%
xparam = data_ei.num
yparam = data_ei.scale_d_p
xlabel = 'num_ei'
ylabel = 'scale_d_p_ei'
#%%
ie_num = 12
data = data_ei.size[:,:,ie_num]
title = '''effects_on_patternsize
ie_scale%.3f'''%(data_ei.scale_ie[ie_num])
fig1 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
plt.savefig(title.replace('\n','_')+'.png') 

data = data_ei.pkf_spon[:,:,ie_num]
title = '''effects_on_pkFreq_spon
ie_scale%.3f'''%(data_ei.scale_ie[ie_num])
fig2 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
plt.savefig(title.replace('\n','_')+'.png') 

data = data_ei.pkf_adapt[:,:,ie_num]
title = '''effects_on_pkFreq_adapt
ie_scale%.3f'''%(data_ei.scale_ie[ie_num])
fig3 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
plt.savefig(title.replace('\n','_')+'.png') 

data = data_ei.rate_spon[:,:,ie_num]
title = '''effects_on_rate_spon
ie_scale%.3f'''%(data_ei.scale_ie[ie_num])
fig4 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
plt.savefig(title.replace('\n','_')+'.png') 

data = data_ei.jump_dist[:,:,ie_num]
title = '''effects_on_jump_dist
ie_scale%.3f'''%(data_ei.scale_ie[ie_num])
fig5 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
plt.savefig(title.replace('\n','_')+'.png') 
    
data = data_ei.alpha[:,:,ie_num]
title = '''effects_on_alpha
ie_scale%.3f'''%(data_ei.scale_ie[ie_num])
fig5 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
plt.savefig(title.replace('\n','_')+'.png') 



#%%


#%%
data_ee = mydata.mydata()

data_ee.scale_d_p = np.linspace(0.85,1.15,13)
data_ee.num = np.linspace(125, 297, 10,dtype=int)
data_ee.scale_ie = np.linspace(1.156-0.15,1.156+0.15,25)
#%%
#file_n = 0
data_ee.size = np.zeros([data_ee.scale_d_p.shape[0], data_ee.num.shape[0], \
                         data_ee.scale_ie.shape[0]])
data_ee.pkf_spon = np.zeros(data_ee.size.shape)
data_ee.pkf_adapt = np.zeros(data_ee.size.shape)
data_ee.rate_spon = np.zeros(data_ee.size.shape)
data_ee.jump_dist = np.zeros(data_ee.size.shape)
data_ee.alpha = np.zeros(data_ee.size.shape)
#%%
data_anly = mydata.mydata()
file_n = 0
datapath = '/run/user/719/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-THAP/shencong/attention/tmp/dp_num_ee/'
for ii in range(data_ee.scale_d_p.shape[0]):
    for jj in range(data_ee.num.shape[0]):
        for kk in range(data_ee.scale_ie.shape[0]):
            
            try:
                data_anly.load(datapath+'data_anly%d.file'%file_n)
            except FileNotFoundError:
                file_n += 1
                break
            data_ee.size[ii,jj,kk] = data_anly.patterns_size_mean
            data_ee.pkf_spon[ii,jj,kk] = data_anly.peakF_spon
            data_ee.pkf_adapt[ii,jj,kk] = data_anly.peakF_adapt
            data_ee.rate_spon[ii,jj,kk] = data_anly.spon_rate
            data_ee.jump_dist[ii,jj,kk] = data_anly.jump_dist_mean
            try:
                data_ee.alpha[ii,jj,kk] = data_anly.alpha_dist[0,0]
            except IndexError:
                data_ee.alpha[ii,jj,kk] = data_anly.alpha_dist
            file_n += 1

#%%
xparam = data_ee.num
yparam = data_ee.scale_d_p
xlabel = 'num_ee'
ylabel = 'scale_d_p_ee'
#%%
ie_num = 12
data = data_ee.size[:,:,ie_num]
title = '''effects_on_patternsize
ie_scale%.3f'''%(data_ee.scale_ie[ie_num])
fig1 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
plt.savefig(title.replace('\n','_')+'.png') 

data = data_ee.pkf_spon[:,:,ie_num]
title = '''effects_on_pkFreq_spon
ie_scale%.3f'''%(data_ee.scale_ie[ie_num])
fig2 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
plt.savefig(title.replace('\n','_')+'.png') 

data = data_ee.pkf_adapt[:,:,ie_num]
title = '''effects_on_pkFreq_adapt
ie_scale%.3f'''%(data_ee.scale_ie[ie_num])
fig3 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
plt.savefig(title.replace('\n','_')+'.png') 

data = data_ee.rate_spon[:,:,ie_num]
title = '''effects_on_rate_spon
ie_scale%.3f'''%(data_ee.scale_ie[ie_num])
fig4 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
plt.savefig(title.replace('\n','_')+'.png') 

data = data_ee.jump_dist[:,:,ie_num]
title = '''effects_on_jump_dist
ie_scale%.3f'''%(data_ee.scale_ie[ie_num])
fig5 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
plt.savefig(title.replace('\n','_')+'.png') 
    
data = data_ee.alpha[:,:,ie_num]
title = '''effects_on_alpha
ie_scale%.3f'''%(data_ee.scale_ie[ie_num])
fig5 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
plt.savefig(title.replace('\n','_')+'.png') 



#%%
def plot_data(data, xparam, xlabel,\
             yparam, ylabel, title):
    
# (data, xtick, xticklabel, xlabel, \
#               ytick, yticklabel, ylabel, title):
    
    xtick = np.arange(xparam.shape[0])[::2]
    xticklabel = np.round(xparam,3)[::2]
    ytick = np.arange(yparam.shape[0])[::2]
    yticklabel = np.round(yparam,3)[::2]
    
    fig,ax1 = plt.subplots(1,1, figsize=[6,6])
    im = ax1.matshow(data)
    plt.colorbar(im, ax=ax1)
    ax1.set_xticks(xtick)#np.arange(scale_ie_1.shape[0])[::5])
    ax1.set_xticklabels(xticklabel)#(np.round(scale_ie_1,3)[::5])
    ax1.set_yticks(ytick)#np.arange(scale_ie_1.shape[0])[::5])
    ax1.set_yticklabels(yticklabel)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    return fig




