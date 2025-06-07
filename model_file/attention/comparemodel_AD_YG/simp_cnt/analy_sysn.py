#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 22:25:17 2020

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
data_nee = mydata.mydata()
data_nee.n = np.linspace(220,290,8)
data_nee.ie_r_e = np.arange(3.05,3.36,0.05)#np.linspace(1.4,1.6,10):#[1.5]: #np.linspace(1.,1.2,5):
# data_nee.w_ie = np.linspace(17,23,7)#np.linspace(1.077,1.277,10):#[1.177]:#np.linspace(1,1,1):
# data_nee.ie_r_i = np.linspace(2.5,3.0,15)#np.linspace(1.156-0.08,1.156+0.08,15):#np.linspace(1.156-0.175,1.156+0.175,31):#np.linspace(1.157,1.357,10):#[1.257]: #np.linspace(0.95,1.1,5):
# data_nee.w_ii = np.linspace(20,27,8)#
#%%
# data_nee.scale_d_p = np.linspace(0.85,1.15,13)
# data_nee.num = np.linspace(204, 466, 10, dtype=int)
# data_nee.scale_ie = np.linspace(1.156-0.15,1.156+0.15,25)
#%%
#file_n = 0
data_nee.size = np.zeros([data_nee.n.shape[0], data_nee.ie_r_e.shape[0]])
data_nee.pkf_spon = np.zeros(data_nee.size.shape)
data_nee.pkf_adapt = np.zeros(data_nee.size.shape)
data_nee.rate_spon = np.zeros(data_nee.size.shape)
data_nee.jump_dist = np.zeros(data_nee.size.shape)
data_nee.alpha = np.zeros(data_nee.size.shape)
data_nee.pattern_on_ratio = np.zeros(data_nee.size.shape)
#%%
data_anly = mydata.mydata()
file_n = 0
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/num/nee/'
for ii in range(data_nee.n.shape[0]):
    for jj in range(data_nee.ie_r_e.shape[0]):
        # for kk in range(data_nee.ie_r_i.shape[0]):
        #     for ll in range(data_nee.w_ii.shape[0]):
                
        data_anly.load(datapath+'data_anly%d.file'%file_n)
        data_nee.size[ii,jj] = data_anly.patterns_size_mean
        data_nee.pkf_spon[ii,jj] = data_anly.peakF_spon
        data_nee.pkf_adapt[ii,jj] = data_anly.peakF_adapt
        data_nee.rate_spon[ii,jj] = data_anly.spon_rate
        data_nee.jump_dist[ii,jj] = data_anly.jump_dist_mean
        data_nee.alpha[ii,jj] = data_anly.alpha_dist[0,0]
        data_nee.pattern_on_ratio[ii,jj] = data_anly.pattern_on_ratio
        file_n += 1
#%%
xparam = data_nee.ie_r_e
yparam = data_nee.n
xlabel = 'ie_r_e'
ylabel = 'nee'
#%%
savefig = 1
#ie_r_e_ind = 3; w_ie_ind = 3
data = data_nee.size#[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_patternsize'''#%(data_nee.ie_r_e[ie_r_e_ind], data_nee.w_ie[w_ie_ind])
fig1 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

data = data_nee.pkf_spon#[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_pkFreq_spon'''#%(data_nee.ie_r_e[ie_r_e_ind], data_nee.w_ie[w_ie_ind])
fig2 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title, clim=1)
if savefig: plt.savefig(title.replace('\n','_')+'.png')

data = data_nee.pkf_adapt#[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_pkFreq_adapt'''#%(data_nee.ie_r_e[ie_r_e_ind], data_nee.w_ie[w_ie_ind])
fig3 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title, clim=1)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

data = data_nee.rate_spon#[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_rate_spon'''#%(data_nee.ie_r_e[ie_r_e_ind], data_nee.w_ie[w_ie_ind])
fig4 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

data = data_nee.jump_dist#[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_jump_dist'''#%(data_nee.ie_r_e[ie_r_e_ind], data_nee.w_ie[w_ie_ind])
fig5 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 
    
data = data_nee.alpha#[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_alpha'''#%(data_nee.ie_r_e[ie_r_e_ind], data_nee.w_ie[w_ie_ind])
fig5 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

data = data_nee.pattern_on_ratio#[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_pattern_on_ratio'''#%(data_nee.ie_r_e[ie_r_e_ind], data_nee.w_ie[w_ie_ind])
fig5 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

#%%
data_nie = mydata.mydata()
data_nie.n = np.linspace(145,205,7)
data_nie.ie_r_e = np.arange(3.05,3.36,0.05)#np.linspace(1.4,1.6,10):#[1.5]: #np.linspace(1.,1.2,5):
# data_nie.w_ie = np.linspace(17,23,7)#np.linspace(1.077,1.277,10):#[1.177]:#np.linspace(1,1,1):
# data_nie.ie_r_i = np.linspace(2.5,3.0,15)#np.linspace(1.156-0.08,1.156+0.08,15):#np.linspace(1.156-0.175,1.156+0.175,31):#np.linspace(1.157,1.357,10):#[1.257]: #np.linspace(0.95,1.1,5):
# data_nie.w_ii = np.linspace(20,27,8)#
#%%
# data_nie.scale_d_p = np.linspace(0.85,1.15,13)
# data_nie.num = np.linspace(204, 466, 10, dtype=int)
# data_nie.scale_ie = np.linspace(1.156-0.15,1.156+0.15,25)
#%%
#file_n = 0
data_nie.size = np.zeros([data_nie.n.shape[0], data_nie.ie_r_e.shape[0]])
data_nie.pkf_spon = np.zeros(data_nie.size.shape)
data_nie.pkf_adapt = np.zeros(data_nie.size.shape)
data_nie.rate_spon = np.zeros(data_nie.size.shape)
data_nie.jump_dist = np.zeros(data_nie.size.shape)
data_nie.alpha = np.zeros(data_nie.size.shape)
data_nie.pattern_on_ratio = np.zeros(data_nie.size.shape)
#%%
data_anly = mydata.mydata()
file_n = 0
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/num/nie/'
for ii in range(data_nie.n.shape[0]):
    for jj in range(data_nie.ie_r_e.shape[0]):
        # for kk in range(data_nie.ie_r_i.shape[0]):
        #     for ll in range(data_nie.w_ii.shape[0]):
                
        data_anly.load(datapath+'data_anly%d.file'%file_n)
        data_nie.size[ii,jj] = data_anly.patterns_size_mean
        data_nie.pkf_spon[ii,jj] = data_anly.peakF_spon
        data_nie.pkf_adapt[ii,jj] = data_anly.peakF_adapt
        data_nie.rate_spon[ii,jj] = data_anly.spon_rate
        data_nie.jump_dist[ii,jj] = data_anly.jump_dist_mean
        data_nie.alpha[ii,jj] = data_anly.alpha_dist[0,0]
        data_nie.pattern_on_ratio[ii,jj] = data_anly.pattern_on_ratio
        file_n += 1
#%%
xparam = data_nie.ie_r_e
yparam = data_nie.n
xlabel = 'ie_r_e'
ylabel = 'nie'
#%%
savefig = 1
#ie_r_e_ind = 3; w_ie_ind = 3
data = data_nie.size#[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_patternsize'''#%(data_nie.ie_r_e[ie_r_e_ind], data_nie.w_ie[w_ie_ind])
fig1 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

data = data_nie.pkf_spon#[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_pkFreq_spon'''#%(data_nie.ie_r_e[ie_r_e_ind], data_nie.w_ie[w_ie_ind])
fig2 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title, clim=1)
if savefig: plt.savefig(title.replace('\n','_')+'.png')

data = data_nie.pkf_adapt#[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_pkFreq_adapt'''#%(data_nie.ie_r_e[ie_r_e_ind], data_nie.w_ie[w_ie_ind])
fig3 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title, clim=1)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

data = data_nie.rate_spon#[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_rate_spon'''#%(data_nie.ie_r_e[ie_r_e_ind], data_nie.w_ie[w_ie_ind])
fig4 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

data = data_nie.jump_dist#[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_jump_dist'''#%(data_nie.ie_r_e[ie_r_e_ind], data_nie.w_ie[w_ie_ind])
fig5 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 
    
data = data_nie.alpha#[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_alpha'''#%(data_nie.ie_r_e[ie_r_e_ind], data_nie.w_ie[w_ie_ind])
fig5 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

data = data_nie.pattern_on_ratio#[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_pattern_on_ratio'''#%(data_nie.ie_r_e[ie_r_e_ind], data_nie.w_ie[w_ie_ind])
fig5 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

#%%
data_nei = mydata.mydata()
data_nei.n = np.linspace(360,440,9)
data_nei.ie_r_i = np.arange(2.8,3.11,0.05)#np.linspace(1.4,1.6,10):#[1.5]: #np.linspace(1.,1.2,5):
# data_nei.w_ie = np.linspace(17,23,7)#np.linspace(1.077,1.277,10):#[1.177]:#np.linspace(1,1,1):
# data_nei.ie_r_i = np.linspace(2.5,3.0,15)#np.linspace(1.156-0.08,1.156+0.08,15):#np.linspace(1.156-0.175,1.156+0.175,31):#np.linspace(1.157,1.357,10):#[1.257]: #np.linspace(0.95,1.1,5):
# data_nei.w_ii = np.linspace(20,27,8)#
#%%
# data_nei.scale_d_p = np.linspace(0.85,1.15,13)
# data_nei.num = np.linspace(204, 466, 10, dtype=int)
# data_nei.scale_ie = np.linspace(1.156-0.15,1.156+0.15,25)
#%%
#file_n = 0
data_nei.size = np.zeros([data_nei.n.shape[0], data_nei.ie_r_e.shape[0]])
data_nei.pkf_spon = np.zeros(data_nei.size.shape)
data_nei.pkf_adapt = np.zeros(data_nei.size.shape)
data_nei.rate_spon = np.zeros(data_nei.size.shape)
data_nei.jump_dist = np.zeros(data_nei.size.shape)
data_nei.alpha = np.zeros(data_nei.size.shape)
data_nei.pattern_on_ratio = np.zeros(data_nei.size.shape)
#%%
data_anly = mydata.mydata()
file_n = 0
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/num/nei/'
for ii in range(data_nei.n.shape[0]):
    for jj in range(data_nei.ie_r_i.shape[0]):
        # for kk in range(data_nei.ie_r_i.shape[0]):
        #     for ll in range(data_nei.w_ii.shape[0]):
                
        data_anly.load(datapath+'data_anly%d.file'%file_n)
        data_nei.size[ii,jj] = data_anly.patterns_size_mean
        data_nei.pkf_spon[ii,jj] = data_anly.peakF_spon
        data_nei.pkf_adapt[ii,jj] = data_anly.peakF_adapt
        data_nei.rate_spon[ii,jj] = data_anly.spon_rate
        data_nei.jump_dist[ii,jj] = data_anly.jump_dist_mean
        data_nei.alpha[ii,jj] = data_anly.alpha_dist[0,0]
        data_nei.pattern_on_ratio[ii,jj] = data_anly.pattern_on_ratio
        file_n += 1
#%%
xparam = data_nei.ie_r_i
yparam = data_nei.n
xlabel = 'ie_r_i'
ylabel = 'nei'
#%%
savefig = 1
#ie_r_e_ind = 3; w_ie_ind = 3
data = data_nei.size#[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_patternsize'''#%(data_nei.ie_r_e[ie_r_e_ind], data_nei.w_ie[w_ie_ind])
fig1 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

data = data_nei.pkf_spon#[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_pkFreq_spon'''#%(data_nei.ie_r_e[ie_r_e_ind], data_nei.w_ie[w_ie_ind])
fig2 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title, clim=1)
if savefig: plt.savefig(title.replace('\n','_')+'.png')

data = data_nei.pkf_adapt#[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_pkFreq_adapt'''#%(data_nei.ie_r_e[ie_r_e_ind], data_nei.w_ie[w_ie_ind])
fig3 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title, clim=1)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

data = data_nei.rate_spon#[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_rate_spon'''#%(data_nei.ie_r_e[ie_r_e_ind], data_nei.w_ie[w_ie_ind])
fig4 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

data = data_nei.jump_dist#[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_jump_dist'''#%(data_nei.ie_r_e[ie_r_e_ind], data_nei.w_ie[w_ie_ind])
fig5 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 
    
data = data_nei.alpha#[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_alpha'''#%(data_nei.ie_r_e[ie_r_e_ind], data_nei.w_ie[w_ie_ind])
fig5 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

data = data_nei.pattern_on_ratio#[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_pattern_on_ratio'''#%(data_nei.ie_r_e[ie_r_e_ind], data_nei.w_ie[w_ie_ind])
fig5 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 
#%%
data_nii = mydata.mydata()
data_nii.n = np.linspace(220,300,9)
data_nii.ie_r_i = np.arange(2.8,3.11,0.05)#np.linspace(1.4,1.6,10):#[1.5]: #np.linspace(1.,1.2,5):
# data_nii.w_ie = np.linspace(17,23,7)#np.linspace(1.077,1.277,10):#[1.177]:#np.linspace(1,1,1):
# data_nii.ie_r_i = np.linspace(2.5,3.0,15)#np.linspace(1.156-0.08,1.156+0.08,15):#np.linspace(1.156-0.175,1.156+0.175,31):#np.linspace(1.157,1.357,10):#[1.257]: #np.linspace(0.95,1.1,5):
# data_nii.w_ii = np.linspace(20,27,8)#
#%%
# data_nii.scale_d_p = np.linspace(0.85,1.15,13)
# data_nii.num = np.linspace(204, 466, 10, dtype=int)
# data_nii.scale_ie = np.linspace(1.156-0.15,1.156+0.15,25)
#%%
#file_n = 0
data_nii.size = np.zeros([data_nii.n.shape[0], data_nii.ie_r_i.shape[0]])
data_nii.pkf_spon = np.zeros(data_nii.size.shape)
data_nii.pkf_adapt = np.zeros(data_nii.size.shape)
data_nii.rate_spon = np.zeros(data_nii.size.shape)
data_nii.jump_dist = np.zeros(data_nii.size.shape)
data_nii.alpha = np.zeros(data_nii.size.shape)
data_nii.pattern_on_ratio = np.zeros(data_nii.size.shape)
#%%
data_anly = mydata.mydata()
file_n = 0
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/num/nii/'
for ii in range(data_nii.n.shape[0]):
    for jj in range(data_nii.ie_r_i.shape[0]):
        # for kk in range(data_nii.ie_r_i.shape[0]):
        #     for ll in range(data_nii.w_ii.shape[0]):
                
        data_anly.load(datapath+'data_anly%d.file'%file_n)
        data_nii.size[ii,jj] = data_anly.patterns_size_mean
        data_nii.pkf_spon[ii,jj] = data_anly.peakF_spon
        data_nii.pkf_adapt[ii,jj] = data_anly.peakF_adapt
        data_nii.rate_spon[ii,jj] = data_anly.spon_rate
        data_nii.jump_dist[ii,jj] = data_anly.jump_dist_mean
        data_nii.alpha[ii,jj] = data_anly.alpha_dist[0,0]
        data_nii.pattern_on_ratio[ii,jj] = data_anly.pattern_on_ratio
        file_n += 1
#%%
xparam = data_nii.ie_r_i
yparam = data_nii.n
xlabel = 'ie_r_i'
ylabel = 'nii'
#%%
savefig = 1
#ie_r_e_ind = 3; w_ie_ind = 3
data = data_nii.size#[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_patternsize'''#%(data_nii.ie_r_e[ie_r_e_ind], data_nii.w_ie[w_ie_ind])
fig1 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

data = data_nii.pkf_spon#[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_pkFreq_spon'''#%(data_nii.ie_r_e[ie_r_e_ind], data_nii.w_ie[w_ie_ind])
fig2 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title, clim=1)
if savefig: plt.savefig(title.replace('\n','_')+'.png')

data = data_nii.pkf_adapt#[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_pkFreq_adapt'''#%(data_nii.ie_r_e[ie_r_e_ind], data_nii.w_ie[w_ie_ind])
fig3 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title, clim=1)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

data = data_nii.rate_spon#[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_rate_spon'''#%(data_nii.ie_r_e[ie_r_e_ind], data_nii.w_ie[w_ie_ind])
fig4 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

data = data_nii.jump_dist#[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_jump_dist'''#%(data_nii.ie_r_e[ie_r_e_ind], data_nii.w_ie[w_ie_ind])
fig5 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 
    
data = data_nii.alpha#[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_alpha'''#%(data_nii.ie_r_e[ie_r_e_ind], data_nii.w_ie[w_ie_ind])
fig5 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

data = data_nii.pattern_on_ratio#[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_pattern_on_ratio'''#%(data_nii.ie_r_e[ie_r_e_ind], data_nii.w_ie[w_ie_ind])
fig5 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

#%%
data_dgk = mydata.mydata()
#data_dgk.n = np.linspace(220,300,9)
data_dgk.delta_gk = np.linspace(5,15,11)
data_dgk.ie_r_e = np.linspace(3.10,3.2,3)
data_dgk.ie_r_i = np.linspace(2.65,2.75,3)#np.linspace(1.4,1.6,10):#[1.5]: #np.linspace(1.,1.2,5):
    
data_dgk.size = np.zeros([data_dgk.ie_r_e.shape[0], data_dgk.ie_r_i.shape[0], data_dgk.delta_gk.shape[0]])
data_dgk.pkf_spon = np.zeros(data_dgk.size.shape)
data_dgk.pkf_adapt = np.zeros(data_dgk.size.shape)
data_dgk.rate_spon = np.zeros(data_dgk.size.shape)
data_dgk.jump_dist = np.zeros(data_dgk.size.shape)
data_dgk.alpha = np.zeros(data_dgk.size.shape)
data_dgk.pattern_on_ratio = np.zeros(data_dgk.size.shape)
#%%
data_anly = mydata.mydata()
file_n = 0
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/dgkS2/'
for ii in range(data_dgk.ie_r_e.shape[0]):
    for jj in range(data_dgk.ie_r_i.shape[0]):
         for kk in range(data_dgk.delta_gk.shape[0]):
        #     for ll in range(data_dgk.w_ii.shape[0]):
                
            data_anly.load(datapath+'data_anly%d.file'%file_n)
            data_dgk.size[ii,jj,kk] = data_anly.patterns_size_mean
            data_dgk.pkf_spon[ii,jj,kk] = data_anly.peakF_spon
            data_dgk.pkf_adapt[ii,jj,kk] = data_anly.peakF_adapt
            data_dgk.rate_spon[ii,jj,kk] = data_anly.spon_rate
            data_dgk.jump_dist[ii,jj,kk] = data_anly.jump_dist_mean
            data_dgk.alpha[ii,jj,kk] = data_anly.alpha_dist[0,0]
            data_dgk.pattern_on_ratio[ii,jj,kk] = data_anly.pattern_on_ratio
            file_n += 1
#%%
xparam = data_dgk.delta_gk
yparam = data_dgk.ie_r_e
xlabel = 'delta_gk'
ylabel = 'ie_r_e'

figa = visualize_data(data_dgk, np.s_[:,1,:], xparam, yparam, xlabel, ylabel, savefig = 1)
#%%
def plot_data(data, xparam, xlabel,\
             yparam, ylabel, title, clim=None):
    
# (data, xtick, xticklabel, xlabel, \
#               ytick, yticklabel, ylabel, title):
    
    xtick = np.arange(xparam.shape[0])[::2]
    xticklabel = np.round(xparam,3)[::2]
    ytick = np.arange(yparam.shape[0])[::2]
    yticklabel = np.round(yparam,3)[::2]
    
    fig,ax1 = plt.subplots(1,1, figsize=[6,3])
    im = ax1.matshow(data)
    plt.colorbar(im, ax=ax1)
    if clim is not None: im.set_clim(2, 8)
    ax1.set_xticks(xtick)#np.arange(scale_ie_1.shape[0])[::5])
    ax1.set_xticklabels(xticklabel)#(np.round(scale_ie_1,3)[::5])
    ax1.set_yticks(ytick)#np.arange(scale_ie_1.shape[0])[::5])
    ax1.set_yticklabels(yticklabel)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    return fig
#%%
def visualize_data(data_in, slic, xparam, yparam, xlabel, ylabel, savefig = 0):
    
    #ie_r_e_ind = 3; w_ie_ind = 3
    data = data_in.size[slic]#[ie_r_e_ind,w_ie_ind,:,:]
    title = '''effects_on_patternsize'''#%(data_in.ie_r_e[ie_r_e_ind], data_in.w_ie[w_ie_ind])
    fig1 = plot_data(data, xparam, xlabel, \
                  yparam, ylabel, title)
    if savefig: plt.savefig(title.replace('\n','_')+'.png') 
    
    data = data_in.pkf_spon[slic]#[ie_r_e_ind,w_ie_ind,:,:]
    title = '''effects_on_pkFreq_spon'''#%(data_in.ie_r_e[ie_r_e_ind], data_in.w_ie[w_ie_ind])
    fig2 = plot_data(data, xparam, xlabel, \
                  yparam, ylabel, title, clim=1)
    if savefig: plt.savefig(title.replace('\n','_')+'.png')
    
    data = data_in.pkf_adapt[slic]#[ie_r_e_ind,w_ie_ind,:,:]
    title = '''effects_on_pkFreq_adapt'''#%(data_in.ie_r_e[ie_r_e_ind], data_in.w_ie[w_ie_ind])
    fig3 = plot_data(data, xparam, xlabel, \
                  yparam, ylabel, title, clim=1)
    if savefig: plt.savefig(title.replace('\n','_')+'.png') 
    
    data = data_in.rate_spon[slic]#[ie_r_e_ind,w_ie_ind,:,:]
    title = '''effects_on_rate_spon'''#%(data_in.ie_r_e[ie_r_e_ind], data_in.w_ie[w_ie_ind])
    fig4 = plot_data(data, xparam, xlabel, \
                  yparam, ylabel, title)
    if savefig: plt.savefig(title.replace('\n','_')+'.png') 
    
    data = data_in.jump_dist[slic]#[ie_r_e_ind,w_ie_ind,:,:]
    title = '''effects_on_jump_dist'''#%(data_in.ie_r_e[ie_r_e_ind], data_in.w_ie[w_ie_ind])
    fig5 = plot_data(data, xparam, xlabel, \
                  yparam, ylabel, title)
    if savefig: plt.savefig(title.replace('\n','_')+'.png') 
        
    data = data_in.alpha[slic]#[ie_r_e_ind,w_ie_ind,:,:]
    title = '''effects_on_alpha'''#%(data_in.ie_r_e[ie_r_e_ind], data_in.w_ie[w_ie_ind])
    fig6 = plot_data(data, xparam, xlabel, \
                  yparam, ylabel, title)
    if savefig: plt.savefig(title.replace('\n','_')+'.png') 
    
    data = data_in.pattern_on_ratio[slic]#[ie_r_e_ind,w_ie_ind,:,:]
    title = '''effects_on_pattern_on_ratio'''#%(data_in.ie_r_e[ie_r_e_ind], data_in.w_ie[w_ie_ind])
    fig7 = plot_data(data, xparam, xlabel, \
                  yparam, ylabel, title)
    if savefig: plt.savefig(title.replace('\n','_')+'.png') 
    
    return fig1, fig2, fig3, fig4, fig5, fig6, fig7