#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 22:36:04 2020

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
data_wt = mydata.mydata()

data_wt.ie_r_e = np.linspace(3.10,3.45,6)#np.linspace(1.4,1.6,10):#[1.5]: #np.linspace(1.,1.2,5):
data_wt.w_ie = np.linspace(17,23,7)#np.linspace(1.077,1.277,10):#[1.177]:#np.linspace(1,1,1):
data_wt.ie_r_i = np.linspace(2.5,3.0,15)#np.linspace(1.156-0.08,1.156+0.08,15):#np.linspace(1.156-0.175,1.156+0.175,31):#np.linspace(1.157,1.357,10):#[1.257]: #np.linspace(0.95,1.1,5):
data_wt.w_ii = np.linspace(20,27,8)#
#%%
# data_wt.scale_d_p = np.linspace(0.85,1.15,13)
# data_wt.num = np.linspace(204, 466, 10, dtype=int)
# data_wt.scale_ie = np.linspace(1.156-0.15,1.156+0.15,25)
#%%
#file_n = 0
data_wt.size = np.zeros([data_wt.ie_r_e.shape[0], data_wt.w_ie.shape[0], \
                         data_wt.ie_r_i.shape[0], data_wt.w_ii.shape[0]])
data_wt.pkf_spon = np.zeros(data_wt.size.shape)
data_wt.pkf_adapt = np.zeros(data_wt.size.shape)
data_wt.rate_spon = np.zeros(data_wt.size.shape)
data_wt.jump_dist = np.zeros(data_wt.size.shape)
data_wt.alpha = np.zeros(data_wt.size.shape)
data_wt.pattern_on_ratio = np.zeros(data_wt.size.shape)
#%%
data_anly = mydata.mydata()
file_n = 0
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/wt/'
for ii in range(data_wt.ie_r_e.shape[0]):
    for jj in range(data_wt.w_ie.shape[0]):
        for kk in range(data_wt.ie_r_i.shape[0]):
            for ll in range(data_wt.w_ii.shape[0]):
                
                data_anly.load(datapath+'data_anly%d.file'%file_n)
                data_wt.size[ii,jj,kk,ll] = data_anly.patterns_size_mean
                data_wt.pkf_spon[ii,jj,kk,ll] = data_anly.peakF_spon
                data_wt.pkf_adapt[ii,jj,kk,ll] = data_anly.peakF_adapt
                data_wt.rate_spon[ii,jj,kk,ll] = data_anly.spon_rate
                data_wt.jump_dist[ii,jj,kk,ll] = data_anly.jump_dist_mean
                data_wt.alpha[ii,jj,kk,ll] = data_anly.alpha_dist[0,0]
                data_wt.pattern_on_ratio[ii,jj,kk,ll] = data_anly.pattern_on_ratio
                file_n += 1

#%%
xparam = data_wt.w_ie
yparam = data_wt.ie_r_e
xlabel = 'w_ie'
ylabel = 'ie_r_e'
#%%
savefig = 1
ie_r_i_ind = 7; w_ii_ind = 3
data = data_wt.size[:,:,ie_r_i_ind,w_ii_ind]
title = '''effects_on_patternsize
ie_r_i%.3f_w_ii:%.1f'''%(data_wt.ie_r_i[ie_r_i_ind], data_wt.w_ii[w_ii_ind])
fig1 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

data = data_wt.pkf_spon[:,:,ie_r_i_ind,w_ii_ind]
title = '''effects_on_pkFreq_spon
ie_r_i%.3f_w_ii:%.1f'''%(data_wt.ie_r_i[ie_r_i_ind], data_wt.w_ii[w_ii_ind])
fig2 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title, clim=1)
if savefig: plt.savefig(title.replace('\n','_')+'.png')

data = data_wt.pkf_adapt[:,:,ie_r_i_ind,w_ii_ind]
title = '''effects_on_pkFreq_adapt
ie_r_i%.3f_w_ii:%.1f'''%(data_wt.ie_r_i[ie_r_i_ind], data_wt.w_ii[w_ii_ind])
fig3 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title, clim=1)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

data = data_wt.rate_spon[:,:,ie_r_i_ind,w_ii_ind]
title = '''effects_on_rate_spon
ie_r_i%.3f_w_ii:%.1f'''%(data_wt.ie_r_i[ie_r_i_ind], data_wt.w_ii[w_ii_ind])
fig4 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

data = data_wt.jump_dist[:,:,ie_r_i_ind,w_ii_ind]
title = '''effects_on_jump_dist
ie_r_i%.3f_w_ii:%.1f'''%(data_wt.ie_r_i[ie_r_i_ind], data_wt.w_ii[w_ii_ind])
fig5 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 
    
data = data_wt.alpha[:,:,ie_r_i_ind,w_ii_ind]
title = '''effects_on_alpha
ie_r_i%.3f_w_ii:%.1f'''%(data_wt.ie_r_i[ie_r_i_ind], data_wt.w_ii[w_ii_ind])
fig5 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

data = data_wt.pattern_on_ratio[:,:,ie_r_i_ind,w_ii_ind]
title = '''effects_on_pattern_on_ratio
ie_r_i%.3f_w_ii:%.1f'''%(data_wt.ie_r_i[ie_r_i_ind], data_wt.w_ii[w_ii_ind])
fig5 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png')
#%%
xparam = data_wt.w_ii
yparam = data_wt.ie_r_i
xlabel = 'w_ii'
ylabel = 'ie_r_i'
#%%
savefig = 0
ie_r_e_ind = 3; w_ie_ind = 3
data = data_wt.size[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_patternsize
ie_r_e%.3f_w_ie:%.1f'''%(data_wt.ie_r_e[ie_r_e_ind], data_wt.w_ie[w_ie_ind])
fig1 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

data = data_wt.pkf_spon[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_pkFreq_spon
ie_r_e%.3f_w_ie:%.1f'''%(data_wt.ie_r_e[ie_r_e_ind], data_wt.w_ie[w_ie_ind])
fig2 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title, clim=1)
if savefig: plt.savefig(title.replace('\n','_')+'.png')

data = data_wt.pkf_adapt[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_pkFreq_adapt
ie_r_e%.3f_w_ie:%.1f'''%(data_wt.ie_r_e[ie_r_e_ind], data_wt.w_ie[w_ie_ind])
fig3 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title, clim=1)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

data = data_wt.rate_spon[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_rate_spon
ie_r_e%.3f_w_ie:%.1f'''%(data_wt.ie_r_e[ie_r_e_ind], data_wt.w_ie[w_ie_ind])
fig4 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

data = data_wt.jump_dist[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_jump_dist
ie_r_e%.3f_w_ie:%.1f'''%(data_wt.ie_r_e[ie_r_e_ind], data_wt.w_ie[w_ie_ind])
fig5 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 
    
data = data_wt.alpha[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_alpha
ie_r_e%.3f_w_ie:%.1f'''%(data_wt.ie_r_e[ie_r_e_ind], data_wt.w_ie[w_ie_ind])
fig5 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

data = data_wt.pattern_on_ratio[ie_r_e_ind,w_ie_ind,:,:]
title = '''effects_on_pattern_on_ratio
ie_r_e%.3f_w_ie:%.1f'''%(data_wt.ie_r_e[ie_r_e_ind], data_wt.w_ie[w_ie_ind])
fig5 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 
#%%
xparam = data_wt.ie_r_i
yparam = data_wt.w_ie
xlabel = 'ie_r_i'
ylabel = 'w_ie'

#%%
savefig = 1
ie_r_e_ind = 3; w_ii_ind = 3
data = data_wt.size[ie_r_e_ind,:,:,w_ii_ind]
title = '''effects_on_patternsize
ie_r_e%.3f_w_ii:%.1f'''%(data_wt.ie_r_e[ie_r_e_ind], data_wt.w_ii[w_ie_ind])
fig1 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

data = data_wt.pkf_spon[ie_r_e_ind,:,:,w_ii_ind]
title = '''effects_on_pkFreq_spon
ie_r_e%.3f_w_ii:%.1f'''%(data_wt.ie_r_e[ie_r_e_ind], data_wt.w_ii[w_ie_ind])
fig2 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title, clim=1)
if savefig: plt.savefig(title.replace('\n','_')+'.png')

data = data_wt.pkf_adapt[ie_r_e_ind,:,:,w_ii_ind]
title = '''effects_on_pkFreq_adapt
ie_r_e%.3f_w_ii:%.1f'''%(data_wt.ie_r_e[ie_r_e_ind], data_wt.w_ii[w_ie_ind])
fig3 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title, clim=1)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

data = data_wt.rate_spon[ie_r_e_ind,:,:,w_ii_ind]
title = '''effects_on_rate_spon
ie_r_e%.3f_w_ii:%.1f'''%(data_wt.ie_r_e[ie_r_e_ind], data_wt.w_ii[w_ie_ind])
fig4 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

data = data_wt.jump_dist[ie_r_e_ind,:,:,w_ii_ind]
title = '''effects_on_jump_dist
ie_r_e%.3f_w_ii:%.1f'''%(data_wt.ie_r_e[ie_r_e_ind], data_wt.w_ii[w_ie_ind])
fig5 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 
    
data = data_wt.alpha[ie_r_e_ind,:,:,w_ii_ind]
title = '''effects_on_alpha
ie_r_e%.3f_w_ii:%.1f'''%(data_wt.ie_r_e[ie_r_e_ind], data_wt.w_ii[w_ie_ind])
fig5 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

data = data_wt.pattern_on_ratio[ie_r_e_ind,:,:,w_ii_ind]
title = '''effects_on_pattern_on_ratio
ie_r_e%.3f_w_ii:%.1f'''%(data_wt.ie_r_e[ie_r_e_ind], data_wt.w_ii[w_ie_ind])
fig5 = plot_data(data, xparam, xlabel, \
              yparam, ylabel, title)
if savefig: plt.savefig(title.replace('\n','_')+'.png') 

 
#%%
def plot_data(data, xparam, xlabel,\
             yparam, ylabel, title, clim=None):
    
# (data, xtick, xticklabel, xlabel, \
#               ytick, yticklabel, ylabel, title):
    
    xtick = np.arange(xparam.shape[0])[::2]
    xticklabel = np.round(xparam,3)[::2]
    ytick = np.arange(yparam.shape[0])[::2]
    yticklabel = np.round(yparam,3)[::2]
    
    fig,ax1 = plt.subplots(1,1, figsize=[6,6])
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
