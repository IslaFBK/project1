#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 12:29:31 2021

@author: shni2598
"""


import matplotlib as mpl
mpl.use('Agg')
import scipy.stats
#import load_data_dict
import mydata
import brian2.numpy_ as np
from brian2.only import *
#import post_analysis as psa
import firing_rate_analysis as fra
import frequency_analysis as fqa
import fano_mean_match
import connection as cn
import pickle
import sys
import os
import matplotlib.pyplot as plt
import shutil
#%%
data_dir = 'raw_data/'
analy_type = 'state'
datapath = data_dir
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/twoarea_hz/state/adpt_logi_1stim/' + data_dir
#datapath = '/z:/brian2/NeuroNet_brian/model_file/attention/two_area/data/twoarea_hz/state/adpt_logi_1stim/' + data_dir
#datapath = 'Z:\\brian2\\NeuroNet_brian\\model_file\\attention\\two_area\\data\\twoarea_hz\\state\\adpt_logi_1stim\\' + data_dir
sys_argv = int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num
#loop_num = 560
good_dir = 'good/'
goodsize_dir = 'good_size/'

savefile_name = 'data_anly_wvt' #'data_anly' data_anly_temp

fftplot = 1; get_wvt = 1
getfano = 1
get_nscorr = 1; get_nscorr_t = 1
get_TunningCurve = 1; get_HzTemp = 1
firing_rate_long = 1

if loop_num%4 == 0: save_img = 1
else: save_img = 0

if loop_num%10 ==0: get_ani = 1
else: get_ani = 0

#%%
clr = plt.rcParams['axes.prop_cycle'].by_key()['color']

data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
data_anly = mydata.mydata()


if analy_type == 'state': # fbrg: feedback range
    title = 'hz_2irie%.2f_2ndgk%.1f_pk2e1e%.2f'%(data.param.ie_r_i2, data.param.new_delta_gk_2, \
                                               data.inter.param.peak_p_e2_e1)
        
        
n_StimAmp = data.a1.param.stim1.n_StimAmp
n_perStimAmp = data.a1.param.stim1.n_perStimAmp
stim_amp = 200*2**np.arange(n_StimAmp)

#%%
'''spontanous rate'''
dt = 1/10000;
end = int(20/dt); start = int(5/dt)
spon_rate1 = np.sum((data.a1.ge.t < end) & (data.a1.ge.t >= start))/15/data.a1.param.Ne
spon_rate2 = np.sum((data.a2.ge.t < end) & (data.a2.ge.t >= start))/15/data.a2.param.Ne

#data_anly.spon_rate1 = spon_rate1
#data_anly.spon_rate2 = spon_rate2
'''adapt rate'''
start = int(round((data.a1.param.stim1.stim_on[n_StimAmp*n_perStimAmp,0] - 2000)/1000/dt))
end = int(round((data.a1.param.stim1.stim_on[n_StimAmp*n_perStimAmp,0])/1000/dt))
adapt_rate1 = np.sum((data.a1.ge.t < end) & (data.a1.ge.t >= start))/2/data.a1.param.Ne
#data_anly.adapt_rate1 = adapt_rate1
adapt_rate2 = np.sum((data.a2.ge.t < end) & (data.a2.ge.t >= start))/2/data.a2.param.Ne
#data_anly.adapt_rate2 = adapt_rate2

title += '_1hz%.2f_1adphz%.2f_2hz%.2f_2adphz%.2f'%(spon_rate1,adapt_rate1,spon_rate2,adapt_rate2)

#%%
#if get_wvt:
mua_range_1 = 5
mua_loca_1 = [0,0]
mua_neuron_1 = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca_1, mua_range_1, data.a1.param.width)

#%%
'''sens'''
stim_dura = data.a1.param.stim1.stim_on[0,1] - data.a1.param.stim1.stim_on[0,0]
fig, ax = plt.subplots(2,n_StimAmp, figsize=[12,6])
ax2_hz = []
for st in range(n_StimAmp):
    '''no att'''
    window = 5
    start_time = data.a1.param.stim1.stim_on[st*n_perStimAmp,0] #- 300
    end_time = data.a1.param.stim1.stim_on[(st+1)*n_perStimAmp-1,1] + window#+ 500 
    
    
    data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = window)
    mua = data.a1.ge.spk_rate.spk_rate.reshape(data.a1.param.Ne,-1)[mua_neuron_1]
    mua = mua.mean(0)/(window/1000)
    
    coef, freq = fqa.mycwt(mua, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)
    
    #st = 1
    wvt_mean_amp = np.zeros([coef.shape[0], stim_dura])
    dura_anly = data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp].copy()
    dura_anly -= start_time
    dura_anly_n = dura_anly.shape[0]
    for dura in dura_anly:
        #dura -= start_time
        wvt_mean_amp += np.abs(coef[:,dura[0]:dura[1]])/dura_anly_n
    
    fig, ax[0,st], ax2 = fqa.plot_cwt(wvt_mean_amp, freq, base = 10, colorbar=True, fig=fig,ax=ax[0,st])
    ax[0,st].set_title('%.1fHz'%stim_amp[st])
    ax2_hz.append(ax2)
    
    if st == 0:
        data_anly.wvt = mydata.mydata()
        data_anly.wvt.wvt_resp_sens = np.zeros([*wvt_mean_amp.shape, n_StimAmp*2])
        data_anly.wvt.freq_sens = freq
    data_anly.wvt.wvt_resp_sens[:,:,st] = wvt_mean_amp
    #stim_dura = data.a1.param.stim1.stim_on[0,1] - data.a1.param.stim1.stim_on[0,0]
    '''att'''
    #st = 0
    start_time = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp,0] #- 300
    end_time = data.a1.param.stim1.stim_on[(st+n_StimAmp+1)*n_perStimAmp-1,0] + window#+ 500 
    
    #window = 5
    data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = window)
    mua = data.a1.ge.spk_rate.spk_rate.reshape(data.a1.param.Ne,-1)[mua_neuron_1]
    mua = mua.mean(0)/(window/1000)
    
    coef, freq = fqa.mycwt(mua, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)
    #
    #st = 1
    wvt_mean_amp_att = np.zeros([coef.shape[0], stim_dura])
    dura_anly = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp:(st+n_StimAmp+1)*n_perStimAmp].copy()
    dura_anly -= start_time
    dura_anly_n = dura_anly.shape[0]
    for dura in dura_anly:
        #dura -= start_time
        wvt_mean_amp_att += np.abs(coef[:,dura[0]:dura[1]])/dura_anly_n
    
    fig, ax[1,st], ax2 = fqa.plot_cwt(wvt_mean_amp_att, freq, base = 10, colorbar=True, fig=fig,ax=ax[1,st])
    ax[1,st].set_title('%.1fHz att'%stim_amp[st])
    ax2_hz.append(ax2)
    
    data_anly.wvt.wvt_resp_sens[:,:,st+n_StimAmp] = wvt_mean_amp_att
    
ax2_hz[0].set_ylabel('Hz')
ax2_hz[1].set_ylabel('Hz')
ax[1,1].set_xlabel('ms')

title_ = title + '\n_wvt_sens_%d'%loop_num
fig.suptitle(title_)#+title)

#if save_img: plt.savefig(title.replace('\n','')+'_tunecv'+'_%d'%loop_num+'.png')
fig.savefig(title_.replace('\n','')+'.png')
plt.close()
#%%
'''asso'''
stim_dura = data.a1.param.stim1.stim_on[0,1] - data.a1.param.stim1.stim_on[0,0]
fig, ax = plt.subplots(2,n_StimAmp, figsize=[12,6])
ax2_hz = []
#a1a2_delay = 10
for st in range(n_StimAmp):
    '''no attention'''
    window = 5
    start_time = data.a1.param.stim1.stim_on[st*n_perStimAmp,0] #- 300
    end_time = data.a1.param.stim1.stim_on[(st+1)*n_perStimAmp-1,1] + 200 #+ 500 
    
    
    data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = window)
    mua = data.a2.ge.spk_rate.spk_rate.reshape(data.a2.param.Ne,-1)[mua_neuron_1]
    mua = mua.mean(0)/(window/1000)
    
    coef, freq = fqa.mycwt(mua, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)
    
    #st = 1
    ext_len = 50 # ms
    wvt_mean_amp = np.zeros([coef.shape[0], stim_dura + ext_len])
    dura_anly = data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp].copy()
    dura_anly[:,1] += ext_len # a1a2_delay
    dura_anly -= start_time
    dura_anly_n = dura_anly.shape[0]
    for dura in dura_anly:
        #dura -= start_time
        wvt_mean_amp += np.abs(coef[:,dura[0]:dura[1]])/dura_anly_n
    
    fig, ax[0,st], ax2 = fqa.plot_cwt(wvt_mean_amp[:27], freq[:27], base = 10, colorbar=True, fig=fig,ax=ax[0,st])
    ax[0,st].set_title('%.1fHz'%stim_amp[st])
    ax2_hz.append(ax2)

    if st == 0:
        #data_anly.wvt = mydata.mydata()
        data_anly.wvt.wvt_resp_asso = np.zeros([*wvt_mean_amp.shape, n_StimAmp*2])
        data_anly.wvt.freq_asso = freq
    data_anly.wvt.wvt_resp_asso[:,:,st] = wvt_mean_amp
    
    #stim_dura = data.a2.param.stim1.stim_on[0,1] - data.a2.param.stim1.stim_on[0,0]
    '''attention'''
    #st = 0
    start_time = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp,0] #- 300
    end_time = data.a1.param.stim1.stim_on[(st+n_StimAmp+1)*n_perStimAmp-1,1] + 200 #+ 500 
    
    #window = 5
    data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = window)
    mua = data.a2.ge.spk_rate.spk_rate.reshape(data.a2.param.Ne,-1)[mua_neuron_1]
    mua = mua.mean(0)/(window/1000)
    
    coef, freq = fqa.mycwt(mua, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)
    #
    #st = 1
    wvt_mean_amp_att = np.zeros([coef.shape[0], stim_dura + ext_len])
    dura_anly = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp:(st+n_StimAmp+1)*n_perStimAmp].copy()
    dura_anly[:,1] += ext_len # a1a2_delay
    dura_anly -= start_time
    dura_anly_n = dura_anly.shape[0]
    #ii=0
    for dura in dura_anly:
        #print(ii);ii+=1
        #dura -= start_time
        wvt_mean_amp_att += np.abs(coef[:,dura[0]:dura[1]])/dura_anly_n
    
    fig, ax[1,st], ax2 = fqa.plot_cwt(wvt_mean_amp_att[:27], freq[:27], base = 10, colorbar=True, fig=fig,ax=ax[1,st])
    ax[1,st].set_title('%.1fHz att'%stim_amp[st])
    ax2_hz.append(ax2)
    
    data_anly.wvt.wvt_resp_asso[:,:,st+n_StimAmp] = wvt_mean_amp_att

ax2_hz[0].set_ylabel('Hz')
ax2_hz[1].set_ylabel('Hz')
ax[1,1].set_xlabel('ms')

title_ = title + '\n_wvt_asso_%d'%loop_num
fig.suptitle(title_)#+title)

#if save_img: plt.savefig(title.replace('\n','')+'_tunecv'+'_%d'%loop_num+'.png')
fig.savefig(title_.replace('\n','')+'.png')
plt.close()
#%%
'''no att; wvt long'''
st = 1  
start_time = data.a1.param.stim1.stim_on[st*n_perStimAmp,0] - 300
end_time = data.a1.param.stim1.stim_on[st*n_perStimAmp+5,1] + 200 


stim_onset = data.a1.param.stim1.stim_on[st*n_perStimAmp:(st*n_perStimAmp+6),0] - start_time



#start_time = 5e3; end_time = 20e3
window = 5
data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = window)
mua_1 = data.a1.ge.spk_rate.spk_rate.reshape(data.a1.param.Ne,-1)[mua_neuron_1]
mua_1 = mua_1.mean(0)/(window/1000)
data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = window)
mua_2 = data.a2.ge.spk_rate.spk_rate.reshape(data.a2.param.Ne,-1)[mua_neuron_1]
mua_2 = mua_2.mean(0)/(window/1000)

coef_1, freq_1 = fqa.mycwt(mua_1, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)
coef_2, freq_2 = fqa.mycwt(mua_2, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)

#coef, freq = fqa.mycwt(mua, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)
plt_freq_i = 27
plt_h = coef[:plt_freq_i].shape[0] -1

fig, ax = plt.subplots(4,1, figsize=[10,12])
fig, ax[1], ax2 = fqa.plot_cwt(coef_1[:plt_freq_i], freq_1[:plt_freq_i], base = 10, colorbar=True, fig=fig,ax=ax[1])
ax2.set_ylabel('Hz')
fig, ax[0], ax2 = fqa.plot_cwt(coef_2[:plt_freq_i], freq_2[:plt_freq_i], base = 10, colorbar=True, fig=fig,ax=ax[0])

#fig, ax, ax2 = fqa.plot_cwt(coef, freq, base = 10)

for onset_t in stim_onset:
    ax[0].plot([onset_t,onset_t],[plt_h,0], c=clr[1])
    ax[1].plot([onset_t,onset_t],[plt_h,0], c=clr[1])
#ax[1].set_xlabel('ms')
##ax[1].set_ylabel('Hz')

ax[0].set_title('asso_noatt')
ax[1].set_title('sens_noatt')

#%%
'''att; wvt long'''
st = 1  
start_time = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp,0] - 300
end_time = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp+5,1] + 200 


stim_onset = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp:((st+n_StimAmp)*n_perStimAmp+6),0] - start_time



#start_time = 5e3; end_time = 20e3
window = 5
data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = window)
mua_1 = data.a1.ge.spk_rate.spk_rate.reshape(data.a1.param.Ne,-1)[mua_neuron_1]
mua_1 = mua_1.mean(0)/(window/1000)
data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = window)
mua_2 = data.a2.ge.spk_rate.spk_rate.reshape(data.a2.param.Ne,-1)[mua_neuron_1]
mua_2 = mua_2.mean(0)/(window/1000)

coef_1, freq_1 = fqa.mycwt(mua_1, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)
coef_2, freq_2 = fqa.mycwt(mua_2, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)

#coef, freq = fqa.mycwt(mua, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)

plt_freq_i = 27
plt_h = coef[:plt_freq_i].shape[0] -1

#fig, ax = plt.subplots(2,1, figsize=[10,6])
fig, ax[3], ax2 = fqa.plot_cwt(coef_1[:plt_freq_i], freq_1[:plt_freq_i], base = 10, colorbar=True, fig=fig,ax=ax[3])
ax2.set_ylabel('Hz')
fig, ax[2], ax2 = fqa.plot_cwt(coef_2[:plt_freq_i], freq_2[:plt_freq_i], base = 10, colorbar=True, fig=fig,ax=ax[2])

#fig, ax, ax2 = fqa.plot_cwt(coef, freq, base = 10)

for onset_t in stim_onset:
    ax[2].plot([onset_t,onset_t],[plt_h,0], c=clr[1])
    ax[3].plot([onset_t,onset_t],[plt_h,0], c=clr[1])
ax[3].set_xlabel('ms')
#ax[1].set_ylabel('Hz')

ax[2].set_title('asso_att')
ax[3].set_title('sens_att')

title_ = title + '\n_wvt_long_sti%.1f_%d'%(stim_amp[st],loop_num)
fig.suptitle(title_)#+title)

#if save_img: plt.savefig(title.replace('\n','')+'_tunecv'+'_%d'%loop_num+'.png')
fig.savefig(title_.replace('\n','')+'.png')
plt.close()

#%%
data_anly.save(data_anly.class2dict(), datapath+savefile_name+'%d.file'%loop_num)
#%%


# title_ = title + '\nstim%.1f_noatt'%(stim_amp[st])
# fig.suptitle(title_)
# savetitle = title_.replace('\n','')




#%%
# fig, ax = plt.subplots(1,1)
# plt.plot(freq[::-1], wvt_mean_amp_att.mean(1)[::-1], ls='--')
# # ax.xaxis.set_visible(False)
# # # if colorbar: plt.colorbar(imcwt, ax=ax)

# # #ax.get_position()
# # ax2 = fig.add_axes(ax.get_position(),frame_on=False)
# # fix_bound = (freq[0]/freq[1])**0.5
# # ax2.set_xlim([freq[-1]/fix_bound,freq[0]*fix_bound])
# # ax2.yaxis.set_visible(False)

# # ax2 = fig.add_subplot(111,label='yaxis')
# # #ax2.yaxis.set_ticks(freq_wvt)
# # ax2.set_ylim([freq_wvt[-1]/(2**(1/10)),freq_wvt[0]*(2**(1/10))])
# # ax2.xaxis.set_visible(False)

# plt.xscale('log',base=10)
# plt.yscale('log',base=10)

# #%%
# fig, ax = plt.subplots(1,1)
# #plt.imshow(wvt_mean_amp_att)
# fig, ax, ax2 = fqa.plot_cwt(coef_2, freq, base = 10, colorbar=True, fig=fig,ax=ax)





#%%
# #%%
# x = np.arange(2000)/1000
# d = np.sin(2*np.pi*4*x)
# coef, freq = fqa.mycwt(d, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)
# fig, ax, ax2 = fqa.plot_cwt(coef, freq, base = 10)


# #%%
# start_time = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp,0] - 300
# end_time = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp+5,0] + 500 
    
# #start_time = 5e3; end_time = 20e3
# window = 5
# data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = window)
# mua = data.a1.ge.spk_rate.spk_rate.reshape(data.a1.param.Ne,-1)[mua_neuron_1]
# mua = mua.mean(0)/(window/1000)

# coef, freq = fqa.mycwt(mua, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)

# fig, ax, ax2 = fqa.plot_cwt(coef, freq, base = 10)
# plt.xlabel('ms')
# plt.ylabel('Hz')
# title_ = title + '\nstim%.1f_att'%(stim_amp[st])
# fig.suptitle(title_)
# savetitle = title_.replace('\n','')
# ##nsfile = savetitle+'_nct_%d'%(loop_num)+'.png'
# #fig.savefig(savetitle+'_wvt_st%.1f_%d'%(stim_amp[st],loop_num)+'.png')
# #plt.close()    
#%%    
    
    
    
    
    