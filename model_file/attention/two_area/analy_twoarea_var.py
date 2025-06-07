#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 22:03:00 2021

@author: shni2598
"""


import matplotlib as mpl
mpl.use('Agg')
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
#datapath = '/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/'
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/chg_adpt_netsize/'
# import glob
# #path = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/dp_num_ie/'
# path = ''
# FilenamesList = glob.glob(path+'*.mp4')
# path = './good/'
# FilenamesList += glob.glob(path+'*.mp4')
# indList = [None]*len(FilenamesList)
# for i in range(len(FilenamesList)):
#     indList[i] = FilenamesList[i].split('.')[-2].split('_')[-1]
    
# if sys.argv[1] in indList:
#     print('True')
#     sys.exit("Exit, file already exists!")


analy_type = 'fano'
datapath = ''
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/spon_1area/'
sys_argv = int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num
good_dir = 'good/'
goodsize_dir = 'good_size/'
#%%
data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
#%%
'''spontanous rate'''
data_anly = mydata.mydata()
dt = 1/10000;
end = int(15/dt); start = int(5/dt)
spon_rate1 = np.sum((data.a1.ge.t < end) & (data.a1.ge.t > start))/10/data.a1.param.Ne
spon_rate2 = np.sum((data.a2.ge.t < end) & (data.a2.ge.t > start))/10/data.a2.param.Ne

data_anly.spon_rate1 = spon_rate1
data_anly.spon_rate2 = spon_rate2
#data_anly.invalid_pii = data.a1.param.p_peak_ii >= 1
#data_anly.save(data_anly.class2dict(), datapath+'data_anly%d.file'%loop_num)
# """
# spon_rate_good = spon_rate < 9.5 and spon_rate > 2.5
# """
# '''MSD'''
# data.a1.ge.get_MSD(start_time=4000, end_time=20000, n_neuron=data.a1.param.Ne, window = 15, jump_interval=np.array([15]), fit_stableDist='pylevy')
# stable_good = data.a1.ge.MSD.stableDist_param[0,0] < 1.8 and data.a1.ge.MSD.stableDist_param[0,0] > 1.35
# data_anly.alpha_dist = data.a1.ge.MSD.stableDist_param

# '''pattern'''
# start_time = 4e3; end_time = 20e3
# data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 10)
# data.a1.ge.get_centre_mass(detect_pattern=True)

# #data_anly.pattern = data.a1.ge.centre_mass.pattern
# #data_anly.pattern_size = data.a1.ge.centre_mass.pattern_size
# data_anly.patterns_size_mean = data.a1.ge.centre_mass.pattern_size[data.a1.ge.centre_mass.pattern].mean()
# data_anly.pattern_on_ratio = data.a1.ge.centre_mass.pattern.sum()/len(data.a1.ge.centre_mass.pattern)
# data_anly.patterns_size_std = data.a1.ge.centre_mass.pattern_size[data.a1.ge.centre_mass.pattern].std()
# data_anly.jump_dist_mean = data.a1.ge.centre_mass.jump_dist.mean()
#%%

if analy_type == 'fft':
    #data_anly.invalid_pei = data.a1.param.p_peak_ei >= 1    
    title = "erie%.3f_\non%.2f_sz%.1f_alpha%.2f_hz%.2f"%(data.a1.param.ie_r_e,\
                                                      data_anly.pattern_on_ratio, data_anly.patterns_size_mean,\
                                                      data.a1.ge.MSD.stableDist_param[0,0],spon_rate)    
if analy_type == 'stim':
    #data_anly.invalid_pei = data.a1.param.p_peak_ei >= 1    
    title = "erie1%.3f_erie2%.3f_w12%.2f_w21%.2f_\nhz1%.2f_hz2%.2f"%(data.param.ie_r_e1, data.param.ie_r_e2,\
                                                                     data.inter.param.w_e1_e2_mean, data.inter.param.w_e2_e1_mean,\
                                                                     spon_rate1, spon_rate2)    
if analy_type == 'toa1':
    #data_anly.invalid_pei = data.a1.param.p_peak_ei >= 1    
    title = "w21e%.3f_w21i%.3f_p21i%.2f_\n1hz%.2f_2hz%.2f"%(data.inter.param.w_e2_e1_mean, data.inter.param.w_e2_i1_mean,\
                                                                     data.inter.param.peak_p_e2_i1,\
                                                                     spon_rate1, spon_rate2)    
if analy_type == 'toa2':
    #data_anly.invalid_pei = data.a1.param.p_peak_ei >= 1    
    title = "w12e%.2f_w12i%.2f_p12i%.2f_\n1hz%.2f_2hz%.2f"%(data.inter.param.w_e1_e2_mean, data.inter.param.w_e1_i2_mean,\
                                                                     data.inter.param.peak_p_e1_i2,\
                                                                     spon_rate1, spon_rate2)    

if analy_type == 'a1a2':
    #data_anly.invalid_pei = data.a1.param.p_peak_ei >= 1    
    title = "w12e%.2f_w12i%.2f_p12i%.2f_\nw21e%.2f_w21i%.2f_p21i%.2f_\n1hz%.2f_2hz%.2f"%(data.inter.param.w_e1_e2_mean, data.inter.param.w_e1_i2_mean,data.inter.param.peak_p_e1_i2,\
                                                                                         data.inter.param.w_e2_e1_mean, data.inter.param.w_e2_i1_mean,data.inter.param.peak_p_e2_i1,\
                                                                                        spon_rate1, spon_rate2)    
if analy_type == 'stim':
    #data_anly.invalid_pei = data.a1.param.p_peak_ei >= 1    
    title = "w12e%.2f_w12i%.2f_p12i%.2f_\nw21e%.2f_w21i%.2f_p21i%.2f_\n1hz%.2f_2hz%.2f"%(data.inter.param.w_e1_e2_mean, data.inter.param.w_e1_i2_mean,data.inter.param.peak_p_e1_i2,\
                                                                                         data.inter.param.w_e2_e1_mean, data.inter.param.w_e2_i1_mean,data.inter.param.peak_p_e2_i1,\
                                                                                        spon_rate1, spon_rate2)    
if analy_type == 'stim2':
    #data_anly.invalid_pei = data.a1.param.p_peak_ei >= 1    
    title = "w12e%.2f_w12i%.2f_\n1hz%.2f_2hz%.2f"%(data.inter.param.w_e1_e2_mean, data.inter.param.w_e1_i2_mean,\
                                                   spon_rate1, spon_rate2)    

if analy_type == 'stim3':
    #data_anly.invalid_pei = data.a1.param.p_peak_ei >= 1    
    title = "1eier%.3f_w12e%.2f_w12i%.2f_\n1hz%.2f_2hz%.2f"%(data.param.ie_r_e1, data.inter.param.w_e1_e2_mean, data.inter.param.w_e1_i2_mean,\
                                                   spon_rate1, spon_rate2)    
 
if analy_type == 'one_stim':
    #data_anly.invalid_pei = data.a1.param.p_peak_ei >= 1    
    title = "eier%.3f_iier%.3f_\n1hz%.2f"%(data.param.ie_r_e1, data.param.ie_r_i1, \
                                                   spon_rate1)    

if analy_type == 'two_stim':
    #data_anly.invalid_pei = data.a1.param.p_peak_ei >= 1    
    title = "12e%.3f_12i%.3f_\n1hz%.2f_2hz%.2f"%(data.inter.param.w_e1_e2_mean/4, data.inter.param.w_e1_i2_mean/4, \
                                                   spon_rate1, spon_rate2)  
if analy_type == 'two_stim21':
    #data_anly.invalid_pei = data.a1.param.p_peak_ei >= 1    
    title = "21e%.3f_21i%.3f_\n1hz%.2f_2hz%.2f"%(data.inter.param.w_e2_e1_mean/7, data.inter.param.w_e2_i1_mean/7, \
                                                   spon_rate1, spon_rate2)   
        # if data_anly.invalid_pei:
    #     title = 'p1_'+title
if analy_type == 'fano':
    #data_anly.invalid_pei = data.a1.param.p_peak_ei >= 1    
    title = 'fano'  


clr = plt.rcParams['axes.prop_cycle'].by_key()['color']
#%%
'''fft'''
def find_peakF(coef, freq, lwin):
    dF = freq[1] - freq[0]
    #Fwin = 0.3
    #lwin = 3#int(Fwin/dF)
    win = np.ones(lwin)/lwin
    coef_avg = np.convolve(np.abs(coef[1:]), win, mode='same')
    peakF = freq[1:][coef_avg.argmax()]
    return peakF
#%%
mua_loca = [0, 0]
mua_range = 5 
mua_neuron = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca, mua_range, data.a1.param.width)
#%%

start_time = 5e3; end_time = 15e3
window = 5
data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 5)
mua = data.a1.ge.spk_rate.spk_rate.reshape(data.a1.param.Ne,-1)[mua_neuron]
mua = mua.mean(0)/(window/1000)
#%%
def plot_fft(data_fft):
    fig, ax = plt.subplots(2,2,figsize=[9,9])
    
    fs = 1000
    #data_fft = mua[:]
    coef, freq = fqa.myfft(data_fft, fs)
    # data_anly.coef_spon = coef
    # data_anly.freq_spon = freq
    
    peakF_spon = find_peakF(coef, freq, 3)
    
    freq_max = 20
    ind_len = freq[freq<freq_max].shape[0] # int(20/(fs/2)*(len(data_fft)/2)) + 1
    ax[0,0].plot(freq[1:ind_len], np.abs(coef[1:ind_len]),label='spon')
    ax[0,1].loglog(freq[1:ind_len], np.abs(coef[1:ind_len]),label='spon_log')
    
    freq_max = 150
    ind_len = freq[freq<freq_max].shape[0] # int(20/(fs/2)*(len(data_fft)/2)) + 1
    ax[1,0].plot(freq[1:ind_len], np.abs(coef[1:ind_len]),label='spon')
    ax[1,1].loglog(freq[1:ind_len], np.abs(coef[1:ind_len]),label='spon_log')
    
    for i in range(2):
        for j in range(2):
            ax[i,j].legend()
    
    return fig, ax, peakF_spon, coef, freq
#%%
fig, ax, peakF_spon, coef, freq = plot_fft(mua)
data_anly.coef_spon_a1 = coef
data_anly.freq_spon_a1 = freq

title1 = title + '_pf%.2f_a1'%(peakF_spon)
savetitle = title1.replace('\n','')
fig.suptitle(title1)
fftfile = savetitle+'_fft_%d'%(loop_num)+'.png'
#fig.savefig(fftfile)

data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 5)
mua = data.a2.ge.spk_rate.spk_rate.reshape(data.a2.param.Ne,-1)[mua_neuron]
mua = mua.mean(0)/(window/1000)

fig, ax, peakF_spon, coef, freq = plot_fft(mua)
data_anly.coef_spon_a2 = coef
data_anly.freq_spon_a2 = freq

title2 = title + '_pf%.2f_a2'%(peakF_spon)
savetitle = title2.replace('\n','')
fig.suptitle(title2)
fftfile = savetitle+'_fft_%d'%(loop_num)+'.png'
fig.savefig(fftfile)

#%%
# data_fft = mua[22000:32000]
# coef, freq = fqa.myfft(data_fft, fs)
# data_anly.coef_adapt = coef
# data_anly.freq_adapt = freq

# peakF_adapt = find_peakF(coef, freq, 1)

# freq_max = 20
# ind_len = freq[freq<freq_max].shape[0] # int(20/(fs/2)*(len(data_fft)/2)) + 1
# ax1t=ax1.twinx()
# im2 = ax1t.plot(freq[1:ind_len], np.abs(coef[1:ind_len]),'tab:orange',label='adapt')
# ax1t.plot([3,3],[0,np.max(np.abs(coef[1:ind_len]))],'r--')
# ax1t.plot([8,8],[0,np.max(np.abs(coef[1:ind_len]))],'r--')
# ax2.loglog(freq[1:ind_len], np.abs(coef[1:ind_len]),label='adapt_log')
# ax2.loglog([3,3],[0,np.max(np.abs(coef[1:ind_len]))],'r--')
# ax2.loglog([8,8],[0,np.max(np.abs(coef[1:ind_len]))],'r--')
# lms = im1 + im2
# labs = [l.get_label() for l in lms]
# ax1.legend(lms, labs)#, loc=0)
# ax2.legend()
# #ax2.set_title('adapt')
# title = title + '_pf%.2f_pfadpt%.2f'%(peakF_spon,peakF_adapt)
# savetitle = title.replace('\n','')

# fig.suptitle(title)
# data_anly.peakF_spon = peakF_spon
# data_anly.peakF_adapt = peakF_adapt

# peakF_good = peakF_spon > 3# and peakF_adapt > 3

# fftfile = savetitle+'_fft_%d'%(loop_num)+'.png'
# fig.savefig(fftfile)

# #plt.plot(freq[1:], np.abs(coef[1:]))

#%%
'''fano'''
data_anly.fano = mydata.mydata()

stim_loc = np.array([0,0])

neuron = np.arange(data.a1.param.Ne)

dist = cn.coordination.lattice_dist(data.a1.param.e_lattice,63,stim_loc)
neu_pool = [None]*1
neu_pool[0] = neuron[(dist >= 0) & (dist <= 10)]

fr_bin = np.array([200]) 
    
simu_time_tot = data.param.simutime#29000

N_stim = data.a1.param.stim.stim_amp_scale.shape[0]

data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])

fanomm = fano_mean_match.fano_mean_match()
fanomm.bin_count_interval = 0.5
fanomm.spk_sparmat = data.a1.ge.spk_matrix[neu_pool[0],:]
fanomm.method = 'regression' # 'mean' or 'regression'

for win in [50,100,150]:
    fanomm.win = win
    fanomm.stim_onoff = data.a1.param.stim.stim_on[0:100].copy()
    fano_mean_noatt, fano_std_noatt, _ = fanomm.get_fano()
    
    data_anly.fano.fano_mean_noatt = fano_mean_noatt
    data_anly.fano.fano_std_noatt = fano_std_noatt
    
    # print(np.sum(np.isnan(_)))
    # print(np.sum(np.isnan(fano_mean_noatt)))
    # print(np.sum(np.isnan(fano_std_noatt)))
    fanomm.stim_onoff = data.a1.param.stim.stim_on[100:200].copy()
    
    fano_mean_att, fano_std_att, _ = fanomm.get_fano()
    # print(np.sum(np.isnan(_)))
    data_anly.fano.fano_mean_att = fano_mean_att
    data_anly.fano.fano_std_att = fano_std_att
    
    fig, ax = plt.subplots(1,1)
    ax.errorbar(np.arange(fano_mean_noatt.shape[0])*10-100,fano_mean_noatt,fano_std_noatt,label='no attention')
    ax.errorbar(np.arange(fano_mean_att.shape[0])*10-100,fano_mean_att,fano_std_att,label='attention')
    ax.set_xlabel('ms')
    ax.set_ylabel('fano')
    plt.legend()
    title3 = title + '_win%.1f'%fanomm.win#%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean())
    fig.suptitle(title3)
    savetitle = title3.replace('\n','')
    fanofile = savetitle+'_%d'%(loop_num)+'.png'
    fig.savefig(fanofile)
#%%
'''firing rate'''
'''
firing rate-location
'''
stim_loc = np.array([0,0])

dist = cn.coordination.lattice_dist(data.a1.param.e_lattice,63,stim_loc)
dist_bin = np.arange(0, 31.5*2**0.5, 2.5)
n_in_bin = [None]*(dist_bin.shape[0]-1)
neuron = np.arange(data.a1.param.Ne)

for i in range(len(dist_bin)-1):
    n_in_bin[i] = neuron[(dist >= dist_bin[i]) & (dist < dist_bin[i+1])]


simu_time_tot = data.param.simutime#29000

#N_stim = data.a1.param.stim.stim_amp_scale.shape[0]

data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])

hz_loc = fra.tuning_curve(data.a1.ge.spk_matrix, data.a1.param.stim.stim_on, n_in_bin)
#%%
hz_loc_spon = np.zeros([2, len(n_in_bin)])
'''no attention'''
spon_onff = np.array([[5000,10000]])
hz_loc_spon[0,:] = fra.tuning_curve(data.a1.ge.spk_matrix, spon_onff, n_in_bin)

'''attention'''
spon_adpt_stt = data.a1.param.stim.stim_on[100,0] - 2000 
spon_adpt_end = data.a1.param.stim.stim_on[100,0]
spon_onff = np.array([[spon_adpt_stt,spon_adpt_end]])
hz_loc_spon[1,:] = fra.tuning_curve(data.a1.ge.spk_matrix, spon_onff, n_in_bin)

data_anly.hz_loc = hz_loc
data_anly.hz_loc_spon = hz_loc_spon

'''plot'''
stim_amp = 400
hz_loc_mean_noatt = hz_loc[:100,:].mean(0)
hz_loc_mean_att = hz_loc[100:200,:].mean(0)

plt.figure(figsize=[8,6])
dist_bin_plot = (dist_bin+(dist_bin[1]-dist_bin[0])*0.5)[:-1]
    
plt.plot(dist_bin_plot, hz_loc_mean_noatt, ls='--', marker='o', c=clr[0], label = 'stim_amp: %.1f Hz'%(stim_amp))
plt.plot(dist_bin_plot, hz_loc_spon[0], ls='--', marker='o', c=clr[1], label = 'spontaneous')

plt.plot(dist_bin_plot, hz_loc_mean_att, ls='-', marker='o', c=clr[0], label = 'attention; stim_amp: %.1f Hz'%(stim_amp))
plt.plot(dist_bin_plot, hz_loc_spon[1], ls='-', marker='o', c=clr[1], label = 'attention; spontaneous')


#title = "2e1e:%.3f_2e1i:%.3f_1hz:%.2f_2hz:%.2f"%(scale_w_21_e[ie_ind//scale_w_21_i.shape[0]], scale_w_21_i[ie_ind%scale_w_21_i.shape[0]], spon_rate1,spon_rate2)
#plt.title('average firing rate versus distance to stimulus centre; 2e1e:%.3f 2e1i%.3f\n Spike-count bin: %.1f ms\n'%(scale_w_21_e[ie_ind//scale_w_21_i.shape[0]], scale_w_21_i[ie_ind%scale_w_21_i.shape[0]],fr_bin[b])+title)
plt.title('average firing rate versus distance to stimulus centre; Spike-count bin: %.1f ms\n'%(data.a1.param.stim.stim_on[0,1]-data.a1.param.stim.stim_on[0,0]))#+title)

plt.xlim([dist_bin[0],dist_bin[-1]])
plt.xlabel('distance')
plt.ylabel('Hz')
plt.legend()
plt.savefig(title.replace(':','')+'_tunecv'+'_%d'%loop_num+'.png')
#%%
'''firing rate time'''
mua_loca_1 = [0, 0]
mua_range_1 = 5
mua_neuron_1 = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca_1, mua_range_1, data.a1.param.width)
window = 5
dura_onoff = data.a1.param.stim.stim_on.copy()
dura_onoff[:,0] -= 100
dura_onoff[:,1] += 100
hz_t = fra.firing_rate_time_multi(data.a1.ge, mua_neuron_1, dura_onoff, window=window)
data_anly.hz_t = hz_t

hz_t_noatt_mean = hz_t[:100,:].mean(0)
hz_t_noatt_std = hz_t[:100,:].std(0)

hz_t_att_mean = hz_t[100:200,:].mean(0)
hz_t_att_std = hz_t[100:200,:].std(0)

#hz_t_mean = hz_t[:,:, 0].reshape(n_amp_stim,n_per_amp,-1).mean(1)
plt.figure(figsize=[8,6])
#for i in range(n_amp_stim_att):
t_plot = np.arange(dura_onoff[0,1] - dura_onoff[0,0]) - 100
plt.plot(t_plot, hz_t_noatt_mean, ls='--', c=clr[0], label = 'stim_amp: %.1f Hz'%(stim_amp))
plt.fill_between(t_plot, hz_t_noatt_mean-hz_t_noatt_std,hz_t_noatt_mean+hz_t_noatt_std, \
                 ls='--', facecolor=clr[0], edgecolor=clr[0], alpha=0.2)
#for i in range(n_amp_stim_att,n_amp_stim):
plt.plot(t_plot, hz_t_att_mean, ls='-', c=clr[0], label = 'attention; stim_amp: %.1f Hz'%(stim_amp)) 
plt.fill_between(t_plot, hz_t_att_mean-hz_t_att_std,hz_t_att_mean+hz_t_att_std, \
                 ls='-', facecolor=clr[0], edgecolor=clr[0], alpha=0.2)
plt.xlabel('ms')
plt.ylabel('Hz')
#title = "2e1e:%.3f_2e1i:%.3f_1hz:%.2f_2hz:%.2f"%(scale_w_21_e[ie_ind//scale_w_21_i.shape[0]], scale_w_21_i[ie_ind%scale_w_21_i.shape[0]], spon_rate1,spon_rate2)
#title = ''
plt.title('firing rate-time; senssory')
plt.legend()
#plt.savefig(save_dir+title.replace(':','')+'_temp'+'_%d'%ie_ind+'.png')
plt.savefig(title.replace(':','')+'_temp'+'_%d'%loop_num+'.png')
#%%
# plt.figure(figsize=[8,6]);
# plt.fill_between(np.arange(3), np.arange(3)-1,np.arange(3)+1, edgecolor=clr[0], ls='--',  alpha=0.2,label = 'stim_amp: %.1f Hz'%(600))
#%%
'''
spk_respon = np.zeros([neu_pool[0].shape[0], N_stim])

for i in range(N_stim):
    spk_respon[:,i] = data.a1.ge.spk_matrix[neu_pool[0], data.a1.param.stim.stim_on[i,0]*10:(data.a1.param.stim.stim_on[i,0]+fr_bin[0])*10].sum(1).A[:,0]


data_anly.var_noatt = np.var(spk_respon[:,:20], 1)
data_anly.var_att = np.var(spk_respon[:,20:], 1)

data_anly.mean_noatt = np.mean(spk_respon[:,:20], 1)
data_anly.mean_att = np.mean(spk_respon[:,20:], 1)

data_anly.fano_noatt = data_anly.var_noatt/data_anly.mean_noatt
data_anly.fano_att = data_anly.var_att/data_anly.mean_att

#%%
fig, ax = plt.subplots(2,1,figsize=[6,7])
ax[0].hist(data_anly.fano_noatt,alpha=0.7,label='no-attention')
ax[0].hist(data_anly.fano_att,alpha=0.7,label='attention')
ax[0].set_title('mean-fano-noatt: %.2f; mean-fano-att: %.2f'%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean()))
ax[0].set_xlabel('fano-factor')
ax[0].legend()

ax[1].bar([0], data_anly.fano_noatt.mean(), yerr=data_anly.fano_noatt.std(),color=clr[0],capsize=10)  
ax[1].bar([1], data_anly.fano_att.mean(), yerr=data_anly.fano_att.std(),color=clr[1],capsize=10)  
ax[1].set_xticks(np.arange(2))
ax[1].set_xticklabels(['fano-noatt', 'fano-att'])

title3 = title + '_fanona%.2f_fanoa%.2f'%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean())
fig.suptitle(title3)
savetitle = title3.replace('\n','')

fanofile = savetitle+'_fano_%d'%(loop_num)+'.png'
#fig.savefig(fanofile)
#%%
bins = np.linspace(data_anly.mean_noatt.min(), data_anly.mean_noatt.max()+0.1, 6)
neu_response = [None]*(bins.shape[0]-1)
neu_ind = np.arange(data_anly.mean_noatt.shape[0])
for i in range(len(neu_response)):
    neu_response[i] = neu_ind[(data_anly.mean_noatt >= bins[i]) & (data_anly.mean_noatt < bins[i+1])]

fano_response = np.zeros([2, len(neu_response), 2])
for i in range(len(neu_response)):
    fano_response[0,i,0] = data_anly.fano_noatt[neu_response[i]].mean()
    fano_response[1,i,0] = data_anly.fano_noatt[neu_response[i]].std()
    fano_response[0,i,1] = data_anly.fano_att[neu_response[i]].mean()
    fano_response[1,i,1] = data_anly.fano_att[neu_response[i]].std()

data_anly.fano_respons = fano_response
#%%
fig = plt.figure()
bins_plot = (bins + (bins[1]-bins[0])/2)[:-1]
plt.errorbar(bins_plot, fano_response[0,:,0], fano_response[1,:,0], ls='--', marker='o', c=clr[0], label='no-attention')
plt.errorbar(bins_plot, fano_response[0,:,1], fano_response[1,:,1], ls='-', marker='o', c=clr[1], label='attention')
plt.legend()
plt.xlabel('spikes')
plt.ylabel('fano')
plt.title('fano-response')
title3 = title + '_fanona%.2f_fanoa%.2f'%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean())
#fig.suptitle(title3)
savetitle = title3.replace('\n','')

fanofile = savetitle+'_fano_resp_%d'%(loop_num)+'.png'
#fig.savefig(fanofile)

#%%
fig = plt.figure()
plt.scatter(data_anly.mean_noatt, data_anly.var_noatt, s=10, c='b',label='no attention')
plt.scatter(data_anly.mean_att, data_anly.var_att, s=10, c='y',label='attention')

plot_mean = np.vstack((data_anly.mean_noatt, data_anly.mean_att))
plot_var = np.vstack((data_anly.var_noatt, data_anly.var_att))
plt.plot(plot_mean, plot_var, c='b',lw=0.6)
plt.legend()
plt.xlabel('mean')
plt.ylabel('var')
title3 = title + '_varna%.2f_vara%.2f'%(data_anly.var_noatt.mean(), data_anly.var_att.mean())
fig.suptitle(title3)
savetitle = title3.replace('\n','')

varfile = savetitle+'_var_%d'%(loop_num)+'.png'
#fig.savefig(varfile)
'''
#%%

#%%
'''
slop = (plot_var[1,:] - plot_var[0,:])/(plot_mean[1,:] - plot_mean[0,:])
slop = slop[np.logical_not((slop == np.inf) | (slop == -np.inf))]
fig = plt.figure()
plt.hist(slop, 15)
plt.xlabel('_dvar/dmean')
title4 = title3 + '_dvardmean'
savetitle = title4.replace('\n','')
slopfile = savetitle+'_%d'%(loop_num)+'.png'
fig.savefig(slopfile)
'''
#%%
'''
mean_inc = ((plot_mean[1,:] - plot_mean[0,:]) > 0).sum()/ plot_mean.shape[1]
var_dec = ((plot_var[1,:] - plot_var[0,:]) < 0).sum()/ plot_mean.shape[1]

mean_inc_var_dec = (((plot_mean[1,:] - plot_mean[0,:]) > 0) & ((plot_var[1,:] - plot_var[0,:]) < 0)).sum()/plot_mean.shape[1]
#%%
fig, ax = plt.subplots(2,1,figsize=[6,7])
ax[0].bar(np.arange(3),(mean_inc, var_dec, mean_inc_var_dec))
ax[0].set_xticks(np.arange(3))
ax[0].set_xticklabels(['mean+', 'var-', 'mean+var-'])
ax[0].set_title('percentage of neurons with increasing mean and decreasing var')
ax[1].bar([0], data_anly.mean_noatt.mean(), yerr=data_anly.mean_noatt.std(),color=clr[0],capsize=10)  
ax[1].bar([1], data_anly.mean_att.mean(), yerr=data_anly.mean_att.std(),color=clr[1],capsize=10)  

ax[1].bar([2], data_anly.var_noatt.mean(), yerr=data_anly.var_noatt.std(),color=clr[0],capsize=10)  
ax[1].bar([3], data_anly.var_att.mean(), yerr=data_anly.var_att.std(),color=clr[1],capsize=10)  
ax[1].set_xticks(np.arange(4))
ax[1].set_xticklabels(['mean-noatt', 'mean-att', 'var-noatt', 'var-att'])
ax[1].set_title('population change of mean and var with or without attention')

title5 = title3 + '_popu'
savetitle = title5.replace('\n','')
varfile2 = savetitle+'_%d'%(loop_num)+'.png'
fig.savefig(varfile2)
'''
#%%
'''animation'''
first_stim = 20; last_stim = 23
start_time = data.a1.param.stim.stim_on[first_stim,0] - 400
end_time = data.a1.param.stim.stim_on[last_stim,0] + 400

data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 15)
data.a1.ge.get_centre_mass()
data.a1.ge.overlap_centreandspike()

data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 15)
data.a2.ge.get_centre_mass()
data.a2.ge.overlap_centreandspike()

#data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ni, window = 10)
#frames = int(end_time - start_time)
frames = data.a1.ge.spk_rate.spk_rate.shape[2]

stim_on_off = data.a1.param.stim.stim_on-start_time
stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]
stim = [[[[31,31]], [stim_on_off], [[6]*stim_on_off.shape[0]]], None]
adpt = [None, [[[31,31]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
ani = fra.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, spkrate2=data.a2.ge.spk_rate.spk_rate, \
                                        frames = frames, start_time = start_time, interval_movie=10, anititle='',stim=stim, adpt=adpt)
savetitle = title.replace('\n','')

moviefile = savetitle+'_att_%d'%loop_num+'.mp4'

if loop_num%1 == 0:
    ani.save(moviefile)
    pass

first_stim = 0; last_stim = 3
start_time = data.a1.param.stim.stim_on[first_stim,0] - 400
end_time = data.a1.param.stim.stim_on[last_stim,0] + 400

data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 15)
data.a1.ge.get_centre_mass()
data.a1.ge.overlap_centreandspike()

data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 15)
data.a2.ge.get_centre_mass()
data.a2.ge.overlap_centreandspike()

#data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ni, window = 10)
#frames = int(end_time - start_time)
frames = data.a1.ge.spk_rate.spk_rate.shape[2]

stim_on_off = data.a1.param.stim.stim_on-start_time
stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]
stim = [[[[31,31]], [stim_on_off], [[6]*stim_on_off.shape[0]]], None]
#adpt = [None, [[[31,31]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
adpt = None
ani = fra.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, spkrate2=data.a2.ge.spk_rate.spk_rate, \
                                        frames = frames, start_time = start_time, interval_movie=10, anititle='',stim=stim, adpt=adpt)
savetitle = title.replace('\n','')

moviefile = savetitle+'_noatt_%d'%loop_num+'.mp4'

if loop_num%1 == 0:
    ani.save(moviefile)
    pass
#%%
data_anly.save(data_anly.class2dict(), datapath+'data_anly%d.file'%loop_num)




