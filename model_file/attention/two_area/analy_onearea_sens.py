#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 23:00:37 2021

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


analy_type = 'sens'
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
title = 'sens'
title = title + '_eier%.2f_iier%.2f'%(data.param.ie_r_e1, data.param.ie_r_i1)


'''spontanous rate'''
data_anly = mydata.mydata()
dt = 1/10000;
end = int(15/dt); start = int(5/dt)
spon_rate1 = np.sum((data.a1.ge.t < end) & (data.a1.ge.t > start))/10/data.a1.param.Ne
#spon_rate2 = np.sum((data.a2.ge.t < end) & (data.a2.ge.t > start))/10/data.a2.param.Ne

data_anly.spon_rate1 = spon_rate1
#data_anly.spon_rate2 = spon_rate2
title = title + '_hzspon%.2f'%spon_rate1
'''pattern size'''
start_time = 5e3; end_time = 15e3
data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 10)
data.a1.ge.get_centre_mass(detect_pattern=True)

#data_anly.pattern = data.a1.ge.centre_mass.pattern
#data_anly.pattern_size = data.a1.ge.centre_mass.pattern_size
data_anly.patterns_size_mean = data.a1.ge.centre_mass.pattern_size[data.a1.ge.centre_mass.pattern].mean()
data_anly.pattern_on_ratio = data.a1.ge.centre_mass.pattern.sum()/data.a1.ge.centre_mass.pattern.shape[0]
data_anly.patterns_size_std = data.a1.ge.centre_mass.pattern_size[data.a1.ge.centre_mass.pattern].std()
title = title + '\n_ptsz%.1f_pton%.2f'%(data_anly.patterns_size_mean, data_anly.pattern_on_ratio)

'''MSD alpha'''
data.a1.ge.get_MSD(start_time=5000, end_time=15000, n_neuron=data.a1.param.Ne, window = 10, jump_interval=np.array([15]), fit_stableDist='pylevy')
data_anly.alpha = data.a1.ge.MSD.stableDist_param[0,0]
title = title + '_alpha%.2f'%data.a1.ge.MSD.stableDist_param[0,0]
#%%
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
    
    freq_max = 200
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

titlefft = title + '_pf%.2f_a1'%(peakF_spon)
fig.suptitle(titlefft)
savetitle = titlefft.replace('\n','')
fftfile = savetitle+'_fft_%d'%(loop_num)+'.png'
fig.savefig(fftfile)
plt.close()
# '''firing rate'''
'''
firing rate-location
'''
stim_loc = np.array([0,0])

dist = cn.coordination.lattice_dist(data.a1.param.e_lattice,data.a1.param.width,stim_loc)
dist_bin = np.arange(0, data.a1.param.width/2*2**0.5, 2.5)
n_in_bin = [None]*(dist_bin.shape[0]-1)
neuron = np.arange(data.a1.param.Ne)

for i in range(len(dist_bin)-1):
    n_in_bin[i] = neuron[(dist >= dist_bin[i]) & (dist < dist_bin[i+1])]


simu_time_tot = data.param.simutime#29000

#N_stim = data.a1.param.stim.stim_amp_scale.shape[0]

data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])

hz_loc = fra.tuning_curve(data.a1.ge.spk_matrix, data.a1.param.stim.stim_on, n_in_bin)
#%%
hz_loc_spon = np.zeros([1, len(n_in_bin)])
'''no attention'''
spon_onff = np.array([[5000,15000]])
hz_loc_spon[0,:] = fra.tuning_curve(data.a1.ge.spk_matrix, spon_onff, n_in_bin)

# '''attention'''
# spon_adpt_stt = data.a1.param.stim.stim_on[100,0] - 2000 
# spon_adpt_end = data.a1.param.stim.stim_on[100,0]
# spon_onff = np.array([[spon_adpt_stt,spon_adpt_end]])
# hz_loc_spon[1,:] = fra.tuning_curve(data.a1.ge.spk_matrix, spon_onff, n_in_bin)

data_anly.hz_loc = hz_loc
data_anly.hz_loc_spon = hz_loc_spon

'''plot'''
stim_amp = np.array([400,600,800])#400
plt.figure(figsize=[8,7])
dist_bin_plot = (dist_bin+(dist_bin[1]-dist_bin[0])*0.5)[:-1]
for st in range(stim_amp.shape[0]):
    hz_loc_mean_noatt = hz_loc[st*30:st*30+30,:].mean(0)
    #hz_loc_std_noatt = hz_loc[st*30:st*30+30,:].std(0)
    hz_loc_sem_noatt = scipy.stats.sem(hz_loc[st*30:st*30+30,:])#.std(0)

    #hz_loc_mean_att = hz_loc[100:200,:].mean(0)
    
    #plt.figure(figsize=[8,6])
    #dist_bin_plot = (dist_bin+(dist_bin[1]-dist_bin[0])*0.5)[:-1]
        
    plt.errorbar(dist_bin_plot, hz_loc_mean_noatt, hz_loc_sem_noatt, ls='--', marker='o', c=clr[st], label = 'stim_amp: %.1f Hz'%(stim_amp[st]))
    #plt.plot(dist_bin_plot, hz_loc_spon[0], ls='--', marker='o', c=clr[1], label = 'spontaneous')
    
    #plt.plot(dist_bin_plot, hz_loc_mean_att, ls='-', marker='o', c=clr[0], label = 'attention; stim_amp: %.1f Hz'%(stim_amp))
    #plt.plot(dist_bin_plot, hz_loc_spon[1], ls='-', marker='o', c=clr[1], label = 'attention; spontaneous')
    
plt.plot(dist_bin_plot, hz_loc_spon[0], ls='--', marker='o', c=clr[st+1], label = 'spontaneous')
    #title = "2e1e:%.3f_2e1i:%.3f_1hz:%.2f_2hz:%.2f"%(scale_w_21_e[ie_ind//scale_w_21_i.shape[0]], scale_w_21_i[ie_ind%scale_w_21_i.shape[0]], spon_rate1,spon_rate2)
    #plt.title('average firing rate versus distance to stimulus centre; 2e1e:%.3f 2e1i%.3f\n Spike-count bin: %.1f ms\n'%(scale_w_21_e[ie_ind//scale_w_21_i.shape[0]], scale_w_21_i[ie_ind%scale_w_21_i.shape[0]],fr_bin[b])+title)
#plt.title(title+'\n'+'average firing rate versus distance to stimulus centre; Spike-count bin: %.1f ms\n'%(data.a1.param.stim.stim_on[0,1]-data.a1.param.stim.stim_on[0,0]))#+title)
#titlefft = title + '_pf%.2f_a1'%(peakF_spon)
plt.xlim([dist_bin[0],dist_bin[-1]])
plt.xlabel('distance')
plt.ylabel('Hz')
plt.legend()
plt.suptitle(title+'\n'+'average firing rate versus distance to stimulus centre; Spike-count bin: %.1f ms\n'%(data.a1.param.stim.stim_on[0,1]-data.a1.param.stim.stim_on[0,0]))#+title))
savetitle = title.replace('\n','')
tunecvfile = savetitle+'_tunecv'+'_%d'%loop_num+'.png'
plt.savefig(tunecvfile)
plt.close()
#plt.savefig(title.replace(':','')+'_tunecv'+'_%d'%loop_num+'.png')
#%%
'''firing rate time'''
mua_loca_1 = [0, 0]
mua_range_1 = 5
mua_neuron_1 = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca_1, mua_range_1, data.a1.param.width)
window = 5
dura_onoff = data.a1.param.stim.stim_on.copy()
dura_onoff[:,0] -= 100
dura_onoff[:,1] += 100
hz_t = fra.firing_rate_time_multi(data.a1.ge, mua_neuron_1, dura_onoff, window=window, n_neu_all=data.a1.param.Ne)
data_anly.hz_t = hz_t


plt.figure(figsize=[8,6])
t_plot = np.arange(dura_onoff[0,1] - dura_onoff[0,0]) - 100

for st in range(stim_amp.shape[0]):

    hz_t_noatt_mean = hz_t[st*30:st*30+30,:].mean(0)
    #hz_t_noatt_std = hz_t[st*30:st*30+30,:].std(0)
    hz_t_noatt_sem = scipy.stats.sem(hz_t[st*30:st*30+30,:])#.std(0)
    
    #hz_t_att_mean = hz_t[100:200,:].mean(0)
    #hz_t_att_std = hz_t[100:200,:].std(0)
    
    #hz_t_mean = hz_t[:,:, 0].reshape(n_amp_stim,n_per_amp,-1).mean(1)
    #for i in range(n_amp_stim_att):
    plt.plot(t_plot, hz_t_noatt_mean, ls='--', c=clr[st], label = 'stim_amp: %.1f Hz'%(stim_amp[st]))
    plt.fill_between(t_plot, hz_t_noatt_mean-hz_t_noatt_sem,hz_t_noatt_mean+hz_t_noatt_sem, \
                     ls='--', facecolor=clr[st], edgecolor=clr[st], alpha=0.2)
    #for i in range(n_amp_stim_att,n_amp_stim):
    # plt.plot(t_plot, hz_t_att_mean, ls='-', c=clr[0], label = 'attention; stim_amp: %.1f Hz'%(stim_amp[st])) 
    # plt.fill_between(t_plot, hz_t_att_mean-hz_t_att_std,hz_t_att_mean+hz_t_att_std, \
    #                  ls='-', facecolor=clr[0], edgecolor=clr[0], alpha=0.2)
plt.xlabel('ms')
plt.ylabel('Hz')
#title = "2e1e:%.3f_2e1i:%.3f_1hz:%.2f_2hz:%.2f"%(scale_w_21_e[ie_ind//scale_w_21_i.shape[0]], scale_w_21_i[ie_ind%scale_w_21_i.shape[0]], spon_rate1,spon_rate2)
#title = ''
#plt.title(title+'\n'+'firing rate-time; senssory')
plt.legend()
#plt.savefig(save_dir+title.replace(':','')+'_temp'+'_%d'%ie_ind+'.png')
#plt.savefig(title.replace(':','')+'_temp'+'_%d'%loop_num+'.png')

plt.title(title+'\n'+'firing rate-time; senssory')
savetitle = title.replace('\n','')
tempfile = savetitle+'_temp'+'_%d'%loop_num+'.png'
plt.savefig(tempfile)
plt.close()
#%%
'''animation'''
first_stim = 28; last_stim = 31
start_time = data.a1.param.stim.stim_on[first_stim,0] - 400
end_time = data.a1.param.stim.stim_on[last_stim,0] + 400

data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 15)
data.a1.ge.get_centre_mass()
data.a1.ge.overlap_centreandspike()

# data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 15)
# data.a2.ge.get_centre_mass()
# data.a2.ge.overlap_centreandspike()

#data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ni, window = 10)
#frames = int(end_time - start_time)
frames = data.a1.ge.spk_rate.spk_rate.shape[2]

stim_on_off = data.a1.param.stim.stim_on-start_time
stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]
#stim = [[[[31,31]], [stim_on_off], [[6]*stim_on_off.shape[0]]], None]
stim = [[[[31,31]], [stim_on_off], [[6]*stim_on_off.shape[0]]]]
adpt = None
#adpt = [None, [[[31,31]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
#adpt = [[[[31,31]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
ani = fra.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, spkrate2=None, \
                                        frames = frames, start_time = start_time, interval_movie=10, anititle=title,stim=stim, adpt=adpt)
savetitle = title.replace('\n','')

moviefile = savetitle+'_st1_%d'%loop_num+'.mp4'

if loop_num%1 == 0:
    ani.save(moviefile)
    pass

first_stim = 58; last_stim = 61
start_time = data.a1.param.stim.stim_on[first_stim,0] - 400
end_time = data.a1.param.stim.stim_on[last_stim,0] + 400

data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 15)
data.a1.ge.get_centre_mass()
data.a1.ge.overlap_centreandspike()

# data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 15)
# data.a2.ge.get_centre_mass()
# data.a2.ge.overlap_centreandspike()

#data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ni, window = 10)
#frames = int(end_time - start_time)
frames = data.a1.ge.spk_rate.spk_rate.shape[2]

stim_on_off = data.a1.param.stim.stim_on-start_time
stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]
#stim = [[[[31,31]], [stim_on_off], [[6]*stim_on_off.shape[0]]], None]
stim = [[[[31,31]], [stim_on_off], [[6]*stim_on_off.shape[0]]]]

#adpt = [None, [[[31,31]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
adpt = None
ani = fra.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, spkrate2=None, \
                                        frames = frames, start_time = start_time, interval_movie=10, anititle=title,stim=stim, adpt=adpt)
savetitle = title.replace('\n','')

moviefile = savetitle+'_st2_%d'%loop_num+'.mp4'

if loop_num%1 == 0:
    ani.save(moviefile)
    pass
#%%
data_anly.save(data_anly.class2dict(), datapath+'data_anly%d.file'%loop_num)
