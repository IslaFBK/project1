#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 19:43:25 2021

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
import frequency_analysis as fa
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


analy_type = 'two_stim'
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
end = int(10/dt); start = int(5/dt)
spon_rate1 = np.sum((data.a1.ge.t < end) & (data.a1.ge.t > start))/5/data.a1.param.Ne
spon_rate2 = np.sum((data.a2.ge.t < end) & (data.a2.ge.t > start))/5/data.a2.param.Ne

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
        # if data_anly.invalid_pei:
    #     title = 'p1_'+title

#%%
'''spon'''
if loop_num == 0:
    #first_stim = 9; last_stim = 10
    start_time = 5000#data.a1.param.stim.stim_on[first_stim,0] - 100
    end_time = 7000#data.a1.param.stim.stim_on[last_stim,0] + 400
    
    data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 10)
    data.a1.ge.get_centre_mass()
    data.a1.ge.overlap_centreandspike()
    
    data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 10)
    data.a2.ge.get_centre_mass()
    data.a2.ge.overlap_centreandspike()
    
    #data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ni, window = 10)
    #frames = int(end_time - start_time)
    frames = data.a1.ge.spk_rate.spk_rate.shape[2]
    
    # stim_on_off = data.a1.param.stim.stim_on-(data.a1.param.stim.stim_on[first_stim,0] - 100)
    # stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]
    # stim = [[[[31,31]], [stim_on_off], [[6]*stim_on_off.shape[0]]], None]
    
    ani = firing_rate_analysis.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, spkrate2=data.a2.ge.spk_rate.spk_rate, \
                                            frames = frames, start_time = start_time, interval_movie=10, anititle='',stim=None, adpt=None)
    savetitle = title.replace('\n','')
    
    moviefile = '1spon'+savetitle+'_%d'%loop_num+'.mp4'
    #if loop_num%1 == 0:
    ani.save(moviefile)
# '''fft'''
# def find_peakF(coef, freq, lwin):
#     dF = freq[1] - freq[0]
#     #Fwin = 0.3
#     #lwin = 3#int(Fwin/dF)
#     win = np.ones(lwin)/lwin
#     coef_avg = np.convolve(np.abs(coef[1:]), win, mode='same')
#     peakF = freq[1:][coef_avg.argmax()]
#     return peakF
# #%%
# start_time = 0; end_time = 32e3
# data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 10)
# mua = data.a1.ge.spk_rate.spk_rate.reshape(data.a1.param.Ne,-1)[data.a1.param.chg_adapt_neuron]
# mua = mua.mean(0)/0.01

# fig, [ax1,ax2] = plt.subplots(1,2,figsize=[9,4])

# fs = 1000
# data_fft = mua[4000:20000]
# coef, freq = fa.myfft(data_fft, fs)
# data_anly.coef_spon = coef
# data_anly.freq_spon = freq

# peakF_spon = find_peakF(coef, freq, 3)

# freq_max = 20
# ind_len = freq[freq<freq_max].shape[0] # int(20/(fs/2)*(len(data_fft)/2)) + 1
# im1 = ax1.plot(freq[1:ind_len], np.abs(coef[1:ind_len]),label='spon')
# ax2.loglog(freq[1:ind_len], np.abs(coef[1:ind_len]),label='spon_log')

# ax1.set_title('spon')

# data_fft = mua[22000:32000]
# coef, freq = fa.myfft(data_fft, fs)
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
'''
firing rate-location
'''
# stim_loc = np.array([0,0])

# dist = cn.coordination.lattice_dist(data.a1.param.e_lattice,62,stim_loc)
# dist_bin = np.arange(0, 31.5*2**0.5, 2.5)
# n_in_bin = [None]*(dist_bin.shape[0]-1)
# neuron = np.arange(data.a1.param.Ne)

# for i in range(len(dist_bin)-1):
#     n_in_bin[i] = neuron[(dist >= dist_bin[i]) & (dist < dist_bin[i+1])]


# simu_time_tot = 28000

# N_stim = data.a1.param.stim.stim_amp_scale.shape[0]

# data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])
# #%%
# fr_bin = np.array([150, 200, 250]) 

# hz_loc = np.zeros([N_stim,len(dist_bin)-1,len(fr_bin)])
# #n_per_stim = 5

       
# #%%
# for b in range(len(fr_bin)):
#     for i in range(N_stim):
#         for j in range(len(n_in_bin)):
#         #data.a1.ge.spk_matrix[]
    
#             hz_loc[i, j, b] = data.a1.ge.spk_matrix[n_in_bin[j], data.a1.param.stim.stim_on[i,0]*10:(data.a1.param.stim.stim_on[i,0]+fr_bin[b])*10].sum()/n_in_bin[j].shape[0]
# #%%

# hz_spon = np.zeros([len(n_in_bin)])

# for j in range(len(n_in_bin)):
#     hz_spon[j] = data.a1.ge.spk_matrix[n_in_bin[j], 5000*10:10000*10].sum()/n_in_bin[j].shape[0]


# data_anly.hz_loc = hz_loc
# data_anly.hz_spon = hz_spon

'''
firing rate-time
'''
N_stim = data.a1.param.stim.stim_amp_scale.shape[0]
dura = 400
hz_t = np.zeros([N_stim, 400, 2])
mua_loca_1 = [0, 0]
mua_range_1 = 5
mua_neuron_1 = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca_1, mua_range_1, data.a1.param.width)
window = 5

for i in range(N_stim):
    
    start_time = data.a1.param.stim.stim_on[i,0]; end_time = data.a1.param.stim.stim_on[i,0] + dura
    data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = window)
    hz_t[i, :data.a1.ge.spk_rate.spk_rate.shape[-1], 0] = data.a1.ge.spk_rate.spk_rate.reshape(data.a1.param.Ne,-1)[mua_neuron_1].mean(0)/(window/1000)
    
    data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = window)
    hz_t[i, :data.a2.ge.spk_rate.spk_rate.shape[-1], 1] = data.a2.ge.spk_rate.spk_rate.reshape(data.a2.param.Ne,-1)[mua_neuron_1].mean(0)/(window/1000)

data_anly.hz_t = hz_t

if loop_num%1 == 0:
    stim_amp = np.arange(2,5)*200
    clr = ['b', 'tab:orange', 'g']
    #hz_t /= 81
    
    hz_t_mean_1 = hz_t[:,:,0].reshape(3,10,-1).mean(1)
    hz_t_mean_2 = hz_t[:,:,1].reshape(3,10,-1).mean(1)
    
    fig, ax = plt.subplots(1,2,figsize=[10,5])
    for i in range(3):
        ax[0].plot(hz_t_mean_1[i], c=clr[i], label = 'stim_amp: %.1f Hz; a1'%(stim_amp[i]))
        ax[1].plot(hz_t_mean_2[i], c=clr[i], ls='-.', label = 'stim_amp: %.1f Hz; a2'%(stim_amp[i]))
    ax[1].set_ylim(ax[0].get_ylim())                                 
    for axis in ax: axis.legend()
    plt.xlabel('ms')
    plt.ylabel('Hz')
    plt.suptitle('average firing rate versus time (10 trials)\n'+title)
    savetitle = title.replace('\n','')

    hz_t_file = savetitle+'_temp_%d'%loop_num+'.png'
    plt.savefig(hz_t_file)

#%%
'''wavelet'''
# if loop_num%5 == 0:

#     mua_loca_1 = [0, 0]
#     mua_range_1 = 5
#     mua_neuron_1 = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca_1, mua_range_1, data.a1.param.width)
#     window = 5
#     start_time = 5000; end_time = 20000
#     data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = window)
    
#     mua = data.a1.ge.spk_rate.spk_rate.reshape(data.a1.param.Ne,-1)[mua_neuron_1].mean(0)/(window/1000)
    
#     coef, freq = fa.mycwt(mua, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)
#     fig = fa.plot_cwt(coef, freq, base = 10)
#     plt.xlabel('ms')
#     plt.ylabel('Hz')
#     plt.title('wavelet\n'+title)
    
#     savetitle = title.replace('\n','')
    
#     wvt_file = savetitle+'_wvt_%d'%loop_num+'.png'
#     fig.savefig(wvt_file)

#%%


'''
#%%
for b in range(len(fr_bin)):
    hz_loc_tmp = hz_loc[:,:,b].reshape(6,5,-1)
    
    hz_loc_mean = hz_loc_tmp.mean(1)
    #plt.matshow(hz_loc_mean)#[:,0,:])
    
    stim_amp = np.unique(data.a1.param.stim.stim_amp_scale)*200
    
    plt.figure()
    for i in range(hz_loc_mean.shape[0]):
        plt.plot(hz_loc_mean[i], '-o', label = 'stim_amp: %.1f Hz'%(stim_amp[i]))
        plt.title('counts bins: %.1f ms'%fr_bin[b])
    plt.legend()
#%%
spk_mua = data.a1.ge.spk_matrix[n_in_bin[0],:].sum(0).A[0]
#%%
spk_mua_mean = np.convolve(spk_mua, np.ones(100),'same')/0.01/len(n_in_bin[0])
#%%
stim_t = np.vstack((data.a1.param.stim.stim_on[:,0],data.a1.param.stim.stim_on[:,0]))

stim_t_2 =  np.vstack((np.ones(len(data.a1.param.stim.stim_on[:,0])), np.ones(len(data.a1.param.stim.stim_on[:,0]))*100))

#%%
plt.figure()
plt.plot(spk_mua_mean[::10])
plt.plot(stim_t, stim_t_2)
#%%
sua = data.a1.ge.spk_matrix[2000, :].A[0]
sua_mean = np.convolve(sua, np.ones(100),'same')/0.01
plt.figure()
plt.plot(sua_mean[::10])
plt.plot(stim_t, stim_t_2)
#%%
plt.figure()
plt.plot(sua.nonzero()[0]/10,np.ones(sua.nonzero()[0].shape),'|')
plt.plot(stim_t, stim_t_2)

#%%
'''
#%%
'''animation'''
first_stim = 9; last_stim = 10
start_time = data.a1.param.stim.stim_on[first_stim,0] - 100
end_time = data.a1.param.stim.stim_on[last_stim,0] + 400

data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 10)
data.a1.ge.get_centre_mass()
data.a1.ge.overlap_centreandspike()

data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 10)
data.a2.ge.get_centre_mass()
data.a2.ge.overlap_centreandspike()

#data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ni, window = 10)
#frames = int(end_time - start_time)
frames = data.a1.ge.spk_rate.spk_rate.shape[2]

stim_on_off = data.a1.param.stim.stim_on-(data.a1.param.stim.stim_on[first_stim,0] - 100)
stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]
stim = [[[[31,31]], [stim_on_off], [[6]*stim_on_off.shape[0]]], None]

ani = firing_rate_analysis.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, spkrate2=data.a2.ge.spk_rate.spk_rate, \
                                        frames = frames, start_time = start_time, interval_movie=10, anititle='',stim=stim, adpt=None)
savetitle = title.replace('\n','')

moviefile = savetitle+'_%d'%loop_num+'.mp4'
#if loop_num%1 == 0:
    #ani.save(moviefile)
#%%
'''
start_time = 4e3; end_time = 6e3
data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 10)
data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 10)

#data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ni, window = 10)
#data.a1.ge.get_centre_mass(detect_pattern=True)
#pattern_size2 = data.a1.ge.centre_mass.pattern_size.copy()
#pattern_size2[np.invert(data.a1.ge.centre_mass.pattern)] = 0

frames = data.a1.ge.spk_rate.spk_rate.shape[2]
# ani = firing_rate_analysis.show_pattern(data.a1.ge.spk_rate.spk_rate, None, frames = frames, start_time = start_time, anititle=title,\
#                                         show_pattern_size=True, pattern_ctr=data.a1.ge.centre_mass.centre_ind, \
#                                             pattern_size=pattern_size2)
ani = firing_rate_analysis.show_pattern(data.a1.ge.spk_rate.spk_rate, data.a2.ge.spk_rate.spk_rate, frames = frames, start_time = start_time, \
                                        interval_movie=20, anititle=title, show_pattern_size=False)

#data.a1.param.scale_ie_1

#savetitle = title
savetitle = title.replace('\n','')

moviefile = savetitle+'_1'+'_%d'%loop_num+'.mp4'
ani.save(moviefile)

start_time = 12e3; end_time = 14e3
data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 10)
data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 10)

#data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ni, window = 10)
#data.a1.ge.get_centre_mass(detect_pattern=True)
#pattern_size2 = data.a1.ge.centre_mass.pattern_size.copy()
#pattern_size2[np.invert(data.a1.ge.centre_mass.pattern)] = 0

frames = data.a1.ge.spk_rate.spk_rate.shape[2]
# ani = firing_rate_analysis.show_pattern(data.a1.ge.spk_rate.spk_rate, None, frames = frames, start_time = start_time, anititle=title,\
#                                         show_pattern_size=True, pattern_ctr=data.a1.ge.centre_mass.centre_ind, \
#                                             pattern_size=pattern_size2)
ani = firing_rate_analysis.show_pattern(data.a1.ge.spk_rate.spk_rate, data.a2.ge.spk_rate.spk_rate, frames = frames, start_time = start_time, \
                                        interval_movie=20, anititle=title, show_pattern_size=False)

#data.a1.param.scale_ie_1

#savetitle = title
savetitle = title.replace('\n','')

moviefile = savetitle+'_2'+'_%d'%loop_num+'.mp4'
ani.save(moviefile)
'''
#%%
'''
start_time = int(5e3); end_time = int(6e3)
#data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 10)
#data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ni, window = 10)
frames = int(end_time - start_time)
#frames = data.a1.ge.spk_rate.spk_rate.shape[2]
# ani = firing_rate_analysis.show_pattern(data.a1.ge.spk_rate.spk_rate, None, frames = frames, start_time = start_time, anititle=title,\
#                                         show_pattern_size=True, pattern_ctr=data.a1.ge.centre_mass.centre_ind, \
#                                             pattern_size=pattern_size2)
ani = firing_rate_analysis.show_timev(data.a1.ge.v[:,start_time:end_time].reshape(63,63,-1),\
                                      data.a1.gi.v[:,start_time:end_time].reshape(32,32,-1), vrange=[[-80,-50],[-80,-50]],\
                                      frames = frames, start_time = start_time, \
                                      interval_movie=20, anititle=title)

#data.a1.param.scale_ie_1

#savetitle = title
savetitle = title.replace('\n','')

moviefile = savetitle+'_v'+'_%d'%loop_num+'.mp4'
ani.save(moviefile)
'''
# start_time = 12e3; end_time = 14e3
# data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 10)
# data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 10)

# #data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ni, window = 10)
# #data.a1.ge.get_centre_mass(detect_pattern=True)
# #pattern_size2 = data.a1.ge.centre_mass.pattern_size.copy()
# #pattern_size2[np.invert(data.a1.ge.centre_mass.pattern)] = 0

# frames = data.a1.ge.spk_rate.spk_rate.shape[2]
# # ani = firing_rate_analysis.show_pattern(data.a1.ge.spk_rate.spk_rate, None, frames = frames, start_time = start_time, anititle=title,\
# #                                         show_pattern_size=True, pattern_ctr=data.a1.ge.centre_mass.centre_ind, \
# #                                             pattern_size=pattern_size2)
# ani = firing_rate_analysis.show_pattern(data.a1.ge.spk_rate.spk_rate, data.a2.ge.spk_rate.spk_rate, frames = frames, start_time = start_time, \
#                                         interval_movie=20, anititle=title, show_pattern_size=False)

# #data.a1.param.scale_ie_1

# #savetitle = title
# savetitle = title.replace('\n','')

# moviefile = savetitle+'_2'+'_%d'%loop_num+'.mp4'
# ani.save(moviefile)



#del data.a1.ge.spk_rate

#%%

'''rate'''
'''
# e_lattice = cn.coordination.makelattice(int(np.sqrt(data.a1.param.Ne).round()),data.a1.param.width,[0,0])

mua_loca_1 = [0,0]#data.a1.param.chg_adapt_loca #[0, 0]
mua_range_1 = 5#data.a1.param.chg_adapt_range #6 
mua_neuron_1 = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca_1, mua_range_1, data.a1.param.width)

mua_neuron_2 = mua_neuron_1
#%%

start_time = 4e3; end_time = 6e3
data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 10)
data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 10)

mua1 = data.a1.ge.spk_rate.spk_rate.reshape(data.a1.param.Ne,-1)[mua_neuron_1]
mua1 = mua1.mean(0)/0.01

mua2 = data.a2.ge.spk_rate.spk_rate.reshape(data.a2.param.Ne,-1)[mua_neuron_2]
mua2 = mua2.mean(0)/0.01

fig, ax = plt.subplots(4,1,figsize=[8,6])
ax[0].plot(np.arange(len(mua1))+start_time, mua1,label='sensory')
ax[1].plot(np.arange(len(mua2))+start_time, mua2,label='associate')

start_time = 12e3; end_time = 14e3
data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 10)
data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 10)

mua1 = data.a1.ge.spk_rate.spk_rate.reshape(data.a1.param.Ne,-1)[mua_neuron_1]
mua1 = mua1.mean(0)/0.01

mua2 = data.a2.ge.spk_rate.spk_rate.reshape(data.a2.param.Ne,-1)[mua_neuron_2]
mua2 = mua2.mean(0)/0.01

ax[2].plot(np.arange(len(mua1))+start_time, mua1,label='sensory')
ax[3].plot(np.arange(len(mua2))+start_time, mua2,label='associate')
for axis in ax:
    axis.legend()
#ax[0].set_title(title)
fig.suptitle(title)

savetitle = title.replace('\n','')

ratefile = savetitle+'_rate_%d'%loop_num+'.png'
fig.savefig(ratefile)

del data.a1.ge.spk_rate
'''
#%%
"""
'''move good results'''
if spon_rate_good and stable_good and peakF_good:
    if data_anly.patterns_size_mean <= 7.2:
        if not os.path.isdir(goodsize_dir):
            os.mkdir(goodsize_dir)
        shutil.copyfile(moviefile, goodsize_dir+moviefile)
        shutil.copyfile(ratefile, goodsize_dir+ratefile)
        shutil.copyfile(fftfile, goodsize_dir+fftfile)
    if not os.path.isdir(good_dir):
        os.mkdir(good_dir)
    shutil.move(moviefile, good_dir)
    shutil.move(ratefile, good_dir)
    shutil.move(fftfile, good_dir)
#%%
"""
#%%
#data_anly.save(data_anly.class2dict(), datapath+'data_anly%d.file'%loop_num)

#%%
"""
''' MSD '''
data.a1.ge.get_MSD(start_time=3000, end_time=10000, window = 15, jump_interval=np.arange(1,1000,5), fit_stableDist='Matlab')

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
"""
