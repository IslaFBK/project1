#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 17:30:40 2021

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
import connection as cn
import pickle
import sys
import os
import matplotlib.pyplot as plt
import shutil
from cfc_analysis import cfc
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

data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
#%%
title = 'sens'
title = title + '_stim%.1f'%(data.a1.param.stim.maxrate_stim)
title = title + '_eier%.2f_iier%.2f'%(data.param.ie_r_e1, data.param.ie_r_i1)
#savefftfile = True
savefftfile = False; saveratefile = False;
savecwtfile = False; 
saveMSDfile = False; savecfcfile = False
saveani_adapt = False; saveani_stim = False

if loop_num%25 == 0:
    savefftfile = True; saveratefile = True;
    savecwtfile =True; 
    saveMSDfile = True; savecfcfile = False
    saveani_adapt = True; saveani_stim = False
#%%
data_anly = mydata.mydata()
'''spon rate'''
dt = 1/10000;
end = int(20/dt); start = int(5/dt)
spon_rate = np.sum((data.a1.ge.t < end) & (data.a1.ge.t > start))/15/data.a1.param.Ne
data_anly.spon_rate = spon_rate

title = title + '_hzspon%.2f'%spon_rate
#%%
'''pattern size'''
start_time = 5e3; end_time = 20e3
data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 10)
data.a1.ge.get_centre_mass(detect_pattern=True)

#data_anly.pattern = data.a1.ge.centre_mass.pattern
#data_anly.pattern_size = data.a1.ge.centre_mass.pattern_size
data_anly.patterns_size_mean = data.a1.ge.centre_mass.pattern_size[data.a1.ge.centre_mass.pattern].mean()
data_anly.pattern_on_ratio = data.a1.ge.centre_mass.pattern.sum()/data.a1.ge.centre_mass.pattern.shape[0]
data_anly.patterns_size_std = data.a1.ge.centre_mass.pattern_size[data.a1.ge.centre_mass.pattern].std()
title = title + '_ptsz%.1f_pton%.2f'%(data_anly.patterns_size_mean, data_anly.pattern_on_ratio)

'''MSD alpha'''
data.a1.ge.get_MSD(start_time=5000, end_time=20000, n_neuron=data.a1.param.Ne, window = 15, jump_interval=np.array([15]), fit_stableDist='pylevy')
data_anly.alpha = data.a1.ge.MSD.stableDist_param[0,0]
title = title + '_alpha%.2f'%data.a1.ge.MSD.stableDist_param[0,0]
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

def plot_fft(data_fft, fs = 1000, freq_max1=20, freq_max2 = 200, fig=None, ax=None, label=''):
    if fig is None:
        fig, ax = plt.subplots(2,2,figsize=[9,9])
    
    #fs = 1000
    #data_fft = mua[:]
    coef, freq = fqa.myfft(data_fft, fs)
    # data_anly.coef_spon = coef
    # data_anly.freq_spon = freq
    
    peakF_spon = find_peakF(coef, freq, 3)
    
    #freq_max1 = 20
    ind_len = freq[freq<freq_max1].shape[0] # int(20/(fs/2)data_fft*(len(data_fft)/2)) + 1
    ax[0,0].plot(freq[1:ind_len], np.abs(coef[1:ind_len]),label=label+'_linear')
    ax[0,1].loglog(freq[1:ind_len], np.abs(coef[1:ind_len]),label=label+'_loglog')
    
    #freq_max2 = 150
    ind_len = freq[freq<freq_max2].shape[0] # int(20/(fs/2)*(len(data_fft)/2)) + 1
    ax[1,0].plot(freq[1:ind_len], np.abs(coef[1:ind_len]),label=label+'_linear')
    ax[1,1].loglog(freq[1:ind_len], np.abs(coef[1:ind_len]),label=label+'_loglog')
    
    for i in range(2):
        for j in range(2):
            ax[i,j].legend()
    
    return fig, ax, peakF_spon, coef, freq
#%%
mua_loca = [0, 0]
mua_range = 5 
mua_neuron = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca, mua_range, data.a1.param.width)
#%%

start_time = 5e3; end_time = 20e3
window = 5
data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 5)
mua = data.a1.ge.spk_rate.spk_rate.reshape(data.a1.param.Ne,-1)[mua_neuron]
mua = mua.mean(0)/(window/1000)

fig, ax, peakF_spon, coef, freq = plot_fft(mua, fs = 1000, fig=None, ax=None, label='spon')
data_anly.coef_spon = coef
data_anly.freq_spon = freq
data_anly.pkf_spon = peakF_spon

start_time = 20e3; end_time = 35e3
window = 5
data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 5)
mua = data.a1.ge.spk_rate.spk_rate.reshape(data.a1.param.Ne,-1)[mua_neuron]
mua = mua.mean(0)/(window/1000)

fig, ax, peakF_adapt, coef, freq = plot_fft(mua, fs = 1000, fig=fig, ax=ax, label='adapt')
data_anly.coef_adapt = coef
data_anly.freq_adapt = freq
data_anly.pkf_adapt = peakF_adapt


titlefft = title + '_pf_spon%.2f_pf_adapt%.2f_fft'%(peakF_spon,peakF_adapt)
fig.suptitle(titlefft)

savetitle = titlefft.replace('\n','')
fftfile = savetitle+'_%d'%(loop_num)+'.png'
if savefftfile == True:
    fig.savefig(fftfile)
#%%
#fft_t = 2*np.random.randn(1000)
# cwt_t = 2*np.sin(2*np.pi*np.arange(10000)/1000)
# cwt_t2 = 2*np.sin(4*np.pi*np.arange(10000)/1000)
# cwt_t3 = np.zeros(10000)
# cwt_t3[:5000] = cwt_t[:5000]
# cwt_t3[5000:] = cwt_t2[5000:]
#%%
'''firing rate'''
start_time = 18e3; end_time = 25e3
window = 5
data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 5)
mua = data.a1.ge.spk_rate.spk_rate.reshape(data.a1.param.Ne,-1)[mua_neuron]
mua = mua.mean(0)/(window/1000)
fig = plt.figure(figsize=[9,5])
plt.plot(np.arange(mua.shape[0])+start_time, mua)
plt.xlabel('ms')
plt.ylabel('Hz')
titlerate = title + '_pf_spon%.2f_pf_adapt%.2f_rate'%(peakF_spon,peakF_adapt)
fig.suptitle(titlerate)

savetitle = titlerate.replace('\n','')
ratefile = savetitle+'_%d'%(loop_num)+'.png'
if saveratefile == True:
    fig.savefig(ratefile)

#%%
'''cwt'''
fig, ax = plt.subplots(2,1,figsize=[9,9])

start_time = 5e3; end_time = 15e3
window = 5
data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 5)
mua = data.a1.ge.spk_rate.spk_rate.reshape(data.a1.param.Ne,-1)[mua_neuron]
mua = mua.mean(0)/(window/1000)

coef, freq = fqa.mycwt(mua, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)

fig, ax[0], ax02 = fqa.plot_cwt(coef, freq, base = 10, fig=fig, ax=ax[0])
ax[0].set_title('spon')

start_time = 20e3; end_time = 30e3
window = 5
data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 5)
mua = data.a1.ge.spk_rate.spk_rate.reshape(data.a1.param.Ne,-1)[mua_neuron]
mua = mua.mean(0)/(window/1000)

coef, freq = fqa.mycwt(mua, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)

#fig, ax = plt.subplots(3,1,figsize=[9,9])
fig, ax[1], ax12 = fqa.plot_cwt(coef, freq, base = 10, fig=fig, ax=ax[1])
ax[1].set_title('adapt')

# start_time = 40e3; end_time = 50e3
# window = 5
# data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 5)
# mua = data.a1.ge.spk_rate.spk_rate.reshape(data.a1.param.Ne,-1)[mua_neuron]
# mua = mua.mean(0)/(window/1000)

# coef, freq = fqa.mycwt(mua, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)

# #fig, ax = plt.subplots(3,1,figsize=[9,9])
# fig, ax[2], ax22 = fqa.plot_cwt(coef, freq, base = 10, fig=fig, ax=ax[2])
# ax[2].set_title('stim')

titlecwt = title + '_cwt'#%(peakF_spon,peakF_adapt)
fig.suptitle(titlecwt)

savetitle = titlecwt.replace('\n','')
cwtfile = savetitle+'_%d'%(loop_num)+'.png'
if savecwtfile:
    fig.savefig(cwtfile)
#%%

''' MSD '''
data.a1.ge.get_MSD(start_time=5000, end_time=20000, n_neuron = data.a1.param.Ne, \
                   sample_interval = 1, slide_interval = 1, dt = 0.1, \
                   window = 10, jump_interval=np.logspace(0.1,3,10), fit_stableDist='Matlab')

data_anly.MSD = data.a1.ge.MSD
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

alpha = data.a1.ge.MSD.stableDist_param[2,0,0]
titleMSD = title + '_alpha%.2f'%alpha
ax1.set_title(titleMSD)

savetitle = titleMSD.replace('\n','')
MSDfile = savetitle+'_%d'%(loop_num)+'.png'
if saveMSDfile:
    fig.savefig(MSDfile)

#%%
'''cfc'''
if savecfcfile:
    start_time = 40e3; end_time = 55e3
    window = 5
    data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 5)
    mua = data.a1.ge.spk_rate.spk_rate.reshape(data.a1.param.Ne,-1)[mua_neuron]
    mua = mua.mean(0)/(window/1000)
    
    findcfc = cfc.cfc()
    Fs = 1000;
    #timeDim = 0;
    phaseBand = np.arange(1,14.1,0.5)
    ampBand = np.arange(10,101,5) 
    phaseBandWid = 0.49 ;
    ampBandWid = 5 ;
    
    band1 = np.concatenate((phaseBand - phaseBandWid, ampBand - ampBandWid)).reshape(1,-1)
    band2 = np.concatenate((phaseBand + phaseBandWid, ampBand + ampBandWid)).reshape(1,-1)
    subBand = np.concatenate((band1,band2),0)
    subBand = subBand.T
    #
    ##%%
    
    findcfc.timeDim = -1;
    findcfc.Fs = Fs; 
    findcfc.phaseBand = subBand[:len(phaseBand)];
    findcfc.ampBand = subBand[len(phaseBand):]
    #findcfc.find_cfc_from_rawsig(mua)
    
    #findcfc.section_input_to_find_MI_cfc = [4000,40000]
    MI_raw, MI_surr, meanBinAmp = findcfc.find_cfc_from_rawsig(mua,return_Ampdist=True)
    #%%
    phaseBandWid = 0.5#0.49 ;
    ampBandWid = 5 ;
    phaseBand = np.arange(1,14.1,0.5)
    ampBand = np.arange(10,101,5) 
    
    fig, [ax1,ax2] = plt.subplots(2,1, figsize=[7,9])
    #x_range = np.arange(phaseBand[0]-phaseBandWid/2, phaseBand[-1]+phaseBandWid/2+1)
    #y_range = np.arange(ampBand[0]-ampBandWid/2, ampBand[-1]+ampBandWid/2+1)
    
    #im = ax1.imshow(np.flip(MI_raw.T, 0), aspect='auto', extent=[phaseBand[0]-phaseBandWid/2, phaseBand[-1]+phaseBandWid/2, ampBand[0]-ampBandWid/2, ampBand[-1]+ampBandWid/2])
    imc = ax1.contourf(phaseBand, ampBand, MI_raw.T, 15)#, aspect='auto')
    imcc = ax1.contour(phaseBand, ampBand, MI_raw.T, 15, colors='k', linewidths=0.3)#, aspect='auto')
    
    imc2 = ax2.contourf(phaseBand, ampBand, MI_surr.T, 15)#, aspect='auto')
    imcc2 = ax2.contour(phaseBand, ampBand, MI_surr.T, 15, colors='k', linewidths=0.3)#, aspect='auto')
    
    #imc2 = ax1.contour(phaseBand, ampBand, MI_raw.T, 15)#, aspect='auto')
    
    #imc = ax1.contourf(MI_raw.T, 12, extent=[phaseBand[0]-phaseBandWid/2, phaseBand[-1]+phaseBandWid/2, ampBand[0]-ampBandWid/2, ampBand[-1]+ampBandWid/2])#, aspect='auto')
    #imc = ax1.contourf(MI_raw.T, 12, origin='lower', extent=[phaseBand[0]-phaseBandWid/2, phaseBand[-1]+phaseBandWid/2, ampBand[0]-ampBandWid/2, ampBand[-1]+ampBandWid/2])#extent=[phaseBand[0]-phaseBandWid/2, phaseBand[-1]+phaseBandWid/2, ampBand[0]-ampBandWid/2, ampBand[-1]+ampBandWid/2])#, aspect='auto')
    
    #imi = ax2.imshow(np.flip(MI_raw_mat.T, 0), aspect='auto', extent=[phaseBand[0]-phaseBandWid/2, phaseBand[-1]+phaseBandWid/2, ampBand[0]-ampBandWid/2, ampBand[-1]+ampBandWid/2])
    plt.colorbar(imc, ax=ax1)
    plt.colorbar(imc2, ax=ax2)
    ax1.set_xlabel('phase frequency (Hz)')
    ax1.set_ylabel('Amplitude frequency (Hz)')
    ax1.set_title('raw')
    ax2.set_xlabel('phase frequency (Hz)')
    ax2.set_ylabel('Amplitude frequency (Hz)')
    ax2.set_title('minus-surr')
    #plt.suptitle('ee1.20_ei1.27_ie1.2137_ii1.08_dsi4.44_dse5.00\n/headnode1/shni2598/brian2/NeuroNet_brian/\nmodel_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp')
    
    titlecfc = title + '_cfc'
    plt.suptitle(titlecfc)
    
    savetitle = titlecfc.replace('\n','')
    cfcfile = savetitle+'_%d'%(loop_num)+'.png'
    fig.savefig(cfcfile)

#%%
'''animation'''

start_time = 19e3; end_time = 22e3
data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 10)
#data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ni, window = 10)

#data.a1.ge.get_MSD(start_time=3000, end_time=10000, n_neuron=data.a1.param.Ne, window = 15, jump_interval=np.array([15]), fit_stableDist='pylevy')
data.a1.ge.get_centre_mass()
data.a1.ge.overlap_centreandspike()

#data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ni, window = 10)
#frames = int(end_time - start_time)
frames = data.a1.ge.spk_rate.spk_rate.shape[2]

#stim_on_off = data.a1.param.stim.stim_on-start_time
#stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]
#stim_on_off = np.array([[1000,3000]])

#stim = [[[[31,31]], [stim_on_off], [[6]*stim_on_off.shape[0]]], None]
adpt = [[[[32,32]], [[[1000, 3000]]], [[6]]]]
ani = fra.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, \
                                        frames = frames, start_time = start_time, interval_movie=15, anititle='',stim=None, adpt=adpt)
savetitle = title.replace('\n','')

moviefile = savetitle+'_adapt_%d'%loop_num+'.mp4'

if saveani_adapt:
    ani.save(moviefile)


# start_time = 39e3; end_time = 42e3
# data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 10)
# #data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ni, window = 10)

# #data.a1.ge.get_MSD(start_time=3000, end_time=10000, n_neuron=data.a1.param.Ne, window = 15, jump_interval=np.array([15]), fit_stableDist='pylevy')
# data.a1.ge.get_centre_mass()
# data.a1.ge.overlap_centreandspike()

# #data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ni, window = 10)
# #frames = int(end_time - start_time)
# frames = data.a1.ge.spk_rate.spk_rate.shape[2]

# #stim_on_off = data.a1.param.stim.stim_on-start_time
# #stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]
# stim_on_off = np.array([[1000,3000]])

# stim = [[[[32,32],[64,0]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]]]
# #adpt = [[[[31,31]], [[[1000, 3000]]], [[6]]]]
# ani = fra.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, \
#                                         frames = frames, start_time = start_time, interval_movie=15, anititle='',stim=stim, adpt=None)
# savetitle = title.replace('\n','')

# moviefile = savetitle+'_stim_%d'%loop_num+'.mp4'
# if saveani_stim:
#     ani.save(moviefile)
#%%
data_anly.save(data_anly.class2dict(), datapath+'data_anly%d.file'%loop_num)





