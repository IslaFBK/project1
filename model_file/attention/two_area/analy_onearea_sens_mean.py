#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 17:58:24 2021

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
datapath = ''#'/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/onearea_4096/verify/asso_electrode1/'
#sys_argv = int(sys.argv[1])
#loop_num = sys_argv #rep_ind*20 + ie_num
good_dir = 'good/'
goodsize_dir = 'good_size/'
save_dir = 'mean_results/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#%%
def find_peakF(coef, freq, lwin):
    dF = freq[1] - freq[0]
    #Fwin = 0.3
    #lwin = 3#int(Fwin/dF)
    win = np.ones(lwin)/lwin
    coef_avg = np.convolve(np.abs(coef[1:]), win, mode='same')
    peakF = freq[1:][coef_avg.argmax()]
    return peakF

def plot_fft(freq, coef, freq_max1=20, freq_max2 = 200, fig=None, ax=None, label=''):
    if fig is None:
        fig, ax = plt.subplots(2,2,figsize=[9,9])
    
    #fs = 1000
    #data_fft = mua[:]
    #coef, freq = fqa.myfft(data_fft, fs)
    # data_anly.coef_spon = coef
    # data_anly.freq_spon = freq
    
    #peakF = find_peakF(coef, freq, 3)
    
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
    
    return fig, ax, #peakF#, coef, freq
#%%
data_anly = mydata.mydata()
data = mydata.mydata()
loop_num = 0
mua_loc_ind = 0
#data_anly_file = 'data_anly_electrode2_'
#data_anly_file = 'data_anly'

data_anly.load(datapath+'data_anly%d.file'%(loop_num))

freq_spon = data_anly.freq_spon
coef_spon_mean = np.zeros(data_anly.coef_spon.shape)
freq_adapt = data_anly.freq_adapt
coef_adapt_mean = np.zeros(data_anly.coef_adapt.shape)

spon_rate = 0
#alpha = 0
#MI_raw = np.zeros(data_anly.cfc.MI_raw.shape)
#MI_surr = np.zeros(data_anly.cfc.MI_surr.shape)

MSD = np.zeros(data_anly.MSD.MSD.shape)
#data_anly.MSD.jump_interval, data_anly.MSD.MSD


repeat = 50
para_n = 28#81
mua_loca_all = np.array([[0,0]])#np.array([[0,0],[-32,-32],[0,32]])
divd = repeat

for para in range(para_n):
    #print(para_n)
    #for mua_loc_ind in range(1):#3
    coef_spon_mean[:] = 0
    coef_adapt_mean[:] = 0
    spon_rate = 0
    #alpha = 0
    #MI_raw[:] = 0
    #MI_surr[:] = 0
    MSD[:] = 0
    mua_loca = mua_loca_all[mua_loc_ind]
    divd = repeat
    for loop_num in range(para*repeat, para*repeat+repeat):

        #try:data_anly.load(datapath+'data_anly_electrode_loc%d_%d.file'%(mua_loc_ind,loop_num))
        try:data_anly.load(datapath+'data_anly%d.file'%(loop_num))
        except FileNotFoundError:
            divd -= 1
            continue
        
        coef_spon_mean += np.abs(data_anly.coef_spon)
        coef_adapt_mean += np.abs(data_anly.coef_adapt)
        #MI_raw += data_anly.cfc.MI_raw
        #MI_surr += data_anly.cfc.MI_surr            
        MSD += data_anly.MSD.MSD
        spon_rate += data_anly.spon_rate
    
    data.load(datapath+'data%d.file'%loop_num)

    coef_spon_mean /= divd
    coef_adapt_mean /= divd
    #MI_raw /= divd
    #MI_surr /= divd
    MSD /= divd
    spon_rate /= divd; #print(spon_rate)
    
    title = 'sens'#'asso'
    title = title + '_eier%.3f_iier%.3f\n'%(data.param.ie_r_e1, data.param.ie_r_i1)
    pkf_spon = find_peakF(coef_spon_mean, freq_spon, 3)#freq_spon[1:][coef_spon_mean[1:].argmax()]
    pkf_adapt = find_peakF(coef_adapt_mean, freq_adapt, 3)#freq_adapt[1:][coef_adapt_mean[1:].argmax()]
    
    title += '_hz%.2f_pkfspon%.2f_pkfadpt%.2f_eleloc%d'%(spon_rate,pkf_spon,pkf_adapt,mua_loc_ind)
    '''fft'''
    fig, ax = plt.subplots(2,2,figsize=[9,9])
    fig, ax = plot_fft(freq_spon, coef_spon_mean, freq_max1=20, freq_max2 = 200, fig=fig, ax=ax, label='spon')
    fig, ax = plot_fft(freq_adapt, coef_adapt_mean, freq_max1=20, freq_max2 = 200, fig=fig, ax=ax, label='adapt')
    for i in range(2):
        for j in range(2):
            ax[i,j].plot([3,3],[0,6],'r--',lw=0.5)
            ax[i,j].plot([4,4],[0,6],'r--',lw=0.5)
            ax[i,j].plot([8,8],[0,6],'r--',lw=0.5)
    
    fig.suptitle('mean spectrum(50 realizations)'+title+'\n[%.1f, %.1f]'%(mua_loca[0],mua_loca[1]))
    savetitle = title.replace('\n','')
    fftfile = savetitle + '_fft_param%d'%(para) + '.png'
    fig.savefig(save_dir + fftfile)
    plt.close(fig)
    '''MSD'''
    
    fig, ax1 = plt.subplots(1,1)
    ax1.loglog(data_anly.MSD.jump_interval, data_anly.MSD.MSD)

    # ax2 = ax1.twinx()
    # ax2.set_ylim([0,2.5])
    # err_up = data.a1.ge.MSD.stableDist_param[:,2,0] - data.a1.ge.MSD.stableDist_param[:,0,0]
    # err_down = data.a1.ge.MSD.stableDist_param[:,0,0] - data.a1.ge.MSD.stableDist_param[:,1,0]
    
    # googfit = np.abs(err_up - err_down)/(err_up + err_down) < 0.01
    # ax2.errorbar(data.a1.ge.MSD.jump_interval[googfit], \
    #              data.a1.ge.MSD.stableDist_param[googfit,0,0], \
    #              yerr=data.a1.ge.MSD.stableDist_param[googfit,2,0] - data.a1.ge.MSD.stableDist_param[googfit,0,0], fmt='ro')
    # #ax2.errorbar(data.a1.ge.MSD.jump_interval, \
    # #             data.a1.ge.MSD.stableDist_param[:], yerr=0)
    
    #alpha = data.a1.ge.MSD.stableDist_param[2,0,0]
    #titleMSD = title + '_alpha%.2f'%alpha
    ax1.set_title('mean MSD(50 realizations)'+title+'\n[%.1f, %.1f]'%(mua_loca[0],mua_loca[1]))
    
    savetitle = title.replace('\n','')
    MSDfile = savetitle+'_MSD_param%d'%(para) + '.png'
    # if saveMSDfile:
    fig.savefig(save_dir + MSDfile)
    plt.close(fig)
    
    '''cfc'''
    # phaseBand = data_anly.cfc.phaseBand #np.arange(1,14.1,0.5)
    # ampBand = data_anly.cfc.ampBand #np.arange(10,101,5) 
    
    # fig, [ax1,ax2] = plt.subplots(2,1, figsize=[7,9])
    # #x_range = np.arange(phaseBand[0]-phaseBandWid/2, phaseBand[-1]+phaseBandWid/2+1)
    # #y_range = np.arange(ampBand[0]-ampBandWid/2, ampBand[-1]+ampBandWid/2+1)
    
    # #im = ax1.imshow(np.flip(MI_raw.T, 0), aspect='auto', extent=[phaseBand[0]-phaseBandWid/2, phaseBand[-1]+phaseBandWid/2, ampBand[0]-ampBandWid/2, ampBand[-1]+ampBandWid/2])
    # imc = ax1.contourf(phaseBand, ampBand, MI_raw.T, 15)#, aspect='auto')
    # imcc = ax1.contour(phaseBand, ampBand, MI_raw.T, 15, colors='k', linewidths=0.3)#, aspect='auto')
    
    # imc2 = ax2.contourf(phaseBand, ampBand, MI_surr.T, 15)#, aspect='auto')
    # imcc2 = ax2.contour(phaseBand, ampBand, MI_surr.T, 15, colors='k', linewidths=0.3)#, aspect='auto')
    
    # #imc2 = ax1.contour(phaseBand, ampBand, MI_raw.T, 15)#, aspect='auto')
    
    # #imc = ax1.contourf(MI_raw.T, 12, extent=[phaseBand[0]-phaseBandWid/2, phaseBand[-1]+phaseBandWid/2, ampBand[0]-ampBandWid/2, ampBand[-1]+ampBandWid/2])#, aspect='auto')
    # #imc = ax1.contourf(MI_raw.T, 12, origin='lower', extent=[phaseBand[0]-phaseBandWid/2, phaseBand[-1]+phaseBandWid/2, ampBand[0]-ampBandWid/2, ampBand[-1]+ampBandWid/2])#extent=[phaseBand[0]-phaseBandWid/2, phaseBand[-1]+phaseBandWid/2, ampBand[0]-ampBandWid/2, ampBand[-1]+ampBandWid/2])#, aspect='auto')
    
    # #imi = ax2.imshow(np.flip(MI_raw_mat.T, 0), aspect='auto', extent=[phaseBand[0]-phaseBandWid/2, phaseBand[-1]+phaseBandWid/2, ampBand[0]-ampBandWid/2, ampBand[-1]+ampBandWid/2])
    # plt.colorbar(imc, ax=ax1)
    # plt.colorbar(imc2, ax=ax2)
    # ax1.set_xlabel('phase frequency (Hz)')
    # ax1.set_ylabel('Amplitude frequency (Hz)')
    # ax1.set_title('raw')
    # ax2.set_xlabel('phase frequency (Hz)')
    # ax2.set_ylabel('Amplitude frequency (Hz)')
    # ax2.set_title('minus-surr')
    # #plt.suptitle('ee1.20_ei1.27_ie1.2137_ii1.08_dsi4.44_dse5.00\n/headnode1/shni2598/brian2/NeuroNet_brian/\nmodel_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp')
    
    # #titlecfc = title + '_cfc'
    # fig.suptitle('mean spectrum(50 realizations)'+title+'\n[%.1f, %.1f]'%(mua_loca[0],mua_loca[1]))
    # savetitle = title.replace('\n','')
    # cfcfile = savetitle + '_cfc_param%d'%(para) + '.png'
    # fig.savefig(save_dir + cfcfile)
    # plt.close(fig)
        

        #%%

        

# 'data_anly_electrode_loc%d_%d.file'%(mua_loc_ind,loop_num))