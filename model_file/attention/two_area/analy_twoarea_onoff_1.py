#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 23:31:05 2021

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
import find_change_pts
import connection as cn
import pickle
import sys
import os
import matplotlib.pyplot as plt
import shutil
#%%
data_dir = 'raw_data/'
analy_type = 'fbrgbig4'
datapath = data_dir
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/twoarea_hz/on_off/'+data_dir
sys_argv = int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num
#loop_num = 0

good_dir = 'good/'
goodsize_dir = 'good_size/'

savefile_name = 'data_anly_thr' #'data_anly' data_anly_temp
save_apd = '_thr'

onff_method = 'threshold'

thre_spon = 4
thre_stim = [12, 15, 30]

fftplot = 1; getfano = 1
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

n_StimAmp = data.a1.param.stim1.n_StimAmp
n_perStimAmp = data.a1.param.stim1.n_perStimAmp
stim_amp = 200*2**np.arange(n_StimAmp)

if analy_type == 'fbrgbig4': # fbrg: feedback range
    title = '1irie%.2f_1e2e%.1f_pk2e1e%.2f'%(data.param.ie_r_i1, data.inter.param.w_e1_e2_mean/5, \
                                               data.inter.param.peak_p_e2_e1)
#%%
mua_loca = [0, 0]
mua_range = 5 
mua_neuron = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca, mua_range, data.a1.param.width)
#%%
simu_time_tot = data.param.simutime#29000

#N_stim = data.a1.param.stim.stim_amp_scale.shape[0]
#%%
'''onoff'''
data_anly.onoff = mydata.mydata()

findonoff = find_change_pts.MUA_findchangepts()
#%%
'''spon onoff'''
start = 5000; end = 20000
analy_dura = np.array([[start,end]])
data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])

findonoff.MinThreshold = None # None # 1000
findonoff.MaxNumChanges = int(round(((analy_dura[0,1]-analy_dura[0,0]))/1000*8*1.2)) #15
findonoff.smooth_window = None #52



#%%
# analy_dura = data.a1.param.stim1.stim_on[20:40].copy()
# analy_dura[:,1] -= (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
#analy_dura[:,0] += (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
if onff_method == 'threshold':
    spon_onoff = findonoff.analyze(data.a1.ge.spk_matrix[mua_neuron], analy_dura=analy_dura, method = 'threshold', threshold=thre_spon)
else:
    spon_onoff = findonoff.analyze(data.a1.ge.spk_matrix[mua_neuron], analy_dura=analy_dura, method = 'comparing')

data_anly.onoff.spon = spon_onoff

#%%
stim_num = 0
end_plot_time = 3000
n_ind, t = data.a1.ge.spk_matrix[mua_neuron,analy_dura[stim_num,0]*10:(analy_dura[stim_num,0]+end_plot_time)*10].nonzero()


#%%
fig,ax = plt.subplots(2,1, figsize=[15,6])
# ax[0].plot(t/10, n_ind, '|')
# ax[0].plot(show_stim_on)  
# ax[1].plot(np.arange(mua_.shape[0]),mua_)
#ax[0].plot(np.array(smt_mua_mat[stim_on_new[20,0]:stim_on_new[20,1]]))
ax[0].plot(spon_onoff.mua[stim_num][:end_plot_time])
for i in range(spon_onoff.cpts[stim_num].shape[0]):
    if spon_onoff.cpts[stim_num][i] >= end_plot_time : break
    ax[0].plot([spon_onoff.cpts[stim_num][i],spon_onoff.cpts[stim_num][i]],[0,spon_onoff.mua[stim_num][:end_plot_time].max()], c=clr[1])
ax[1].plot(spon_onoff.onoff_bool[stim_num][:end_plot_time]*80)
ax[1].plot(t/10, n_ind, '|')
fig.suptitle(title + 'on-off; plot')

savetitle = title.replace('\n','')
onofffile = savetitle+'_spon_t'+save_apd+'_%d'%(loop_num)+'.png'
fig.savefig(onofffile)
plt.close()
#%%
fig, ax = plt.subplots(1,4, figsize=[15,6])
hr = ax[0].hist(np.concatenate(spon_onoff.on_t),bins=20, density=True)
mu = np.concatenate(spon_onoff.on_t).mean()
ax[0].plot([mu,mu],[0,hr[0].max()*1.2])
ax[0].set_title('on period; spon; mean:%.2f'%mu)
hr = ax[1].hist(np.concatenate(spon_onoff.off_t),bins=20, density=True)
mu = np.concatenate(spon_onoff.off_t).mean()
ax[1].plot([mu,mu],[0,hr[0].max()*1.2])
ax[1].set_title('off period; spon; mean:%.2f'%mu)
hr = ax[2].hist(np.concatenate(spon_onoff.on_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000),bins=20, density=True)
mu = np.concatenate(spon_onoff.on_amp).mean()
ax[2].plot([mu,mu],[0,hr[0].max()*1.2])
ax[2].set_title('on rate; spon; mean:%.2f'%mu)
hr = ax[3].hist(np.concatenate(spon_onoff.off_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000),bins=20, density=True)
mu = np.concatenate(spon_onoff.off_amp).mean()
ax[3].plot([mu,mu],[0,hr[0].max()*1.2])
ax[3].set_title('off rate; spon; mean:%.2f'%mu)
ax[0].set_yscale('log')
ax[1].set_yscale('log')
fig.suptitle(title + 'on-off dist')

savetitle = title.replace('\n','')
onofffile = savetitle+'_spon_dis'+save_apd+'_%d'%(loop_num)+'.png'
fig.savefig(onofffile)
plt.close()

#%%
'''stim;'''
data_anly.onoff.stim_noatt = []
data_anly.onoff.stim_att = []

for n in range(n_StimAmp):
    
    analy_dura = data.a1.param.stim1.stim_on[n*n_perStimAmp:(n+1)*n_perStimAmp].copy()
    analy_dura[:,1] -= (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
    #analy_dura[:,0] += (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
    findonoff.MinThreshold = None # None # 1000
    findonoff.MaxNumChanges = int(((analy_dura[0,1]-analy_dura[0,0]))/1000*8*1.2) #15
    findonoff.smooth_window = None #52
    
    # start = 5000; end = 20000
    # analy_dura = np.array([[start,end]])
    
    #analy_dura = data.a1.param.stim1.stim_on[20:40].copy()
    #analy_dura[:,1] -= (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
    #analy_dura[:,0] += (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
    #stim_onoff = findonoff.analyze(data.a1.ge.spk_matrix[mua_neuron], analy_dura=analy_dura)

    if onff_method == 'threshold':
        stim_onoff = findonoff.analyze(data.a1.ge.spk_matrix[mua_neuron], analy_dura=analy_dura, method = 'threshold', threshold=thre_stim[n])
    else:
        stim_onoff = findonoff.analyze(data.a1.ge.spk_matrix[mua_neuron], analy_dura=analy_dura, method = 'comparing')

    
    data_anly.onoff.stim_noatt.append(stim_onoff)
    
    stim_num = 0
    n_ind, t = data.a1.ge.spk_matrix[mua_neuron,analy_dura[stim_num,0]*10:analy_dura[stim_num,1]*10].nonzero()
    
    '''on-off plot'''
    fig, ax = plt.subplots(4,1, figsize=[15,12])
    '''no att'''
    ax[0].plot(stim_onoff.mua[stim_num]/mua_neuron.shape[0]/(findonoff.mua_win/1000))
    for i in range(stim_onoff.cpts[stim_num].shape[0]):
        ax[0].plot([stim_onoff.cpts[stim_num][i],stim_onoff.cpts[stim_num][i]],[0,80], c=clr[1])
    ax[0].set_title('stim; no att')
    ax[1].plot(stim_onoff.onoff_bool[stim_num]*80)
    ax[1].plot(t/10, n_ind, '|')
    ax[0].xaxis.set_visible(False)
    ax[1].xaxis.set_visible(False) 
   
    # fig,ax = plt.subplots(2,1)
    # # ax[0].plot(t/10, n_ind, '|')
    # # ax[0].plot(show_stim_on)  
    # # ax[1].plot(np.arange(mua_.shape[0]),mua_)
    # #ax[0].plot(np.array(smt_mua_mat[stim_on_new[20,0]:stim_on_new[20,1]]))
    # ax[0].plot(stim_onoff.mua[stim_num])
    # for i in range(stim_onoff.cpts[stim_num].shape[0]):
    #     ax[0].plot([stim_onoff.cpts[stim_num][i],stim_onoff.cpts[stim_num][i]],[0,80], c=clr[1])
    # ax[1].plot(stim_onoff.onoff_bool[stim_num]*80)
    # ax[1].plot(t/10, n_ind, '|')
    
    # fig,ax = plt.subplots(2,2)
    # ax[0,0].hist(np.concatenate(stim_onoff.on_t),bins=20, density=True)
    # ax[0,1].hist(np.concatenate(stim_onoff.off_t),bins=20, density=True)
    # ax[1,0].hist(np.concatenate(stim_onoff.on_amp),bins=20, density=True)
    # ax[1,1].hist(np.concatenate(stim_onoff.off_amp),bins=20, density=True)
    # ax[0,0].set_yscale('log')
    # ax[0,1].set_yscale('log')
    
    '''att'''
    analy_dura = data.a1.param.stim1.stim_on[n*n_perStimAmp:(n+1)*n_perStimAmp].copy()
    #analy_dura[:,1] -= (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
    analy_dura[:,0] += (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
    findonoff.MinThreshold = None # None # 1000
    findonoff.MaxNumChanges = int(((analy_dura[0,1]-analy_dura[0,0]))/1000*8*1.2) #15
    findonoff.smooth_window = None #52
    
    # start = 5000; end = 20000
    # analy_dura = np.array([[start,end]])
    
    #analy_dura = data.a1.param.stim1.stim_on[20:40].copy()
    #analy_dura[:,1] -= (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
    #analy_dura[:,0] += (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
    #stim_onoff_att = findonoff.analyze(data.a1.ge.spk_matrix[mua_neuron], analy_dura=analy_dura)
    if onff_method == 'threshold':
        stim_onoff_att = findonoff.analyze(data.a1.ge.spk_matrix[mua_neuron], analy_dura=analy_dura, method = 'threshold', threshold=thre_stim[n])
    else:
        stim_onoff_att = findonoff.analyze(data.a1.ge.spk_matrix[mua_neuron], analy_dura=analy_dura, method = 'comparing')    
    
    data_anly.onoff.stim_att.append(stim_onoff_att)
    
    stim_num = 0
    n_ind, t = data.a1.ge.spk_matrix[mua_neuron,analy_dura[stim_num,0]*10:analy_dura[stim_num,1]*10].nonzero()
     
    '''att'''
    ax[2].plot(stim_onoff_att.mua[stim_num]/mua_neuron.shape[0]/(findonoff.mua_win/1000))
    for i in range(stim_onoff_att.cpts[stim_num].shape[0]):
        ax[2].plot([stim_onoff_att.cpts[stim_num][i],stim_onoff_att.cpts[stim_num][i]],[0,80], c=clr[1])
    ax[2].set_title('stim; att')
    ax[3].plot(stim_onoff_att.onoff_bool[stim_num]*80)
    ax[3].plot(t/10, n_ind, '|')
    ax[2].xaxis.set_visible(False)      
    fig.suptitle(title + 'stim: %.1f hz'%stim_amp[n])

    savetitle = title.replace('\n','')
    onofffile = savetitle+'_stim%d_t'%n+save_apd+'_%d'%(loop_num)+'.png'
    fig.savefig(onofffile)
    plt.close()      
    
    
    # fig,ax = plt.subplots(2,1)
    # # ax[0].plot(t/10, n_ind, '|')
    # # ax[0].plot(show_stim_on)  
    # # ax[1].plot(np.arange(mua_.shape[0]),mua_)
    # #ax[0].plot(np.array(smt_mua_mat[stim_on_new[20,0]:stim_on_new[20,1]]))
    # ax[0].plot(stim_onoff_att.mua[stim_num])
    # for i in range(stim_onoff_att.cpts[stim_num].shape[0]):
    #     ax[0].plot([stim_onoff_att.cpts[stim_num][i],stim_onoff_att.cpts[stim_num][i]],[0,80], c=clr[1])
    # ax[1].plot(stim_onoff_att.onoff_bool[stim_num]*80)
    # ax[1].plot(t/10, n_ind, '|')
    fig,ax = plt.subplots(2,4, figsize=[15,6])
    hr = ax[0,0].hist(np.concatenate(stim_onoff.on_t),bins=20, density=True)
    mu = np.concatenate(stim_onoff.on_t).mean()
    ax[0,0].plot([mu,mu],[0,hr[0].max()*1.2])
    ax[0,0].set_title('on period; no att; mean:%.2f'%mu)
    hr = ax[0,1].hist(np.concatenate(stim_onoff.off_t),bins=20, density=True)
    mu = np.concatenate(stim_onoff.off_t).mean()
    ax[0,1].plot([mu,mu],[0,hr[0].max()*1.2])
    ax[0,1].set_title('off period; no att; mean:%.2f'%mu)
    hr = ax[0,2].hist(np.concatenate(stim_onoff.on_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000),bins=20, density=True)
    mu = (np.concatenate(stim_onoff.on_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000)).mean()
    ax[0,2].plot([mu,mu],[0,hr[0].max()*1.2])
    ax[0,2].set_title('on rate; no att; mean:%.2f'%mu)
    hr = ax[0,3].hist(np.concatenate(stim_onoff.off_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000),bins=20, density=True)
    mu = (np.concatenate(stim_onoff.off_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000)).mean()
    ax[0,3].plot([mu,mu],[0,hr[0].max()*1.2])
    ax[0,3].set_title('off rate; no att; mean:%.2f'%mu)
    ax[0,0].set_yscale('log')
    ax[0,1].set_yscale('log')
    
    #fig,ax = plt.subplots(2,4)
    hr = ax[1,0].hist(np.concatenate(stim_onoff_att.on_t),bins=20, density=True)
    mu = np.concatenate(stim_onoff_att.on_t).mean()
    ax[1,0].plot([mu,mu],[0,hr[0].max()*1.2])
    ax[1,0].set_title('on period; att; mean:%.2f'%mu)
    hr = ax[1,1].hist(np.concatenate(stim_onoff_att.off_t),bins=20, density=True)
    mu = np.concatenate(stim_onoff_att.off_t).mean()
    ax[1,1].plot([mu,mu],[0,hr[0].max()*1.2])
    ax[1,1].set_title('off period; att; mean:%.2f'%mu)
    hr = ax[1,2].hist(np.concatenate(stim_onoff_att.on_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000),bins=20, density=True)
    mu = (np.concatenate(stim_onoff_att.on_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000)).mean()
    ax[1,2].plot([mu,mu],[0,hr[0].max()*1.2])
    ax[1,2].set_title('on rate; att; mean:%.2f'%mu)
    hr = ax[1,3].hist(np.concatenate(stim_onoff_att.off_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000),bins=20, density=True)
    mu = (np.concatenate(stim_onoff_att.off_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000)).mean()
    ax[1,3].plot([mu,mu],[0,hr[0].max()*1.2])
    ax[1,3].set_title('off rate; att; mean:%.2f'%mu)
    ax[1,0].set_yscale('log')
    ax[1,1].set_yscale('log')
    fig.suptitle(title + 'stim: %.1f hz'%stim_amp[n])

    savetitle = title.replace('\n','')
    onofffile = savetitle+'_stim%d_dis'%n+save_apd+'%d'%(loop_num)+'.png'
    fig.savefig(onofffile)
    plt.close()  
    # ax[1,0].hist(np.concatenate(stim_onoff_att.on_t),bins=20, density=True)
    # ax[1,0].set_title('on period; att')
    # ax[1,1].hist(np.concatenate(stim_onoff_att.off_t),bins=20, density=True)
    # ax[1,1].set_title('off period; att')
    # ax[1,2].hist(np.concatenate(stim_onoff_att.on_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000),bins=20, density=True)
    # ax[1,2].set_title('on rate; att')
    # ax[1,3].hist(np.concatenate(stim_onoff_att.off_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000),bins=20, density=True)
    # ax[1,3].set_title('off rate; att')
    # ax[1,0].set_yscale('log')
    # ax[1,1].set_yscale('log')
#%%
data_anly.save(data_anly.class2dict(), datapath+savefile_name+'%d.file'%loop_num)
#%%

"""
'''spon onoff'''
start = 5000; end = 20000
analy_dura = np.array([[start,end]])
#data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])

findonoff.MinThreshold = None # None # 1000
findonoff.MaxNumChanges = int(round(((analy_dura[0,1]-analy_dura[0,0]))/1000*8*1.2)) #15
findonoff.smooth_window = None #52


threshold = 4

# analy_dura = data.a1.param.stim1.stim_on[20:40].copy()
# analy_dura[:,1] -= (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
#analy_dura[:,0] += (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
spon_onoff = findonoff.analyze(data.a1.ge.spk_matrix[mua_neuron], analy_dura=analy_dura,  method = "threshold", threshold=threshold)
#data_anly.onoff.spon = spon_onoff


stim_num = 0
str_plot_time = 3000; end_plot_time = 6000
n_ind, t = data.a1.ge.spk_matrix[mua_neuron,(analy_dura[stim_num,0]+str_plot_time)*10:(analy_dura[stim_num,0]+end_plot_time)*10].nonzero()



fig,ax = plt.subplots(2,1)
# ax[0].plot(t/10, n_ind, '|')
# ax[0].plot(show_stim_on)  
# ax[1].plot(np.arange(mua_.shape[0]),mua_)
#ax[0].plot(np.array(smt_mua_mat[stim_on_new[20,0]:stim_on_new[20,1]]))
ax[0].plot(spon_onoff.mua[stim_num][str_plot_time:end_plot_time])
for i in range(spon_onoff.cpts[stim_num].shape[0]):
    if spon_onoff.cpts[stim_num][i] < str_plot_time : continue
    if spon_onoff.cpts[stim_num][i] >= end_plot_time : break
    ax[0].plot([spon_onoff.cpts[stim_num][i]-str_plot_time,spon_onoff.cpts[stim_num][i]-str_plot_time],[0,spon_onoff.mua[stim_num][:end_plot_time].max()], c=clr[1])
ax[1].plot(spon_onoff.onoff_bool[stim_num][str_plot_time:end_plot_time]*80)
ax[1].plot(t/10, n_ind, '|')
# fig.suptitle(title + 'on-off; plot')

# savetitle = title.replace('\n','')
# onofffile = savetitle+'_spon_t_%d'%(loop_num)+'.png'
# #fig.savefig(onofffile)
# #plt.close()







#%%
[12, 15, 30]
#%%
loop_num=0
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/twoarea_hz/on_off/'+data_dir

data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)

data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])
#%%
# start = (data.a1.param.stim1.stim_on[20,0]-2000)*10
# end = (data.a1.param.stim1.stim_on[25,1]+300)*10

# stim_on_new = data.a1.param.stim1.stim_on.copy()
# stim_on_new -= int(round(start/10))

#%%
st_stim = 0; ed_stim = 5
analy_dura = data.a1.param.stim1.stim_on[st_stim:ed_stim].copy()
analy_dura[:,1] -= (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
#analy_dura[:,0] += (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
findonoff.MinThreshold = None # None # 1000
findonoff.MaxNumChanges = int(round((analy_dura[0,1]-analy_dura[0,0])/1000*8*1.2)) #15
findonoff.smooth_window = None #52

threshold = 12
# start = 5000; end = 20000
# analy_dura = np.array([[start,end]])

#analy_dura = data.a1.param.stim1.stim_on[20:40].copy()
#analy_dura[:,1] -= (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
#analy_dura[:,0] += (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
# threshold comparing

stim_onoff = findonoff.analyze(data.a1.ge.spk_matrix[mua_neuron], analy_dura=analy_dura,  method = "threshold", threshold=threshold)

#data_anly.onoff.stim_noatt.append(stim_onoff)

stim_num = 4
n_ind, t = data.a1.ge.spk_matrix[mua_neuron,analy_dura[stim_num,0]*10:analy_dura[stim_num,1]*10].nonzero()

'''on-off plot'''
fig, ax = plt.subplots(4,1, figsize=[15,12])
'''no att'''
ax[0].plot(stim_onoff.mua[stim_num]/mua_neuron.shape[0]/(findonoff.mua_win/1000))
#ax[0].plot(stim_onoff.mua[stim_num])

for i in range(stim_onoff.cpts[stim_num].shape[0]):
    ax[0].plot([stim_onoff.cpts[stim_num][i],stim_onoff.cpts[stim_num][i]],[0,80], c=clr[1])
ax[0].set_title('stim; no att')
ax[1].plot(stim_onoff.onoff_bool[stim_num]*80)
ax[1].plot(t/10, n_ind, '|')
ax[0].xaxis.set_visible(False)
ax[1].xaxis.set_visible(False) 



# start = (data.a1.param.stim1.stim_on[20,0]-2000)*10
# end = (data.a1.param.stim1.stim_on[25,1]+300)*10

# stim_on_new = data.a1.param.stim1.stim_on.copy()
# stim_on_new -= int(round(start/10))


analy_dura = data.a1.param.stim1.stim_on[st_stim:ed_stim].copy()
#analy_dura[:,1] -= (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
analy_dura[:,0] += (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
findonoff.MinThreshold = None # None # 1000
findonoff.MaxNumChanges = int(round(((analy_dura[0,1]-analy_dura[0,0]))/1000*8*1.2)) #15
findonoff.smooth_window = None #52

# start = 5000; end = 20000
# analy_dura = np.array([[start,end]])

#analy_dura = data.a1.param.stim1.stim_on[20:40].copy()
#analy_dura[:,1] -= (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
#analy_dura[:,0] += (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
# threshold comparing

stim_onoff_att = findonoff.analyze(data.a1.ge.spk_matrix[mua_neuron], analy_dura=analy_dura,  method = "threshold", threshold=threshold)

#data_anly.onoff.stim_noatt.append(stim_onoff_att)

#stim_num = 4
n_ind, t = data.a1.ge.spk_matrix[mua_neuron,analy_dura[stim_num,0]*10:analy_dura[stim_num,1]*10].nonzero()

'''on-off plot'''
#fig, ax = plt.subplots(4,1, figsize=[15,12])
'''att'''
#ax[0].plot(stim_onoff.mua[stim_num])


ax[2].plot(stim_onoff_att.mua[stim_num]/mua_neuron.shape[0]/(findonoff.mua_win/1000))
#ax[2].plot(stim_onoff_att.mua[stim_num])

for i in range(stim_onoff_att.cpts[stim_num].shape[0]):
    ax[2].plot([stim_onoff_att.cpts[stim_num][i],stim_onoff_att.cpts[stim_num][i]],[0,80], c=clr[1])
ax[2].set_title('stim; att')
ax[3].plot(stim_onoff_att.onoff_bool[stim_num]*80)
ax[3].plot(t/10, n_ind, '|')
ax[2].xaxis.set_visible(False)  








#%%
data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])
n_ind, t = data.a1.ge.spk_matrix[mua_neuron,start:end].nonzero()

mua = data.a1.ge.spk_matrix[mua_neuron,start:end].sum(0).A[0]

mua = np.convolve(mua,np.ones(100))/0.01/mua_neuron.shape[0]
#%%
show_stim_on = np.zeros(int(round((end-start)/10)))
for i in range(20,26):
    show_stim_on[stim_on_new[i,0]:stim_on_new[i,1]] = 60

#%%

fig,ax = plt.subplots(2,1)
ax[0].plot(t/10, n_ind, '|')
ax[0].plot(show_stim_on)
ax[1].plot(np.arange(mua.shape[0])/10,mua)
ax[1].plot(show_stim_on)
#data.a1.ge.
#%%
mua_ = mua[0:-1:10]
#%%
fig,ax = plt.subplots(2,1)
# ax[0].plot(t/10, n_ind, '|')
# ax[0].plot(show_stim_on)
ax[1].plot(np.arange(mua_.shape[0]),mua_)

#%%
eng = matlab.engine.start_matlab('-nodisplay')
#%%
mua_mat = matlab.double(list(mua_), size=[mua_.size,1])
smt_mua_mat = eng.smoothdata(mua_mat,'rlowess')
smt_mua_ = np.array(smt_mua_mat)
#%%
fig,ax = plt.subplots(2,1)
ax[0].plot(t/10, n_ind, '|')
ax[0].plot(show_stim_on)
ax[1].plot(np.arange(mua_.shape[0]),mua_)
ax[1].plot(np.arange(mua_.shape[0]),smt_mua_)
#%%
MaxNumChanges = 15; #(R.step_tot/10 - 4e3)/125+10;% fft shows it is 4 Hz and the visualiztion proves it
MinDistance = 5;
MinThreshold = 4000
#cpts = eng.findchangepts(smt_mua_mat[stim_on_new[20,0]+1500:stim_on_new[20,1]],'statistic','mean','MaxNumChanges',MaxNumChanges,'MinDistance',MinDistance);
cpts = eng.findchangepts(smt_mua_mat[stim_on_new[20,0]:stim_on_new[20,0]+1500],'statistic','mean','MinThreshold',MinThreshold,'MinDistance',MinDistance);

#A = [1,A]; % make the first one manully since the function misses it.
cpts_ary = np.array(cpts)
cpts_ary = np.concatenate(([0],cpts_ary.reshape(-1)))
cpts_ary += 0


fig,ax = plt.subplots(2,1)
# ax[0].plot(t/10, n_ind, '|')
# ax[0].plot(show_stim_on)  
# ax[1].plot(np.arange(mua_.shape[0]),mua_)
ax[0].plot(np.array(smt_mua_mat[stim_on_new[20,0]:stim_on_new[20,1]]))
for i in range(cpts_ary.shape[0]):
    ax[0].plot([cpts_ary[i],cpts_ary[i]],[0,80], c=clr[1])
    #ax[0].plot([cpts_ary[i]],[80], c=clr[1])
#%%
cpts__, findonoff.mat_eng = find_change_pts.find_change_pts(mua_[stim_on_new[20,0]:stim_on_new[20,0]+1500].reshape(-1), \
                    smooth_method = 'rlowess', \
                    smooth_window = None, \
                    chg_pts_statistic = 'mean', MinThreshold = MinThreshold, \
                    MaxNumChanges = None, MinDistance = 5, eng=findonoff.mat_eng)



#%%
a = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]);    
a = matlab.double(list(a), size=[a.size,1])
cpts_a = eng.findchangepts(a,'statistic','mean');
cpts_ary_a = np.array(cpts_a)
#%%
import find_change_pts
#%%
'''class'''
findonoff = find_change_pts.MUA_findchangepts()
#%%
findonoff.MinThreshold = MinThreshold # None #
findonoff.MaxNumChanges = None #15
#%%
analy_dura = data.a1.param.stim1.stim_on[20:22].copy()
analy_dura[:,1] -= (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
on_t, off_t, cpts_cls, onoff_bool, mua_tmp, on_amp, off_amp = findonoff.analyze(data.a1.ge.spk_matrix[mua_neuron], analy_dura=analy_dura)





#%%
fig,ax = plt.subplots(2,1)
# ax[0].plot(t/10, n_ind, '|')
# ax[0].plot(show_stim_on)  
# ax[1].plot(np.arange(mua_.shape[0]),mua_)
#ax[0].plot(np.array(smt_mua_mat[stim_on_new[20,0]:stim_on_new[20,1]]))
ax[0].plot(mua_tmp[0])
for i in range(cpts_cls[0].shape[0]):
    ax[0].plot([cpts_cls[0][i],cpts_cls[0][i]],[0,80], c=clr[1])
ax[1].plot(onoff_bool[0])
#%%
fig,ax = plt.subplots(2,1)
# ax[0].plot(t/10, n_ind, '|')
# ax[0].plot(show_stim_on)  
# ax[1].plot(np.arange(mua_.shape[0]),mua_)
ax[0].plot(mua_tmp[0])    
#%%
cpts, eng = find_change_pts.find_change_pts(mua_tmp[0], smooth_method = 'rlowess', smooth_window = None, \
                    chg_pts_statistic = 'mean', MinThreshold = None, \
                    MaxNumChanges = 15, MinDistance = 5, eng=findonoff.mat_eng)
    
#%%
MaxNumChanges = 15; #(R.step_tot/10 - 4e3)/125+10;% fft shows it is 4 Hz and the visualiztion proves it
MinDistance = 5;
MinThreshold = 4000

mua_mat = matlab.double(list(mua_tmp[0]), size=[mua_tmp[0].size,1])
smt_mua_mat = eng.smoothdata(mua_mat,'rlowess')
smt_mua_ = np.array(smt_mua_mat)
#cpts = eng.findchangepts(smt_mua_mat[stim_on_new[20,0]+1500:stim_on_new[20,1]],'statistic','mean','MaxNumChanges',MaxNumChanges,'MinDistance',MinDistance);
cpts = eng.findchangepts(smt_mua_mat[stim_on_new[20,0]:stim_on_new[20,0]+1500],'statistic','mean','MinThreshold',MinThreshold,'MinDistance',MinDistance);

#A = [1,A]; % make the first one manully since the function misses it.
cpts_ary = np.array(cpts)
cpts_ary = np.concatenate(([0],cpts_ary.reshape(-1)))
cpts_ary += 0


fig,ax = plt.subplots(2,1)
# ax[0].plot(t/10, n_ind, '|')
# ax[0].plot(show_stim_on)  
# ax[1].plot(np.arange(mua_.shape[0]),mua_)
ax[0].plot(np.array(smt_mua_mat[stim_on_new[20,0]:stim_on_new[20,1]]))
for i in range(cpts_ary.shape[0]):
    ax[0].plot([cpts_ary[i],cpts_ary[i]],[0,80], c=clr[1])    
#%%





#%%
simu_time_tot = data.param.simutime#29000

#N_stim = data.a1.param.stim.stim_amp_scale.shape[0]
#%%
start = (data.a1.param.stim1.stim_on[20,0])*10
end = (data.a1.param.stim1.stim_on[20,0] + 1500)*10

stim_on_new = data.a1.param.stim1.stim_on.copy()
stim_on_new -= int(round(start/10))

#%%
data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])
n_ind, t = data.a1.ge.spk_matrix[mua_neuron,start:end].nonzero()

mua = data.a1.ge.spk_matrix[mua_neuron,start:end].sum(0).A[0]

mua = np.convolve(mua,np.ones(100),mode='valid')#/0.01/mua_neuron.shape[0]
#%%
show_stim_on = np.zeros(int(round((end-start)/10)))
for i in range(20,26):
    show_stim_on[stim_on_new[i,0]:stim_on_new[i,1]] = 60
    
#%%
mua_ = mua[0::10]
#%%
mua_ = mua_tmp[0]

#%%
fig,ax = plt.subplots(2,1)
# ax[0].plot(t/10, n_ind, '|')
# ax[0].plot(show_stim_on)
ax[1].plot(np.arange(mua_.shape[0]),mua_)

#%%
eng = matlab.engine.start_matlab('-nodisplay')
#%%
mua_mat = matlab.double(list(mua_), size=[mua_.size,1])
smt_mua_mat = findonoff.mat_eng.smoothdata(mua_mat,'rlowess')
smt_mua_ = np.array(smt_mua_mat)

#%%
MaxNumChanges = 15; #(R.step_tot/10 - 4e3)/125+10;% fft shows it is 4 Hz and the visualiztion proves it
MinDistance = 5;
MinThreshold = 4000
#cpts = eng.findchangepts(smt_mua_mat[stim_on_new[20,0]+1500:stim_on_new[20,1]],'statistic','mean','MaxNumChanges',MaxNumChanges,'MinDistance',MinDistance);
cpts = eng.findchangepts(smt_mua_mat,'statistic','mean','MinThreshold',MinThreshold,'MinDistance',MinDistance);

#A = [1,A]; % make the first one manully since the function misses it.
cpts_ary = np.array(cpts).reshape(-1).astype(int)
cpts_ary = np.concatenate(([0],cpts_ary.reshape(-1)))
cpts_ary += 0


fig,ax = plt.subplots(2,1)
# ax[0].plot(t/10, n_ind, '|')
# ax[0].plot(show_stim_on)  
# ax[1].plot(np.arange(mua_.shape[0]),mua_)
ax[0].plot(np.array(smt_mua_mat))
for i in range(cpts_ary.shape[0]):
    ax[0].plot([cpts_ary[i],cpts_ary[i]],[0,80], c=clr[1])
    #ax[0].plot([cpts_ary[i]],[80], c=clr[1])

#%%
plt.figure()
plt.plot(mua_tmp[0])
plt.plot(mua_)
#%%

#%%
'''class'''
findonoff = find_change_pts.MUA_findchangepts()
#%%
findonoff.MinThreshold = MinThreshold # None #
findonoff.MaxNumChanges = None #15
findonoff.smooth_window = 52
#%%
analy_dura = data.a1.param.stim1.stim_on[20:40].copy()
analy_dura[:,1] -= (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
#analy_dura[:,0] += (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
on_t, off_t, cpts_cls, onoff_bool, mua_tmp, on_amp, off_amp = findonoff.analyze(data.a1.ge.spk_matrix[mua_neuron], analy_dura=analy_dura)

#%%
stim_num = 1
n_ind, t = data.a1.ge.spk_matrix[mua_neuron,analy_dura[stim_num,0]*10:analy_dura[stim_num,1]*10].nonzero()

#%%
fig,ax = plt.subplots(2,1)
# ax[0].plot(t/10, n_ind, '|')
# ax[0].plot(show_stim_on)  
# ax[1].plot(np.arange(mua_.shape[0]),mua_)
#ax[0].plot(np.array(smt_mua_mat[stim_on_new[20,0]:stim_on_new[20,1]]))
ax[0].plot(mua_tmp[stim_num])
for i in range(cpts_cls[stim_num].shape[0]):
    ax[0].plot([cpts_cls[stim_num][i],cpts_cls[stim_num][i]],[0,80], c=clr[1])
ax[1].plot(onoff_bool[stim_num]*80)
ax[1].plot(t/10, n_ind, '|')
#%%
fig,ax = plt.subplots(2,1)
ax[0].hist(np.concatenate(on_t),bins=20)
ax[1].hist(np.concatenate(off_t),bins=20)
ax[0].set_yscale('log')
ax[1].set_yscale('log')

  
#%%
'''class'''
findonoff = find_change_pts.MUA_findchangepts()
#%%
findonoff.MinThreshold = 1000 # None #
findonoff.MaxNumChanges = None #15

#analy_dura = data.a1.param.stim1.stim_on[20:22].copy()
#analy_dura[:,0] += (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)

start = 5000; end = 20000
analy_dura = np.array([[start,end]])
on_t, off_t, cpts_cls, onoff_bool, mua_tmp, on_amp, off_amp = findonoff.analyze(data.a1.ge.spk_matrix[mua_neuron], analy_dura=analy_dura)

#%%
#data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])
n_ind, t = data.a1.ge.spk_matrix[mua_neuron,start*10:end*10].nonzero()

#%%
fig,ax = plt.subplots(2,1)
# ax[0].plot(t/10, n_ind, '|')
# ax[0].plot(show_stim_on)  
# ax[1].plot(np.arange(mua_.shape[0]),mua_)
#ax[0].plot(np.array(smt_mua_mat[stim_on_new[20,0]:stim_on_new[20,1]]))
ax[0].plot(mua_tmp[0])
for i in range(cpts_cls[0].shape[0]):
    ax[0].plot([cpts_cls[0][i],cpts_cls[0][i]],[0,80], c=clr[1])
ax[1].plot(onoff_bool[0]*80)
ax[1].plot(t/10, n_ind, '|')
#%%
fig,ax = plt.subplots(2,1)
# ax[0].plot(t/10, n_ind, '|')
# ax[0].plot(show_stim_on)  
# ax[1].plot(np.arange(mua_.shape[0]),mua_)
# ax[0].plot(mua_tmp[0])  
#%%
fig,ax = plt.subplots(2,1)
ax[0].hist(on_t[0])
ax[1].hist(off_t[0])
ax[0].set_yscale('log')
ax[1].set_yscale('log')
"""






