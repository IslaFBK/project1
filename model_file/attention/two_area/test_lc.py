#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 17:04:32 2021

@author: shni2598
"""

import fano_mean_match_test
import mydata
#%%
data_dir = 'raw_data/'
analy_type = 'fano'
#datapath = data_dir
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/twoarea_fano/highff_2stim/'+data_dir
sys_argv = 78#int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num
good_dir = 'good/'
goodsize_dir = 'good_size/'

fftplot = True; getfano = True
get_TunningCurve = True; get_HzTemp = True
get_ani = True
#%%
clr = plt.rcParams['axes.prop_cycle'].by_key()['color']

data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
data_anly = mydata.mydata()

#%%
#if getfano:
data_anly.fano = mydata.mydata()

stim_loc = np.array([0,0])

neuron = np.arange(data.a1.param.Ne)

dist = cn.coordination.lattice_dist(data.a1.param.e_lattice,data.a1.param.width,stim_loc)
neu_pool = [None]*1
neu_pool[0] = neuron[(dist >= 0) & (dist <= 10)]

#fr_bin = np.array([200]) 
    
simu_time_tot = data.param.simutime#29000
#%%
#N_stim = data.a1.param.stim.stim_amp_scale.shape[0]

data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])

fanomm = fano_mean_match_test.fano_mean_match_2()
fanomm.bin_count_interval = 0.5
fanomm.spk_sparmat = data.a1.ge.spk_matrix[neu_pool[0],:]
fanomm.method = 'regression' # 'mean' or 'regression'
fanomm.mean_match_across_condition = True # if do mean matching across different condition e.g. attention or no-attention condition

N_stim = int(round(data.a1.param.stim1.stim_amp_scale.shape[0]/2))
win = 100#[50,100,150]:
fanomm.win = win
fanomm.stim_onoff = data.a1.param.stim1.stim_on[0:N_stim].copy()
fanomm.stim_onoff_2 = data.a1.param.stim1.stim_on[N_stim:N_stim*2].copy() #  on and off time of each stimulus under condition 2; this variable takes effect only when self.mean_match_across_condition =  True

fano_mean_noatt, fano_std_noatt, _, fano_mean_att, fano_std_att, _ = fanomm.get_fano()

# data_anly.fano.fano_mean_noatt = fano_mean_noatt
# data_anly.fano.fano_std_noatt = fano_std_noatt

# # print(np.sum(np.isnan(_)))
# # print(np.sum(np.isnan(fano_mean_noatt)))
# # print(np.sum(np.isnan(fano_std_noatt)))
# fanomm.stim_onoff = data.a1.param.stim1.stim_on[N_stim:N_stim*2].copy()

# fano_mean_att, fano_std_att, _ = fanomm.get_fano()
# # print(np.sum(np.isnan(_)))
# data_anly.fano.fano_mean_att = fano_mean_att
# data_anly.fano.fano_std_att = fano_std_att

fig, ax = plt.subplots(1,1, figsize=[8,6])
ax.errorbar(np.arange(fano_mean_noatt.shape[0])*10-100,fano_mean_noatt,fano_std_noatt,label='no attention')
ax.errorbar(np.arange(fano_mean_att.shape[0])*10-100,fano_mean_att,fano_std_att,label='attention')
ax.set_xlabel('ms')
ax.set_ylabel('fano')
plt.legend()
# title3 = title + '_win%.1f'%fanomm.win#%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean())
# fig.suptitle(title3)
# savetitle = title3.replace('\n','')
# fanofile = savetitle+'_%d'%(loop_num)+'.png'
#fig.savefig(fanofile)
#plt.close()

#%%
#N_stim = data.a1.param.stim.stim_amp_scale.shape[0]

data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])

fanomm = fano_mean_match_test.fano_mean_match_2()
fanomm.bin_count_interval = 0.5
fanomm.spk_sparmat = data.a1.ge.spk_matrix[neu_pool[0],:]
fanomm.method = 'regression' # 'mean' or 'regression'
fanomm.mean_match_across_condition = False # if do mean matching across different condition e.g. attention or no-attention condition

N_stim = int(round(data.a1.param.stim1.stim_amp_scale.shape[0]/2))
win = 100#[50,100,150]:
fanomm.win = win
fanomm.stim_onoff = data.a1.param.stim1.stim_on[0:N_stim].copy()
#fanomm.stim_onoff_2 = data.a1.param.stim1.stim_on[N_stim:N_stim*2].copy() #  on and off time of each stimulus under condition 2; this variable takes effect only when self.mean_match_across_condition =  True

fano_mean_noatt, fano_std_noatt, _ = fanomm.get_fano()

# data_anly.fano.fano_mean_noatt = fano_mean_noatt
# data_anly.fano.fano_std_noatt = fano_std_noatt

# print(np.sum(np.isnan(_)))
# print(np.sum(np.isnan(fano_mean_noatt)))
# print(np.sum(np.isnan(fano_std_noatt)))
fanomm.stim_onoff = data.a1.param.stim1.stim_on[N_stim:N_stim*2].copy()

fano_mean_att, fano_std_att, _ = fanomm.get_fano()
# print(np.sum(np.isnan(_)))
# data_anly.fano.fano_mean_att = fano_mean_att
# data_anly.fano.fano_std_att = fano_std_att

fig, ax = plt.subplots(1,1, figsize=[8,6])
ax.errorbar(np.arange(fano_mean_noatt.shape[0])*10-100,fano_mean_noatt,fano_std_noatt,label='no attention')
ax.errorbar(np.arange(fano_mean_att.shape[0])*10-100,fano_mean_att,fano_std_att,label='attention')
ax.set_xlabel('ms')
ax.set_ylabel('fano')
plt.legend()


#%%
data_anly.fano = mydata.mydata()

stim_loc = np.array([0,0])

neuron = np.arange(data.a1.param.Ne)

dist = cn.coordination.lattice_dist(data.a1.param.e_lattice,data.a1.param.width,stim_loc)
neu_pool = [None]*1
neu_pool[0] = neuron[(dist >= 0) & (dist <= 10)]
#%%
data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])

fanomm = fano_mean_match_test.fano_mean_match_2()
fanomm.bin_count_interval = 0.25
fanomm.spk_sparmat = data.a1.ge.spk_matrix[neu_pool[0],:]
fanomm.method = 'regression' # 'mean' or 'regression'
fanomm.seed = 101
fanomm.mean_match_across_condition = 1 # if do mean matching across different condition e.g. attention or no-attention condition

N_stim = int(round(data.a1.param.stim1.stim_amp_scale.shape[0]/2))
win = 100#[50,100,150]:
fanomm.win = win
fanomm.stim_onoff = data.a1.param.stim1.stim_on[0:N_stim].copy()
fanomm.stim_onoff_2 = data.a1.param.stim1.stim_on[N_stim:N_stim*2].copy() #  on and off time of each stimulus under condition 2; this variable takes effect only when self.mean_match_across_condition =  True


#fano_mean_noatt, fano_std_noatt, _, mean_record, common_mean_hist_all_1 = fanomm.get_fano()
fano_mean_noatt, fano_std_noatt, _, mean_record_1, fano_mean_att, fano_std_att, _, mean_record_2, common_mean_hist_all_2 = fanomm.get_fano()
#%%
mean_record.mean(0)

#%%
#%%
fig, ax = plt.subplots(1,1, figsize=[8,6])
ax.errorbar(np.arange(fano_mean_noatt.shape[0])*10-100,fano_mean_noatt,fano_std_noatt,label='no attention')
ax.errorbar(np.arange(fano_mean_att.shape[0])*10-100,fano_mean_att,fano_std_att,label='attention')
ax.set_xlabel('ms')
ax.set_ylabel('fano')
plt.legend()
#%%
plt.figure()
plt.plot(mean_record.mean(0), 'x')
plt.plot(mean_record_1.mean(0), 'x')
plt.plot(mean_record_2.mean(0), 'o')


#%%
plt.figure()
plt.plot(common_mean_hist_all_1, 'x')
plt.plot(common_mean_hist_all_2, '-')


#%%
data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])

fanomm = fano_mean_match.fano_mean_match()
fanomm.bin_count_interval = 0.25
fanomm.spk_sparmat = data.a1.ge.spk_matrix[neu_pool[0],:]
fanomm.method = 'regression' # 'mean' or 'regression'
fanomm.seed = 101
fanomm.mean_match_across_condition = 1 # if do mean matching across different condition e.g. attention or no-attention condition

N_stim = int(round(data.a1.param.stim1.stim_amp_scale.shape[0]/2))
win = 100#[50,100,150]:
fanomm.win = win
fanomm.stim_onoff = data.a1.param.stim1.stim_on[0:N_stim].copy()
fanomm.stim_onoff_2 = data.a1.param.stim1.stim_on[N_stim:N_stim*2].copy() #  on and off time of each stimulus under condition 2; this variable takes effect only when self.mean_match_across_condition =  True


#fano_mean_noatt, fano_std_noatt, _, mean_record, common_mean_hist_all_1 = fanomm.get_fano()
fano_mean_noatt_f, fano_std_noatt_f, _, fano_mean_att_f, fano_std_att_f, _ = fanomm.get_fano()
#%%

fig, ax = plt.subplots(1,1, figsize=[8,6])
ax.errorbar(np.arange(fano_mean_noatt_f.shape[0])*10-100,fano_mean_noatt_f,fano_std_noatt_f,label='no attention')
ax.errorbar(np.arange(fano_mean_att_f.shape[0])*10-100,fano_mean_att_f,fano_std_att_f,label='attention')
ax.set_xlabel('ms')
ax.set_ylabel('fano')
plt.legend()
#%%
mean_record.mean(0)

#%%
plt.figure()
plt.plot(mean_record.mean(0), 'x')
plt.plot(mean_record_1.mean(0), 'x')
plt.plot(mean_record_2.mean(0), 'o')


















#%%