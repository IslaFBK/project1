#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 15:11:28 2021

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

from scipy.io import savemat
#%%
data_dir = 'raw_data/'
analy_type = 'state'
#datapath = data_dir
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/twoarea_hz/on_off/test_longstim/'+data_dir
sys_argv = int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num
#loop_num = 0

good_dir = 'good/'
goodsize_dir = 'good_size/'

savefile_name = 'data_anly_onoff' #'data_anly' data_anly_temp
save_apd = ''

onff_method = 'threshold'

thre_spon = 5#4
thre_stim = [20] #[12, 15, 30]

fftplot = 1; getfano = 1
get_nscorr = 1; get_nscorr_t = 1
get_TunningCurve = 1; get_HzTemp = 1
firing_rate_long = 1

if loop_num%4 == 0: save_img = 1
else: save_img = 0

if loop_num%10 ==0: get_ani = 0
else: get_ani = 0

save_analy_file = False
#%%
clr = plt.rcParams['axes.prop_cycle'].by_key()['color']


data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
data_anly = mydata.mydata()

n_StimAmp = data.a1.param.stim1.n_StimAmp
n_perStimAmp = data.a1.param.stim1.n_perStimAmp
stim_amp = [400] #200*2**np.arange(n_StimAmp)

if analy_type == 'fbrgbig4': # fbrg: feedback range
    title = '1irie%.2f_1e2e%.1f_pk2e1e%.2f'%(data.param.ie_r_i1, data.inter.param.w_e1_e2_mean/5, \
                                               data.inter.param.peak_p_e2_e1)
        
if analy_type == 'state': # fbrg: feedback range
    title = 'hz_2irie%.2f_2ndgk%.1f_pk2e1e%.2f'%(data.param.ie_r_i2, data.param.new_delta_gk_2, \
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
# start = 5000; end = 20000
# analy_dura = np.array([[start,end]])
# data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])
#%%
simu_time_tot = data.param.simutime
data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10],mat_type='csc')
data.a2.ge.get_sparse_spk_matrix([data.a2.param.Ne, simu_time_tot*10],mat_type='csc')

#%%
st = 0

# data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+n_StimAmp)*n_perStimAmp]


MUA = fra.get_spkcount_sparmat_multi(data.a1.ge.spk_matrix[mua_neuron], data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp],\
                           sum_activity=True, \
                   sample_interval = 10,  window = 10, dt = 0.1)

MUA_a2 = fra.get_spkcount_sparmat_multi(data.a2.ge.spk_matrix[mua_neuron], data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp:(st+n_StimAmp+1)*n_perStimAmp],\
                           sum_activity=True, \
                   sample_interval = 10,  window = 10, dt = 0.1)

#%%    
plt.figure()
plt.plot(MUA_a2[0])

#%%
MUA_mat = {'mua': MUA}
savemat("MUA.mat", MUA_mat)
#%%
MUA_mat_a2 = {'mua_a2': MUA_a2}
savemat("MUA_a2.mat", MUA_mat_a2)
#%%
n_ind, t_ind = data.a1.ge.spk_matrix[mua_neuron,data.a1.param.stim1.stim_on[0,0]*10:data.a1.param.stim1.stim_on[0,1]*10].nonzero()
#%%
plt.figure()
plt.plot(t_ind,n_ind,'|')
#%%
import matlab.engine
from scipy.sparse import csc_matrix

#%%
eng = matlab.engine.start_matlab('-nodisplay')
#%%
MUA_mat = matlab.double(MUA.tolist(), size=[MUA.shape[0],MUA.shape[1]])
R = eng.HMM(MUA_mat)
#%%
#MUA_mat = matlab.double(MUA.tolist(), size=[MUA.shape[0],MUA.shape[1]])
R = eng.HMM(results.mua)
#%%
for key in R:
    R[key] = np.squeeze(np.array(R[key]))
#%%
R = mydata.mydata(R)
R.state_inferred = R.state_inferred.astype(int)

#%%
import HMM_py
hmm_analy_md = HMM_py.HMM_onoff()
hmm_analy_md.mat_eng = eng
#hmm_analy.mat_eng.quit()
#hmm_analy.mat_eng = matlab.engine.start_matlab('-nodisplay')

results_md = hmm_analy_md.analyze(data.a1.ge.spk_matrix[mua_neuron], data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+n_StimAmp)*n_perStimAmp])
#%%
for i in range(len(results_md.onset_t)):
    print(i)
    conc = np.sort(np.concatenate((results_md.onset_t[i], results_md.offset_t[i])))
    # print(conc[:5], results.cpts[i][:5])
    # print(conc.shape, results.cpts[i].shape)
    # print(np.where(conc!=results.cpts[i])[0])
    ifsame = np.all(conc==results_md.cpts[i])
    print(ifsame)
    
#%%

class HMM_onoff:
    
    def __init__(self):
        
        #self.stim_dura = None
        self.dt = 0.1 # ms
        self.mua_sampling_interval = 10 # ms
        self.mua_win = 10 # ms 
        self.mat_eng = None # matlab engine  
        
    def analyze(self, spk_mat, analy_dura):#, method = "comparing", threshold=None):

        # analy_dura : ms
        on_t = [None for _ in range(len(analy_dura))]
        off_t = [None for _ in range(len(analy_dura))]
        onset_t = [None for _ in range(len(analy_dura))]
        offset_t = [None for _ in range(len(analy_dura))]
        cpts = [None for _ in range(len(analy_dura))]
        #onoff_bool = [None for _ in range(len(analy_dura))]
        mua = [None for _ in range(len(analy_dura))]
        on_amp = [None for _ in range(len(analy_dura))]
        off_amp = [None for _ in range(len(analy_dura))]
        
        if not isinstance(spk_mat, csc_matrix):        
            spk_mat = spk_mat.tocsc()
            
        for i in range(len(analy_dura)):    

            mua[i] = find_MUA(spk_mat[:,analy_dura[i,0]*10:analy_dura[i,1]*10], self.dt, self.mua_sampling_interval, self.mua_win)
            
            mua[i] = matlab.double(mua[i].tolist(), size=[1,mua[i].shape[0]])
            
        R = self.mat_eng.HMM(mua)
        for key in R:
            if key == 'state_inferred':
                for t in range(len(R[key])):
                    R[key][t] = np.squeeze(np.array(R[key][t])).astype(int)
            # elif key == 'mua':
            #     for t in range(len(R[key])):
            #         R[key][t] = np.squeeze(np.array(R[key][t]))               
            else:
                R[key] = np.squeeze(np.array(R[key]))
            # cpts[i], self.mat_eng = find_change_pts(mua[i], smooth_method = self.smooth_method, smooth_window = self.smooth_window, \
            #         chg_pts_statistic = self.chg_pts_statistic, MinThreshold = self.MinThreshold, \
            #         MaxNumChanges = self.MaxNumChanges, MinDistance = int(round(self.MinDistance/self.mua_sampling_interval)), eng=self.mat_eng)
            
            # onoff_bool[i], on_t[i], off_t[i], on_amp[i], off_amp[i] = on_off_properties(mua[i], cpts[i],  method = method, threshold = threshold)
        
        
        for i in range(len(analy_dura)): 
            mua[i] = np.squeeze(np.array(mua[i])) 
            on_t[i], off_t[i], on_amp[i], off_amp[i], onset_t[i], offset_t[i], cpts[i] = on_off_properties(mua[i], R['state_inferred'][i])
        
        R['onoff_bool'] = R.pop('state_inferred')
        results = mydata.mydata(R)
        results.on_t = on_t
        results.off_t = off_t
        results.onset_t = onset_t
        results.offset_t = offset_t
        results.cpts = cpts
        #results.onoff_bool = onoff_bool
        results.mua = mua
        results.on_amp = on_amp
        results.off_amp = off_amp
        
        return results
#%%
hmm_analy = HMM_onoff()
hmm_analy.mat_eng = eng
#hmm_analy.mat_eng.quit()
#hmm_analy.mat_eng = matlab.engine.start_matlab('-nodisplay')

results = hmm_analy.analyze(data.a1.ge.spk_matrix[mua_neuron], data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+n_StimAmp)*n_perStimAmp])
#results = hmm_analy.analyze(data.a1.ge.spk_matrix[mua_neuron], data.a1.param.stim1.stim_on[:4])

#%%

for i in range(len(results.onset_t)):
    print(i)
    conc = np.sort(np.concatenate((results.onset_t[i], results.offset_t[i])))
    # print(conc[:5], results.cpts[i][:5])
    # print(conc.shape, results.cpts[i].shape)
    # print(np.where(conc!=results.cpts[i])[0])
    ifsame = np.all(conc==results.cpts[i])
    print(ifsame)
#%%
conc = np.sort(np.concatenate((results.onset_t[3], results.offset_t[3])))

#%%
#R.state_inferred
#for onoff_bool in R.state_inferred:
def on_off_properties(data, onoff_bool):
     
    # duration of on-state
    onoff_bool_tmp = np.concatenate(([0], onoff_bool, [0]))
    c = np.where(np.diff(onoff_bool_tmp) != 0)[0]
    on_t = c[1::2] - c[0::2]
    on_amp = []
     
    for pts_str, pts_end in zip(c[0::2],c[1::2]):
        on_amp.append(np.mean(data[pts_str:pts_end]))
    on_amp = np.array(on_amp)
    
    onset_t = c[0::2]
    offset_t = c[1::2]
    #print(onset_t[-5:])
    #print(offset_t[-5:])
    transition_t = np.where(np.diff(onoff_bool) != 0)[0] + 1
    
    if onoff_bool[0] == 1:
        on_t = np.delete(on_t,0) # discard first points if they are on states
        on_amp = np.delete(on_amp,0)
        onset_t = np.delete(onset_t,0)
        # print('onoff_bool[0] == 1')
        # print(onset_t[-5:])
    if onoff_bool[-1] == 1:
        on_t = np.delete(on_t,-1) # discard end points if they are on states
        on_amp = np.delete(on_amp,-1)
        offset_t = np.delete(offset_t,-1)
        # print('onoff_bool[-1] == 1')
        # print(offset_t[-5:])           
    # duration of off-state
    onoff_bool_tmp = np.concatenate(([1], onoff_bool, [1]))
    c = np.where(np.diff(onoff_bool_tmp) != 0)[0]
    off_t = c[1::2] - c[0::2]
    off_amp = []
    
    for pts_str, pts_end in zip(c[0::2],c[1::2]):
        off_amp.append(np.mean(data[pts_str:pts_end]))
    off_amp = np.array(off_amp)
    
    if onoff_bool[0] == 0:
        off_t = np.delete(off_t,0) # discard first points if they are off states
        off_amp = np.delete(off_amp,0)
    if onoff_bool[-1] == 0:
        off_t = np.delete(off_t,-1)  # discard end points if they are off states
        off_amp = np.delete(off_amp,-1)
    
    del onoff_bool_tmp        
    
    return on_t, off_t, on_amp, off_amp, onset_t, offset_t, transition_t

#%%
def find_MUA(spk_mat, dt, sampling_interval, win):
    
    sampling_interval = int(np.round(sampling_interval/dt))
    #start_time = int(np.round(start_time/dt))
    #end_time = int(np.round(end_time/dt))
    window_step = int(np.round(win/dt))

    sample_t = np.arange(0, spk_mat.shape[1]-window_step+1, sampling_interval)

    # spk_mat = spk_mat.tocsc()
    #spk_mat = csc_matrix((np.ones(len(spikei),dtype=int),(spikei,spiket-start_time)),(n_neuron,end_time-start_time))
    mua = np.zeros([len(sample_t)], dtype=int)
        
    for i in range(len(sample_t)):
        mua[i] = spk_mat.indptr[sample_t[i]+window_step] - spk_mat.indptr[sample_t[i]]
        #neu, counts = np.unique(spk_mat.indices[spk_mat.indptr[sample_t[i]]:spk_mat.indptr[sample_t[i]+window_step]],return_counts=True)
        #spk_rate[:, i][neu] += counts
    return mua



