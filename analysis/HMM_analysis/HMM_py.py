#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 20:11:38 2021

@author: shni2598
"""

import numpy as np
import matlab.engine
from scipy.sparse import csc_matrix
import mydata
#%%
class HMM_onoff:
    
    def __init__(self):
        
        #self.stim_dura = None
        self.dt = 0.1 # ms
        self.mua_sampling_interval = 10 # ms
        self.mua_win = 10 # ms 
        self.mat_eng = None # matlab engine  
        
    def analyze(self, spk_mat, analy_dura, cross_validation=True):#, method = "comparing", threshold=None):

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
            
        R = self.mat_eng.HMM_mat(mua, cross_validation)
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
            on_t[i], off_t[i], on_amp[i], off_amp[i], onset_t[i], offset_t[i], cpts[i] = on_off_properties(mua[i], R['state_inferred'][i],self.mua_sampling_interval)
        
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
    
    
def on_off_properties(data, onoff_bool, sampling_interval):
     
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
    
    on_t *= sampling_interval
    off_t *= sampling_interval
    onset_t *= sampling_interval
    offset_t *= sampling_interval
    transition_t *= sampling_interval
    
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
