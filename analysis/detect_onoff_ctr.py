#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 11:13:37 2021

@author: shni2598
"""


import numpy as np
#import matlab.engine
import mydata
from scipy.sparse import csc_matrix

#import firing_rate_analysis as fra

class detect_onoff:
    '''use centre of mass of pattern to detect on and off state of firing rate'''
    def __init__(self):
        
        #self.stim_dura = None
        self.dt = 0.1 # ms
        self.mua_sampling_interval = 1 # ms
        self.mua_win = 10 # ms 
        
        #self.smooth_method = 'rlowess' 
        #self.smooth_window = None
        #self.chg_pts_statistic = 'mean'
        #self.MinThreshold = None
        #self.MaxNumChanges = None
        #self.MinDistance = 5 # ms
        #self.mat_eng = None # matlab engine  
       # self.threshold_range = 7
        
    def analyze(self, spk_mat, mua_neuron, analy_dura, threshold_range=7, mua_loc=np.array([31.5,31.5])):
        '''
        Parameters
        ----------
        spk_mat : sparse matrix
            time and index of neruons emitting spike.
        analy_dura : 2-D array
            each row of 'analy_dura' indicating the start and end time points for on-off detection.
        method : string
            "threshold" or "comparing", The default is "threshold".
            the method to define on and off       
        threshold: scalar
            threshold for discriminating on and off state; only used for method = "threshold"

        Returns
        -------
        results : an object of class of mydata.mydata
            an object recording results

        '''
        # analy_dura : ms
        on_t = [None for _ in range(len(analy_dura))]
        off_t = [None for _ in range(len(analy_dura))]
        cpts = [None for _ in range(len(analy_dura))]
        onoff_bool = [None for _ in range(len(analy_dura))]
        mua = [None for _ in range(len(analy_dura))]
        on_amp = [None for _ in range(len(analy_dura))]
        off_amp = [None for _ in range(len(analy_dura))]
        onset_t = [None for _ in range(len(analy_dura))]
        offset_t = [None for _ in range(len(analy_dura))]
        centre = [None for _ in range(len(analy_dura))]


        for i in range(len(analy_dura)):    
            # mua = fra.get_spike_rate(spk_mat, analy_dura[i,0],analy_dura[i,1], indiv_rate=True, popu_rate=False, \
            #                    sample_interval=self.mua_sampling_interval, n_neuron=spk_mat.shape[0], \
            #                        window = self.mua_win, dt=self.dt, reshape_indiv_rate=False,save_results_to_input=False)
            mua[i], centre[i], onoff_bool[i] = get_mua_onoff(spk_mat, mua_neuron, analy_dura[i,0], analy_dura[i,1], threshold_range, mua_loc,\
                   sample_interval = self.mua_sampling_interval,  window = self.mua_win, dt = self.dt)
            
        for i in range(len(analy_dura)): 
            #mua[i] = np.squeeze(np.array(mua[i])) 
            on_t[i], off_t[i], on_amp[i], off_amp[i], onset_t[i], offset_t[i], cpts[i] = on_off_properties(mua[i], onoff_bool[i],self.mua_sampling_interval)

        #R['onoff_bool'] = R.pop('state_inferred')
        results = mydata.mydata()
        results.on_t = on_t
        results.off_t = off_t
        results.onset_t = onset_t
        results.offset_t = offset_t
        results.cpts = cpts
        results.onoff_bool = onoff_bool
        results.centre = centre
        results.mua = mua
        results.on_amp = on_amp
        results.off_amp = off_amp
        
        return results




#%%
def get_mua_onoff(spk_sparmat, mua_neuron, start_time, end_time, threshold_range, mua_loc = np.array([31.5,31.5]),\
                   sample_interval = 1,  window = 10, dt = 0.1):
    
    sample_interval = int(np.round(sample_interval/dt))
    start_time = int(np.round(start_time/dt))
    end_time = int(np.round(end_time/dt))
    window_step = int(np.round(window/dt))
    
    
###################
    len_hori = int(np.sqrt(spk_sparmat.shape[0]).round())
    len_vert = int(np.sqrt(spk_sparmat.shape[0]).round())
    x_hori = np.cos((np.arange(len_hori)+0.5)/len_hori*2*np.pi).reshape([-1,1])
    y_hori = np.sin((np.arange(len_hori)+0.5)/len_hori*2*np.pi).reshape([-1,1])
    xy_hori = np.concatenate((x_hori,y_hori),axis=1)
    
    x_vert = np.cos((np.arange(len_vert)+0.5)/len_vert*2*np.pi).reshape([-1,1])
    y_vert = np.sin((np.arange(len_vert)+0.5)/len_vert*2*np.pi).reshape([-1,1])
    xy_vert = np.concatenate((x_vert,y_vert),axis=1)
    hw_h = 0.5*(len_hori); hw_v = 0.5*(len_vert)
    
    #mua_loc = np.array([31.5,31.5])
    
#################   
    if not isinstance(spk_sparmat, csc_matrix):        
        spk_sparmat = spk_sparmat.tocsc()
    
    sample_t = np.arange(start_time, end_time-window_step+1, sample_interval)
    #spk_count = np.zeros([spk_sparmat.shape[0], sample_t.shape[0]], dtype=int)
    spk_count = np.zeros([spk_sparmat.shape[0]], dtype=int)
    mua = np.zeros([sample_t.shape[0]], dtype=int)
    centre = np.zeros([sample_t.shape[0], 2])
    onoff_bool = np.zeros(sample_t.shape[0], int)
    
    for i in range(sample_t.shape[0]):
        spk_count[:] = 0
        neu, counts = np.unique(spk_sparmat.indices[spk_sparmat.indptr[sample_t[i]]:spk_sparmat.indptr[sample_t[i]+window_step]],return_counts=True)
        #spk_count[:, i][neu] += counts
        spk_count[neu] += counts
        mua[i] = spk_count[mua_neuron].sum()
        spk_count = spk_count.reshape(len_vert, len_hori)
        
        centre_i = find_centreMass(spk_count, len_hori, len_vert, xy_hori, xy_vert)
        if centre_i is None:
            if i == 0:
                centre[i,:] = 0
            else:
                centre[i,:] = centre[i-1,:]
            #onoff_bool[i] = 0
        else:
            centre[i,:] = centre_i
        
        
            dist_h = (mua_loc[1] - centre[i,1] + hw_h)%(2*hw_h) - hw_h
            dist_v = (mua_loc[0] - centre[i,0] + hw_v)%(2*hw_v) - hw_v
            dist = np.sqrt(dist_h**2 + dist_v**2)
            if dist <= threshold_range:
                onoff_bool[i] = 1
        
        spk_count = spk_count.reshape(-1)
        
    return mua, centre, onoff_bool


#%%
def find_centreMass(spk_count, len_hori, len_vert, xy_hori, xy_vert):
    

    centre = np.zeros(2, dtype=float)


    #for ind in range(len(slide_ary)):
    if np.all(spk_count == 0):
        centre = None

    else:
        sum_hori = np.sum(spk_count, axis=0)
        sum_vert = np.sum(spk_count, axis=1)
        ctr_hori = np.dot(sum_hori, xy_hori)
        ctr_vert = np.dot(sum_vert, xy_vert)
        '''
        if ctr_hori[1] >= 0:
            ind_hori = int((npa.arctan2(ctr_hori[1],ctr_hori[0])*len_hori)/(2*np.pi))
        else:
            ind_hori = int(((2*np.pi+np.arctan2(ctr_hori[1],ctr_hori[0]))*len_hori)/(2*np.pi))
        if ctr_vert[1] >= 0:
            ind_vert = int((np.arctan2(ctr_vert[1],ctr_vert[0])*len_vert)/(2*np.pi))
        else:
            ind_vert = int(((2*np.pi+np.arctan2(ctr_vert[1],ctr_vert[0]))*len_vert)/(2*np.pi))
        '''
        centre[1] = np.angle(np.array([ctr_hori[0] + 1j*ctr_hori[1]]))[0]
        centre[0] = np.angle(np.array([ctr_vert[0] + 1j*ctr_vert[1]]))[0]
        centre[0] = wrapTo2Pi(np.array([centre[0]]))*len_vert/(2*np.pi)
#        if ind_vert > 62: 
#            ind_vert = 62
        centre[1] = wrapTo2Pi(np.array([centre[1]]))*len_hori/(2*np.pi)
        
        centre -= 0.5
        
    return centre
    
#%%
def wrapTo2Pi(angle):
    #positiveinput = (angle > 0)
    angle = np.mod(angle, 2*np.pi)
    #angle[(angle==0) & positiveinput] = 2*np.pi
    return angle

def wrapToPi(angle):
    select = (angle < -np.pi) | (angle > np.pi) 
    angle[select] = wrapTo2Pi(angle[select] + np.pi) - np.pi
    return angle   
#%%  
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