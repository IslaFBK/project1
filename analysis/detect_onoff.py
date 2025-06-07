#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 17:34:41 2021

@author: shni2598
"""

import numpy as np
import matlab.engine
import mydata
#import firing_rate_analysis as fra

class MUA_findchangepts:
    
    def __init__(self):
        
        #self.stim_dura = None
        self.dt = 0.1 # ms
        self.mua_sampling_interval = 1 # ms
        self.mua_win = 10 # ms 
        
        self.smooth_method = 'rlowess' 
        self.smooth_window = None
        self.chg_pts_statistic = 'mean'
        self.MinThreshold = None
        self.MaxNumChanges = None
        self.MinDistance = 5 # ms
        self.mat_eng = None # matlab engine  
    
    def analyze(self, spk_mat, analy_dura, method = "comparing", threshold=None):
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
        for i in range(len(analy_dura)):    
            # mua = fra.get_spike_rate(spk_mat, analy_dura[i,0],analy_dura[i,1], indiv_rate=True, popu_rate=False, \
            #                    sample_interval=self.mua_sampling_interval, n_neuron=spk_mat.shape[0], \
            #                        window = self.mua_win, dt=self.dt, reshape_indiv_rate=False,save_results_to_input=False)
            mua[i] = find_MUA(spk_mat[:,analy_dura[i,0]*10:analy_dura[i,1]*10], self.dt, self.mua_sampling_interval, self.mua_win)
                
            cpts[i], self.mat_eng = find_change_pts(mua[i], smooth_method = self.smooth_method, smooth_window = self.smooth_window, \
                    chg_pts_statistic = self.chg_pts_statistic, MinThreshold = self.MinThreshold, \
                    MaxNumChanges = self.MaxNumChanges, MinDistance = int(round(self.MinDistance/self.mua_sampling_interval)), eng=self.mat_eng)
            
            onoff_bool[i], on_t[i], off_t[i], on_amp[i], off_amp[i], onset_t[i], offset_t[i] \
                = on_off_properties(mua[i], cpts[i],  method = method, threshold = threshold)
        
        
        results = mydata.mydata()
        results.on_t = on_t
        results.off_t = off_t
        results.cpts = cpts
        results.onoff_bool = onoff_bool
        results.mua = mua
        results.on_amp = on_amp
        results.off_amp = off_amp
        results.onset_t = onset_t
        results.offset_t = offset_t
        
        return results
        
        
#%%
def find_MUA(spk_mat, dt, sampling_interval, win):
    
    sampling_interval = int(np.round(sampling_interval/dt))
    #start_time = int(np.round(start_time/dt))
    #end_time = int(np.round(end_time/dt))
    window_step = int(np.round(win/dt))

    sample_t = np.arange(0, spk_mat.shape[1]-window_step+1, sampling_interval)

    #spiket = spike.t #np.round(spike.t/dt).astype(int)
    #spikei = spike.i[(spiket >= start_time) & (spiket < end_time)]
    #spiket = spiket[(spiket >= start_time) & (spiket < end_time)]
    
    spk_mat = spk_mat.tocsc()
    #spk_mat = csc_matrix((np.ones(len(spikei),dtype=int),(spikei,spiket-start_time)),(n_neuron,end_time-start_time))
    mua = np.zeros([len(sample_t)], dtype=int)
        
    for i in range(len(sample_t)):
        mua[i] = spk_mat.indptr[sample_t[i]+window_step] - spk_mat.indptr[sample_t[i]]
        #neu, counts = np.unique(spk_mat.indices[spk_mat.indptr[sample_t[i]]:spk_mat.indptr[sample_t[i]+window_step]],return_counts=True)
        #spk_rate[:, i][neu] += counts
    return mua
#%%
def on_off_properties(data, cpts, method = "comparing", threshold = None):
    '''
    Parameters
    ----------
    data : array
        time series data.
    cpts : array
        change points.
    method : string
        "threshold" or "comparing", The default is "comparing".
        the method to define on and off
    threshold: scalar
        threshold for discriminating on and off state; only used for method = "threshold"
    
    Returns
    -------
    onoff_bool : array
        time series with the same length as 'data', indicating on or off state.
    on_t : array
        periods of each on states.
    off_t : array
        periods of each off states.
    on_amp : array
        mean value of each on states.
    off_amp : array
        mean value of each off states.
    '''
    # adapted from GuoZhang and Qi's code
    
    
    if method == "comparing":
        # comparing the mean value in each section with the mean of its two neighbour
        # if it's greater than its two neighbour then the corresponding section is regarded as on state, otherwise off state
        cpts = np.concatenate(([0], cpts))
        
        mean_data = np.zeros(data.shape)
        for i in range(cpts.shape[0]-1):
            mean_data[cpts[i]:cpts[i+1]] = np.mean(data[cpts[i]:cpts[i+1]])
        #mean_data[-1] = mean_data[-2]
        mean_data[cpts[-1]:] = np.mean(data[cpts[-1]:])
        
        onoff_bool = np.zeros(data.shape, int)
        on_amp = []
        off_amp = []
        for i in range(cpts.shape[0]):
            if i == 0:
                if mean_data[cpts[i]] > mean_data[cpts[i+1]]:
                    onoff_bool[cpts[i]:cpts[i+1]] = 1
                    on_amp.append(mean_data[cpts[i]])
                else:
                    off_amp.append(mean_data[cpts[i]])
                    
            elif i == cpts.shape[0]-1:
                if mean_data[cpts[i]] > mean_data[cpts[i-1]]:
                    onoff_bool[cpts[i]:] = 1
                    on_amp.append(mean_data[cpts[i]])
                else:
                    off_amp.append(mean_data[cpts[i]])
            else:
                if mean_data[cpts[i]] > mean_data[cpts[i-1]] and mean_data[cpts[i]] > mean_data[cpts[i+1]]:
                    onoff_bool[cpts[i]:cpts[i+1]] = 1
                    on_amp.append(mean_data[cpts[i]])
                else:
                    off_amp.append(mean_data[cpts[i]])
        on_amp = np.array(on_amp)
        off_amp = np.array(off_amp)
        
        # duration of on-state
        onoff_bool_tmp = np.concatenate(([0], onoff_bool, [0]))
        c = np.where(np.diff(onoff_bool_tmp) != 0)[0]
        onset_t = c[0::2]
        offset_t = c[1::2]
        on_t = c[1::2] - c[0::2]
        
        if onoff_bool[0] == 1:
            on_t = np.delete(on_t,0) # discard first points if they are on states
            on_amp = np.delete(on_amp,0)
            onset_t = np.delete(onset_t,0)
        if onoff_bool[-1] == 1:
            on_t = np.delete(on_t,-1) # discard end points if they are on states
            on_amp = np.delete(on_amp,-1)
            offset_t = np.delete(offset_t,-1)
        # duration of off-state
        onoff_bool_tmp = np.concatenate(([1], onoff_bool, [1]))
        c = np.where(np.diff(onoff_bool_tmp) != 0)[0]
        off_t = c[1::2] - c[0::2]
        if onoff_bool[0] == 0:
            off_t = np.delete(off_t,0) # discard first points if they are off states
            off_amp = np.delete(off_amp,0)
        if onoff_bool[-1] == 0:
            off_t = np.delete(off_t,-1)  # discard end points if they are off states
            off_amp = np.delete(off_amp,-1)
        del onoff_bool_tmp
        
        return onoff_bool, on_t, off_t, on_amp, off_amp, onset_t, offset_t
    
    
    if method == "threshold":
        # determine on or off state by comparing the mean value of each section with 'threshold'
        cpts = np.concatenate(([0], cpts))
        mean_data = np.zeros(data.shape)
        for i in range(cpts.shape[0]-1):
            mean_data[cpts[i]:cpts[i+1]] = np.mean(data[cpts[i]:cpts[i+1]])
        #mean_data[-1] = mean_data[-2]
        mean_data[cpts[-1]:] = np.mean(data[cpts[-1]:])
        
        onoff_bool = mean_data >= threshold
        
        # duration of on-state
        onoff_bool_tmp = np.concatenate(([0], onoff_bool, [0]))
        c = np.where(np.diff(onoff_bool_tmp) != 0)[0]
        onset_t = c[0::2]
        offset_t = c[1::2]
        on_t = c[1::2] - c[0::2]
        on_amp = []
     
        for pts_str, pts_end in zip(c[0::2],c[1::2]):
            on_amp.append(np.mean(data[pts_str:pts_end]))
        on_amp = np.array(on_amp)
        
        if onoff_bool[0] == 1:
            on_t = np.delete(on_t,0) # discard first points if they are on states
            on_amp = np.delete(on_amp,0)
            onset_t = np.delete(onset_t,0)
        if onoff_bool[-1] == 1:
            on_t = np.delete(on_t,-1) # discard end points if they are on states
            on_amp = np.delete(on_amp,-1) 
            offset_t = np.delete(offset_t,-1)
                
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
        
        return onoff_bool, on_t, off_t, on_amp, off_amp, onset_t, offset_t

#%%
def find_change_pts(data, smooth_method = 'rlowess', smooth_window = None, \
                    chg_pts_statistic = 'mean', MinThreshold = None, \
                    MaxNumChanges = None, MinDistance = 5, eng=None):
    '''

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    smooth_method : TYPE, optional
        DESCRIPTION. The default is 'rlowess'.
    smooth_window : TYPE, optional
        DESCRIPTION. The default is None.
    chg_pts_statistic : TYPE, optional
        DESCRIPTION. The default is 'mean'.
    MinThreshold : TYPE, optional
        DESCRIPTION. The default is None.
    MaxNumChanges : TYPE, optional
        DESCRIPTION. The default is None.
    MinDistance : scalar; unit: ms
        DESCRIPTION. The default is 5.
    eng : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    cpts : numpy array
        change points index.

    '''
    
    if not isinstance(data, np.ndarray):
        raise Exception('Error: data must be numpy array!')
    else:
        if len(data.shape) != 1:
            raise Exception('Error: data must be 1-D numpy array!')    
    
    if eng is None:
        eng = start_Mat_Eng()
        
    if MinThreshold is not None and MaxNumChanges is not None:
        raise Exception("'MinThreshold' and 'MaxNumChanges' cannot be simultaneously used!")
    
    data = matlab.double(list(data), size=[data.size,1])
    
    if smooth_window is None: 
        data_smt = eng.smoothdata(data, smooth_method)
    else:
        data_smt = eng.smoothdata(data, smooth_method, smooth_window)
    
    if MaxNumChanges is not None:
        cpts = eng.findchangepts(data_smt, 'statistic', chg_pts_statistic,'MaxNumChanges', MaxNumChanges,'MinDistance',MinDistance)
    elif MinThreshold is not None:
        cpts = eng.findchangepts(data_smt, 'statistic', chg_pts_statistic,'MinThreshold', MinThreshold,'MinDistance',MinDistance)
    else:
        raise Exception("Either 'MaxNumChanges' or 'MinThreshold' should be provided for matlab 'findchangepts'!")
    
    cpts = np.array(cpts).reshape(-1).astype(int)
    cpts -= 1
    #cpts = np.concatenate(([0], cpts, [data.size[0]-1]))
    
    return cpts, eng
#%%    

def start_Mat_Eng(args='-nodisplay'):
    ''' start Matlab engine'''
    return matlab.engine.start_matlab(args) # '-nodisplay'










#%%
# #import numpy as np
# import fitStable_mat

# data = rnd #np.random.randn(100)

# fitparam, eng = fitStable_mat.fitStable(data)

# eng.quit()
# #%%
# import levy
# #%%
# rnd = levy.random(1.5, 0.5, 0, 1.2, shape=(100,))

# #%%
# param = levy.fit_levy(rnd)



