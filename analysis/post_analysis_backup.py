#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 12:24:45 2020

@author: shni2598
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:10:28 2019

@author: shni2598
"""
import brian2.numpy_ as np
from brian2.only import *
#%%
#class analysis:
#    
#    def __init__(data_dict, output):
#        
#        self.data = data_dict
#        self.cv_wind = 5*ms
#        self.cv_wind_type = 'flat'
#        self.output_type = ('cv_rate')
#        
#        if 'cv_rate' in output():
#            self.get_cv_rate(self.data['popu_rate'])
#        
#    
#    def get_cv_rate(popu_rate):
#        
#        smth_popu_rate = popu_rate.smooth_rate(self.cv_wind_type, self.cv_wind)
#        
#        smth_popu_rate = smth_popu_rate/Hz
#        
#        self.cv_rate = np.std(smth_popu_rate)/np.mean(smth_popu_rate)
#    
#%%
#dtest = {'a':None, 'b':None}
#'a' in dtest.keys()
#
##%%
#{'cv_rate':{cv_rate_e}}

#%%

import pickle
from scipy.sparse import csr_matrix
import matplotlib
#matplotlib.use('Qt5Agg')
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.linear_model import LinearRegression

import pdb
#%%
#class analysis:
    
    # generate spike sparse matrix from time and neuron index
def index_to_matrix(spike, row, col, dt):
    
    #Ne = 63*63;  dt = 0.1*ms; simu_step = round(25*second/dt);
    return csr_matrix((np.ones(len(spike.i),int),(spike.i, (np.round(spike.t/dt)).astype(int))),shape=(row, col))

    

# extract the spike data of a sub-group of neurons from the orrignal spike data
def subsample_spike_neuron(spk_data, ind): # ind: index of sub-group neurons; spk_data: orriginal spike data(sparse matrix)
    
    row_ind = np.array([], dtype=int)
    col_ind = np.array([], dtype=int)
    for i in ind:
        col_ind = np.concatenate((col_ind, spk_data.indices[spk_data.indptr[i]: spk_data.indptr[i+1]]))
        row_ind = np.concatenate((row_ind, i*np.ones(spk_data.indptr[i+1] - spk_data.indptr[i], dtype=int)))
    
    return row_ind, col_ind
#%%    
def detect_avalanche_window(spike, row, col, cv_window, rate_window, aval_window, 
                            dt = 0.1*ms, discard_t=200*ms):
    
    cv_window = round(cv_window/dt); rate_window = round(rate_window/dt); 
    aval_window = round(aval_window/dt)
    
    dt = dt/ms; discard_t = discard_t/ms
    
    #pdb.set_trace()
    
    spk_e_t = np.round(((spike.t/ms)[spike.t[:]/ms>=discard_t]/dt)).astype(int) -round(discard_t/dt) # dispose the first 200ms data
    spk_e_ind = spike.i[spike.t[:]/ms>=discard_t]
    
    spk = csr_matrix((np.ones(len(spk_e_ind),dtype=int), (spk_e_ind, spk_e_t) ), shape = (row, col-round(discard_t/dt)))
    spk_popu = spk.sum(axis=0).A[0]
    
    num_cv = int(len(spk_popu)/cv_window) # number cv window of whole population spike train 
    num_rate = int(cv_window/rate_window) # number rate window in each cv window
    
    cv = np.zeros(num_cv) # an array storing cv value of each cv_window
    
    
    for i in range(0, num_cv):
        spk_per_cv  = spk_popu[i*cv_window:(i+1)*cv_window]
        rate = np.zeros(num_rate)
        for j in range(0,num_rate):
            rate[j] = (spk_per_cv[j*rate_window:(j+1)*rate_window]).sum()
            
        cv[i] = np.std(rate)/np.mean(rate)
    
    #pdb.set_trace()
    # start detecting avalanche 
    
    aval_st = [None]*len(cv)
    
    for i in range(len(aval_st)):
        
        spk_per_cv = spk_popu[i*cv_window:(i+1)*cv_window]
        spk_aval_cv = np.zeros(int(cv_window/aval_window), dtype=int) # number of aval_window in each cv_window
        aval_time = np.array([],int)
        aval_size = np.array([],int)            
        
        for j in range(int(cv_window/aval_window)):
            
            spk_aval_cv[j] = np.sum(spk_per_cv[j*aval_window:(j+1)*aval_window])
        
        #pdb.set_trace()
        
        threshold = 0.1*np.median(spk_aval_cv)
        
        if spk_aval_cv[0]>threshold:
           is_aval_pre = True
           dura = aval_window
           size = spk_aval_cv[0]
        else:
            is_aval_pre = False
            dura = 0
            size = 0
            
        for k in range(1, len(spk_aval_cv)):
            if not(is_aval_pre) and spk_aval_cv[k]>threshold:
                dura += aval_window
                size += spk_aval_cv[k]
                is_aval_pre = True
            elif is_aval_pre and spk_aval_cv[k]>threshold:
                dura += aval_window
                size += spk_aval_cv[k]
            elif is_aval_pre and not(spk_aval_cv[k]>threshold):
                aval_time = np.concatenate((aval_time, [dura]))
                aval_size = np.concatenate((aval_size, [size]))
                dura = 0
                size = 0
                is_aval_pre = False
                #pdb.set_trace()
            else: continue
        #pdb.set_trace()
        aval_st[i] = np.concatenate((aval_time.reshape(1,-1), aval_size.reshape(1,-1)),0)
    
    return aval_st, cv
    
    
    #return cv
 
#    def detect_avalanche_window(spike, row, col, cv_window, rate_window, aval_window, 
#                                dt = 0.1*ms, discard_t=200*ms):
#        
#        cv_window = round(cv_window/dt); rate_window = round(rate_window/dt); 
#        aval_window = round(aval_window/dt)
#        
#        dt = dt/ms; discard_t = discard_t/ms
#        
#        spk_e_t = np.round(((spike.t/ms)[spike.t[:]/ms>=discard_t]/dt)).astype(int) -round(discard_t/dt) # dispose the first 200ms data
#        spk_e_ind = spike.i[spike.t[:]/ms>=discard_t]
#        
#        spk = csr_matrix((np.ones(len(spk_e_ind),dtype=int), (spk_e_ind, spk_e_t) ), shape = (row, col-round(discard_t/dt)))
#        spk_popu = spk.sum(axis=0).A[0]
#        
#        if rate_window%2 ==0:
#            rate_kernel = np.ones(rate_window+1)/((rate_window+dt)/10000)/row 
#        else:
#            rate_kernel = np.ones(rate_window)/(rate_window/10000)/row 
#        
#        spk_rate = np.convolve(spk_popu, rate_kernel, mode='valid')
#        
#        #cv_window = 0.5*second; 
#        intvl = np.arange(0,spk_rate.shape[0],cv_window)
#        #intvl = np.concatenate((intvl, spk1_rate.shape))
#        cv = np.zeros(len(intvl)-1)
#        for ind in range(len(intvl)-1):
#            cv[ind] = np.std(spk_rate[intvl[ind]:intvl[ind+1]])/np.mean(spk_rate[intvl[ind]:intvl[ind+1]])
#        
#        spk_popu = spk_popu[0:intvl[-1]]
#        
##        spk_aval_sum = np.zeros(int(spk_popu.shape[0]/aval_window), dtype=int)
##        for ind in range(int(spk_popu.shape[0]/aval_window)):
##            
##            spk_aval_sum[ind] = np.sum(spk_popu[ind*aval_window:(ind+1)*aval_window])
##        
##        threshold = 0.3*np.median(spk_aval_sum)
##            
#        aval_st = [None]*cv.shape[0]
#        
#        for i in range(len(aval_st)):
#            
#            spk_cv_window = spk_popu[i*cv_window:(i+1)*cv_window]
#            spk_aval_cv = np.zeros(int(spk_cv_window.shape[0]/aval_window), dtype=int)
#            aval_time = np.array([],int)
#            aval_size = np.array([],int)            
#            
#            for j in range(int(cv_window/aval_window)):
#                
#                spk_aval_cv[j] = np.sum(spk_cv_window[j*aval_window:(j+1)*aval_window])
#            
#            threshold = 0.1*np.median(spk_aval_cv)
#            
#            if spk_aval_cv[0]>threshold:
#               is_aval_pre = True
#               dura = aval_window
#               size = spk_aval_cv[0]
#            else:
#                is_aval_pre = False
#                dura = 0
#                size = 0
#                
#            for k in range(1, len(spk_aval_cv)):
#                if not(is_aval_pre) and spk_aval_cv[k]>threshold:
#                    dura += aval_window
#                    size += spk_aval_cv[k]
#                    is_aval_pre = True
#                elif is_aval_pre and spk_aval_cv[k]>threshold:
#                    dura += aval_window
#                    size += spk_aval_cv[i]
#                elif is_aval_pre and not(spk_aval_cv[k]>threshold):
#                    aval_time = np.concatenate((aval_time, [dura]))
#                    aval_size = np.concatenate((aval_size, [size]))
#                    dura = 0
#                    size = 0
#                    is_aval_pre = False
#                else: continue
#            
#            aval_st[i] = np.concatenate((aval_time.reshape(1,-1), aval_size.reshape(1,-1)),0)
#        
#        return aval_st, cv


    
    
#        spk_e1_spr = csr_matrix((np.ones(len(spike_e1.i),dtype=int), (spike_e1.i, (np.round(spike_e1.t/(dt))).astype(int))), shape = (3969, simu_step))
#        spk_popu = spk_e1_spr.sum(axis=0).A[0]
#        
#        rate_wind = 50*ms; dt = 0.1*ms
#        if int(round(rate_wind/dt))%2 ==0:
#            rate_kernel = np.ones(int(round(rate_wind/dt))+1)/((rate_wind+dt)/second)/Ne 
#        else:
#            rate_kernel = np.ones(int(round(rate_wind/dt)))/(rate_wind/second)/Ne 
#        
#        #rate_kernel = np.ones(int(round(rate_wind/dt)))/(rate_wind/second)/Ne 
#        spk1_rate = np.convolve(spk_popu, rate_kernel, mode='valid')
   
#%%
    
def sortandgroup_cv(cv, avalanche, cv_group_num):      
    arg = np.argsort(cv)
    sort_aval = [None]*(len(avalanche))
    for i in range(len(cv)):
        sort_aval[i] = avalanche[arg[i]]
        
    aval_group = [None]*(int(len(sort_aval)/cv_group_num))
    for i in range(len(aval_group)):
        group_i = np.array([[],[]])
        for j in range(cv_group_num):
            group_i = np.concatenate((group_i, sort_aval[i*cv_group_num+j]), axis=1)                
        
        aval_group[i] = group_i
        
    return aval_group
#%%    
def detect_avalanche(spike, row, col, dt=0.1*ms, discard_t=200*ms): # window = 0.1*ms,
    
    dt = dt/ms; discard_t = discard_t/ms
    #window = window/dt
    
    spk_e_t = np.round(((spike.t/ms)[spike.t[:]/ms>=discard_t]/dt)).astype(int) -round(discard_t/dt) # dispose the first 200ms data
    spk_e_ind = spike.i[spike.t[:]/ms>=discard_t]
    spk_e_aval = csr_matrix((np.ones(len(spk_e_t),dtype=int), (spk_e_ind, spk_e_t) ), shape = (row, col-round(discard_t/dt)))
    
    #nbins = np.round(spk_e_aval.shape[1]/window)
    
    
    #spk_popu, bins = np.histogram(spk_e_t, bins=nbins, range=(0, col-round(discard_t/dt)))
    
    spk_popu = spk_e_aval.sum(axis=0)
    spk_popu = spk_popu.A[0]
#        
    aval_size = np.array([])
    aval_time = np.array([])
    
    if spk_popu[0]>0: 
        fire_pre = True
        dura = 1
        size = spk_popu[0]
    else: 
        fire_pre = False
        dura = 0
        size = 0
    
    for i in range(1,len(spk_popu)):
        
        if not(fire_pre) and spk_popu[i]:
            dura += 1
            size += spk_popu[i]
            fire_pre = True
        elif fire_pre and spk_popu[i]:
            dura += 1
            size += spk_popu[i]
        elif fire_pre and not(spk_popu[i]):
            aval_time = np.concatenate((aval_time, [dura]))
            aval_size = np.concatenate((aval_size, [size]))
            dura = 0
            size = 0
            fire_pre = False
        else: continue
    
    return aval_time, aval_size
#%%    
def visualize_aval_distribution(aval, drange = None, returnfig = False, nbins = 34):
    
    #plt.figure()data//data.file
    freq, abins = np.histogram(aval, bins = nbins, range=drange)        
    #freq, abins, c = plt.hist(aval,bins=nbins)
    binback = abins[:]
    abins = abins + np.concatenate((0.5*np.diff(abins),[0]))
    #abins = abins + 0.5*(abins[1]-abins[0])
    abins = np.delete(abins, -1)       
    
    abins = abins[freq!=0]
    freq = freq[freq!=0]
    freq = freq/np.sum(freq)
    
    fit_power = LinearRegression()

    fit_power.fit(np.log(abins).reshape(-1,1),np.log(freq).reshape(-1,1))
    #
    xx = np.array([[np.log(min(abins))],[np.log(max(abins))]])
    #plt.plot(xxt, fit_power.predict(xxt), label='fit: tau_t=%.4f'%fit_power.coef_)
    
    #plt.figure()
    fig, ax = plt.subplots(1,1)
    ax.loglog(abins,freq, '*', label='data')
    ax.plot(np.exp(xx), np.exp(fit_power.predict(xx)), label='fit: tau=%.4f'%fit_power.coef_)
    ax.legend()
    #ax.set_xlabel('log(t)')
    ax.set_ylabel('frequency')
    
    coef = fit_power.coef_[0][0]
    
    if returnfig:
        return freq, abins, binback, coef, fig, ax
    else:
        return freq, abins, binback, coef
#%%    
def visualize_t_size_distribution(dtime, dsize, returnfig = False):
    
    t_ind = np.sort(np.array(list(set(dtime))))

    ts_each = np.array([[],[],[]])
    for t in t_ind:
        
        mean_s = np.mean(dsize[dtime == t])
        std_s = np.std(dsize[dtime == t])
        ts_each = np.concatenate((ts_each, [[t],[mean_s],[std_s]]), 1)
        
        
    fit_power_ts = LinearRegression()
    
    fit_power_ts.fit(np.log(ts_each[0,:]).reshape(-1,1),np.log(ts_each[1,:]).reshape(-1,1))
    #fit_power_ts.coef_
    xxts = np.array([[np.log(min(ts_each[0,:]))],[np.log(max(ts_each[0,:]))]])
#        plt.figure()
    #plt.plot(np.log(aval_time), np.log(aval_size), '*')
    fig, ax = plt.subplots(1,1)
    ax.loglog(ts_each[0,:], ts_each[1,:], '*', label='data')
    ax.plot(np.exp(xxts), np.exp(fit_power_ts.predict(xxts)), label='fit: tau_ts=%.4f'%fit_power_ts.coef_)
    ax.set_xlabel('log(t)')
    ax.set_ylabel('log(s)')
    ax.legend()
    
    if returnfig:
        return fig, ax, fit_power_ts.coef_[0][0]
    
    ## .............
#        fit_power_ts = LinearRegression()
#
#        fit_power_ts.fit(np.log(dtime).reshape(-1,1),np.log(dsize).reshape(-1,1))
#        fit_power_ts.coef_
#        
#        #plt.figure()
#        fig, ax = plt.subplots(1,1)
#        #plt.plot(np.log(dtime), np.log(dsize), '*')
#        
#        ax.loglog(dtime, dsize, '*', label='data')
#        xxts = np.array([[np.log(min(dtime))],[np.log(max(dtime))]])
#        ax.loglog(np.exp(xxts), np.exp(fit_power_ts.predict(xxts)), label='fit: tau=%.4f'%fit_power_ts.coef_)
#        ax.set_xlabel('log(t)')
#        ax.set_ylabel('log(s)')
#        ax.legend()
#        #plt.savefig('t_size.png')
#        #plt.savefig('t_size%s.png' %sys.argv[1])
#        if returnfig:
#            return fig, ax
#%%        
def get_spike_rate(spike, start_time, end_time, indiv_rate = False, popu_rate = False, \
                   sample_interval = 1*ms, n_neuron = 3969, window = 10*ms, dt = 0.1*ms):
    '''
    spike: an object that contains time index and neuron index for spikes
    start_time: the time in recorded spike data to start to calculate the firing rate 
    end_time: the time in recorded spike data to stop the calculation of firing rate 
    indiv_rate: whether to output the number of spikes for each neurons in 'window'
    popu_rate: whether to output the population firing rate
    sample_interval: time interval between each firing rate calculation
    n_neuron: total number of neurons
    window: number of spikes in 'window'/'window' = firing rate
    dt: simulation time step
    '''
    #window = 10*ms #steps, 5ms
    sample_interval = int(sample_interval/dt)
    start_time = int(np.round(start_time/dt))
    end_time = int(np.round(end_time/dt))
    window_step = int(np.round(window/dt))
    #total_step = end_time - start_time #int(np.round(ani_time/dt))
    
    sample_t = np.arange(start_time, end_time-window_step+1, sample_interval)
    #spk = np.zeros([n_neuron, total_step-window_step+1], dtype=np.int8)
    spk = np.zeros([n_neuron, len(sample_t)], dtype=np.int8)
    t_spk = (np.round(spike.t/dt)).astype(int)

    #for stp in range(spk.shape[1]):
    for t in range(len(sample_t)):    
        neu, counts = np.unique(spike.i[(t_spk >= (sample_t[t])) & (t_spk < (window_step+sample_t[t]))], return_counts=True)
        spk[:, t][neu] += counts
    if indiv_rate:
        spk = spk.reshape(int(n_neuron**0.5),int(n_neuron**0.5),spk.shape[1])
        pop_rate = (spk.sum(0).sum(0))/(window/(1*second))/n_neuron
    else:
        pop_rate = spk.sum(0)/(window/(1*second))/n_neuron
    
    if indiv_rate and (not popu_rate): return spk
    elif (not indiv_rate) and popu_rate: return pop_rate
    elif indiv_rate and popu_rate: return spk, pop_rate
    else: return 0
#        fig, ax1= plt.subplots(1,1)
#        
#        
#        value1=ax1.matshow(spk[:,:,0], cmap=plt.cm.get_cmap('viridis', 5))
#        ax1.axis('off')
#        #cmap=plt.cm.get_cmap('binary', 3)
#        
#        def updatev(iii):
#            value1.set_array(spk[:,:,iii])
#            return value1
#        #
#        cbaxes = fig.add_axes([0.1, 0.06, 0.8, 0.03]) 
#        cb=fig.colorbar(value1, cax = cbaxes, orientation='horizontal',ticks=[0,1,2,3,4]) 
#        value1.set_clim(vmin=0, vmax=4)
#        
#        #fig.suptitle('ie_ratio:%s\ncentre of pattern and number of spikes per 10ms\n ' %ie_ratio)
#        ani=animation.FuncAnimation(fig, updatev, frames=spk.shape[1], interval=0.1)  
#        ani.show()
#%%
'''
spk: N*N*t array; a 3D array containing the number of spikes for each neuron in a sliding short period of time window.
slide_interval: the time steps interval between two successive centre-of-mass calculation.
jump_interval: the time steps interval between two successive centre-of-mass jump distance calculation
'''
def get_centre_mass(spk, slide_interval, jump_interval, dt=0.1*ms):
    
#        slide_interval = round(slide_interval/dt)
#        jump_interval = round(jump_interval/dt)
    
    len_hori = spk.shape[1]; len_vert = spk.shape[0];
    x_hori = np.cos(np.arange(len_hori)/len_hori*2*np.pi).reshape([-1,1])
    y_hori = np.sin(np.arange(len_hori)/len_hori*2*np.pi).reshape([-1,1])
    xy_hori = np.concatenate((x_hori,y_hori),axis=1)
    
    x_vert = np.cos(np.arange(len_vert)/len_vert*2*np.pi).reshape([-1,1])
    y_vert = np.sin(np.arange(len_vert)/len_vert*2*np.pi).reshape([-1,1])
    xy_vert = np.concatenate((x_vert,y_vert),axis=1)
    
    slide_ary = np.arange(0,spk.shape[2],slide_interval)
    centre_ind = np.zeros([len(slide_ary), 2], dtype=int)
    #jump_size = np.zeros([len(slide_ary), 2])
    centre = np.zeros([len(slide_ary), 2], dtype=float)
    for ind in range(len(slide_ary)):
        sum_hori = np.sum(spk[:,:,slide_ary[ind]], axis=0)
        sum_vert = np.sum(spk[:,:,slide_ary[ind]], axis=1)
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
        centre[ind,1] = np.angle(np.array([ctr_hori[0] + 1j*ctr_hori[1]]))[0]
        centre[ind,0] = np.angle(np.array([ctr_vert[0] + 1j*ctr_vert[1]]))[0]
        ind_vert = int(self.wrapTo2Pi(np.array([centre[ind,0]]))*len_vert/(2*np.pi))
        ind_hori = int(self.wrapTo2Pi(np.array([centre[ind,1]]))*len_hori/(2*np.pi))
        centre_ind[ind] = [ind_vert, ind_hori]
    jump_size = centre[:len(centre)-int(jump_interval/slide_interval), :] - centre[int(jump_interval/slide_interval):, :]
    jump_size = self.wrapToPi(jump_size)
    jump_size[:,0] = jump_size[:,0]*len_vert/(2*np.pi)
    jump_size[:,1] = jump_size[:,1]*len_hori/(2*np.pi)
    jump_dist = np.sqrt(np.sum(jump_size**2,1))
    
    return centre_ind, jump_size, jump_dist
    
#    def get_centre_mass2(spk, slide_interval, jump_interval=15*ms, dt=0.1*ms):
#        
#        slide_interval = round(slide_interval/dt)
#        jump_interval = round(jump_interval/dt)
#        
#        len_hori = spk.shape[1]; len_vert = spk.shape[0]
#        
#        cordi_hori = 1j*np.arange(len_hori)/len_hori*2*np.pi
#        cordi_vert = 1j*np.arange(len_vert)/len_vert*2*np.pi
#        
#        cordi_hori, cordi_vert = np.meshgrid(cordi_hori, cordi_vert)
#        
#        slide_ary = np.arange(0,spk.shape[2],slide_interval)
#        centre = np.zeros([len(slide_ary),2])
#        for i in range(len(slide_ary)):
#            centre[i,1] = np.angle(np.sum(np.exp(cordi_hori)*spk[:,:,slide_ary[i]]))
#            centre[i,0] = np.angle(np.sum(np.exp(cordi_vert)*spk[:,:,slide_ary[i]]))
#            for j in range(2):
#                if centre[i,j] == 0:
#                    centre[i,j] = centre[max(0,i-1), j]
#        
#        jump_size = self.wrapToPi(centre[:len(slide_ary)-int(jump_interval/slide_interval),:] - centre[int(jump_interval/slide_interval):,:])
#        #jump_dist = np.sum(jump_size*jump_size,1)
#        jump_size[:,0] = jump_size[:,0]*spk.shape[0]/(2*np.pi)
#        jump_size[:,1] = jump_size[:,1]*spk.shape[1]/(2*np.pi)
#        jump_dist = np.sqrt(np.sum(jump_size**2,1))
#        
#        centre = self.wrapTo2Pi(centre)
#        centre[:,0] = centre[:,0]*spk.shape[0]/(2*np.pi)
#        centre[:,1] = centre[:,1]*spk.shape[1]/(2*np.pi)
#        #centre = np.round(centre)
#        centre = centre.astype(int)
#        
#        return centre, jump_size, jump_dist

def wrapTo2Pi(angle):
    positiveinput = (angle > 0)
    angle = np.mod(angle, 2*np.pi)
    angle[(angle==0) & positiveinput] = 2*np.pi
    return angle


def wrapToPi(angle):
    select = (angle < -np.pi) | (angle > np.pi) 
    angle[select] = self.wrapTo2Pi(angle[select] + np.pi) - np.pi
    return angle    


def overlap_centreandspike(centre, img, show_trajectory = False):
    
    #centre = centre_mass(spk)
    #centre = centre)
    if not show_trajectory:
        for ind in range(img.shape[2]):
            img[:,:,ind][centre[ind,0], centre[ind,1]] = -1 # make the centre of mass on the plot show different colour        
        return img
    else:
        for ind in range(img.shape[2]):
            img[centre[ind,0],centre[ind,1],ind:] = -1
        return img
#%%
'''
draw the trajectory of the centre of mass of pattern.
axx: the axis object of subplots
centre_ind: an N*2 array, the coordinate of the centre of mass
centre_ind[:,1]:x axis coordinate; centre_ind[:,0]:y axis coordinate
'''
def plot_traj(axx, centre_ind):
#plt.figure()
    axx.plot(centre_ind[0,1], centre_ind[0,0],'og',label='start')
    for ind in range(len(centre_ind)-1):
        if abs(centre_ind[ind,0]-centre_ind[ind+1,0])>31 and abs(centre_ind[ind,1]-centre_ind[ind+1,1])>31:
            if centre_ind[ind,0] > centre_ind[ind+1,0]:
                if centre_ind[ind,1] > centre_ind[ind+1,1]:
                    axx.plot([centre_ind[ind,1]-63, centre_ind[ind+1,1]], [centre_ind[ind,0]-63, centre_ind[ind+1,0]],'b')
                    axx.plot([centre_ind[ind,1], centre_ind[ind+1,1]+63], [centre_ind[ind,0], centre_ind[ind+1,0]+63],'b')
                else:
                    axx.plot([centre_ind[ind+1,1], centre_ind[ind,1]+63], [centre_ind[ind+1,0], centre_ind[ind,0]-63],'b')
                    axx.plot([centre_ind[ind,1], centre_ind[ind+1,1]-63], [centre_ind[ind,0], centre_ind[ind+1,0]+63],'b')
            else:
                if centre_ind[ind,1] > centre_ind[ind+1,1]:
                    axx.plot([centre_ind[ind,1]-63, centre_ind[ind+1,1]], [centre_ind[ind,0]+63,centre_ind[ind+1,0]],'b')
                    axx.plot([centre_ind[ind,1], centre_ind[ind+1,1]+63], [centre_ind[ind,0],centre_ind[ind+1,0]-63],'b')
                else:
                    axx.plot([centre_ind[ind+1,1],centre_ind[ind,1]+63], [centre_ind[ind+1,0],centre_ind[ind,0]+63],'b')
                    axx.plot([centre_ind[ind+1,1]-63,centre_ind[ind,1]], [centre_ind[ind+1,0]-63,centre_ind[ind,0]],'b')
        elif abs(centre_ind[ind,0]-centre_ind[ind+1,0])>31:
            if centre_ind[ind,0]-centre_ind[ind+1,0]>0:
                axx.plot([centre_ind[ind+1,1],centre_ind[ind,1]], [centre_ind[ind+1,0],centre_ind[ind,0]-63],'b')
                axx.plot([centre_ind[ind+1,1],centre_ind[ind,1]], [centre_ind[ind+1,0]+63,centre_ind[ind,0]],'b')
            else:
                axx.plot([centre_ind[ind+1,1],centre_ind[ind,1]], [centre_ind[ind+1,0],centre_ind[ind,0]+63],'b')
                axx.plot([centre_ind[ind+1,1],centre_ind[ind,1]], [centre_ind[ind+1,0]-63,centre_ind[ind,0]],'b')
        
        elif abs(centre_ind[ind,1]-centre_ind[ind+1,1])>31:
            if centre_ind[ind,1]-centre_ind[ind+1,1]>0:
                axx.plot([centre_ind[ind,1]-63,centre_ind[ind+1,1]], [centre_ind[ind,0],centre_ind[ind+1,0]],'b')
                axx.plot([centre_ind[ind,1],centre_ind[ind+1,1]+63], [centre_ind[ind,0],centre_ind[ind+1,0]],'b')
            else:
                axx.plot([centre_ind[ind+1,1]-63,centre_ind[ind,1]], [centre_ind[ind+1,0],centre_ind[ind,0]],'b')
                axx.plot([centre_ind[ind+1,1],centre_ind[ind,1]+63], [centre_ind[ind+1,0],centre_ind[ind,0]],'b')
        else:
            axx.plot([centre_ind[ind,1],centre_ind[ind+1,1]], [centre_ind[ind,0],centre_ind[ind+1,0]],'b')
    axx.plot(centre_ind[ind,1], centre_ind[ind,0],'or',label='end')
    axx.legend()
#%%
def fourier_transform(spk, simulation_time, dt=0.1*ms, discard_period = 100*ms):
    
    t_ind = np.round(spk.t/dt).astype(int);
    i_ind = spk.i[t_ind > int(discard_period/dt)]
    t_ind = t_ind[t_ind > int(discard_period/dt)]
    csr_matrix((np.ones(len(t_ind), dtype=int), (spk.i, t_ind)), shape = (row, col))
#%%
'''
#with open ('/import/headnode1/shni2598/brian2/data/yifan_criticality/data1.file', 'rb') as file:
#    data1 = pickle.load(file)
with open ('data.file', 'rb') as file:
    data = pickle.load(file)
#data1.spike_e['i']
#bb=np.round(((spike_e.t/ms)[spike_e.t[:]/ms>=discard_t]/dt)).astype(int) -round(discard_t/dt)
#%%
class classdata:
    
    pass
#%%
class spk_data:
    def __init__(spk_dict):
        for key in spk_dict:
            setattr(key, spk_dict[key])
        
#%%
a=classdata()
a.d1 = 1
a.d2 = 2   
a.d2.d22 = 2    
#%%
Ne = 63*63; sim_t = 10000
anly = analysis()
spike_e = spk_data(data.spike_e)
cls_size, cls_time = anly.detect_avalanche(spike_e, Ne, 10000)
fig, ax = anly.visualize_aval_distribution(cls_time, returnfig=True)
#ax.set_xlabel('log')
#plt.show(fig)

#%%
plt.figure()        
freq_time, bins_time, c_time = plt.hist(cls_time,bins=34)
bins_new_time = bins_time + 0.5*(bins_time[1]-bins_time[0])
bins_new_time = np.delete(bins_new_time, -1)       
#%%
bins_new_time = bins_new_time[freq_time!=0]
freq_time = freq_time[freq_time!=0]
freq_time = freq_time/np.sum(freq_time)

#plt.figure()
#plt.plot(np.log(bins_new_time), np.log(freq_time), '*', label='avalanche_duration')

#%%
with open ('/import/headnode1/shni2598/brian2/data/yifan_criticality2/data1.file', 'rb') as file:
    data1 = pickle.load(file)
#%%
N = 63*63; sim_t = 500000
anly = analysis()
spike_e1 = spk_data(data1.spike_e)
size1, time1 = anly.detect_avalanche(spike_e1, N, sim_t)
fig, ax = anly.visualize_aval_distribution(time1, returnfig=True, nbins='auto')
fig, ax = anly.visualize_aval_distribution(size1, returnfig=True, nbins='auto')
#ax.set_xlabel('log')
#plt.show(fig)
#%%
fig1, ax1 = anly.visualize_t_size_distribution(time1, size1)
'''







