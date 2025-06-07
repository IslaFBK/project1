#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 16:53:20 2020

@author: shni2598
"""

import brian2.numpy_ as np
#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt
import connection as cn
#import warnings
from scipy import sparse
from brian2.only import *
import time
import mydata
#import firing_rate_analysis
import os
import datetime
import poisson_stimuli as psti
import pre_process_sc
import preprocess_2area
import build_two_areas
import sys
import pickle
#%%
'''
choose approximatey 1/4 neurons with constant space
'''
N_side = 63
lattice_bool = np.zeros([N_side, N_side],bool)
lattice_bool[:,0::2] = True
lattice_bool[1::2,:] = False
N_choosen = np.arange(N_side*N_side)[lattice_bool.reshape(-1)]


#%%
step = 2; N_x_all = 63; N_y_all = 63;
x_sec = np.arange(0, N_x_all, step)
y_sec = np.arange(0, N_y_all, step)
x_sec[-1] = N_x_all
y_sec[-1] = N_y_all

lattice_bool = np.zeros([N_y_all, N_x_all],bool)
#%%
selection_p = 1/4
for j in range(len(y_sec)-1):
    for i in range(len(x_sec)-1):
        section_size = (x_sec[i+1] - x_sec[i])*(y_sec[j+1] - y_sec[j])
        select_num = section_size*selection_p
        p = 1 - (select_num - int(select_num))
        if np.random.rand() < p:
            select_num = int(select_num)
        else:
            select_num = int(select_num) + 1
        tmp = np.zeros(section_size, bool)
        tmp[np.random.choice(np.arange(section_size),select_num,replace=False)] = True
        lattice_bool[y_sec[j]:y_sec[j+1], x_sec[i]:x_sec[i+1]] = tmp.reshape(y_sec[j+1] - y_sec[j], x_sec[i+1] - x_sec[i])
        
#%%
plt.matshow(lattice_bool)
#%%
np.arange(N_y_all*N_x_all)[lattice_bool.reshape(-1)].shape
#%%
N_side = 63
lattice_bool = np.zeros([N_side, N_side],bool)
lattice_bool.reshape(-1)[ijwd_inter.inter_e_neuron_1] = True
lattice_bool.reshape(N_side, N_side)
plt.matshow(lattice_bool)
N_side = 63
lattice_bool = np.zeros([N_side, N_side],bool)
lattice_bool.reshape(-1)[ijwd_inter.inter_e_neuron_2] = True
lattice_bool.reshape(N_side, N_side)
plt.matshow(lattice_bool)

#%%


#%%
prefs.codegen.target = 'numpy'




#%%
n_neuron = 3969

stim_dura = 200; separate_dura = np.array([300,500])
dt_stim = 10
stim_dura //= dt_stim;
separate_dura //= dt_stim
stim_num = 4
stim_amp_scale = np.ones(4)
stim_amp_scale[2:] *= 2
stim_amp_base = 200

sepa = np.random.rand(stim_num)*(separate_dura[1]-separate_dura[0]) + separate_dura[0]
#sepa /= dt_stim
sepa = sepa.astype(int)

stim_rate = psti.input_spkrate(maxrate = [stim_amp_base], sig=[6], position=[[0, 0]])#*Hz


stim = np.zeros([int(round(stim_num*stim_dura+sepa.sum())), n_neuron])
for i in range(stim_num):
    stim[i*stim_dura + sepa[:i].sum(): i*stim_dura + sepa[:i].sum()+stim_dura, :] = stim_amp_scale[i] * stim_rate
#%%
plt.figure()
plt.imshow(stim, aspect='auto')
#%%
start_scope()

stim *= 0.1

stimulus = TimedArray(stim*Hz, dt=10*ms)
p_input =  PoissonGroup(3969, rates='''stimulus(t,i)''')

spk_m = SpikeMonitor(p_input, record = True)

run(2500*ms)
#%%
plt.figure()
plt.plot(spk_m.t/ms, spk_m.i, '|')

#%%
cost_n = np.zeros(10)
for i in range(10):
    start_scope()
    prefs.codegen.target = 'numpy'
    
    t_n = NeuronGroup(500, 'rates : Hz', threshold='rand()<rates*dt')
    t_n.rates = 500*Hz
    
    tic = time.perf_counter()
    
    run(2500*ms)
    cost_n[i] = time.perf_counter() - tic
    print('total time elapsed:',time.perf_counter() - tic)
#%%
cost_p = np.zeros(10)
for i in range(10):
    start_scope()
    prefs.codegen.target = 'numpy'
    
    t_p = PoissonGroup(500, 500*Hz)
    
    tic = time.perf_counter()
    
    run(2500*ms)
    cost_p[i] = time.perf_counter() - tic
    print('total time elapsed:',time.perf_counter() - tic)

#%%
start_scope()
posi_stim = NeuronGroup(3969, \
                        '''rates =  bkg_rates + stim*scale(t) : Hz
                        bkg_rates : Hz
                        stim : Hz
                        ''', threshold='rand()<rates*dt')

posi_stim.bkg_rates = 0*Hz
posi_stim.stim = psti.input_spkrate(maxrate = [stim_amp_base], sig=[6], position=[[0, 0]])*0.1*Hz

init = np.zeros(100)
scale_stim = np.concatenate((init, scale_stim))
scale = TimedArray(scale_stim, dt=10*ms)
spk_m = SpikeMonitor(posi_stim, record = True)

run(3500*ms)

#%%
plt.figure()
plt.plot(spk_m.t/ms, spk_m.i, '|')

#%%
n_neuron = 3969

stim_dura = 200; separate_dura = np.array([300,500])
dt_stim = 10
stim_dura //= dt_stim;
separate_dura //= dt_stim
stim_num = 4
stim_amp_scale = np.ones(4)
stim_amp_scale[2:] *= 2
stim_amp_base = 200

sepa = np.random.rand(stim_num)*(separate_dura[1]-separate_dura[0]) + separate_dura[0]
#sepa /= dt_stim
sepa = sepa.astype(int)

stim_rate = psti.input_spkrate(maxrate = [stim_amp_base], sig=[6], position=[[0, 0]])#*Hz


scale_stim = np.zeros([int(round(stim_num*stim_dura+sepa.sum()))])#, n_neuron])
for i in range(stim_num):
    scale_stim[i*stim_dura + sepa[:i].sum(): i*stim_dura + sepa[:i].sum()+stim_dura] = stim_amp_scale[i] #* stim_rate
#%%
plt.figure()
plt.plot(scale_stim)#, aspect='auto')
#%%
scale = TimedArray(scale_stim, dt=10*ms)
p_input =  PoissonGroup(3969, rates='''stimulus(t,i)''')
#%%
class get_stim_scale:
    
    def __init__(self):
        self.seed = 10
        self.stim_dura = 200 # ms
        self.separate_dura = np.array([300,500]) # ms
        self.dt_stim = 10 # ms
        self.stim_amp_scale = None
        
    def get_scale(self):
        stim_num = self.stim_amp_scale.shape[0]
        stim_dura = self.stim_dura//self.dt_stim;
        separate_dura = self.separate_dura//self.dt_stim
        np.random.seed(self.seed)
        sepa = np.random.rand(stim_num)*(separate_dura[1]-separate_dura[0]) + separate_dura[0]
        sepa = sepa.astype(int)
        self.scale_stim = np.zeros([int(round(stim_num*stim_dura+sepa.sum()))])#, n_neuron])
        self.stim_on = np.zeros([stim_num, 2], int) 
        for i in range(stim_num):
            self.scale_stim[i*stim_dura + sepa[:i].sum(): i*stim_dura + sepa[:i].sum()+stim_dura] = self.stim_amp_scale[i] #* stim_rate
            self.stim_on[i] = np.array([i*stim_dura + sepa[:i].sum(), i*stim_dura + sepa[:i].sum()+stim_dura]) * self.dt_stim
                
        pass

#%%
stim_scale_cls = get_stim_scale()

stim_scale_cls.stim_amp_scale = np.ones(4)
stim_scale_cls.stim_amp_scale[2:] *= 2
stim_scale_cls.get_scale()
#%%
start_scope()
posi_stim = NeuronGroup(3969, \
                        '''rates =  bkg_rates + stim*scale(t) : Hz
                        bkg_rates : Hz
                        stim : Hz
                        ''', threshold='rand()<rates*dt')

posi_stim.bkg_rates = 0*Hz
stim_amp_base = 20
posi_stim.stim = psti.input_spkrate(maxrate = [stim_amp_base], sig=[6], position=[[0, 0]])*0.1*Hz

init = np.zeros(100)
scale_stim = np.concatenate((init, stim_scale_cls.scale_stim))
scale = TimedArray(scale_stim, dt=10*ms)
spk_m = SpikeMonitor(posi_stim, record = True)

st_m = StateMonitor(posi_stim, ('stim','rates'), record=[1984])
run(3900*ms)

#%%
plt.figure()
plt.plot(spk_m.t/ms, spk_m.i, '|')        
#%%
plt.figure()
plt.plot(stim_scale_cls.scale_stim)#, aspect='auto')    
#%%
plt.figure()
plt.plot(st_m.t/ms, st_m.stim[0], '-')        
#%%
plt.figure()
plt.plot(st_m.t/ms, st_m.rates[0], '-')        
    
    
    
    
    