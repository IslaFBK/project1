#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 23:23:24 2021

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
from analysis import mydata
#import firing_rate_analysis
import os
import datetime
from connection import poisson_stimuli as psti
from connection import pre_process_sc
from connection import preprocess_2area
from connection import build_two_areas
from connection import get_stim_scale
from connection import adapt_gaussian
import sys
import pickle

#%%
prefs.codegen.target = 'cython'

# dir_cache = './cache/cache%s' %sys.argv[1]
dir_cache = './cache'
prefs.codegen.runtime.cython.cache_dir = dir_cache
prefs.codegen.max_cache_dir_size = 120
#%%
data_dir = 'raw_data/'
if not os.path.exists(data_dir):
    try: os.makedirs(data_dir)
    except FileExistsError:
        pass
#%%    
#sys_argv = int(sys.argv[1]) # define which (the index of) parameter you want to use in this simulation; get the index from script input argument
sys_argv = 0 # define which (the index of) parameter you want to use in this simulation

#%%
record_LFP = True

loop_num = -1

repeat = 20

tau_k_ = 60 # ms
chg_adapt_range = 7 
w_extnl_ = 10 # nS
tau_s_r_ = 1 # ms

# swap parameters
for ie_r_i1 in np.arange(1.52,1.521,0.02):#[1.40]:
    for ie_r_i2 in [1.06]:#np.arange(1.04, 1.071, 0.01):
        for delta_gk_1 in [3.5]:#np.linspace(7,15,9)]]:
            for delta_gk_2 in [16]:
                for new_delta_gk_2 in [0.5]:#np.arange(0.5, 2.1, 0.5):#[2]:#np.arange(0.15,0.31,0.05)*delta_gk_:#np.linspace(1,5,5):
                    for tau_s_di_ in np.linspace(4.4,4.4,1):
                        for tau_s_de_ in np.linspace(5.,5.,1):
                            for scale_w_12_e in np.arange(2.6, 2.61, 0.2):#np.arange(0.8,1.21,0.05):[2.5]
                                for scale_w_12_i in scale_w_12_e*np.array([1]):#np.array([0.8,0.9,1.0,1.1]):#np.arange(0.9,1.11,0.1):#np.arange(0.8,1.21,0.05):#[scale_w_12_e]:#0.2*np.arange(0.8,1.21,0.05):
                                    for scale_w_21_e in np.arange(0.30, 0.31, 0.02):#np.arange(0.8,1.21,0.05):#[1]:#np.arange(0.8,1.21,0.05):[0.3]
                                        for scale_w_21_i in scale_w_21_e*np.array([1]):#np.arange(0.9,1.11,0.1):#np.arange(0.8,1.21,0.05):#[1]:#[scale_w_21_e]:#0.35*np.arange(0.8,1.21,0.05):
                                            for tau_p_d_e1_e2 in [5]:
                                                for tau_p_d_e1_i2 in [5]:
                                                    for tau_p_d_e2_e1 in [15]:#np.arange(7,14.1,1):
                                                        for tau_p_d_e2_i1 in [tau_p_d_e2_e1]:#[15]:#np.arange(7,14.1,1):#[10]:
                                                            for peak_p_e1_e2 in [0.4]:#np.arange(0.3,0.451,0.05):
                                                                for peak_p_e1_i2 in [0.4]:#np.arange(0.25,0.401,0.05):
                                                                    for peak_p_e2_e1 in [0.13]:#[0.1,0.12,0.14]:#np.arange(0.10):#[0.15]: # 0.18
                                                                        for peak_p_e2_i1 in [0.1]:#peak_p_e2_e1*np.repeat(np.arange(0.9,1.1,0.03),repeat):#:[0.15]*repeat: # 0.18
                                                                            for rp in [None]*repeat:
                                                                                loop_num += 1
                                                                                if loop_num == sys_argv:
                                                                                    print('loop_num:',loop_num)
                                                                                    break
                                                                                else: continue
                                                                                break
                                                                            else: continue
                                                                            break
                                                                        else: continue
                                                                        break
                                                                    else: continue
                                                                    break
                                                                else: continue
                                                                break
                                                            else: continue                    
                                                            break
                                                        else: continue
                                                        break
                                                    else: continue
                                                    break
                                                else: continue
                                                break
                                            else: continue
                                            break
                                        else: continue
                                        break
                                    else: continue
                                    break    
                                else: continue
                                break
                            else: continue
                            break
                        else: continue
                        break
                    else: continue
                    break
                else: continue
                break
            else: continue
            break
        else: continue
        break
    else: continue
    break


if loop_num != sys_argv: sys.exit("Error: wrong PBS_array_id")                    

#%%
def find_w_e(w_i, num_i, num_e, ie_ratio):
    return (w_i*num_i)/num_e/ie_ratio
               
def find_w_i(w_e, num_e, num_i, ie_ratio):
    return (w_e*num_e)*ie_ratio/num_i

w_ie_1 = 115; w_ii_1 = 100
ie_r_e = 2.76*6.5/5.8; ie_r_e1 = 1.56
ie_r_i = 2.450*6.5/5.8; #ie_r_i1 = 1.40

ijwd1 = pre_process_sc.get_ijwd()
ijwd1.Ne = 64*64; ijwd1.Ni = 32*32
ijwd1.width = 64#79
#ijwd1.w_ee_mean *= 2; ijwd1.w_ei_mean *= 2; ijwd1.w_ie_mean *= 2; ijwd1.w_ii_mean *= 2;
#scale_d_p = 1 #np.sqrt(8/5) 
ijwd1.decay_p_ee = 7 #* scale_d_p_ee#scale_d_p # decay constant of e to e connection probability as distance increases
ijwd1.decay_p_ei = 9 #* scale_d_p_ei# scale_d_p # decay constaw_ie_ in [115]nt of e to i connection probability as distance increases
ijwd1.decay_p_ie = 19 #* scale_d_p_ie#scale_d_p_i#* scale_d_p # decay constant of i to e connection probability as distance increases
ijwd1.decay_p_ii = 19 #* scale_d_p_ii#* scale_d_p # decay constant of i to i connection probability as distance increases
ijwd1.delay = [0.5,2.5]

num_ee = 240; num_ei = 400; num_ie = 150; num_ii = 230

ijwd1.mean_SynNumIn_ee = num_ee     ; # p = 0.08
ijwd1.mean_SynNumIn_ei = num_ei #* 8/5     ; # p = 0.125
ijwd1.mean_SynNumIn_ie = num_ie  #scale_d_p_i    ; # p = 0.2
ijwd1.mean_SynNumIn_ii = num_ii# 221 * scale_d_p_i#* 8/5     ; # p = 0.25

ijwd1.w_ee_mean = find_w_e(w_ie_1, num_ie, num_ee, ie_r_e*ie_r_e1)#find_w_e(w_ie_, num_ie, num_ee, ie_r_e)
ijwd1.w_ei_mean = find_w_e(w_ii_1, num_ii, num_ei, ie_r_i*ie_r_i1)
ijwd1.w_ie_mean = w_ie_1 #find_w_i(w_ee_, num_ee, num_ie, ie_r_e*ie_r_e1)#w_ie_ #find_w_i(w_ee_, num_ee, num_ie, ie_r_e)
ijwd1.w_ii_mean = w_ii_1 #find_w_i(w_ei_, num_ei, num_ii, ie_r_i)#w_ii_
        
ijwd1.generate_ijw()
#ijwd1.generate_d_rand()
ijwd1.generate_d_dist()

# ijwd1.w_ee *= scale_ee_1#tau_s_de_scale_d_p_i
# ijwd1.w_ei *= scale_ei_1 #tau_s_de_ #5*nS
# ijwd1.w_ie *= scale_ie_1#tau_s_di_#25*nS
# ijwd1.w_ii *= scale_ii_1#tau_s_di_#
param_a1 = {**ijwd1.__dict__}



del param_a1['i_ee'], param_a1['j_ee'], param_a1['w_ee'], param_a1['d_ee'], param_a1['dist_ee']  
del param_a1['i_ei'], param_a1['j_ei'], param_a1['w_ei'], param_a1['d_ei'], param_a1['dist_ei'] 
del param_a1['i_ie'], param_a1['j_ie'], param_a1['w_ie'], param_a1['d_ie'], param_a1['dist_ie'] 
del param_a1['i_ii'], param_a1['j_ii'], param_a1['w_ii'], param_a1['d_ii'], param_a1['dist_ii']

#w_ee_2 = 4*5.8; w_ei_2 = 5*5.8
w_ie_2 = 115; w_ii_2 = 140
ie_r_e = 2.76*6.5/5.8; ie_r_e2 = 1.06 #1.03
ie_r_i = 2.450*6.5/5.8; #ie_r_i2 = 1.04

ijwd2 = pre_process_sc.get_ijwd()
ijwd2.Ne = 64*64; ijwd2.Ni = 32*32
ijwd2.width = 64#79
#ijwd2.w_ee_mean *= 2; ijwd2.w_ei_mean *= 2; ijwd2.w_ie_mean *= 2; ijwd2.w_ii_mean *= 2;
#scale_d_p = 1 #np.sqrt(8/5) 
ijwd2.decay_p_ee = 7#8*0.8 #* scale_d_p_ee#scale_d_p # decay constant of e to e connection probability as distance increases
ijwd2.decay_p_ei = 9#10*0.82 #* scale_d_p_ei# scale_d_p # decay constant of e to i connection probability as distance increases
ijwd2.decay_p_ie = 19#20*0.86 #* scale_d_p_ie#scale_d_p_i#* scale_d_p # decay constant of i to e connection probability as distance increases
ijwd2.decay_p_ii = 19#20*0.86 #* scale_d_p_ii#* scale_d_p # decay constant of i to i connection probability as distance increases
ijwd2.delay = [0.5,2.5]

num_ee = 240; num_ei = 400; num_ie = 150; num_ii = 230

ijwd2.mean_SynNumIn_ee = num_ee#num_ee     ; # p = 0.08
ijwd2.mean_SynNumIn_ei = num_ei#num_ei #* 8/5     ; # p = 0.125
ijwd2.mean_SynNumIn_ie = num_ie#num_ie  #scale_d_p_i    ; # p = 0.2
ijwd2.mean_SynNumIn_ii = num_ii#num_ii# 221 * scale_d_p_i#* 8/5     ; # p = 0.25

# ijwd2.w_ee_mean = w_ee_2#find_w_e(w_ie_, num_ie, num_ee, ie_r_e)#find_w_e(w_ie_, num_ie, num_ee, ie_r_e)
# ijwd2.w_ei_mean = w_ei_2#find_w_e(w_ii_, num_ii, num_ei, ie_r_i)
# ijwd2.w_ie_mean = find_w_i(w_ee_2, num_ee, num_ie, ie_r_e*ie_r_e2)#w_ie_ #find_w_i(w_ee_, num_ee, num_ie, ie_r_e)
# ijwd2.w_ii_mean = find_w_i(w_ei_2, num_ei, num_ii, ie_r_i*ie_r_i2)#w_ii_

ijwd2.w_ee_mean = find_w_e(w_ie_2, num_ie, num_ee, ie_r_e*ie_r_e2)#find_w_e(w_ie_, num_ie, num_ee, ie_r_e)
ijwd2.w_ei_mean = find_w_e(w_ii_2, num_ii, num_ei, ie_r_i*ie_r_i2)
ijwd2.w_ie_mean = w_ie_2 #find_w_i(w_ee_, num_ee, num_ie, ie_r_e*ie_r_e1)#w_ie_ #find_w_i(w_ee_, num_ee, num_ie, ie_r_e)
ijwd2.w_ii_mean = w_ii_2 #find_w_i(w_ei_, num_ei, num_ii, ie_r_i)#w_ii_

   
ijwd2.generate_ijw()
#ijwd2.generate_d_rand()
ijwd2.generate_d_dist()

param_a2 = {**ijwd2.__dict__}

del param_a2['i_ee'], param_a2['j_ee'], param_a2['w_ee'], param_a2['d_ee'], param_a2['dist_ee'] 
del param_a2['i_ei'], param_a2['j_ei'], param_a2['w_ei'], param_a2['d_ei'], param_a2['dist_ei']
del param_a2['i_ie'], param_a2['j_ie'], param_a2['w_ie'], param_a2['d_ie'], param_a2['dist_ie'] 
del param_a2['i_ii'], param_a2['j_ii'], param_a2['w_ii'], param_a2['d_ii'], param_a2['dist_ii']


ijwd_inter = preprocess_2area.get_ijwd_2()

ijwd_inter.Ne1 = 64*64; ijwd_inter.Ne2 = 64*64; 
ijwd_inter.width1 = 64; ijwd_inter.width2 = 64; 
ijwd_inter.p_inter_area_1 = 1/2; ijwd_inter.p_inter_area_2 = 1/2
ijwd_inter.section_width_1 = 4;  ijwd_inter.section_width_2 = 4; 
ijwd_inter.peak_p_e1_e2 = peak_p_e1_e2; ijwd_inter.tau_p_d_e1_e2 = tau_p_d_e1_e2
ijwd_inter.peak_p_e1_i2 = peak_p_e1_i2; ijwd_inter.tau_p_d_e1_i2 = tau_p_d_e1_i2        
ijwd_inter.peak_p_e2_e1 = peak_p_e2_e1; ijwd_inter.tau_p_d_e2_e1 = tau_p_d_e2_e1
ijwd_inter.peak_p_e2_i1 = peak_p_e2_i1; ijwd_inter.tau_p_d_e2_i1 = tau_p_d_e2_i1

ijwd_inter.w_e1_e2_mean = 5*scale_w_12_e; ijwd_inter.w_e1_i2_mean = 5*scale_w_12_i
ijwd_inter.w_e2_e1_mean = 5*scale_w_21_e; ijwd_inter.w_e2_i1_mean = 5*scale_w_21_i

ijwd_inter.generate_ijwd()

param_inter = {**ijwd_inter.__dict__}

del param_inter['i_e1_e2'], param_inter['j_e1_e2'], param_inter['w_e1_e2'], param_inter['d_e1_e2'] 
del param_inter['i_e1_i2'], param_inter['j_e1_i2'], param_inter['w_e1_i2'], param_inter['d_e1_i2'] 
del param_inter['i_e2_e1'], param_inter['j_e2_e1'], param_inter['w_e2_e1'], param_inter['d_e2_e1'] 
del param_inter['i_e2_i1'], param_inter['j_e2_i1'], param_inter['w_e2_i1'], param_inter['d_e2_i1']
#%%
# plt.figure()
# plt.plot(ijwd_inter.e_lattice1[ijwd_inter.inter_e_neuron_1][:,0], ijwd_inter.e_lattice1[ijwd_inter.inter_e_neuron_1][:,1],'o')

#%%
start_scope()

twoarea_net = build_two_areas.two_areas()

group_e_1, group_i_1, syn_ee_1, syn_ei_1, syn_ie_1, syn_ii_1,\
group_e_2, group_i_2, syn_ee_2, syn_ei_2, syn_ie_2, syn_ii_2,\
syn_e1e2, syn_e1i2, syn_e2e1, syn_e2i1 = twoarea_net.build(ijwd1, ijwd2, ijwd_inter)
#%%
'''record LFP'''
if record_LFP:
    from connection import get_LFP
    
    LFP_elec = np.array([[0,0],[-32,-32]])
    i_LFP, j_LFP, w_LFP = get_LFP.get_LFP(ijwd1.e_lattice, LFP_elec, width = ijwd1.width, LFP_sigma = 8, LFP_effect_range = 2.5)
    
    group_LFP_record = NeuronGroup(len(LFP_elec), model = get_LFP.LFP_recordneuron)
    syn_LFP = Synapses(group_e_1, group_LFP_record, model = get_LFP.LFP_syn)
    syn_LFP.connect(i=i_LFP, j=j_LFP)
    syn_LFP.w[:] = w_LFP[:]
#%%
# #lfp_p = ijwd1.e_lattice[syn_LFP.i[int(syn_LFP.i.shape[0]/2):]]
# lfp_p = ijwd1.e_lattice[syn_LFP.i[:int(syn_LFP.i.shape[0])]]

# #%%
# plt.figure()
# plt.scatter(ijwd1.e_lattice[:,0], ijwd1.e_lattice[:,1])
# plt.scatter(lfp_p[:,0], lfp_p[:,1], s=syn_LFP.w*10)
#%%

# chg_adapt_loca = [0, 0]
# chg_adapt_range = 6 
# chg_adapt_neuron = cn.findnearbyneuron.findnearbyneuron(ijwd2.e_lattice, chg_adapt_loca, chg_adapt_range, ijwd2.width)

chg_adapt_loca = [0, 0]
#chg_adapt_range = 6 
#chg_adapt_neuron = cn.findnearbyneuron.findnearbyneuron(ijwd2.e_lattice, chg_adapt_loca, chg_adapt_range, ijwd2.width)

adapt_value = adapt_gaussian.get_adaptation(base_amp = delta_gk_2, \
    max_decrease = [delta_gk_2-new_delta_gk_2], sig=[chg_adapt_range], position=[chg_adapt_loca], n_side=int(round((ijwd2.Ne)**0.5)), width=ijwd2.width)

#%%

'''external input'''#%%
'''stimulus'''
# ## no attention
# stim_scale_cls = get_stim_scale.get_stim_scale()
# stim_amp_scale = np.ones(100)
# for i in range(1):
#     stim_amp_scale[i*100:i*100+100] = i+2

# stim_scale_cls.stim_amp_scale = stim_amp_scale
# stim_scale_cls.get_scale()

# transient = 15000
# init = np.zeros(transient//stim_scale_cls.dt_stim)
# stim_scale_cls.scale_stim = np.concatenate((init, stim_scale_cls.scale_stim))
# stim_scale_cls.stim_on += transient
# ## attention
# stim_scale_cls_att = get_stim_scale.get_stim_scale()
# stim_amp_scale = np.ones(100)
# for i in range(1):
#     stim_amp_scale[i*100:i*100+100] = i+2

# stim_scale_cls_att.stim_amp_scale = stim_amp_scale
# stim_scale_cls_att.get_scale()

# inter_time = 4000
# suplmt = (inter_time // stim_scale_cls.dt_stim) - (stim_scale_cls.scale_stim.shape[0] - stim_scale_cls.stim_on[-1,1] // stim_scale_cls.dt_stim) # supply '0' between non-attention and attention stimuli amplitude

# stim_scale_cls.scale_stim = np.concatenate((stim_scale_cls.scale_stim, np.zeros(suplmt), stim_scale_cls_att.scale_stim))
# stim_scale_cls.stim_amp_scale = np.concatenate((stim_scale_cls.stim_amp_scale, stim_scale_cls_att.stim_amp_scale))
# stim_scale_cls_att.stim_on += stim_scale_cls.stim_on[-1,1] + inter_time
# stim_scale_cls.stim_on = np.vstack((stim_scale_cls.stim_on, stim_scale_cls_att.stim_on))
#stim_scale_cls = get_stim_scale.get_stim_scale()
'''stim 1; constant amplitude'''
'''no attention'''
stim_dura = 300 # ms duration of each stimulus presentation
transient = 3000 # ms initial transient period; when add stimulus
inter_time = 2000 # ms interval between block of trials without attention and with attention


stim_scale_cls = get_stim_scale.get_stim_scale()
stim_scale_cls.seed = 10
n_StimAmp = 1
n_perStimAmp = 1
stim_amp_scale = np.ones(n_StimAmp*n_perStimAmp)
for i in range(n_StimAmp):
    stim_amp_scale[i*n_perStimAmp:i*n_perStimAmp+n_perStimAmp] = 2**(i)

stim_scale_cls.stim_amp_scale = stim_amp_scale
stim_scale_cls.stim_dura = stim_dura
stim_scale_cls.separate_dura = np.array([300,600])
stim_scale_cls.get_scale()
stim_scale_cls.n_StimAmp = n_StimAmp
stim_scale_cls.n_perStimAmp = n_perStimAmp

#transient = 3000 # ms
init = np.zeros(transient//stim_scale_cls.dt_stim)
stim_scale_cls.scale_stim = np.concatenate((init, stim_scale_cls.scale_stim))
stim_scale_cls.stim_on += transient
'''attention'''
stim_scale_cls_att = get_stim_scale.get_stim_scale()
stim_scale_cls_att.seed = 15
n_StimAmp = 1
n_perStimAmp = 1
stim_amp_scale = np.ones(n_StimAmp*n_perStimAmp)
for i in range(n_StimAmp):
    stim_amp_scale[i*n_perStimAmp:i*n_perStimAmp+n_perStimAmp] = 2**(i)

stim_scale_cls_att.stim_amp_scale = stim_amp_scale
stim_scale_cls_att.stim_dura = stim_dura
stim_scale_cls_att.separate_dura = np.array([300,600])
stim_scale_cls_att.get_scale()

#inter_time = 1000
suplmt = (inter_time // stim_scale_cls.dt_stim) - (stim_scale_cls.scale_stim.shape[0] - stim_scale_cls.stim_on[-1,1] // stim_scale_cls.dt_stim) # supply '0' between non-attention and attention stimuli amplitude

stim_scale_cls.scale_stim = np.concatenate((stim_scale_cls.scale_stim, np.zeros(suplmt), stim_scale_cls_att.scale_stim))
stim_scale_cls.stim_amp_scale = np.concatenate((stim_scale_cls.stim_amp_scale, stim_scale_cls_att.stim_amp_scale))
stim_scale_cls_att.stim_on += stim_scale_cls.stim_on[-1,1] + inter_time
stim_scale_cls.stim_on = np.vstack((stim_scale_cls.stim_on, stim_scale_cls_att.stim_on))
#%%
scale_1 = TimedArray(stim_scale_cls.scale_stim, dt=10*ms)
data_ = mydata.mydata()
param_a1 = {**param_a1, 'stim1':data_.class2dict(stim_scale_cls)}
#%%
'''stim 2; varying amplitude'''
'''no attention'''
stim_scale_cls = get_stim_scale.get_stim_scale()
stim_scale_cls.seed = 10
n_StimAmp = 1
n_perStimAmp = 1
stim_amp_scale = np.ones(n_StimAmp*n_perStimAmp)
for i in range(n_StimAmp):
    stim_amp_scale[i*n_perStimAmp:i*n_perStimAmp+n_perStimAmp] = 2**(i)

stim_scale_cls.stim_amp_scale = stim_amp_scale
stim_scale_cls.stim_dura = stim_dura
stim_scale_cls.separate_dura = np.array([300,600])
stim_scale_cls.get_scale()
stim_scale_cls.n_StimAmp = n_StimAmp
stim_scale_cls.n_perStimAmp = n_perStimAmp

#transient = 3000
init = np.zeros(transient//stim_scale_cls.dt_stim)
stim_scale_cls.scale_stim = np.concatenate((init, stim_scale_cls.scale_stim))
stim_scale_cls.stim_on += transient
'''attention'''
stim_scale_cls_att = get_stim_scale.get_stim_scale()
stim_scale_cls_att.seed = 15
n_StimAmp = 1
n_perStimAmp = 1
stim_amp_scale = np.ones(n_StimAmp*n_perStimAmp)
for i in range(n_StimAmp):
    stim_amp_scale[i*n_perStimAmp:i*n_perStimAmp+n_perStimAmp] = 2**(i)

stim_scale_cls_att.stim_amp_scale = stim_amp_scale
stim_scale_cls_att.stim_dura = stim_dura
stim_scale_cls_att.separate_dura = np.array([300,600])
stim_scale_cls_att.get_scale()

#inter_time = 1000
suplmt = (inter_time // stim_scale_cls.dt_stim) - (stim_scale_cls.scale_stim.shape[0] - stim_scale_cls.stim_on[-1,1] // stim_scale_cls.dt_stim) # supply '0' between non-attention and attention stimuli amplitude

stim_scale_cls.scale_stim = np.concatenate((stim_scale_cls.scale_stim, np.zeros(suplmt), stim_scale_cls_att.scale_stim))
stim_scale_cls.stim_amp_scale = np.concatenate((stim_scale_cls.stim_amp_scale, stim_scale_cls_att.stim_amp_scale))
stim_scale_cls_att.stim_on += stim_scale_cls.stim_on[-1,1] + inter_time
stim_scale_cls.stim_on = np.vstack((stim_scale_cls.stim_on, stim_scale_cls_att.stim_on))
#%%
scale_2 = TimedArray(stim_scale_cls.scale_stim, dt=10*ms)

param_a1 = {**param_a1, 'stim2':data_.class2dict(stim_scale_cls)}
#%%
posi_stim_e1 = NeuronGroup(ijwd1.Ne, \
                        '''rates =  bkg_rates + stim_1*scale_1(t) + stim_2*scale_2(t) : Hz
                        bkg_rates : Hz
                        stim_1 : Hz
                        stim_2 : Hz
                        ''', threshold='rand()<rates*dt')

posi_stim_e1.bkg_rates = 0*Hz
posi_stim_e1.stim_1 = psti.input_spkrate(maxrate = [200], sig=[6], position=[[0, 0]])*Hz
posi_stim_e1.stim_2 = psti.input_spkrate(maxrate = [200], sig=[6], position=[[-32, -32]])*Hz
#posi_stim_e1.stim_2 = psti.input_spkrate(maxrate = [200], sig=[6], position=[[-32, 0]])*Hz

synapse_e_extnl = cn.model_neu_syn_AD.synapse_e_AD
syn_extnl_e1 = Synapses(posi_stim_e1, group_e_1, model=synapse_e_extnl, on_pre='x_E_extnl_post += w')
syn_extnl_e1.connect('i==j')
syn_extnl_e1.w = w_extnl_*nS#*tau_s_de_*nS

#%%

# scale = TimedArray(stim_scale_cls.scale_stim, dt=10*ms)

# param_a1 = {**param_a1, 'stim':stim_scale_cls}

# posi_stim_e1 = NeuronGroup(ijwd1.Ne, \
#                         '''rates =  bkg_rates + stim*scale(t) : Hz
#                         bkg_rates : Hz
#                         stim : Hz
#                         ''', threshold='rand()<rates*dt')

# posi_stim_e1.bkg_rates = 0*Hz
# posi_stim_e1.stim = psti.input_spkrate(maxrate = [200], sig=[6], position=[[0, 0]], n_side=63, width=63)*Hz

# synapse_e_extnl = cn.model_neu_syn_AD.synapse_e_AD
# syn_extnl_e1 = Synapses(posi_stim_e1, group_e_1, model=synapse_e_extnl, on_pre='x_E_extnl_post += w')
# syn_extnl_e1.connect('i==j')
# syn_extnl_e1.w = w_extnl_*nS#*tau_s_de_*nS



group_e_1.tau_s_de = tau_s_de_*ms; 
group_e_1.tau_s_di = tau_s_di_*ms
group_e_1.tau_s_re = group_e_1.tau_s_ri = tau_s_r_*ms

group_e_1.tau_s_de_inter = tau_s_de_*ms #5.0*ms; 
group_e_1.tau_s_re_inter = 1*ms
group_e_1.tau_s_de_extnl = 5.0*ms #5.0*ms
group_e_1.tau_s_re_extnl = 1*ms

group_i_1.tau_s_de = tau_s_de_*ms
group_i_1.tau_s_di = tau_s_di_*ms
group_i_1.tau_s_re = group_i_1.tau_s_ri = tau_s_r_*ms

group_i_1.tau_s_de_inter = tau_s_de_*ms #5.0*ms; 
group_i_1.tau_s_re_inter = 1*ms
group_i_1.tau_s_de_extnl = 5.0*ms #5.0*ms
group_i_1.tau_s_re_extnl = 1*ms

#syn_pois_e.effect = 0
#syn_pois_i.effect = 0
#group_e_v2.v = np.random.random(Ne)*35*mV-85*mV
#group_i_v2.v = np.random.random(Ni)*35*mV-85*mV
#seed(1000)
group_e_1.v = np.random.random(ijwd1.Ne)*35*mV-85*mV
group_i_1.v = np.random.random(ijwd1.Ni)*35*mV-85*mV
group_e_1.delta_gk = delta_gk_1*nS
group_e_1.tau_k = tau_k_*ms
#group_e_v2.v = np.random.random(Ne)*10*mV-60*mV
#group_i_v2.v = np.random.random(Ni)*10*mV-60*mV
group_e_1.I_extnl_crt = 0.51*nA # 0.25 0.51*nA
group_i_1.I_extnl_crt = 0.60*nA # 0.25 0.60*nA


group_e_2.tau_s_de = tau_s_de_*ms; 
group_e_2.tau_s_di = tau_s_di_*ms
group_e_2.tau_s_re = group_e_2.tau_s_ri = tau_s_r_*ms

group_e_2.tau_s_de_inter = tau_s_de_*ms #5.0*ms; 
group_e_2.tau_s_re_inter = 1*ms
group_e_2.tau_s_de_extnl = 5.0*ms #5.0*ms
group_e_2.tau_s_re_extnl = 1*ms

group_i_2.tau_s_de = tau_s_de_*ms
group_i_2.tau_s_di = tau_s_di_*ms
group_i_2.tau_s_re = group_i_2.tau_s_ri = tau_s_r_*ms

group_i_2.tau_s_de_inter = tau_s_de_*ms #5.0*ms; 
group_i_2.tau_s_re_inter = 1*ms
group_i_2.tau_s_de_extnl = 5.0*ms #5.0*ms
group_i_2.tau_s_re_extnl = 1*ms

#syn_pois_e.effect = 0
#syn_pois_i.effect = 0
#group_e_v2.v = np.random.random(Ne)*35*mV-85*mV
#group_i_v2.v = np.random.random(Ni)*35*mV-85*mV
#seed(1000)
group_e_2.v = np.random.random(ijwd2.Ne)*35*mV-85*mV
group_i_2.v = np.random.random(ijwd2.Ni)*35*mV-85*mV
group_e_2.delta_gk = delta_gk_2*nS
group_e_2.tau_k = tau_k_*ms
#group_e_v2.v = np.random.random(Ne)*10*mV-60*mV
#group_i_v2.v = np.random.random(Ni)*10*mV-60*mV
group_e_2.I_extnl_crt = 0.51*nA #0.51*nA
group_i_2.I_extnl_crt = 0.60*nA #0.60*nA

# scale_w_12 = 0.18#0.2
# scale_w_21 = 0.22#0.2
#%%
spk_e_1 = SpikeMonitor(group_e_1, record = True)
spk_i_1 = SpikeMonitor(group_i_1, record = True)
spk_e_2 = SpikeMonitor(group_e_2, record = True)
spk_i_2 = SpikeMonitor(group_i_2, record = True)
#ve1_moni = StateMonitor(group_e_1, ('v'), dt = 1*ms, record = True)
#vi1_moni = StateMonitor(group_i_1, ('v'), dt = 1*ms, record = True)

# lfp_moni = StateMonitor(group_LFP_record, ('lfp1','lfp2','lfp3'), record = True)

if record_LFP:
    lfp_moni = StateMonitor(group_LFP_record, ('lfp'), record = True)
#%%
net = Network(collect())
net.store('state1')

#%%
#change_ie(4.4)
#syn_ie_1.w = w_ie*usiemens

print('ie_w: %fnsiemens' %(syn_ie_1.w[0]/nsiemens))
#Ne = 63*63; Ni = 1000;
C = 0.25*nF # capacitance
g_l = 16.7*nS # leak capacitance
v_l = -70*mV # leak voltage
v_threshold = -50*mV
v_reset = -70*mV# -60*mV
v_rev_I = -80*mV
v_rev_E = 0*mV
v_k = -85*mV
#tau_k = 80*ms# 80*ms
#delta_gk = 10*nS #10*nS
t_ref = 5*ms # refractory period
#new_delta_gk_ = 0
#tau_s_de = 5*ms
#tau_s_di = 3*ms
#tau_s_re = 1*ms
#tau_s_ri = 1*ms
#tau_x_re = 1*ms
tic = time.perf_counter()
#seed(10)
simu_time_tot = (stim_scale_cls.stim_on[-1,1] + 500)*ms
#simu_time1 = 10000*ms#2000*ms
simu_time1 = (stim_scale_cls.stim_on[n_StimAmp*n_perStimAmp-1,1] + round(inter_time/2))*ms

simu_time2 = simu_time_tot - simu_time1
# simu_time3 = 2000*ms
# simu_time4 = 1000*ms
# simu_time5 = 5000*ms

#simu_time2 = 2000*ms#8000*ms
#simu_time_tot = 29000*ms
#extnl_e1.rates = 0*Hz
net.run(simu_time1, profile=False) #,namespace={'tau_k': 80*ms}
# net.run(simu_time2, profile=False) #,namespace={'tau_k': 80*ms}
# net.run(simu_time3, profile=False) #,namespace={'tau_k': 80*ms}
group_e_2.delta_gk[:] = adapt_value*nS

net.run(simu_time2, profile=False) #,namespace={'tau_k': 80*ms}
# net.run(simu_time5, profile=False) #,namespace={'tau_k': 80*ms}

#extnl_e.rates = bkg_rate2e
#net.run(simu_time3, profile=False) #,namespace={'tau_k': 80*ms}
#group_input.active = True
#group_e_1.delta_gk[chg_adapt_neuron] = 2*nS; group_e_1.tau_k[chg_adapt_neuron] = 40*ms
#net.run(simu_time2, report = 'text', profile=False) #,namespace={'tau_k': 80*ms}
print('total time elapsed:',np.round((time.perf_counter() - tic)/60,2), 'min')
#%%
spk_tstep_e1 = np.round(spk_e_1.t/(0.1*ms)).astype(int)
spk_tstep_i1 = np.round(spk_i_1.t/(0.1*ms)).astype(int)
spk_tstep_e2 = np.round(spk_e_2.t/(0.1*ms)).astype(int)
spk_tstep_i2 = np.round(spk_i_2.t/(0.1*ms)).astype(int)

now = datetime.datetime.now()
#param_1 = {'delta_gk':10,'tau_s_di':3,'tau_s_de':5,'ie_ratio':3.1725,'sti_onset':'10s'}

param_all = {'delta_gk_1':delta_gk_1,
             'delta_gk_2':delta_gk_2,
         'new_delta_gk_2':new_delta_gk_2,
         'tau_k': tau_k_,
         #'new_tau_k':40,
         'tau_s_di':tau_s_di_,
         'tau_s_de':tau_s_de_,
         'tau_s_r':tau_s_r_,
         #'scale_d_p_i':scale_d_p_i,
         'num_ee':num_ee,
         'num_ei':num_ei,
         'num_ii':num_ii,
         'num_ie':num_ie,
         #'ie_ratio':ie_ratio_,
         #'mean_J_ee': ijwd.mean_J_ee,
         #'chg_adapt_range':6, 
         #'p_ee':p_ee,
         'simutime':int(round(simu_time_tot/ms)),
         #'chg_adapt_time': simu_time1/ms,
         'chg_adapt_range': chg_adapt_range,
         'chg_adapt_loca': chg_adapt_loca,
         #'chg_adapt_neuron': chg_adapt_neuron,
         #'scale_ee_1': scale_ee_1,
         #'scale_ei_1': scale_ei_1,
         #'scale_ie_1': scale_ie_1,
         #'scale_ii_1': scale_ii_1,
         'ie_r_e': ie_r_e,
         'ie_r_e1':ie_r_e1,   
         'ie_r_e2':ie_r_e2,
         'ie_r_i': ie_r_i,
         'ie_r_i1': ie_r_i1,
         'ie_r_i2': ie_r_i2,
         't_ref': t_ref/ms}
#param_a2 = {**param_a2, **param_new_2}

#param = {}
#param_12 = {'scale_e_12':scale_e_12[i], 'scale_e_21':scale_e_21[j]}

#data = {'param':param_12, 'a1':{'param':param_1,'e':{'i':spk_1.i[:],'t':spk_1_tstep}},
#                            'a2':{'param':param_2,'e':{'i':spk_2.i[:],'t':spk_2_tstep}}}
data = {'datetime':now.strftime("%Y-%m-%d %H:%M:%S"), 'dt':0.1, 'loop_num':loop_num, 'data_dir': os.getcwd(),
        'param':param_all,
        'a1':{'param':param_a1,
              #'LFP':{'lfp1':lfp_moni.lfp1/nA, 'lfp2':lfp_moni.lfp2/nA, 'lfp3':lfp_moni.lfp3/nA},
              'ge':{'i':spk_e_1.i[:],'t':spk_tstep_e1},    
              'gi':{'i':spk_i_1.i[:],'t':spk_tstep_i1}},
        'a2':{'param':param_a2,
              'ge':{'i':spk_e_2.i[:],'t':spk_tstep_e2},
              'gi':{'i':spk_i_2.i[:],'t':spk_tstep_i2}},
        'inter':{'param':param_inter}}
if record_LFP:
    data['a1']['ge']['LFP'] = lfp_moni.lfp[:]/nA
##'v':ve1_moni.v[:]/mV},   
#'v':vi1_moni.v[:]/mV}},
#data = mydata.mydata(data)
#%%
''' save data to disk'''
with open(data_dir+'data%d.file'%loop_num, 'wb') as file:
    pickle.dump(data, file)

'''load data from disk'''
data_dir = 'raw_data/'
datapath = data_dir

data_load = mydata.mydata()
data_load.load(datapath+'data%d.file'%loop_num)    

'''or load data from 'dictionary' '''
# data_load = mydata.mydata(data)
    
#%%
''' animation '''
from analysis import firing_rate_analysis as fra

#first_stim = 1*n_perStimAmp -1; last_stim = 1*n_perStimAmp
#first_stim = 0; last_stim = 0
start_time = transient - 500  #data.a1.param.stim1.stim_on[first_stim,0] - 300
end_time = int(round(simu_time_tot/ms))   #data.a1.param.stim1.stim_on[last_stim,0] + 1500

data_load.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data_load.a1.param.Ne, window = 15)
data_load.a1.ge.get_centre_mass()
data_load.a1.ge.overlap_centreandspike()

data_load.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data_load.a2.param.Ne, window = 15)
data_load.a2.ge.get_centre_mass()
data_load.a2.ge.overlap_centreandspike()

frames = data_load.a1.ge.spk_rate.spk_rate.shape[2]

stim_on_off = data_load.a1.param.stim1.stim_on-start_time
stim_on_off = stim_on_off[stim_on_off[:,0]>=0]#[:int(last_stim-first_stim)+1]

#stim = [[[[31.5,31.5]], [stim_on_off], [[6]*stim_on_off.shape[0]]],None]
stim = [[[[31.5,31.5],[63.5,-0.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]],None]
#stim = [[[[31.5,31.5],[31.5,-0.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]],None]

#adpt = None
adpt_start = data_load.a1.param.stim1.stim_on[n_StimAmp*n_perStimAmp, 0] - start_time - round(inter_time/2)
adpt = [None, [[[31.5,31.5]], [[[adpt_start, data_load.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[chg_adapt_range]]]]

#adpt = [None, [[[31,31]], [[[0, data_load.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
#adpt = [[[[31,31]], [[[0, data_load.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
title = 'animation'
ani = fra.show_pattern(spkrate1=data_load.a1.ge.spk_rate.spk_rate, spkrate2=data_load.a2.ge.spk_rate.spk_rate, \
                                        frames = frames, start_time = start_time, interval_movie=15, anititle=title,stim=stim, adpt=adpt)

    
    
    
