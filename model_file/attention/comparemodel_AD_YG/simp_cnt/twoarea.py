#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 12:29:07 2020

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
import sys
import pickle
import preprocess_2area
#%%
prefs.codegen.target = 'cython'

dir_cache = './cache/cache%s' %sys.argv[1]
prefs.codegen.runtime.cython.cache_dir = dir_cache
prefs.codegen.max_cache_dir_size = 30
#%%
sys_argv = int(sys.argv[1])
#%%
loop_num = -1

#param_in = [[0.08,6.5]]
#repeat = 10
for scale_d_p_ee in [1]:#np.linspace(0.85,1.15,13):
    for scale_d_p_ei in [1]:#np.linspace(0.85,1.15,13):#[1]
        for scale_d_p_ii in np.linspace(1,1,1):
            for scale_d_p_ie in np.linspace(1,1,1):#np.linspace(0.85,1.15,13):
                for num_ee in [260]:#np.linspace(125, 297, 10,dtype=int): 
                    for num_ei in [400]:#np.linspace(204, 466, 10, dtype=int):#[320]:               
                        for num_ie in [175]:#[260]:#np.linspace(246, 300, 6,dtype=int):#[221]:#np.linspace(156, 297, 10,dtype=int):
                            for num_ii in [260]:#np.linspace(135, 170,6,dtype=int):#[129]:#np.linspace(93, 177, 10,dtype=int):
                                for ie_r_e in [3.1]:#np.linspace(3.10,3.45,6):#np.linspace(1.4,1.6,10):#[1.5]: #np.linspace(1.,1.2,5):
                                    for w_ie_ in [17]:#np.linspace(17,23,7):#np.linspace(1.077,1.277,10):#[1.177]:#np.linspace(1,1,1):
                                        for ie_r_i in [2.786]:#np.linspace(2.5,3.0,15):#np.linspace(1.156-0.08,1.156+0.08,15):#np.linspace(1.156-0.175,1.156+0.175,31):#np.linspace(1.157,1.357,10):#[1.257]: #np.linspace(0.95,1.1,5):
                                            for w_ii_ in [23]:#np.linspace(20,27,8):#np.arange(1.4,1.81,0.1)]]:#np.linspace(1.,1.,1):
                                                for w_extnl_ in [1.5]:
                                                    for delta_gk_ in [12]:#np.linspace(7,15,9)]]:
                                                        for new_delta_gk_ in [12/4]:#np.arange(0.15,0.31,0.05)*delta_gk_:#np.linspace(1,5,5):
                                                            for tau_s_di_ in np.linspace(4.444,4.444,1):
                                                                for tau_s_de_ in np.linspace(5.,5.,1):
                                                                    for tau_s_r_ in np.linspace(1,1,1):
                                                                        #for decay_p_ie_p_ii in [20]:    
                                                                            #for ie_ratio_ in 3.375*np.arange(0.94, 1.21, 0.02):#(np.arange(0.7,1.56,0.05)-0.02):#np.linspace(2.4, 4.5, 20):
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
                
if loop_num != sys_argv: sys.exit("Error: wrong PBS_array_id")                    

#%%
def find_w_e(w_i, num_i, num_e, ie_ratio):
    return (w_i*num_i)/num_e/ie_ratio
               

ijwd1 = pre_process_sc.get_ijwd()
ijwd1.Ne = 63*63; ijwd1.Ni = 32*32
ijwd1.width = 62#79
#ijwd1.w_ee_mean *= 2; ijwd1.w_ei_mean *= 2; ijwd1.w_ie_mean *= 2; ijwd1.w_ii_mean *= 2;
scale_d_p = 1 #np.sqrt(8/5) 
ijwd1.decay_p_ee = 7 #* scale_d_p_ee#scale_d_p # decay constant of e to e connection probability as distance increases
ijwd1.decay_p_ei = 9 #* scale_d_p_ei# scale_d_p # decay constant of e to i connection probability as distance increases
ijwd1.decay_p_ie = 20 #* scale_d_p_ie#scale_d_p_i#* scale_d_p # decay constant of i to e connection probability as distance increases
ijwd1.decay_p_ii = 20 #* scale_d_p_ii#* scale_d_p # decay constant of i to i connection probability as distance increases

ijwd1.mean_SynNumIn_ee = num_ee     ; # p = 0.08
ijwd1.mean_SynNumIn_ei = num_ei #* 8/5     ; # p = 0.125
ijwd1.mean_SynNumIn_ie = num_ie  #scale_d_p_i    ; # p = 0.2
ijwd1.mean_SynNumIn_ii = num_ii# 221 * scale_d_p_i#* 8/5     ; # p = 0.25

ijwd1.w_ee_mean = find_w_e(w_ie_, num_ie, num_ee, ie_r_e)
ijwd1.w_ei_mean = find_w_e(w_ii_, num_ii, num_ei, ie_r_i)
ijwd1.w_ie_mean = w_ie_
ijwd1.w_ii_mean = w_ii_
        
ijwd1.generate_ijw()
ijwd1.generate_d_rand()

# ijwd1.w_ee *= scale_ee_1#tau_s_de_scale_d_p_i
# ijwd1.w_ei *= scale_ei_1 #tau_s_de_ #5*nS
# ijwd1.w_ie *= scale_ie_1#tau_s_di_#25*nS
# ijwd1.w_ii *= scale_ii_1#tau_s_di_#
param = {**ijwd1.__dict__}



del param['i_ee'], param['j_ee'], param['w_ee'], param['d_ee'] 
del param['i_ei'], param['j_ei'], param['w_ei'], param['d_ei'] 
del param['i_ie'], param['j_ie'], param['w_ie'], param['d_ie'] 
del param['i_ii'], param['j_ii'], param['w_ii'], param['d_ii']

ijwd2 = pre_process_sc.get_ijwd()
ijwd2.Ne = 63*63; ijwd2.Ni = 32*32
ijwd2.width = 62#79
#ijwd2.w_ee_mean *= 2; ijwd2.w_ei_mean *= 2; ijwd2.w_ie_mean *= 2; ijwd2.w_ii_mean *= 2;
scale_d_p = 1 #np.sqrt(8/5) 
ijwd2.decay_p_ee = 7 #* scale_d_p_ee#scale_d_p # decay constant of e to e connection probability as distance increases
ijwd2.decay_p_ei = 9 #* scale_d_p_ei# scale_d_p # decay constant of e to i connection probability as distance increases
ijwd2.decay_p_ie = 20 #* scale_d_p_ie#scale_d_p_i#* scale_d_p # decay constant of i to e connection probability as distance increases
ijwd2.decay_p_ii = 20 #* scale_d_p_ii#* scale_d_p # decay constant of i to i connection probability as distance increases

ijwd2.mean_SynNumIn_ee = num_ee     ; # p = 0.08
ijwd2.mean_SynNumIn_ei = num_ei #* 8/5     ; # p = 0.125
ijwd2.mean_SynNumIn_ie = num_ie  #scale_d_p_i    ; # p = 0.2
ijwd2.mean_SynNumIn_ii = num_ii# 221 * scale_d_p_i#* 8/5     ; # p = 0.25

ijwd2.w_ee_mean = find_w_e(w_ie_, num_ie, num_ee, ie_r_e)
ijwd2.w_ei_mean = find_w_e(w_ii_, num_ii, num_ei, ie_r_i)
ijwd2.w_ie_mean = w_ie_
ijwd2.w_ii_mean = w_ii_
        
ijwd2.generate_ijw()
ijwd2.generate_d_rand()

ijwd_inter = preprocess_2area.get_ijwd_2()
ijwd_inter.peak_p_e1_e2 = 0.5; ijwd_inter.tau_p_d_e1_e2 = 4.2
ijwd_inter.peak_p_e1_i2 = 0.5; ijwd_inter.tau_p_d_e1_i2 = 4.2        
ijwd_inter.peak_p_e2_e1 = 0.28; ijwd_inter.tau_d_e2_e1 = 7
ijwd_inter.peak_p_e2_i1 = 0.28; ijwd_inter.tau_d_e2_i1 = 7
ijwd_inter.generate_ijwd()
#%%

chg_adapt_loca = [0, 0]
chg_adapt_range = 6 * scale_d_p
chg_adapt_neuron = cn.findnearbyneuron.findnearbyneuron(ijwd2.e_lattice, chg_adapt_loca, chg_adapt_range, ijwd2.width)

#%%
start_scope()

neuronmodel_e = cn.model_neu_syn_AD.neuron_e_AD
neuronmodel_i = cn.model_neu_syn_AD.neuron_i_AD

synapse_e = cn.model_neu_syn_AD.synapse_e_AD
synapse_i = cn.model_neu_syn_AD.synapse_i_AD

#%%
group_e_1 =NeuronGroup(ijwd1.Ne, model=neuronmodel_e,
                     threshold='v>v_threshold', method='euler',
                     reset='''v = v_reset
                              g_k += delta_gk''', refractory='(t-lastspike)<t_ref')

group_i_1 =NeuronGroup(ijwd1.Ni, model=neuronmodel_i,
                     threshold='v>v_threshold', method='euler',
                     reset='v = v_reset', refractory='(t-lastspike)<t_ref')

syn_ee_1 = Synapses(group_e_1, group_e_1, model=synapse_e, 
                  on_pre='''x_E_post += w''')
syn_ei_1 = Synapses(group_e_1, group_i_1, model=synapse_e, 
                  on_pre='''x_E_post += w''')
syn_ie_1 = Synapses(group_i_1, group_e_1, model=synapse_i, 
                  on_pre='''x_I_post += w''')
syn_ii_1 = Synapses(group_i_1, group_i_1, model=synapse_i, 
                  on_pre='''x_I_post += w''')

'''external input'''
# #stim_rate = psti.input_spkrate(maxrate = [800,800], sig=[6,6], position=[[0, 0],[31.5,31.5]])*Hz
# bkg_rate2e = 850*Hz
# bkg_rate2i = 1000*Hz
# extnl_e = PoissonGroup(ijwd1.Ne, bkg_rate2e)
# extnl_i = PoissonGroup(ijwd1.Ni, bkg_rate2i)

# #tau_x_re = 1*ms
# synapse_e_extnl = cn.model_neu_syn_AD.synapse_e_AD
# syn_extnl_e = Synapses(extnl_e, group_e_1, model=synapse_e_extnl, on_pre='x_E_extnl_post += w')
# syn_extnl_i = Synapses(extnl_i, group_i_1, model=synapse_e_extnl, on_pre='x_E_extnl_post += w')

# syn_extnl_e.connect('i==j')
# syn_extnl_i.connect('i==j')

# #w_extnl_ = 1.5 # nS
# syn_extnl_e.w = w_extnl_*5.8*nS#*tau_s_de_*nS
# syn_extnl_i.w = w_extnl_*5.8*nS#*tau_s_de_*nS


syn_ee_1.connect(i=ijwd1.i_ee, j=ijwd1.j_ee)
syn_ei_1.connect(i=ijwd1.i_ei, j=ijwd1.j_ei)
syn_ie_1.connect(i=ijwd1.i_ie, j=ijwd1.j_ie)
syn_ii_1.connect(i=ijwd1.i_ii, j=ijwd1.j_ii)


#tau_s_di_try = tau_s_di_
syn_ee_1.w = ijwd1.w_ee*nsiemens/tau_s_r_ * 5.8 #tau_s_de_
syn_ei_1.w = ijwd1.w_ei*nsiemens/tau_s_r_ * 5.8 #tau_s_de_ #5*nS
syn_ii_1.w = ijwd1.w_ii*nsiemens/tau_s_r_ * 6.5 #tau_s_di_#25*nS
syn_ie_1.w = ijwd1.w_ie*nsiemens/tau_s_r_ * 6.5 #tau_s_di_#


def set_delay(syn, delay_up):
    #n = len(syn)
    syn.delay = delay_up*ms
    #syn.down.delay = (delay_up + 1)*ms
    
    return syn 

#d_ee, d_ie, d_ei, d_ii = generate_d_rand(4,len(i_ee),len(i_ie),len(i_ei),len(i_ii))
syn_ee_1 = set_delay(syn_ee_1, ijwd1.d_ee)
syn_ie_1 = set_delay(syn_ie_1, ijwd1.d_ie)
syn_ei_1 = set_delay(syn_ei_1, ijwd1.d_ei)
syn_ii_1 = set_delay(syn_ii_1, ijwd1.d_ii)

#syn_pois_e = set_delay(syn_pois_e)
#syn_pois_i = set_delay(syn_pois_i)

#tau_s_de_ = 5.8; tau_s_di_ = 6.5
#delta_gk_ = 10


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
group_e_1.delta_gk = delta_gk_*nS
group_e_1.tau_k = 80*ms
#group_e_v2.v = np.random.random(Ne)*10*mV-60*mV
#group_i_v2.v = np.random.random(Ni)*10*mV-60*mV
group_e_1.I_extnl_crt = 0.51*nA #0.51*nA
group_i_1.I_extnl_crt = 0.60*nA #0.60*nA

#%%
group_e_2 =NeuronGroup(ijwd2.Ne, model=neuronmodel_e,
                     threshold='v>v_threshold', method='euler',
                     reset='''v = v_reset
                              g_k += delta_gk''', refractory='(t-lastspike)<t_ref')

group_i_2 =NeuronGroup(ijwd2.Ni, model=neuronmodel_i,
                     threshold='v>v_threshold', method='euler',
                     reset='v = v_reset', refractory='(t-lastspike)<t_ref')

syn_ee_2 = Synapses(group_e_2, group_e_2, model=synapse_e, 
                  on_pre='''x_E_post += w''')
syn_ei_2 = Synapses(group_e_2, group_i_2, model=synapse_e, 
                  on_pre='''x_E_post += w''')
syn_ie_2 = Synapses(group_i_2, group_e_2, model=synapse_i, 
                  on_pre='''x_I_post += w''')
syn_ii_2 = Synapses(group_i_2, group_i_2, model=synapse_i, 
                  on_pre='''x_I_post += w''')

'''external input'''
# #stim_rate = psti.input_spkrate(maxrate = [800,800], sig=[6,6], position=[[0, 0],[31.5,31.5]])*Hz
# bkg_rate2e = 850*Hz
# bkg_rate2i = 1000*Hz
# extnl_e = PoissonGroup(ijwd2.Ne, bkg_rate2e)
# extnl_i = PoissonGroup(ijwd2.Ni, bkg_rate2i)

# #tau_x_re = 1*ms
# synapse_e_extnl = cn.model_neu_syn_AD.synapse_e_AD
# syn_extnl_e = Synapses(extnl_e, group_e_2, model=synapse_e_extnl, on_pre='x_E_extnl_post += w')
# syn_extnl_i = Synapses(extnl_i, group_i_2, model=synapse_e_extnl, on_pre='x_E_extnl_post += w')

# syn_extnl_e.connect('i==j')
# syn_extnl_i.connect('i==j')

# #w_extnl_ = 1.5 # nS
# syn_extnl_e.w = w_extnl_*5.8*nS#*tau_s_de_*nS
# syn_extnl_i.w = w_extnl_*5.8*nS#*tau_s_de_*nS


syn_ee_2.connect(i=ijwd2.i_ee, j=ijwd2.j_ee)
syn_ei_2.connect(i=ijwd2.i_ei, j=ijwd2.j_ei)
syn_ie_2.connect(i=ijwd2.i_ie, j=ijwd2.j_ie)
syn_ii_2.connect(i=ijwd2.i_ii, j=ijwd2.j_ii)


#tau_s_di_try = tau_s_di_
syn_ee_2.w = ijwd2.w_ee*nsiemens/tau_s_r_ * 5.8 #tau_s_de_
syn_ei_2.w = ijwd2.w_ei*nsiemens/tau_s_r_ * 5.8 #tau_s_de_ #5*nS
syn_ii_2.w = ijwd2.w_ii*nsiemens/tau_s_r_ * 6.5 #tau_s_di_#25*nS
syn_ie_2.w = ijwd2.w_ie*nsiemens/tau_s_r_ * 6.5 #tau_s_di_#


def set_delay(syn, delay_up):
    #n = len(syn)
    syn.delay = delay_up*ms
    #syn.down.delay = (delay_up + 1)*ms
    
    return syn 

#d_ee, d_ie, d_ei, d_ii = generate_d_rand(4,len(i_ee),len(i_ie),len(i_ei),len(i_ii))
syn_ee_2 = set_delay(syn_ee_2, ijwd2.d_ee)
syn_ie_2 = set_delay(syn_ie_2, ijwd2.d_ie)
syn_ei_2 = set_delay(syn_ei_2, ijwd2.d_ei)
syn_ii_2 = set_delay(syn_ii_2, ijwd2.d_ii)

#syn_pois_e = set_delay(syn_pois_e)
#syn_pois_i = set_delay(syn_pois_i)

#tau_s_de_ = 5.8; tau_s_di_ = 6.5
#delta_gk_ = 10


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
group_e_2.delta_gk = delta_gk_*nS
group_e_2.tau_k = 80*ms
#group_e_v2.v = np.random.random(Ne)*10*mV-60*mV
#group_i_v2.v = np.random.random(Ni)*10*mV-60*mV
group_e_2.I_extnl_crt = 0.51*nA #0.51*nA
group_i_2.I_extnl_crt = 0.60*nA #0.60*nA
#%%
syn_e1e2 = Synapses(group_e_1, group_e_2, model=synapse_e, 
                  on_pre='''x_E_inter_post += w''')
syn_e1i2 = Synapses(group_e_1, group_i_2, model=synapse_e, 
                  on_pre='''x_E_inter_post += w''')
syn_e2e1 = Synapses(group_e_2, group_e_1, model=synapse_e, 
                  on_pre='''x_E_inter_post += w''')
syn_e2i1 = Synapses(group_e_2, group_i_1, model=synapse_e, 
                  on_pre='''x_E_inter_post += w''')

syn_e1e2.connect(i=ijwd_inter.i_e1_e2, j=ijwd_inter.j_e1_e2)
syn_e1i2.connect(i=ijwd_inter.i_e1_i2, j=ijwd_inter.j_e1_i2)
syn_e2e1.connect(i=ijwd_inter.i_e2_e1, j=ijwd_inter.j_e2_e1)
syn_e2i1.connect(i=ijwd_inter.i_e2_i1, j=ijwd_inter.j_e2_i1)

syn_e1e2.w = ijwd_inter.w_e1_e2*5*nS
syn_e1i2.w = ijwd_inter.w_e1_i2*5*nS
syn_e2e1.w = ijwd_inter.w_e2_e1*5*nS
syn_e2i1.w = ijwd_inter.w_e2_i1*5*nS

syn_e1e2.delay = ijwd_inter.d_e1_e2*ms
syn_e1i2.delay = ijwd_inter.d_e1_i2*ms
syn_e2e1.delay = ijwd_inter.d_e2_e1*ms
syn_e2i1.delay = ijwd_inter.d_e2_i1*ms
#%%
spk_e_1 = SpikeMonitor(group_e_1, record = True)
spk_i_1 = SpikeMonitor(group_i_1, record = True)
spk_e_2 = SpikeMonitor(group_e_2, record = True)
spk_i_2 = SpikeMonitor(group_i_2, record = True)

#lfp_moni = StateMonitor(group_LFP_record, ('lfp1','lfp2','lfp3'), record = True)
#%%
net = Network(collect())
net.store('state1')
#%%
#scale_ee = 1.55#1.05; 
#scale_ei = 1.5#1.
#scale_ie = 1.5#0.95
#scale_ii = 1.5#1.00
#syn_ee_1.w = ijwd1.w_ee*nsiemens * 5.8 * scale_ee#tau_s_de_
#syn_ei_1.w = ijwd1.w_ei*nsiemens * 5.8 * scale_ei #tau_s_de_ #5*nS
#syn_ie_1.w = ijwd1.w_ie*nsiemens * 6. * scale_ie#tau_s_di_#25*nS
#syn_ii_1.w = ijwd1.w_ii*nsiemens * 6. * scale_ii#tau_s_di_#
scale_w_12 = 0.18#0.2
scale_w_21 = 0.22#0.2
syn_e1e2.w = ijwd_inter.w_e1_e2*5*nS * scale_w_12
syn_e1i2.w = ijwd_inter.w_e1_i2*5*nS * scale_w_12
syn_e2e1.w = ijwd_inter.w_e2_e1*5*nS * scale_w_21
syn_e2i1.w = ijwd_inter.w_e2_i1*5*nS * scale_w_21

syn_ee_1.w = ijwd1.w_ee*nsiemens/tau_s_r_ * 5.8 #tau_s_de_
syn_ei_1.w = ijwd1.w_ei*nsiemens/tau_s_r_ * 5.8 #tau_s_de_ #5*nS 
syn_ii_1.w = ijwd1.w_ii*nsiemens/tau_s_r_ * 6.5 #tau_s_di_#25*nS
syn_ie_1.w = ijwd1.w_ie*nsiemens/tau_s_r_ * 6.5 * 1.03#tau_s_di_#

#tau_d = 100*ms
#%%
#change_ie(4.4)
#syn_ie_1.w = w_ie*usiemens
allrun = 1

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
t_ref = 4*ms # refractory period
#new_delta_gk_ = 0
#tau_s_de = 5*ms
#tau_s_di = 3*ms
#tau_s_re = 1*ms
#tau_s_ri = 1*ms
#tau_x_re = 1*ms
tic = time.perf_counter()
#seed(10)
simu_time1 = 3000*ms#2000*ms
simu_time2 = 12000*ms
#simu_time2 = 2000*ms#8000*ms
#simu_time3 = 1000*ms
simu_time_tot = 32000*ms
#group_input.active = False
# extnl_e.rates = bkg_rate2e
net.run(simu_time1, profile=False) #,namespace={'tau_k': 80*ms}
#extnl_e.rates = bkg_rate2e + stim_rate
if allrun:
    group_e_2.delta_gk[chg_adapt_neuron] = 2*nS
    net.run(simu_time2, profile=False) #,namespace={'tau_k': 80*ms}
#extnl_e.rates = bkg_rate2e
#net.run(simu_time3, profile=False) #,namespace={'tau_k': 80*ms}
#group_input.active = True
#group_e_1.delta_gk[chg_adapt_neuron] = 2*nS; group_e_1.tau_k[chg_adapt_neuron] = 40*ms
#net.run(simu_time2, report = 'text', profile=False) #,namespace={'tau_k': 80*ms}
print('total time elapsed:',time.perf_counter() - tic)
#%%
spk_tstep_e1 = np.round(spk_e_1.t/(0.1*ms)).astype(int)
spk_tstep_i1 = np.round(spk_i_1.t/(0.1*ms)).astype(int)
spk_tstep_e2 = np.round(spk_e_2.t/(0.1*ms)).astype(int)
spk_tstep_i2 = np.round(spk_i_2.t/(0.1*ms)).astype(int)
now = datetime.datetime.now()
#param_1 = {'delta_gk':10,'tau_s_di':3,'tau_s_de':5,'ie_ratio':3.1725,'sti_onset':'10s'}

param_new = {'delta_gk':delta_gk_,
         'new_delta_gk':new_delta_gk_,
         #'new_tau_k':40,
         'tau_s_di':tau_s_di_,
         'tau_s_de':tau_s_de_,
         'tau_s_r':tau_s_r_,
         #'scale_d_p_i':scale_d_p_i,
         'num_ee':num_ee,
         'num_ei':num_ei,
         'num_ii':num_ii,
         'num_ie':num_ie,
         'scale_d_p_ee':scale_d_p_ee,
         'scale_d_p_ei':scale_d_p_ei,
         'scale_d_p_ii':scale_d_p_ii,
         'scale_d_p_ie':scale_d_p_ie,
         #'ie_ratio':ie_ratio_,
         #'mean_J_ee': ijwd2.mean_J_ee,
         #'chg_adapt_range':6, 
         #'p_ee':p_ee,
         #'simutime':simu_time_tot/ms,
         #'chg_adapt_time': simu_time1/ms,
         'chg_adapt_range': chg_adapt_range,
         'chg_adapt_loca': chg_adapt_loca,
         'chg_adapt_neuron': chg_adapt_neuron,
         #'scale_ee_1': scale_ee_1,
         #'scale_ei_1': scale_ei_1,
         #'scale_ie_1': scale_ie_1,
         #'scale_ii_1': scale_ii_1,
         'ie_r_e': ie_r_e,
         'ie_r_i': ie_r_i,
         't_ref': t_ref/ms}
param = {**param, **param_new}

#param = {}
#param_12 = {'scale_e_12':scale_e_12[i], 'scale_e_21':scale_e_21[j]}

#data = {'param':param_12, 'a1':{'param':param_1,'e':{'i':spk_1.i[:],'t':spk_1_tstep}},
#                            'a2':{'param':param_2,'e':{'i':spk_2.i[:],'t':spk_2_tstep}}}
data = {'datetime':now.strftime("%Y-%m-%d %H:%M:%S"), 'dt':0.1, 'loop_num':loop_num, 'data_dir': os.getcwd(),
        'a1':{'param':param,
              #'LFP':{'lfp1':lfp_moni.lfp1/nA, 'lfp2':lfp_moni.lfp2/nA, 'lfp3':lfp_moni.lfp3/nA},
              'ge':{'i':spk_e_1.i[:],'t':spk_tstep_e1},
              'gi':{'i':spk_i_1.i[:],'t':spk_tstep_i1}},
        'a2':{'ge':{'i':spk_e_2.i[:],'t':spk_tstep_e2},
              'gi':{'i':spk_i_2.i[:],'t':spk_tstep_i2}}}


#data = mydata.mydata(data)
# with open('data%d.file'%loop_num, 'wb') as file:
#     pickle.dump(data, file)
data1 = mydata.mydata(data)
#data1.load(data)
#%%
start_time = 0e3; end_time = 6e3
data1.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data1.a1.param.Ne, window = 10)
#data1.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data1.a1.param.Ni, window = 10)
# data1.a1.ge.get_centre_mass(detect_pattern=True)
# #data1.a1.ge.overlap_centreandspike()
# pattern_size2 = data1.a1.ge.centre_mass.pattern_size.copy()
# pattern_size2[np.invert(data1.a1.ge.centre_mass.pattern)] = 0
data1.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data1.a1.param.Ne, window = 10)

#title = "dgk{:.1f}_ndgk{:.1f}_alpha{:.2f}".format(data.a1.param.delta_gk,data.a1.param.new_delta_gk,data.a1.ge.MSD.stableDist_param[0,0])
#title = "dgk{:.1f}_ndgk{:.1f}_alpha{:.2f}".format(data.a1.param.delta_gk,data.a1.param.new_delta_gk,data.a1.ge.MSD.stableDist_param[0,0])
#title = "ie%.3f_dgk%.1f_ndgk%.2f_tauk%.1f_alpha%.2f"%(data.a1.param.scale_ie_1, data.a1.param.delta_gk, data.a1.param.new_delta_gk, data.a1.param.tau_k, data.a1.ge.MSD.stableDist_param[0,0])
#title = "ei%.3f_ii%.3f_dgk%.1f_ndgk%.2f_alpha%.2f"%(data.a1.param.scale_ei_1, data.a1.param.scale_ii_1, data.a1.param.delta_gk, data.a1.param.new_delta_gk, data.a1.ge.MSD.stableDist_param[0,0])
#title = "ee%.3f_ie%.3f_dgk%.1f_ndgk%.2f"%(data1.a1.param.scale_ee_1, data1.a1.param.scale_ie_1, data1.a1.param.delta_gk, data1.a1.param.new_delta_gk)

frames = data1.a1.ge.spk_rate.spk_rate.shape[2]
ani = firing_rate_analysis.show_pattern(data1.a1.ge.spk_rate.spk_rate, data1.a2.ge.spk_rate.spk_rate, frames = frames, start_time = start_time, anititle=title)
# ani = firing_rate_analysis.show_pattern(data1.a1.ge.spk_rate.spk_rate, None, frames = frames, start_time = start_time, anititle=title,\
#                                         show_pattern_size=True, pattern_ctr=data1.a1.ge.centre_mass.centre_ind, \
#                                             pattern_size=pattern_size2)
#%%
data1.a1.ge.get_spike_rate(start_time=0, end_time=12000,\
                           sample_interval = 1, n_neuron = ijwd.Ne, window = 10, dt = 0.1, reshape_indiv_rate = True)
data1.a2.ge.get_spike_rate(start_time=0, end_time=12000,\
                           sample_interval = 1, n_neuron = ijwd.Ne, window = 10, dt = 0.1, reshape_indiv_rate = True)


    
mua_adpt1 = data1.a1.ge.spk_rate.spk_rate.reshape(ijwd1.Ne,-1)[chg_adapt_neuron].mean(0)/0.01
mua_adpt2 = data1.a2.ge.spk_rate.spk_rate.reshape(ijwd2.Ne,-1)[chg_adapt_neuron].mean(0)/0.01

#%%
fig, [ax1,ax2] = plt.subplots(2,1)
ax1.plot(mua_adpt1[0:15000])
ax2.plot(mua_adpt2[0:15000])

#%%
def find_peakF(coef, freq):
    dF = freq[1] - freq[0]
    Fwin = 0.5
    lwin = int(Fwin/dF)
    win = np.ones(lwin)/lwin
    coef_avg = np.convolve(np.abs(coef[1:]), win, mode='same')
    peakF = freq[1:][coef_avg.argmax()]
    return peakF
#%%
fig, [ax1,ax2] = plt.subplots(1,2,figsize=[8,4])
coef, freq = fa.myfft(mua_adpt1[5000:15000], 1000)
freq_max = 20
ind_len = freq[freq<freq_max].shape[0] 

# plt.figure()
ax1.plot(freq[1:ind_len], np.abs(coef[1:ind_len]))
ax1.plot([3,3],[0,np.max(np.abs(coef[1:ind_len]))],'r--')
ax1.plot([8,8],[0,np.max(np.abs(coef[1:ind_len]))],'r--')
ax1.set_title('spon')
coef, freq = fa.myfft(mua_adpt2[5000:15000], 1000)
# freq_max = 20
ind_len = freq[freq<freq_max].shape[0] 

# plt.figure()
ax2.plot(freq[1:ind_len], np.abs(coef[1:ind_len]))
ax2.plot([3,3],[0,np.max(np.abs(coef[1:ind_len]))],'r--')
ax2.plot([8,8],[0,np.max(np.abs(coef[1:ind_len]))],'r--')
ax2.set_title('adapt')

#%%
net.restore('state1')
#%%
fig, [ax1,ax2] = plt.subplots(1,2,figsize=[8,4])
coef, freq = fa.myfft(mua_adpt1[2000:12000], 1000)
freq_max = 20
ind_len = freq[freq<freq_max].shape[0] 

# plt.figure()
ax1.plot(freq[1:ind_len], np.abs(coef[1:ind_len]))
ax1.plot([3,3],[0,np.max(np.abs(coef[1:ind_len]))],'r--')
ax1.plot([8,8],[0,np.max(np.abs(coef[1:ind_len]))],'r--')
ax1.set_title('spon')
coef, freq = fa.myfft(mua_adpt2[2000:12000], 1000)
# freq_max = 20
ind_len = freq[freq<freq_max].shape[0] 

# plt.figure()
ax2.plot(freq[1:ind_len], np.abs(coef[1:ind_len]))
ax2.plot([3,3],[0,np.max(np.abs(coef[1:ind_len]))],'r--')
ax2.plot([8,8],[0,np.max(np.abs(coef[1:ind_len]))],'r--')
ax2.set_title('adapt')
#%%
dt = 1/10000;
end = int(15/dt); start = int(5/dt)
spon_rate = np.sum((data1.a1.ge.t < end) & (data1.a1.ge.t > start))/10/data1.a1.param.Ne
print(spon_rate)
spon_rate = np.sum((data1.a1.gi.t < end) & (data1.a1.gi.t > start))/10/data1.a1.param.Ni
print(spon_rate)