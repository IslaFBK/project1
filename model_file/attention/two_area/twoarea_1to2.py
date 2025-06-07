#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 18:23:15 2021

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
import get_stim_scale
import sys
import pickle

#%%
prefs.codegen.target = 'cython'

dir_cache = './cache/cache%s' %sys.argv[1]
prefs.codegen.runtime.cython.cache_dir = dir_cache
prefs.codegen.max_cache_dir_size = 80
#%%
sys_argv = int(sys.argv[1])
#%%
loop_num = -1

#param_in = [[0.08,6.5]]
#repeat = 10

# for num_ee in [240]:#np.linspace(125, 297, 10,dtype=int): 
#     for num_ei in [400]:#np.linspace(204, 466, 10, dtype=int):#[320]:               
#         for num_ie in [150]:#[260]:#np.linspace(246, 300, 6,dtype=int):#[221]:#np.linspace(156, 297, 10,dtype=int):
#             for num_ii in [230]:#np.linspace(135, 170,6,dtype=int):#[129]:#np.linspace(93, 177, 10,dtype=int):
#for ie_r_e in [2.76*6.5/5.8]:# np.linspace(3.07,3.13,7):#np.arange(2.5,4.0,0.04):#[3.1]:#np.linspace(3.10,3.45,6):#np.linspace(1.4,1.6,10):#[1.5]: #np.linspace(1.,1.2,5):
#    for ie_r_e1 in np.array([0.96,0.98,1.02,1.04]):
#        for ie_r_e2 in [1]:#np.arange(0.96,1.041,0.02):
#            for w_ee_ in [4*5.8]:#[4]:
#               for w_ie_ in [None]:#[23.12]:#[17]:#np.linspace(17,23,7):#np.linspace(1.077,1.277,10):#[1.177]:#np.linspace(1,1,1):
#                    for ie_r_i in [2.450*6.5/5.8]:#[2.719]:#[2.817]:#[2.786]:#np.linspace(2.5,3.0,15):#np.linspace(1.156-0.08,1.156+0.08,15):#np.linspace(1.156-0.175,1.156+0.175,31):#np.linspace(1.157,1.357,10):#[1.257]: #np.linspace(0.95,1.1,5):
#                        for w_ii_ in [None]:#[25]:#np.linspace(20,27,8):#np.arange(1.4,1.81,0.1)]]:#np.linspace(1.,1.,1):
#                            for w_ei_ in [5*5.8]:#[6.35]:
for w_extnl_ in [10]:
    for delta_gk_1 in [5]:#np.linspace(7,15,9)]]:
        for delta_gk_2 in [16]:
            for new_delta_gk_2 in [16/5]:#np.arange(0.15,0.31,0.05)*delta_gk_:#np.linspace(1,5,5):
                for tau_s_di_ in np.linspace(4.4,4.4,1):
                    for tau_s_de_ in np.linspace(5.,5.,1):
                        for tau_s_r_ in [1]:#np.linspace(1,1,1):
                            for scale_w_12_e in np.arange(0.8,1.21,0.05):
                                for scale_w_12_i in np.arange(0.8,1.21,0.05):#[scale_w_12_e]:#0.2*np.arange(0.8,1.21,0.05):
                                    for scale_w_21_e in [1]:#np.arange(0.8,1.21,0.05):
                                        for scale_w_21_i in [1]:#[scale_w_21_e]:#0.35*np.arange(0.8,1.21,0.05):
                                            for tau_p_d_e1_e2 in [5]:
                                                for tau_p_d_e1_i2 in [5]:
                                                    for peak_p_e1_e2 in [0.45]:#np.arange(0.3,0.451,0.05):
                                                        for peak_p_e1_i2 in [0.45]:#np.arange(0.25,0.401,0.05):
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
    #                             else: continue
    #                             break
    #                         else: continue
    #                         break
    #                     else: continue
    #                     break
    #                 else: continue
    #                 break
    #             else: continue
    #             break
    #         else: continue
    #         break
    #     else: continue
    #     break
    # else: continue
    # break
               
    #             else: continue
    #             break
    #         else: continue
    #         break
    #     else: continue
    #     break
    # else: continue
    # break

if loop_num != sys_argv: sys.exit("Error: wrong PBS_array_id")                    

#%%
def find_w_e(w_i, num_i, num_e, ie_ratio):
    return (w_i*num_i)/num_e/ie_ratio
               
def find_w_i(w_e, num_e, num_i, ie_ratio):
    return (w_e*num_e)*ie_ratio/num_i

w_ie_1 = 115; w_ii_1 = 100
ie_r_e = 2.76*6.5/5.8; ie_r_e1 = 1.6
ie_r_i = 2.450*6.5/5.8; ie_r_i1 = 1.648

ijwd1 = pre_process_sc.get_ijwd()
ijwd1.Ne = 63*63; ijwd1.Ni = 32*32
ijwd1.width = 63#79
#ijwd1.w_ee_mean *= 2; ijwd1.w_ei_mean *= 2; ijwd1.w_ie_mean *= 2; ijwd1.w_ii_mean *= 2;
#scale_d_p = 1 #np.sqrt(8/5) 
ijwd1.decay_p_ee = 7 #* scale_d_p_ee#scale_d_p # decay constant of e to e connection probability as distance increases
ijwd1.decay_p_ei = 9 #* scale_d_p_ei# scale_d_p # decay constaw_ie_ in [115]nt of e to i connection probability as distance increases
ijwd1.decay_p_ie = 19 #* scale_d_p_ie#scale_d_p_i#* scale_d_p # decay constant of i to e connection probability as distance increases
ijwd1.decay_p_ii = 19 #* scale_d_p_ii#* scale_d_p # decay constant of i to i connection probability as distance increases

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
ijwd1.generate_d_rand()

# ijwd1.w_ee *= scale_ee_1#tau_s_de_scale_d_p_i
# ijwd1.w_ei *= scale_ei_1 #tau_s_de_ #5*nS
# ijwd1.w_ie *= scale_ie_1#tau_s_di_#25*nS
# ijwd1.w_ii *= scale_ii_1#tau_s_di_#
param_a1 = {**ijwd1.__dict__}



del param_a1['i_ee'], param_a1['j_ee'], param_a1['w_ee'], param_a1['d_ee'] 
del param_a1['i_ei'], param_a1['j_ei'], param_a1['w_ei'], param_a1['d_ei'] 
del param_a1['i_ie'], param_a1['j_ie'], param_a1['w_ie'], param_a1['d_ie'] 
del param_a1['i_ii'], param_a1['j_ii'], param_a1['w_ii'], param_a1['d_ii']



w_ee_2 = 4*5.8; w_ei_2 = 5*5.8
ie_r_e = 2.76*6.5/5.8; ie_r_e2 = 1
ie_r_i = 2.450*6.5/5.8; ie_r_i2 = 1

ijwd2 = pre_process_sc.get_ijwd()
ijwd2.Ne = 63*63; ijwd2.Ni = 32*32
ijwd2.width = 63#79
#ijwd2.w_ee_mean *= 2; ijwd2.w_ei_mean *= 2; ijwd2.w_ie_mean *= 2; ijwd2.w_ii_mean *= 2;
#scale_d_p = 1 #np.sqrt(8/5) 
ijwd2.decay_p_ee = 7#8*0.8 #* scale_d_p_ee#scale_d_p # decay constant of e to e connection probability as distance increases
ijwd2.decay_p_ei = 9#10*0.82 #* scale_d_p_ei# scale_d_p # decay constant of e to i connection probability as distance increases
ijwd2.decay_p_ie = 19#20*0.86 #* scale_d_p_ie#scale_d_p_i#* scale_d_p # decay constant of i to e connection probability as distance increases
ijwd2.decay_p_ii = 19#20*0.86 #* scale_d_p_ii#* scale_d_p # decay constant of i to i connection probability as distance increases

num_ee = 240; num_ei = 400; num_ie = 150; num_ii = 230

ijwd2.mean_SynNumIn_ee = num_ee#num_ee     ; # p = 0.08
ijwd2.mean_SynNumIn_ei = num_ei#num_ei #* 8/5     ; # p = 0.125
ijwd2.mean_SynNumIn_ie = num_ie#num_ie  #scale_d_p_i    ; # p = 0.2
ijwd2.mean_SynNumIn_ii = num_ii#num_ii# 221 * scale_d_p_i#* 8/5     ; # p = 0.25

ijwd2.w_ee_mean = w_ee_2#find_w_e(w_ie_, num_ie, num_ee, ie_r_e)#find_w_e(w_ie_, num_ie, num_ee, ie_r_e)
ijwd2.w_ei_mean = w_ei_2#find_w_e(w_ii_, num_ii, num_ei, ie_r_i)
ijwd2.w_ie_mean = find_w_i(w_ee_2, num_ee, num_ie, ie_r_e*ie_r_e2)#w_ie_ #find_w_i(w_ee_, num_ee, num_ie, ie_r_e)
ijwd2.w_ii_mean = find_w_i(w_ei_2, num_ei, num_ii, ie_r_i*ie_r_i2)#w_ii_
   
ijwd2.generate_ijw()
ijwd2.generate_d_rand()

param_a2 = {**ijwd2.__dict__}

del param_a2['i_ee'], param_a2['j_ee'], param_a2['w_ee'], param_a2['d_ee'] 
del param_a2['i_ei'], param_a2['j_ei'], param_a2['w_ei'], param_a2['d_ei'] 
del param_a2['i_ie'], param_a2['j_ie'], param_a2['w_ie'], param_a2['d_ie'] 
del param_a2['i_ii'], param_a2['j_ii'], param_a2['w_ii'], param_a2['d_ii']

ijwd_inter = preprocess_2area.get_ijwd_2()

ijwd_inter.p_inter_area_1 = 1/4; ijwd_inter.p_inter_area_2 = 1/4
ijwd_inter.peak_p_e1_e2 = peak_p_e1_e2; ijwd_inter.tau_p_d_e1_e2 = tau_p_d_e1_e2
ijwd_inter.peak_p_e1_i2 = peak_p_e1_i2; ijwd_inter.tau_p_d_e1_i2 = tau_p_d_e1_i2        
ijwd_inter.peak_p_e2_e1 = 0.4; ijwd_inter.tau_d_e2_e1 = 7
ijwd_inter.peak_p_e2_i1 = 0.4; ijwd_inter.tau_d_e2_i1 = 9

ijwd_inter.w_e1_e2_mean = 4*scale_w_12_e; ijwd_inter.w_e1_i2_mean = 4*scale_w_12_i
ijwd_inter.w_e2_e1_mean = 7*scale_w_21_e*0; ijwd_inter.w_e2_i1_mean = 9.6*scale_w_21_i*0

ijwd_inter.generate_ijwd()

param_inter = {**ijwd_inter.__dict__}

del param_inter['i_e1_e2'], param_inter['j_e1_e2'], param_inter['w_e1_e2'], param_inter['d_e1_e2'] 
del param_inter['i_e1_i2'], param_inter['j_e1_i2'], param_inter['w_e1_i2'], param_inter['d_e1_i2'] 
del param_inter['i_e2_e1'], param_inter['j_e2_e1'], param_inter['w_e2_e1'], param_inter['d_e2_e1'] 
del param_inter['i_e2_i1'], param_inter['j_e2_i1'], param_inter['w_e2_i1'], param_inter['d_e2_i1']
#%%
start_scope()

twoarea_net = build_two_areas.two_areas()

group_e_1, group_i_1, syn_ee_1, syn_ei_1, syn_ie_1, syn_ii_1,\
group_e_2, group_i_2, syn_ee_2, syn_ei_2, syn_ie_2, syn_ii_2,\
syn_e1e2, syn_e1i2, syn_e2e1, syn_e2i1 = twoarea_net.build(ijwd1, ijwd2, ijwd_inter)
#%%

chg_adapt_loca = [0, 0]
chg_adapt_range = 6 
chg_adapt_neuron = cn.findnearbyneuron.findnearbyneuron(ijwd2.e_lattice, chg_adapt_loca, chg_adapt_range, ijwd2.width)

#%%

'''external input'''
# #stim_rate = psti.input_spkrate(maxrate = [800,800], sig=[6,6], position=[[0, 0],[31.5,31.5]])*Hz
# stim_rate = psti.input_spkrate(maxrate = [800], sig=[6], position=[[0, 0]])*Hz

# #bkg_rate2e = 0*Hz#850*Hz
# #bkg_rate2i = 1000*Hz
# extnl_e1 = PoissonGroup(ijwd1.Ne, stim_rate)
# #extnl_i = PoissonGroup(ijwd1.Ni, bkg_rate2i)

# #tau_x_re = 1*ms
# synapse_e_extnl = cn.model_neu_syn_AD.synapse_e_AD
# syn_extnl_e1 = Synapses(extnl_e1, group_e_1, model=synapse_e_extnl, on_pre='x_E_extnl_post += w')
# #syn_extnl_i = Synapses(extnl_i, group_i_1, model=synapse_e_extnl, on_pre='x_E_extnl_post += w')

# syn_extnl_e1.connect('i==j')
# #syn_extnl_i.connect('i==j')

# #w_extnl_ = 1.5 # nS
# syn_extnl_e1.w = w_extnl_*nS#*tau_s_de_*nS
# #syn_extnl_i.w = w_extnl_*nS#*tau_s_de_*nS

# #syn_pois_e = set_delay(syn_pois_e)
# #syn_pois_i = set_delay(syn_pois_i)

# #tau_s_de_ = 5.8; tau_s_di_ = 6.5
# #delta_gk_ = 10

stim_scale_cls = get_stim_scale.get_stim_scale()
stim_amp_scale = np.ones(30)
for i in range(3):
    stim_amp_scale[i*10:i*10+10] = i+2

stim_scale_cls.stim_amp_scale = stim_amp_scale
stim_scale_cls.get_scale()

transient = 10000
init = np.zeros(transient//stim_scale_cls.dt_stim)
stim_scale_cls.scale_stim = np.concatenate((init, stim_scale_cls.scale_stim))
stim_scale_cls.stim_on += transient
scale = TimedArray(stim_scale_cls.scale_stim, dt=10*ms)

param_a1 = {**param_a1, 'stim':stim_scale_cls}

posi_stim_e1 = NeuronGroup(ijwd1.Ne, \
                        '''rates =  bkg_rates + stim*scale(t) : Hz
                        bkg_rates : Hz
                        stim : Hz
                        ''', threshold='rand()<rates*dt')

posi_stim_e1.bkg_rates = 0*Hz
posi_stim_e1.stim = psti.input_spkrate(maxrate = [200], sig=[6], position=[[0, 0]])*Hz

synapse_e_extnl = cn.model_neu_syn_AD.synapse_e_AD
syn_extnl_e1 = Synapses(posi_stim_e1, group_e_1, model=synapse_e_extnl, on_pre='x_E_extnl_post += w')
syn_extnl_e1.connect('i==j')
syn_extnl_e1.w = w_extnl_*nS#*tau_s_de_*nS



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
group_e_1.tau_k = 60*ms
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
group_e_2.tau_k = 60*ms
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
ve1_moni = StateMonitor(group_e_1, ('v'), dt = 1*ms, record = True)
vi1_moni = StateMonitor(group_i_1, ('v'), dt = 1*ms, record = True)

#lfp_moni = StateMonitor(group_LFP_record, ('lfp1','lfp2','lfp3'), record = True)
#%%
net = Network(collect())
net.store('state1')

#%%
#scale_ee = 1.55#1.05; 
#scale_ei = 1.5#1.
#scale_ie = 1.5#0.95
#scale_ii = 1.5#1.00
#syn_ee_1.w = ijwd.w_ee*nsiemens * 5.8 * scale_ee#tau_s_de_
#syn_ei_1.w = ijwd.w_ei*nsiemens * 5.8 * scale_ei #tau_s_de_ #5*nS
#syn_ie_1.w = ijwd.w_ie*nsiemens * 6. * scale_ie#tau_s_di_#25*nS
#syn_ii_1.w = ijwd.w_ii*nsiemens * 6. * scale_ii#tau_s_di_#

#tau_d = 100*ms
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
simu_time1 = 28000*ms#2000*ms
simu_time2 = 5000*ms
simu_time3 = 2000*ms
simu_time4 = 1000*ms
simu_time5 = 5000*ms

#simu_time2 = 2000*ms#8000*ms
simu_time_tot = 28000*ms
#extnl_e1.rates = 0*Hz
net.run(simu_time1, profile=False) #,namespace={'tau_k': 80*ms}
# net.run(simu_time2, profile=False) #,namespace={'tau_k': 80*ms}
# net.run(simu_time3, profile=False) #,namespace={'tau_k': 80*ms}
# group_e_2.delta_gk[chg_adapt_neuron] = new_delta_gk_2*nS
# net.run(simu_time4, profile=False) #,namespace={'tau_k': 80*ms}
# net.run(simu_time5, profile=False) #,namespace={'tau_k': 80*ms}

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

param_all = {'delta_gk_1':delta_gk_1,
             'delta_gk_2':delta_gk_2,
         'new_delta_gk_2':new_delta_gk_2,
         'tau_k': 60,
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
         'ie_r_e1':ie_r_e1,   
         'ie_r_e2':ie_r_e2,
         'ie_r_i': ie_r_i,
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
              'ge':{'i':spk_e_1.i[:],'t':spk_tstep_e1,'v':ve1_moni.v[:]/mV},
              'gi':{'i':spk_i_1.i[:],'t':spk_tstep_i1,'v':vi1_moni.v[:]/mV}},
        'a2':{'param':param_a2,
              'ge':{'i':spk_e_2.i[:],'t':spk_tstep_e2},
              'gi':{'i':spk_i_2.i[:],'t':spk_tstep_i2}},
        'inter':{'param':param_inter}}


#data = mydata.mydata(data)
with open('data%d.file'%loop_num, 'wb') as file:
    pickle.dump(data, file)
#%%