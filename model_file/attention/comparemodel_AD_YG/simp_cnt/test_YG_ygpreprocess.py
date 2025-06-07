#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:08:03 2020

@author: shni2598
"""

'''
YG neuron model

YG pre_process

test effect of common-neighbour factor on pattern size
'''
from connection import pre_process
from connection import connect_interareas 
import connection as cn
#from analysis import post_analysis
import poisson_stimuli as psti
import matplotlib as mpl
mpl.use('Agg')
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
import time

import brian2.numpy_ as np
import brian2.only
from brian2.only import *
#import pickle
import sys
import pickle
#%%
prefs.codegen.target = 'cython'

dir_cache = './cache/cache%s' %sys.argv[1]
prefs.codegen.runtime.cython.cache_dir = dir_cache
prefs.codegen.max_cache_dir_size = 30
#%%
sys_argv = int(sys.argv[1])
#%%
loop_num = -1

# param_in = [[0.08,6.5]]
# repeat = 10
# for p_ee, tau_s_di_ in param_in*repeat:
#     for scale_ee_1 in np.linspace(1,1,1):
#         for delta_gk_ in [3]:
#             for tau_s_de_ in [5.8]:
#                 for ie_ratio_ in 3.375*(np.arange(0.7,1.56,0.05)-0.02):#np.linspace(2.4, 4.5, 20):
#                     loop_num += 1
#                     if loop_num == sys_argv:
#                         print(loop_num)
#                         break
#                 else: continue
#                 break
#             else: continue
#             break
#         else: continue
#         break
#     else: continue
#     break
for cn_scale_wire_ in [1, 2]:
    for scale_ee_1 in [1.200]:#np.arange(1,1.26,0.05):#[1.15]: #np.linspace(1.,1.2,5):
        for scale_ei_1 in np.linspace(1,1,1):
            #for scale_ie_1 in np.linspace(0.7,1.3,20)*1.0225:#np.arange(0.9825,1.1225,0.02):#[1.0625]: #np.linspace(0.95,1.1,5):
            for ie_ratio_ in 20/4*200/320/1.15*np.linspace(0.7,1.3,20)*1.0225:    
                for scale_ii_1 in np.linspace(1.,1.,1):
                    for w_extnl_ in [1.5]:
                        for delta_gk_ in [13]:#np.linspace(7,14,8):
                            for new_delta_gk_ in [1.95]:#np.arange(0.15,0.31,0.05)*delta_gk_:#np.linspace(1,5,5):
                                for tau_s_di_ in [5.7772]:#np.arange(0.7,1.31,0.1)*4.444:#np.linspace(4.444,4.444,1):
                                    for tau_s_de_ in [5.8]:#np.arange(0.7,1.11,0.1)*5.8:#np.linspace(5.8,5.8,1):
                                        for tau_s_r_ in np.linspace(1,1,1):
                                            for decay_p_ie_p_ii in [20]:
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
 
if loop_num != sys_argv: sys.exit("wrong PBS_array_id")
#%%

# ijwd = pre_process_sc.get_ijwd()
# ijwd.Ne = 63*63; ijwd.Ni = 32*32
# ijwd.width = 62#79
# #ijwd.w_ee_mean *= 2; ijwd.w_ei_mean *= 2; ijwd.w_ie_mean *= 2; ijwd.w_ii_mean *= 2;
# scale_d_p = 1 #np.sqrt(8/5) 
# ijwd.decay_p_ee = 8 * scale_d_p # decay constant of e to e connection probability as distance increases
# ijwd.decay_p_ei = 10 * scale_d_p # decay constant of e to i connection probability as distance increases
# ijwd.decay_p_ie = decay_p_ie_p_ii * scale_d_p # decay constant of i to e connection probability as distance increases
# ijwd.decay_p_ii = decay_p_ie_p_ii * scale_d_p # decay constant of i to i connection probability as distance increases

# ijwd.mean_SynNumIn_ee = 320 * 5/8    ; # p = 0.08
# ijwd.mean_SynNumIn_ei = 500 #* 8/5     ; # p = 0.125
# ijwd.mean_SynNumIn_ie = 200 * 5/8    ; # p = 0.2
# ijwd.mean_SynNumIn_ii = 250 #* 8/5     ; # p = 0.25

# ijwd.generate_ijw()
# ijwd.generate_d_rand()

# ijwd.w_ee *= scale_ee_1#tau_s_de_
# ijwd.w_ei *= scale_ei_1 #tau_s_de_ #5*nS
# ijwd.w_ie *= scale_ie_1#tau_s_di_#25*nS
# ijwd.w_ii *= scale_ii_1#tau_s_di_#
# param = {**ijwd.__dict__}

#%%
Ne = 3969; Ni = 1024

ijwd = pre_process.get_ijwd(Ni=Ni)
ijwd.p_ee = 0.05
ijwd.p_ei = 0.125
ijwd.p_ie = 0.125
ijwd.p_ii = 0.25
ijwd.w_ee_dist = 'normal'
ijwd.hybrid = 0.
ijwd.cn_scale_weight = 1
ijwd.cn_scale_wire = cn_scale_wire_
ijwd.iter_num=5

ijwd.ie_ratio = ie_ratio_
scale_ee_1 = scale_ee_1; scale_ei_1 = scale_ei_1
ijwd.mean_J_ee = 4*10**-3*scale_ee_1 # usiemens
ijwd.sigma_J_ee = 1.9*10**-3*scale_ee_1 # usiemens

ijwd.w_ei = 5*10**(-3)*scale_ei_1 #usiemens
ijwd.w_ii = 25*10**(-3)*scale_ii_1

ijwd.change_dependent_para()
param = {**ijwd.__dict__}

ijwd.generate_ijw()
ijwd.generate_d_rand()
#%%
chg_adapt_loca = [0, 0]
chg_adapt_range = 6
chg_adapt_neuron = cn.findnearbyneuron.findnearbyneuron(ijwd.lattice_ext, chg_adapt_loca, chg_adapt_range, ijwd.width)
#%%
start_scope()

neuronmodel_e = '''
dv/dt = (1/C)*(-g_l*(v - v_l) + (-g_k)*(v - v_k) + (-g_I)*(v - v_rev_I) +(-g_E-g_E_inter-g_E_extnl)*(v - v_rev_E) + I_extnl) : volt (unless refractory)

dg_k/dt = -g_k/tau_k :siemens
delta_gk : siemens
tau_k : second

g_I : siemens
g_E : siemens
g_E_inter : siemens
I_extnl : amp

g_E_extnl = w_extnl*s_extnl: siemens
w_extnl: siemens
ds_extnl/dt = -s_extnl/tau_s_de_extnl + x : 1
dx/dt = -x/tau_x_re : Hz
tau_s_de_extnl : second
'''

neuronmodel_i = '''
dv/dt = (1/C)*(-g_l*(v - v_l) + (-g_I)*(v - v_rev_I) +(-g_E-g_E_inter-g_E_extnl)*(v - v_rev_E) + I_extnl) : volt (unless refractory)
g_I : siemens
g_E : siemens
g_E_inter : siemens
I_extnl : amp

g_E_extnl = w_extnl*s_extnl: siemens
w_extnl: siemens
ds_extnl/dt = -s_extnl/tau_s_de_extnl + x : 1
dx/dt = -x/tau_x_re : Hz
tau_s_de_extnl : second
'''

synapse_e = '''
w: siemens
g_E_post = w*s : siemens (summed)
ds/dt = -s/tau_s_de + rect_puls*(1 - s) : 1 (clock-driven)
rect_puls = (1/tau_s_re)*effect : Hz
effect : integer
tau_s_de : second
'''
synapse_i = '''
w: siemens
g_I_post = w*s : siemens (summed)
ds/dt = -s/tau_s_di + rect_puls*(1 - s) : 1 (clock-driven)
rect_puls = (1/tau_s_ri)*effect : Hz
effect : integer
tau_s_di : second
'''
#%%
group_e_1 =NeuronGroup(ijwd.Ne, model=neuronmodel_e,
                     threshold='v>v_threshold', method='euler',
                     reset='''v = v_reset
                              g_k += delta_gk''', refractory='(t-lastspike)<t_ref')

group_i_1 =NeuronGroup(ijwd.Ni, model=neuronmodel_i,
                     threshold='v>v_threshold', method='euler',
                     reset='v = v_reset', refractory='(t-lastspike)<t_ref')

syn_ee_1 = Synapses(group_e_1, group_e_1, model=synapse_e, method='euler',
                  on_pre={'up':'effect += 1', 'down': 'effect -= 1'})
syn_ei_1 = Synapses(group_e_1, group_i_1, model=synapse_e, method='euler',
                  on_pre={'up':'effect += 1', 'down': 'effect -= 1'})
syn_ie_1 = Synapses(group_i_1, group_e_1, model=synapse_i, method='euler',
                  on_pre={'up':'effect += 1', 'down': 'effect -= 1'})
syn_ii_1 = Synapses(group_i_1, group_i_1, model=synapse_i, method='euler',
                  on_pre={'up':'effect += 1', 'down': 'effect -= 1'})

#extnl_input_e = PoissonGroup(ijwd.Ne, 850*Hz)
#extnl_input_i = PoissonGroup(ijwd.Ni, 1000*Hz)
#
#syn_extnl_e = Synapses(extnl_input_e, group_e_1, method='euler', on_pre='x_post += 1/tau_x_re')
#syn_extnl_i = Synapses(extnl_input_i, group_i_1, method='euler', on_pre='x_post += 1/tau_x_re')

#%%
syn_ee_1.connect(i=ijwd.i_ee, j=ijwd.j_ee)
syn_ei_1.connect(i=ijwd.i_ei, j=ijwd.j_ei)
syn_ie_1.connect(i=ijwd.i_ie, j=ijwd.j_ie)
syn_ii_1.connect(i=ijwd.i_ii, j=ijwd.j_ii)
#syn_extnl_e.connect('i==j')
#syn_extnl_i.connect('i==j')
#group_e_1.w_extnl = 2*nS
#group_i_1.w_extnl = 2*nS

#%%
syn_ee_1.w = ijwd.w_ee*usiemens
syn_ei_1.w = ijwd.w_ei*usiemens #5*nS
syn_ii_1.w = ijwd.w_ii*usiemens #25*nS
syn_ie_1.w = ijwd.w_ie*usiemens
#w_ext = 2*nS
#syn_pois_e.w = w_ext
#syn_pois_i.w = w_ext
#%%
def set_delay(syn, delay_up):
    #n = len(syn)
    syn.up.delay = delay_up*ms
    syn.down.delay = (delay_up + 1)*ms
    
    return syn 
#%%
#ijwd.generate_d_dist()
#ijwd.generate_d_rand()
syn_ee_1 = set_delay(syn_ee_1, ijwd.d_ee)
syn_ie_1 = set_delay(syn_ie_1, ijwd.d_ie)
syn_ei_1 = set_delay(syn_ei_1, ijwd.d_ei)
syn_ii_1 = set_delay(syn_ii_1, ijwd.d_ii)

#syn_pois_e = set_delay(syn_pois_e)
#syn_pois_i = set_delay(syn_pois_i)

#syn_ee_v2.up.delay = 3*ms; syn_ee_v2.down.delay = 4*ms; 
#syn_ie_v2.up.delay = 3*ms; syn_ie_v2.down.delay = 4*ms; 
#syn_ei_v2.up.delay = 3*ms; syn_ei_v2.down.delay = 4*ms; 
#syn_ii_v2.up.delay = 3*ms; syn_ii_v2.down.delay = 4*ms; 

syn_ee_1.effect = 0; syn_ee_1.tau_s_de = tau_s_de_*ms
syn_ie_1.effect = 0; syn_ie_1.tau_s_di = tau_s_di_*ms
syn_ei_1.effect = 0; syn_ei_1.tau_s_de = tau_s_de_*ms
syn_ii_1.effect = 0; syn_ii_1.tau_s_di = tau_s_di_*ms

#syn_pois_e.effect = 0
#syn_pois_i.effect = 0
#group_e_v2.v = np.random.random(Ne)*35*mV-85*mV
#group_i_v2.v = np.random.random(Ni)*35*mV-85*mV
#seed(1000)
group_e_1.v = np.random.random(ijwd.Ne)*35*mV-85*mV
group_i_1.v = np.random.random(ijwd.Ni)*35*mV-85*mV
group_e_1.delta_gk = delta_gk_*nS
group_e_1.tau_k = 80*ms
group_e_1.tau_s_de_extnl = 5*ms
group_i_1.tau_s_de_extnl = 5*ms

#group_e_v2.v = np.random.random(ijwd.Ne)*10*mV-60*mV
#group_i_v2.v = np.random.random(ijwd.Ni)*10*mV-60*mV
group_e_1.I_extnl = 0.51*nA
group_i_1.I_extnl = 0.60*nA

#%%
spk_e = SpikeMonitor(group_e_1, record = True)
spk_i = SpikeMonitor(group_i_1, record = True)

#%%
net = Network(collect())
net.store('state1')
#%%
#ijwd.change_ie(4.4)
#syn_ie_1.w = ijwd.w_ie*usiemens

print('ie_w: %fnsiemens' %(syn_ie_1.w[0]/nsiemens))
#Ne = 63*63; Ni = 1000;
C = 0.25*nF # capacitance
g_l = 16.7*nS # leak capacitance
v_l = -70*mV # leak voltage
v_threshold = -50*mV
v_reset = -60*mV
v_rev_I = -80*mV
v_rev_E = 0*mV
v_k = -85*mV
#tau_k = 80*ms# 80*ms
#delta_gk = 10*nS #10*nS
t_ref = 4*ms # refractory period

#tau_s_de = 5*ms
#tau_s_di = 3*ms
tau_s_re = 1*ms
tau_s_ri = 1*ms
tau_x_re = 1*ms
tic = time.perf_counter()
#seed(10)
simu_time1 = 5000*ms#2000*ms
simu_time2 = 5000*ms#8000*ms
simu_time = simu_time1+simu_time2
#simu_time3 = 2000*ms

#group_input.active = False
net.run(simu_time1, profile=False) #,namespace={'tau_k': 80*ms}


group_e_1.delta_gk[chg_adapt_neuron] = new_delta_gk_*nS
net.run(simu_time2, profile=False) #,namespace={'tau_k': 80*ms}

#group_input.active = True
#group_e_1.delta_gk[chg_adapt_neuron] = 2*nS; group_e_1.tau_k[chg_adapt_neuron] = 40*ms
#net.run(simu_time2, report = 'text', profile=False) #,namespace={'tau_k': 80*ms}

#%%
import datetime
now = datetime.datetime.now()
#spk_1_tstep = np.round(spk_1.t/(0.1*ms)).astype(int)
spk_tstep_e = np.round(spk_e.t/(0.1*ms)).astype(int)
spk_tstep_i = np.round(spk_i.t/(0.1*ms)).astype(int)

#param_1 = {'delta_gk':10,'tau_s_di':3,'tau_s_de':5,'ie_ratio':3.1725,'sti_onset':'10s'}

param_new = {'delta_gk':delta_gk_,
         #'new_delta_gk':2,'new_tau_k':40,
         'tau_s_di':tau_s_di_,
         'tau_s_de' : tau_s_de_,
         'ie_ratio':ie_ratio_,
         #'mean_J_ee': ijwd.mean_J_ee,
         #'chg_adapt_range':6,
         
         #'p_ee':p_ee,
         'scale_ee_1': scale_ee_1,
         'scale_ei_1': scale_ei_1,
         #'scale_ie_1': scale_ie_1,
         'scale_ii_1': scale_ii_1,
         'chg_adapt_neuron':chg_adapt_neuron,
         'simutime':simu_time/ms}
param = {**param, **param_new}

#param_12 = {'scale_e_12':scale_e_12[i], 'scale_e_21':scale_e_21[j]}

#data = {'param':param_12, 'a1':{'param':param_1,'e':{'i':spk_1.i[:],'t':spk_1_tstep}},
#                            'a2':{'param':param_2,'e':{'i':spk_2.i[:],'t':spk_2_tstep}}}
data = {'datetime':now.strftime("%Y-%m-%d %H:%M:%S"),
        'a1':{'param':param,'ge':{'i':spk_e.i[:],'t':spk_tstep_e},
        'gi':{'i':spk_i.i[:],'t':spk_tstep_i}}}

with open('data%d.file'%loop_num, 'wb') as file:
    pickle.dump(data, file)
#%%
#import load_data_dict
#import post_analysis as psa
##%%
#spkdata2 = load_data_dict.data_multiarea(data)
#
##spkdata.a1.e.t = spkdata.a1.e.t*0.1*ms
#spkdata2.a2.e.t = spkdata2.a2.e.t*0.1*ms
#
##%%
#spkrate3, centre_ind3, jump_size3, jump_dist3 = psa.get_rate_centre_jumpdist(spkdata2.a2.e, starttime=0*ms, endtime=1000*ms, binforrate=5*ms, sample_interval=1*ms, slide_interval = 1, jump_interval = 15, dt=0.1*ms)
#spkrate3 = psa.overlap_centreandspike(centre_ind3, spkrate3, show_trajectory = False)
#anititle=''#'change adaptation in region near centre to %.1fnS at 1000 ms\ndelta_gk: %.1fnS; tau_s_di: %.1fms; ie_ratio: %.3f'%(spke1.param['new_delta_gk'], spke1.param['delta_gk'], spke1.param['tau_s_di'], spke1.param['ie_ratio'])
#ani3 = psa.show_pattern(spkrate3, spkrate2=0, area_num = 1, frames = 500, start_time = 0, anititle=anititle)
##%%
#net.restore('state1')






