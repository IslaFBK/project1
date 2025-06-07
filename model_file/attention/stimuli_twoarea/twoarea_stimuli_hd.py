#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 18:00:55 2020

@author: shni2598
"""

'''
add two input stimuli into sensory area. The input fring rate have Gaussian profile 
'''
from connection import pre_process
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
prefs.codegen.target = 'cython'

dir_cache = '/headnode1/shni2598/brian2/brian_cache/cache%s' %sys.argv[1]
prefs.codegen.runtime.cython.cache_dir = dir_cache
prefs.codegen.max_cache_dir_size = 25
#%%poisson_stimuli
print('start run')
Ne = 63*63
Ni = 1024
width = 62
delay_v1v2 = [1,5] # spike transmission delay between two areas

n_inter_area_1 = int(0.5*Ne) # number of excitatory neurons form inter-areal projections in area 1
n_inter_area_2 = int(0.5*Ne) # number of excitatory neurons form inter-areal projections in area 2

syn_per_inter_neuron_1 = int(1.125*10**6/n_inter_area_1)


inter_e_neuron_1 = np.random.choice(Ne, n_inter_area_1, replace = False)
inter_e_neuron_1 = np.sort(inter_e_neuron_1)

inter_e_neuron_2 = np.random.choice(Ne, n_inter_area_2, replace = False)
inter_e_neuron_2 = np.sort(inter_e_neuron_2)

lattice_ex_1 = cn.coordination.makelattice(int(Ne**0.5), width, [0, 0])
lattice_ex_2 = cn.coordination.makelattice(int(Ne**0.5), width, [0, 0])

lattice_in_1 = cn.quasi_lattice.lattice(width, Ni)
lattice_in_2 = cn.quasi_lattice.lattice(width, Ni)


#%%
periodic_boundary = True; interarea_dist = 0;

peak_p_v1e_v2e = 0.85; tau_d_v1e_v2e = 5
i_v1e_v2e, j_v1e_v2e, dist_v1e_v2e = cn.connect_2lattice.expo_decay(lattice_ex_1, lattice_ex_2, inter_e_neuron_1, \
                                                      width, periodic_boundary, interarea_dist, \
                                                      peak_p_v1e_v2e, tau_d_v1e_v2e, src_equal_trg = False, self_cnt = False)

peak_p_v1e_v2i = 0.85; tau_d_v1e_v2i = 5
i_v1e_v2i, j_v1e_v2i, dist_v1e_v2i = cn.connect_2lattice.expo_decay(lattice_ex_1, lattice_in_2, inter_e_neuron_1, \
                                                      width, periodic_boundary, interarea_dist, \
                                                      peak_p_v1e_v2i, tau_d_v1e_v2i, src_equal_trg = False, self_cnt = False)

peak_p_v2e_v1e = 0.4; tau_d_v2e_v1e = 8
i_v2e_v1e, j_v2e_v1e, dist_v2e_v1e = cn.connect_2lattice.expo_decay(lattice_ex_2, lattice_ex_1, inter_e_neuron_2, \
                                                      width, periodic_boundary, interarea_dist, \
                                                      peak_p_v2e_v1e, tau_d_v2e_v1e, src_equal_trg = False, self_cnt = False)

peak_p_v2e_v1i = 0.4; tau_d_v2e_v1i = 8
i_v2e_v1i, j_v2e_v1i, dist_v2e_v1i = cn.connect_2lattice.expo_decay(lattice_ex_2, lattice_in_1, inter_e_neuron_2, \
                                                      width, periodic_boundary, interarea_dist, \
                                                      peak_p_v2e_v1i, tau_d_v2e_v1i, src_equal_trg = False, self_cnt = False)

#%%
'''
ind_src = inter_e_neuron_2[749]
print(len(i_v2e_v1e)/(63*63))
ind_src = inter_e_neuron_1[1708]
print(len(i_v1e_v2e)/(63*63))
'''
#%%
'''
plt.figure()
ind_src = inter_e_neuron_1[28]
plt.plot(lattice2[j_e_v2[i_e_v1==ind_src]][:,0], lattice2[j_e_v2[i_e_v1==ind_src]][:,1], 'or')
plt.plot(lattice1[ind_src][0], lattice1[ind_src][1], 'ob')

plt.xlim(-31.5,94)
plt.ylim(-31.5,31.5)
plt.show()
#%%
# v2e to v1e
# plt.figure()
fig, [ax1,ax2] = plt.subplots(1,2,figsize=(14,6))
ind_src = inter_e_neuron_2[749]
ax1.plot(lattice_ex_1[j_v2e_v1e[i_v2e_v1e==ind_src]][:,0], lattice_ex_1[j_v2e_v1e[i_v2e_v1e==ind_src]][:,1], 'or', label='target')
ax1.legend()
ax2.plot(lattice_ex_2[ind_src][0], lattice_ex_2[ind_src][1], 'ob', label='source')
ax2.legend()
ax1.set_title('v1', fontsize=16); ax2.set_title('v2',fontsize=16)
ax1.set_xlim(-32.,32.)
ax1.set_ylim(-32.,32.)
ax2.set_xlim(-32.,32.)
ax2.set_ylim(-32.,32.)
plt.show()
#%%
# v1e to v2e
# plt.figure()
fig, [ax1,ax2] = plt.subplots(1,2,figsize=(14,6))
ind_src = inter_e_neuron_1[1708]
ax2.plot(lattice_ex_2[j_v1e_v2e[i_v1e_v2e==ind_src]][:,0], lattice_ex_2[j_v1e_v2e[i_v1e_v2e==ind_src]][:,1], 'or', label='target')
ax2.legend()
ax1.plot(lattice_ex_1[ind_src][0], lattice_ex_1[ind_src][1], 'ob', label='source')
ax1.legend()
ax1.set_title('v1', fontsize=16); ax2.set_title('v2',fontsize=16)
ax1.set_xlim(-32.,32.)
ax1.set_ylim(-32.,32.)
ax2.set_xlim(-32.,32.)
ax2.set_ylim(-32.,32.)
plt.show()
#%%
plt.figure()
ind_src = inter_e_neuron_2[34]
plt.plot(lattice_in_1[j_v2e_v1i[i_v2e_v1i==ind_src]][:,0], lattice_in_1[j_v2e_v1i[i_v2e_v1i==ind_src]][:,1], 'or')
plt.plot(lattice_ex_2[ind_src][0], lattice_ex_2[ind_src][1], 'ob')

plt.xlim(-31.5,94)
plt.ylim(-31.5,31.5)
plt.show()
#%%
plt.figure()
ind_src = inter_e_neuron_1[100]
plt.plot(lattice2[j_e_v2[i_e_v1==ind_src]][:,0], lattice2[j_e_v2[i_e_v1==ind_src]][:,1], 'or')
plt.plot(lattice1[ind_src][0], lattice1[ind_src][1], 'ob')

plt.xlim(-31.5,94)
plt.ylim(-31.5,31.5)
plt.show()
'''
#%%

ijwd1 = pre_process.get_ijwd(Ni=Ni)
scale_ee_1 = 0.7; scale_ei_1 = 0.7
ijwd1.mean_J_ee = 4*10**-3*scale_ee_1 # usiemens
ijwd1.sigma_J_ee = 1.9*10**-3*scale_ee_1 # usiemens
ijwd1.generate_ijw()
ijwd1.generate_d_rand()
ijwd1.w_ei = 5*10**(-3)#*scale_ei_1 #usiemens
ijwd1.w_ii = 25*10**(-3) #* 1.4 
#%%
ijwd2 = pre_process.get_ijwd(Ni=Ni)
scale_ee_2 = 1.4; scale_ei_2 = 1.4
ijwd2.mean_J_ee = 4*10**-3*scale_ee_2 # usiemens
ijwd2.sigma_J_ee = 1.9*10**-3*scale_ee_2 # usiemens
ijwd2.generate_ijw()
ijwd2.generate_d_rand()
ijwd2.w_ei = 5*10**(-3)*scale_ei_2*1.05 #usiemens
ijwd2.w_ii = 25*10**(-3) * 1.4

#%%
#pickle.dump(ijwd1, open('ijwd1.file','wb'))
#pickle.dump(ijwd2, open('ijwd2.file','wb'))
#%%
#np.mean(ijwd2.w_ee[:])
#%%

start_scope()

neuronmodel_e = '''
dv/dt = (1/C)*(-g_l*(v - v_l) + (-g_k)*(v - v_k) + (-g_I)*(v - v_rev_I) +(-g_E-g_E_inter-g_E_extnl)*(v - v_rev_E) + I_extnl) : volt (unless refractory)
dg_k/dt = -g_k/tau_k :siemens

g_I : siemens
g_E : siemens
g_E_inter : siemens
I_extnl : amp

g_E_extnl =  w_extnl*s_extnl: siemens
w_extnl: siemens
ds_extnl/dt = -s_extnl/tau_s_de + x : 1
dx/dt = -x/tau_x_re : Hz
'''
#neuronmodel_e = '''
#dv/dt = (1/C)*(-g_l*(v - v_l) + (-g_k)*(v - v_k) + (-g_I)*(v - v_rev_I) +(-g_E-g_E_inter)*(v - v_rev_E) + I_extnl) : volt (unless refractory)
#dg_k/dt = -g_k/tau_k :siemens
#
#g_I : siemens
#g_E : siemens
#g_E_inter : siemens
#I_extnl : amp
#'''

neuronmodel_i = '''
dv/dt = (1/C)*(-g_l*(v - v_l) + (-g_I)*(v - v_rev_I) +(-g_E-g_E_inter)*(v - v_rev_E) + I_extnl) : volt (unless refractory)
g_I : siemens
g_E : siemens
g_E_inter : siemens
I_extnl : amp
'''

synapse_e = '''
w: siemens
g_E_post = w*s : siemens (summed)
ds/dt = -s/tau_s_de + rect_puls*(1 - s) : 1 (clock-driven)
rect_puls = (1/tau_s_re)*effect : Hz
effect : integer
'''
synapse_i = '''
w: siemens
g_I_post = w*s : siemens (summed)
ds/dt = -s/tau_s_di + rect_puls*(1 - s) : 1 (clock-driven)
rect_puls = (1/tau_s_ri)*effect : Hz
effect : integer
'''


synapse_e_v1v2 = '''
w: siemens
g_E_inter_post = w*s : siemens (summed)
ds/dt = -s/tau_s_de + rect_puls*(1 - s) : 1 (clock-driven)
rect_puls = (1/tau_s_re)*effect : Hz
effect : integer
'''
synapse_extnl = '''


'''
#synapse_ext = '''
#w: siemens
#g_ext_post = w*s : siemens (summed)
#ds/dt = -s/tau_s_de + rect_puls : 1 (clock-driven)
#rect_puls = (1/tau_s_re)*effect : Hz
#effect : 1
#
#''' group_e_v1, group_i_v1, syn_ee_v1, syn_ei_v1, syn_ie_v1, syn_ii_v1, vexi, spike_e
#%%
group_e_v1 =NeuronGroup(ijwd1.Ne, model=neuronmodel_e,
                     threshold='v>v_threshold', method='euler',
                     reset='''v = v_reset
                              g_k += delta_gk''', refractory='(t-lastspike)<t_ref')

group_i_v1 =NeuronGroup(ijwd1.Ni, model=neuronmodel_i,
                     threshold='v>v_threshold', method='euler',
                     reset='v = v_reset', refractory='(t-lastspike)<t_ref')

syn_ee_v1 = Synapses(group_e_v1, group_e_v1, model=synapse_e, method='euler',
                  on_pre={'up':'effect += 1', 'down': 'effect -= 1'})
syn_ei_v1 = Synapses(group_e_v1, group_i_v1, model=synapse_e, method='euler',
                  on_pre={'up':'effect += 1', 'down': 'effect -= 1'})
syn_ie_v1= Synapses(group_i_v1, group_e_v1, model=synapse_i, method='euler',
                  on_pre={'up':'effect += 1', 'down': 'effect -= 1'})
syn_ii_v1 = Synapses(group_i_v1, group_i_v1, model=synapse_i, method='euler',
                  on_pre={'up':'effect += 1', 'down': 'effect -= 1'})


group_input = PoissonGroup(ijwd1.Ne, psti.input_spkrate(maxrate = 800, sig=6, position=[[-31.5, -31.5],[0, 0]])*Hz)
syn_extnl_e_v1 = Synapses(group_input, group_e_v1, method='euler', on_pre='x_post += 1/tau_x_re')
#%%
syn_ee_v1.connect(i=ijwd1.i_ee, j=ijwd1.j_ee)
syn_ei_v1.connect(i=ijwd1.i_ei, j=ijwd1.j_ei)
syn_ie_v1.connect(i=ijwd1.i_ie, j=ijwd1.j_ie)
syn_ii_v1.connect(i=ijwd1.i_ii, j=ijwd1.j_ii)
syn_extnl_e_v1.connect('i==j')
group_e_v1.w_extnl = 2*nS
#%%
syn_ee_v1.w = ijwd1.w_ee*usiemens
syn_ei_v1.w = ijwd1.w_ei*usiemens #5*nS
syn_ii_v1.w = ijwd1.w_ii*usiemens #25*nS
syn_ie_v1.w = ijwd1.w_ie*usiemens
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
#ijwd1.generate_d_dist()
#ijwd1.generate_d_rand()
syn_ee_v1 = set_delay(syn_ee_v1, ijwd1.d_ee)
syn_ie_v1 = set_delay(syn_ie_v1, ijwd1.d_ie)
syn_ei_v1 = set_delay(syn_ei_v1, ijwd1.d_ei)
syn_ii_v1 = set_delay(syn_ii_v1, ijwd1.d_ii)
#syn_pois_e = set_delay(syn_pois_e)
#syn_pois_i = set_delay(syn_pois_i)

#syn_ee_v1.up.delay = 3*ms; syn_ee_v1.down.delay = 4*ms; 
#syn_ie_v1.up.delay = 3*ms; syn_ie_v1.down.delay = 4*ms; 
#syn_ei_v1.up.delay = 3*ms; syn_ei_v1.down.delay = 4*ms; 
#syn_ii_v1.up.delay = 3*ms; syn_ii_v1.down.delay = 4*ms; 

syn_ee_v1.effect = 0
syn_ie_v1.effect = 0
syn_ei_v1.effect = 0
syn_ii_v1.effect = 0
#syn_pois_e.effect = 0
#syn_pois_i.effect = 0
#group_e_v1.v = np.random.random(Ne)*35*mV-85*mV
#group_i_v1.v = np.random.random(Ni)*35*mV-85*mV
group_e_v1.v = np.random.random(ijwd1.Ne)*35*mV-85*mV
group_i_v1.v = np.random.random(ijwd1.Ni)*35*mV-85*mV
#group_e_v1.v = np.random.random(ijwd1.Ne)*10*mV-60*mV
#group_i_v1.v = np.random.random(ijwd1.Ni)*10*mV-60*mV
group_e_v1.I_extnl = 0.51*nA
group_i_v1.I_extnl = 0.60*nA

#%%
group_e_v2 =NeuronGroup(ijwd2.Ne, model=neuronmodel_e,
                     threshold='v>v_threshold', method='euler',
                     reset='''v = v_reset
                              g_k += delta_gk''', refractory='(t-lastspike)<t_ref')

group_i_v2 =NeuronGroup(ijwd2.Ni, model=neuronmodel_i,
                     threshold='v>v_threshold', method='euler',
                     reset='v = v_reset', refractory='(t-lastspike)<t_ref')

syn_ee_v2 = Synapses(group_e_v2, group_e_v2, model=synapse_e, method='euler',
                  on_pre={'up':'effect += 1', 'down': 'effect -= 1'})
syn_ei_v2 = Synapses(group_e_v2, group_i_v2, model=synapse_e, method='euler',
                  on_pre={'up':'effect += 1', 'down': 'effect -= 1'})
syn_ie_v2 = Synapses(group_i_v2, group_e_v2, model=synapse_i, method='euler',
                  on_pre={'up':'effect += 1', 'down': 'effect -= 1'})
syn_ii_v2 = Synapses(group_i_v2, group_i_v2, model=synapse_i, method='euler',
                  on_pre={'up':'effect += 1', 'down': 'effect -= 1'})

#%%
syn_ee_v2.connect(i=ijwd2.i_ee, j=ijwd2.j_ee)
syn_ei_v2.connect(i=ijwd2.i_ei, j=ijwd2.j_ei)
syn_ie_v2.connect(i=ijwd2.i_ie, j=ijwd2.j_ie)
syn_ii_v2.connect(i=ijwd2.i_ii, j=ijwd2.j_ii)

#%%
syn_ee_v2.w = ijwd2.w_ee*usiemens
syn_ei_v2.w = ijwd2.w_ei*usiemens #5*nS
syn_ii_v2.w = ijwd2.w_ii*usiemens #25*nS
syn_ie_v2.w = ijwd2.w_ie*usiemens
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
#ijwd2.generate_d_dist()
#ijwd2.generate_d_rand()
syn_ee_v2 = set_delay(syn_ee_v2, ijwd2.d_ee)
syn_ie_v2 = set_delay(syn_ie_v2, ijwd2.d_ie)
syn_ei_v2 = set_delay(syn_ei_v2, ijwd2.d_ei)
syn_ii_v2 = set_delay(syn_ii_v2, ijwd2.d_ii)
#syn_pois_e = set_delay(syn_pois_e)
#syn_pois_i = set_delay(syn_pois_i)

#syn_ee_v2.up.delay = 3*ms; syn_ee_v2.down.delay = 4*ms; 
#syn_ie_v2.up.delay = 3*ms; syn_ie_v2.down.delay = 4*ms; 
#syn_ei_v2.up.delay = 3*ms; syn_ei_v2.down.delay = 4*ms; 
#syn_ii_v2.up.delay = 3*ms; syn_ii_v2.down.delay = 4*ms; 

syn_ee_v2.effect = 0
syn_ie_v2.effect = 0
syn_ei_v2.effect = 0
syn_ii_v2.effect = 0
#syn_pois_e.effect = 0
#syn_pois_i.effect = 0
#group_e_v2.v = np.random.random(Ne)*35*mV-85*mV
#group_i_v2.v = np.random.random(Ni)*35*mV-85*mV
group_e_v2.v = np.random.random(ijwd2.Ne)*35*mV-85*mV
group_i_v2.v = np.random.random(ijwd2.Ni)*35*mV-85*mV
#group_e_v2.v = np.random.random(ijwd2.Ne)*10*mV-60*mV
#group_i_v2.v = np.random.random(ijwd2.Ni)*10*mV-60*mV
group_e_v2.I_extnl = 0.51*nA
group_i_v2.I_extnl = 0.60*nA

#%%

scale_e_12 = np.concatenate([np.array([0]),np.arange(0.2,1.4,0.1)])
scale_e_21 = np.concatenate([np.array([0]),np.arange(0.2,1.4,0.1)])

sys_argv = int(sys.argv[1])
loop_num = -1
for i in range(len(scale_e_12)):
    for j in range(len(scale_e_21)):
        loop_num += 1
        if loop_num == sys_argv:
            print(i,j)
            break
    else:
        continue
    break

#%%
'''
scale = np.array([0, 0.2, 0.6, 0.9, 1.1])
sys_argv = int(sys.argv[1])
loop_num = -1
for i in range(len(scale)):
    loop_num += 1
    if loop_num == sys_argv:
        print(i)
        break
'''
#%%
syn_ee_v1v2 = Synapses(group_e_v1, group_e_v2, model=synapse_e_v1v2, method='euler',
                  on_pre={'up':'effect += 1', 'down': 'effect -= 1'})


syn_ee_v1v2.connect(i=i_v1e_v2e, j=j_v1e_v2e)
syn_ee_v1v2.w = np.random.normal(4, 1.9, len(i_v1e_v2e))*nS*scale_e_12[i]


d_ee_v1v2 = np.random.uniform(delay_v1v2[0], delay_v1v2[1], len(i_v1e_v2e))
syn_ee_v1v2 = set_delay(syn_ee_v1v2, d_ee_v1v2)

syn_ee_v1v2.effect = 0
#%%
#i_v1e_v2e, j_v1e_v2e,
#i_v1e_v2i, j_v1e_v2i,
#i_v2e_v1e, j_v2e_v1e,
#i_v2e_v1i, j_v2e_v1i,


syn_ei_v1v2 = Synapses(group_e_v1, group_i_v2, model=synapse_e_v1v2, method='euler',
                  on_pre={'up':'effect += 1', 'down': 'effect -= 1'})


syn_ei_v1v2.connect(i=i_v1e_v2i, j=j_v1e_v2i)
syn_ei_v1v2.w = 5*nS*scale_e_12[i] # np.random.normal(4, 1.9, len(i_v1e_v2i))*nS


d_ei_v1v2 = np.random.uniform(delay_v1v2[0], delay_v1v2[1], len(i_v1e_v2i))
syn_ei_v1v2 = set_delay(syn_ei_v1v2, d_ei_v1v2)
del d_ei_v1v2

syn_ei_v1v2.effect = 0

#%%
syn_ee_v2v1 = Synapses(group_e_v2, group_e_v1, model=synapse_e_v1v2, method='euler',
                  on_pre={'up':'effect += 1', 'down': 'effect -= 1'})


syn_ee_v2v1.connect(i=i_v2e_v1e, j=j_v2e_v1e)
syn_ee_v2v1.w = np.random.normal(4, 1.9, len(i_v2e_v1e))*nS*scale_e_21[j]


d_ee_v2v1 = np.random.uniform(delay_v1v2[0], delay_v1v2[1], len(i_v2e_v1e))
syn_ee_v2v1 = set_delay(syn_ee_v2v1, d_ee_v2v1)
del d_ee_v2v1

syn_ee_v2v1.effect = 0

#%%
syn_ei_v2v1 = Synapses(group_e_v2, group_i_v1, model=synapse_e_v1v2, method='euler',
                  on_pre={'up':'effect += 1', 'down': 'effect -= 1'})


syn_ei_v2v1.connect(i=i_v2e_v1i, j=j_v2e_v1i)
syn_ei_v2v1.w = 5*nS*scale_e_21[j] # np.random.normal(4, 1.9, len(i_v2e_v1i))*nS


d_ei_v2v1 = np.random.uniform(delay_v1v2[0], delay_v1v2[1], len(i_v2e_v1i))
syn_ei_v2v1 = set_delay(syn_ei_v2v1, d_ei_v2v1)
del d_ei_v2v1

syn_ei_v2v1.effect = 0

#%%
'''
syn_ee_v1v2.w = 0.5*np.random.normal(4, 1.9, len(i_v1e_v2e))*nS #0*nS 
syn_ei_v1v2.w = 0.5*5*nS #np.random.normal(4, 1.9, len(i_v1e_v2i))*nS #0*nS 
syn_ee_v2v1.w = 0.5*np.random.normal(4, 1.9, len(i_v2e_v1e))*nS #0*nS  #0*nS 
syn_ei_v2v1.w = 0.5*5*nS # np.random.normal(4, 1.9, len(i_v2e_v1i))*nS #0*nS 

group_e_v2.I_extnl = 0.51*nA
group_i_v2.I_extnl = 0.6*nA
'''
#%%
#vexi = StateMonitor(group_e, ('v','delta_gk'), record=np.arange(net_ijwd.Ne),dt=0.5*ms,name='v_moni_1')

#v_v1 = StateMonitor(group_e_v1, ('v'), record=True,dt=0.5*ms,name='v_v1')
#v_v2 = StateMonitor(group_e_v2, ('v'), record=True,dt=0.5*ms,name='v_v2')

spk_v1 = SpikeMonitor(group_e_v1, record = True)
spk_v2 = SpikeMonitor(group_e_v2, record = True)

#spike_e = SpikeMonitor(group_e, record = True)
#rate_e = PopulationRateMonitor(group_e)

#%%

net = Network(collect())
net.store('state1')
#%%
print('ie_w: %fnsiemens' %(syn_ie_v1.w[0]/nsiemens))
#Ne = 63*63; Ni = 1000;
C = 0.25*nF # capacitance
g_l = 16.7*nS # leak capacitance
v_l = -70*mV # leak voltage
v_threshold = -50*mV
v_reset = -60*mV
v_rev_I = -80*mV
v_rev_E = 0*mV
v_k = -85*mV
tau_k = 80*ms# 80*ms
delta_gk = 10*nS #10*nS
t_ref = 4*ms # refractory period

tau_s_de = 5*ms
tau_s_di = 3*ms
tau_s_re = 1*ms
tau_s_ri = 1*ms
tau_x_re = 1*ms
tic = time.perf_counter()
#seed(10)
simu_time1 = 1000*ms
simu_time2 = 5000*ms

group_input.active = False
net.run(simu_time1, report = 'text', profile=False) #,namespace={'tau_k': 80*ms}
group_input.active = True
net.run(simu_time2, report = 'text', profile=False) #,namespace={'tau_k': 80*ms}

print('%ds'%(time.perf_counter() - tic))
#%%
import scipy.io as sio
#%%
#loop_num = 1
sio.savemat('indexv1%s.mat'%loop_num, {'ind':spk_v1.i[:]})
spk_tstep = np.round(spk_v1.t/(0.1*ms)).astype(int)
sio.savemat('timev1%s.mat'%loop_num, {'time':spk_tstep})

sio.savemat('indexv2%s.mat'%loop_num, {'ind':spk_v2.i[:]})
spk_tstep = np.round(spk_v2.t/(0.1*ms)).astype(int)
sio.savemat('timev2%s.mat'%loop_num, {'time':spk_tstep})
#%%
'''
vrec1 = v_v1.v[:] #extract membrane potential data
vrec1 = vrec1.reshape(63,63,vrec1.shape[1])
vrec1 = vrec1/mV
vrec2 = v_v2.v[:] #extract membrane potential data
vrec2 = vrec2.reshape(63,63,vrec2.shape[1])
vrec2 = vrec2/mV
#%% plot the membrane potential of the first neuron 
#figure()
#plot(np.arange(5000), vrec[0,0,:]/mV)
#%% make animation of the dynamics of membrane potential
fig, [ax1,ax2]= plt.subplots(1,2)
value1=ax1.matshow(vrec1[:,:,0])
value2=ax2.matshow(vrec2[:,:,0])
def updatev(i):
    value1.set_array(vrec1[:,:,i])
    value2.set_array(vrec2[:,:,i])
    return value1, value2

cbaxes1 = fig.add_axes([0.1, 0.1, 0.35, 0.03]) 
cbaxes2 = fig.add_axes([0.55, 0.1, 0.35, 0.03]) 
cb1 = fig.colorbar(value1, cax = cbaxes1, orientation='horizontal') 
cb2 = fig.colorbar(value2, cax = cbaxes2, orientation='horizontal') 
value1.set_clim(vmin=-85, vmax=-50)
value2.set_clim(vmin=-85, vmax=-50)  

ani=animation.FuncAnimation(fig, updatev, frames=int(800), interval=10)   


'''
#%%
"""
analy = post_analysis.analysis()

#analy.get_spike_rate_animation

#anly_tmp = post_analysis.analysis()
spk1 = analy.get_spike_rate(spk_v1, start_time=0*ms, end_time=simu_time, sample_interval = 1*ms, \
                            indiv_rate = True, popu_rate = False,\
                            n_neuron = 3969, window = 10*ms, dt = 0.1*ms)
spk2 = analy.get_spike_rate(spk_v2, start_time=0*ms, end_time=simu_time, sample_interval = 1*ms, \
                            indiv_rate = True, popu_rate = False,
                            n_neuron = 3969, window = 10*ms, dt = 0.1*ms)

centre_ind1, jump_size1, jump_dist1 = analy.get_centre_mass(spk1, slide_interval=1, jump_interval=15)
centre_ind2, jump_size2, jump_dist2 = analy.get_centre_mass(spk2, slide_interval=1, jump_interval=15)

spk1 = analy.overlap_centreandspike(centre_ind1, spk1[:,:,:])
spk2 = analy.overlap_centreandspike(centre_ind2, spk2[:,:,:])

#centre = analy.get_centre_mass(spk)
#spk = analy.overlap_centreandspike(centre, spk)
#%%
'''
show trajectory of centre of mass of pattern 
'''
fig, [ax1,ax2]= plt.subplots(1,2)
fig.suptitle('sensory to association strength: %.2f * default\nassociation to sensory strength: %.2f * default'
             %(scale_e_12[i],scale_e_21[j]))
ax1.set_title('sensory')
ax2.set_title('association')

cmap=plt.cm.get_cmap('Blues', 7) # viridis Blues
cmap.set_under('red')
bounds = [0, 1, 2, 3, 4, 5, 6]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cbaxes = fig.add_axes([0.2, 0.15, 0.6, 0.03]) 

cb = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap,
                                norm=norm,
                                boundaries=bounds,
                                ticks=bounds,
                                spacing='proportional',
                                orientation='horizontal') #horizontal vertical
cb.set_label('number of spikes')

value1=ax1.matshow(spk1[:,:,-1], cmap=cb.cmap)
value2=ax2.matshow(spk2[:,:,-1], cmap=cb.cmap)

value1.set_clim(vmin=0, vmax=6)
value2.set_clim(vmin=0, vmax=6)
ax1.axis('off')
ax2.axis('off')
fig.savefig("traj_interarea_strength%d_%.1f%.1f.png"%(loop_num,scale_e_12[i],scale_e_21[j]))
#%%
'''
show pattern animation
'''
fig, [ax1,ax2]= plt.subplots(1,2)
fig.suptitle('sensory to association strength: %.2f\nassociation to sensory strength: %.2f'
             %(scale_e_12[i],scale_e_21[j]))
ax1.set_title('sensory')
ax2.set_title('association')

cmap=plt.cm.get_cmap('Blues', 7) # viridis Blues
cmap.set_under('red')
bounds = [0, 1, 2, 3, 4, 5, 6]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cbaxes = fig.add_axes([0.2, 0.15, 0.6, 0.03]) 

cb = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap,
                                norm=norm,
                                boundaries=bounds,
                                ticks=bounds,
                                spacing='proportional',
                                orientation='horizontal') #horizontal vertical
cb.set_label('number of spikes')

titleaxes = fig.add_axes([0.3, 0.75, 0.4, 0.05])
titleaxes.axis('off')
title = titleaxes.text(0.5,0.95, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=titleaxes.transAxes, ha="center")
time_title = np.arange(spk1.shape[2])

value1=ax1.matshow(spk1[:,:,:], cmap=cb.cmap)
value2=ax2.matshow(spk2[:,:,:], cmap=cb.cmap)
def updatev(i):
    value1.set_array(spk1[:,:,i])
    value2.set_array(spk2[:,:,i])
    title.set_text(u"time: {} ms".format(time_title[i]))
    return value1, value2

#cbaxes1 = fig.add_axes([0.1, 0.1, 0.35, 0.03]) 
#cbaxes2 = fig.add_axes([0.55, 0.1, 0.35, 0.03]) 
#cb1 = fig.colorbar(value1, cax = cbaxes1, orientation='horizontal', ticks=[0,1,2,3,4]) 
#cb2 = fig.colorbar(value2, cax = cbaxes2, orientation='horizontal', ticks=[0,1,2,3,4]) 
value1.set_clim(vmin=0, vmax=6)
value2.set_clim(vmin=0, vmax=6)
ax1.axis('off')
ax2.axis('off')
ani=animation.FuncAnimation(fig, updatev, frames=2000, interval=30)    # frames=spk1.shape[2]
#%%
ani.save("interarea_strength%d_%.1f%.1f.mp4"%(loop_num,scale_e_12[i],scale_e_21[j]))

#%%
'''
plt.figure()
plt.plot(spk_v1.t[:]/ms, spk_v1.i[:],'.')
'''
#%%

'''
net.restore('state1')
'''
#%%
'''
fig, ax1= plt.subplots(1,1)


value1=ax1.matshow(spk[:,:,0], cmap=plt.cm.get_cmap('viridis', 5))
ax1.axis('off')
#cmap=plt.cm.get_cmap('binary', 3)

def updatev(iii):
    value1.set_array(spk[:,:,iii])
    return value1
#
cbaxes = fig.add_axes([0.1, 0.06, 0.8, 0.03]) 
cb=fig.colorbar(value1, cax = cbaxes, orientation='horizontal',ticks=[0,1,2,3,4]) 
value1.set_clim(vmin=0, vmax=4)

#fig.suptitle('ie_ratio:%s\ncentre of pattern and number of spikes per 10ms\n ' %ie_ratio)
ani=animation.FuncAnimation(fig, updatev, frames=spk.shape[2], interval=0.1)  
'''
"""