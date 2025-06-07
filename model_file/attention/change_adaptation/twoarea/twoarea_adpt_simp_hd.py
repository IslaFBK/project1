# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:17:27 2020

@author: nishe
"""

'''
connectivity in both layers are simple
add two input stimuli into sensory area. The input fring rate have Gaussian profile 
change the adaptation locally in layer 2 to see how it will effect the dynamics of layer 1
'''
import matplotlib as mpl
mpl.use('Agg')
from connection import pre_process
from connection import connect_interareas 
import connection as cn
#from analysis import post_analysis
import poisson_stimuli as psti
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
import time

import brian2.numpy_ as np
import brian2.only
from brian2.only import *
#import pickle
import sys
prefs.codegen.target = 'cython'

dir_cache = './cache/cache%s' %sys.argv[1]
prefs.codegen.runtime.cython.cache_dir = dir_cache
prefs.codegen.max_cache_dir_size = 100
#%%
print('start running')
#%%
scale_e_12 = np.concatenate([np.array([0]),np.arange(0.2,1.4,0.1)])
scale_e_21 = np.concatenate([np.array([0]),np.arange(0.2,1.4,0.1)])

#scale_e_12 = np.array([0, 0, 0])
#scale_e_21 = np.array([0, 0, 0])
sys_argv = int(sys.argv[1])
loop_num = -1
for i in range(len(scale_e_12)):
    for j in range(len(scale_e_21)):
        for ie_ratio_1 in 3.1725*np.arange(1,1.05,0.02):
            for chg_adapt_range in np.arange(7,10):
                for new_delta_gk in np.arange(3):
                    loop_num += 1
                    if loop_num == sys_argv:
                        print(i,j,loop_num)
                        break
                else:continue
                break
            else:continue
            break
        else:continue
        break
    else:continue
    break
#%%
Ne = 3969; Ni = 1024
#%%
scale_ee_1 = 0.7; scale_ei_1 = 0.7

ijwd1 = pre_process.get_ijwd(Ni=Ni)
ijwd1.w_ee_dist = 'normal'
ijwd1.hybrid = 0.
ijwd1.cn_scale_weight = 1
ijwd1.cn_scale_wire = 1
ijwd1.iter_num=1

ijwd1.ie_ratio = ie_ratio_1 # 3.1725
ijwd1.mean_J_ee = 4*10**-3*scale_ee_1 # usiemens
ijwd1.sigma_J_ee = 1.9*10**-3*scale_ee_1 # usiemens
ijwd1.change_dependent_para()
ijwd1.generate_ijw()
ijwd1.generate_d_rand()
ijwd1.w_ei = 5*10**(-3)#*scale_ei_1 #usiemens
ijwd1.w_ii = 25*10**(-3) #* 1.4 
#%%
ijwd2 = pre_process.get_ijwd(Ni=Ni)
ijwd2.w_ee_dist = 'normal'
ijwd2.hybrid = 0.
ijwd2.cn_scale_weight = 1
ijwd2.cn_scale_wire = 1
ijwd2.iter_num=1

scale_ee_2 = 1.4; scale_ei_2 = 1.4
ijwd2.ie_ratio = 2.880
ijwd2.mean_J_ee = 4*10**-3*scale_ee_2 # usiemens
ijwd2.sigma_J_ee = 1.9*10**-3*scale_ee_2 # usiemens
ijwd2.change_dependent_para()
ijwd2.generate_ijw()
ijwd2.generate_d_rand()
ijwd2.w_ei = 5*10**(-3)*scale_ei_2*1.05 #usiemens
ijwd2.w_ii = 25*10**(-3) * 1.4
#%%
ijd_inter = connect_interareas.get_inter_ijwd()
ijd_inter.generate_ij(ijwd1, ijwd2)
ijd_inter.generate_d_rand()
#%%
'''
ind_src = inter_e_neuron_2[749]
print(len(i_v2e_v1e)/(63*63))
ind_src = inter_e_neuron_1[1708]
print(len(i_v1e_v2e)/(63*63))
'''
#%%
'''
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Qt5Agg')
#%%
plt.figure()
ind_src = ijd_inter.inter_e_neuron_1[28]
plt.plot(ijwd2.lattice_ext[ijd_inter.j_1e_2e[ijd_inter.i_1e_2e==ind_src]][:,0], ijwd2.lattice_ext[ijd_inter.j_1e_2e[ijd_inter.i_1e_2e==ind_src]][:,1], 'or')
plt.plot(ijwd1.lattice_ext[ind_src][0], ijwd1.lattice_ext[ind_src][1], 'ob')

plt.xlim(-31.5,31.5)
plt.ylim(-31.5,31.5)
plt.show()
#%%
# v2e to v1e
# plt.figure()
fig, [ax1,ax2] = plt.subplots(1,2,figsize=(14,6))
ind_src = ijd_inter.inter_e_neuron_2[749]
ax1.plot(ijwd1.lattice_ext[ijd_inter.j_2e_1e[ijd_inter.i_2e_1e==ind_src]][:,0], ijwd1.lattice_ext[ijd_inter.j_2e_1e[ijd_inter.i_2e_1e==ind_src]][:,1], 'or', label='target')
ax1.legend()
ax2.plot(ijwd2.lattice_ext[ind_src][0], ijwd2.lattice_ext[ind_src][1], 'ob', label='source')
ax2.legend()
ax1.set_title('v1', fontsize=16); ax2.set_title('v2',fontsize=16)
ax1.set_xlim(-32.,32.)
ax1.set_ylim(-32.,32.)
ax2.set_xlim(-32.,32.)
ax2.set_ylim(-32.,32.)
plt.show()
'''
#%%
chg_adapt_loca = [0, 0]
#chg_adapt_range = 6
chg_adapt_neuron = cn.findnearbyneuron.findnearbyneuron(ijwd2.lattice_ext, chg_adapt_loca, chg_adapt_range, ijwd2.width)

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

synapse_e_12 = '''
w: siemens
g_E_inter_post = w*s : siemens (summed)
ds/dt = -s/tau_s_de + rect_puls*(1 - s) : 1 (clock-driven)
rect_puls = (1/tau_s_re)*effect : Hz
effect : integer
tau_s_de : second
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
group_e_1 =NeuronGroup(ijwd1.Ne, model=neuronmodel_e,
                     threshold='v>v_threshold', method='euler',
                     reset='''v = v_reset
                              g_k += delta_gk''', refractory='(t-lastspike)<t_ref')

group_i_1 =NeuronGroup(ijwd1.Ni, model=neuronmodel_i,
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

stim_rate = psti.input_spkrate(maxrate = [800], sig=[6], position=[[0, 0]])*Hz
group_input = PoissonGroup(ijwd1.Ne, stim_rate)
syn_extnl_e_1 = Synapses(group_input, group_e_1, method='euler', on_pre='x_post += 1/tau_x_re')
#%%
syn_ee_1.connect(i=ijwd1.i_ee, j=ijwd1.j_ee)
syn_ei_1.connect(i=ijwd1.i_ei, j=ijwd1.j_ei)
syn_ie_1.connect(i=ijwd1.i_ie, j=ijwd1.j_ie)
syn_ii_1.connect(i=ijwd1.i_ii, j=ijwd1.j_ii)
syn_extnl_e_1.connect('i==j')
group_e_1.w_extnl = 2*nS
#%%
syn_ee_1.w = ijwd1.w_ee*usiemens
syn_ei_1.w = ijwd1.w_ei*usiemens #5*nS
syn_ii_1.w = ijwd1.w_ii*usiemens #25*nS
syn_ie_1.w = ijwd1.w_ie*usiemens
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
syn_ee_1 = set_delay(syn_ee_1, ijwd1.d_ee)
syn_ie_1 = set_delay(syn_ie_1, ijwd1.d_ie)
syn_ei_1 = set_delay(syn_ei_1, ijwd1.d_ei)
syn_ii_1 = set_delay(syn_ii_1, ijwd1.d_ii)
#syn_pois_e = set_delay(syn_pois_e)
#syn_pois_i = set_delay(syn_pois_i)

syn_ee_1.effect = 0; syn_ee_1.tau_s_de = 5*ms
syn_ie_1.effect = 0; syn_ie_1.tau_s_di = 3*ms
syn_ei_1.effect = 0; syn_ei_1.tau_s_de = 5*ms
syn_ii_1.effect = 0; syn_ii_1.tau_s_di = 3*ms
#syn_pois_e.effect = 0
#syn_pois_i.effect = 0
#group_e_v1.v = np.random.random(Ne)*35*mV-85*mV
#group_i_v1.v = np.random.random(Ni)*35*mV-85*mV
group_e_1.v = np.random.random(ijwd1.Ne)*35*mV-85*mV
group_i_1.v = np.random.random(ijwd1.Ni)*35*mV-85*mV
group_e_1.delta_gk = 10*nS
group_e_1.tau_k = 80*ms
group_e_1.tau_s_de_extnl = 5*ms
#group_e_v1.v = np.random.random(ijwd1.Ne)*10*mV-60*mV
#group_i_v1.v = np.random.random(ijwd1.Ni)*10*mV-60*mV
group_e_1.I_extnl = 0.51*nA
group_i_1.I_extnl = 0.60*nA

#%%
group_e_2 =NeuronGroup(ijwd2.Ne, model=neuronmodel_e,
                     threshold='v>v_threshold', method='euler',
                     reset='''v = v_reset
                              g_k += delta_gk''', refractory='(t-lastspike)<t_ref')

group_i_2 =NeuronGroup(ijwd2.Ni, model=neuronmodel_i,
                     threshold='v>v_threshold', method='euler',
                     reset='v = v_reset', refractory='(t-lastspike)<t_ref')

syn_ee_2 = Synapses(group_e_2, group_e_2, model=synapse_e, method='euler',
                  on_pre={'up':'effect += 1', 'down': 'effect -= 1'})
syn_ei_2 = Synapses(group_e_2, group_i_2, model=synapse_e, method='euler',
                  on_pre={'up':'effect += 1', 'down': 'effect -= 1'})
syn_ie_2 = Synapses(group_i_2, group_e_2, model=synapse_i, method='euler',
                  on_pre={'up':'effect += 1', 'down': 'effect -= 1'})
syn_ii_2 = Synapses(group_i_2, group_i_2, model=synapse_i, method='euler',
                  on_pre={'up':'effect += 1', 'down': 'effect -= 1'})

#%%
syn_ee_2.connect(i=ijwd2.i_ee, j=ijwd2.j_ee)
syn_ei_2.connect(i=ijwd2.i_ei, j=ijwd2.j_ei)
syn_ie_2.connect(i=ijwd2.i_ie, j=ijwd2.j_ie)
syn_ii_2.connect(i=ijwd2.i_ii, j=ijwd2.j_ii)

#%%
syn_ee_2.w = ijwd2.w_ee*usiemens
syn_ei_2.w = ijwd2.w_ei*usiemens #5*nS
syn_ii_2.w = ijwd2.w_ii*usiemens #25*nS
syn_ie_2.w = ijwd2.w_ie*usiemens
#w_ext = 2*nS
#syn_pois_e.w = w_ext
#syn_pois_i.w = w_ext
#%%
#ijwd2.generate_d_dist()
#ijwd2.generate_d_rand()
syn_ee_2 = set_delay(syn_ee_2, ijwd2.d_ee)
syn_ie_2 = set_delay(syn_ie_2, ijwd2.d_ie)
syn_ei_2 = set_delay(syn_ei_2, ijwd2.d_ei)
syn_ii_2 = set_delay(syn_ii_2, ijwd2.d_ii)

#syn_pois_e = set_delay(syn_pois_e)
#syn_pois_i = set_delay(syn_pois_i)

#syn_ee_v2.up.delay = 3*ms; syn_ee_v2.down.delay = 4*ms; 
#syn_ie_v2.up.delay = 3*ms; syn_ie_v2.down.delay = 4*ms; 
#syn_ei_v2.up.delay = 3*ms; syn_ei_v2.down.delay = 4*ms; 
#syn_ii_v2.up.delay = 3*ms; syn_ii_v2.down.delay = 4*ms; 

syn_ee_2.effect = 0; syn_ee_2.tau_s_de = 4.0*ms
syn_ie_2.effect = 0; syn_ie_2.tau_s_di = 3.5*ms
syn_ei_2.effect = 0; syn_ei_2.tau_s_de = 4.0*ms
syn_ii_2.effect = 0; syn_ii_2.tau_s_di = 3.5*ms

#syn_pois_e.effect = 0
#syn_pois_i.effect = 0
#group_e_v2.v = np.random.random(Ne)*35*mV-85*mV
#group_i_v2.v = np.random.random(Ni)*35*mV-85*mV
#seed(1000)
group_e_2.v = np.random.random(ijwd2.Ne)*35*mV-85*mV
group_i_2.v = np.random.random(ijwd2.Ni)*35*mV-85*mV
group_e_2.delta_gk = 12*nS
group_e_2.tau_k = 80*ms
group_e_2.tau_s_de_extnl = 5*ms
#group_e_v2.v = np.random.random(ijwd2.Ne)*10*mV-60*mV
#group_i_v2.v = np.random.random(ijwd2.Ni)*10*mV-60*mV
group_e_2.I_extnl = 0.51*nA
group_i_2.I_extnl = 0.60*nA


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
syn_ee_12 = Synapses(group_e_1, group_e_2, model=synapse_e_12, method='euler',
                  on_pre={'up':'effect += 1', 'down': 'effect -= 1'})

syn_ee_12.connect(i=ijd_inter.i_1e_2e, j=ijd_inter.j_1e_2e)
syn_ee_12.w = np.random.normal(4, 1.9, len(ijd_inter.i_1e_2e))*nS*scale_e_12[i]
syn_ee_12.tau_s_de = 5*ms

#d_ee_v1v2 = np.random.uniform(delay_12[0], delay_v1v2[1], len(i_v1e_v2e))
syn_ee_12 = set_delay(syn_ee_12, ijd_inter.d_1e_2e)

syn_ee_12.effect = 0
#%%
syn_ei_12 = Synapses(group_e_1, group_i_2, model=synapse_e_12, method='euler',
                  on_pre={'up':'effect += 1', 'down': 'effect -= 1'})

syn_ei_12.connect(i=ijd_inter.i_1e_2i, j=ijd_inter.j_1e_2i)
syn_ei_12.w = 5*nS*scale_e_12[i] # np.random.normal(4, 1.9, len(i_v1e_v2i))*nS
syn_ei_12.tau_s_de = 5*ms

#d_ei_v1v2 = np.random.uniform(delay_v1v2[0], delay_v1v2[1], len(i_v1e_v2i))
syn_ei_12 = set_delay(syn_ei_12, ijd_inter.d_1e_2i)

syn_ei_12.effect = 0

#%%
syn_ee_21 = Synapses(group_e_2, group_e_1, model=synapse_e_12, method='euler',
                  on_pre={'up':'effect += 1', 'down': 'effect -= 1'})

syn_ee_21.connect(i=ijd_inter.i_2e_1e, j=ijd_inter.j_2e_1e)
syn_ee_21.w = np.random.normal(4, 1.9, len(ijd_inter.i_2e_1e))*nS*scale_e_21[j]
syn_ee_21.tau_s_de = 5*ms

syn_ee_21 = set_delay(syn_ee_21, ijd_inter.d_2e_1e)

syn_ee_21.effect = 0

#%%
syn_ei_21 = Synapses(group_e_2, group_i_1, model=synapse_e_12, method='euler',
                  on_pre={'up':'effect += 1', 'down': 'effect -= 1'})

syn_ei_21.connect(i=ijd_inter.i_2e_1i, j=ijd_inter.j_2e_1i)
syn_ei_21.w = 5*nS*scale_e_21[j] # np.random.normal(4, 1.9, len(i_v2e_v1i))*nS
syn_ei_21.tau_s_de = 5*ms

syn_ei_21 = set_delay(syn_ei_21, ijd_inter.d_2e_1i)

syn_ei_21.effect = 0

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

spk_1 = SpikeMonitor(group_e_1, record = True)
spk_2 = SpikeMonitor(group_e_2, record = True)

#spike_e = SpikeMonitor(group_e, record = True)
#rate_e = PopulationRateMonitor(group_e)

#%%
net = Network(collect())
net.store('state1')
#%%
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
simu_time1 = 1000*ms#2000*ms
simu_time2 = 2000*ms#8000*ms
simu_time3 = 1000*ms
simu_time4 = 6000*ms
simu_time5 = 2000*ms

tic = time.perf_counter(); toc = time.perf_counter()

group_input.rates = 0*Hz
net.run(simu_time1, profile=False); print(time.perf_counter() - tic); tic=time.perf_counter()
 #,namespace={'tau_k': 80*ms}
group_input.rates = stim_rate #True
net.run(simu_time2, profile=False); print(time.perf_counter() - tic); tic=time.perf_counter()
group_input.rates = 0*Hz #False
net.run(simu_time3, profile=False); print(time.perf_counter() - tic); tic=time.perf_counter()

group_e_2.delta_gk[chg_adapt_neuron] = new_delta_gk*nS; group_e_2.tau_k[chg_adapt_neuron] = 40*ms
net.run(simu_time4, profile=False); print(time.perf_counter() - tic); tic=time.perf_counter() 
group_input.rates = stim_rate #True
net.run(simu_time5, profile=False); print(time.perf_counter() - tic); tic=time.perf_counter() 

print('total simulation time:',time.perf_counter() - toc,'s')
#group_input.active = True
#net.run(simu_time3, report = 'text', profile=False) #,namespace={'tau_k': 80*ms}

#%%
spk_1_tstep = np.round(spk_1.t/(0.1*ms)).astype(int)
spk_2_tstep = np.round(spk_2.t/(0.1*ms)).astype(int)

param_1 = {'delta_gk':10,'tau_s_di':3,'tau_s_de':5,'ie_ratio':ie_ratio_1,'sti_onset':'10s'}

param_2 = {'chg_adapt_time':'2s','delta_gk':12,'new_delta_gk':new_delta_gk,'new_tau_k':40,'tau_s_di':3.5,\
         'tau_s_de':4.0, 'ie_ratio':2.880, 'chg_adapt_range':chg_adapt_range}

param_12 = {'scale_e_12':scale_e_12[i], 'scale_e_21':scale_e_21[j]}

data = {'param':param_12, 'a1':{'param':param_1,'e':{'i':spk_1.i[:],'t':spk_1_tstep}},
                            'a2':{'param':param_2,'e':{'i':spk_2.i[:],'t':spk_2_tstep}}}
#%%
import pickle
with open('data%s.file'%loop_num, 'wb') as file:
    pickle.dump(data, file)
#%%    
print('simulation done!')
#%%
"""
import load_data_dict
import post_analysis as psa
#%%
spkdata = load_data_dict.data_multiarea(data)

spkdata.a1.e.t = spkdata.a1.e.t*0.1*ms
spkdata.a2.e.t = spkdata.a2.e.t*0.1*ms

#%%
chg_adapt_loca = [0, 0]
chg_adapt_range = spkdata.a2.param['chg_adapt_range']
chg_adapt_neuron = cn.findnearbyneuron.findnearbyneuron(ijwd1.lattice_ext, chg_adapt_loca, chg_adapt_range, 62)
#%%
spkrate2, centre_ind2, jump_size2, jump_dist2 = psa.get_rate_centre_jumpdist(spkdata.a2.e, starttime=0*ms, endtime=500*ms, binforrate=10*ms, sample_interval=1*ms, slide_interval = 1, jump_interval = 15, dt=0.1*ms, show_trajectory=False)
spkrate2 = psa.overlap_centreandspike(centre_ind2, spkrate2, show_trajectory = False)
anititle=''#'change adaptation in region near centre to %.1fnS at 1000 ms\ndelta_gk: %.1fnS; tau_s_di: %.1fms; ie_ratio: %.3f'%(spke1.param['new_delta_gk'], spke1.param['delta_gk'], spke1.param['tau_s_di'], spke1.param['ie_ratio'])
ani2 = psa.show_pattern(spkrate2, spkrate2=0, area_num = 1, frames = 500, bottom_up=1, top_down=1, stimu_onset = -1, start_time = 0, anititle=anititle)
#ani1.save('chg_adapt_%d_%.1f_%.1f_%.1f_%.1f_%.3f_%d.png'%savename)
#%%
spkrate1, centre_ind1, jump_size1, jump_dist1 = psa.get_rate_centre_jumpdist(spkdata.a1.e, starttime=1800*ms, endtime=2500*ms, binforrate=10*ms, sample_interval=1*ms, slide_interval = 1, jump_interval = 15, dt=0.1*ms, show_trajectory=False)
spkrate1 = psa.overlap_centreandspike(centre_ind1, spkrate1, show_trajectory = False)
anititle=''#'change adaptation in region near centre to %.1fnS at 1000 ms\ndelta_gk: %.1fnS; tau_s_di: %.1fms; ie_ratio: %.3f'%(spke1.param['new_delta_gk'], spke1.param['delta_gk'], spke1.param['tau_s_di'], spke1.param['ie_ratio'])
ani1 = psa.show_pattern(spkrate1, spkrate2=0, area_num = 1, frames = 600, bottom_up=1, top_down=1, stimu_onset = -1, start_time = 0, anititle=anititle)

#%%
chg_adapt_loca = [0, 0]
chg_adapt_range = spkdata.a2.param['chg_adapt_range']
chg_adapt_neuron = cn.findnearbyneuron.findnearbyneuron(ijwd2.lattice_ext, chg_adapt_loca, chg_adapt_range, 62)

#%%
rate_chgadapt_neu = spkrate2.reshape(3969,-1)[chg_adapt_neuron]
poprate_rate_chgadapt_neu = rate_chgadapt_neu.sum(0)/len(chg_adapt_neuron)/0.01

plt.figure(figsize=[12,6])
plt.plot(np.arange(len(poprate_rate_chgadapt_neu)),poprate_rate_chgadapt_neu)
plt.plot([1000,1000],[0,140],'r--', label='time of adaptation change')
plt.xlabel('time(ms)')
plt.ylabel('rate(Hz)')
#title = ''''adpt_range:%d,new_tau_k:%.0f,new_delta_gk:%.0f,delta_gk:%.0f
#tau_s_di:%.1f,tau_s_de:%.1f,ie_ratio:%.3f'''%savename
plt.title(title)#('firing rate of neurons with decreased adaptation\nnew_delta_gk:%.1f;delta_gk:%.1f\ntau_s_di: %.1fms; ie_ratio: %.3f'%(spke1.param['new_delta_gk'], spke1.param['delta_gk'], spke1.param['tau_s_di'], spke1.param['ie_ratio']))
plt.legend()
#%%
plt.figure()
stat = plt.hist2d(centre_ind2[:,1], centre_ind2[:,0], bins=np.linspace(0,62,10))
#plt.matshow(stat[0])
plt.colorbar()
"""

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
plt.figure()
plt.plot(spk_v1.t[:]/ms, spk_v1.i[:],'.')
'''
"""
#%%
net.restore('state1')



