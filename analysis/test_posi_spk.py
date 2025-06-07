#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 11:30:09 2021

@author: shni2598
"""


import brian2.numpy_ as np
import poisson_stimuli as psti
import get_stim_scale
import connection as cn
import mydata
import matplotlib.pyplot as plt
from brian2.only import *

#%%
start_scope()

'''stim 1; constant amplitude'''
'''no attention'''
stim_scale_cls = get_stim_scale.get_stim_scale()
stim_scale_cls.seed = 10
n_StimAmp = 1
n_perStimAmp = 50
stim_amp_scale = np.ones(n_StimAmp*n_perStimAmp)
for i in range(n_StimAmp):
    stim_amp_scale[i*n_perStimAmp:i*n_perStimAmp+n_perStimAmp] = 2**i

stim_scale_cls.stim_amp_scale = stim_amp_scale
stim_scale_cls.stim_dura = 250
stim_scale_cls.separate_dura = np.array([300,600])
stim_scale_cls.get_scale()
stim_scale_cls.n_StimAmp = n_StimAmp
stim_scale_cls.n_perStimAmp = n_perStimAmp

transient = 0 # 20000
init = np.zeros(transient//stim_scale_cls.dt_stim)
stim_scale_cls.scale_stim = np.concatenate((init, stim_scale_cls.scale_stim))
stim_scale_cls.stim_on += transient
# #%%
# '''attention'''
# stim_scale_cls_att = get_stim_scale.get_stim_scale()
# stim_scale_cls_att.seed = 15
# n_StimAmp = 3
# n_perStimAmp = 50
# stim_amp_scale = np.ones(n_StimAmp*n_perStimAmp)
# for i in range(n_StimAmp):
#     stim_amp_scale[i*n_perStimAmp:i*n_perStimAmp+n_perStimAmp] = 2**i

# stim_scale_cls_att.stim_amp_scale = stim_amp_scale
# stim_scale_cls_att.stim_dura = 250
# stim_scale_cls_att.separate_dura = np.array([300,600])
# stim_scale_cls_att.get_scale()

# inter_time = 4000
# suplmt = (inter_time // stim_scale_cls.dt_stim) - (stim_scale_cls.scale_stim.shape[0] - stim_scale_cls.stim_on[-1,1] // stim_scale_cls.dt_stim) # supply '0' between non-attention and attention stimuli amplitude

# stim_scale_cls.scale_stim = np.concatenate((stim_scale_cls.scale_stim, np.zeros(suplmt), stim_scale_cls_att.scale_stim))
# stim_scale_cls.stim_amp_scale = np.concatenate((stim_scale_cls.stim_amp_scale, stim_scale_cls_att.stim_amp_scale))
# stim_scale_cls_att.stim_on += stim_scale_cls.stim_on[-1,1] + inter_time
# stim_scale_cls.stim_on = np.vstack((stim_scale_cls.stim_on, stim_scale_cls_att.stim_on))


scale_1 = TimedArray(stim_scale_cls.scale_stim, dt=10*ms)


'''stim 2; varying amplitude'''
'''no attention'''
stim_scale_cls = get_stim_scale.get_stim_scale()
stim_scale_cls.seed = 10
n_StimAmp = 1
n_perStimAmp = 50
stim_amp_scale = np.ones(n_StimAmp*n_perStimAmp)
for i in range(n_StimAmp):
    stim_amp_scale[i*n_perStimAmp:i*n_perStimAmp+n_perStimAmp] = 2**i

stim_scale_cls.stim_amp_scale = stim_amp_scale
stim_scale_cls.stim_dura = 250
stim_scale_cls.separate_dura = np.array([300,600])
stim_scale_cls.get_scale()
stim_scale_cls.n_StimAmp = n_StimAmp
stim_scale_cls.n_perStimAmp = n_perStimAmp

transient = 0 # 20000
init = np.zeros(transient//stim_scale_cls.dt_stim)
stim_scale_cls.scale_stim = np.concatenate((init, stim_scale_cls.scale_stim))
stim_scale_cls.stim_on += transient


scale_2 = TimedArray(stim_scale_cls.scale_stim, dt=10*ms)


posi_stim_e1 = NeuronGroup(64*64, \
                        '''rates =  bkg_rates + stim_1*scale_1(t) + stim_2*scale_2(t) : Hz
                        bkg_rates : Hz
                        stim_1 : Hz
                        stim_2 : Hz
                        ''', threshold='rand()<rates*dt')

posi_stim_e1.bkg_rates = 0*Hz
posi_stim_e1.stim_1 = psti.input_spkrate(maxrate = [200], sig=[6], position=[[0, 0]])*Hz
posi_stim_e1.stim_2 = psti.input_spkrate(maxrate = [200], sig=[6], position=[[-32, -32]])*Hz
#posi_stim_e1.stim_2 = psti.input_spkrate(maxrate = [200], sig=[6], position=[[-32, 0]])*Hz
# #%%
# synapse_e_extnl = cn.model_neu_syn_AD.synapse_e_AD
# syn_extnl_e1 = Synapses(posi_stim_e1, group_e_1, model=synapse_e_extnl, on_pre='x_E_extnl_post += w')
# syn_extnl_e1.connect('i==j')
# syn_extnl_e1.w = w_extnl_*nS#*tau_s_de_*nS
#%%
spk_e_1 = SpikeMonitor(posi_stim_e1, record = True)


net = Network(collect())
seed(0)
net.store('state1')
#%%
data_ext = mydata.mydata()
data_ext.a1 = mydata.mydata()
data_ext.a1.ge = mydata.mydata()
e_lattice = cn.coordination.makelattice(64, 64, [0,0])

mua_loca = [0, 0]
mua_range = 10 
mua_neuron_1 = cn.findnearbyneuron.findnearbyneuron(e_lattice, mua_loca, mua_range, 64)
mua_loca = [-32, -32]
mua_range = 10 
mua_neuron_2 = cn.findnearbyneuron.findnearbyneuron(e_lattice, mua_loca, mua_range, 64)

spk_sum_1 = 0
spk_sum_2 = 0
spk_1_all = []
spk_2_all = []
#%%
spk_less = 0
import random
for i in range(20):
    seed(random.randint(0, 30000))
    net.run(stim_scale_cls.stim_on[-1,1]*ms)
    print(spk_e_1.i.shape)

    

# #%%
# net.run(stim_scale_cls.stim_on[-1,1]*ms)
# #%%
# net.restore('state1')


    data_ext.a1.ge.t = np.round(spk_e_1.t/(0.1*ms)).astype(int)
    data_ext.a1.ge.i = spk_e_1.i
    
    simu_time_tot = stim_scale_cls.stim_on[-1,1]
    data_ext.a1.ge.get_sparse_spk_matrix([64*64, simu_time_tot*10], 'csr')


    spk_1 = data_ext.a1.ge.spk_matrix[mua_neuron_1,:].sum()
    spk_2 = data_ext.a1.ge.spk_matrix[mua_neuron_2,:].sum()
    spk_1_all.append(spk_1)
    spk_2_all.append(spk_2)
    
    spk_less += int(spk_1<spk_2)
    print(spk_1, spk_2, spk_1<spk_2)
    spk_sum_1 += spk_1
    spk_sum_2 += spk_2
    
    net.restore('state1', restore_random_state  = True)

print(spk_sum_1/20)
print(spk_sum_2/20)
#%%
scipy.stats.wilcoxon(np.array(spk_1_all) - np.array(spk_2_all))

#%%
plt.figure()
plt.scatter(data.a1.param.e_lattice[:,0], data.a1.param.e_lattice[:,1], s=0.5)
plt.scatter(data.a1.param.i_lattice[:,0], data.a1.param.i_lattice[:,1], s=0.5)
#%%
plt.figure()
plt.scatter(data.a2.param.e_lattice[:,0], data.a2.param.e_lattice[:,1], s=0.5)
plt.scatter(data.a2.param.i_lattice[:,0], data.a2.param.i_lattice[:,1], s=0.5)
#%%
plt.figure()
plt.scatter(data.a1.param.e_lattice[:,0], data.a1.param.e_lattice[:,1], s=0.5)

for n in range(len(n_in_bin)):

    plt.scatter(data.a1.param.e_lattice[n_in_bin[n]][:,0], data.a1.param.e_lattice[n_in_bin[n]][:,1], s=1.5, marker='^')


