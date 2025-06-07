#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 16:32:27 2021

@author: shni2598
"""

from brian2.only import *
import connection as cn
cn.model_neu_syn_AD.neuron_e_AD
#%%

prefs.codegen.target = 'cython'
prefs.codegen.target = 'numpy'
prefs.codegen.target = 'standalone'


group_e_1 = NeuronGroup(10, model=cn.model_neu_syn_AD.neuron_e_AD, #model='''dv/dt = ...'''
             threshold='v>v_threshold', method='euler',
             reset='''v = v_reset
                      g_k += delta_gk''', refractory='(t-lastspike)<t_ref')
                      
syn_ee_1 = Synapses(group_e_1, group_e_1, model=self.synapse_model_e, 
                            on_pre='''x_E_post += w''') 

syn_ee_1.connect(i=ijwd1.i_ee, j=ijwd1.j_ee)

synapse_e_AD = '''
w: siemens
'''
syn_ee_1.w = = ijwd1.w_ee*nsiemens


syn_ee_1.delay = [1 2] * ms

#%%
group_input_test = PoissonGroup(100,300*Hz)

spk_sti_test = SpikeMonitor(group_input_test, record = True)

net_test = Network(group_input_test, spk_sti_test)

group_input_test.active = False
net_test.run(1000*ms)
group_input_test.active = True
net_test.run(1000*ms)
group_input_test.active = False
net_test.run(1000*ms)

plt.figure()
plt.plot(spk_sti_test.t/ms, spk_sti_test.i,'|')
plt.xlim([0,3000])
plt.xlabel('ms')
plt.ylabel('neuron',fontsize=12)
#%%
#%%
group_input_test = PoissonGroup(100,300*Hz)

spk_sti_test = SpikeMonitor(group_input_test, record = True)

net_test = Network(group_input_test, spk_sti_test)

group_input_test.rates = 0*Hz
net_test.run(1000 * ms)
group_input_test.rates = 300*Hz
net_test.run(1000*ms)
group_input_test.rates = 0*Hz
net_test.run(1000*ms)

plt.figure()
plt.plot(spk_sti_test.t/ms, spk_sti_test.i,'|')
plt.xlim([0,3000])
plt.xlabel('ms')
plt.ylabel('neuron',fontsize=12)
#%%
x = np.arange(10)
y = np.concatenate((np.ones(5)*10, np.ones(5)))
fig, ax = plt.subplots(1,1)
ax.plot(x[:5], y[:5])
ax.set_ylim([0,12])

axt = ax.twinx()
axt.plot(x[5:], y[5:])
axt.set_ylim([0,2])

#%%
import scipy.sparse as sparse

#%%
dt = 0.1*10**-3
T = 1
t = np.arange(0,T,dt)
f = 100
y = 2*np.sin(2*np.pi*f*t)

gamma_T_step = round((1/f/dt))

tri_n = 10
spk_t = []
neu_i = []
for tri in range(tri_n):
    spk_gam_t = np.random.choice(np.arange(0, y.shape[0], gamma_T_step), 20, replace=False) + tri*t.shape[0]
    spk_ran_t = np.random.randint(0, y.shape[0], 10) + tri*t.shape[0]
    spk_t_t = np.concatenate((spk_gam_t, spk_ran_t))
    neu_i_t = np.zeros(spk_t_t.shape, dtype=int)
    spk_t.append(spk_t_t)
    neu_i.append(neu_i_t)

spk_t = np.concatenate(spk_t)
neu_i = np.concatenate(neu_i)


lfp = np.tile(y ,tri_n)



spk_matrix = sparse.coo_matrix((np.ones(len(spk_t),dtype=int),(neu_i, spk_t)), shape=[1, t.shape[0]*10])
spk_matrix = spk_matrix.tocsc()

dura = np.tile(np.array([0,1000]),tri_n).reshape(10,2)

R = spk_cohe.spk_lfp_coherence(spk_matrix, lfp, dura, discard_init = 200, hfwin_lfp_seg=150, dt = 0.1)        


plt_len = (R.freq <= 150).sum()
stim_amp = [400]
ref_sig = 'LFP'
st = 0
fig, ax = plt.subplots(3, 1, figsize=[4*1, 10])

ax[0].plot(R.freq[:plt_len], R.cohe[:plt_len], label='cohe,noatt;%dHz'%stim_amp[st])
ax[1].loglog(R.freq[1:plt_len], R.staRef_pw[1:plt_len], label='sta'+ref_sig+'_pw,noatt;%dHz'%stim_amp[st])
ax[2].plot(np.arange(R.staRef.shape[0])/10, R.staRef, label='sta'+ref_sig+',noatt;%dHz'%stim_amp[st])







#%%
plt.figure()
plt.plot(t, y)


