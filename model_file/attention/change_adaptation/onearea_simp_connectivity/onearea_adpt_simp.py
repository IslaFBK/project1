# -*- coding: utf-8 -*-
"""
Created on Tue May 19 11:57:30 2020

@author: nishe
"""

import brian2.numpy_ as np
from brian2.only import *
from connection import pre_process
import connection as cn
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import sys
import scipy.io as sio
#%%
np.random.seed(10)
ijwd = pre_process.get_ijwd()
ijwd.w_ee_dist = 'normal'
ijwd.hybrid = 0.
ijwd.cn_scale_weight = 1
ijwd.cn_scale_wire = 1
ijwd.iter_num=1
ijwd.generate_ijw()
ijwd.generate_d_rand()

#%%
chg_adapt_range = 6
ie_ratio = 3.206
tau_s_de = 4.5*ms
tau_s_di = 4*ms
new_delta_gk = 2#*nS
delta_gk = 10#*nS
#w_ee_c = ijwd.w_ee.copy()
#%%
chg_adapt_range = 6
#ie_ratio = 3.206
tau_s_de = 5*ms
tau_s_di = 3*ms
new_delta_gk = 10#*nS
delta_gk = 10#*nS
#%%
chg_adapt_loca = [0, 0]
#chg_adapt_range = 5
chg_adapt_neuron = cn.findnearbyneuron.findnearbyneuron(ijwd.lattice_ext, chg_adapt_loca, chg_adapt_range, ijwd.width)

#%%
start_scope()

neuronmodel_e = '''
dv/dt = (1/C)*(-g_l*(v - v_l) + (-g_k)*(v - v_k) + (-g_I)*(v - v_rev_I) +(-g_E-g_E_inter)*(v - v_rev_E) + I_extnl) : volt (unless refractory)
dg_k/dt = -g_k/tau_k :siemens
delta_gk : siemens 
g_I : siemens
g_E : siemens
g_E_inter : siemens
I_extnl : amp
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

#synapse_ext = '''
#w: siemens
#g_ext_post = w*s : siemens (summed)
#ds/dt = -s/tau_s_de + rect_puls : 1 (clock-driven)
#rect_puls = (1/tau_s_re)*effect : Hz
#effect : 1
#
#''' group_e_v1, group_i_v1, syn_ee_v1, syn_ei_v1, syn_ie_v1, syn_ii_v1, vexi, spike_e
#%%
group_e_v1 =NeuronGroup(ijwd.Ne, model=neuronmodel_e,
                     threshold='v>v_threshold', method='euler',
                     reset='''v = v_reset
                              g_k += delta_gk''', refractory='(t-lastspike)<t_ref')

group_i_v1 =NeuronGroup(ijwd.Ni, model=neuronmodel_i,
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
#%%
syn_ee_v1.connect(i=ijwd.i_ee, j=ijwd.j_ee)
syn_ei_v1.connect(i=ijwd.i_ei, j=ijwd.j_ei)
syn_ie_v1.connect(i=ijwd.i_ie, j=ijwd.j_ie)
syn_ii_v1.connect(i=ijwd.i_ii, j=ijwd.j_ii)

#%%
syn_ee_v1.w = ijwd.w_ee*usiemens
syn_ei_v1.w = ijwd.w_ei*usiemens #5*nS
syn_ii_v1.w = ijwd.w_ii*usiemens #25*nS
syn_ie_v1.w = ijwd.w_ie*usiemens
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
syn_ee_v1 = set_delay(syn_ee_v1, ijwd.d_ee)
syn_ie_v1 = set_delay(syn_ie_v1, ijwd.d_ie)
syn_ei_v1 = set_delay(syn_ei_v1, ijwd.d_ei)
syn_ii_v1 = set_delay(syn_ii_v1, ijwd.d_ii)
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
group_e_v1.v = np.random.random(ijwd.Ne)*35*mV-85*mV
group_i_v1.v = np.random.random(ijwd.Ni)*35*mV-85*mV
group_e_v1.delta_gk = delta_gk*nS
#group_e_v1.v = np.random.random(ijwd.Ne)*10*mV-60*mV
#group_i_v1.v = np.random.random(ijwd.Ni)*10*mV-60*mV
group_e_v1.I_extnl = 0.51*nA
group_i_v1.I_extnl = 0.6*nA

#%%
#v_v1 = StateMonitor(group_e_v1, ('v'), record=True,dt=0.5*ms,name='v_v1')
spk_e1 = SpikeMonitor(group_e_v1, record = True)

#%%
net = Network(collect())
#%%
ijwd.change_ie(3.15)
syn_ie_v1.w = ijwd.w_ie*usiemens

#net.add(spk_e1)
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
#delta_gk = 10*nS #10*nS
t_ref = 4*ms # refractory period

'''
tau_s_de_list = np.arange(2,8,0.5)*ms
tau_s_di_list = np.arange(2,10,0.5)*ms
sys_argv = int(sys.argv[1])
loop_num = 0
for i in range(len(tau_s_de_list)):
    for j in range(len(tau_s_di_list)):
        loop_num += 1
        if loop_num == sys_argv:
            print(i,j)
            break
    else:
        continue
    break
'''        
      
#tau_s_de = 5*ms # 5*ms
#tau_s_di = 3*ms #tau_s_di*ms # 3*ms
tau_s_re = 1*ms
tau_s_ri = 1*ms
tic = time.perf_counter()
#seed(10)
simu_time1 = 1000*ms
simu_time2 = 2000*ms

net.run(simu_time1, report = 'text', profile=False) #,namespace={'tau_k': 80*ms}
group_e_v1.delta_gk[chg_adapt_neuron] = new_delta_gk*nS
net.run(simu_time2, report = 'text', profile=False) #,namespace={'tau_k': 80*ms}

print('%ds'%(time.perf_counter() - tic))

#%%
import post_analysis as psa
from scipy.fftpack import fft
#%%
spkrate1, centre_ind1, jump_size1, jump_dist1 = psa.get_rate_centre_jumpdist(spk_e1, starttime=0*ms, endtime=3000*ms, binforrate=10*ms, sample_interval=1*ms, slide_interval = 1, jump_interval = 15, dt=0.1*ms, show_trajectory=False)

poprate_rate_chgadapt_neu = spkrate1.reshape(3969,-1)[chg_adapt_neuron]
poprate_rate_chgadapt_neu = poprate_rate_chgadapt_neu.sum(0)/len(chg_adapt_neuron)
#%%
plt.figure()
plt.plot(poprate_rate_chgadapt_neu)
#%%
spkrate1 = psa.overlap_centreandspike(centre_ind1, spkrate1, show_trajectory = False)
#anititle='change adaptation in region near centre to %.1fnS at 1000 ms\ntau_s_di: %.1fms; ie_ratio: %.3f'%(new_delta_gk, tau_s_di/ms, ie_ratio)
anititle=''
ani1 = psa.show_pattern(spkrate1, spkrate2=0, area_num = 1, frames = 5500, bottom_up=1, top_down=1, stimu_onset = -1, start_time = 0, anititle=anititle)

#%%
net.restore('state1')













