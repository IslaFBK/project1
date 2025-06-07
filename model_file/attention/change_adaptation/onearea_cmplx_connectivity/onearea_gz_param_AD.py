# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 22:01:25 2020

@author: nishe
"""

'''
p_ee = 0.08 or 0.16
tau_s_di = 6*ms or 3*ms
Adam neuron model
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
loop_num = -1

param_in = [[0.08,6],[0.16,3]]

for p_ee, tau_s_di in param_in:
    for repeat in range(3):
        loop_num += 1
        if loop_num == sys_argv:
            print(loop_num)
            break
    else: continue
    break

if loop_num != sys_argv: sys.exit("wrong PBS_array_id")

Ne = 3969; Ni = 1024

ijwd = pre_process.get_ijwd(Ni=Ni)
ijwd.p_ee = p_ee
ijwd.w_ee_dist = 'lognormal'
ijwd.hybrid = 0.4
ijwd.cn_scale_weight = 2
ijwd.cn_scale_wire = 2
ijwd.iter_num=5

scale_ee_2 = 1.; scale_ei_2 = 1.
ijwd.ie_ratio = 3.375
ijwd.mean_J_ee = 4*10**-3*scale_ee_2 # usiemens
ijwd.sigma_J_ee = 1.9*10**-3*scale_ee_2 # usiemens
ijwd.generate_ijw()
ijwd.generate_d_rand()
ijwd.w_ei = 5*10**(-3)*scale_ei_2 #usiemens
ijwd.w_ii = 25*10**(-3)
#%%
#chg_adapt_loca = [0, 0]
#chg_adapt_range = 6
#chg_adapt_neuron = cn.findnearbyneuron.findnearbyneuron(ijwd.lattice_ext, chg_adapt_loca, chg_adapt_range, ijwd.width)
#%%
start_scope()
#%%

neuronmodel_e = cn.model_neu_syn_AD.neuron_e_AD
neuronmodel_i = cn.model_neu_syn_AD.neuron_i_AD

synapse_e = cn.model_neu_syn_AD.synapse_e_AD
synapse_i = cn.model_neu_syn_AD.synapse_i_AD
#%%
group_e_2 =NeuronGroup(ijwd.Ne, model=neuronmodel_e,
                     threshold='v>v_threshold', method='euler',
                     reset='''v = v_reset
                              g_k += delta_gk''', refractory='(t-lastspike)<t_ref')

group_i_2 =NeuronGroup(ijwd.Ni, model=neuronmodel_i,
                     threshold='v>v_threshold', method='euler',
                     reset='v = v_reset', refractory='(t-lastspike)<t_ref')

syn_ee_2 = Synapses(group_e_2, group_e_2, model=synapse_e, 
                  on_pre='x_E_post += w')
syn_ei_2 = Synapses(group_e_2, group_i_2, model=synapse_e, 
                  on_pre='x_E_post += w')
syn_ie_2 = Synapses(group_i_2, group_e_2, model=synapse_i, 
                  on_pre='x_I_post += w')
syn_ii_2 = Synapses(group_i_2, group_i_2, model=synapse_i, 
                  on_pre='x_I_post += w')

#%%
syn_ee_2.connect(i=ijwd.i_ee, j=ijwd.j_ee)
syn_ei_2.connect(i=ijwd.i_ei, j=ijwd.j_ei)
syn_ie_2.connect(i=ijwd.i_ie, j=ijwd.j_ie)
syn_ii_2.connect(i=ijwd.i_ii, j=ijwd.j_ii)

#%%
tau_s_di_try = tau_s_di
syn_ee_2.w = ijwd.w_ee*usiemens * 5
syn_ei_2.w = ijwd.w_ei*usiemens * 5 #5*nS
syn_ii_2.w = ijwd.w_ii*usiemens * tau_s_di_try#25*nS
syn_ie_2.w = ijwd.w_ie*usiemens * tau_s_di_try
#w_ext = 2*nS
#syn_pois_e.w = w_ext
#syn_pois_i.w = w_ext
#%%
def set_delay(syn, delay_up):
    #n = len(syn)
    syn.delay = delay_up*ms
    #syn.down.delay = (delay_up + 1)*ms
    
    return syn 
#%%
#ijwd.generate_d_dist()
#ijwd.generate_d_rand()
syn_ee_2 = set_delay(syn_ee_2, ijwd.d_ee)
syn_ie_2 = set_delay(syn_ie_2, ijwd.d_ie)
syn_ei_2 = set_delay(syn_ei_2, ijwd.d_ei)
syn_ii_2 = set_delay(syn_ii_2, ijwd.d_ii)

#syn_pois_e = set_delay(syn_pois_e)
#syn_pois_i = set_delay(syn_pois_i)

#syn_ee_v2.up.delay = 3*ms; syn_ee_v2.down.delay = 4*ms; 
#syn_ie_v2.up.delay = 3*ms; syn_ie_v2.down.delay = 4*ms; 
#syn_ei_v2.up.delay = 3*ms; syn_ei_v2.down.delay = 4*ms; 
#syn_ii_v2.up.delay = 3*ms; syn_ii_v2.down.delay = 4*ms; 

#syn_ee_2.effect = 0; syn_ee_2.tau_s_de = 5.0*ms
#syn_ie_2.effect = 0; syn_ie_2.tau_s_di = 6*ms
#syn_ei_2.effect = 0; syn_ei_2.tau_s_de = 5.0*ms
#syn_ii_2.effect = 0; syn_ii_2.tau_s_di = 6*ms

group_e_2.tau_s_de = 5.0*ms; 
group_e_2.tau_s_di = tau_s_di_try*ms
group_e_2.tau_s_re = group_e_2.tau_s_ri = 1*ms

group_e_2.tau_s_de_inter = 5.0*ms; 
group_e_2.tau_s_re_inter = 1*ms
group_e_2.tau_s_de_extnl = 5.0*ms
group_e_2.tau_s_re_extnl = 1*ms

group_i_2.tau_s_de = 5.0*ms
group_i_2.tau_s_di = tau_s_di_try*ms
group_i_2.tau_s_re = group_i_2.tau_s_ri = 1*ms

group_i_2.tau_s_de_inter = 5.0*ms; 
group_i_2.tau_s_re_inter = 1*ms
group_i_2.tau_s_de_extnl = 5.0*ms
group_i_2.tau_s_re_extnl = 1*ms

#syn_pois_e.effect = 0
#syn_pois_i.effect = 0
#group_e_v2.v = np.random.random(Ne)*35*mV-85*mV
#group_i_v2.v = np.random.random(Ni)*35*mV-85*mV
#seed(1000)
group_e_2.v = np.random.random(ijwd.Ne)*35*mV-85*mV
group_i_2.v = np.random.random(ijwd.Ni)*35*mV-85*mV
group_e_2.delta_gk = 10*nS
group_e_2.tau_k = 80*ms
#group_e_v2.v = np.random.random(ijwd.Ne)*10*mV-60*mV
#group_i_v2.v = np.random.random(ijwd.Ni)*10*mV-60*mV
group_e_2.I_extnl = 0.51*nA
group_i_2.I_extnl = 0.60*nA

#%%
spk = SpikeMonitor(group_e_2, record = True)
#%%
net = Network(collect())
net.store('state1')
#%%
#ijwd.change_ie(4.4)
#syn_ie_2.w = ijwd.w_ie*usiemens

print('ie_w: %fnsiemens' %(syn_ie_2.w[0]/nsiemens))
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
#tau_s_re = 1*ms
#tau_s_ri = 1*ms
#tau_x_re = 1*ms
tic = time.perf_counter()
#seed(10)
simu_time1 = 5000*ms#2000*ms
#simu_time2 = 2000*ms#8000*ms
#simu_time3 = 2000*ms

#group_input.active = False
net.run(simu_time1, report = 'text', profile=False) #,namespace={'tau_k': 80*ms}
#group_input.active = True
#group_e_2.delta_gk[chg_adapt_neuron] = 2*nS; group_e_2.tau_k[chg_adapt_neuron] = 40*ms
#net.run(simu_time2, report = 'text', profile=False) #,namespace={'tau_k': 80*ms}
print('total time elapsed:',time.perf_counter() - tic)
#%%
#spk_1_tstep = np.round(spk_1.t/(0.1*ms)).astype(int)
spk_tstep = np.round(spk.t/(0.1*ms)).astype(int)

#param_1 = {'delta_gk':10,'tau_s_di':3,'tau_s_de':5,'ie_ratio':3.1725,'sti_onset':'10s'}

param = {'chg_adapt_time':'2s','delta_gk':10,'new_delta_gk':2,'new_tau_k':40,'tau_s_di':tau_s_di_try,\
         'tau_s_de':5.0, 'ie_ratio':3.375, 'chg_adapt_range':6, 'p_ee':p_ee}

#param_12 = {'scale_e_12':scale_e_12[i], 'scale_e_21':scale_e_21[j]}

#data = {'param':param_12, 'a1':{'param':param_1,'e':{'i':spk_1.i[:],'t':spk_1_tstep}},
#                            'a2':{'param':param_2,'e':{'i':spk_2.i[:],'t':spk_2_tstep}}}
data = {'a1':{'param':param,'e':{'i':spk.i[:],'t':spk_tstep}}}

with open('data%d.file'%loop_num, 'wb') as file:
    pickle.dump(data, file)
#%%
#import load_data_dict
#import post_analysis as psa
##%%
#spkdata2 = load_data_dict.data_multiarea(data)
#
##spkdata.a1.e.t = spkdata.a1.e.t*0.1*ms
#spkdata2.a1.e.t = spkdata2.a1.e.t*0.1*ms
#
##%%
#spkrate3, centre_ind3, jump_size3, jump_dist3 = psa.get_rate_centre_jumpdist(spkdata2.a1.e, starttime=0*ms, endtime=1000*ms, binforrate=5*ms, sample_interval=1*ms, slide_interval = 1, jump_interval = 15, dt=0.1*ms)
#spkrate3 = psa.overlap_centreandspike(centre_ind3, spkrate3, show_trajectory = False)
#anititle=''#'change adaptation in region near centre to %.1fnS at 1000 ms\ndelta_gk: %.1fnS; tau_s_di: %.1fms; ie_ratio: %.3f'%(spke1.param['new_delta_gk'], spke1.param['delta_gk'], spke1.param['tau_s_di'], spke1.param['ie_ratio'])
#ani3 = psa.show_pattern(spkrate3, spkrate2=0, area_num = 1, frames = 500, start_time = 0, anititle=anititle)
#%%
net.restore('state1')





