#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:41:53 2020

@author: shni2598
"""

import brian2.numpy_ as np
import matplotlib.pyplot as plt
import connection as cn
#import warnings
from scipy import sparse
from brian2.only import *
import time
import mydata
import firing_rate_analysis
import frequency_analysis as fa
import os
import datetime

import pre_process_sc
#%%
prefs.codegen.target = 'cython'

dir_cache = './cache' #%sys.argv[1]
prefs.codegen.runtime.cython.cache_dir = dir_cache
prefs.codegen.max_cache_dir_size = 1000
#%%
ijwd = pre_process_sc.get_ijwd()
#ijwd.Ne = 77*77; ijwd.Ni = 39*39
#ijwd.width = 76
#ijwd.w_ee_mean *= 2; ijwd.w_ei_mean *= 2; ijwd.w_ie_mean *= 2; ijwd.w_ii_mean *= 2; 

ijwd.Ne = 80*80; ijwd.Ni = 40*40
ijwd.width = 79

scale_d_p = np.sqrt(8/5)
ijwd.decay_p_ee = 8 * scale_d_p# decay constant of e to e connection probability as distance increases
ijwd.decay_p_ei = 10 * scale_d_p# decay constant of e to i connection probability as distance increases
ijwd.decay_p_ie = 20 * scale_d_p# decay constant of i to e connection probability as distance increases
ijwd.decay_p_ii = 20 * scale_d_p# decay constant of i to i connection probability as distance increases

ijwd.mean_SynNumIn_ee = 320#*5/8     ; # p = 0.08
ijwd.mean_SynNumIn_ei = 500 * 8/5#     ; # p = 0.125
ijwd.mean_SynNumIn_ie = 200#*5/8     ; # p = 0.2
ijwd.mean_SynNumIn_ii = 250 * 8/5#     ; # p = 0.25

ijwd.generate_ijw()
ijwd.generate_d_rand()
#%%

chg_adapt_loca = [0, 0]
chg_adapt_range = 6 * scale_d_p
chg_adapt_neuron = cn.findnearbyneuron.findnearbyneuron(ijwd.e_lattice, chg_adapt_loca, chg_adapt_range, ijwd.width)



#%%
start_scope()

neuronmodel_e = cn.model_neu_syn_AD.neuron_e_AD
neuronmodel_i = cn.model_neu_syn_AD.neuron_i_AD

synapse_e = cn.model_neu_syn_AD.synapse_e_AD
synapse_i = cn.model_neu_syn_AD.synapse_i_AD

group_e_1 =NeuronGroup(ijwd.Ne, model=neuronmodel_e,
                     threshold='v>v_threshold', method='euler',
                     reset='''v = v_reset
                              g_k += delta_gk''', refractory='(t-lastspike)<t_ref')

group_i_1 =NeuronGroup(ijwd.Ni, model=neuronmodel_i,
                     threshold='v>v_threshold', method='euler',
                     reset='v = v_reset', refractory='(t-lastspike)<t_ref')
#
#syn_ee_1 = Synapses(group_e_1, group_e_1, model=synapse_e, 
#                  on_pre='''x_E_post += w*u*L
#                            L -= u*L''')
#syn_ei_1 = Synapses(group_e_1, group_i_1, model=synapse_e, 
#                  on_pre='''x_E_post += w*u*L
#                            L -= u*L''')
#syn_ie_1 = Synapses(group_i_1, group_e_1, model=synapse_i, 
#                  on_pre='''x_I_post += w*u*L
#                            L -= u*L''')
#syn_ii_1 = Synapses(group_i_1, group_i_1, model=synapse_i, 
#                  on_pre='''x_I_post += w*u*L
#                            L -= u*L''')
syn_ee_1 = Synapses(group_e_1, group_e_1, model=synapse_e, 
                  on_pre='''x_E_post += w''')
syn_ei_1 = Synapses(group_e_1, group_i_1, model=synapse_e, 
                  on_pre='''x_E_post += w''')
syn_ie_1 = Synapses(group_i_1, group_e_1, model=synapse_i, 
                  on_pre='''x_I_post += w''')
syn_ii_1 = Synapses(group_i_1, group_i_1, model=synapse_i, 
                  on_pre='''x_I_post += w''')

'''external input'''
#stim_rate = psti.input_spkrate(maxrate = [800,800], sig=[6,6], position=[[0, 0],[31.5,31.5]])*Hz
bkg_rate2e = 850*Hz
bkg_rate2i = 1000*Hz
extnl_e = PoissonGroup(ijwd.Ne, bkg_rate2e)
extnl_i = PoissonGroup(ijwd.Ni, bkg_rate2i)

#tau_x_re = 1*ms
synapse_e_extnl = cn.model_neu_syn_AD.synapse_e_AD
syn_extnl_e = Synapses(extnl_e, group_e_1, model=synapse_e_extnl, on_pre='x_E_extnl_post += w')
syn_extnl_i = Synapses(extnl_i, group_i_1, model=synapse_e_extnl, on_pre='x_E_extnl_post += w')

syn_extnl_e.connect('i==j')
syn_extnl_i.connect('i==j')

w_extnl_ = 1.5 # nS
syn_extnl_e.w = w_extnl_*5.8*nS#*tau_s_de_*nS
syn_extnl_i.w = w_extnl_*5.8*nS#*tau_s_de_*nS


syn_ee_1.connect(i=ijwd.i_ee, j=ijwd.j_ee)
syn_ei_1.connect(i=ijwd.i_ei, j=ijwd.j_ei)
syn_ie_1.connect(i=ijwd.i_ie, j=ijwd.j_ie)
syn_ii_1.connect(i=ijwd.i_ii, j=ijwd.j_ii)


#tau_s_di_try = tau_s_di_
syn_ee_1.w = ijwd.w_ee*nsiemens * 5.8 #tau_s_de_
syn_ei_1.w = ijwd.w_ei*nsiemens * 5.8 #tau_s_de_ #5*nS
syn_ii_1.w = ijwd.w_ii*nsiemens * 6.5#tau_s_di_#25*nS
syn_ie_1.w = ijwd.w_ie*nsiemens * 6.5 #tau_s_di_#
#w_ext = 2*nS
#syn_pois_e.w = w_ext
#syn_pois_i.w = w_ext
#u_base = 0.3
#syn_ee_1.u = u_base; syn_ee_1.L = 1
#syn_ei_1.u = u_base; syn_ee_1.L = 1
#syn_ii_1.u = u_base; syn_ee_1.L = 1
#syn_ie_1.u = u_base; syn_ee_1.L = 1
#tau_f = 1500*ms; tau_d = 200*ms

def set_delay(syn, delay_up):
    #n = len(syn)
    syn.delay = delay_up*ms
    #syn.down.delay = (delay_up + 1)*ms
    
    return syn 

#generate_d_dist()
#generate_d_rand()
#def generate_d_rand(delay,len_i_ee,len_i_ie,len_i_ei,len_i_ii):
#        
#    d_ee = np.random.uniform(0, delay, len_i_ee)    
#    d_ie = np.random.uniform(0, delay, len_i_ie)
#    d_ei = np.random.uniform(0, delay, len_i_ei)
#    d_ii = np.random.uniform(0, delay, len_i_ii)
#    
#    return d_ee, d_ie, d_ei, d_ii

#d_ee, d_ie, d_ei, d_ii = generate_d_rand(4,len(i_ee),len(i_ie),len(i_ei),len(i_ii))
syn_ee_1 = set_delay(syn_ee_1, ijwd.d_ee)
syn_ie_1 = set_delay(syn_ie_1, ijwd.d_ie)
syn_ei_1 = set_delay(syn_ei_1, ijwd.d_ei)
syn_ii_1 = set_delay(syn_ii_1, ijwd.d_ii)

#syn_pois_e = set_delay(syn_pois_e)
#syn_pois_i = set_delay(syn_pois_i)

tau_s_de_ = 5.8; tau_s_di_ = 4.444#6.5
delta_gk_ = 10


group_e_1.tau_s_de = tau_s_de_*ms; 
group_e_1.tau_s_di = tau_s_di_*ms
group_e_1.tau_s_re = group_e_1.tau_s_ri = 1*ms

group_e_1.tau_s_de_inter = tau_s_de_*ms #5.0*ms; 
group_e_1.tau_s_re_inter = 1*ms
group_e_1.tau_s_de_extnl = tau_s_de_*ms #5.0*ms
group_e_1.tau_s_re_extnl = 1*ms

group_i_1.tau_s_de = tau_s_de_*ms
group_i_1.tau_s_di = tau_s_di_*ms
group_i_1.tau_s_re = group_i_1.tau_s_ri = 1*ms

group_i_1.tau_s_de_inter = tau_s_de_*ms #5.0*ms; 
group_i_1.tau_s_re_inter = 1*ms
group_i_1.tau_s_de_extnl = tau_s_de_*ms #5.0*ms
group_i_1.tau_s_re_extnl = 1*ms

#syn_pois_e.effect = 0
#syn_pois_i.effect = 0
#group_e_v2.v = np.random.random(Ne)*35*mV-85*mV
#group_i_v2.v = np.random.random(Ni)*35*mV-85*mV
#seed(1000)
group_e_1.v = np.random.random(ijwd.Ne)*35*mV-85*mV
group_i_1.v = np.random.random(ijwd.Ni)*35*mV-85*mV
group_e_1.delta_gk = delta_gk_*nS
group_e_1.tau_k = 80*ms
#group_e_v2.v = np.random.random(Ne)*10*mV-60*mV
#group_i_v2.v = np.random.random(Ni)*10*mV-60*mV
group_e_1.I_extnl_crt = 0*nA #0.51*nA
group_i_1.I_extnl_crt = 0*nA #0.60*nA

#%%
spk_e = SpikeMonitor(group_e_1, record = True)
spk_i = SpikeMonitor(group_i_1, record = True)
#lfp_moni = StateMonitor(group_LFP_record, ('lfp1','lfp2','lfp3'), record = True)
#%%
net = Network(collect())
net.store('state1')
#%%
scale_ee = 1.15#1.05 * 0.85 * 1.1; 
scale_ei = 1. #1.0 * 0.87 * 1.1
scale_ie = 1.062#0.95 * 0.88 * 1.1
scale_ii = 1. #1.07 * 0.88 * 1.1
scale_2_ri = 1.
syn_ee_1.w = ijwd.w_ee*nsiemens * 5.8 * scale_ee#tau_s_de_
syn_ei_1.w = ijwd.w_ei*nsiemens * 5.8 * scale_ei #tau_s_de_ #5*nS
syn_ie_1.w = ijwd.w_ie*nsiemens * 6.5 * scale_ie / scale_2_ri#tau_s_di_#25*nS
syn_ii_1.w = ijwd.w_ii*nsiemens * 6.5 * scale_ii / scale_2_ri#tau_s_di_#
group_e_1.delta_gk = 12*nS
group_e_1.tau_k = 80*ms

group_e_1.tau_s_di = 4.444*ms
group_i_1.tau_s_di = 4.444*ms
group_e_1.tau_s_de = 5.8*ms
group_i_1.tau_s_de = 5.8*ms

group_i_1.tau_s_re = 1*ms
group_i_1.tau_s_ri = 1*ms * scale_2_ri
group_e_1.tau_s_re = 1*ms
group_e_1.tau_s_ri = 1*ms * scale_2_ri

print(scale_ee,scale_ei,scale_ie,scale_ii,scale_2_ri, group_e_1.delta_gk[0]/nS)
#tau_d = 100*ms
#%%
#change_ie(4.4)
#syn_ie_1.w = w_ie*usiemens

print('ie_w: %fnsiemens' %(syn_ie_1.w[0]/nsiemens))
#Ne = 63*63; Ni = 1000;
C = 0.25*nF # 0.25*nF # capacitance
g_l = 16.7*nS # leak capacitance
v_l = -70*mV # leak voltage
v_threshold = -50*mV
v_reset = -80*mV# -60*mV
v_rev_I = -80*mV
v_rev_E = 0*mV
v_k = -85*mV
#tau_k = 80*ms# 80*ms
#delta_gk = 10*nS #10*nS
t_ref = 4*ms # refractory period
new_delta_gk_ = 4
#tau_s_de = 5*ms
#tau_s_di = 3*ms
#tau_s_re = 1*ms
#tau_s_ri = 1*ms
#tau_x_re = 1*ms
tic = time.perf_counter()
#seed(10)
simu_time1 = 4000*ms#2000*ms
simu_time2 = 10000*ms
#simu_time2 = 2000*ms#8000*ms
#simu_time3 = 1000*ms
#simu_time_tot = 30000*ms
#group_input.active = False
extnl_e.rates = bkg_rate2e
net.run(simu_time1, profile=False) #,namespace={'tau_k': 80*ms}
#extnl_e.rates = bkg_rate2e + stim_rate

group_e_1.delta_gk[chg_adapt_neuron] = new_delta_gk_*nS
net.run(simu_time2, profile=False) #,namespace={'tau_k': 80*ms}

# extnl_e.rates = bkg_rate2e
# net.run(simu_time3, profile=False) #,namespace={'tau_k': 80*ms}
#group_input.active = True
#group_e_1.delta_gk[chg_adapt_neuron] = 2*nS; group_e_1.tau_k[chg_adapt_neuron] = 40*ms
#net.run(simu_time2, report = 'text', profile=False) #,namespace={'tau_k': 80*ms}
print('total time elapsed:',time.perf_counter() - tic)
#%%
spk_tstep_e = np.round(spk_e.t/(0.1*ms)).astype(int)
spk_tstep_i = np.round(spk_i.t/(0.1*ms)).astype(int)
now = datetime.datetime.now()
loop_num = 0
#param_1 = {'delta_gk':10,'tau_s_di':3,'tau_s_de':5,'ie_ratio':3.1725,'sti_onset':'10s'}
'''
param_new = {'delta_gk':delta_gk_,
         'new_delta_gk':new_delta_gk_,
         #'new_tau_k':40,
         'tau_s_di':tau_s_di_,
         'tau_s_de':tau_s_de_, 'ie_ratio':ie_ratio_,
         'mean_J_ee': ijwd.mean_J_ee,
         #'chg_adapt_range':6, 
         'p_ee':p_ee,
         'simutime':simu_time_tot/ms,
         'chg_adapt_time': simu_time1/ms,
         'chg_adapt_range': chg_adapt_range,
         'chg_adapt_loca': chg_adapt_loca,}
param = {**param, **param_new}
'''
param = {}
#param_12 = {'scale_e_12':scale_e_12[i], 'scale_e_21':scale_e_21[j]}

#data = {'param':param_12, 'a1':{'param':param_1,'e':{'i':spk_1.i[:],'t':spk_1_tstep}},
#                            'a2':{'param':param_2,'e':{'i':spk_2.i[:],'t':spk_2_tstep}}}
data = {'datetime':now.strftime("%Y-%m-%d %H:%M:%S"), 'dt':0.1, 'loop_num':loop_num, 'data_dir': os.getcwd(),
        'a1':{'param':param,
              #'LFP':{'lfp1':lfp_moni.lfp1/nA, 'lfp2':lfp_moni.lfp2/nA, 'lfp3':lfp_moni.lfp3/nA},
              'ge':{'i':spk_e.i[:],'t':spk_tstep_e},
              'gi':{'i':spk_i.i[:],'t':spk_tstep_i}}}


data1 = mydata.mydata(data)
#%%
start_time = 2000; end_time = 6000
frames = end_time - start_time

data1.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time,\
                           sample_interval = 1, n_neuron = ijwd.Ne, window = 10, dt = 0.1, reshape_indiv_rate = True)
data1.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time,\
                           sample_interval = 1, n_neuron = ijwd.Ni, window = 10, dt = 0.1, reshape_indiv_rate = True)

ani = firing_rate_analysis.show_pattern(data1.a1.ge.spk_rate.spk_rate, data1.a1.gi.spk_rate.spk_rate, frames = frames-50, start_time = start_time, anititle='animation')
#%%
data1.a1.ge.get_MSD(start_time=4000, end_time=14000, n_neuron=80*80,window = 15, jump_interval=np.array([10,15]), fit_stableDist='pylevy')
#data1.a1.ge.get_MSD(start_time=12000, end_time=21000,  window = 15, jump_interval=np.array([10,15,30]), fit_stableDist='Matlab')
print(data1.a1.ge.MSD.stableDist_param)

print(spk_e.i.shape[0]/3969/10)
#%%
net.restore('state1')
print("net.restore('state1')",spk_e.i.shape[0]/3969/10)
#%%
data1.a1.ge.get_spike_rate(start_time=0, end_time=14000,\
                           sample_interval = 1, n_neuron = ijwd.Ne, window = 10, dt = 0.1, reshape_indiv_rate = True)

mua_adpt = data1.a1.ge.spk_rate.spk_rate.reshape(ijwd.Ne,-1)[chg_adapt_neuron].mean(0)
#%%
plt.figure()
plt.plot(mua_adpt[0:14000])
#%%
coef, freq = fa.myfft(mua_adpt[0:14000], 1000)
plt.figure()
plt.plot(freq[1:], np.abs(coef[1:]))
#%%
coef, freq = fa.myfft(mua_adpt[14000:24000], 1000)
plt.figure()
plt.plot(freq[1:], np.abs(coef[1:]))
#%%
data1.a1.ge.get_spike_rate(start_time=0, end_time=14000,\
                           sample_interval = 8, n_neuron = ijwd.Ne, window = 8.1, dt = 0.1, reshape_indiv_rate = True)

mua_adpt = data1.a1.ge.spk_rate.spk_rate.reshape(3969,-1)#[chg_adapt_neuron].sum(0)

plt.figure()
plt.imshow(mua_adpt,aspect = 'auto')
#%%
start_time = 3000; end_time = 10000
frames = end_time - start_time

data1.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time,\
                           sample_interval = 1, n_neuron = ijwd.Ne, window = 15, dt = 0.1, reshape_indiv_rate = True)
#%%
data1.a1.ge.get_centre_mass()
#%%

coef, freq = fa.myfft(data1.a1.ge.centre_mass.centre[:,0], 1000)
#%%
plt.figure()
plt.plot(freq[1:], np.abs(coef[1:]))
#%%
plt.figure()
plt.plot(data1.a1.ge.t/10,data1.a1.ge.i,'|')
#%%
plt.figure()
plt.imshow(data1.a1.ge.spk_rate.spk_rate[:,:,100])
#%%
'''
sample_interval = 10; window = 10; dt = 0.1;
start_time = 0; end_time = 14000
sample_interval = int(np.round(sample_interval/dt))
start_time = int(np.round(start_time/dt))
end_time = int(np.round(end_time/dt))
window_step = int(np.round(window/dt))

sample_t = np.arange(0, end_time-start_time-window_step+1, sample_interval)

spiket = data1.a1.ge.t #np.round(spike.t/dt).astype(int)
spikei = data1.a1.ge.i[(spiket >= start_time) & (spiket < end_time)]
spiket = spiket[(spiket >= start_time) & (spiket < end_time)]

spk_mat = csc_matrix((np.ones(len(spikei),dtype=int),(spikei,spiket-start_time)),(3969,end_time-start_time))
spk_rate = np.zeros([3969, len(sample_t)], dtype=np.int8)

counts_all = 0
for i in range(len(sample_t)):
    
    neu, counts = np.unique(spk_mat.indices[spk_mat.indptr[sample_t[i]]:spk_mat.indptr[sample_t[i]+window_step]],return_counts=True)
    counts_all += counts.sum()
    spk_rate[:, i][neu] += counts
'''
#%%
plt.figure()
plt.scatter( ijwd.e_lattice[:,0], ijwd.e_lattice[:,1])
plt.figure()
plt.scatter( ijwd.i_lattice[:,0], ijwd.i_lattice[:,1])
#%%
pre_neuron = 5500
pre = ijwd.e_lattice[pre_neuron]; post = ijwd.e_lattice[ijwd.j_ee[ijwd.i_ee==pre_neuron]]
plt.figure()
plt.plot(post[:,0],post[:,1], '.')
#%%
pre_neuron = 1000
pre = ijwd.i_lattice[pre_neuron]; post = ijwd.e_lattice[ijwd.j_ie[ijwd.i_ie==pre_neuron]]
plt.figure()
plt.plot(post[:,0],post[:,1], '.')
#%%
pre_neuron = 1000
pre = ijwd.i_lattice[pre_neuron]; post = ijwd.i_lattice[ijwd.j_ii[ijwd.i_ii==pre_neuron]]
plt.figure()
plt.plot(post[:,0],post[:,1], '.')
#%%
data1.a1.ge.get_MSD(start_time=1000, end_time=10000,jump_interval=np.arange(1,1000,2), fit_stableDist=True)

#%%
fig, ax1 = plt.subplots(1,1)

ax1.loglog(data1.a1.ge.MSD.jump_interval,data1.a1.ge.MSD.MSD)
ax2 = ax1.twinx()
stableDist_param
ax2.plot(data1.a1.ge.MSD.jump_interval, data1.a1.ge.MSD.stableDist_param[:,0,0])



#%%
plt.figure();
plt.plot(data1.a1.ge.MSD.jump_interval,data1.a1.ge.MSD.MSD)

#%%
net.restore('state1')

#%%
ijwd_63 = expo_connection.get_ijwd()
ijwd_63.Ne = 63*63; ijwd_63.Ni = 32*32
ijwd_63.width = 62
ijwd_63.generate_ijw()
ijwd_63.generate_d_rand()
#%%
def plot_outgoing(i, j, neuron_src, lattice_src, lattice_trg):
    #pre_neuron = 5500
    pre = lattice_src[neuron_src]; post = lattice_trg[j[i==neuron_src]]
    plt.figure()
    plt.plot(post[:,0],post[:,1], '.')
    plt.plot(pre[0], pre[1], 'or')
#%%
plot_outgoing(ijwd_63.i_ie, ijwd_63.j_ie, 0, ijwd_63.i_lattice, ijwd_63.e_lattice)   
plot_outgoing(ijwd.i_ie, ijwd.j_ie, 0, ijwd.i_lattice, ijwd.e_lattice)   
   
#%%
plot_outgoing(ijwd_63.i_ii, ijwd_63.j_ii, 0, ijwd_63.i_lattice, ijwd_63.i_lattice)   
plot_outgoing(ijwd.i_ii, ijwd.j_ii, 0, ijwd.i_lattice, ijwd.i_lattice)   

