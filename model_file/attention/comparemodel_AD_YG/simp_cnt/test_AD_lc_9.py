#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 22:43:00 2020

@author: shni2598
"""

'''
change number of ee and ie connections
synapse conductance decay time
delay time
'''
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
from cfc_analysis import cfc
#%%
prefs.codegen.target = 'cython'

dir_cache = './cache/cache%s' %0#%sys.argv[1]
prefs.codegen.runtime.cython.cache_dir = dir_cache
prefs.codegen.max_cache_dir_size = 1000
#%%
sys_argv = 0#int(sys.argv[1])
#%%
loop_num = -1

#param_in = [[0.08,6.5]]
#repeat = 10

for scale_ee_1 in [1.00]:#np.arange(1,1.26,0.05):#[1.15]: #np.linspace(1.,1.2,5):
    for scale_ei_1 in [1.]:#np.linspace(1,1,1):
        for scale_ie_1 in [1.]:#np.arange(0.9825,1.1225,0.02):#[1.0625]: #np.linspace(0.95,1.1,5):
            for scale_ii_1 in np.linspace(1.,1.,1):
                for w_extnl_ in [1.5]:
                    for delta_gk_ in [12]:#np.linspace(7,14,8):
                        for new_delta_gk_ in [2.5]:#np.arange(0.15,0.31,0.05)*delta_gk_:#np.linspace(1,5,5):
                            for tau_s_di_ in [4.444]:#np.arange(0.7,1.31,0.1)*4.444:#np.linspace(4.444,4.444,1):
                                for tau_s_de_ in [5.8]:#np.arange(0.7,1.11,0.1)*5.8:#np.linspace(5.8,5.8,1):
                                    for tau_s_r_ in np.linspace(1,1,1):
                                        for decay_p_ie_p_ii in [20]:
                                            for delay_ in [4]:#np.arange(2,5.1,0.5):
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
                    
if loop_num != sys_argv: sys.exit("Error: wrong PBS_array_id")                    

#%%
8*0.8/8.815
Out[176]: 0.7260351673284176

7/9
Out[177]: 0.7777777777777778

  7/8*0.8
Out[183]: 0.7000000000000001

7/(8*0.8)
Out[184]: 1.09375

7/(8*0.8)*8.815
Out[185]: 9.64140625

7/(8*0.8)*8.815/(8*0.82)
Out[186]: 1.4697265625

7/(8*0.8)*8.815/(10*0.82)
Out[187]: 1.17578125

320*7/(8*0.8)*8.815/(10*0.82)              
#%%
ijwd = pre_process_sc.get_ijwd()
ijwd.Ne = 63*63; ijwd.Ni = 32*32
ijwd.width = 62#79
#ijwd.w_ee_mean *= 2; ijwd.w_ei_mean *= 2; ijwd.w_ie_mean *= 2; ijwd.w_ii_mean *= 2;
scale_d_p = 1 #np.sqrt(8/5) 
ijwd.decay_p_ee = 7#scale_d_p # decay constant of e to e connection probability as distance increases
ijwd.decay_p_ei = 9.5#*1.125#scale_d_p # decay constant of e to i connection probability as distance increases
ijwd.decay_p_ie = 20 #scale_d_p # decay constant of i to e connection probability as distance increases
ijwd.decay_p_ii = 20 #scale_d_p # decay constant of i to i connection probability as distance increases

ijwd.mean_SynNumIn_ee = 220#200 320 * 6/8 * 1   ; # p = 0.08                          200
ijwd.mean_SynNumIn_ei = 390#500 500 #500 * 1      #500 #* 8/5     ; # p = 0.125       320  
ijwd.mean_SynNumIn_ie = 150#125 200 * 5/8     ; # p = 0.2                             129
ijwd.mean_SynNumIn_ii = 260#250 250 * 1    # 250 * 8/5     ; # p = 0.25               221  

ijwd.w_ee_mean = 4
ijwd.w_ei_mean = 5
ijwd.w_ie_mean = 20
ijwd.w_ii_mean = 25

ijwd.delay = delay_
ijwd.generate_ijw()
ijwd.generate_d_rand()

ijwd.w_ee *= scale_ee_1#tau_s_de_
ijwd.w_ei *= scale_ei_1 #tau_s_de_ #5*nS
ijwd.w_ie *= scale_ie_1#tau_s_di_#25*nS
ijwd.w_ii *= scale_ii_1#tau_s_di_#
param = {**ijwd.__dict__}
#%%

chg_adapt_loca = [0, 0]
chg_adapt_range = 5 * scale_d_p
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
stim_rate = psti.input_spkrate(maxrate = [800,800], sig=[6,6], position=[[0, 0],[31.5,31.5]])*Hz
# bkg_rate2e = 850*Hz
# bkg_rate2i = 1000*Hz
extnl_e = PoissonGroup(ijwd.Ne, stim_rate)
# extnl_i = PoissonGroup(ijwd.Ni, bkg_rate2i)

# #tau_x_re = 1*ms
synapse_e_extnl = cn.model_neu_syn_AD.synapse_e_AD
syn_extnl_e = Synapses(extnl_e, group_e_1, model=synapse_e_extnl, on_pre='x_E_extnl_post += w')
# syn_extnl_i = Synapses(extnl_i, group_i_1, model=synapse_e_extnl, on_pre='x_E_extnl_post += w')

syn_extnl_e.connect('i==j')
# syn_extnl_i.connect('i==j')

# #w_extnl_ = 1.5 # nS
syn_extnl_e.w = w_extnl_*5.8*nS#*tau_s_de_*nS
# syn_extnl_i.w = w_extnl_*5.8*nS#*tau_s_de_*nS


syn_ee_1.connect(i=ijwd.i_ee, j=ijwd.j_ee)
syn_ei_1.connect(i=ijwd.i_ei, j=ijwd.j_ei)
syn_ie_1.connect(i=ijwd.i_ie, j=ijwd.j_ie)
syn_ii_1.connect(i=ijwd.i_ii, j=ijwd.j_ii)


#tau_s_di_try = tau_s_di_
syn_ee_1.w = ijwd.w_ee*nsiemens/tau_s_r_ * 5.8 #tau_s_de_
syn_ei_1.w = ijwd.w_ei*nsiemens/tau_s_r_ * 5.8 #tau_s_de_ #5*nS
syn_ii_1.w = ijwd.w_ii*nsiemens/tau_s_r_ * 6.5 #tau_s_di_#25*nS
syn_ie_1.w = ijwd.w_ie*nsiemens/tau_s_r_ * 6.5 #tau_s_di_#
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

#d_ee, d_ie, d_ei, d_ii = generate_d_rand(4,len(i_ee),len(i_ie),len(i_ei),len(i_ii))
syn_ee_1 = set_delay(syn_ee_1, ijwd.d_ee)
syn_ie_1 = set_delay(syn_ie_1, ijwd.d_ie)
syn_ei_1 = set_delay(syn_ei_1, ijwd.d_ei)
syn_ii_1 = set_delay(syn_ii_1, ijwd.d_ii)

#syn_pois_e = set_delay(syn_pois_e)
#syn_pois_i = set_delay(syn_pois_i)

#tau_s_de_ = 5.8; tau_s_di_ = 6.5
#delta_gk_ = 10


group_e_1.tau_s_de = tau_s_de_*ms; 
group_e_1.tau_s_di = tau_s_di_*ms
group_e_1.tau_s_re = group_e_1.tau_s_ri = tau_s_r_*ms

group_e_1.tau_s_de_inter = tau_s_de_*ms #5.0*ms; 
group_e_1.tau_s_re_inter = 1*ms
group_e_1.tau_s_de_extnl = 5.0*ms #5.0*ms tau_s_de_
group_e_1.tau_s_re_extnl = 1*ms

group_i_1.tau_s_de = tau_s_de_*ms
group_i_1.tau_s_di = tau_s_di_*ms
group_i_1.tau_s_re = group_i_1.tau_s_ri = tau_s_r_*ms

group_i_1.tau_s_de_inter = tau_s_de_*ms #5.0*ms; 
group_i_1.tau_s_re_inter = 1*ms
group_i_1.tau_s_de_extnl = 5.0*ms #5.0*ms tau_s_de_
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
group_e_1.I_extnl_crt = 0.51*nA #0.51*nA
group_i_1.I_extnl_crt = 0.60*nA #0.60*nA

#%%
spk_e = SpikeMonitor(group_e_1, record = True)
spk_i = SpikeMonitor(group_i_1, record = True)
#lfp_moni = StateMonitor(group_LFP_record, ('lfp1','lfp2','lfp3'), record = True)
#g_moni_e = StateMonitor(group_e_1, ('g_E','g_I','I_extnl_crt','v','I_exi','I_inh','I_extnl_spk'), record = [0,1])
#g_moni_i = StateMonitor(group_i_1, ('g_E','g_I','I_extnl_crt','v','I_exi','I_inh','I_extnl_spk'), record = [0,1])

#%%
net = Network(collect())
net.store('state1')
#%%
scale_ee = 1.15#1.05;                                                                
scale_ei = 1.02 #1.
scale_ie = 0.95 *1.04  #1.115 1.13 0.95 1.07                           #0.87
scale_ii = 0.74   #1.00            0.89                          #0.83
syn_ee_1.w = ijwd.w_ee*nsiemens * 5.8 * scale_ee#tau_s_de_
syn_ei_1.w = ijwd.w_ei*nsiemens * 5.8 * scale_ei #tau_s_de_ #5*nS
syn_ie_1.w = ijwd.w_ie*nsiemens * 6.5 * scale_ie#tau_s_di_#25*nS
syn_ii_1.w = ijwd.w_ii*nsiemens * 6.5 * scale_ii#tau_s_di_#

#tau_d = 100*ms
group_e_1.tau_s_di = 4.*ms            #4.444
group_i_1.tau_s_di = 4.*ms            #4.444

group_e_1.tau_s_de = 5.5*ms
group_i_1.tau_s_de = 5.5*ms

group_e_1.delta_gk = 10*nS
# t_ref = 5.0*ms
#%%
allrun = 0
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
simu_time1 = 8000*ms#2000*ms
simu_time2 = 10000*ms
#simu_time2 = 2000*ms#8000*ms
#simu_time3 = 1000*ms
simu_time_tot = 20000*ms
#group_input.active = False
#extnl_e.rates = bkg_rate2e
extnl_e.rates = 0*Hz
net.run(simu_time1, profile=False) #,namespace={'tau_k': 80*ms}
#extnl_e.rates = bkg_rate2e + stim_rate
if allrun:
    group_e_1.delta_gk[chg_adapt_neuron] = 2.5*nS
    net.run(simu_time2, profile=False) #,namespace={'tau_k': 80*ms}
    # extnl_e.rates = stim_rate
    # net.run(simu_time2, profile=False)
#extnl_e.rates = bkg_rate2e
#net.run(simu_time3, profile=False) #,namespace={'tau_k': 80*ms}
#group_input.active = True
#group_e_1.delta_gk[chg_adapt_neuron] = 2*nS; group_e_1.tau_k[chg_adapt_neuron] = 40*ms
#net.run(simu_time2, report = 'text', profile=False) #,namespace={'tau_k': 80*ms}
print('total time elapsed:',time.perf_counter() - tic)

#%%
spk_tstep_e = np.round(spk_e.t/(0.1*ms)).astype(int)
spk_tstep_i = np.round(spk_i.t/(0.1*ms)).astype(int)
now = datetime.datetime.now()
#param_1 = {'delta_gk':10,'tau_s_di':3,'tau_s_de':5,'ie_ratio':3.1725,'sti_onset':'10s'}
scale_ei_1
param_new = {'delta_gk':delta_gk_,
         'new_delta_gk':new_delta_gk_,
         #'new_tau_k':40,
         'tau_s_di':tau_s_di_,
         'tau_s_de':tau_s_de_,
         'tau_s_r':tau_s_r_,
         #'ie_ratio':ie_ratio_,
         #'mean_J_ee': ijwd.mean_J_ee,
         #'chg_adapt_range':6, 
         #'p_ee':p_ee,
         'simutime':simu_time_tot/ms,
         'chg_adapt_time': simu_time1/ms,
         'chg_adapt_range': chg_adapt_range,
         'chg_adapt_loca': chg_adapt_loca,
         'scale_ee_1': scale_ee_1,
         'scale_ei_1': scale_ei_1,
         'scale_ie_1': scale_ie_1,
         'scale_ii_1': scale_ii_1}
param = {**param, **param_new}

#param = {}scale_ei_1
#param_12 = {'scale_e_12':scale_e_12[i], 'scale_e_21':scale_e_21[j]}

#data = {'param':param_12, 'a1':{'param':param_1,'e':{'i':spk_1.i[:],'t':spk_1_tstep}},
#                            'a2':{'param':param_2,'e':{'i':spk_2.i[:],'t':spk_2_tstep}}}
data = {'datetime':now.strftime("%Y-%m-%d %H:%M:%S"), 'dt':0.1, 'loop_num':loop_num, 'data_dir': os.getcwd(),
        'a1':{'param':param,
              #'LFP':{'lfp1':lfp_moni.lfp1/nA, 'lfp2':lfp_moni.lfp2/nA, 'lfp3':lfp_moni.lfp3/nA},
              'ge':{'i':spk_e.i[:],'t':spk_tstep_e},
              'gi':{'i':spk_i.i[:],'t':spk_tstep_i}}}


data1 = mydata.mydata(data)
#data1.load(data)
#%%
start_time = 4e3; end_time = 8e3
data1.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data1.a1.param.Ne, window = 10)
#data1.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data1.a1.param.Ni, window = 10)
data1.a1.ge.get_centre_mass(detect_pattern=True)
#data1.a1.ge.overlap_centreandspike()
pattern_size2 = data1.a1.ge.centre_mass.pattern_size.copy()
pattern_size2[np.invert(data1.a1.ge.centre_mass.pattern)] = 0

#title = "dgk{:.1f}_ndgk{:.1f}_alpha{:.2f}".format(data.a1.param.delta_gk,data.a1.param.new_delta_gk,data.a1.ge.MSD.stableDist_param[0,0])
#title = "dgk{:.1f}_ndgk{:.1f}_alpha{:.2f}".format(data.a1.param.delta_gk,data.a1.param.new_delta_gk,data.a1.ge.MSD.stableDist_param[0,0])
#title = "ie%.3f_dgk%.1f_ndgk%.2f_tauk%.1f_alpha%.2f"%(data.a1.param.scale_ie_1, data.a1.param.delta_gk, data.a1.param.new_delta_gk, data.a1.param.tau_k, data.a1.ge.MSD.stableDist_param[0,0])
#title = "ei%.3f_ii%.3f_dgk%.1f_ndgk%.2f_alpha%.2f"%(data.a1.param.scale_ei_1, data.a1.param.scale_ii_1, data.a1.param.delta_gk, data.a1.param.new_delta_gk, data.a1.ge.MSD.stableDist_param[0,0])
title = "ee%.3f_ie%.3f_dgk%.1f_ndgk%.2f"%(data1.a1.param.scale_ee_1, data1.a1.param.scale_ie_1, data1.a1.param.delta_gk, data1.a1.param.new_delta_gk)

frames = data1.a1.ge.spk_rate.spk_rate.shape[2]
#ani = firing_rate_analysis.show_pattern(data1.a1.ge.spk_rate.spk_rate, data1.a1.gi.spk_rate.spk_rate, frames = frames, start_time = start_time, anititle=title)
ani = firing_rate_analysis.show_pattern(data1.a1.ge.spk_rate.spk_rate, None, frames = frames, start_time = start_time, anititle=title,\
                                        show_pattern_size=True, pattern_ctr=data1.a1.ge.centre_mass.centre_ind, \
                                            pattern_size=pattern_size2)
#%%
centre_ind, centre, jump_size, jump_dist, pattern, pattern_size = firing_rate_analysis.get_centre_mass(data1.a1.ge.spk_rate.spk_rate, 1, 1, 15, detect_pattern=True)
#%%
pattern_size_n = pattern_size.copy()
pattern_size_n[np.invert(pattern)]=0
#%%
ani = firing_rate_analysis.show_pattern(data1.a1.ge.spk_rate.spk_rate, None, \
                                        frames = frames, start_time = start_time, anititle=title,\
                                        show_pattern_size=True, pattern_ctr=data1.a1.ge.centre_mass.centre_ind, pattern_size=pattern_size_n)
#%%
net.restore('state1')
#%%
start_time = 4e3; end_time = 12e3
data1.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data1.a1.param.Ne, window = 10)
data1.a1.ge.get_centre_mass()
fig,ax=plt.subplots(1,1)
ax = firing_rate_analysis.plot_traj(ax, data1.a1.ge.centre_mass.centre[2000:4000])
ax.set_xlim([-0.5,62.5])
ax.set_ylim([-0.5,62.5])

#%%
data1.a1.ge.get_MSD(start_time=5000, end_time=10000, n_neuron=63*63,window = 15, jump_interval=np.array([15]), fit_stableDist='pylevy')
#data1.a1.ge.get_MSD(start_time=12000, end_time=21000,  window = 15, jump_interval=np.array([10,15,30]), fit_stableDist='Matlab')
print(data1.a1.ge.MSD.stableDist_param)

#%%
data1.a1.ge.get_spike_rate(start_time=0, end_time=24000,\
                           sample_interval = 1, n_neuron = ijwd.Ne, window = 10, dt = 0.1, reshape_indiv_rate = True)

mua_adpt = data1.a1.ge.spk_rate.spk_rate.reshape(ijwd.Ne,-1)[chg_adapt_neuron].mean(0)/0.01
#%%
plt.figure()
plt.plot(mua_adpt[0:20000])
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
coef, freq = fa.myfft(mua_adpt[4000:24000], 1000)
freq_max = 20
ind_len = freq[freq<freq_max].shape[0] 

# plt.figure()
ax1.plot(freq[1:ind_len], np.abs(coef[1:ind_len]))
ax1.plot([3,3],[0,np.max(np.abs(coef[1:ind_len]))],'r--')
ax1.plot([8,8],[0,np.max(np.abs(coef[1:ind_len]))],'r--')
ax1.set_title('spon')
coef, freq = fa.myfft(mua_adpt[10000:18000], 1000)
freq_max = 20
ind_len = freq[freq<freq_max].shape[0] 

# plt.figure()
ax2.plot(freq[1:ind_len], np.abs(coef[1:ind_len]))
ax2.plot([3,3],[0,np.max(np.abs(coef[1:ind_len]))],'r--')
ax2.plot([8,8],[0,np.max(np.abs(coef[1:ind_len]))],'r--')
ax2.set_title('adapt')
#%%
dt = 1/10000;
end = int(4/dt); start = int(1/dt)
spon_rate = np.sum((data1.a1.ge.t < end) & (data1.a1.ge.t > start))/3/data1.a1.param.Ne
print(spon_rate)
spon_rate = np.sum((data1.a1.gi.t < end) & (data1.a1.gi.t > start))/3/data1.a1.param.Ni
print(spon_rate)

#print(data1.a1.ge.i.shape[0]/3969/4)
#%%
net.restore('state1')
#%%
plt.figure()
plt.plot(data1.a1.ge.t, data1.a1.ge.i,'|')
#%%
plt.figure()
plt.plot(g_moni_e.t/ms, g_moni_e.g_E[0]/nS)
#%%
plt.figure()
plt.plot(g_moni_i.t/ms, g_moni_i.g_I[0]/nS)
#%%
plt.figure()
plt.plot(g_moni_i.t/ms, g_moni_i.I_extnl_crt[0]/nA)
#%%
plt.figure()
plt.plot(g_moni_i.t/ms, g_moni_i.I_exi[0]/nA)
#%%
plt.figure()
plt.plot(g_moni_i.t/ms, g_moni_i.I_inh[0]/nA)
#%%
plt.figure()
plt.plot(g_moni_i.t/ms, g_moni_i.v[0]/mV)
#%%
findcfc = cfc.cfc()
#%%
Fs = 1000;
timeDim = 0;
phaseBand = np.arange(1,14.1,0.5)
ampBand = np.arange(20,101,5) 
phaseBandWid = 0.49 ;
ampBandWid = 5 ;

band1 = np.concatenate((phaseBand - phaseBandWid, ampBand - ampBandWid)).reshape(1,-1)
band2 = np.concatenate((phaseBand + phaseBandWid, ampBand + ampBandWid)).reshape(1,-1)
subBand = np.concatenate((band1,band2),0)
subBand = subBand.T
#%%
phaseBand_in = subBand[:27]
ampBand_in = subBand[27:]
#%%
findcfc.Fs = Fs
findcfc.phaseBand = phaseBand_in
findcfc.ampBand = ampBand_in
findcfc.optionMethod = 1
findcfc.optionSur = 2
#%%
MI_raw_mat, MI_surr_mat = findcfc.find_cfc_from_rawsig(mua_adpt[6000:14000])

#%%
phaseBandWid = 0.5#0.49 ;
ampBandWid = 5 ;
phaseBand = np.arange(1,14.1,0.5)
ampBand = np.arange(20,101,5) 

fig, [ax1,ax2] = plt.subplots(2,1, figsize=[7,9])
#x_range = np.arange(phaseBand[0]-phaseBandWid/2, phaseBand[-1]+phaseBandWid/2+1)
#y_range = np.arange(ampBand[0]-ampBandWid/2, ampBand[-1]+ampBandWid/2+1)

#im = ax1.imshow(np.flip(MI_raw.T, 0), aspect='auto', extent=[phaseBand[0]-phaseBandWid/2, phaseBand[-1]+phaseBandWid/2, ampBand[0]-ampBandWid/2, ampBand[-1]+ampBandWid/2])
imc = ax1.contourf(phaseBand, ampBand, MI_raw_mat.T, 15)#, aspect='auto')
imcc = ax1.contour(phaseBand, ampBand, MI_raw_mat.T, 15, colors='k', linewidths=0.3)#, aspect='auto')

imc2 = ax2.contourf(phaseBand, ampBand, MI_surr_mat.T, 15)#, aspect='auto')
imcc2 = ax2.contour(phaseBand, ampBand, MI_surr_mat.T, 15, colors='k', linewidths=0.3)#, aspect='auto')

#imc2 = ax1.contour(phaseBand, ampBand, MI_raw.T, 15)#, aspect='auto')

#imc = ax1.contourf(MI_raw.T, 12, extent=[phaseBand[0]-phaseBandWid/2, phaseBand[-1]+phaseBandWid/2, ampBand[0]-ampBandWid/2, ampBand[-1]+ampBandWid/2])#, aspect='auto')
#imc = ax1.contourf(MI_raw.T, 12, origin='lower', extent=[phaseBand[0]-phaseBandWid/2, phaseBand[-1]+phaseBandWid/2, ampBand[0]-ampBandWid/2, ampBand[-1]+ampBandWid/2])#extent=[phaseBand[0]-phaseBandWid/2, phaseBand[-1]+phaseBandWid/2, ampBand[0]-ampBandWid/2, ampBand[-1]+ampBandWid/2])#, aspect='auto')

#imi = ax2.imshow(np.flip(MI_raw_mat.T, 0), aspect='auto', extent=[phaseBand[0]-phaseBandWid/2, phaseBand[-1]+phaseBandWid/2, ampBand[0]-ampBandWid/2, ampBand[-1]+ampBandWid/2])
plt.colorbar(imc, ax=ax1)
plt.colorbar(imc2, ax=ax2)
ax1.set_xlabel('phase frequency (Hz)')
ax1.set_ylabel('Amplitude frequency (Hz)')
ax1.set_title('raw')
ax2.set_xlabel('phase frequency (Hz)')
ax2.set_ylabel('Amplitude frequency (Hz)')
ax2.set_title('minus-surr')
plt.suptitle('ee1.20_ei1.27_ie1.2137_ii1.08_dsi4.44_dse5.00\n/headnode1/shni2598/brian2/NeuroNet_brian/\nmodel_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp')
#%%
#'/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/dp_num_ii/'
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/dp_num_ii/'

analy_type = 'ii'
sys_argv = 32#int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num
good_dir = 'good'
#%%
data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
#%%
dt = 1/10000;
end = int(20/dt); start = int(4/dt)
spon_rate = np.sum((data.a1.gi.t < end) & (data.a1.gi.t > start))/16/data.a1.param.Ni
spon_rate = np.sum((data.a1.ge.t < end) & (data.a1.ge.t > start))/16/data.a1.param.Ne
print(spon_rate)

#%%
start_time = 19e3; end_time = 20e3
data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 10)
#data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ni, window = 10)
data.a1.ge.get_centre_mass(detect_pattern=True)
pattern_size2 = data.a1.ge.centre_mass.pattern_size.copy()
pattern_size2[np.invert(data.a1.ge.centre_mass.pattern)] = 0
#%%
frames = data.a1.ge.spk_rate.spk_rate.shape[2]
ani = firing_rate_analysis.show_pattern(data.a1.ge.spk_rate.spk_rate, None, frames = frames, start_time = start_time, anititle='',\
                                        show_pattern_size=True, pattern_ctr=data.a1.ge.centre_mass.centre_ind, \
                                            pattern_size=pattern_size2)
#%%
frames = data.a1.ge.spk_rate.spk_rate.shape[2]
ani = show_pattern(data.a1.ge.spk_rate.spk_rate, None, frames = frames, start_time = start_time, anititle='',\
                                        show_pattern_size=True, pattern_ctr=data.a1.ge.centre_mass.centre_ind, \
                                            pattern_size=pattern_size2)

#%%
fig = plt.figure(figsize = (6,6))
ax1=fig.add_subplot(111, label="1",frame_on=False)
ax2=fig.add_subplot(111, label="2",frame_on=False)
ax2.set_xlim([-0.5,62.5])
ax2.set_ylim([-0.5,62.5])
ax1.axis('off')
ax2.axis('off')
ax1.imshow(data.a1.ge.spk_rate.spk_rate[:,:,0])

circle = plt.Circle([data.a1.ge.centre_mass.centre_ind[0,1],62-data.a1.ge.centre_mass.centre_ind[0,0]],pattern_size2[0], lw=1.5,color='r',fill=False)
#circle = plt.Circle([0,0],3)
ax2.add_patch(circle)
#%%
def show_pattern(spkrate1, spkrate2=None, frames = 1000, start_time = 0, anititle='', show_pattern_size=False, pattern_ctr=None, pattern_size=None):
    '''
    create amination for the firing pattern of network
    input:
    spkrate1, spkrate2: (N*N*t array)
    frames: number of frames of amination
    start_time: the real-time of the simulation that the first frame of 'spkrate' cooresponds to
    anititle: set the title of amination
    '''
    if spkrate2 is None:
        
        #from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig = plt.figure(figsize = (6,6))
        ax1=fig.add_subplot(111, label="1",frame_on=False)
        ax1.axis('off');
        # if show_pattern_size:
        #     ax2=fig.add_subplot(111, label="2",frame_on=False)
        #     ax2.set_xlim([-0.5,62.5])
        #     ax2.set_ylim([-0.5,62.5])       
        #     ax2.axis('off')
        
        #fig,[ax1,ax2] = plt.subplots(1,2,figsize = (12,6))
        #ax2.set_xlim([-0.5,62.5]);ax2.set_ylim([-0.5,62.5])
        #ax1= plt.subplot(111)
        #divider = make_axes_locatable(ax1)
        #cbaxes = divider.append_axes("bottom", size="5%", pad=0.05)
        cbaxes = fig.add_axes([0.2, 0.15, 0.6, 0.03]) 
        #fig.colorbar(img1, cax=cax1)
        
        cmap_spk=plt.cm.get_cmap('Blues', 7) # viridis Blues
        cmap_c = np.array([1.,0.,0.,1.])
        cmap_stimulus = np.array([88/255,150/255.,0.,1.])
        cmap = np.vstack((cmap_stimulus,cmap_c,cmap_spk(range(7))))
        cmap = mpl.colors.ListedColormap(cmap)
        #cmap.set_under('red')
        bounds = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        #cbaxes = fig.add_axes([0.2, 0.1, 0.6, 0.03]) 
        
        cb = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap,
                                        norm=norm,
                                        boundaries=bounds,
                                        ticks=np.array([-2,-1, 0, 1, 2, 3, 4, 5, 6])+0.5,
                                        spacing='proportional',
                                        orientation='horizontal') #horizontal vertical
        cb.ax.set_xticklabels(['stimulus','ctr', 0, 1, 2, 3, 4, 5, 6])
        cb.set_label('number of spikes')
        
        titleaxes = fig.add_axes([0.3, 0.75, 0.4, 0.05])
        #titleaxes = divider.append_axes("top", size="5%", pad=0.01)
        titleaxes.axis('off')
        title = titleaxes.text(0.5,0.05, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=titleaxes.transAxes, ha="center")
        time_title = np.arange(spkrate1.shape[2]) + start_time
        
        value1=ax1.matshow(spkrate1[:,:,0], cmap=cb.cmap, norm=cb.norm)
        #value2=ax2.matshow(spk2[:,:,:], cmap=cb.cmap)
        #if show_pattern_size: 
            #ax2 = fig.add_axes([0, 0, 1, 1]) 
            #ax2= plt.subplot(111)
        if show_pattern_size:
            circle = plt.Circle([pattern_ctr[0,1],pattern_ctr[0,0]],pattern_size[0], lw=1.5,color='r',fill=False)
            ax1.add_patch(circle)
        #circle.remove()
        #a=1
        # def init():
        #     value1=ax1.matshow(spkrate1[:,:,0], cmap=cb.cmap, norm=cb.norm)
        #     title.set_text(u"time: {} ms".format(time_title[0]))
        #     #if show_pattern_size:
        #     circle.set_center([pattern_ctr[0,1],62-pattern_ctr[0,0]])
        #     circle.set_radius(pattern_size[0])
        #     ax2.add_patch(circle)
        #     print(a)
        #     return value1,title,circle
            #return title,circle
            #return value1,title,
        
        def updatev(i):
            value1.set_array(spkrate1[:,:,i])
            #value2.set_array(spk2[:,:,i])
            title.set_text(u"time: {} ms".format(time_title[i]))
            #if show_pattern_size:
            if show_pattern_size:
                circle.center = [pattern_ctr[i,1],pattern_ctr[i,0]]
                circle.radius = pattern_size[i]
            #print('size')
                return value1, title,circle,#, value2
            #return title,circle,
            else:
                return value1, title,
        
        #value1.set_clim(vmin=0, vmax=6)

        #if stimu_onset >= 0: fig.suptitle('onset of stimulus:%dms'%(stimu_onset))
        fig.suptitle(anititle)
        ani=animation.FuncAnimation(fig, updatev,  frames=frames, interval=10, blit=True)    # frames=spk1.shape[2] 
        return ani
    # init_func=init
