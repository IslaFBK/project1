# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 10:28:34 2020

@author: nishe
"""

'''
Adam neuron model, change adaptation
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
import os
#%%
prefs.codegen.target = 'cython'

dir_cache = './cache/cache%s' %sys.argv[1]
prefs.codegen.runtime.cython.cache_dir = dir_cache
prefs.codegen.max_cache_dir_size = 30
#%%
sys_argv = int(sys.argv[1])
loop_num = -1

repeat = 5
for p_ee in [0.08]*repeat:
    for scale_ee_ in [0.85+0.3/9]: #np.arange(1,2):
        for delta_gk_ in [10]: #np.arange(2, 8):
            for new_delta_gk_ in [0]:
                for tau_s_de_ in [5.8]:
                    for tau_s_di_ in [6.5]:
                        for ie_ratio_ in 3.375*np.arange(0.85, 1.15, 0.02):#np.linspace(2.4, 4.5, 20):
                            loop_num += 1
                            if loop_num == sys_argv:
                                print(loop_num)
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

Ne = 3969; Ni = 1024

ijwd = pre_process.get_ijwd(Ni=Ni)
ijwd.p_ee = p_ee
ijwd.w_ee_dist = 'lognormal'
ijwd.hybrid = 0.4
ijwd.cn_scale_weight = 2
ijwd.cn_scale_wire = 2
ijwd.iter_num = 5

ijwd.ie_ratio = ie_ratio_
scale_ee_1 = scale_ee_;#scale_ee_ei_ii; 
scale_ei_1 = 1#scale_ee_ei_ii; 
scale_ii_1 = 1 #scale_ee_ei_ii

ijwd.mean_J_ee = 4*10**-3*scale_ee_1 # usiemens
ijwd.sigma_J_ee = 1.9*10**-3*scale_ee_1 # usiemens

ijwd.w_ei = 5*10**(-3)*scale_ei_1 #usiemens
ijwd.w_ii = 25*10**(-3)*scale_ii_1

ijwd.change_dependent_para()
param = {**ijwd.__dict__}

ijwd.generate_ijw()
ijwd.generate_d_rand()
#%%

LFP_loca = np.meshgrid(np.linspace(-31.5, 31.5-(63/4), 4), np.linspace(31.5-(63/4), -31.5, 4))
LFP_loca = np.concatenate((LFP_loca[0].reshape(-1,1), LFP_loca[1].reshape(-1,1)),1)
#LFP_loca[5,:] = np.array([-1,1]) 
#%%
def LFP_w(lattice_ext, LFP_loca):
    i = np.array([], int);
    j = np.array([], int);
    w = np.array([], float);
    sigma = 8.
    for k in range(len(LFP_loca)):
        dist = cn.coordination.lattice_dist(lattice_ext,62,LFP_loca[k])
        ww = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-((dist**2)/(2*sigma**2)))
        ww = ww[dist <= sigma*2.5]
        ii = np.where(dist <= sigma*2.5)[0]
#        ii = np.where(ww >= 0.01)[0]
        i = np.concatenate((i,ii))
        j = np.concatenate((j, np.ones(len(ii),int)*k))
        w = np.concatenate((w, ww))
    return i, j, w
#%%
lfpe_i, lfpe_j, lfpe_w =  LFP_w(ijwd.lattice_ext, LFP_loca)

param_new = {'LFP_loca':LFP_loca, 'LFP_sigma':8}
param = {**param, **param_new}


#%%
chg_adapt_loca = [0, 0]
chg_adapt_range = 6
chg_adapt_neuron = cn.findnearbyneuron.findnearbyneuron(ijwd.lattice_ext, chg_adapt_loca, chg_adapt_range, ijwd.width)
#%%
start_scope()
#%%

neuronmodel_e = cn.model_neu_syn_AD.neuron_e_AD
neuronmodel_i = cn.model_neu_syn_AD.neuron_i_AD

synapse_e = cn.model_neu_syn_AD.synapse_e_AD
synapse_i = cn.model_neu_syn_AD.synapse_i_AD
#%%
group_e_1 =NeuronGroup(ijwd.Ne, model=neuronmodel_e,
                     threshold='v>v_threshold', method='rk2',
                     reset='''v = v_reset
                              g_k += delta_gk''', refractory='(t-lastspike)<t_ref')

group_i_1 =NeuronGroup(ijwd.Ni, model=neuronmodel_i,
                     threshold='v>v_threshold', method='rk2',
                     reset='v = v_reset', refractory='(t-lastspike)<t_ref')

syn_ee_1 = Synapses(group_e_1, group_e_1, model=synapse_e, 
                  on_pre='x_E_post += w')
syn_ei_1 = Synapses(group_e_1, group_i_1, model=synapse_e, 
                  on_pre='x_E_post += w')
syn_ie_1 = Synapses(group_i_1, group_e_1, model=synapse_i, 
                  on_pre='x_I_post += w')
syn_ii_1 = Synapses(group_i_1, group_i_1, model=synapse_i, 
                  on_pre='x_I_post += w')
#%%
'''external input'''
#stim_rate = psti.input_spkrate(maxrate = [800,800], sig=[6,6], position=[[0, 0],[31.5,31.5]])*Hz
bkg_rate2e = 850*Hz
bkg_rate2i = 1000*Hz
extnl_e = PoissonGroup(ijwd.Ne, bkg_rate2e)
extnl_i = PoissonGroup(ijwd.Ni, bkg_rate2i)

#tau_x_re = 1*ms
syn_extnl_e = Synapses(extnl_e, group_e_1, model=synapse_e, on_pre='x_E_extnl_post += w')
syn_extnl_i = Synapses(extnl_i, group_i_1, model=synapse_e, on_pre='x_E_extnl_post += w')

syn_extnl_e.connect('i==j')
syn_extnl_i.connect('i==j')

w_extnl_ = 1.5 # nS
syn_extnl_e.w = w_extnl_*5.8*nS#*tau_s_de_*nS
syn_extnl_i.w = w_extnl_*5.8*nS#*tau_s_de_*nS
#%%
LFP_recordneuron = '''
lfp1 : amp
lfp2 : amp
lfp3 : amp
'''
LFP_syn = '''
w : 1
lfp1_post = (I_exi_pre - I_inh_pre)*w*(1) : amp (summed)
lfp2_post = (I_exi_pre + I_inh_pre)*w*(-1) : amp (summed)
lfp3_post = (I_exi_pre + I_inh_pre + I_extnl_spk_pre)*w*(-1) : amp (summed)
'''
group_LFP_record = NeuronGroup(len(LFP_loca), model = LFP_recordneuron)
syn_lfpe = Synapses(group_e_1, group_LFP_record, model = LFP_syn)
syn_lfpe.connect(i=lfpe_i, j=lfpe_j)
syn_lfpe.w[:] = lfpe_w
#%%
syn_ee_1.connect(i=ijwd.i_ee, j=ijwd.j_ee)
syn_ei_1.connect(i=ijwd.i_ei, j=ijwd.j_ei)
syn_ie_1.connect(i=ijwd.i_ie, j=ijwd.j_ie)
syn_ii_1.connect(i=ijwd.i_ii, j=ijwd.j_ii)

#%%
#tau_s_di_try = tau_s_di_
syn_ee_1.w = ijwd.w_ee*usiemens * 5.8 #tau_s_de_
syn_ei_1.w = ijwd.w_ei*usiemens * 5.8 #tau_s_de_ #5*nS
syn_ii_1.w = ijwd.w_ii*usiemens * 6. #tau_s_di_#25*nS
syn_ie_1.w = ijwd.w_ie*usiemens * 6. #tau_s_di_#
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
syn_ee_1 = set_delay(syn_ee_1, ijwd.d_ee)
syn_ie_1 = set_delay(syn_ie_1, ijwd.d_ie)
syn_ei_1 = set_delay(syn_ei_1, ijwd.d_ei)
syn_ii_1 = set_delay(syn_ii_1, ijwd.d_ii)

#syn_pois_e = set_delay(syn_pois_e)
#syn_pois_i = set_delay(syn_pois_i)

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
#group_e_v2.v = np.random.random(ijwd.Ne)*10*mV-60*mV
#group_i_v2.v = np.random.random(ijwd.Ni)*10*mV-60*mV
group_e_1.I_extnl_crt = 0*nA #0.51*nA
group_i_1.I_extnl_crt = 0*nA #0.60*nA

#%%
spk_e = SpikeMonitor(group_e_1, record = True)
spk_i = SpikeMonitor(group_i_1, record = True)
lfp_moni = StateMonitor(group_LFP_record, ('lfp1','lfp2','lfp3'), record = True)
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
#tau_s_re = 1*ms
#tau_s_ri = 1*ms
#tau_x_re = 1*ms
tic = time.perf_counter()
#seed(10)
simu_time1 = 10000*ms#2000*ms
simu_time2 = 20000*ms
#simu_time2 = 2000*ms#8000*ms
#simu_time3 = 1000*ms
simu_time_tot = 30000*ms
#group_input.active = False
extnl_e.rates = bkg_rate2e
net.run(simu_time1, profile=False) #,namespace={'tau_k': 80*ms}
#extnl_e.rates = bkg_rate2e + stim_rate
group_e_1.delta_gk[chg_adapt_neuron] = new_delta_gk_*nS
net.run(simu_time2, profile=False) #,namespace={'tau_k': 80*ms}
#extnl_e.rates = bkg_rate2e
#net.run(simu_time3, profile=False) #,namespace={'tau_k': 80*ms}
#group_input.active = True
#group_e_1.delta_gk[chg_adapt_neuron] = 2*nS; group_e_1.tau_k[chg_adapt_neuron] = 40*ms
#net.run(simu_time2, report = 'text', profile=False) #,namespace={'tau_k': 80*ms}
print('total time elapsed:',time.perf_counter() - tic)
#%%
import datetime
now = datetime.datetime.now()

#spk_1_tstep = np.round(spk_1.t/(0.1*ms)).astype(int)
spk_tstep_e = np.round(spk_e.t/(0.1*ms)).astype(int)
spk_tstep_i = np.round(spk_i.t/(0.1*ms)).astype(int)

#param_1 = {'delta_gk':10,'tau_s_di':3,'tau_s_de':5,'ie_ratio':3.1725,'sti_onset':'10s'}

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

#param_12 = {'scale_e_12':scale_e_12[i], 'scale_e_21':scale_e_21[j]}

#data = {'param':param_12, 'a1':{'param':param_1,'e':{'i':spk_1.i[:],'t':spk_1_tstep}},
#                            'a2':{'param':param_2,'e':{'i':spk_2.i[:],'t':spk_2_tstep}}}
data = {'datetime':now.strftime("%Y-%m-%d %H:%M:%S"), 'dt':0.1, 'loop_num':loop_num, 'data_dir': os.getcwd(),
        'a1':{'param':param,
              'LFP':{'lfp1':lfp_moni.lfp1/nA, 'lfp2':lfp_moni.lfp2/nA, 'lfp3':lfp_moni.lfp3/nA},
              'ge':{'i':spk_e.i[:],'t':spk_tstep_e},
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
#spkdata2.a1.e.t = spkdata2.a1.e.t*0.1*ms
#
##%%
#spkrate3, centre_ind3, jump_size3, jump_dist3 = psa.get_rate_centre_jumpdist(spkdata2.a1.e, starttime=0*ms, endtime=1000*ms, binforrate=5*ms, sample_interval=1*ms, slide_interval = 1, jump_interval = 15, dt=0.1*ms)
#spkrate3 = psa.overlap_centreandspike(centre_ind3, spkrate3, show_trajectory = False)
#anititle=''#'change adaptation in region near centre to %.1fnS at 1000 ms\ndelta_gk: %.1fnS; tau_s_di: %.1fms; ie_ratio: %.3f'%(spke1.param['new_delta_gk'], spke1.param['delta_gk'], spke1.param['tau_s_di'], spke1.param['ie_ratio'])
#ani3 = psa.show_pattern(spkrate3, spkrate2=0, area_num = 1, frames = 500, start_time = 0, anititle=anititle)
#%%
net.restore('state1')


