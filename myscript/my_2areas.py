#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 2 16:37 2024

Move the variables that need to be traversed in advance out of the loop

@author: liujianing
"""

import brian2.numpy_ as np
import matplotlib.pyplot as plt
import connection as cn
from scipy import sparse
from brian2.only import *
import time
from analysis import mydata
import os
import tempfile
import datetime
from connection import poisson_stimuli as psti
from connection import pre_process_sc
from connection import preprocess_2area
from connection import build_one_area
from connection import build_two_areas
from connection import get_stim_scale
from connection import adapt_gaussian
import sys
import pickle
import itertools
import gc
from analysis import firing_rate_analysis as fra
from analysis import my_analysis as mya
from joblib import Parallel, delayed
from pathlib import Path

from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde
from scipy.stats import levy_stable
from levy import fit_levy, levy

plt.rcParams.update({
    "text.usetex": True,  # 启用 LaTeX 渲染
    "font.family": "serif",  # 使用衬线字体（类似 LaTeX 默认）
    "font.serif": ["Times New Roman"]  # 指定字体
})

#%%
prefs.codegen.target = 'cython'

dir_cache = 'cache/'
prefs.codegen.runtime.cython.cache_dir = dir_cache
prefs.codegen.max_cache_dir_size = 120

#%% OS operation
# test if data_dir exists, if not, create one.
# FileExistsError means if menu is create by other progress or thread, ignore it.
root_dir = '2area/'
if not os.path.exists(root_dir):
    try: os.makedirs(root_dir)
    except FileExistsError:
        pass
data_dir = f'{root_dir}/raw_data/'
if not os.path.exists(data_dir):
    try: os.makedirs(data_dir)
    except FileExistsError:
        pass
graph_dir = f'{root_dir}/graph/'
if not os.path.exists(graph_dir):
    try: os.makedirs(graph_dir)
    except FileExistsError:
        pass
vedio_dir = f'{root_dir}/vedio/'
if not os.path.exists(vedio_dir):
    try: os.makedirs(vedio_dir)
    except FileExistsError:
        pass
state_dir = f'{root_dir}/state/'
if not os.path.exists(state_dir):
    try: os.makedirs(state_dir)
    except FileExistsError:
        pass
MSD_dir = f'./{graph_dir}/MSD/'
Path(MSD_dir).mkdir(parents=True, exist_ok=True)
pdx_dir = f'./{graph_dir}/pdx/'
Path(pdx_dir).mkdir(parents=True, exist_ok=True)
combined_dir = f'./{graph_dir}/combined'
Path(combined_dir).mkdir(parents=True, exist_ok=True)

start = time.perf_counter()
#%% adjustable parameters
def find_w_e(w_i,num_i,num_e,ie_ratio):
    return w_i/(num_e/num_i*ie_ratio)
def find_w_i(w_e,num_e,num_i,ie_ratio):
    return w_e*(num_e/num_i*ie_ratio)


#%%
# common title & path
common_title = rf''
common_path = f''

# check if data file exists
# if 1:
if not os.path.exists(f"{data_dir}data_{common_path}.file") or 1:
    #%% fixed parameters
    record_LFP = True
    # global(1st and 2nd area) physical parameters
    C = 0.25*nF # capacitance
    v_threshold = -50*mV
    v_reset = -70*mV# -60*mV
    t_ref = 4*ms # refractory period
    v_l = -70*mV # leak voltage
    g_l_E = 16.7*nS
    g_l_I = 25*nS
    v_k = -85*mV
    tau_k_ = 60   # ms
    delta_gk_1 = 1.9
    delta_gk_2 = 6.5
    v_rev_E = 0*mV
    v_rev_I = -80*mV
    tau_s_de_ = 5
    tau_s_di_ = 4.5
    tau_s_r_ = 1  # ms

    # in-degree
    num_ee = 270
    num_ei = 350
    num_ie = 130
    num_ii = 180
    # mean synaptic weight 1
    w_ee_1 = 7.857
    w_ei_1 = 10.847
    w_ie_1 = 35.534
    w_ii_1 = 45
    # mean synaptic weight 2
    w_ee_2 = 11
    w_ei_2 = 13.805
    w_ie_2 = 41.835
    w_ii_2 = 50
    # IE-ratio 1
    ie_r_e1 = 2.1775 #
    ie_r_i1 = 2.1336 # 1.52
    # IE-ratio 2
    ie_r_e2 = 1.8312
    ie_r_i2 = 1.8627

    # inter mean weight
    scale_w_12_e = 3.656
    scale_w_12_i = scale_w_12_e
    scale_w_21_e = 0.578
    scale_w_21_i = scale_w_21_e

    # inter decay
    tau_p_d_e1_e2 = 8
    tau_p_d_e1_i2 = tau_p_d_e1_e2
    tau_p_d_e2_e1 = 8
    tau_p_d_e2_i1 = tau_p_d_e2_e1

    #inter probability peak
    peak_p_e1_e2 = 0.4
    peak_p_e1_i2 = 0.4
    peak_p_e2_e1 = 0.4
    peak_p_e2_i1 = 0.4

    #%% build connection set 1
    # neuron quantity
    ijwd1 = pre_process_sc.get_ijwd()
    ijwd1.Ne = 64*64
    ijwd1.Ni = 32*32
    ijwd1.width = 64

    # decay
    ijwd1.decay_p_ee = 7.5
    ijwd1.decay_p_ei = 9.5 # 8.5/9
    ijwd1.decay_p_ie = 19  # 15/19
    ijwd1.decay_p_ii = 19  # 15/19

    ijwd1.delay = [0.5,2.5] # [min,max]

    # K_a'b'ab in-degree
    ijwd1.mean_SynNumIn_ee = num_ee
    ijwd1.mean_SynNumIn_ei = num_ei
    ijwd1.mean_SynNumIn_ie = num_ie
    ijwd1.mean_SynNumIn_ii = num_ii

    # mean synaptic weight
    ijwd1.w_ee_mean = w_ee_1
    ijwd1.w_ei_mean = w_ei_1
    ijwd1.w_ie_mean = w_ie_1
    ijwd1.w_ii_mean = w_ii_1

    ijwd1.generate_ijw()    # generate synaptics and weight
    ijwd1.generate_d_dist() # generate delay

    # load parameters in area 1
    param_a1 = {**ijwd1.__dict__}
    # delete trivial parameters
    del param_a1['i_ee'], param_a1['j_ee'], param_a1['w_ee'], param_a1['d_ee'], param_a1['dist_ee']
    del param_a1['i_ei'], param_a1['j_ei'], param_a1['w_ei'], param_a1['d_ei'], param_a1['dist_ei']
    del param_a1['i_ie'], param_a1['j_ie'], param_a1['w_ie'], param_a1['d_ie'], param_a1['dist_ie']
    del param_a1['i_ii'], param_a1['j_ii'], param_a1['w_ii'], param_a1['d_ii'], param_a1['dist_ii']

    #%% build connection set 2
    # neuron quantity
    ijwd2 = pre_process_sc.get_ijwd()
    ijwd2.Ne = 64*64
    ijwd2.Ni = 32*32
    ijwd2.width = 64

    # decay
    ijwd2.decay_p_ee = 7.5
    ijwd2.decay_p_ei = 9.5
    ijwd2.decay_p_ie = 19  # 15/19
    ijwd2.decay_p_ii = 19  # 15/19

    ijwd2.delay = [0.5,2.5]

    # in-degree
    ijwd2.mean_SynNumIn_ee = num_ee
    ijwd2.mean_SynNumIn_ei = num_ei
    ijwd2.mean_SynNumIn_ie = num_ie
    ijwd2.mean_SynNumIn_ii = num_ii

    # mean synaptic weight
    ijwd2.w_ee_mean = w_ee_2
    ijwd2.w_ei_mean = w_ei_2
    ijwd2.w_ie_mean = w_ie_2
    ijwd2.w_ii_mean = w_ii_2

    ijwd2.generate_ijw()
    ijwd2.generate_d_dist()

    # load parameters in area 2
    param_a2 = {**ijwd2.__dict__}
    # delete trivial parameters
    del param_a2['i_ee'], param_a2['j_ee'], param_a2['w_ee'], param_a2['d_ee'], param_a2['dist_ee']
    del param_a2['i_ei'], param_a2['j_ei'], param_a2['w_ei'], param_a2['d_ei'], param_a2['dist_ei']
    del param_a2['i_ie'], param_a2['j_ie'], param_a2['w_ie'], param_a2['d_ie'], param_a2['dist_ie']
    del param_a2['i_ii'], param_a2['j_ii'], param_a2['w_ii'], param_a2['d_ii'], param_a2['dist_ii']

    #%% build inter connection set
    ijwd_inter = preprocess_2area.get_ijwd_2()
    ijwd_inter.Ne1 = 64*64; ijwd_inter.Ne2 = 64*64; 
    ijwd_inter.width1 = 64; ijwd_inter.width2 = 64; 
    ijwd_inter.p_inter_area_1 = 1/2; ijwd_inter.p_inter_area_2 = 1/2
    ijwd_inter.section_width_1 = 4;  ijwd_inter.section_width_2 = 4; 
    ijwd_inter.peak_p_e1_e2 = peak_p_e1_e2; ijwd_inter.tau_p_d_e1_e2 = tau_p_d_e1_e2
    ijwd_inter.peak_p_e1_i2 = peak_p_e1_i2; ijwd_inter.tau_p_d_e1_i2 = tau_p_d_e1_i2        
    ijwd_inter.peak_p_e2_e1 = peak_p_e2_e1; ijwd_inter.tau_p_d_e2_e1 = tau_p_d_e2_e1
    ijwd_inter.peak_p_e2_i1 = peak_p_e2_i1; ijwd_inter.tau_p_d_e2_i1 = tau_p_d_e2_i1
    
    ijwd_inter.w_e1_e2_mean = scale_w_12_e; ijwd_inter.w_e1_i2_mean = scale_w_12_i
    ijwd_inter.w_e2_e1_mean = scale_w_21_e; ijwd_inter.w_e2_i1_mean = scale_w_21_i

    ijwd_inter.generate_ijwd()

    param_inter = {**ijwd_inter.__dict__}

    del param_inter['i_e1_e2'], param_inter['j_e1_e2'], param_inter['w_e1_e2'], param_inter['d_e1_e2'] 
    del param_inter['i_e1_i2'], param_inter['j_e1_i2'], param_inter['w_e1_i2'], param_inter['d_e1_i2'] 
    del param_inter['i_e2_e1'], param_inter['j_e2_e1'], param_inter['w_e2_e1'], param_inter['d_e2_e1'] 
    del param_inter['i_e2_i1'], param_inter['j_e2_i1'], param_inter['w_e2_i1'], param_inter['d_e2_i1']

    #%%
    start_scope()
    twoarea_net = build_two_areas.two_areas()

    # Brain2 build
    group_e_1, group_i_1, syn_ee_1, syn_ei_1, syn_ie_1, syn_ii_1,\
    group_e_2, group_i_2, syn_ee_2, syn_ei_2, syn_ie_2, syn_ii_2,\
    syn_e1e2, syn_e1i2, syn_e2e1, syn_e2i1 = twoarea_net.build(ijwd1, ijwd2, ijwd_inter)
    #%%
    '''record LFP'''
    if record_LFP:
        from connection import get_LFP

        LFP_elec = np.array([[0,0],[-32,-32]])
        i_LFP,j_LFP,w_LFP = get_LFP.get_LFP(ijwd1.e_lattice,LFP_elec,
                                            width=ijwd1.width,
                                            LFP_sigma=8,LFP_effect_range=2.5)
        group_LFP_record = NeuronGroup(len(LFP_elec),
                                    model=get_LFP.LFP_recordneuron)
        syn_LFP = Synapses(group_e_1,group_LFP_record,model=get_LFP.LFP_syn)
        syn_LFP.connect(i=i_LFP,j=j_LFP)
        syn_LFP.w[:] = w_LFP[:]
                                                
    #%%
    '''stim 1; constant amplitude'''
    '''no attention''' # ?background?
    stim_dura = 1000 # ms duration of each stimulus presentation
    transient = 3000 # ms initial transient period; when add stimulus
    inter_time = 2000 # ms interval between trials without and with attention

    stim_scale_cls = get_stim_scale.get_stim_scale()
    stim_scale_cls.seed = 10 # random seed
    n_StimAmp = 1
    n_perStimAmp = 1
    stim_amp_scale = np.ones(n_StimAmp*n_perStimAmp)
    for i in range(n_StimAmp):
        stim_amp_scale[i*n_perStimAmp:i*n_perStimAmp+n_perStimAmp] = 2**(i)

    stim_scale_cls.stim_amp_scale = stim_amp_scale
    stim_scale_cls.stim_dura = stim_dura
    stim_scale_cls.separate_dura = np.array([300,600])
    stim_scale_cls.get_scale()
    stim_scale_cls.n_StimAmp = n_StimAmp
    stim_scale_cls.n_perStimAmp = n_perStimAmp

    # concatenate
    init = np.zeros(transient//stim_scale_cls.dt_stim)
    stim_scale_cls.scale_stim = np.concatenate((init,stim_scale_cls.scale_stim))
    stim_scale_cls.stim_on += transient

    #%%
    scale_1 = TimedArray(stim_scale_cls.scale_stim, dt=10*ms)
    data_ = mydata.mydata()
    param_a1 = {**param_a1, 'stim1':data_.class2dict(stim_scale_cls)}

    #%%
    N_e_ext = 1600
    N_i_ext = 1600
    pois_bkgExt_e1 = PoissonInput(target=group_e_1,
                                target_var='x_E_extnl',
                                N=N_e_ext,
                                rate=1*Hz,
                                weight=5*nS)
    pois_bkgExt_e2 = PoissonInput(target=group_e_2,
                                target_var='x_E_extnl',
                                N=N_e_ext,
                                rate=1*Hz,
                                weight=5*nS)
    pois_bkgExt_i1 = PoissonInput(target=group_i_1,
                                target_var='x_E_extnl',
                                N=N_i_ext,
                                rate=1*Hz,
                                weight=5*nS)
    pois_bkgExt_i2 = PoissonInput(target=group_i_2,
                                target_var='x_E_extnl',
                                N=N_i_ext,
                                rate=1*Hz,
                                weight=5*nS)

    #%%


    #%% group 1
    group_e_1.tau_s_de = tau_s_de_*ms; 
    group_e_1.tau_s_di = tau_s_di_*ms
    group_e_1.tau_s_re = group_e_1.tau_s_ri = tau_s_r_*ms

    group_e_1.tau_s_de_inter = tau_s_de_*ms #5.0*ms; 
    group_e_1.tau_s_re_inter = 1*ms
    group_e_1.tau_s_de_extnl = 5.0*ms #5.0*ms
    group_e_1.tau_s_re_extnl = 1*ms

    group_i_1.tau_s_de = tau_s_de_*ms
    group_i_1.tau_s_di = tau_s_di_*ms
    group_i_1.tau_s_re = group_i_1.tau_s_ri = tau_s_r_*ms

    group_i_1.tau_s_de_inter = tau_s_de_*ms #5.0*ms; 
    group_i_1.tau_s_re_inter = 1*ms
    group_i_1.tau_s_de_extnl = 5.0*ms #5.0*ms
    group_i_1.tau_s_re_extnl = 1*ms

    group_e_1.v = np.random.random(ijwd1.Ne)*35*mV-85*mV
    group_i_1.v = np.random.random(ijwd1.Ni)*35*mV-85*mV
    group_e_1.delta_gk = delta_gk_1*nS
    group_e_1.tau_k = tau_k_*ms
    group_e_1.g_l = g_l_E
    group_i_1.g_l = g_l_I

    group_e_1.I_extnl_crt = 0*nA # 0.25 0.51*nA
    group_i_1.I_extnl_crt = 0*nA # 0.25 0.60*nA

    #%% group 2
    group_e_2.tau_s_de = tau_s_de_*ms; 
    group_e_2.tau_s_di = tau_s_di_*ms
    group_e_2.tau_s_re = group_e_2.tau_s_ri = tau_s_r_*ms

    group_e_2.tau_s_de_inter = tau_s_de_*ms #5.0*ms; 
    group_e_2.tau_s_re_inter = 1*ms
    group_e_2.tau_s_de_extnl = 5.0*ms #5.0*ms
    group_e_2.tau_s_re_extnl = 1*ms

    group_i_2.tau_s_de = tau_s_de_*ms
    group_i_2.tau_s_di = tau_s_di_*ms
    group_i_2.tau_s_re = group_i_2.tau_s_ri = tau_s_r_*ms

    group_i_2.tau_s_de_inter = tau_s_de_*ms #5.0*ms; 
    group_i_2.tau_s_re_inter = 1*ms
    group_i_2.tau_s_de_extnl = 5.0*ms #5.0*ms
    group_i_2.tau_s_re_extnl = 1*ms

    group_e_2.v = np.random.random(ijwd1.Ne)*35*mV-85*mV
    group_i_2.v = np.random.random(ijwd1.Ni)*35*mV-85*mV
    group_e_2.delta_gk = delta_gk_2*nS
    group_e_2.tau_k = tau_k_*ms
    group_e_2.g_l = g_l_E
    group_i_2.g_l = g_l_I

    group_e_2.I_extnl_crt = 0*nA # 0.25 0.51*nA
    group_i_2.I_extnl_crt = 0*nA # 0.25 0.60*nA

    #%%
    spk_e_1 = SpikeMonitor(group_e_1, record = True)
    spk_i_1 = SpikeMonitor(group_i_1, record = True)
    spk_e_2 = SpikeMonitor(group_e_2, record = True)
    spk_i_2 = SpikeMonitor(group_i_2, record = True)

    if record_LFP:
        lfp_moni = StateMonitor(group_LFP_record, ('lfp'), record = True)

    #%%
    net = Network(collect())
    net.store('state1')

    #%%
    tic = time.perf_counter()

    simu_time_tot = (stim_scale_cls.stim_on[-1,1] + 500)*ms
    simu_time1 = (stim_scale_cls.stim_on[n_StimAmp*n_perStimAmp-1,1] + round(inter_time/2))*ms
    simu_time2 = simu_time_tot - simu_time1

    net.run(simu_time1, profile=False) #,namespace={'tau_k': 80*ms}
    net.run(simu_time2, profile=False) #,namespace={'tau_k': 80*ms}

    print(f'computation elapsed: {np.round((time.perf_counter() - tic)/60,2)} min')

    #%%
    spk_tstep_e1 = np.round(spk_e_1.t/(0.1*ms)).astype(int)
    spk_tstep_i1 = np.round(spk_i_1.t/(0.1*ms)).astype(int)
    spk_tstep_e2 = np.round(spk_e_2.t/(0.1*ms)).astype(int)
    spk_tstep_i2 = np.round(spk_i_2.t/(0.1*ms)).astype(int)

    now = datetime.datetime.now()

    param_all = {'delta_gk_1':delta_gk_1,
                 'delta_gk_2':delta_gk_2,
                 'tau_k': tau_k_,
                 'tau_s_di':tau_s_di_,
                 'tau_s_de':tau_s_de_,
                 'tau_s_r':tau_s_r_,
                 'num_ee':num_ee,
                 'num_ei':num_ei,
                 'num_ii':num_ii,
                 'num_ie':num_ie,
                 'simutime':int(round(simu_time_tot/ms)),
                 'ie_r_e1': ie_r_e1,
                 'ie_r_i1': ie_r_i1,
                 'ie_r_e2': ie_r_e2,
                 'ie_r_i2': ie_r_i2,
                 't_ref': t_ref/ms}

    data = {'datetime':now.strftime("%Y-%m-%d %H:%M:%S"), 
            'dt':0.1, 
            'data_dir': os.getcwd(),
            'param':param_all,
            'a1':{'param':param_a1,
                'ge':{'i':spk_e_1.i[:],'t':spk_tstep_e1},    
                'gi':{'i':spk_i_1.i[:],'t':spk_tstep_i1}},
            'a2':{'param':param_a2,
              'ge':{'i':spk_e_2.i[:],'t':spk_tstep_e2},
              'gi':{'i':spk_i_2.i[:],'t':spk_tstep_i2}},
            'inter':{'param':param_inter}}
    if record_LFP:
        data['a1']['ge']['LFP'] = lfp_moni.lfp[:]/nA

    ''' save data to disk'''
    with open(f"{data_dir}data_{common_path}.file", 'wb') as file:
        pickle.dump(data, file)

print(f'loading data')
'''load data from disk'''
data_load = mydata.mydata()
data_load.load(f"{data_dir}data_{common_path}.file")

# reclaim time parameters
'''stim 1; constant amplitude'''
'''no attention''' # ?background?
stim_dura = 1000 # ms duration of each stimulus presentation
transient = 3000 # ms initial transient period; when add stimulus
inter_time = 2000 # ms interval between trials without and with attention

stim_scale_cls = get_stim_scale.get_stim_scale()
stim_scale_cls.seed = 10 # random seed
n_StimAmp = 1
n_perStimAmp = 1
stim_amp_scale = np.ones(n_StimAmp*n_perStimAmp)
for i in range(n_StimAmp):
    stim_amp_scale[i*n_perStimAmp:i*n_perStimAmp+n_perStimAmp] = 2**(i)

stim_scale_cls.stim_amp_scale = stim_amp_scale
stim_scale_cls.stim_dura = stim_dura
stim_scale_cls.separate_dura = np.array([300,600])
stim_scale_cls.get_scale()
stim_scale_cls.n_StimAmp = n_StimAmp
stim_scale_cls.n_perStimAmp = n_perStimAmp

# concatenate
init = np.zeros(transient//stim_scale_cls.dt_stim)
stim_scale_cls.scale_stim = np.concatenate((init,stim_scale_cls.scale_stim))
stim_scale_cls.stim_on += transient

simu_time_tot = (stim_scale_cls.stim_on[-1,1] + 500)*ms
simu_time1 = (stim_scale_cls.stim_on[n_StimAmp*n_perStimAmp-1,1] + round(inter_time/2))*ms
simu_time2 = simu_time_tot - simu_time1

#%% analyze
print(f'analyzing data')
start_time = transient - 500  #data.a1.param.stim1.stim_on[first_stim,0] - 300
end_time = int(round(simu_time_tot/ms))   #data.a1.param.stim1.stim_on[last_stim,0] + 1500
window = 15
data_load.a1.ge.get_spike_rate(start_time=start_time,
                               end_time=end_time,
                               sample_interval=1,
                               n_neuron = data_load.a1.param.Ne,
                               window = window)
data_load.a1.ge.get_centre_mass()
data_load.a1.ge.overlap_centreandspike()

data_load.a2.ge.get_spike_rate(start_time=start_time,
                               end_time=end_time,
                               sample_interval=1,
                               n_neuron = data_load.a1.param.Ne,
                               window = window)
data_load.a2.ge.get_centre_mass()
data_load.a2.ge.overlap_centreandspike()

frames = data_load.a1.ge.spk_rate.spk_rate.shape[2]

stim_on_off = data_load.a1.param.stim1.stim_on-start_time
stim_on_off = stim_on_off[stim_on_off[:,0]>=0]#[:int(last_stim-first_stim)+1]
#%% graphs and veideos

# MSD
def find_best_linear_region(log_time, log_msd, min_points=5):
    n = len(log_time)
    best_r2 = -np.inf
    best_range = min_points
    for end in range(min_points, n+1):
        x = log_time[:end].reshape(-1, 1)
        y = log_msd[:end]
        model = LinearRegression().fit(x, y)
        r2 = model.score(x, y)
        if r2 > best_r2:
            best_r2 = r2
            best_end = end
    return 0, best_end
# area 1
if not os.path.exists(f'{MSD_dir}/MSD1_{common_path}.png') or 0:
    # linear step
    jump_interval1 = np.linspace(1, 1000, 100)

    # # exponential step
    # jump_interval = np.around(np.logspace(0, 3, num=100, base=10))

    data_load.a1.ge.get_MSD(start_time=start_time,
                            end_time=end_time,
                            sample_interval=1,
                            n_neuron = data_load.a1.param.Ne,
                            window = window,
                            dt = 0.1,
                            slide_interval=1,
                            jump_interval=jump_interval1,
                            fit_stableDist='pylevy')
    MSD1 = data_load.a1.ge.MSD.MSD
    jump_interval1 = data_load.a1.ge.MSD.jump_interval
    log_jump_interval1 = np.log10(jump_interval1)
    log_MSD1 = np.log10(MSD1)
    start1, end1 = find_best_linear_region(log_jump_interval1, log_MSD1, min_points=5)
    x_fit1 = log_jump_interval1[start1:end1]
    y_fit1 = log_MSD1[start1:end1]
    model1 = LinearRegression().fit(x_fit1.reshape(-1, 1), y_fit1)
    y_pred1 = model1.predict(x_fit1.reshape(-1, 1))
    slope1 = model1.coef_[0]
    slope_str1 = f'{slope1:.2f}'
    plt.figure(figsize=(6, 6))
    plt.plot(jump_interval1, MSD1, color="#000000")
    plt.plot(10**x_fit1, 10**y_pred1, 'r--', label='Linear Fit')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\tau$ (ms)', fontsize=16)
    plt.ylabel('MSD (gridpoint$^2$)', fontsize=16)
    plt.text(
        0.2, 0.8,
        rf'$\tau^{{{slope_str1}}}$',
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    plt.savefig(f'{MSD_dir}/MSD1_{common_path}.png', dpi=300, bbox_inches='tight')
    plt.close()
# area 2
if not os.path.exists(f'{MSD_dir}/MSD2_{common_path}.png') or 0:
    # linear step
    jump_interval2 = np.linspace(1, 1000, 100)

    # # exponential step
    # jump_interval = np.around(np.logspace(0, 3, num=100, base=10))

    data_load.a2.ge.get_MSD(start_time=start_time,
                            end_time=end_time,
                            sample_interval=1,
                            n_neuron = data_load.a2.param.Ne,
                            window = window,
                            dt = 0.1,
                            slide_interval=1,
                            jump_interval=jump_interval2,
                            fit_stableDist='pylevy')
    MSD2 = data_load.a2.ge.MSD.MSD
    jump_interval2 = data_load.a2.ge.MSD.jump_interval    
    log_jump_interval2 = np.log10(jump_interval2)
    log_MSD2 = np.log10(MSD2)
    start2, end2 = find_best_linear_region(log_jump_interval2, log_MSD2, min_points=5)
    x_fit2 = log_jump_interval2[start2:end2]
    y_fit2 = log_MSD2[start2:end2]
    model2 = LinearRegression().fit(x_fit2.reshape(-1, 1), y_fit2)
    y_pred2 = model2.predict(x_fit2.reshape(-1, 1))
    slope2 = model2.coef_[0]
    slope_str2 = f'{slope2:.2f}'
    plt.figure(figsize=(6, 6))
    plt.plot(jump_interval2, MSD2, color="#000000")
    plt.plot(10**x_fit2, 10**y_pred2, 'r--', label='Linear Fit')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\tau$ (ms)', fontsize=16)
    plt.ylabel('MSD (gridpoint$^2$)', fontsize=16)
    plt.text(
        0.2, 0.8,
        rf'$\tau^{{{slope_str2}}}$',
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    plt.savefig(f'{MSD_dir}/MSD2_{common_path}.png', dpi=300, bbox_inches='tight')
    plt.close()

# pdx
# area 1
if not os.path.exists(f'{pdx_dir}/pdx1_{common_path}.png') or 1:
    jump_x1 = data_load.a1.ge.centre_mass.jump_size[:,0]
    plt.figure(figsize=(6, 6))
    bins = np.arange(-41, 41 + 2, 2)
    plt.hist(jump_x1, 
                bins=bins, 
                density=True, 
                alpha=0.5, 
                label='histogram', 
                rwidth=0.8, 
                color="#000000")
    # kde = gaussian_kde(jump_x)
    # x = np.linspace(jump_x.min(), jump_x.max(), 200)
    # plt.plot(x, kde(x), label='KDE')
    # fit stable distribution
    params1, nll1 = fit_levy(jump_x1)
    alpha1, beta1, mu1, sigma1 = params1.get()
    x1 = np.linspace(jump_x1.min(), jump_x1.max(), 200)
    pdf_fit1 = levy(x1, alpha1, beta1, mu1, sigma1)
    plt.plot(x1, pdf_fit1, 'r-', label='Levy fit')
    plt.xlabel(r'$\Delta$ x (gridpoint)', fontsize=16)
    plt.ylabel('Probability density', fontsize=16)
    plt.text(
        0.95, 0.95,
        rf'$\alpha: {alpha1:.2f}\\ \beta:{beta1:.2f}\\ \mu: {mu1:.2f}\\ \sigma: {sigma1:.2f}$',
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    plt.xlim(-45, 45)
    plt.legend()
    plt.savefig(f'{pdx_dir}/pdx1_{common_path}.png', dpi=300, bbox_inches='tight')
    plt.close()
# area 2
if not os.path.exists(f'{pdx_dir}/pdx2_{common_path}.png') or 1:
    jump_x2 = data_load.a2.ge.centre_mass.jump_size[:,0]
    plt.figure(figsize=(6, 6))
    bins = np.arange(-41, 41 + 2, 2)
    plt.hist(jump_x2, 
                bins=bins, 
                density=True, 
                alpha=0.5, 
                label='histogram', 
                rwidth=0.8, 
                color="#000000")
    # kde = gaussian_kde(jump_x)
    # x = np.linspace(jump_x.min(), jump_x.max(), 200)
    # plt.plot(x, kde(x), label='KDE')
    # fit stable distribution
    params2, nll2 = fit_levy(jump_x2)
    alpha2, beta2, mu2, sigma2 = params2.get()
    x2 = np.linspace(jump_x2.min(), jump_x2.max(), 200)
    pdf_fit2 = levy(x2, alpha2, beta2, mu2, sigma2)
    plt.plot(x2, pdf_fit2, 'r-', label='Levy fit')
    plt.xlabel(r'$\Delta$ x (gridpoint)', fontsize=16)
    plt.ylabel('Probability density', fontsize=16)
    plt.text(
        0.95, 0.95,
        rf'$\alpha: {alpha2:.2f}\\ \beta:{beta2:.2f}\\ \mu: {mu2:.2f}\\ \sigma: {sigma2:.2f}$',
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    plt.xlim(-45, 45)
    plt.legend()
    plt.savefig(f'{pdx_dir}/pdx2_{common_path}.png', dpi=300, bbox_inches='tight')
    plt.close()

# combined graph
import matplotlib.image as mpimg
# area 1
if not os.path.exists(f'{combined_dir}/1{common_path}.png') or 1:
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # MSD
    img_msd1 = mpimg.imread(f'{MSD_dir}/MSD1_{common_path}.png')
    ax[0].imshow(img_msd1)
    ax[0].axis('off')
    # pdx
    img_pdx1 = mpimg.imread(f'{pdx_dir}/pdx1_{common_path}.png')
    ax[1].imshow(img_pdx1)
    ax[1].axis('off')
    plt.tight_layout()
    plt.savefig(f'{combined_dir}/1{common_path}.png', dpi=300, bbox_inches='tight')
    plt.close()
# area 2
if not os.path.exists(f'{combined_dir}/2{common_path}.png') or 1:
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # MSD
    img_msd2 = mpimg.imread(f'{MSD_dir}/MSD2_{common_path}.png')
    ax[0].imshow(img_msd2)
    ax[0].axis('off')
    # pdx
    img_pdx2 = mpimg.imread(f'{pdx_dir}/pdx2_{common_path}.png')
    ax[1].imshow(img_pdx2)
    ax[1].axis('off')
    plt.tight_layout()
    plt.savefig(f'{combined_dir}/2{common_path}.png', dpi=300, bbox_inches='tight')
    plt.close()

# if not os.path.exists(f'./{trajectory_dir}/trajectory_{common_path}.png'):
#     # trajectory
#     _ = mya.plot_trajectory(
#         data=continous_centre,
#         title=f'Centre trajectory of \n {common_title}',
#         save_path=f'./{trajectory_dir}/trajectory_{common_path}.png'
#         )
# animation
# 2 areas
if not os.path.exists(f'./{vedio_dir}/{common_path}_pattern.mp4') or 0:
    # Animation
    title = f'Animation \n {common_title}'
    ani = fra.show_pattern(spkrate1=data_load.a1.ge.spk_rate.spk_rate,
                           spkrate2=data_load.a2.ge.spk_rate.spk_rate,
                           frames = frames,
                           start_time = start_time,
                           interval_movie=15,
                           anititle=title,
                           stim=None, 
                           adpt=None)
    ani.save(f'./{vedio_dir}/{common_path}_pattern.mp4',writer='ffmpeg',fps=60,dpi=100)

# release RAM
plt.close('all')
gc.collect()

print(f'total time elapsed: {np.round((time.perf_counter() - start)/60,2)} min')