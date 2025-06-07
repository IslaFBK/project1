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
import datetime
from connection import poisson_stimuli as psti
from connection import pre_process_sc
from connection import preprocess_2area
from connection import build_one_area
from connection import get_stim_scale
from connection import adapt_gaussian
import sys
import pickle
import itertools
import gc
from analysis import firing_rate_analysis as fra
from analysis import my_analysis as mya
from joblib import Parallel, delayed

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
data_dir = 'parallel/raw_data/'
if not os.path.exists(data_dir):
    try: os.makedirs(data_dir)
    except FileExistsError:
        pass
graph_dir = 'parallel/graph/'
if not os.path.exists(graph_dir):
    try: os.makedirs(graph_dir)
    except FileExistsError:
        pass
vedio_dir = 'parallel/vedio/'
if not os.path.exists(vedio_dir):
    try: os.makedirs(vedio_dir)
    except FileExistsError:
        pass

start = time.perf_counter()
#%% adjustable parameters
def find_w_e(w_i,num_i,num_e,ie_ratio):
    return w_i/(num_e/num_i*ie_ratio)
def fine_w_i(w_e,num_e,num_i,ie_ratio):
    return w_e*(num_e/num_i*ie_ratio)

# appoint looping parameters
params_loop = {
    'num_ee': np.arange(100, 400+1, 50),
    'num_ei': np.arange(100, 400+1, 50),
    'num_ie': np.arange(100, 400+1, 50),
    'num_ii': np.arange(100, 400+1, 50)
}

# generate looping parameter combinations
loop_combinations = list(itertools.product(*params_loop.values()))
# get total looping number
loop_total = 1
for arr in params_loop.values():
    loop_total *= len(arr)
#%% ready to loop
def compute_and_save(comb, loop_num):
    num_ee, num_ei, num_ie, num_ii = comb
    print(f'looping {loop_num} in {loop_total},\n num_ee={num_ee},num_ei={num_ei},num_ie={num_ie},num_ii={num_ii}')

    #%% fixed parameters
    record_LFP = True
    # global(1st and 2nd area) physical parameters
    C = 0.25*nF # capacitance
    v_threshold = -50*mV
    v_reset = -70*mV# -60*mV
    t_ref = 5*ms # refractory period
    v_l = -70*mV # leak voltage
    g_l_E = 16.7*nS
    g_l_I = 6.5*nS
    v_k = -85*mV
    tau_k_ = 60   # ms
    delta_gk_1 = 1.9
    delta_gk_2 = 6.5
    v_rev_I = -80*mV
    v_rev_E = 0*mV
    tau_s_de_ = 5
    tau_s_di_ = 4.5
    tau_s_r_ = 1  # ms

    # parameters of adaptation(2nd area)
    new_delta_gk_2 = 0.5
    chg_adapt_range = 7
    w_extnl_ = 5 # nS

    # inter mean weight
    scale_w_12_e = 2.6
    scale_w_12_i = scale_w_12_e
    scale_w_21_e = 0.3
    scale_w_21_i = scale_w_21_e

    # inter decay
    tau_p_d_e1_e2 = 5
    tau_p_d_e1_i2 = tau_p_d_e1_e2
    tau_p_d_e2_e1 = 15
    tau_p_d_e2_i1 = tau_p_d_e2_e1

    #inter probability peak
    peak_p_e1_e2 = 0.3
    peak_p_e1_i2 = 0.3
    peak_p_e2_e1 = 0.2
    peak_p_e2_i1 = 0.5

    # mean synaptic weight
    w_ie_1 = 115
    w_ii_1 = 100

    # IE-ratio
    ie_r_e1 = 2.3641 #
    ie_r_i1 = 1.9706 # 1.52

    #%% build connection set
    # neuron quantity
    ijwd1 = pre_process_sc.get_ijwd()
    ijwd1.Ne = 64*64
    ijwd1.Ni = 32*32
    ijwd1.width = 64

    # distribution 
    ijwd1.decay_p_ee = 7
    ijwd1.decay_p_ei = 8.5 # 8.5/9
    ijwd1.decay_p_ie = 15  # 15/19
    ijwd1.decay_p_ii = 15  # 15/19

    ijwd1.delay = [0.5,2.5] # [min,max]

    # K_a'b'ab
    ijwd1.mean_SynNumIn_ee = num_ee
    ijwd1.mean_SynNumIn_ei = num_ei
    ijwd1.mean_SynNumIn_ie = num_ie
    ijwd1.mean_SynNumIn_ii = num_ii

    # mean synaptic weight
    ijwd1.w_ee_mean = find_w_e(w_ie_1, num_ie, num_ee, ie_r_e1)
    ijwd1.w_ei_mean = find_w_e(w_ii_1, num_ii, num_ei, ie_r_i1)
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

    #%%
    start_scope()
    onearea_net = build_one_area.one_area()

    # Brain2 build
    group_e_1, group_i_1, syn_ee_1, syn_ei_1, syn_ie_1, syn_ii_1 = onearea_net.build(ijwd1)
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
    N_e_ext = 1700
    N_i_ext = 1760
    pois_bkgExt_e1 = PoissonInput(target=group_e_1,
                                target_var='x_E_extnl',
                                N=N_e_ext,
                                rate=1*Hz,
                                weight=5*nS)

    pois_bkgExt_i1 = PoissonInput(target=group_i_1,
                                target_var='x_E_extnl',
                                N=N_i_ext,
                                rate=1*Hz,
                                weight=5*nS)

    #%%


    #%%
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

    #%%
    spk_e_1 = SpikeMonitor(group_e_1, record = True)
    spk_i_1 = SpikeMonitor(group_i_1, record = True)

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

    print(f'single loop of {loop_num} elapsed: {np.round((time.perf_counter() - tic)/60,2)} min')

    #%%
    spk_tstep_e1 = np.round(spk_e_1.t/(0.1*ms)).astype(int)
    spk_tstep_i1 = np.round(spk_i_1.t/(0.1*ms)).astype(int)

    now = datetime.datetime.now()

    param_all = {'delta_gk_1':delta_gk_1,
                'delta_gk_2':delta_gk_2,
                'new_delta_gk_2':new_delta_gk_2,
                'tau_k': tau_k_,
                'tau_s_di':tau_s_di_,
                'tau_s_de':tau_s_de_,
                'tau_s_r':tau_s_r_,
                'num_ee':num_ee,
                'num_ei':num_ei,
                'num_ii':num_ii,
                'num_ie':num_ie,
                'simutime':int(round(simu_time_tot/ms)),
                'chg_adapt_range': chg_adapt_range,
                'ie_r_i1': ie_r_i1,
                't_ref': t_ref/ms}

    data = {'datetime':now.strftime("%Y-%m-%d %H:%M:%S"), 
            'dt':0.1, 
            'loop_num':0, 
            'data_dir': os.getcwd(),
            'param':param_all,
            'a1':{'param':param_a1,
                'ge':{'i':spk_e_1.i[:],'t':spk_tstep_e1},    
                'gi':{'i':spk_i_1.i[:],'t':spk_tstep_i1}}}
    if record_LFP:
        data['a1']['ge']['LFP'] = lfp_moni.lfp[:]/nA

    #%%
    # common title & path
    EE = str('{EE}')
    EI = str('{EI}')
    IE = str('{IE}')
    II = str('{II}')
    common_title = rf'$K^{EE}$={num_ee}, $K^{EI}$={num_ei}, $K^{IE}$={num_ie}, $K^{II}$={num_ii}'
    common_path = f'EE{num_ee:03d}_EI{num_ei:03d}_IE{num_ie:03d}_II{num_ii:03d}'

    ''' save data to disk'''
    with open(f"{data_dir}data_{common_path}.file", 'wb') as file:
        pickle.dump(data, file)

Parallel(n_jobs=-1)(delayed(compute_and_save)(comb, i+1)
                    for i, comb in enumerate(loop_combinations))
print(f'total time elapsed: {np.round((time.perf_counter() - start)/60,2)} min')