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
from pathlib import Path

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
root_dir = 'lambda/'
if not os.path.exists(root_dir):
    try: os.makedirs(root_dir)
    except FileExistsError:
        pass
data_dir = 'lambda/raw_data/'
if not os.path.exists(data_dir):
    try: os.makedirs(data_dir)
    except FileExistsError:
        pass
graph_dir = 'lambda/graph/'
if not os.path.exists(graph_dir):
    try: os.makedirs(graph_dir)
    except FileExistsError:
        pass
vedio_dir = 'lambda/vedio/'
if not os.path.exists(vedio_dir):
    try: os.makedirs(vedio_dir)
    except FileExistsError:
        pass
state_dir = 'lambda/state/'
if not os.path.exists(state_dir):
    try: os.makedirs(state_dir)
    except FileExistsError:
        pass
trajectory_dir = f'./{graph_dir}/trajectory/'
Path(trajectory_dir).mkdir(parents=True, exist_ok=True)
jump_dir = f'./{graph_dir}/jump/'
Path(jump_dir).mkdir(parents=True, exist_ok=True)
coactivity_dir = f'./{graph_dir}/coactivity/'
Path(coactivity_dir).mkdir(parents=True, exist_ok=True)

start = time.perf_counter()
#%% adjustable parameters
def find_w_e(w_i,num_i,num_e,ie_ratio):
    return w_i/(num_e/num_i*ie_ratio)
def fine_w_i(w_e,num_e,num_i,ie_ratio):
    return w_e*(num_e/num_i*ie_ratio)

# appoint looping parameters
params_loop = {
    'N_e_ext': np.arange(1200, 2000+1, 50),
    'N_i_ext': np.arange(1360, 2160+1, 50)
}

# generate looping parameter combinations
loop_combinations = list(itertools.product(*params_loop.values()))
# get total looping number
loop_total = len(loop_combinations)

#%% ready to loop
def compute_save_draw(comb, loop_num):
    N_e_ext, N_i_ext = comb
    print(f'looping {loop_num} in {loop_total},\n N_e_ext={N_e_ext}, N_i_ext={N_i_ext}')

    #%%
    # common title & path
    E = str('{E}')
    I = str('{I}')
    bg = str('{bg}')
    common_title = rf'$\lambda^{E}_{bg}$={num_ee}, $\lambda^{I}_{bg}$={num_ei}'
    common_path = f'E{N_e_ext:04d}_I{N_i_ext:04d}'

    # check if data file exists
    # if exists, skip computation
    if os.path.exists(f'{data_dir}data_{common_path}.file'):
        print(f"Data file for {common_path} exists, skipping computation.")
        return None

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
    num_ee = 275
    num_ei = 200
    num_ie = 115
    num_ii = 95
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

    # decide if analyze (p_peak>1) (not compatible with states_parallel.py)
    p_peak_ee=param_a1['p_peak_ee']
    p_peak_ei=param_a1['p_peak_ei']
    p_peak_ie=param_a1['p_peak_ie']
    p_peak_ii=param_a1['p_peak_ii']
    p_peak = np.max([p_peak_ee,p_peak_ei,p_peak_ie,p_peak_ii])
    if p_peak>1:
        return None

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
            'loop_num':loop_num, 
            'data_dir': os.getcwd(),
            'param':param_all,
            'a1':{'param':param_a1,
                'ge':{'i':spk_e_1.i[:],'t':spk_tstep_e1},    
                'gi':{'i':spk_i_1.i[:],'t':spk_tstep_i1}}}
    if record_LFP:
        data['a1']['ge']['LFP'] = lfp_moni.lfp[:]/nA

    ''' save data to disk'''
    with open(f"{data_dir}data_{common_path}.file", 'wb') as file:
        pickle.dump(data, file)
    
    '''load data from disk'''
    data_load = mydata.mydata()
    data_load.load(f"{data_dir}data_E{N_e_ext:04d}_I{N_i_ext:04d}.file")

    #%% analyze
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

    frames = data_load.a1.ge.spk_rate.spk_rate.shape[2]

    stim_on_off = data_load.a1.param.stim1.stim_on-start_time
    stim_on_off = stim_on_off[stim_on_off[:,0]>=0]#[:int(last_stim-first_stim)+1]
    #%% graphs and veideos
    # unwrap periodic trajection
    centre = data_load.a1.ge.centre_mass.centre
    continous_centre = mya.unwrap_periodic_path(centre=centre, width=63)
    continous_jump_size = np.diff(continous_centre)
    continous_jump_dist = np.sqrt(np.sum(continous_jump_size**2,1))

    if not os.path.exists(f'./{trajectory_dir}/trajectory_{common_path}.png'):
        # trajectory
        _ = mya.plot_trajectory(
            data=continous_centre,
            title=f'Centre trajectory of \n {common_title}',
            save_path=f'./{trajectory_dir}/trajectory_{common_path}.png'
            )

    if not os.path.exists(f'./{jump_dir}/jump_{common_path}.png'):
        # pdf power law distribution check
        alpha_jump, r2_jump, _, tail_points_jump = mya.check_jump_power_law(
            continous_jump_dist,
            tail_fraction=0.9,
            save_path=f'./{jump_dir}/jump_{common_path}.png',
            title=f'Jump step distribution of \n {common_title}'
        )
    if tail_points_jump < 8:
        alpha_jump = None
        r2_jump = None

    if not os.path.exists(f'./{coactivity_dir}/coactivity_{common_path}.png'):
        # spike statistic
        alpha_spike, r2_spike, _, tail_points_spike = mya.check_coactive_power_law(
            data_load.a1.ge.spk_rate,
            tail_fraction=1,
            save_path=f'./{coactivity_dir}/coactivity_{common_path}.png',
            title=f'Coactivity distribution of \n {common_title}',
            min_active=1  # 忽略少于1个神经元同时放电的情况
        )
    if tail_points_spike < 8:
        alpha_spike = None
        r2_spike = None

    if not os.path.exists(f'./{vedio_dir}/{common_path}_pattern.mp4'):
        # Animation
        title = f'Animation \n {common_title}'
        ani = fra.show_pattern(spkrate1=data_load.a1.ge.spk_rate.spk_rate,
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

    # record parameters & states
    Analyzer.alpha_r2_collector(
        N_e_ext=N_e_ext,
        N_i_ext=N_i_ext,
        alpha_jump=alpha_jump,
        r2_jump=r2_jump,
        alpha_spike=alpha_spike,
        r2_spike=r2_spike
        )
    return None

Analyzer = fra.CriticalityAnalyzer()
Parallel(n_jobs=-1)(delayed(compute_save_draw)(comb, i+1)
                    for i, comb in enumerate(loop_combinations))

now = datetime.datetime.now()
data_states = {'datetime':now.strftime("%Y-%m-%d %H:%M:%S"), 
               'dt':0.1, 
               'data_dir': os.getcwd(),
               'params': Analyzer.params,
               'states': Analyzer.states
               }

''' save phase data to disk'''
print('saving states data to disk...')
with open(f"{state_dir}auto_states.file", 'wb') as file:
    pickle.dump(data_states, file)
print(f'data states of {loop_total} states saved to {state_dir}')

Analyzer.plot_phase_diagrams(
    save_path=f'{state_dir}/Phase_diagrams_of_loops{loop_total}.png'
    )
# Analyzer.alpha_r2_collector(
#     N_e_ext=N_e_ext,
#     N_i_ext=N_i_ext,
#     alpha_jump=alpha_jump,
#     r2_jump=r2_jump,
#     alpha_spike=alpha_spike,
#     r2_spike=r2_spike)

print(f'total time elapsed: {np.round((time.perf_counter() - start)/60,2)} min')