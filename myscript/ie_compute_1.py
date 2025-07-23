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
root_dir = 'ie_ratio/'
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
video_dir = f'{root_dir}/vedio/'
if not os.path.exists(video_dir):
    try: os.makedirs(video_dir)
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

#%% ready to loop
def compute_save_draw(comb, loop_num):
    # 为每个进程设置独立的matplotlib缓存目录
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()
    ie_r_e1, ie_r_i1 = comb
    print(f'looping {loop_num} in {loop_total}')

    #%%
    # common title & path
    common_title = rf'$\zeta^{{E}}$: {ie_r_e1:.4f}, $\zeta^{{I}}$: {ie_r_i1:.4f}'
    common_path = f're{ie_r_e1:.4f}_ri{ie_r_i1:.4f}'

    # check if data file exists
    # if 1:
    if not os.path.exists(f"{data_dir}data_{common_path}.file") or 0:
        print(f'computing {loop_num} in {loop_total}')
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
        # mean synaptic weight
        w_ee_1 = 11
        w_ii_1 = 50
        w_ei_1 = find_w_e(w_ii_1, num_ii, num_ei, ie_r_e1)
        w_ie_1 = find_w_i(w_ee_1, num_ee, num_ie, ie_r_i1)

        #%% build connection set
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
        stim_scale_cls.seed = (loop_num-1) % (2**32) # random seed
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
    
    print(f'loading {loop_num} in {loop_total}')
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
    stim_scale_cls.seed = (loop_num-1) % (2**32) # random seed
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
    print(f'analyzing {loop_num} in {loop_total}')
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


    if not os.path.exists(f'{MSD_dir}/MSD_{common_path}.png') or 0:
        # linear step
        jump_interval = np.linspace(1, 1000, 100)

        # # exponential step
        # jump_interval = np.around(np.logspace(0, 3, num=100, base=10))

        data_load.a1.ge.get_MSD(start_time=start_time,
                                end_time=end_time,
                                sample_interval=1,
                                n_neuron = data_load.a1.param.Ne,
                                window = window,
                                dt = 0.1,
                                slide_interval=1,
                                jump_interval=jump_interval,
                                fit_stableDist='pylevy')
        MSD = data_load.a1.ge.MSD.MSD
        jump_interval = data_load.a1.ge.MSD.jump_interval
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
        
        log_jump_interval = np.log10(jump_interval)
        log_MSD = np.log10(MSD)
        start, end = find_best_linear_region(log_jump_interval, log_MSD, min_points=5)
        x_fit = log_jump_interval[start:end]
        y_fit = log_MSD[start:end]
        model = LinearRegression().fit(x_fit.reshape(-1, 1), y_fit)
        y_pred = model.predict(x_fit.reshape(-1, 1))
        slope = model.coef_[0]
        slope_str = f'{slope:.2f}'
        plt.figure(figsize=(6, 6))
        plt.plot(jump_interval, MSD, color="#000000")
        plt.plot(10**x_fit, 10**y_pred, 'r--', label='Linear Fit')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\tau$ (ms)', fontsize=16)
        plt.ylabel('MSD (gridpoint$^2$)', fontsize=16)
        plt.text(
            0.2, 0.8,
            rf'$\tau^{{{slope_str}}}$',
            transform=plt.gca().transAxes,
            fontsize=14,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
        plt.savefig(f'{MSD_dir}/MSD_{common_path}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # unwrap periodic trajection
    centre = data_load.a1.ge.centre_mass.centre
    continous_centre = mya.unwrap_periodic_path(centre=centre, width=63)
    continous_jump_size = np.diff(continous_centre)
    continous_jump_dist = np.sqrt(np.sum(continous_jump_size**2,1))
    jump_x = data_load.a1.ge.centre_mass.jump_size[:,0]
    # pdx
    if not os.path.exists(f'{pdx_dir}/pdx_{common_path}.png') or 1:
        plt.figure(figsize=(6, 6))
        bins = np.arange(-41, 41 + 2, 2)
        plt.hist(jump_x, 
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
        params, nll = fit_levy(jump_x)
        alpha, beta, mu, sigma = params.get()
        x = np.linspace(jump_x.min(), jump_x.max(), 200)
        pdf_fit = levy(x, alpha, beta, mu, sigma)
        plt.plot(x, pdf_fit, 'r-', label='Levy fit')
        plt.xlabel(r'$\Delta$ x (gridpoint)', fontsize=16)
        plt.ylabel('Probability density', fontsize=16)
        plt.text(
            0.95, 0.95,
            rf'$\alpha: {alpha:.2f}\\ \beta:{beta:.2f}\\ \mu: {mu:.2f}\\ \sigma: {sigma:.2f}$',
            transform=plt.gca().transAxes,
            fontsize=14,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
        plt.xlim(-45, 45)
        plt.legend()
        plt.savefig(f'{pdx_dir}/pdx_{common_path}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # combined graph
    if not os.path.exists(f'{combined_dir}/{common_path}.png') or 1:
        import matplotlib.image as mpimg
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        # MSD
        img_msd = mpimg.imread(f'{MSD_dir}/MSD_{common_path}.png')
        ax[0].imshow(img_msd)
        ax[0].axis('off')
        # pdx
        img_pdx = mpimg.imread(f'{pdx_dir}/pdx_{common_path}.png')
        ax[1].imshow(img_pdx)
        ax[1].axis('off')
        plt.tight_layout()
        plt.savefig(f'{combined_dir}/{common_path}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # if not os.path.exists(f'./{trajectory_dir}/trajectory_{common_path}.png'):
    #     # trajectory
    #     _ = mya.plot_trajectory(
    #         data=continous_centre,
    #         title=f'Centre trajectory of \n {common_title}',
    #         save_path=f'./{trajectory_dir}/trajectory_{common_path}.png'
    #         )

    if not os.path.exists(f'./{video_dir}/{common_path}_pattern.mp4') or 0:
        # Animation
        title = f'Animation \n {common_title}'
        ani = fra.show_pattern(spkrate1=data_load.a1.ge.spk_rate.spk_rate,
                            frames = frames,
                            start_time = start_time,
                            interval_movie=15,
                            anititle=title,
                            stim=None, 
                            adpt=None)
        ani.save(f'./{video_dir}/{common_path}_pattern.mp4',writer='ffmpeg',fps=60,dpi=100)

    # release RAM
    plt.close('all')
    gc.collect()

    # record parameters & states
    return None

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

# appoint looping parameters
params_loop = {
    'ie_r_e1': np.linspace(1.8, 1.8, 1),
    'ie_r_i1': np.linspace(2.4, 2.4, 1)
}

# generate looping parameter combinations
loop_combinations = list(itertools.product(*params_loop.values()))
# get total looping number
loop_total = len(loop_combinations)

Parallel(n_jobs=-1)(delayed(compute_save_draw)(comb, i+1)
                    for i, comb in enumerate(loop_combinations))


print(f'total time elapsed: {np.round((time.perf_counter() - start)/60,2)} min')