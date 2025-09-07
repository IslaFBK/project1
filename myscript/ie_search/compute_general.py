import brian2.numpy_ as np
from brian2.only import *
import pickle
import os
import datetime
import time
import connection as cn
from connection import poisson_stimuli as psti
from connection import pre_process_sc
from connection import preprocess_2area
from connection import build_one_area
from connection import build_two_areas
from connection import get_stim_scale
from analysis import mydata
from analysis import firing_rate_analysis as fra
from pathlib import Path

import logging
logging.getLogger('brian2').setLevel(logging.WARNING)

#%% OS operation
# test if data_dir exists, if not, create one.
# FileExistsError means if menu is create by other progress or thread, ignore it.
root_dir = 'ie_ratio_2/'
Path(root_dir).mkdir(parents=True, exist_ok=True)
data_dir = f'{root_dir}/raw_data/'
Path(data_dir).mkdir(parents=True, exist_ok=True)
graph_dir = f'{root_dir}/graph/'
Path(graph_dir).mkdir(parents=True, exist_ok=True)
video_dir = f'{root_dir}/vedio/'
Path(video_dir).mkdir(parents=True, exist_ok=True)
state_dir = f'{root_dir}/state/'
Path(state_dir).mkdir(parents=True, exist_ok=True)
MSD_dir = f'./{graph_dir}/MSD/'
Path(MSD_dir).mkdir(parents=True, exist_ok=True)
pdx_dir = f'./{graph_dir}/pdx/'
Path(pdx_dir).mkdir(parents=True, exist_ok=True)
combined_dir = f'./{graph_dir}/combined'
Path(combined_dir).mkdir(parents=True, exist_ok=True)

def compute_1(comb, seed=10, index=1, sti=False, maxrate=2000, sig=2, sti_type='Gaussian', video=False, save_load=False):
    ie_r_e1, ie_r_i1 = comb

    common_title = rf'$\zeta^{{E}}$: {ie_r_e1:.4f}, $\zeta^{{I}}$: {ie_r_i1:.4f}'
    common_path = f're{ie_r_e1:.4f}_ri{ie_r_i1:.4f}'

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
    w_extnl_ = 10 # nS
    #%% adjustable parameters
    def find_w_e_(w_i_,num_e_,num_i_,ie_ratio_e):
        return w_i_/(num_e_/num_i_*ie_ratio_e)
    def find_w_i_(w_e_,num_e_,num_i_,ie_ratio_i):
        return w_e_*(num_e_/num_i_*ie_ratio_i)
    # in-degree
    num_ee = 270
    num_ei = 350
    num_ie = 130
    num_ii = 180
    # mean synaptic weight
    w_ee_1 = 11
    w_ii_1 = 50
    w_ei_1 = find_w_e_(w_ii_1, num_ei, num_ii, ie_r_e1)
    w_ie_1 = find_w_i_(w_ee_1, num_ee, num_ie, ie_r_i1)

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
        # LFP_elec = np.array([[0,0]])
        i_LFP,j_LFP,w_LFP = get_LFP.get_LFP(ijwd1.e_lattice,LFP_elec,
                                            width=ijwd1.width,
                                            LFP_sigma=8,LFP_effect_range=2.5)
        group_LFP_record = NeuronGroup(len(LFP_elec),
                                    model=get_LFP.LFP_recordneuron)
        syn_LFP = Synapses(group_e_1,group_LFP_record,model=get_LFP.LFP_syn)
        syn_LFP.connect(i=i_LFP,j=j_LFP)
        syn_LFP.w[:] = w_LFP[:]
                                                
    #%%
    # --- 刺激参数设置与时序生成 ---
    stim_dura = 1000      # 每次刺激持续时间（ms）
    transient = 3000      # 仿真初始预热期（ms），用于网络稳定
    inter_time = 2000     # 两次刺激间隔（ms）

    stim_scale_cls = get_stim_scale.get_stim_scale()  # 自定义类，统一管理刺激参数和时序
    stim_scale_cls.seed = seed                        # 随机种子，保证可复现
    n_StimAmp = 1         # 刺激强度组数
    n_perStimAmp = 1      # 每组强度的重复次数
    stim_amp_scale = np.ones(n_StimAmp*n_perStimAmp)  # 初始化刺激强度数组
    for i in range(n_StimAmp):
        stim_amp_scale[i*n_perStimAmp:i*n_perStimAmp+n_perStimAmp] = 2**(i)  # 生成不同强度

    stim_scale_cls.stim_amp_scale = stim_amp_scale
    stim_scale_cls.stim_dura = stim_dura
    stim_scale_cls.separate_dura = np.array([300,600])  # 刺激间隔期（ms），可自定义
    stim_scale_cls.get_scale()                          # 自动生成完整刺激时序
    stim_scale_cls.n_StimAmp = n_StimAmp
    stim_scale_cls.n_perStimAmp = n_perStimAmp

    # 拼接预热期和刺激时序，保证与仿真时间轴对齐
    init = np.zeros(transient//stim_scale_cls.dt_stim)
    stim_scale_cls.scale_stim = np.concatenate((init,stim_scale_cls.scale_stim))
    stim_scale_cls.stim_on += transient

    # Brian2专用时序数组
    scale_1 = TimedArray(stim_scale_cls.scale_stim, dt=10*ms)
    data_ = mydata.mydata()
    param_a1 = {**param_a1, 'stim1':data_.class2dict(stim_scale_cls)}  # 保存刺激参数

    #%% Background
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

    #%% Stimulus (Gaussian/Uniform)
    if sti == True:
        posi_stim_e1 = NeuronGroup(ijwd1.Ne, \
                                '''rates =  bkg_rates + stim_1*scale_1(t) : Hz
                                bkg_rates : Hz
                                stim_1 : Hz
                                ''', threshold='rand()<rates*dt')

        posi_stim_e1.bkg_rates = 0*Hz
        posi_stim_e1.stim_1 = psti.input_spkrate(maxrate = [maxrate], sig=[sig], position=[[0, 0]], sti_type=sti_type)*Hz
        #posi_stim_e1.stim_2 = psti.input_spkrate(maxrate = [200], sig=[6], position=[[-32, -32]])*Hz

        synapse_e_extnl = cn.model_neu_syn_AD.synapse_e_AD
        syn_extnl_e1 = Synapses(posi_stim_e1, group_e_1, 
                                model=synapse_e_extnl, on_pre='x_E_extnl_post += w')
        syn_extnl_e1.connect('i==j')
        syn_extnl_e1.w = w_extnl_*nS#*tau_s_de_*nS

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

    simu_time_tot = (stim_scale_cls.stim_on[-1,1] + 15)*ms
    simu_time1 = (stim_scale_cls.stim_on[n_StimAmp*n_perStimAmp-1,1] + round(inter_time/2))*ms
    simu_time2 = simu_time_tot - simu_time1

    net.run(simu_time1, profile=False) #,namespace={'tau_k': 80*ms}
    net.run(simu_time2, profile=False) #,namespace={'tau_k': 80*ms}

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
            'data_dir': os.getcwd(),
            'param':param_all,
            'a1':{'param':param_a1,
                'ge':{'i':spk_e_1.i[:],'t':spk_tstep_e1},    
                'gi':{'i':spk_i_1.i[:],'t':spk_tstep_i1}}}
    if record_LFP:
        data['a1']['ge']['LFP'] = lfp_moni.lfp[:]/nA

    if save_load:
        # save and load
        ''' save data to disk'''
        with open(f"{data_dir}data_{index}.file", 'wb') as file:
            pickle.dump(data, file)
        '''load data from disk'''
        data_load = mydata.mydata()
        data_load.load(f"{data_dir}data_{index}.file")
    else:
        # directly use mydata module
        data_load = mydata.mydata(data)

    #%% analysis
    start_time = transient  #data.a1.param.stim1.stim_on[first_stim,0] - 300
    end_time = int(round(simu_time_tot/ms))   #data.a1.param.stim1.stim_on[last_stim,0] + 1500
    window = 15
    data_load.a1.ge.get_spike_rate(start_time=start_time,
                                   end_time=end_time,
                                   sample_interval=1,
                                   n_neuron = data_load.a1.param.Ne,
                                   window = window)
    spk_rate = data_load.a1.ge.spk_rate.spk_rate
    frames = data_load.a1.ge.spk_rate.spk_rate.shape[2]

    data_load.a1.ge.get_centre_mass()
    centre = data_load.a1.ge.centre_mass.centre

    data_load.a1.ge.overlap_centreandspike()
    
    stim_on_off = data_load.a1.param.stim1.stim_on-start_time
    stim_on_off = stim_on_off[stim_on_off[:,0]>=0]

    stim = None
    if sti:
        stim = [[[[31.5,31.5]], 
                [stim_on_off], 
                [[sig]*stim_on_off.shape[0]]]]
        
    jump_interval = np.linspace(1, 1000, 100)
    data_load.a1.ge.get_MSD(start_time=start_time,
                            end_time=end_time,
                            sample_interval=1,
                            n_neuron = data_load.a1.param.Ne,
                            window = window,
                            dt = 0.1,
                            slide_interval=1,
                            jump_interval=jump_interval,
                            fit_stableDist='pylevy')
    msd = data_load.a1.ge.MSD.MSD
    jump_interval = data_load.a1.ge.MSD.jump_interval

    pdx = data_load.a1.ge.centre_mass.jump_size[:,1]

    if video:
        # Animation
        title = f'Animation \n {common_title}'
        ani = fra.show_pattern(spkrate1=data_load.a1.ge.spk_rate.spk_rate,
                               frames = frames,
                               start_time = start_time,
                               interval_movie=15,
                               anititle=title,
                               stim=stim,
                               adpt=None)
        ani.save(f'./{video_dir}/1area_{index}_{common_path}_{sig}_pattern.mp4',writer='ffmpeg',fps=60,dpi=100)
    return {
        'data': data_load,
        'msd': msd,
        'jump_interval': jump_interval,
        'pdx': pdx,
        'spk_rate': spk_rate,
        'centre': centre
    }

def compute_2(comb, seed=10, index=1, sti=False, maxrate=2000, sig=2, sti_type='Gaussian', video=False, save_load=False):
    ie_r_e1, ie_r_i1, ie_r_e2, ie_r_i2 = comb

    # common title & path
    common_title = (rf'$\zeta^{{E1}}$: {ie_r_e1:.4f}, '
                    rf'$\zeta^{{I1}}$: {ie_r_i1:.4f}, '
                    rf'$\zeta^{{E2}}$: {ie_r_e2:.4f}, '
                    rf'$\zeta^{{I2}}$: {ie_r_i2:.4f}')
    common_path = f're1{ie_r_e1:.4f}_ri1{ie_r_i1:.4f}_re2{ie_r_e2:.4f}_ri2{ie_r_i2:.4f}'

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
    w_extnl_ = 10 # nS
    #%% adjustable parameters
    def find_w_e_(w_i_,num_e_,num_i_,ie_ratio_e):
        return w_i_/(num_e_/num_i_*ie_ratio_e)
    def find_w_i_(w_e_,num_e_,num_i_,ie_ratio_i):
        return w_e_*(num_e_/num_i_*ie_ratio_i)
    # in-degree
    num_ee = 270
    num_ei = 350
    num_ie = 130
    num_ii = 180
    # mean synaptic weight 1
    w_ee_1 = 11
    w_ii_1 = 50
    # mean synaptic weight 2
    w_ee_2 = 11
    w_ii_2 = 50

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
    ijwd1.w_ei_mean = find_w_e_(w_ii_1, num_ei, num_ii, ie_r_e1)
    ijwd1.w_ie_mean = find_w_i_(w_ee_1, num_ee, num_ie, ie_r_i1)
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
    ijwd2.w_ei_mean = find_w_e_(w_ii_2, num_ei, num_ii, ie_r_e2)
    ijwd2.w_ie_mean = find_w_i_(w_ee_2, num_ee, num_ie, ie_r_i2)
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
    stim_scale_cls.seed = seed # random seed
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

    #%% Stimulus (Gaussian/Uniform)
    if sti == True:
        posi_stim_e1 = NeuronGroup(ijwd1.Ne, \
                                '''rates =  bkg_rates + stim_1*scale_1(t) : Hz
                                bkg_rates : Hz
                                stim_1 : Hz
                                ''', threshold='rand()<rates*dt')

        posi_stim_e1.bkg_rates = 0*Hz
        posi_stim_e1.stim_1 = psti.input_spkrate(maxrate = [maxrate], sig=[sig], position=[[0, 0]], sti_type=sti_type)*Hz
        #posi_stim_e1.stim_2 = psti.input_spkrate(maxrate = [200], sig=[6], position=[[-32, -32]])*Hz

        synapse_e_extnl = cn.model_neu_syn_AD.synapse_e_AD
        syn_extnl_e1 = Synapses(posi_stim_e1, group_e_1, 
                                model=synapse_e_extnl, on_pre='x_E_extnl_post += w')
        syn_extnl_e1.connect('i==j')
        syn_extnl_e1.w = w_extnl_*nS#*tau_s_de_*nS

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

    simu_time_tot = (stim_scale_cls.stim_on[-1,1] + 15)*ms
    simu_time1 = (stim_scale_cls.stim_on[n_StimAmp*n_perStimAmp-1,1] + round(inter_time/2))*ms
    simu_time2 = simu_time_tot - simu_time1

    net.run(simu_time1, profile=False) #,namespace={'tau_k': 80*ms}
    net.run(simu_time2, profile=False) #,namespace={'tau_k': 80*ms}

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

    if save_load:
        # save and load
        ''' save data to disk'''
        with open(f"{data_dir}data_{index}.file", 'wb') as file:
            pickle.dump(data, file)
        '''load data from disk'''
        data_load = mydata.mydata()
        data_load.load(f"{data_dir}data_{index}.file")
    else:
        # directly use mydata module
        data_load = mydata.mydata(data)

    #%% analysis
    start_time = transient  #data.a1.param.stim1.stim_on[first_stim,0] - 300
    end_time = int(round(simu_time_tot/ms))   #data.a1.param.stim1.stim_on[last_stim,0] + 1500
    window = 15
    # area 1
    data_load.a1.ge.get_spike_rate(start_time=start_time,
                                   end_time=end_time,
                                   sample_interval=1,
                                   n_neuron = data_load.a1.param.Ne,
                                   window = window)
    spk_rate1 = data_load.a1.ge.spk_rate.spk_rate

    data_load.a1.ge.get_centre_mass()
    centre1 = data_load.a1.ge.centre_mass.centre

    data_load.a1.ge.overlap_centreandspike()

    jump_interval = np.linspace(1, 1000, 100)
    data_load.a1.ge.get_MSD(start_time=start_time,
                            end_time=end_time,
                            sample_interval=1,
                            n_neuron = data_load.a1.param.Ne,
                            window = window,
                            dt = 0.1,
                            slide_interval=1,
                            jump_interval=jump_interval,
                            fit_stableDist='pylevy')
    msd1 = data_load.a1.ge.MSD.MSD
    jump_interval1 = data_load.a1.ge.MSD.jump_interval

    pdx1 = data_load.a1.ge.centre_mass.jump_size[:,1]

    # area 2
    data_load.a2.ge.get_spike_rate(start_time=start_time,
                                   end_time=end_time,
                                   sample_interval=1,
                                   n_neuron = data_load.a2.param.Ne,
                                   window = window)
    spk_rate2 = data_load.a2.ge.spk_rate.spk_rate

    data_load.a2.ge.get_centre_mass()
    centre2 = data_load.a2.ge.centre_mass.centre

    data_load.a2.ge.overlap_centreandspike()

    jump_interval = np.linspace(1, 1000, 100)
    data_load.a2.ge.get_MSD(start_time=start_time,
                            end_time=end_time,
                            sample_interval=1,
                            n_neuron = data_load.a2.param.Ne,
                            window = window,
                            dt = 0.1,
                            slide_interval=1,
                            jump_interval=jump_interval,
                            fit_stableDist='pylevy')
    msd2 = data_load.a2.ge.MSD.MSD
    jump_interval2 = data_load.a2.ge.MSD.jump_interval

    pdx2 = data_load.a2.ge.centre_mass.jump_size[:,1]
    
    # video prepare
    frames = data_load.a1.ge.spk_rate.spk_rate.shape[2]
    stim_on_off = data_load.a1.param.stim1.stim_on-start_time
    stim_on_off = stim_on_off[stim_on_off[:,0]>=0]

    stim = None
    if sti:
        stim = [[[[31.5,31.5]], 
                [stim_on_off], 
                [[sig]*stim_on_off.shape[0]]],None]
        
    if video:
        # Animation
        title = f'Animation \n {common_title}'
        ani = fra.show_pattern(spkrate1=data_load.a1.ge.spk_rate.spk_rate,
                               spkrate2=data_load.a2.ge.spk_rate.spk_rate,
                               frames = frames,
                               start_time = start_time,
                               interval_movie=15,
                               anititle=title,
                               stim=stim,
                               adpt=None)
        ani.save(f'./{video_dir}/2area_{index}_{common_path}_{sig}_pattern.mp4',writer='ffmpeg',fps=60,dpi=100)
    return {
        'data': data_load,
        'msd1': msd1,
        'msd2': msd2,
        'jump_interval1': jump_interval1,
        'jump_interval2': jump_interval2,
        'pdx1': pdx1,
        'pdx2': pdx2,
        'spk_rate1': spk_rate1,
        'spk_rate2': spk_rate2,
        'centre1': centre1,
        'centre2': centre2
    }
