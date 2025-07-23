# University of Sydney
# Shuzheng Huang
# Creating time 27/05/2022 9:37 pm

##
import brian2.numpy_ as np
import connection as cn
from brian2.only import *
import time
import mydata
import os
import datetime
import poisson_stimulus_SC as psti
import naturalScene as NS
import cla_sharp as CSS
import pre_process_sc
import preprocess_2area
import build_two_areas
import get_stim_scale
import adapt_gaussian
import sys
import inverse_stimulus_sharp as ISS
import change_adaptation as chanAdapt

#%%
def calResponse(pres_T=[], trans_T=[], inter_T=[], seed_num=[], stimulus_type=[], stiSize=[], stiPosition=[], stiMaxRate=[],
                gaussian=[], stimulus_type_top = ['spontaneous'], stiSize_top = [], stiPosition_top = [], stiMaxRate_top = [],
                gaussian_top = [], areaConnection = True, stimPresTimes=1, netWorkSize=64, radius_max=None, imagePath=None,
                chanBotAdapt=False):

    #%%
    prefs.codegen.target = 'cython'

    dir_cache = '../cache'
    prefs.codegen.runtime.cython.cache_dir = dir_cache
    prefs.codegen.max_cache_dir_size = 120

    #%%
    data_dir = 'raw_data/'
    if not os.path.exists(data_dir):
        try:
            os.makedirs(data_dir)
        except FileExistsError:
            pass

    #%%
    sys_argv = int(sys.argv[1])
    record_LFP = True

    #%%
    loop_num = -1
    repeat = 20
    tau_k_ = 60  # ms
    # chg_adapt_range = 7
    w_extnl_ = 5  # nS
    tau_s_r_ = 1  # ms

    ei_sc = 1
    ie_r_i1 = 0.777076 / ei_sc
    ie_r_i2 = 0.6784 / ei_sc

    tau_s_di_ = 4.5
    tau_s_de_ = 5

    delta_gk_1 = 1.9
    new_delta_gk_1 = 1
    stim_dura = pres_T[0]
    t_ref = 4

    adapt_change_shape = 'gaus'  # logi, gaus
    chg_adapt_sharpness = 2.2

    stim2 = 0  # True

    pois_bckgrdExt = 1
    I_extnl_ = 0

    peak_p_e1_e2 = 0.4
    peak_p_e1_i2 = 0.4
    peak_p_e2_e1 = 0.4
    peak_p_e2_i1 = 0.4

    for pois_extnl_r_1 in [8, ]:  # np.arange(8, 8.1, 1):
        for pois_extnl_r_2 in [8, ]:  # np.arange(8, 8.1, 1):
            for delta_gk_2 in 6.5 * np.array([1]):  # np.array([1, 1.05, 1.1]):
                for new_delta_gk_2 in 0.5 * np.array([1]):  # np.array([1.1, 1.2]):
                    for ie_r_e2 in 0.7475 * np.array([
                        0.99, ]):  # np.arange(0.99, 0.991,0.01)/ei_sc:#1.03*np.arange(1.02, 1.021, 0.02):#1.43715*np.arange(1.0,1.01,0.02):#[1.12125]:#[1.41375]:#1.3728*np.arange(1.05,1.051,0.02):#np.arange(1.68, 2.281, 0.03)*0.55*7/14*2.6:#[1.55]:#np.arange(1.51, 1.551, 0.02): #[1.51]:#[2.06]: #np.arange(2.04, 2.121, 0.02): # def:2.06
                        for ie_r_e1 in 0.88 * np.arange(1.00, 1.001,
                                                        0.015) / ei_sc:  # 1.015*1.4*np.arange(1.00, 1.001, 0.02):#1.43715*1.5**np.arange(0.98,1.021, 0.04):#1.21875*np.arange(1.02,1.041, 0.02):#1.486875*np.arange(1.02,1.041, 0.02):#2.0592*1.06*np.arange(0.98,0.99,0.02): #np.arange(1.02, 1.11,0.02): #[2.13]:#np.arange(2.12, 2.141, 0.01): #[2.15]: #[2.88]: #np.arange(2.86, 2.941, 0.02): #def:2.88 #np.arange(2.86, 2.901, 0.02): #2.64
                            for tau_p_d_e1_e2 in [8, ]:
                                for tau_p_d_e1_i2 in [tau_p_d_e1_e2]:
                                    for tau_p_d_e2_e1 in [8, ]:  # np.arange(7,14.1,1):
                                        for tau_p_d_e2_i1 in [tau_p_d_e2_e1]:  # [15]:#np.arange(7,14.1,1):#[10]:
                                            for scale_w_12_e in np.arange(2.6, 2.61, 0.3) * 0.5 / ((
                                                                                                           tau_p_d_e1_e2 / 6) ** 2):  # np.arange(1.4, 1.81, 0.2):#np.arange(0.8,1.21,0.05):[2.5]
                                                for scale_w_12_i in scale_w_12_e * np.arange(1.0, 1.01,
                                                                                             0.1):  # [scale_w_12_e*4/5] scale_w_12_e*np.array([1]):#np.array([0.8,0.9,1.0,1.1]):#np.arange(0.9,1.11,0.1):#np.arange(0.8,1.21,0.05):#[scale_w_12_e]:#0.2*np.arange(0.8,1.21,0.05):
                                                    for scale_w_21_e in np.arange(0.35, 0.351, 0.03) / 4 * 0.5 / ((
                                                                                                                          tau_p_d_e2_e1 / 13) ** 2):  # np.arange(0.8,1.21,0.05):#[1]:#np.arange(0.8,1.21,0.05):[0.3]
                                                        for scale_w_21_i in scale_w_21_e * np.arange(1.0, 1.01,
                                                                                                     0.1):  # [scale_w_21_e*4/5] *np.arange(0.5, 0.91, 0.2):#scale_w_21_e*np.array([1]):#np.arange(0.9,1.11,0.1):#np.arange(0.8,1.21,0.05):#[1]:#[scale_w_21_e]:#0.35*np.arange(0.8,1.21,0.05):

                                                            # for peak_p_e1_e2 in [0.4]:#np.arange(0.3,0.451,0.05):
                                                            #     for peak_p_e1_i2 in [0.4]:#np.arange(0.25,0.401,0.05):
                                                            #         for peak_p_e2_e1 in [0.4]:#np.arange(0.08, 0.121, 0.01):#[0.13]:#[0.1,0.12,0.14]:#np.arange(0.10):#[0.15]: # 0.18
                                                            #             for peak_p_e2_i1 in [0.4]:#np.arange(0.08, 0.121, 0.01):#[0.1]:#peak_p_e2_e1*np.repeat(np.arange(0.9,1.1,0.03),repeat):#:[0.15]*repeat: # 0.18
                                                            # for stim_dura in [600]:
                                                            for stim_amp in np.array([200, 400, 600, 800]):  # , 400, 800])
                                                                for chg_adapt_range in [8.2, ]:  # 8.2 7.5
                                                                    for rp in [None] * repeat:
                                                                        # for decay_p_ie_p_ii in [20]:
                                                                        # for ie_ratio_ in 3.375*np.arange(0.94, 1.21, 0.02):#(np.arange(0.7,1.56,0.05)-0.02):#np.linspace(2.4, 4.5, 20):
                                                                        loop_num += 1
                                                                        if loop_num == sys_argv:
                                                                            print('loop_num:', loop_num)
                                                                            break
                                                                        else:
                                                                            continue
                                                                        break
                                                                    else:
                                                                        continue
                                                                    break
                                                                else:
                                                                    continue
                                                                break
                                                            else:
                                                                continue
                                                            break
                                                        else:
                                                            continue
                                                        break
                                                    else:
                                                        continue
                                                    break
                                                else:
                                                    continue
                                                break
                                            else:
                                                continue
                                            break
                                        else:
                                            continue
                                        break
                                    else:
                                        continue
                                    break
                                else:
                                    continue
                                break
                            else:
                                continue
                            break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                continue
            break
        else:
            continue
        break

    if not pois_bckgrdExt:
        I_extnl_crt2e = I_extnl_  # 0.51 0.40
        I_extnl_crt2i = I_extnl_  # 0.51 0.40

    if loop_num != sys_argv: sys.exit("Error: wrong PBS_array_id")

    #%%
    if len(seed_num):
        seed(seed_num[0])
    else:
        seed_num = None

    #%%
    if len(sys.argv) >= 3:
        sys_argv2 = int(sys.argv[2])
        loop_num2 = -1
        for num_ee1 in [275]:
            for num_ei1 in [200]:
                for num_ie1 in [115]:
                    for num_ii1 in [95]:
                        for num_ee2 in [320]:
                            for num_ei2 in [415]:
                                for num_ie2 in [130]:
                                    for num_ii2 in [180]:
                                        for peak_p_e1_e2 in [0.3]:
                                            for tau_p_d_e1_e2 in [8]:
                                                for peak_p_e1_i2 in [0.3]:
                                                    for tau_p_d_e1_i2 in [8]:
                                                        for peak_p_e2_e1 in [0.2]:
                                                            for tau_p_d_e2_e1 in [8]:
                                                                for peak_p_e2_i1 in [0.5]:
                                                                    for tau_p_d_e2_i1 in [6]:
                                                                        loop_num2 += 1
                                                                        if loop_num2 == sys_argv2:
                                                                            print('loop_num2:', loop_num2)
                                                                            break
                                                                        else:
                                                                            continue
                                                                        break
                                                                    else:
                                                                        continue
                                                                    break
                                                                else:
                                                                    continue
                                                                break
                                                            else:
                                                                continue
                                                            break
                                                        else:
                                                            continue
                                                        break
                                                    else:
                                                        continue
                                                    break
                                                else:
                                                    continue
                                                break
                                            else:
                                                continue
                                            break
                                        else:
                                            continue
                                        break
                                    else:
                                        continue
                                    break
                                else:
                                    continue
                                break
                            else:
                                continue
                            break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                continue
            break

    if len(sys.argv) >= 3:
        if loop_num2 != sys_argv2: sys.exit("Error: wrong PBS_array_id2")
    #else:
        #num_ee1 = 100
        #num_ei1 = 130
        #num_ie1 = 75
        #num_ii1 = 90
        #num_ee2 = 320
        #num_ei2 = 415
        #num_ie2 = 130
        #num_ii2 = 180
        #peak_p_e1_e2 = 0.35
        #tau_p_d_e1_e2 = 6
        #peak_p_e1_i2 = 0.5
        #tau_p_d_e1_i2 = 4
        #peak_p_e2_e1 = 0.3
        #tau_p_d_e2_e1 = 8
        #peak_p_e2_i1 = 0.2
        #tau_p_d_e2_i1 = 12

    #%%
    ie_r_e1 *= (27 / 13) * (num_ie1 / num_ee1)
    ie_r_i1 *= (35 / 18) * (num_ii1 / num_ei1)

    #%%
    def find_w_e(w_i, num_i, num_e, ie_ratio):
        return (w_i * num_i) / num_e / ie_ratio

    def find_w_i(w_e, num_e, num_i, ie_ratio):
        return (w_e * num_e) * ie_ratio / num_i

    #%%
    w_ee_1 = 7.857
    w_ei_1 = 10.847

    ie_r_e = 2.76 * 6.5 / 5.8
    ie_r_i = 2.450 * 6.5 / 5.8

    ijwd1 = pre_process_sc.get_ijwd()
    ijwd1.Ne = netWorkSize * netWorkSize
    ijwd1.Ni = int(netWorkSize/2) * int(netWorkSize/2)
    ijwd1.width = netWorkSize

    ijwd1.decay_p_ee = 7
    ijwd1.decay_p_ei = 8.5
    ijwd1.decay_p_ie = 15
    ijwd1.decay_p_ii = 15
    ijwd1.delay = [0.5, 2.5]

    num_ee = num_ee1
    num_ei = num_ei1
    num_ie = num_ie1
    num_ii = num_ii1

    ijwd1.mean_SynNumIn_ee = num_ee
    ijwd1.mean_SynNumIn_ei = num_ei
    ijwd1.mean_SynNumIn_ie = num_ie
    ijwd1.mean_SynNumIn_ii = num_ii

    ijwd1.w_ee_mean = w_ee_1
    ijwd1.w_ei_mean = w_ei_1
    ijwd1.w_ie_mean = find_w_i(w_ee_1, num_ee, num_ie, ie_r_e * ie_r_e1)
    ijwd1.w_ii_mean = find_w_i(w_ei_1, num_ei, num_ii, ie_r_i * ie_r_i1)
    print('bottom area: ee:%.3f, ei:%.3f, ie:%.3f, ii:%.3f' % (ijwd1.w_ee_mean, ijwd1.w_ei_mean,
                                                               ijwd1.w_ie_mean, ijwd1.w_ii_mean))

    #%%
    ijwd1.generate_ijw()
    ijwd1.generate_d_rand_lowerHigherBound()
    param_a1 = {**ijwd1.__dict__}

    del param_a1['i_ee'], param_a1['j_ee'], param_a1['w_ee'], param_a1['d_ee'], param_a1['dist_ee']
    del param_a1['i_ei'], param_a1['j_ei'], param_a1['w_ei'], param_a1['d_ei'], param_a1['dist_ei']
    del param_a1['i_ie'], param_a1['j_ie'], param_a1['w_ie'], param_a1['d_ie'], param_a1['dist_ie']
    del param_a1['i_ii'], param_a1['j_ii'], param_a1['w_ii'], param_a1['d_ii'], param_a1['dist_ii']

    #%%
    ie_r_e2 *= (27 / 13) * (num_ie2 / num_ee2)
    ie_r_i2 *= (35 / 18) * (num_ii2 / num_ei2)

    #%%
    w_ee_2 = 11
    w_ei_2 = 13.805

    ie_r_e = 2.76 * 6.5 / 5.8
    ie_r_i = 2.450 * 6.5 / 5.8

    ijwd2 = pre_process_sc.get_ijwd()
    ijwd2.Ne = netWorkSize * netWorkSize
    ijwd2.Ni = int(netWorkSize/2) * int(netWorkSize/2)
    ijwd2.width = netWorkSize

    ijwd2.decay_p_ee = 9.5
    ijwd2.decay_p_ei = 13
    ijwd2.decay_p_ie = 19
    ijwd2.decay_p_ii = 19
    ijwd2.delay = [0.5, 2.5]

    num_ee = num_ee2
    num_ei = num_ei2
    num_ie = num_ie2
    num_ii = num_ii2

    ijwd2.mean_SynNumIn_ee = num_ee
    ijwd2.mean_SynNumIn_ei = num_ei
    ijwd2.mean_SynNumIn_ie = num_ie
    ijwd2.mean_SynNumIn_ii = num_ii

    ijwd2.w_ee_mean = w_ee_2
    ijwd2.w_ei_mean = w_ei_2
    ijwd2.w_ie_mean = find_w_i(w_ee_2, num_ee, num_ie, ie_r_e * ie_r_e2)
    ijwd2.w_ii_mean = find_w_i(w_ei_2, num_ei, num_ii, ie_r_i * ie_r_i2)
    print('top area: ee:%.3f, ei:%.3f, ie:%.3f, ii:%.3f' % (ijwd2.w_ee_mean, ijwd2.w_ei_mean,
                                                            ijwd2.w_ie_mean, ijwd2.w_ii_mean))

    #%%
    ijwd2.generate_ijw()
    ijwd2.generate_d_rand_lowerHigherBound()

    param_a2 = {**ijwd2.__dict__}

    del param_a2['i_ee'], param_a2['j_ee'], param_a2['w_ee'], param_a2['d_ee'], param_a2['dist_ee']
    del param_a2['i_ei'], param_a2['j_ei'], param_a2['w_ei'], param_a2['d_ei'], param_a2['dist_ei']
    del param_a2['i_ie'], param_a2['j_ie'], param_a2['w_ie'], param_a2['d_ie'], param_a2['dist_ie']
    del param_a2['i_ii'], param_a2['j_ii'], param_a2['w_ii'], param_a2['d_ii'], param_a2['dist_ii']

    #%%
    ijwd_inter = preprocess_2area.get_ijwd_2()

    ijwd_inter.Ne1 = netWorkSize * netWorkSize
    ijwd_inter.Ne2 = netWorkSize * netWorkSize
    ijwd_inter.width1 = netWorkSize
    ijwd_inter.width2 = netWorkSize

    ijwd_inter.p_inter_area_1 = 1 / 2;
    ijwd_inter.p_inter_area_2 = 1 / 2
    ijwd_inter.section_width_1 = 4;
    ijwd_inter.section_width_2 = 4

    ijwd_inter.peak_p_e1_e2 = peak_p_e1_e2
    ijwd_inter.tau_p_d_e1_e2 = tau_p_d_e1_e2
    ijwd_inter.peak_p_e1_i2 = peak_p_e1_i2
    ijwd_inter.tau_p_d_e1_i2 = tau_p_d_e1_i2
    ijwd_inter.peak_p_e2_e1 = peak_p_e2_e1
    ijwd_inter.tau_p_d_e2_e1 = tau_p_d_e2_e1
    ijwd_inter.peak_p_e2_i1 = peak_p_e2_i1
    ijwd_inter.tau_p_d_e2_i1 = tau_p_d_e2_i1

    if areaConnection:
        ijwd_inter.w_e1_e2_mean = 5  # 5*scale_w_12_e
        ijwd_inter.w_e1_i2_mean = 5  # 5*scale_w_12_i
        ijwd_inter.w_e2_e1_mean = 5  # 5*scale_w_21_e
        ijwd_inter.w_e2_i1_mean = 5  # 5*scale_w_21_i
    else:
        ijwd_inter.w_e1_e2_mean = 0  # 5*scale_w_12_e
        ijwd_inter.w_e1_i2_mean = 0  # 5*scale_w_12_i
        ijwd_inter.w_e2_e1_mean = 0  # 5*scale_w_21_e
        ijwd_inter.w_e2_i1_mean = 0  # 5*scale_w_21_i

    #%%
    ijwd_inter.generate_ijwd()

    param_inter = {**ijwd_inter.__dict__}

    del param_inter['i_e1_e2'], param_inter['j_e1_e2'], param_inter['w_e1_e2'], param_inter['d_e1_e2']
    del param_inter['i_e1_i2'], param_inter['j_e1_i2'], param_inter['w_e1_i2'], param_inter['d_e1_i2']
    del param_inter['i_e2_e1'], param_inter['j_e2_e1'], param_inter['w_e2_e1'], param_inter['d_e2_e1']
    del param_inter['i_e2_i1'], param_inter['j_e2_i1'], param_inter['w_e2_i1'], param_inter['d_e2_i1']

    #%%
    start_scope()

    twoarea_net = build_two_areas.two_areas()

    group_e_1, group_i_1, syn_ee_1, syn_ei_1, syn_ie_1, syn_ii_1, \
    group_e_2, group_i_2, syn_ee_2, syn_ei_2, syn_ie_2, syn_ii_2, \
    syn_e1e2, syn_e1i2, syn_e2e1, syn_e2i1 = twoarea_net.build(ijwd1, ijwd2, ijwd_inter)

    #%%
    if record_LFP:
        import get_LFP

        LFP_elec = np.array([[0, 0], [-32, -32]])
        i_LFP, j_LFP, w_LFP = get_LFP.get_LFP(ijwd2.e_lattice, LFP_elec, width=ijwd2.width, LFP_sigma=7, LFP_effect_range=2.5)

        group_LFP_record_1 = NeuronGroup(len(LFP_elec), model=get_LFP.LFP_recordneuron)
        syn_LFP_1 = Synapses(group_e_1, group_LFP_record_1, model=get_LFP.LFP_syn)
        syn_LFP_1.connect(i=i_LFP, j=j_LFP)
        syn_LFP_1.w[:] = w_LFP[:]

        group_LFP_record_2 = NeuronGroup(len(LFP_elec), model=get_LFP.LFP_recordneuron)
        syn_LFP_2 = Synapses(group_e_2, group_LFP_record_2, model=get_LFP.LFP_syn)
        syn_LFP_2.connect(i=i_LFP, j=j_LFP)
        syn_LFP_2.w[:] = w_LFP[:]

    #%%
    chg_adapt_loca = [0, 0]

    '''gaussian shape'''
    if adapt_change_shape == 'gaus':
        adapt_value_new = adapt_gaussian.get_adaptation(base_amp=delta_gk_2,
                                                        max_decrease=[delta_gk_2 - new_delta_gk_2], sig=[chg_adapt_range],
                                                        position=[chg_adapt_loca], n_side=int(round((ijwd2.Ne) ** 0.5)),
                                                        width=ijwd2.width)

        '''logistic shape'''
    elif adapt_change_shape == 'logi':
        adapt_value_new = adapt_logistic.get_adaptation(base_amp=delta_gk_2, max_decrease=[delta_gk_2 - new_delta_gk_2],
                                                        rang=[chg_adapt_range], sharpness=[chg_adapt_sharpness],
                                                        position=[chg_adapt_loca],
                                                        n_side=round((ijwd2.Ne) ** 0.5), width=ijwd2.width)

    #%%
    stim_scale_cls = get_stim_scale.get_stim_scale()
    stim_scale_cls.seed = 10
    n_perStimAmp = stimPresTimes
    if not isinstance(stim_amp, np.ndarray):
        stim_amp = np.array([stim_amp])
    n_StimAmp = stim_amp.shape[0]
    # stim_dura = 250
    stim_amp_scale = np.ones(n_StimAmp * n_perStimAmp)
    for i in range(n_StimAmp):
        stim_amp_scale[i * n_perStimAmp:i * n_perStimAmp + n_perStimAmp] = stim_amp[i] / 200

    stim_scale_cls.stim_amp_scale = stim_amp_scale
    stim_scale_cls.stim_dura = stim_dura
    stim_scale_cls.separate_dura = np.array([600, 1200])
    stim_scale_cls.get_scale()
    stim_scale_cls.n_StimAmp = n_StimAmp
    stim_scale_cls.n_perStimAmp = n_perStimAmp
    stim_scale_cls.stim_amp = stim_amp

    transient = trans_T[0]
    init = np.zeros(transient // stim_scale_cls.dt_stim)
    stim_scale_cls.scale_stim = np.concatenate((init, stim_scale_cls.scale_stim))
    stim_scale_cls.stim_on += transient

    inter_time = inter_T[0]
    suplmt = (inter_time // stim_scale_cls.dt_stim) - (stim_scale_cls.scale_stim.shape[0] - stim_scale_cls.stim_on[
        -1, 1] // stim_scale_cls.dt_stim)

    #%%
    scale_1 = TimedArray(stim_scale_cls.scale_stim, dt=10 * ms)
    data_ = mydata.mydata()
    param_a1 = {**param_a1, 'stim1': data_.class2dict(stim_scale_cls)}

    #%%
    if stim2:
        scale_2 = TimedArray(stim_scale_cls.scale_stim, dt=10 * ms)
    else:
        scale_2 = TimedArray(np.zeros(stim_scale_cls.scale_stim.shape), dt=10 * ms)

    data_ = mydata.mydata()
    param_a1 = {**param_a1, 'stim2': data_.class2dict(stim_scale_cls)}

    #%%
    posi_stim_e1 = NeuronGroup(ijwd1.Ne, \
                               '''rates =  bkg_rates + stim_1*scale_1(t) + stim_2*scale_2(t) : Hz
                               bkg_rates : Hz
                               stim_1 : Hz
                               stim_2 : Hz
                               ''', threshold='rand()<rates*dt')

    posi_stim_e1.bkg_rates = 0 * Hz

    if stimulus_type[0] == 'poi':
        posi_stim_e1.stim_1 = psti.input_spkrate(maxrate=[stiMaxRate[0]], sig=[stiSize[0]], position=[stiPosition],
                                                 n_side=netWorkSize, width=netWorkSize) * Hz
        posi_stim_e1.stim_2 = 0 * Hz
    elif stimulus_type[0] == 'claSharp':
        posi_stim_e1.stim_1 = CSS.input_claSpkRate_sharp(bgRate=[stiMaxRate[0]], radius=[stiSize[0]],
                                                         position=[stiPosition], gaussian=gaussian,
                                                         n_side=netWorkSize, width=netWorkSize,
                                                         addNoise=0, noiseGroupSize=0,
                                                         noiseProportion=0.2) * Hz
        posi_stim_e1.stim_2 = 0 * Hz
    elif stimulus_type[0] == 'spontaneous':
        posi_stim_e1.stim_1 = 0 * Hz
        posi_stim_e1.stim_2 = 0 * Hz
    elif stimulus_type[0] == 'invSharp':
        posi_stim_e1.stim_1 = ISS.input_invSpkRate_sharp(bgRate=[stiMaxRate[0]], radius=[stiSize[0]],
                                                         position=[stiPosition], gaussian=gaussian,
                                                         n_side=netWorkSize, width=netWorkSize,
                                                         radius_max=radius_max) * Hz
        posi_stim_e1.stim_2 = 0 * Hz
    elif stimulus_type[0] == 'naturalScene':
        posi_stim_e1.stim_1 = NS.input_naturalScene(imagePath=imagePath, strength=[stiMaxRate[0]], radius=[stiSize[0]],
                                                    position=[stiPosition], n_side=netWorkSize, width=netWorkSize) * Hz
        posi_stim_e1.stim_2 = 0 * Hz

    synapse_e_extnl = cn.model_neu_syn_AD.synapse_e_AD
    syn_extnl_e1 = Synapses(posi_stim_e1, group_e_1, model=synapse_e_extnl, on_pre='x_E_extnl_post += w')
    syn_extnl_e1.connect('i==j')
    syn_extnl_e1.w = w_extnl_ * nS

    #%%
    posi_stim_e2 = NeuronGroup(ijwd2.Ne, \
                               '''rates =  bkg_rates + stim_1*scale_1(t) + stim_2*scale_2(t) : Hz
                               bkg_rates : Hz
                               stim_1 : Hz
                               stim_2 : Hz
                               ''', threshold='rand()<rates*dt')

    posi_stim_e2.bkg_rates = 0 * Hz

    if stimulus_type_top[0] == 'poi':
        posi_stim_e2.stim_1 = psti.input_spkrate(maxrate=[stiMaxRate_top[0]], sig=[stiSize_top[0]],
                                                 position=[stiPosition_top], n_side=netWorkSize, width=netWorkSize) * Hz
        posi_stim_e2.stim_2 = 0 * Hz
    elif stimulus_type_top[0] == 'claSharp':
        posi_stim_e2.stim_1 = CSS.input_claSpkRate_sharp(bgRate=[stiMaxRate_top[0]], radius=[stiSize_top[0]],
                                                         position=[stiPosition_top], gaussian=gaussian_top,
                                                         n_side=netWorkSize, width=netWorkSize) * Hz
        posi_stim_e2.stim_2 = 0 * Hz
    elif stimulus_type_top[0] == 'spontaneous':
        posi_stim_e2.stim_1 = 0 * Hz
        posi_stim_e2.stim_2 = 0 * Hz

    synapse_e_extnl = cn.model_neu_syn_AD.synapse_e_AD
    syn_extnl_e2 = Synapses(posi_stim_e2, group_e_2, model=synapse_e_extnl, on_pre='x_E_extnl_post += w')
    syn_extnl_e2.connect('i==j')
    syn_extnl_e2.w = w_extnl_ * nS

    if pois_bckgrdExt:
        pois_bkgExt_e1 = PoissonInput(group_e_1, 'x_E_extnl', 200, pois_extnl_r_1 * 1 * Hz, weight=5 * nS)
        pois_bkgExt_i1 = PoissonInput(group_i_1, 'x_E_extnl', 200, pois_extnl_r_1 * 1.1 * Hz, weight=5 * nS)

        pois_bkgExt_e2 = PoissonInput(group_e_2, 'x_E_extnl', 200, pois_extnl_r_2 * 1 * Hz, weight=5 * nS)
        pois_bkgExt_i2 = PoissonInput(group_i_2, 'x_E_extnl', 200, pois_extnl_r_2 * 1 * Hz, weight=5 * nS)

    #%%
    group_e_1.tau_s_de = tau_s_de_ * ms
    group_e_1.tau_s_di = tau_s_di_ * ms
    group_e_1.tau_s_re = group_e_1.tau_s_ri = tau_s_r_ * ms

    group_e_1.tau_s_de_inter = tau_s_de_ * ms  # 5.0*ms;
    group_e_1.tau_s_re_inter = tau_s_r_ * ms
    group_e_1.tau_s_de_extnl = tau_s_de_ * ms  # 5.0*ms
    group_e_1.tau_s_re_extnl = tau_s_r_ * ms

    group_i_1.tau_s_de = tau_s_de_ * ms
    group_i_1.tau_s_di = tau_s_di_ * ms
    group_i_1.tau_s_re = group_i_1.tau_s_ri = tau_s_r_ * ms

    group_i_1.tau_s_de_inter = tau_s_de_ * ms  # 5.0*ms;
    group_i_1.tau_s_re_inter = tau_s_r_ * ms
    group_i_1.tau_s_de_extnl = tau_s_de_ * ms  # 5.0*ms
    group_i_1.tau_s_re_extnl = tau_s_r_ * ms

    group_e_1.v = np.random.random(ijwd1.Ne) * 35 * mV - 85 * mV
    group_i_1.v = np.random.random(ijwd1.Ni) * 35 * mV - 85 * mV
    #group_e_1.delta_gk = delta_gk_1 * nS

    if not chanBotAdapt:
        print('@@@ the adaptation at the bottom area will NOT be changed @@@')
        group_e_1.delta_gk = delta_gk_1 * nS
    else:
        print('@@@ the adaptation at the bottom area will be changed @@@')
        botAdaptNew = chanAdapt.ChangeAdapt().changeCircle(position=[[0, 0]], radius=[10], adaptBase=delta_gk_1,
                                                           adaptInc=(new_delta_gk_1 - delta_gk_1))
        group_e_1.delta_gk = botAdaptNew * nS

    group_e_1.tau_k = tau_k_ * ms

    if pois_bckgrdExt:
        group_e_1.I_extnl_crt = 0 * nA  # 0.25 0.51*nA 0.35
        group_i_1.I_extnl_crt = 0 * nA  # 0.25 0.60*nA 0.35

    else:
        group_e_1.I_extnl_crt = I_extnl_crt2e * nA  # 0.25 0.51*nA 0.35
        group_i_1.I_extnl_crt = I_extnl_crt2i * nA  # 0.25 0.60*nA 0.35

    group_e_2.tau_s_de = tau_s_de_ * ms
    group_e_2.tau_s_di = tau_s_di_ * ms
    group_e_2.tau_s_re = group_e_2.tau_s_ri = tau_s_r_ * ms

    group_e_2.tau_s_de_inter = tau_s_de_ * ms  # 5.0*ms;
    group_e_2.tau_s_re_inter = tau_s_r_ * ms
    group_e_2.tau_s_de_extnl = tau_s_de_ * ms  # 5.0*ms
    group_e_2.tau_s_re_extnl = tau_s_r_ * ms

    group_i_2.tau_s_de = tau_s_de_ * ms
    group_i_2.tau_s_di = tau_s_di_ * ms
    group_i_2.tau_s_re = group_i_2.tau_s_ri = tau_s_r_ * ms

    group_i_2.tau_s_de_inter = tau_s_de_ * ms  # 5.0*ms;
    group_i_2.tau_s_re_inter = tau_s_r_ * ms
    group_i_2.tau_s_de_extnl = tau_s_de_ * ms  # 5.0*ms
    group_i_2.tau_s_re_extnl = tau_s_r_ * ms

    group_e_2.v = np.random.random(ijwd2.Ne) * 35 * mV - 85 * mV
    group_i_2.v = np.random.random(ijwd2.Ni) * 35 * mV - 85 * mV
    group_e_2.delta_gk = delta_gk_2 * nS
    group_e_2.tau_k = tau_k_ * ms

    if pois_bckgrdExt:
        group_e_2.I_extnl_crt = 0 * nA  # 0.51*nA  0.40 0.35
        group_i_2.I_extnl_crt = 0 * nA  # 0.60*nA  0.40 0.35
    else:
        group_e_2.I_extnl_crt = I_extnl_crt2e * nA  # 0.51*nA  0.40 0.35
        group_i_2.I_extnl_crt = I_extnl_crt2i * nA  # 0.60*nA  0.40 0.35

    #%%
    spk_e_1 = SpikeMonitor(group_e_1, record=True)
    spk_i_1 = SpikeMonitor(group_i_1, record=True)
    spk_e_2 = SpikeMonitor(group_e_2, record=True)
    spk_i_2 = SpikeMonitor(group_i_2, record=True)

    if record_LFP:
        lfp_moni_1 = StateMonitor(group_LFP_record_1, ('lfp'), dt=1 * ms, record=True)
        lfp_moni_2 = StateMonitor(group_LFP_record_2, ('lfp'), dt=1 * ms, record=True)

    net = Network(collect())
    net.store('state1')

    print('ie_w: %fnsiemens' % (syn_ie_1.w[0] / nsiemens))
    C = 0.25 * nF  # capacitance
    # g_l = 16.7*nS # leak capacitance
    v_l = -70 * mV  # leak voltage
    v_threshold = -50 * mV
    v_reset = -70 * mV  # -60*mV
    v_rev_I = -80 * mV
    v_rev_E = 0 * mV
    v_k = -85 * mV

    group_e_2.g_l = 16.7 * nS  # 16.7*nS # leak capacitance
    group_i_2.g_l = 25 * nS  # leak capacitance

    group_e_2.t_ref = t_ref * ms  # 16.7*nS # leak capacitance
    group_i_2.t_ref = t_ref * ms  # leak capacitance

    group_e_1.g_l = 16.7 * nS  # 16.7*nS # leak capacitance
    group_i_1.g_l = 25 * nS  # leak capacitance

    group_e_1.t_ref = t_ref * ms  # 16.7*nS # leak capacitance
    group_i_1.t_ref = t_ref * ms  # leak capacitance

    tic = time.perf_counter()
    simu_time_tot = (stim_scale_cls.stim_on[-1, 1] + 500) * ms

    simu_time1 = (stim_scale_cls.stim_on[n_StimAmp * n_perStimAmp - 1, 1] + int(inter_T[0])) * ms
    simu_time2 = 0 * ms
    T_total = simu_time1 + simu_time2

    print(stim_scale_cls.stim_on, simu_time_tot, simu_time1, simu_time2)

    net.run(simu_time1, profile=False)

    group_e_2.delta_gk[:] = adapt_value_new * nS

    net.run(simu_time2, profile=False)

    print('total time elapsed:', np.round((time.perf_counter() - tic) / 60, 2), 'min')

    spk_tstep_e1 = np.round(spk_e_1.t / (0.1 * ms)).astype(int)
    spk_tstep_i1 = np.round(spk_i_1.t / (0.1 * ms)).astype(int)
    spk_tstep_e2 = np.round(spk_e_2.t / (0.1 * ms)).astype(int)
    spk_tstep_i2 = np.round(spk_i_2.t / (0.1 * ms)).astype(int)

    now = datetime.datetime.now()

    param_all = {'delta_gk_1': delta_gk_1,
                 'delta_gk_2': delta_gk_2,
                 'new_delta_gk_2': new_delta_gk_2,
                 'tau_k': tau_k_,
                 # 'new_tau_k':40,
                 'tau_s_di': tau_s_di_,
                 'tau_s_de': tau_s_de_,
                 'tau_s_r': tau_s_r_,
                 # 'scale_d_p_i':scale_d_p_i,
                 'num_ee1': num_ee1,
                 'num_ei1': num_ei1,
                 'num_ii1': num_ii1,
                 'num_ie1': num_ie1,
                 'num_ee2': num_ee2,
                 'num_ei2': num_ei2,
                 'num_ii2': num_ii2,
                 'num_ie2': num_ie2,
                 'peak_p_e1_e2' : peak_p_e1_e2,
                 'tau_p_d_e1_e2' : tau_p_d_e1_e2,
                 'peak_p_e1_i2' : peak_p_e1_i2,
                 'tau_p_d_e1_i2' : tau_p_d_e1_i2,
                 'peak_p_e2_e1' : peak_p_e2_e1,
                 'tau_p_d_e2_e1' : tau_p_d_e2_e1,
                 'peak_p_e2_i1' : peak_p_e2_i1,
                 'tau_p_d_e2_i1' : tau_p_d_e2_i1,
                 # 'ie_ratio':ie_ratio_,
                 # 'mean_J_ee': ijwd.mean_J_ee,
                 # 'chg_adapt_range':6,
                 # 'p_ee':p_ee,
                 'simutime': int(round(T_total / ms)),
                 # 'chg_adapt_time': simu_time1/ms,
                 'chg_adapt_range': chg_adapt_range,
                 'chg_adapt_loca': chg_adapt_loca,
                 # 'chg_adapt_neuron': chg_adapt_neuron,
                 # 'scale_ee_1': scale_ee_1,
                 # 'scale_ei_1': scale_ei_1,
                 # 'scale_ie_1': scale_ie_1,
                 # 'scale_ii_1': scale_ii_1,
                 'ie_r_e': ie_r_e,
                 'ie_r_e1': ie_r_e1,
                 'ie_r_e2': ie_r_e2,
                 'ie_r_i': ie_r_i,
                 'ie_r_i1': ie_r_i1,
                 'ie_r_i2': ie_r_i2,
                 #'t_ref_a': t_ref_a / ms
                 }

    data = {'datetime': now.strftime("%Y-%m-%d %H:%M:%S"), 'dt': 0.1, 'loop_num': loop_num, 'data_dir': os.getcwd(),
            'param': param_all,
            'a1': {'param': param_a1,
                   # 'LFP':{'lfp1':lfp_moni.lfp1/nA, 'lfp2':lfp_moni.lfp2/nA, 'lfp3':lfp_moni.lfp3/nA},
                   'ge': {'i': spk_e_1.i[:], 't': spk_tstep_e1},
                   'gi': {'i': spk_i_1.i[:], 't': spk_tstep_i1}},
            'a2': {'param': param_a2,
                   'ge': {'i': spk_e_2.i[:], 't': spk_tstep_e2},
                   'gi': {'i': spk_i_2.i[:], 't': spk_tstep_i2}},
            'inter': {'param': param_inter}}

    if record_LFP:
        data['a1']['ge']['LFP'] = lfp_moni_1.lfp[:] / nA
        data['a2']['ge']['LFP'] = lfp_moni_2.lfp[:] / nA

    return_data = {'spk_e_1': spk_e_1, 'spk_e_2': spk_e_2, 'stim_on': stim_scale_cls.stim_on,
                   't_spon': (T_total / second), 'parameters' : data}

    return return_data






