#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 2 16:37 2024

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
from sklearn.linear_model import LinearRegression
import sys
import pickle
from pathlib import Path
from levy import fit_levy, levy

#%%
prefs.codegen.target = 'cython'

dir_cache = 'cache/'
prefs.codegen.runtime.cython.cache_dir = dir_cache
prefs.codegen.max_cache_dir_size = 120

#%%
# test if data_dir exists, if not, create one.
# FileExistsError means if menu is create by other progress or thread, ignore it.
root_dir = 'phasesearch/'
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
MSD_dir = f'./{graph_dir}/MSD/'
Path(MSD_dir).mkdir(parents=True, exist_ok=True)
pdx_dir = f'./{graph_dir}/pdx/'
Path(pdx_dir).mkdir(parents=True, exist_ok=True)
combined_dir = f'./{graph_dir}/combined'
Path(combined_dir).mkdir(parents=True, exist_ok=True)

#%%
# define which (the index of) parameter you want to use in this simulation
sys_argv = 0

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

#%% adjustable parameters
def find_w_e(w_i,num_i,num_e,ie_ratio):
    return w_i/(num_e/num_i*ie_ratio)
def fine_w_i(w_e,num_e,num_i,ie_ratio):
    return w_e*(num_e/num_i*ie_ratio)

# mean synaptic weight 2
w_ee_1 = 11
w_ei_1 = 13.805
w_ie_1 = 41.835
w_ii_1 = 50
# IE-ratio 2
ie_r_e1 = 1.8312
ie_r_i1 = 1.8627

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

# K_a'b'ab
# num_ee = 275
# num_ei = 200
# num_ie = 115
# num_ii = 95
num_ee = 270
num_ei = 350
num_ie = 130
num_ii = 180

# K_a'b'ab
ijwd1.mean_SynNumIn_ee = num_ee
ijwd1.mean_SynNumIn_ei = num_ei
ijwd1.mean_SynNumIn_ie = num_ie
ijwd1.mean_SynNumIn_ii = num_ii

# mean synaptic weight
# ijwd1.w_ee_mean = w_ee_1
# ijwd1.w_ei_mean = w_ei_1
ijwd1.w_ee_mean = w_ee_1
ijwd1.w_ei_mean = w_ei_1
ijwd1.w_ie_mean = w_ie_1
ijwd1.w_ii_mean = w_ii_1

ijwd1.generate_ijw()    # generate synaptics and weight
ijwd1.generate_d_dist() # generate delay

# ?
param_a1 = {**ijwd1.__dict__}
# ?
del param_a1['i_ee'], param_a1['j_ee'], param_a1['w_ee'], param_a1['d_ee'], param_a1['dist_ee']  
del param_a1['i_ei'], param_a1['j_ei'], param_a1['w_ei'], param_a1['d_ei'], param_a1['dist_ei'] 
del param_a1['i_ie'], param_a1['j_ie'], param_a1['w_ie'], param_a1['d_ie'], param_a1['dist_ie'] 
del param_a1['i_ii'], param_a1['j_ii'], param_a1['w_ii'], param_a1['d_ii'], param_a1['dist_ii']

#%% plot network
# plt.figure()

# # plt.plot(ijwd_inter.e_lattice1[ijwd_inter.inter_e_neuron_1][:,0],
# #          ijwd_inter.e_lattice1[ijwd_inter.inter_e_neuron_1][:,1],'o')

# x = ijwd_inter.e_lattice1[ijwd_inter.inter_e_neuron_1][:, 0]
# y = ijwd_inter.e_lattice1[ijwd_inter.inter_e_neuron_1][:, 1]
# heatmap_resolution = (64+32)
# heatmap,xedges,yedges = np.histogram2d(x,y,bins=heatmap_resolution)
# heatmap = heatmap.T
# heatmap = plt.imshow(heatmap,cmap='viridis',origin='lower',aspect='auto',
#                      extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]])
# cbar = plt.colorbar()
# cbar.set_label('Density',fontsize=12)
# plt.savefig('network.png',dpi=300, format='png')

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

#%% ?adapt?
'''
chg_adapt_loca = [0,0]

adapt_value = adapt_gaussian.get_adaptation(base_amp=delta_gk_2,
                                            max_decrease=[delta_gk_2-new_delta_gk_2],
                                            sig=[chg_adapt_range],
                                            position=[chg_adapt_loca],
                                            n_side=int(round((ijwd1.Ne)**0.5)),
                                            width=ijwd1.width)
'''
                                            
#%%
'''stim 1; constant amplitude'''
'''no attention''' # ?background?
stim_dura = 1000 # ms duration of each stimulus presentation
transient = 3000 # ms initial transient period; when add stimulus
inter_time = 2000 # ms interval between trials without and with attention

stim_scale_cls = get_stim_scale.get_stim_scale()
stim_scale_cls.seed = 10 # random seed
n_StimAmp = 1 # 刺激幅值的种类数，这里只有一种
n_perStimAmp = 1 # 每种幅值的重复次数，这里每种只出现1次
stim_amp_scale = np.ones(n_StimAmp*n_perStimAmp) # 创建一个长度为n_SitmAmp*n_perSimAmp的数组，初始值全为1，表示每个刺激的幅值
for i in range(n_StimAmp):
    stim_amp_scale[i*n_perStimAmp:i*n_perStimAmp+n_perStimAmp] = 2**(i) # 遍历每种刺激幅值类型，将其对应的幅值设置为2**i。这里n_StimAmp=1，所以只会设置为1

stim_scale_cls.stim_amp_scale = stim_amp_scale # 把刚刚生成的刺激幅值数组赋值给stim_scale_cls实例，用于后续生成刺激序列
stim_scale_cls.stim_dura = stim_dura # 设置每个刺激的持续时间（毫秒）
stim_scale_cls.separate_dura = np.array([300,600]) # 设置刺激之间的间隔范围（单位毫秒），后续会在这个区间随机采样每个刺激之间的间隔
stim_scale_cls.get_scale() # 调用get_scale()方法，生册灰姑娘最终的刺激时间序列scale_stim和每个刺激的起止时间stim_on(对每个刺激，随机生成间隔，将刺激幅值插入到对应的时间段，得到完整的刺激序列)
stim_scale_cls.n_StimAmp = n_StimAmp
stim_scale_cls.n_perStimAmp = n_perStimAmp #记录刺激种类数和每种的重复次数，便于后续分析

# concatenate
init = np.zeros(transient//stim_scale_cls.dt_stim) # 生成一个长度为transient//dt_stim的全零数组，表示仿真开始的过渡期无刺激
stim_scale_cls.scale_stim = np.concatenate((init,stim_scale_cls.scale_stim)) # 把这个过渡期与后续的刺激序列拼接起来，形成完整的刺激输入
stim_scale_cls.stim_on += transient # 把所有刺激的起止时间(stim_on)整体向后平移transient毫秒，保证刺激实际发生在过渡期后

#%%
scale_1 = TimedArray(stim_scale_cls.scale_stim, dt=10*ms) # 用Brian2的TimedArray创建一个时间序列对象，scale_stim为刺激强度随时间的数组，时间步长为10ms。这个对象可直接用于神经元模型的输入，随访真是件自动取值。
'''
总结： 这段代码的作用是：为神经网络仿真生成一个带有初始过渡期、恒定幅值、随机间隔的刺激序列，并将其封装为 Brian2 可用的 TimedArray，为后续神经元模型提供输入。所有参数都可灵活调整，便于批量仿真和参数空间搜索。
'''
data_ = mydata.mydata()
param_a1 = {**param_a1, 'stim1':data_.class2dict(stim_scale_cls)}

# #%% External-Excitatory
# posi_stim_e1 = NeuronGroup(ijwd1.Ne,
#                            '''rates =  bkg_rates + stim_1*scale_1(t) : Hz
#                            bkg_rates : Hz
#                            stim_1 : Hz
#                            ''', threshold='rand()<rates*dt')

# posi_stim_e1.bkg_rates = 0*Hz
# posi_stim_e1.stim_1 = psti.input_spkrate(maxrate=[0],sig=[6],position=[[0,0]])*Hz

# synapse_e_extnl = cn.model_neu_syn_AD.synapse_e_AD
# syn_extnl_e1 = Synapses(posi_stim_e1, group_e_1, model=synapse_e_extnl, on_pre='x_E_extnl_post += w')
# syn_extnl_e1.connect('i==j')
# syn_extnl_e1.w = w_extnl_*nS#*tau_s_de_*nS

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

# # %% External-Inhibitory
# posi_stim_i1 = NeuronGroup(ijwd1.Ni,
#                            '''rates = bkg_rates + stim_1*scale_1(t) : Hz
#                            bkg_rates : Hz
#                            stim_1 : Hz
#                            ''', threshold='rand()<rates*dt')

# posi_stim_i1.bkg_rates = 1760*Hz
# posi_stim_i1.stim_1 = psti.input_spkrate(maxrate=[200],sig=[6],position=[[0,0]])*Hz

# synapse_i_extnl = cn.model_neu_syn_AD.synapse_i_AD
# syn_extnl_i1 = Synapses(posi_stim_i1, group_i_1, model=synapse_i_extnl, on_pre='x_I_extnl_post += w')
# syn_extnl_i1.connect('i==j')
# syn_extnl_i1.w = w_extnl_*nS

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
# group_e_1.I_extnl_crt = 0.51*nA
# group_i_1.I_extnl_crt = 0.60*nA

#%%
spk_e_1 = SpikeMonitor(group_e_1, record = True)
spk_i_1 = SpikeMonitor(group_i_1, record = True)

if record_LFP:
    lfp_moni = StateMonitor(group_LFP_record, ('lfp'), record = True)

#%%
net = Network(collect())
net.store('state1')

#%%
print('ie_w: %fnsiemens' %(syn_ie_1.w[0]/nsiemens))

tic = time.perf_counter()

simu_time_tot = (stim_scale_cls.stim_on[-1,1] + 500)*ms
simu_time1 = (stim_scale_cls.stim_on[n_StimAmp*n_perStimAmp-1,1] + round(inter_time/2))*ms
simu_time2 = simu_time_tot - simu_time1

net.run(simu_time1, profile=False) #,namespace={'tau_k': 80*ms}
net.run(simu_time2, profile=False) #,namespace={'tau_k': 80*ms}

print('total time elapsed:',np.round((time.perf_counter() - tic)/60,2), 'min')

#%%
spk_tstep_e1 = np.round(spk_e_1.t/(0.1*ms)).astype(int)
spk_tstep_i1 = np.round(spk_i_1.t/(0.1*ms)).astype(int)

now = datetime.datetime.now()

param_all = {'delta_gk_1':delta_gk_1,
             'delta_gk_2':delta_gk_2,
             'new_delta_gk_2':new_delta_gk_2,
             'tau_k': tau_k_,
             #'new_tau_k':40,
             'tau_s_di':tau_s_di_,
             'tau_s_de':tau_s_de_,
             'tau_s_r':tau_s_r_,
             #'scale_d_p_i':scale_d_p_i,
             'num_ee':num_ee,
             'num_ei':num_ei,
             'num_ii':num_ii,
             'num_ie':num_ie,
             #'ie_ratio':ie_ratio_,
             #'mean_J_ee': ijwd.mean_J_ee,
             #'chg_adapt_range':6, 
             #'p_ee':p_ee,
             'simutime':int(round(simu_time_tot/ms)),
             #'chg_adapt_time': simu_time1/ms,
             'chg_adapt_range': chg_adapt_range,
             #'chg_adapt_neuron': chg_adapt_neuron,
             #'scale_ee_1': scale_ee_1,
             #'scale_ei_1': scale_ei_1,
             #'scale_ie_1': scale_ie_1,
             #'scale_ii_1': scale_ii_1,
             #'ie_r_e': ie_r_e,
             #'ie_r_e1':ie_r_e1,
             #'ie_r_i': ie_r_i,
             'ie_r_i1': ie_r_i1,
            #  'ie_r_i2': ie_r_i2,
             't_ref': t_ref/ms}

data = {'datetime':now.strftime("%Y-%m-%d %H:%M:%S"), 
        'dt':0.1, 
        'loop_num':0, 
        'data_dir': os.getcwd(),
        'param':param_all,
        'a1':{'param':param_a1,
        #'LFP':{'lfp1':lfp_moni.lfp1/nA, 'lfp2':lfp_moni.lfp2/nA, 'lfp3':lfp_moni.lfp3/nA},
              'ge':{'i':spk_e_1.i[:],'t':spk_tstep_e1},    
              'gi':{'i':spk_i_1.i[:],'t':spk_tstep_i1}}}
if record_LFP:
    data['a1']['ge']['LFP'] = lfp_moni.lfp[:]/nA

#%%
''' save data to disk'''
with open(f"{data_dir}data{0}.file", 'wb') as file:
    pickle.dump(data, file)

'''load data from disk'''
# data_dir = 'raw_data/'
datapath = data_dir

data_load = mydata.mydata()
data_load.load(f"{data_dir}data{0}.file")

'''or load data from 'dictionary' '''
# data_load = mydata.mydata(data)

#%%
''' animation '''
from analysis import firing_rate_analysis as fra
from analysis import my_analysis as mya

start_time = transient - 500  #data.a1.param.stim1.stim_on[first_stim,0] - 300
end_time = int(round(simu_time_tot/ms))   #data.a1.param.stim1.stim_on[last_stim,0] + 1500

window = 10
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
'''
stim = [[[[31.5,31.5],[63.5,-0.5]], 
         [stim_on_off,stim_on_off], 
         [[6]*stim_on_off.shape[0],
          [6]*stim_on_off.shape[0]]]]
'''
stim = [[[[31.5,31.5]], 
         [stim_on_off], 
         [[6]*stim_on_off.shape[0]]]]

# unwrap periodic trajection
centre = data_load.a1.ge.centre_mass.centre
continous_centre = fra.unwrap_periodic_path(centre=centre, width=63)
continous_jump_size = np.diff(continous_centre)
continous_jump_dist = np.sqrt(np.sum(continous_jump_size**2,1))
# print(type(centre))
# print(dir(centre))

# plt.figure(figsize=(6, 6))
# plt.plot(continous_centre[:,0],continous_centre[:,1])
# # get range, take maximum as general boundary
# x_min, x_max = continous_centre[:, 0].min(), continous_centre[:, 0].max()
# y_min, y_max = continous_centre[:, 1].min(), continous_centre[:, 1].max()
# max_range = max(x_max - x_min, y_max - y_min) * 1.1  # take 10% edge
# # set the same range
# plt.xlim((x_min + x_max - max_range) / 2, (x_min + x_max + max_range) / 2)
# plt.ylim((y_min + y_max - max_range) / 2, (y_min + y_max + max_range) / 2)
# # plot
# plt.title(f'Centre trajectory E{N_e_ext} I{N_i_ext}')
# plt.xlabel('Horizontal Position')
# plt.ylabel('Vertical Position')
# plt.savefig(f'./phasesearch/centre_trajectory.png',
#             dpi=300, bbox_inches='tight')

# # pdf
# _, _, _ = fra.plot_adaptive_histogram(
#     continous_jump_dist,
#     save_path=f'./phasesearch/jump_distribution.png',
#     title=f'Jump distribution E{N_e_ext} I{N_i_ext}',
#     xlabel='Jump Distance (units)'
# )

# # p(X>x) power law check
# _, _, _ = fra.check_power_law(
#     data=continous_jump_dist,
#     tail_fraction=0.8,
#     plot=True,
#     save_path=f'./phasesearch/jump_prob_distribution.png',
#     N_e_ext=N_e_ext,
#     N_i_ext=N_i_ext
# )

# # pdf power law distribution check
# alpha, r2, fit_data = fra.check_power_law_density(
#     continous_jump_dist,
#     tail_fraction=0.8,
#     plot=True,
#     save_path=f'./phasesearch/jump_power_law.png',
#     N_e_ext=N_e_ext,
#     N_i_ext=N_i_ext
# )

# # spike statistic
# alpha, r2, fit_data = fra.analyze_coactive_power_law(
#     data_load.a1.ge.spk_rate,
#     tail_fraction=1,
#     save_path=f'./phasesearch/coactivity_power_law.png',
#     min_active=5  # 忽略少于5个神经元同时放电的情况
# )

# # plot trajectory
# _ = mya.plot_trajectory(
#     data=continous_centre,
#     title=f'Centre trajectory E{N_e_ext} I{N_i_ext}',
#     save_path=f'./phasesearch/centre_trajectory.png',
# )

# # pdf power law distribution check
# alpha_jump, r2_jump, _ = mya.check_jump_power_law(
#     continous_jump_dist,
#     tail_fraction=0.9,
#     save_path=f'./phasesearch/jump_power_law.png',
#     title=f'Jump step distribution E{N_e_ext} I{N_i_ext}'
# )

# # spike statistic
# alpha_spike, r2_spike, _ = mya.check_coactive_power_law(
#     data_load.a1.ge.spk_rate,
#     tail_fraction=1,
#     save_path=f'./phasesearch/coactivity_power_law.png',
#     title=f'Coactivity distribution E{N_e_ext} I{N_i_ext}',
#     min_active=1  # 忽略少于1个神经元同时放电的情况，正整数
# )

if not os.path.exists(f'{root_dir}/MSD.png') or 0:
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
    plt.savefig(f'{root_dir}/MSD.png', dpi=300, bbox_inches='tight')
    plt.close()

# unwrap periodic trajection
centre = data_load.a1.ge.centre_mass.centre
continous_centre = mya.unwrap_periodic_path(centre=centre, width=63)
continous_jump_size = np.diff(continous_centre)
continous_jump_dist = np.sqrt(np.sum(continous_jump_size**2,1))
jump_x = data_load.a1.ge.centre_mass.jump_size[:,0]
# pdx
if not os.path.exists(f'{root_dir}/pdx.png') or 1:
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
    plt.savefig(f'{root_dir}/pdx.png', dpi=300, bbox_inches='tight')
    plt.close()

# combined graph
if not os.path.exists(f'{root_dir}/combined.png') or 1:
    import matplotlib.image as mpimg
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # MSD
    img_msd = mpimg.imread(f'{root_dir}/MSD.png')
    ax[0].imshow(img_msd)
    ax[0].axis('off')
    # pdx
    img_pdx = mpimg.imread(f'{root_dir}/pdx.png')
    ax[1].imshow(img_pdx)
    ax[1].axis('off')
    plt.tight_layout()
    plt.savefig(f'{root_dir}/combined.png', dpi=300, bbox_inches='tight')
    plt.close()

# Animation
title = f'Animation'
ani = fra.show_pattern(spkrate1=data_load.a1.ge.spk_rate.spk_rate,
                       frames = frames,
                       start_time = start_time,
                       interval_movie=15,
                       anititle=title,
                       stim=None, 
                       adpt=None)
ani.save(f'./phasesearch/pattern.mp4',writer='ffmpeg',fps=60,dpi=100)