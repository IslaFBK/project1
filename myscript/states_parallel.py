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

'''
Cause this script scane parameters exactly, it's not flexible enough, now can be fully replaced by state_parallel_auto.py
'''

data_dir = 'parallel/raw_data/'
graph_dir = 'parallel/graph/'
vedio_dir = 'parallel/vedio/'
state_dir = 'parallel/state/'

start = time.perf_counter()
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
loop_total = len(loop_combinations)

def states_compute(comb, loop_num):
    loop_num += 1
    num_ee, num_ei, num_ie, num_ii = comb
    print(f'Processing {loop_num}/{loop_total},\n num_ee={num_ee},num_ei={num_ei},num_ie={num_ie},num_ii={num_ii}')

    '''load data from disk'''
    datapath = data_dir
    # common title & path
    EE = str('{EE}')
    EI = str('{EI}')
    IE = str('{IE}')
    II = str('{II}')
    common_title = rf'$K^{EE}$={num_ee}, $K^{EI}$={num_ei}, $K^{IE}$={num_ie}, $K^{II}$={num_ii}'
    common_path = f'EE{num_ee:03d}_EI{num_ei:03d}_IE{num_ie:03d}_II{num_ii:03d}'
    data_load = mydata.mydata()
    data_load.load(f"{data_dir}data_{common_path}.file")

    # decide if analyze (p_peak>1)
    p_peak_ee=data_load.a1.param.p_peak_ee
    p_peak_ei=data_load.a1.param.p_peak_ei
    p_peak_ie=data_load.a1.param.p_peak_ie
    p_peak_ii=data_load.a1.param.p_peak_ii
    p_peak = np.max([p_peak_ee,p_peak_ei,p_peak_ie,p_peak_ii])
    if p_peak>1:
        return None

    #%% analysis
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

    # # Animation
    # title = f'Animation \n {common_title}'
    # ani = fra.show_pattern(spkrate1=data_load.a1.ge.spk_rate.spk_rate,
    #                     frames = frames,
    #                     start_time = start_time,
    #                     interval_movie=15,
    #                     anititle=title,
    #                     stim=None, 
    #                     adpt=None)
    # ani.save(f'./{vedio_dir}/{common_path}_pattern.mp4',writer='ffmpeg',fps=60,dpi=100)

    # unwrap periodic trajection
    centre = data_load.a1.ge.centre_mass.centre
    continous_centre = mya.unwrap_periodic_path(centre=centre, width=63)
    continous_jump_size = np.diff(continous_centre)
    continous_jump_dist = np.sqrt(np.sum(continous_jump_size**2,1))

    # pdf power law distribution check
    alpha_jump, r2_jump, _ = mya.check_jump_power_law(
        continous_jump_dist,
        tail_fraction=0.9,
        plot=False,
        save_path=f'./{graph_dir}/jump/jump_{common_path}.png',
        title=f'Jump step distribution of \n {common_title}'
    )

    # spike statistic
    alpha_spike, r2_spike, _ = mya.check_coactive_power_law(
        data_load.a1.ge.spk_rate,
        tail_fraction=1,
        plot=False,
        save_path=f'./{graph_dir}/coactivity/coactivity_{common_path}.png',
        title=f'Coactivity distribution of \n {common_title}',
        min_active=1  # 忽略少于1个神经元同时放电的情况
    )

    # release RAM
    plt.close('all')
    gc.collect()

    # record parameters & states
    return {
        'params': {
            'num_ee': num_ee,
            'num_ei': num_ei,
            'num_ie': num_ie,
            'num_ii': num_ii
        },
        'states': {
            'alpha_jump': alpha_jump,
            'r2_jump': r2_jump,
            'alpha_spike': alpha_spike,
            'r2_spike': r2_spike
        }
    }

# parallel compute
results = Parallel(n_jobs=-1)(
    delayed(states_compute)(comb, i+1)
    for i, comb in enumerate(loop_combinations)
)

# establish analyzer to collect result
Analyzer = mya.CriticalityAnalyzer()
for result in results: # not sorted
    if result is not None:
        Analyzer.states_collector(params=result['params'], states=result['states'])

# save states
states_path = ''.join(
    f"{k.replace('num_','')}({v[0]}_{v[-1]})"
    for k, v in sorted(params_loop.items())
)

now = datetime.datetime.now()
data_states = {'datetime':now.strftime("%Y-%m-%d %H:%M:%S"), 
               'dt':0.1, 
               'data_dir': os.getcwd(),
               'params': Analyzer.params,
               'states': Analyzer.states
               }

''' save phase data to disk'''
with open(f"{state_dir}data_{loop_total}_states_{states_path}.file", 'wb') as file:
    pickle.dump(data_states, file)
print(f'data states of {loop_total} states saved to {state_dir}')

# # load phase data and rebuild Analyzer object
# ''' load phase data '''
# with open(f"{data_dir}data_{loop_total}_states_{states_path}.file", 'rb') as file:
#     data_states = pickle.load(file)
# Analyzer = mya.CriticalityAnalyzer()
# Analyzer.params = data_states['params']
# Analyzer.states = data_states['states']

# # draw phase graph
# fixed_params = {'num_ie': 400, 'num_ii': 400}
# fixed_path = "_".join(f"{k}{v}" for k, v in sorted(fixed_params.items()))
# Analyzer.plot_phase_diagrams(
#     param_x='num_ee',
#     param_y='num_ei',
#     state_vars=['alpha_jump', 'r2_jump', 'alpha_spike', 'r2_spike'],
#     fixed_params=fixed_params,  # 固定这两个参数
#     save_path=f'./phasesearch2/states_{loop_total}_{fixed_path}.png'
# )
# print(f'Phase graph of {loop_total} states saved to ./parallel/{fixed_path}')