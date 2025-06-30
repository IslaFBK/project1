import brian2.numpy_ as np
from brian2.only import *
import matplotlib.pyplot as plt
import sys
import pickle
import itertools
import gc
import os
import tempfile
import datetime
import time
import connection as cn
from connection import poisson_stimuli as psti
from connection import pre_process_sc
from connection import preprocess_2area
from connection import build_one_area
from connection import get_stim_scale
from connection import adapt_gaussian
from analysis import mydata
from analysis import firing_rate_analysis as fra
from analysis import my_analysis as mya
from joblib import Parallel, delayed
from pathlib import Path
from myscript.ie_search.compute_MSD_pdx import compute_MSD_pdx
import myscript.ie_search.utils as utils
from myscript.ie_search.batch_repeat import batch_repeat
import myscript.ie_search.load_repeat as load_repeat
import myscript.ie_search.critical_states_search as search

import logging
logging.getLogger('brian2').setLevel(logging.WARNING)

plt.rcParams.update({
    "text.usetex": True,  # 启用 LaTeX 渲染
    "font.family": "serif",  # 使用衬线字体（类似 LaTeX 默认）
    "font.serif": ["Times New Roman"]  # 指定字体
})

#%%
stdout_save = sys.stdout
stderr_save = sys.stderr
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

prefs.codegen.target = 'cython'

sys.stdout = stdout_save
sys.stderr = stderr_save

dir_cache = 'cache/'
prefs.codegen.runtime.cython.cache_dir = dir_cache
prefs.codegen.max_cache_dir_size = 120

#%% OS operation
# test if data_dir exists, if not, create one.
# FileExistsError means if menu is create by other progress or thread, ignore it.
root_dir = 'ie_ratio_2/'
Path(root_dir).mkdir(parents=True, exist_ok=True)
data_dir = f'{root_dir}/raw_data/'
Path(data_dir).mkdir(parents=True, exist_ok=True)
graph_dir = f'{root_dir}/graph/'
Path(graph_dir).mkdir(parents=True, exist_ok=True)
vedio_dir = f'{root_dir}/vedio/'
Path(vedio_dir).mkdir(parents=True, exist_ok=True)
state_dir = f'{root_dir}/state/'
Path(state_dir).mkdir(parents=True, exist_ok=True)
MSD_dir = f'./{graph_dir}/MSD/'
Path(MSD_dir).mkdir(parents=True, exist_ok=True)
pdx_dir = f'./{graph_dir}/pdx/'
Path(pdx_dir).mkdir(parents=True, exist_ok=True)
combined_dir = f'./{graph_dir}/combined'
Path(combined_dir).mkdir(parents=True, exist_ok=True)

#%% pick parameters and run `n_repeat` times
def pick_parameters_and_repeat_compute(param=None, n_repeat=128):
    # common title & path
    # param = (1.8, 2.4)
    ie_r_e1, ie_r_i1 = param
    n_repeat = 128
    common_title = rf'$\zeta^{{E}}$: {ie_r_e1:.4f}, $\zeta^{{I}}$: {ie_r_i1:.4f}'
    common_path = f're{ie_r_e1:.4f}_ri{ie_r_i1:.4f}'

    save_path_MSD = f'{MSD_dir}/{common_path}_{n_repeat}.png'
    save_path_pdx = f'{pdx_dir}/{common_path}_{n_repeat}.png'
    save_path_combined = f'{combined_dir}/{common_path}_{n_repeat}.png'

    batch_repeat(
        param=param,
        n_repeat=n_repeat,
        save_path_MSD=save_path_MSD,
        save_path_pdx=save_path_pdx,
        save_path_combined=save_path_combined
    )

#%% load datas and output graph diract
def pick_parameters_and_repeat_load(param=None, n_repeat=128):
    # common title & path
    # param = (1.8, 2.4)
    ie_r_e1, ie_r_i1 = param
    n_repeat = 128
    common_title = rf'$\zeta^{{E}}$: {ie_r_e1:.4f}, $\zeta^{{I}}$: {ie_r_i1:.4f}'
    common_path = f're{ie_r_e1:.4f}_ri{ie_r_i1:.4f}'
    save_path_MSD = f'{MSD_dir}/{common_path}_{n_repeat}.png'
    save_path_pdx = f'{pdx_dir}/{common_path}_{n_repeat}.png'
    save_path_combined = f'{combined_dir}/{common_path}_{n_repeat}.png'
    load_repeat.load_repeat(
        param=param,
        n_repeat=n_repeat,
        save_path_MSD=save_path_MSD,
        save_path_pdx=save_path_pdx,
        save_path_combined=save_path_combined
    )
#%% find packets
def find_packets(comb, index=1):
    result = compute_MSD_pdx(comb=comb, index=index, video=False)
    spk_rate = result['spk_rate']
    centre = result['centre']
    exist = utils.wave_packet_exist(spk_rate=spk_rate, centre=centre, r=0.6)
    return exist

def packets_exist_graph():
    params_loop = {
        'ie_r_e1': np.linspace(1.8, 2.5, 8),
        'ie_r_i1': np.linspace(1.8, 2.5, 8)
    }
    # generate looping parameter combinations
    loop_combinations = list(itertools.product(*params_loop.values()))
    # parallel compute
    results = Parallel(n_jobs=-1)(
        delayed(find_packets)(comb=comb, index=i+1)
        for i, comb in enumerate(loop_combinations)
    )
    # reshape results as 2D matrix
    results_matrix = np.array(results, dtype=int).reshape(len(params_loop['ie_r_e1']), 
                                                          len(params_loop['ie_r_i1']))
    # generate grid
    x, y = np.meshgrid(params_loop['ie_r_e1'], params_loop['ie_r_i1'], indexing='ij')
    # extend to 1D
    x_flat = x.flatten()
    y_flat = y.flatten()
    results_flat = results_matrix.flatten()
    # draw phase graph
    plt.figure(figsize=(6, 6))
    plt.scatter(y_flat, 
                x_flat, 
                c=results_flat, 
                cmap='Greens', 
                vmin=0, 
                vmax=1, 
                s=80, 
                edgecolors='k')
    plt.xlabel(r'$\zeta^{\rm I}$')
    plt.ylabel(r'$\zeta^{\rm E}$')
    plt.title('Wave Packet Existence Phase Diagram')
    plt.colorbar(label='Wave Packet Exists (1:Yes, 0:No)')
    plt.tight_layout()
    plt.savefig(f'{graph_dir}/Packet_Existence.png')
# start = time.perf_counter()
# packets_exist_graph()
# print(f'total time elapsed: {np.round((time.perf_counter() - start)/60,2)} min')

#%% auto search critical states and run `n_repeat` times
# params_loop = {
#     'ie_r_e1': np.linspace(1.8, 2.5, 8),
#     'ie_r_i1': np.linspace(1.8, 2.5, 8)
# }
# # generate looping parameter combinations
# loop_combinations = list(itertools.product(*params_loop.values()))
# # get total looping number
# loop_total = len(loop_combinations)

#%% 进化算法找态
def evalution_search():
    # 初始参数栅格
    initial_param = {
        'ie_r_e1': np.linspace(1.8, 2.5, 8),
        'ie_r_i1': np.linspace(1.8, 2.5, 8)
    }
    initial_params = list(itertools.product(*initial_param.values()))
    # 运行进化搜索
    history = search.evolve_search(
        initial_params,
        search.eval_func,
        r0=0.1,
        k=0.2,
        max_gen=10,
        n_child=5
    )
    # save
    print('saving')
    with open(f'{state_dir}/evolution.file', 'wb') as file:
        pickle.dump(history, file)

    # load
    print('loading')
    with open(f'{state_dir}/evolution.file', 'rb') as file:
        history = pickle.load(file)

    # print(len(history))

    # draw
    print('drawing')
    save_path = f'{graph_dir}/evaluation.png'
    search.plot_evolution_history(history=history,save_path=save_path)
    def get_min_alpha_critical(history):
        all_points = [h for gen in history for h in gen]
        critical_points = [h for h in all_points if h.get('critical', False)]
        if not critical_points:
            return None
        return min(critical_points, key=lambda h: h['alpha'])

    def get_top_n_alpha_critical(history, n=10):
        all_points = [h for gen in history for h in gen]
        critical_points = [h for h in all_points if h.get('critical', False)]
        if not critical_points:
            return []
        # 按alpha升序排序，取前n个
        top_n = sorted(critical_points, key=lambda h: h['alpha'])[:n]
        return top_n

    min_alpha = get_min_alpha_critical(history=history)
    top10 = get_top_n_alpha_critical(history, n=10)
    print('alpha min critical point:', min_alpha)
    for i, point in enumerate(top10):
        print(f"Top {i+1}: param={point['param']}, alpha={point['alpha']}")

    # recompute n_repeat times and draw statictical MSD and pdx
    if min_alpha is not None:
        pick_parameters_and_repeat_compute(min_alpha['param'])
    else:
        print('critical point not found')
