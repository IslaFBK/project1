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
import traceback
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
from myscript.ie_search.batch_repeat import batch_repeat2
import myscript.ie_search.load_repeat as load_repeat
import myscript.ie_search.critical_states_search as search
import myscript.send_email as send_email
import myscript.ie_search.compute_general as compute

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
recfield_dir = f'./{graph_dir}/recfield'
Path(recfield_dir).mkdir(parents=True, exist_ok=True)

#%% pick parameters and run `n_repeat` times
def pick_parameters_and_repeat_compute(param=None, n_repeat=128, video=False):
    # common title & path
    # param = (1.8, 2.4)
    ie_r_e1, ie_r_i1 = param
    common_title = rf'$\zeta^{{E}}$: {ie_r_e1:.4f}, $\zeta^{{I}}$: {ie_r_i1:.4f}'
    common_path = f're{ie_r_e1:.4f}_ri{ie_r_i1:.4f}'

    save_path_MSD = f'{MSD_dir}/1area_{common_path}_{n_repeat}.png'
    save_path_pdx = f'{pdx_dir}/1area_{common_path}_{n_repeat}.png'
    save_path_combined = f'{combined_dir}/1area_{common_path}_{n_repeat}.png'

    batch_repeat(
        param=param,
        n_repeat=n_repeat,
        save_path_MSD=save_path_MSD,
        save_path_pdx=save_path_pdx,
        save_path_combined=save_path_combined,
        video=video
    )

def pick_parameters_and_repeat_compute2(param=None, n_repeat=128, video=False):
    # common title & path
    # param = (1.8, 2.4)
    root_dir = '2area/'
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
    recfield_dir = f'./{graph_dir}/recfield'
    Path(recfield_dir).mkdir(parents=True, exist_ok=True)
    ie_r_e1, ie_r_i1, ie_r_e2, ie_r_i2 = param
    common_title = (rf'$\zeta^{{E1}}$: {ie_r_e1:.4f}, '
                    rf'$\zeta^{{I1}}$: {ie_r_i1:.4f}, '
                    rf'$\zeta^{{E2}}$: {ie_r_e2:.4f}, '
                    rf'$\zeta^{{I2}}$: {ie_r_i2:.4f}')
    common_path = f're1{ie_r_e1:.4f}_ri1{ie_r_i1:.4f}_re2{ie_r_e2:.4f}_ri2{ie_r_i2:.4f}'

    save_path_MSD = f'{MSD_dir}/2area_{common_path}_{n_repeat}'
    save_path_pdx = f'{pdx_dir}/2area_{common_path}_{n_repeat}'
    save_path_combined = f'{combined_dir}/2area_{common_path}_{n_repeat}'

    batch_repeat2(
        param=param,
        n_repeat=n_repeat,
        save_path_MSD=save_path_MSD,
        save_path_pdx=save_path_pdx,
        save_path_combined=save_path_combined,
        video=video
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

def pick_farthest_critical_point(history):
    # 展开所有critical点
    all_points = [h for gen in history for h in gen]
    critical_points = [h for h in all_points if h.get('critical', False)]
    if not critical_points:
        return None, None

    # 找到alpha最小的点
    min_alpha_point = min(critical_points, key=lambda h: h['alpha'])
    min_param = np.array(min_alpha_point['param'])

    # 计算欧氏距离，选距离最大的
    def euclidean_dist(p):
        return np.linalg.norm(np.array(p['param']) - min_param)

    farthest_point = max(critical_points, key=euclidean_dist)
    return min_alpha_point, farthest_point

def evalution_search(compute=False, repeat_MSD=False):
    if compute:
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

    # pick points
    min_alpha = get_min_alpha_critical(history=history)
    top10 = get_top_n_alpha_critical(history, n=10)
    print('alpha min critical point:', min_alpha)
    for i, point in enumerate(top10):
        print(f"Top {i+1}: param={point['param']}, alpha={point['alpha']}")
    min_alpha_point, farthest_point = pick_farthest_critical_point(history)
    print("alpha最小点:", min_alpha_point)
    print("距离最远的critical点:", farthest_point)

    if repeat_MSD:
        # recompute n_repeat times and draw statictical MSD and pdx
        if min_alpha is not None:
            pick_parameters_and_repeat_compute(min_alpha['param'])
        else:
            print('critical point not found')

# Receptive_field
def last_generation(history):
    last_gen = history[-1]
    last_gen = [h for h in last_gen if h.get('critical', False)]
    return last_gen

def receptive_field(param):
    result0 = compute.compute_1(comb=param, sti=False, save_load=False)
    result1 = compute.compute_1(comb=param, sti=True,  save_load=False)
    spk_rate0 = result0['spk_rate']
    spk_rate1 = result1['spk_rate']
    ie_r_e1, ie_r_i1 = param
    common_path = f're{ie_r_e1:.4f}_ri{ie_r_i1:.4f}'
    save_path1 = f'{graph_dir}/fr_ext{common_path}.png'
    save_path2 = f'{graph_dir}/fr_ext-dist{common_path}.png'
    data_path = f'{state_dir}/fr_ext{common_path}.file'
    r_rf = mya.receptive_field(spk_rate0=spk_rate0, 
                               spk_rate1=spk_rate1, 
                               save_path1 = save_path1,
                               save_path2 = save_path2,
                               data_path=data_path,
                               plot=True)
    return r_rf

def receptive_field_repeat(param, n_repeat, plot=False, 
                           video0=False, video1=False, save_load0=False, save_load1=False):
        
    result0 = Parallel(n_jobs=-1)(
        delayed(compute.compute_1)(comb=param, seed=i, index=i, sti=False, 
                                   video=video0, save_load=save_load0)
        for i in range(n_repeat)
    )
    result1 = Parallel(n_jobs=-1)(
        delayed(compute.compute_1)(comb=param, seed=i, index=i, sti=True,  
                                   video=video1, save_load=save_load1)
        for i in range(n_repeat)
    )
    # 提取所有 spk_rate 并堆叠
    spk_rate0_all = np.stack([r['spk_rate'] for r in result0], axis=0)  # shape: (n_repeat, Nx, Ny, T)
    spk_rate1_all = np.stack([r['spk_rate'] for r in result1], axis=0)

    # 在第一个维度取平均
    spk_rate0_mean = np.mean(spk_rate0_all, axis=0)  # shape: (Nx, Ny, T)
    spk_rate1_mean = np.mean(spk_rate1_all, axis=0)

    ie_r_e1, ie_r_i1 = param
    common_path = f're{ie_r_e1:.4f}_ri{ie_r_i1:.4f}'

    save_path = f'{recfield_dir}/{n_repeat}fr_ext-dist{common_path}.png'
    data_path = f'{state_dir}/{n_repeat}fr_ext{common_path}.file'
    r_rf = mya.receptive_field(spk_rate0=spk_rate0_mean,
                               spk_rate1=spk_rate1_mean,
                               save_path=save_path,
                               data_path=data_path,
                               plot=plot)
    return r_rf

def find_max_min_receptive_field(n_repeat):
    # load
    print('loading')
    with open(f'{state_dir}/evolution.file', 'rb') as file:
        history = pickle.load(file)
    last_gen = last_generation(history=history)

    max_val = -np.inf
    min_val = np.inf
    max_param = None
    min_param = None
    loop_total = len(last_gen)
    loop_num = 0
    r_rf_history = []
    for entry in last_gen:
        loop_num = loop_num + 1
        param = entry['param']
        ie_r_e1, ie_r_i1 = param
        common_path = f're{ie_r_e1:.4f}_ri{ie_r_i1:.4f}'
        save_path = f'{recfield_dir}/{n_repeat}fr_ext-dist{common_path}.png'
        data_path = f'{state_dir}/{n_repeat}fr_ext{common_path}.file'
        if os.path.exists(data_path):
            with open(data_path, 'rb') as file:
                fr_ext = pickle.load(file)
            field = mya.load_receptive_field(fr_ext,
                                             save_path=save_path,
                                             plot=True)
        else:
            field = receptive_field_repeat(param=param, n_repeat=n_repeat)
        
        if field is None:
            print(f"警告：对于参数 {param}, receptive_field_repeat 返回 None, 跳过")
            send_email.send_email('Progress', f"警告：对于参数 {param}, receptive_field_repeat 返回 None, 跳过")
            continue
            
        if 'r_rf' not in field:
            print(f"警告：对于参数 {param}, 返回的字段中缺少 'r_rf' 键, 跳过")
            send_email.send_email('Progress', f"警告：对于参数 {param}, 返回的字段中缺少 'r_rf' 键, 跳过")
            continue

        r_rf = field['r_rf']

        if r_rf > max_val:
            max_val = r_rf
            max_param = param
        if r_rf < min_val:
            min_val = r_rf
            min_param = param
        r_rf_result = [{'r_rf': r_rf, 'max_r_rf': max_val, 'min_r_rf': min_val, 'max_param': max_param, 'min_param': min_param}]
        info = [{'param': param, 'r_rf_result': r_rf_result}]
        r_rf_history.append(info)
        # save
        with open(f'{state_dir}/r_rf_history.file', 'wb') as file:
            pickle.dump(r_rf_history, file)
        # report progress
        print(f'Complete {loop_num} in {loop_total}, \n parameter: {param}, r_rf: {r_rf}. Now, \n max r_rf: {max_val}, max parameter: {max_param}, \n min r_rf: {min_val}, min parameter: {min_param}')

        send_email.send_email('Progress', f'Complete {loop_num} in {loop_total}, \n parameter: {param}, r_rf: {r_rf}. Now, \n max r_rf: {max_val}, max parameter: {max_param}, \n min r_rf: {min_val}, min parameter: {min_param}')

    print(f'最大receptive field参数: {max_param}, 最大值: {max_val}')
    print(f'最小receptive field参数: {min_param}, 最小值: {min_val}')
    # draw
    for entry in last_gen:
        loop_num = loop_num + 1
        param = entry['param']
        ie_r_e1, ie_r_i1 = param
        common_path = f're{ie_r_e1:.4f}_ri{ie_r_i1:.4f}'
        save_path = f'{recfield_dir}/{n_repeat}fr_ext-dist{common_path}.png'
        data_path = f'{state_dir}/{n_repeat}fr_ext{common_path}.file'
        if not os.path.exists(save_path) or 1:
            with open(data_path, 'rb') as file:
                fr_ext = pickle.load(file)
            _ = mya.load_receptive_field(fr_ext=fr_ext, save_path=save_path, plot=True)
    # return {
    #     'max_pm': max_param,
    #     'max_rf': max_val,
    #     'min_pm': min_param,
    #     'min_rf': min_val
    # }

def load_and_draw_receptive_field(param, n_repeat=64):
    ie_r_e1, ie_r_i1 = param
    common_path = f're{ie_r_e1:.4f}_ri{ie_r_i1:.4f}'
    save_path = f'{recfield_dir}/{n_repeat}fr_ext-dist{common_path}.png'
    data_path = f'{state_dir}/{n_repeat}fr_ext{common_path}.file'
    if not os.path.exists(save_path) or 1:
        with open(data_path, 'rb') as file:
            fr_ext = pickle.load(file)
        _ = mya.load_receptive_field(fr_ext=fr_ext, save_path=save_path, plot=True)

#%% Execution area
try:
    send_email.send_email('begin running', 'ie_search.main running')
    #%% test
    param = (1.8512390285440765, 2.399131446733395)
    compute.compute_1(comb=param, seed=10, index=0, sti=True, video=True, save_load=False)

    #%% evalutionary search
    # evalution_search()

    #%% repeat 1 area computation
    # param1 = (1.824478865468595, 2.4061741957998843)
    # param2 = (1.9905682989732332, 2.558780313870593)
    # pick_parameters_and_repeat_compute(param=param2, video=True)

    #%% repeat 2 area computation
    # param = (1.824478865468595, 2.4061741957998843, 1.9905682989732332, 2.558780313870593)
    # pick_parameters_and_repeat_compute2(param=param,
    #                                     n_repeat=128,
    #                                     video=True)

    #%% receptive field
    # param = (1.824478865468595, 2.4061741957998843)
    # receptive_field(param=param)

    #%% repeat receptive field
    # # first layer
    # param = (1.795670364314891, 2.449990451446889)
    # receptive_field_repeat(param=param, n_repeat=64, plot=True)
    # # second layer
    # param = (1.8512390285440765, 2.399131446733395)
    # receptive_field_repeat(param=param, n_repeat=64, plot=True)

    #%% search receptive field
    # result = find_max_min_receptive_field(n_repeat=64)

    #%% repeat 2 area coputation recetive field
    # param = (1.795670364314891, 2.449990451446889, 1.8512390285440765, 2.399131446733395)
    # pick_parameters_and_repeat_compute2(param=param,
    #                                     n_repeat=128,
    #                                     video=True)

    #%% load and draw receptive field
    # # first layer
    # param = (1.795670364314891, 2.449990451446889)
    # load_and_draw_receptive_field(param, n_repeat=64)
    # # second layer
    # param = (1.8512390285440765, 2.399131446733395)
    # load_and_draw_receptive_field(param, n_repeat=64)

    

    send_email.send_email('code executed', 'ie_search.main accomplished')
except Exception:
    # 捕获异常并发送邮件
    error_info = traceback.format_exc()  # 获取完整错误堆栈
    print(error_info)
    send_email.send_error_email(
        subject="script execution error",
        body=f"info: \n\n{error_info}"
    )
    sys.exit(1)  # 退出程序并返回错误码