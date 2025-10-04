import brian2.numpy_ as np
from brian2.only import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.spatial import ConvexHull, Delaunay
from scipy.stats import chi2
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
recfield_dir = f'./{graph_dir}/recfield'
Path(recfield_dir).mkdir(parents=True, exist_ok=True)
LFP_dir = f'./{graph_dir}/LFP'
Path(LFP_dir).mkdir(parents=True, exist_ok=True)

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
            # 评价是否临界，并输出alpha
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
    ellipse_info = search.plot_evolution_history(history=history,save_path=save_path)
    # 保存椭圆边界信息
    with open(f'{state_dir}/critical_ellipse.file', 'wb') as file:
        pickle.dump(ellipse_info, file)
    
    plt.figure(figsize=(14, 7))
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

# 在椭圆中均匀随机采样
def sample_in_ellipse(mean, cov, conf_level, n_samples):
    dim = len(mean)
    threshold = np.sqrt(chi2.ppf(conf_level, df=dim))
    samples = []
    while len(samples) < n_samples:
        # 在包络盒内均匀采样
        box_min = mean - 2*np.sqrt(np.diag(cov))
        box_max = mean + 2*np.sqrt(np.diag(cov))
        point = np.random.uniform(box_min, box_max)
        # 判断是否在椭圆内
        dist = np.sqrt((point-mean) @ np.linalg.inv(cov) @ (point-mean).T)
        if dist < threshold:
            samples.append(point)
    return np.array(samples)

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

# exam different distance firing rate
def receptive_field_repeat(param, n_repeat, plot=False, 
                           video0=False, video1=False, maxrate=5000,
                           save_load0=False, save_load1=False):
    
    if video0:
        result0 = Parallel(n_jobs=-1)(
            delayed(compute.compute_1)(comb=param, seed=i, index=i, sti=False, 
                                    video=(i==0), save_load=save_load0)
            for i in range(n_repeat)
        )
    else:
        result0 = Parallel(n_jobs=-1)(
            delayed(compute.compute_1)(comb=param, seed=i, index=i, sti=False, 
                                    video=False, save_load=save_load0)
            for i in range(n_repeat)
        )
    if video1:
        result1 = Parallel(n_jobs=-1)(
            delayed(compute.compute_1)(comb=param, seed=i, index=i, sti=True, maxrate=maxrate,
                                    video=(i==0), save_load=save_load1)
            for i in range(n_repeat)
        )
    else:
        result1 = Parallel(n_jobs=-1)(
            delayed(compute.compute_1)(comb=param, seed=i, index=i, sti=True, maxrate=maxrate,
                                    video=False, save_load=save_load1)
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

    save_path = f'{recfield_dir}/{n_repeat}_{maxrate}fr_ext-dist{common_path}.png'
    data_path = f'{state_dir}/{n_repeat}_{maxrate}fr_ext{common_path}.file'
    r_rf = mya.receptive_field(spk_rate0=spk_rate0_mean,
                               spk_rate1=spk_rate1_mean,
                               save_path=save_path,
                               data_path=data_path,
                               plot=plot)
    return r_rf

def find_max_min_receptive_field(n_repeat, maxrate=1000):
    # load parameters directly
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
        save_path = f'{recfield_dir}/{n_repeat}_{maxrate}fr_ext-dist{common_path}.png'
        data_path = f'{state_dir}/{n_repeat}_{maxrate}fr_ext{common_path}.file'
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

def find_receptive_field_distribution_in_range(n_repeat, range_path, maxrate=1000, n_sample=1000):
    # 读取椭圆参数
    with open(range_path, 'rb') as file:
        ellipse_info = pickle.load(file)
    mean = ellipse_info['mean']
    cov = ellipse_info['cov']
    conf_level = ellipse_info.get('conf_level', 0.99)

    # 椭圆内采样参数
    params = sample_in_ellipse(mean, cov, conf_level, n_sample)

    max_val = -np.inf
    min_val = np.inf
    max_param = None
    min_param = None
    rf_list = []
    r_rf_history = []
    loop_total = len(params)
    loop_num = 0

    for param in params:
        loop_num += 1
        param_tuple = tuple(param)
        try:
            field = receptive_field_repeat(param=param_tuple, n_repeat=n_repeat, maxrate=maxrate, plot=False)
            r_rf = field['r_rf'] if isinstance(field, dict) and 'r_rf' in field else None
        except Exception as e:
            print(f"参数 {param_tuple} 计算失败: {e}")
            send_email.send_email('Progress', f"参数 {param_tuple} 计算失败: {e}")
            continue
        rf_list.append((param_tuple, r_rf))
        if r_rf is not None:
            if r_rf > max_val:
                max_val = r_rf
                max_param = param_tuple
            if r_rf < min_val:
                min_val = r_rf
                min_param = param_tuple
        
        r_rf_result = [{'r_rf': r_rf, 'max_r_rf': max_val, 'min_r_rf': min_val, 'max_param': max_param, 'min_param': min_param}]
        info = [{'param': param_tuple, 'r_rf_result': r_rf_result}]
        r_rf_history.append(info)

        # 实时保存
        with open(f'{state_dir}/rf_landscape_{n_sample}.file', 'wb') as file:
            pickle.dump(r_rf_history, file)

        # 发邮件报告进度
        send_email.send_email(
            'Progress',
            f'Complete {loop_num} in {loop_total}, \n parameter: {param_tuple}, r_rf: {r_rf}. Now, \n max r_rf: {max_val}, max parameter: {max_param}, \n min r_rf: {min_val}, min parameter: {min_param}'
        )

    print(f'最大receptive field参数: {max_param}, 最大值: {max_val}')
    print(f'最小receptive field参数: {min_param}, 最小值: {min_val}')

    # 画地形图
    x = [p[0] for p, rf in rf_list]
    y = [p[1] for p, rf in rf_list]
    z = [rf for p, rf in rf_list]
    plt.figure(figsize=(7,6))
    sc = plt.scatter(x, y, c=z, cmap='viridis', s=60)
    plt.colorbar(sc, label='Receptive Field')
    plt.xlabel(r'$\zeta^{\rm E}$')
    plt.ylabel(r'$\zeta^{\rm I}$')
    plt.title('Receptive Field Landscape')
    plt.tight_layout()
    plt.savefig(f'{graph_dir}/rf_landscape_{n_sample}.png', dpi=300)
    plt.close()

    # 画3维地形图
    x = [p[0] for p, rf in rf_list]
    y = [p[1] for p, rf in rf_list]
    z = [rf for p, rf in rf_list]
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=40)
    ax.set_xlabel(r'$\zeta^{\rm E}$')
    ax.set_ylabel(r'$\zeta^{\rm I}$')
    ax.set_zlabel('Receptive Field')
    ax.set_title('Receptive Field 3D Landscape')
    fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10, label='Receptive Field')
    plt.tight_layout()
    plt.savefig(f'{graph_dir}/rf_landscape_3d_{n_sample}.png', dpi=300)
    plt.close()

    return {
        'max_param': max_param,
        'max_rf': max_val,
        'min_param': min_param,
        'min_rf': min_val
    }

def load_and_draw_receptive_field(param, maxrate=5000, n_repeat=64):
    ie_r_e1, ie_r_i1 = param
    common_path = f're{ie_r_e1:.4f}_ri{ie_r_i1:.4f}'
    save_path = f'{recfield_dir}/{n_repeat}_{maxrate}fr_ext-dist{common_path}.png'
    data_path = f'{state_dir}/{n_repeat}_{maxrate}fr_ext{common_path}.file'
    if not os.path.exists(save_path) or 1:
        with open(data_path, 'rb') as file:
            fr_ext = pickle.load(file)
        _ = mya.load_receptive_field(fr_ext=fr_ext, save_path=save_path, plot=True)

# find optimized stimuli rate
def check_r_rf_maxrate(param=None, 
                        seq_maxrate=None,
                        n_repeat=128):
    # if param is None:
    #     param = (1.8512390285440765, 2.399131446733395)
    # if seq_maxrate is None:
    #     seq_maxrate = [0, 1, 10, 100, 200, 500, 1000, 2000, 5000]
    ie_r_e1, ie_r_i1 = param
    common_path = f're{ie_r_e1:.4f}_ri{ie_r_i1:.4f}'
    seq_r_rf = []
    loop_num = 0
    loop_total = len(seq_maxrate)
    for maxrate in seq_maxrate:
        loop_num+=1
        r_rf = receptive_field_repeat(param=param, n_repeat=n_repeat, maxrate=maxrate, plot=True, video1=True)
        seq_r_rf.append(r_rf['r_rf'])
        print(f'completed {loop_num} in {loop_total} \n max rate: {maxrate}, r_rf: {r_rf}')
        contents = (
            f"Completed {loop_num} in {loop_total}\n"
            f"max rate: {maxrate}\n"
            f"r_rf: {r_rf}"
        )
        send_email.send_email('Check r_rf (mr)', contents=contents)

    plt.figure(figsize=(5,5))
    plt.plot(seq_maxrate, seq_r_rf, 'o-', label='Mean $r_{rf}$')
    plt.xlabel('Max rate (Hz)')
    plt.ylabel('$r_{rf}$')
    plt.title(f'$r_{{rf}}$-Max rate \\n parameter: {param}')
    plt.savefig(f'{recfield_dir}/r_rf-mr_lin{common_path}_{n_repeat}.png')

    plt.figure(figsize=(5,5))
    plt.plot(seq_maxrate, seq_r_rf, 'o-', label='Mean $r_{rf}$')
    plt.xlabel('Max rate (Hz)')
    plt.ylabel('$r_{rf}$')
    plt.title(f'$r_{{rf}}$-Max rate \\n parameter: {param}')
    plt.savefig(f'{recfield_dir}/r_rf-mr_log{common_path}_{n_repeat}.png')

# exam middle 4 point firing rate (receptive field)
def receptive_field_repeat2(param, n_repeat, plot=False, 
                            video0=False, video1=False, maxrate=1000, sig=2, sti_type='uniform',
                            save_load0=False, save_load1=False):
    
    if video0:
        result0 = Parallel(n_jobs=-1)(
            delayed(compute.compute_1)(comb=param, seed=i, index=i, sti=False, 
                                    video=(i==0), save_load=save_load0)
            for i in range(n_repeat)
        )
    else:
        result0 = Parallel(n_jobs=-1)(
            delayed(compute.compute_1)(comb=param, seed=i, index=i, sti=False, 
                                    video=False, save_load=save_load0)
            for i in range(n_repeat)
        )
    if video1:
        result1 = Parallel(n_jobs=-1)(
            delayed(compute.compute_1)(comb=param, seed=i, index=i, sti=True, maxrate=maxrate, sig=sig, sti_type=sti_type, 
                                    video=(i==0), save_load=save_load1)
            for i in range(n_repeat)
        )
    else:
        result1 = Parallel(n_jobs=-1)(
            delayed(compute.compute_1)(comb=param, seed=i, index=i, sti=True, maxrate=maxrate, sig=sig, sti_type=sti_type, 
                                    video=False, save_load=save_load1)
            for i in range(n_repeat)
        )
    # 提取所有 spk_rate 并堆叠
    spk_rate0_all = np.stack([r['spk_rate'] for r in result0], axis=0)  # shape: (n_repeat, Nx, Ny, T)
    spk_rate1_all = np.stack([r['spk_rate'] for r in result1], axis=0)

    # 在第一个维度(realization)取平均
    spk_rate0_mean = np.mean(spk_rate0_all, axis=0)  # shape: (Nx, Ny, T)
    spk_rate1_mean = np.mean(spk_rate1_all, axis=0)


    # 取中心最近的四个点
    center_indices = [(31, 31), (31, 32), (32, 31), (32, 32)]
    center_spk_rate0 = np.array([spk_rate0_mean[x, y, :] for x, y in center_indices])  # shape: (4, T)
    center_spk_rate1 = np.array([spk_rate1_mean[x, y, :] for x, y in center_indices])  # shape: (4, T)

    # 对这四个点做平均
    center_spk_rate0_mean = np.mean(center_spk_rate0, axis=0)  # shape: (T,)
    center_spk_rate1_mean = np.mean(center_spk_rate1, axis=0)  # shape: (T,)
    center_spk_rate0_tmean = np.mean(center_spk_rate0_mean, axis=0)
    center_spk_rate1_tmean = np.mean(center_spk_rate1_mean, axis=0)

    ratio = center_spk_rate1_tmean/center_spk_rate0_tmean
    diff = center_spk_rate1_tmean-center_spk_rate0_tmean
    return ratio, diff

# exam whole field firing rate (receptive field)
def receptive_field_repeat3(param, n_repeat, plot=False, 
                            video0=False, video1=False, maxrate=1000, sig=2, sti_type='uniform',
                            save_load0=False, save_load1=False):
    
    if video0:
        result0 = Parallel(n_jobs=-1)(
            delayed(compute.compute_1)(comb=param, seed=i, index=i, sti=False, 
                                    video=(i==0), save_load=save_load0)
            for i in range(n_repeat)
        )
    else:
        result0 = Parallel(n_jobs=-1)(
            delayed(compute.compute_1)(comb=param, seed=i, index=i, sti=False, 
                                    video=False, save_load=save_load0)
            for i in range(n_repeat)
        )
    if video1:
        result1 = Parallel(n_jobs=-1)(
            delayed(compute.compute_1)(comb=param, seed=i, index=i, sti=True, maxrate=maxrate, sig=sig, sti_type=sti_type, 
                                    video=(i==0), save_load=save_load1)
            for i in range(n_repeat)
        )
    else:
        result1 = Parallel(n_jobs=-1)(
            delayed(compute.compute_1)(comb=param, seed=i, index=i, sti=True, maxrate=maxrate, sig=sig, sti_type=sti_type, 
                                    video=False, save_load=save_load1)
            for i in range(n_repeat)
        )
    # 提取所有 spk_rate 并堆叠
    spk_rate0_all = np.stack([r['spk_rate'] for r in result0], axis=0)  # shape: (n_repeat, Nx, Ny, T)
    spk_rate1_all = np.stack([r['spk_rate'] for r in result1], axis=0)

    # 在第一个维度(realization)取平均
    spk_rate0_mean = np.mean(spk_rate0_all, axis=0)  # shape: (Nx, Ny, T)
    spk_rate1_mean = np.mean(spk_rate1_all, axis=0)

    # 全场平均
    center_spk_rate0 = np.mean(spk_rate0_mean, axis=(0, 1))  # shape: (T,)
    center_spk_rate1 = np.mean(spk_rate1_mean, axis=(0, 1))  # shape: (T,)

    # 时间再平均
    center_spk_rate0_tmean = np.mean(center_spk_rate0)
    center_spk_rate1_tmean = np.mean(center_spk_rate1)

    ratio = center_spk_rate1_tmean/center_spk_rate0_tmean
    diff = center_spk_rate1_tmean-center_spk_rate0_tmean
    return ratio, diff

from math import ceil, sqrt
# middle 4 point different sig scane
def receptive_field2(param, n_repeat, plot=False, 
                     video0=False, video1=False, maxrate=1000, sti_type='uniform',
                     save_load0=False, save_load1=False):
    # max_sig = ceil(31.5*sqrt(2))
    max_sig = ceil(31.5)
    sigs = np.arange(0, max_sig + 1, 1)
    ratios = []
    diffs = []
    for sig in sigs:
        ratio, diff = receptive_field_repeat2(param, n_repeat, plot=plot, 
                                              video0=video0, video1=video1, maxrate=maxrate, sig=sig, sti_type=sti_type,
                                              save_load0=save_load0, save_load1=save_load1)
        ratios.append(ratio)
        diffs.append(diff)
    return ratios, diffs, sigs

# exam whole field different sig scane
def receptive_field3(param, n_repeat, plot=False, 
                     video0=False, video1=False, maxrate=1000, sti_type='uniform',
                     save_load0=False, save_load1=False):
    # max_sig = ceil(31.5*sqrt(2))
    max_sig = ceil(31.5)
    sigs = np.arange(0, max_sig + 1, 1)
    ratios = []
    diffs = []
    for sig in sigs:
        ratio, diff = receptive_field_repeat3(param, n_repeat, plot=plot, 
                                              video0=video0, video1=video1, maxrate=maxrate, sig=sig, sti_type=sti_type,
                                              save_load0=save_load0, save_load1=save_load1)
        ratios.append(ratio)
        diffs.append(diff)
    return ratios, diffs, sigs

#%% LFP
# compute 1 area centre point LFP, and output FFT
def LFP_1area(param, maxrate=2000, sig=5, dt=0.1, plot=True):
    ie_r_e1, ie_r_i1 = param
    common_path = f're{ie_r_e1:.4f}_ri{ie_r_i1:.4f}'
    result = compute.compute_1(comb=param, sti=True, maxrate=maxrate, sig=sig, sti_type='Uniform', video=True)
    LFP = result['data'].a1.ge.LFP
    save_path = f'{LFP_dir}/1area_FFT_{sig}_{common_path}.eps'
    freqs, power = mya.analyze_LFP_fft(LFP, dt=dt, plot=plot, save_path=save_path)
    return freqs, power

# compute 2 area take 1st layer's centre point LFP, and output FFT
def LFP_2area(param, maxrate=2000, sig=5, dt=0.1, plot=True):
    ie_r_e1, ie_r_i1, ie_r_e2, ie_r_i2 = param
    # common_title = (rf'$\zeta^{{E1}}$: {ie_r_e1:.4f}, '
    #                 rf'$\zeta^{{I1}}$: {ie_r_i1:.4f}, '
    #                 rf'$\zeta^{{E2}}$: {ie_r_e2:.4f}, '
    #                 rf'$\zeta^{{I2}}$: {ie_r_i2:.4f}')
    common_path = f're1{ie_r_e1:.4f}_ri1{ie_r_i1:.4f}_re2{ie_r_e2:.4f}_ri2{ie_r_i2:.4f}'
    result = compute.compute_2(comb=param, sti=True, maxrate=maxrate, sig=sig, sti_type='Uniform', video=True)
    LFP = result['data'].a1.ge.LFP
    save_path = f'{LFP_dir}/2area_FFT_{sig}_{common_path}.eps'
    freqs, power = mya.analyze_LFP_fft(LFP, dt=dt, plot=plot, save_path=save_path)
    return freqs, power

# repeat computing 1 area FFT of LFP, output beta band and gamma band spectrum
def LFP_1area_repeat(param, n_repeat=64, maxrate=500, sig=5, sti_type='Uniform', dt=0.1, plot=True, video=True, save_load=False):
    ie_r_e1, ie_r_i1 = param
    common_path = f're{ie_r_e1:.4f}_ri{ie_r_i1:.4f}'
    save_path_beta = f'{LFP_dir}/beta_1area_FFT_{sig}_{common_path}_{n_repeat}.eps'
    save_path_gamma = f'{LFP_dir}/gamma_1area_FFT_{sig}_{common_path}_{n_repeat}.eps'
    if video:
        results = Parallel(n_jobs=-1)(
            delayed(compute.compute_1)(comb=param, seed=i, index=i, sti=True, maxrate=maxrate, sig=sig, 
                                       sti_type=sti_type, video=(i==0), save_load=save_load)
            for i in range(n_repeat)
        )
    else:
        results = Parallel(n_jobs=-1)(
            delayed(compute.compute_1)(comb=param, seed=i, index=i, sti=True, maxrate=maxrate, sig=sig, 
                                       sti_type=sti_type, video=False, save_load=save_load)
            for i in range(n_repeat)
        )
    # 提取所有LFP
    LFP_list = [r['data'].a1.ge.LFP for r in results]
    # 计算所有频谱
    fft_results = [mya.analyze_LFP_fft(LFP, dt=dt, plot=False) for LFP in LFP_list]
    freqs = fft_results[0][0]
    powers = np.array([fr[1] for fr in fft_results])
    mean_power = np.mean(powers, axis=0)

    # 画平均频谱
    # beta
    if plot:
        plt.figure(figsize=(6,4))
        plt.plot(freqs, mean_power, label='Mean Power')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum without feedback')
        plt.grid(True)
        plt.xlim(15, 30)
        plt.legend()
        x_min, x_max = plt.xlim()
        mask = (freqs >= x_min) & (freqs <= x_max)
        if np.any(mask):
            plt.ylim(np.min(mean_power[mask]), np.max(mean_power[mask]))
        if save_path_beta:
            plt.savefig(save_path_beta, dpi=300, bbox_inches='tight')
        plt.close()
    # gamma
    if plot:
        plt.figure(figsize=(6,4))
        plt.plot(freqs, mean_power, label='Mean Power')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum without feedback')
        plt.grid(True)
        plt.xlim(30, 80)
        plt.legend()
        x_min, x_max = plt.xlim()
        mask = (freqs >= x_min) & (freqs <= x_max)
        if np.any(mask):
            plt.ylim(np.min(mean_power[mask]), np.max(mean_power[mask]))
        if save_path_gamma:
            plt.savefig(save_path_gamma, dpi=300, bbox_inches='tight')
        plt.close()
    return freqs, mean_power

# repeat computing 2 area FFT of LFP, output beta band and gamma band spectrum
def LFP_2area_repeat(param, n_repeat=64, maxrate=500, sig=5, sti_type='Uniform', dt=0.1, plot=True, video=True, save_load=False):
    ie_r_e1, ie_r_i1, ie_r_e2, ie_r_i2 = param
    common_path = f're1{ie_r_e1:.4f}_ri1{ie_r_i1:.4f}_re2{ie_r_e2:.4f}_ri2{ie_r_i2:.4f}'
    save_path_beta = f'{LFP_dir}/beta_2area_FFT_{sig}_{common_path}_{n_repeat}.eps'
    save_path_gamma = f'{LFP_dir}/gamma_2area_FFT_{sig}_{common_path}_{n_repeat}.eps'
    if video:
        results = Parallel(n_jobs=-1)(
            delayed(compute.compute_2)(comb=param, seed=i, index=i, sti=True, maxrate=maxrate, sig=sig, 
                                       sti_type=sti_type, video=(i==0), save_load=save_load)
            for i in range(n_repeat)
        )
    else:
        results = Parallel(n_jobs=-1)(
            delayed(compute.compute_2)(comb=param, seed=i, index=i, sti=True, maxrate=maxrate, sig=sig, 
                                       sti_type=sti_type, video=False, save_load=save_load)
            for i in range(n_repeat)
        )
    # 提取所有LFP
    LFP_list = [r['data'].a1.ge.LFP for r in results]
    # 计算所有频谱
    fft_results = [mya.analyze_LFP_fft(LFP, dt=dt, plot=False) for LFP in LFP_list]
    freqs = fft_results[0][0]
    powers = np.array([fr[1] for fr in fft_results])
    mean_power = np.mean(powers, axis=0)

    # 画平均频谱
    # beta
    if plot:
        plt.figure(figsize=(6,4))
        plt.plot(freqs, mean_power, label='Mean Power')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum with feedback')
        plt.grid(True)
        plt.xlim(15, 30)
        plt.legend()
        x_min, x_max = plt.xlim()
        mask = (freqs >= x_min) & (freqs <= x_max)
        if np.any(mask):
            plt.ylim(np.min(mean_power[mask]), np.max(mean_power[mask]))
        if save_path_beta:
            plt.savefig(save_path_beta, dpi=300, bbox_inches='tight')
        plt.close()
    if plot:
        plt.figure(figsize=(6,4))
        plt.plot(freqs, mean_power, label='Mean Power')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum with feedback')
        plt.grid(True)
        plt.xlim(30, 80)
        plt.legend()
        x_min, x_max = plt.xlim()
        mask = (freqs >= x_min) & (freqs <= x_max)
        if np.any(mask):
            plt.ylim(np.min(mean_power[mask]), np.max(mean_power[mask]))
        if save_path_gamma:
            plt.savefig(save_path_gamma, dpi=300, bbox_inches='tight')
        plt.close()
    return freqs, mean_power

# exam middle point LFP (FFT)
def LFP_diff_repeat(param1, param2, n_repeat=64, maxrate=500, sig=5, sti_type='Uniform', dt=0.1,
                    plot=True, video=True, save_load=False):
    ie_r_e1, ie_r_i1, ie_r_e2, ie_r_i2 = param2
    common_path = f're1{ie_r_e1:.4f}_ri1{ie_r_i1:.4f}_re2{ie_r_e2:.4f}_ri2{ie_r_i2:.4f}'
    save_path_beta = f'{LFP_dir}/beta_diff_FFT_{sig}_{common_path}_{n_repeat}.eps'
    save_path_gamma = f'{LFP_dir}/gamma_diff_FFT_{sig}_{common_path}_{n_repeat}.eps'
    freqs1, mean_power1 = LFP_1area_repeat(param=param1, n_repeat=n_repeat, maxrate=maxrate, sig=sig,
                                           sti_type=sti_type, dt=dt, plot=plot, video=video, save_load=save_load)
    freqs2, mean_power2 = LFP_2area_repeat(param=param2, n_repeat=n_repeat, maxrate=maxrate, sig=sig,
                                           sti_type=sti_type, dt=dt, plot=plot, video=video, save_load=save_load)
    freqs_diff = freqs2
    mean_power_diff = mean_power2-mean_power1
    # beta
    if plot:
        plt.figure(figsize=(6,4))
        plt.plot(freqs_diff, mean_power_diff, label='Mean Power')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum difference')
        plt.grid(True)
        plt.xlim(15, 30)
        plt.legend()
        x_min, x_max = plt.xlim()
        mask = (freqs_diff >= x_min) & (freqs_diff <= x_max)
        if np.any(mask):
            plt.ylim(np.min(mean_power_diff[mask]), np.max(mean_power_diff[mask]))
        if save_path_beta:
            plt.savefig(save_path_beta, dpi=300, bbox_inches='tight')
        plt.close()
    # gamma
    if plot:
        plt.figure(figsize=(6,4))
        plt.plot(freqs_diff, mean_power_diff, label='Mean Power')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum difference')
        plt.grid(True)
        plt.xlim(30, 80)
        plt.legend()
        x_min, x_max = plt.xlim()
        mask = (freqs_diff >= x_min) & (freqs_diff <= x_max)
        if np.any(mask):
            plt.ylim(np.min(mean_power_diff[mask]), np.max(mean_power_diff[mask]))
        if save_path_gamma:
            plt.savefig(save_path_gamma, dpi=300, bbox_inches='tight')
        plt.close()
    return freqs1, mean_power1, freqs2, mean_power2, freqs_diff, mean_power_diff
# 1,2area and diff, compare different sig
def draw_LFP_FFT_compare(param1, param2, n_repeat=64, sigs=[0,5,10,15,20,25], maxrate=500, dt=0.1, 
                         plot=True, video=True, save_load=False):
    
    ie_r_e1, ie_r_i1, ie_r_e2, ie_r_i2 = param2
    common_path1 = f're1{ie_r_e1:.4f}_ri1{ie_r_i1:.4f}'
    common_path2 = f're1{ie_r_e1:.4f}_ri1{ie_r_i1:.4f}_re2{ie_r_e2:.4f}_ri2{ie_r_i2:.4f}'

    save_path_beta1 = f'{LFP_dir}/beta_1area_FFT_{common_path1}_{n_repeat}.eps'
    save_path_beta2 = f'{LFP_dir}/beta_2area_FFT_{common_path2}_{n_repeat}.eps'
    save_path_betad = f'{LFP_dir}/beta_diff_FFT_{common_path2}_{n_repeat}.eps'
    save_path_gamma1 = f'{LFP_dir}/gamma_1area_FFT_{common_path1}_{n_repeat}.eps'
    save_path_gamma2 = f'{LFP_dir}/gamma_2area_FFT_{common_path2}_{n_repeat}.eps'
    save_path_gammad = f'{LFP_dir}/gamma_diff_FFT_{common_path2}_{n_repeat}.eps'

    results_1area = []
    results_2area = []
    results_diff = []

    for sig in sigs:
        freqs1, mean_power1, freqs2, mean_power2, freqs_diff, mean_power_diff = LFP_diff_repeat(
            param1=param1, param2=param2, n_repeat=n_repeat, maxrate=maxrate, sig=sig, dt=dt, plot=plot, video=video, save_load=save_load
            )
        results_1area.append((sig, freqs1, mean_power1))
        results_2area.append((sig, freqs2, mean_power2))
        results_diff.append((sig, freqs1, mean_power_diff))

    # 1area
    if plot:
    # beta
        plt.figure(figsize=(6,4))
        for sig, freqs, power in results_1area:
            plt.plot(freqs, power, label=f'sig={sig}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum without feedback')
        plt.grid(True)
        plt.xlim(15, 30)
        plt.legend()
        x_min, x_max = plt.xlim()
        all_masked_power = []
        for sig, freqs, power in results_1area:
            mask = (freqs >= x_min) & (freqs <= x_max)
            if np.any(mask):
                all_masked_power.append(power[mask])
        if all_masked_power:
            all_masked_power = np.concatenate(all_masked_power)
            plt.ylim(np.min(all_masked_power), np.max(all_masked_power))
        if save_path_beta1:
            plt.savefig(save_path_beta1, dpi=300, bbox_inches='tight')
        plt.close()
    # gamma
        plt.figure(figsize=(6,4))
        for sig, freqs, power in results_1area:
            plt.plot(freqs, power, label=f'sig={sig}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum without feedback')
        plt.grid(True)
        plt.xlim(30, 80)
        plt.legend()
        x_min, x_max = plt.xlim()
        all_masked_power = []
        for sig, freqs, power in results_1area:
            mask = (freqs >= x_min) & (freqs <= x_max)
            if np.any(mask):
                all_masked_power.append(power[mask])
        if all_masked_power:
            all_masked_power = np.concatenate(all_masked_power)
            plt.ylim(np.min(all_masked_power), np.max(all_masked_power))
        if save_path_gamma1:
            plt.savefig(save_path_gamma1, dpi=300, bbox_inches='tight')
        plt.close()

    # 2area
    # beta
        plt.figure(figsize=(6,4))
        for sig, freqs, power in results_2area:
            plt.plot(freqs, power, label=f'sig={sig}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum with feedback')
        plt.grid(True)
        plt.xlim(15, 30)
        plt.legend()
        x_min, x_max = plt.xlim()
        all_masked_power = []
        for sig, freqs, power in results_2area:
            mask = (freqs >= x_min) & (freqs <= x_max)
            if np.any(mask):
                all_masked_power.append(power[mask])
        if all_masked_power:
            all_masked_power = np.concatenate(all_masked_power)
            plt.ylim(np.min(all_masked_power), np.max(all_masked_power))
        if save_path_beta2:
            plt.savefig(save_path_beta2, dpi=300, bbox_inches='tight')
        plt.close()
    # gamma
        plt.figure(figsize=(6,4))
        for sig, freqs, power in results_2area:
            plt.plot(freqs, power, label=f'sig={sig}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum with feedback')
        plt.grid(True)
        plt.xlim(30, 80)
        plt.legend()
        x_min, x_max = plt.xlim()
        all_masked_power = []
        for sig, freqs, power in results_2area:
            mask = (freqs >= x_min) & (freqs <= x_max)
            if np.any(mask):
                all_masked_power.append(power[mask])
        if all_masked_power:
            all_masked_power = np.concatenate(all_masked_power)
            plt.ylim(np.min(all_masked_power), np.max(all_masked_power))
        if save_path_gamma2:
            plt.savefig(save_path_gamma2, dpi=300, bbox_inches='tight')
        plt.close()

    # diff
    # beta
        plt.figure(figsize=(6,4))
        for sig, freqs, power in results_diff:
            plt.plot(freqs, power, label=f'sig={sig}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum difference')
        plt.grid(True)
        plt.xlim(15, 30)
        plt.legend()
        x_min, x_max = plt.xlim()
        all_masked_power = []
        for sig, freqs, power in results_diff:
            mask = (freqs >= x_min) & (freqs <= x_max)
            if np.any(mask):
                all_masked_power.append(power[mask])
        if all_masked_power:
            all_masked_power = np.concatenate(all_masked_power)
            plt.ylim(np.min(all_masked_power), np.max(all_masked_power))
        if save_path_betad:
            plt.savefig(save_path_betad, dpi=300, bbox_inches='tight')
        plt.close()
    # gamma
        plt.figure(figsize=(6,4))
        for sig, freqs, power in results_diff:
            plt.plot(freqs, power, label=f'sig={sig}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum difference')
        plt.grid(True)
        plt.xlim(30, 80)
        plt.legend()
        x_min, x_max = plt.xlim()
        all_masked_power = []
        for sig, freqs, power in results_diff:
            mask = (freqs >= x_min) & (freqs <= x_max)
            if np.any(mask):
                all_masked_power.append(power[mask])
        if all_masked_power:
            all_masked_power = np.concatenate(all_masked_power)
            plt.ylim(np.min(all_masked_power), np.max(all_masked_power))
        if save_path_gammad:
            plt.savefig(save_path_gammad, dpi=300, bbox_inches='tight')
        plt.close()
    

#%% Execution area
try:
    send_email.send_email('begin running', 'ie_search.main running')
    #%% test
    # param = (1.8512390285440765, 2.399131446733395)
    # compute.compute_1(comb=param, seed=10, index=0, sti=True, video=True, save_load=False)
    # # ie-ratio 写错时，e-i对调了：
    # param1 = (1.795670364314891, 2.449990451446889)
    # param2 = (1.795670364314891, 2.449990451446889, 1.8512390285440765, 2.399131446733395)
    # # 改正后：
    # param1 = (2.449990451446889, 1.795670364314891)
    # param2 = (2.449990451446889, 1.795670364314891, 2.399131446733395, 1.8512390285440765)
    #%% evalutionary search
    # evalution_search(compute=False)

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
    # receptive_field(param=param, maxrate=5000)

    #%% repeat receptive field
    # # first layer
    # param = (1.795670364314891, 2.449990451446889)
    # receptive_field_repeat(param=param, n_repeat=128, maxrate=1000, plot=True, video1=True)
    # # second layer
    # param = (1.8870084212830673, 2.3990240481749168)
    # receptive_field_repeat(param=param, n_repeat=128, maxrate=1000, plot=True, video1=True)

    #%% search receptive field
    # result = find_max_min_receptive_field(n_repeat=64, maxrate=1000)
    range_path = f'{state_dir}/critical_ellipse.file'
    result = find_receptive_field_distribution_in_range(n_repeat=64, 
                                                        range_path=range_path, 
                                                        maxrate=1000, 
                                                        n_sample=1000)

    #%% repeat 2 area computation recetive field (MSD, pdx) -> 2area/
    # param = (1.795670364314891, 2.449990451446889, 1.8512390285440765, 2.399131446733395)
    # pick_parameters_and_repeat_compute2(param=param,
    #                                     n_repeat=128,
    #                                     video=True)

    #%% load and draw receptive field
    # # first layer
    # param = (1.795670364314891, 2.449990451446889)
    # load_and_draw_receptive_field(param, maxrate=5000, n_repeat=64)
    # # second layer
    # param = (1.8512390285440765, 2.399131446733395)
    # load_and_draw_receptive_field(param, maxrate=5000, n_repeat=64)

    #%% check r_rf(maxrate)
    # check_r_rf_maxrate(param = (1.8512390285440765, 2.399131446733395),
    #                    seq_maxrate = [0, 1, 10, 100, 200, 500, 1000, 2000, 5000])

    #%% receptive field 2 (exam middle 4 point firing rate while scane stimuli size)
    # def draw_receptive_field2(param, n_repeat, maxrate=1000):
    #     ratios, diffs, sigs = receptive_field2(param, n_repeat, plot=False, 
    #                                      video0=False, video1=False, maxrate=maxrate, sti_type='uniform',
    #                                      save_load0=False, save_load1=False)
    #     ie_r_e1, ie_r_i1 = param
    #     common_path = f're{ie_r_e1:.4f}_ri{ie_r_i1:.4f}'

    #     save_pathr = f'{recfield_dir}/middle_zratio{n_repeat}_{maxrate}fr_ext{common_path}.eps'
    #     save_pathd = f'{recfield_dir}/middle_zdiff{n_repeat}_{maxrate}fr_ext{common_path}.eps'
        
    #     plt.figure(figsize=(5,5))
    #     plt.plot(sigs, ratios, 'o-')
    #     plt.xlabel('Stimuli size')
    #     plt.ylabel('Centre firing rate ratio')
    #     plt.title('Centre firing rate ratio vs. stimuli size')
    #     plt.savefig(save_pathr, dpi=600, format='eps')

    #     plt.figure(figsize=(5,5))
    #     plt.plot(sigs, diffs, 'o-')
    #     plt.xlabel('Stimuli size')
    #     plt.ylabel('Centre firing rate difference')
    #     plt.title('Centre firing rate ratio vs. stimuli size')
    #     plt.savefig(save_pathd, dpi=600, format='eps')
    # param = (1.795670364314891, 2.449990451446889)
    # draw_receptive_field2(param=param, n_repeat=64)

    #%% receptive field 3 (exam whole field firing rate while scane stimuli size)
    # def draw_receptive_field3(param, n_repeat, maxrate=1000):
    #     ratios, diffs, sigs = receptive_field3(param, n_repeat, plot=False, 
    #                                      video0=False, video1=False, maxrate=maxrate, sti_type='uniform',
    #                                      save_load0=False, save_load1=False)
    #     ie_r_e1, ie_r_i1 = param
    #     common_path = f're{ie_r_e1:.4f}_ri{ie_r_i1:.4f}'

    #     save_pathr = f'{recfield_dir}/field_zratio{n_repeat}_{maxrate}fr_ext{common_path}.eps'
    #     save_pathd = f'{recfield_dir}/field_zdiff{n_repeat}_{maxrate}fr_ext{common_path}.eps'
        
    #     plt.figure(figsize=(5,5))
    #     plt.plot(sigs, ratios, 'o-')
    #     plt.xlabel('Stimuli size')
    #     plt.ylabel('Whole field mean firing rate ratio')
    #     plt.title('Whole field mean firing rate ratio vs. stimuli size')
    #     plt.savefig(save_pathr, dpi=600, format='eps')

    #     plt.figure(figsize=(5,5))
    #     plt.plot(sigs, diffs, 'o-')
    #     plt.xlabel('Stimuli size')
    #     plt.ylabel('Whole field mean firing rate difference')
    #     plt.title('Whole field mean firing rate difference vs. stimuli size')
    #     plt.savefig(save_pathd, dpi=600, format='eps')
    # param = (1.795670364314891, 2.449990451446889)
    # draw_receptive_field3(param=param, n_repeat=64)

    #%% LFP
    # def draw_LFP_FFT_2area():
    #     param1 = (1.795670364314891, 2.449990451446889)
    #     param2 = (1.795670364314891, 2.449990451446889, 1.8512390285440765, 2.399131446733395)
    #     sig=0
    #     LFP_1area(param=param1,maxrate=500,sig=sig,dt=0.1,plot=True)
    #     LFP_2area(param=param2,maxrate=500,sig=sig,dt=0.1,plot=True)
    # # draw_LFP_FFT_2area()
    # def draw_LFP_FFT_1area_repeat(n_repeat=64,sig=0):
    #     param = (1.795670364314891, 2.449990451446889)
    #     LFP_1area_repeat(param=param,n_repeat=64,maxrate=500,sig=sig,dt=0.1,plot=True,video=True,save_load=False)
    # def draw_LFP_FFT_2area_repeat(n_repeat=64,sig=0):
    #     param = (1.795670364314891, 2.449990451446889, 1.8512390285440765, 2.399131446733395)
    #     LFP_2area_repeat(param=param,n_repeat=64,maxrate=500,sig=sig,dt=0.1,plot=True,video=True,save_load=False)
    # # draw_LFP_FFT_1area_repeat()
    # # draw_LFP_FFT_2area_repeat()
    # def draw_LFP_FFT_diff_repeat(n_repeat=128):
    #     param1 = (1.795670364314891, 2.449990451446889)
    #     param2 = (1.795670364314891, 2.449990451446889, 1.8512390285440765, 2.399131446733395)
    #     for sig in [0,5,10,15,20,25]:
    #         LFP_diff_repeat(param1=param1, param2=param2, n_repeat=n_repeat, sig=sig)
    # draw_LFP_FFT_diff_repeat()

    # param1 = (1.795670364314891, 2.449990451446889)
    # param2 = (1.795670364314891, 2.449990451446889, 1.8512390285440765, 2.399131446733395)
    # draw_LFP_FFT_compare(param1=param1, param2=param2)





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
# %%
