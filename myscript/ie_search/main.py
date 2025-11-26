import brian2.numpy_ as np
from brian2.only import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
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

# 会画出两层作用在一起时两层分别的MDS和pdx
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
    save_path = f'{graph_dir}/evaluation.eps'
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
# def sample_in_ellipse(mean, cov, conf_level, n_samples):
#     dim = len(mean)
#     threshold = np.sqrt(chi2.ppf(conf_level, df=dim))
#     samples = []
#     while len(samples) < n_samples:
#         # 在包络盒内均匀采样
#         box_min = mean - 2*np.sqrt(np.diag(cov))
#         box_max = mean + 2*np.sqrt(np.diag(cov))
#         point = np.random.uniform(box_min, box_max)
#         # 判断是否在椭圆内
#         dist = np.sqrt((point-mean) @ np.linalg.inv(cov) @ (point-mean).T)
#         if dist < threshold:
#             samples.append(point)
#     return np.array(samples)

def sample_in_ellipse(mean, cov, conf_level, n_samples):
    dim = len(mean)
    threshold = np.sqrt(chi2.ppf(conf_level, df=dim))
    
    # 特征分解
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    samples = []
    while len(samples) < n_samples:
        # 在单位圆内均匀采样
        u = np.random.uniform(0, 1, dim)
        # 转换为球面上的均匀分布
        radius = np.sqrt(u[0]) * threshold  # 确保在椭圆内均匀分布
        angles = 2 * np.pi * u[1]
        
        # 转换为椭圆坐标
        point_sphere = np.array([radius * np.cos(angles), radius * np.sin(angles)])
        
        # 变换到实际的椭圆
        scaling = np.diag(np.sqrt(eigvals))
        point_ellipse = eigvecs @ scaling @ point_sphere + mean
        
        samples.append(point_ellipse)
    
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

# compute receptive field radius and alpha with repeat realizaiton
def rf_and_alpha_repeat(param, n_repeat, plot=False,
                        video0=False, video1=False, maxrate=1000,
                        save_load0=False, save_load1=False):
    # without stimuli
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
    # with stimuli
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
    # 计算rf
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
                               plot=plot)['r_rf']
    # 合并msd
    msds = np.stack([r['msd'] for r in result0])
    jump_interval = result0[0]['jump_interval']
    msd_mean = np.mean(msds, axis=0)
    msd_std = np.std(msds, axis=0)
    # pdx合并
    all_pdx = np.concatenate([r['pdx'] for r in result0])
    # 计算alpha，判断是否临界
    motion_critical, info = utils.is_critical_state(msd=msd_mean, 
                                                    jump_interval=jump_interval, 
                                                    pdx=all_pdx)
    alpha = info['alpha']
    critical = motion_critical
    return {'r_rf': r_rf, 'alpha': alpha, 'critical': critical}

# 已被find_receptive_field_distribution_in_range替代
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

# 上面那个的完全上位
def find_receptive_field_distribution_in_range(n_repeat, range_path, maxrate=1000, 
                                               n_sample=1000, fit=False):
    # 读取椭圆参数
    with open(range_path, 'rb') as file:
        ellipse_info = pickle.load(file)
    mean = ellipse_info['mean']
    cov = ellipse_info['cov']
    conf_level = ellipse_info.get('conf_level', 0.99)
    # 转换椭圆坐标(绘图用)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2 * np.sqrt(vals * chi2.ppf(conf_level, df=2))

    # 椭圆内采样参数
    params = sample_in_ellipse(mean, cov, conf_level, n_sample)
    params = [tuple(p) for p in params]

    # 尝试读取已有历史
    rf_history_path = f'{state_dir}/rf_landscape_{n_sample}.file'
    r_rf_history = []
    computed_params = set()
    if os.path.exists(rf_history_path):
        with open(rf_history_path, 'rb') as file:
            r_rf_history = pickle.load(file)
        # 已经算过的参数
        for info in r_rf_history:
            param = info['param']
            computed_params.add(tuple(param))

    max_val = -np.inf
    min_val = np.inf
    max_param = None
    min_param = None
    rf_list = []
    loop_total = len(params)
    loop_num = 0

    for param in params:
        # # 如果想看此轮采样的点中算了多少个，用这个
        # loop_num += 1
        param_tuple = (param[1], param[0])
        try:
            field = rf_and_alpha_repeat(param=param_tuple, 
                                        n_repeat=n_repeat, 
                                        maxrate=maxrate, 
                                        plot=False)
            r_rf = field['r_rf']
            alpha = field['alpha']
            critical = field['critical']
        except Exception as e:
            print(f"参数 {param_tuple} 计算失败: {e}")
            send_email.send_email('Progress', f"参数 {param_tuple} 计算失败: {e}")
            continue
        
        # 修改1: 当alpha>1.5时跳过当前参数
        if alpha > 1.5:
            print(f"参数 {param_tuple} alpha={alpha:.3f}>1.5,跳过")
            send_email.send_email('Progress', f"参数 {param_tuple} alpha={alpha:.3f}>1.5,跳过")
            continue

        rf_list.append((param_tuple, r_rf))
        if r_rf is not None:
            if r_rf > max_val:
                max_val = r_rf
                max_param = param_tuple
            if r_rf < min_val:
                min_val = r_rf
                min_param = param_tuple
        
        info = {'param': param_tuple, 'r_rf': r_rf, 'alpha': alpha, 'critical': critical}
        r_rf_history.append(info)

        # 如果想看r_rf_history这个量有多少组，即总共计算了多少组，用这个
        loop_num = len(r_rf_history)

        # 实时保存
        with open(f'{state_dir}/rf_landscape_{n_sample}.file', 'wb') as file:
            pickle.dump(r_rf_history, file)

        # 画地形图
        x = [info['param'][0] for info in r_rf_history]
        y = [info['param'][1] for info in r_rf_history]
        z_rf = [info['r_rf'] for info in r_rf_history]
        z_alpha = [info['alpha'] for info in r_rf_history]
        # 创建椭圆对象
        ellipse1 = Ellipse(xy=(mean[1], mean[0]), width=width, height=height, angle=90-theta, 
                            edgecolor='blue', facecolor='none', lw=2, 
                            label='Ellipse Boundary', zorder=4)
        ellipse2 = Ellipse(xy=(mean[1], mean[0]), width=width, height=height, angle=90-theta, 
                            edgecolor='blue', facecolor='none', lw=2, 
                            label='Ellipse Boundary', zorder=4)

        fig, axs = plt.subplots(1, 2, figsize=(7, 3.5))
        # 左：r_rf
        sc1 = axs[0].scatter(x, y, c=z_rf, cmap='viridis', s=60)
        axs[0].add_patch(ellipse1)
        axs[0].set_xlabel(r'$\zeta^{\rm E}$', fontsize=10)
        axs[0].set_ylabel(r'$\zeta^{\rm I}$', fontsize=10)
        axs[0].set_title('Receptive Field Radius', fontsize=11)
        axs[0].tick_params(axis='both', labelsize=10)
        cbar1 = plt.colorbar(sc1, ax=axs[0])
        cbar1.set_label('Receptive Field', fontsize=10)
        cbar1.ax.tick_params(labelsize=10)
        # axs[0].legend(fontsize=9)
        # 右：alpha
        sc2 = axs[1].scatter(x, y, c=z_alpha, cmap='plasma', s=60)
        axs[1].add_patch(ellipse2)
        axs[1].set_xlabel(r'$\zeta^{\rm E}$', fontsize=10)
        axs[1].set_ylabel(r'$\zeta^{\rm I}$', fontsize=10)
        axs[1].set_title(r'$\alpha$', fontsize=11)
        axs[1].tick_params(axis='both', labelsize=10)
        cbar2 = plt.colorbar(sc2, ax=axs[1])
        cbar2.set_label(r'$\alpha$', fontsize=10)
        cbar2.ax.tick_params(labelsize=10)
        # axs[1].legend(fontsize=9)
        plt.tight_layout(pad=1.0)
        plt.savefig(f'{graph_dir}/rf_landscape_{n_sample}.eps', dpi=300)
        plt.close()

        # 画3维地形图
        # 提取有效数据
        x, y, z = [], [], []
        x_bad, y_bad = [], []
        for info in r_rf_history:
            param = info['param']
            r_rf = info['r_rf']
            # 自动舍弃无效点
            if r_rf is None or not isinstance(r_rf, (int, float)) or not (r_rf == r_rf):  # 排除None和NaN
                x_bad.append(param[0])
                y_bad.append(param[1])
                continue
            x.append(param[0])
            y.append(param[1])
            z.append(r_rf)
        
        # 转为numpy数组，避免幂运算报错
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=40)

        if fit and len(x)>6:
            # 曲面拟合
            coeffs = fit_quadratic_surface(x, y, z)

            # 计算拟合值和R²
            X_design = np.column_stack([x**2, y**2, x*y, x, y, np.ones_like(x)])
            z_fit = X_design @ coeffs
            ss_res = np.sum((z - z_fit) ** 2)
            ss_tot = np.sum((z - np.mean(z)) ** 2)
            r2 = 1 - ss_res / ss_tot

            # 生成网格用于画曲面
            x_grid = np.linspace(np.min(x), np.max(x), 50)
            y_grid = np.linspace(np.min(y), np.max(y), 50)
            X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
            Z_grid = (coeffs[0]*X_grid**2 + coeffs[1]*Y_grid**2 + coeffs[2]*X_grid*Y_grid +
                    coeffs[3]*X_grid + coeffs[4]*Y_grid + coeffs[5])

            # 画三维地形图
            surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='coolwarm', alpha=0.5, linewidth=0, antialiased=True)
            ax.set_title(f'Receptive Field 3D Landscape with Quadratic Surface Fit, $R^2$={r2:.3f}')
        else:
            ax.set_title('Receptive Field 3D Landscape')

        # 获取当前z轴范围
        if len(z) > 0:
            zmin, zmax = np.min(z), np.max(z)
            dz = zmax - zmin
            z_bad = np.full(len(x_bad), zmin - 0.05*dz if dz > 0 else zmin - 1)
            # 在最低面下方一点画无效点
            ax.scatter(x_bad, y_bad, z_bad, c='red', marker='x', s=50, label='Invalid r_rf')
            # 可选：加图例
            ax.legend()
        
        ax.set_xlabel(r'$\zeta^{\rm E}$')
        ax.set_ylabel(r'$\zeta^{\rm I}$')
        ax.set_zlabel('Receptive Field')
        fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10, label='Receptive Field')
        plt.tight_layout()
        plt.savefig(f'{graph_dir}/rf_landscape_3d_{n_sample}.png', dpi=300)
        # 俯视视角
        ax.view_init(elev=90, azim=-90)
        plt.tight_layout()
        plt.savefig(f'{graph_dir}/rf_landscape_3d_{n_sample}_top.png', dpi=300)

        plt.close()

        # 发邮件报告进度
        send_email.send_email(
            'Progress',
            f'Complete {loop_num} in {loop_total}, \n parameter: {param_tuple}, \n r_rf: {r_rf}. \n Now, \n max r_rf: {max_val}, \n max parameter: {max_param}, \n min r_rf: {min_val}, \n min parameter: {min_param}'
        )

        if loop_num >= n_sample:
            break

    print(f'最大receptive field参数: {max_param}, 最大值: {max_val}')
    print(f'最小receptive field参数: {min_param}, 最小值: {min_val}')

    return {
        'max_param': max_param,
        'max_rf': max_val,
        'min_param': min_param,
        'min_rf': min_val
    }

def fit_quadratic_surface(x, y, z):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    X = np.column_stack([x**2, y**2, x*y, x, y, np.ones_like(x)])
    coeffs, _, _, _ = np.linalg.lstsq(X, z, rcond=None)
    return coeffs

def plot_rf_landscape_3d(n_sample, fit=True):
    # 读取历史文件
    rf_history_path = f'{state_dir}/rf_landscape_{n_sample}.file'
    with open(rf_history_path, 'rb') as file:
        r_rf_history = pickle.load(file)

    # 提取有效数据
    x, y, z = [], [], []
    x_bad, y_bad = [], []
    for info in r_rf_history:
        param = info[0]['param']
        r_rf = info[0]['r_rf_result'][0]['r_rf']
        # 自动舍弃无效点
        if r_rf is None or not isinstance(r_rf, (int, float)) or not (r_rf == r_rf):  # 排除None和NaN
            x_bad.append(param[0])
            y_bad.append(param[1])
            continue
        x.append(param[0])
        y.append(param[1])
        z.append(r_rf)
    
    # 转为numpy数组，避免幂运算报错
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=40)

    if fit and len(x)>6:
        # 曲面拟合
        coeffs = fit_quadratic_surface(x, y, z)

        # 计算拟合值和R²
        X_design = np.column_stack([x**2, y**2, x*y, x, y, np.ones_like(x)])
        z_fit = X_design @ coeffs
        ss_res = np.sum((z - z_fit) ** 2)
        ss_tot = np.sum((z - np.mean(z)) ** 2)
        r2 = 1 - ss_res / ss_tot

        # 生成网格用于画曲面
        x_grid = np.linspace(np.min(x), np.max(x), 50)
        y_grid = np.linspace(np.min(y), np.max(y), 50)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        Z_grid = (coeffs[0]*X_grid**2 + coeffs[1]*Y_grid**2 + coeffs[2]*X_grid*Y_grid +
                coeffs[3]*X_grid + coeffs[4]*Y_grid + coeffs[5])

        # 画三维地形图
        surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='coolwarm', alpha=0.5, linewidth=0, antialiased=True)
        ax.set_title(f'Receptive Field 3D Landscape with Quadratic Surface Fit, $R^2$={r2:.3f}')
    else:
        ax.set_title('Receptive Field 3D Landscape')

    # 获取当前z轴范围
    if len(z) > 0:
        zmin, zmax = np.min(z), np.max(z)
        dz = zmax - zmin
        z_bad = np.full(len(x_bad), zmin - 0.05*dz if dz > 0 else zmin - 1)
        # 在最低面下方一点画无效点
        ax.scatter(x_bad, y_bad, z_bad, c='red', marker='x', s=50, label='Invalid r_rf')
        # 可选：加图例
        ax.legend()
    
    ax.set_xlabel(r'$\zeta^{\rm E}$')
    ax.set_ylabel(r'$\zeta^{\rm I}$')
    ax.set_zlabel('Receptive Field')
    fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10, label='Receptive Field')
    plt.tight_layout()
    plt.savefig(f'{graph_dir}/rf_landscape_3d_{n_sample}.png', dpi=300)
    # 俯视视角
    ax.view_init(elev=90, azim=-90)
    plt.tight_layout()
    plt.savefig(f'{graph_dir}/rf_landscape_3d_{n_sample}_top.png', dpi=300)

    plt.close()

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
                            video0=False, video1=False, maxrate=1000, sig=2, sti_type='Uniform',
                            save_load0=False, save_load1=False, le=64, li=32):
    
    if video0:
        result0 = Parallel(n_jobs=-1)(
            delayed(compute.compute_1)(comb=param, seed=i, index=i, sti=False, 
                                       video=(i==0), save_load=save_load0, 
                                       le=le, li=li)
            for i in range(n_repeat)
        )
    else:
        result0 = Parallel(n_jobs=-1)(
            delayed(compute.compute_1)(comb=param, seed=i, index=i, sti=False, 
                                       video=False, save_load=save_load0, 
                                       le=le, li=li)
            for i in range(n_repeat)
        )
    if video1:
        result1 = Parallel(n_jobs=-1)(
            delayed(compute.compute_1)(comb=param, seed=i, index=i, sti=True, 
                                       maxrate=maxrate, sig=sig, sti_type=sti_type, 
                                       video=(i==0), save_load=save_load1, 
                                       le=le, li=li)
            for i in range(n_repeat)
        )
    else:
        result1 = Parallel(n_jobs=-1)(
            delayed(compute.compute_1)(comb=param, seed=i, index=i, sti=True, 
                                       maxrate=maxrate, sig=sig, sti_type=sti_type, 
                                       video=False, save_load=save_load1, 
                                       le=le, li=li)
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

#%% draw receptive field 2 (exam middle 4 point firing rate while scane stimuli size)
def draw_receptive_field2(param, n_repeat, maxrate=1000, le=64, li=32):
    ratios, diffs, sigs = receptive_field2(param, n_repeat, plot=False, 
                                           video0=False, video1=False, 
                                           maxrate=maxrate, sti_type='Uniform',
                                           save_load0=False, save_load1=False, 
                                           le=le, li=li)
    ie_r_e1, ie_r_i1 = param
    common_path = f're{ie_r_e1:.4f}_ri{ie_r_i1:.4f}'

    save_pathr = f'{recfield_dir}/middle_zratio{n_repeat}_{maxrate}fr_ext{common_path}.eps'
    save_pathd = f'{recfield_dir}/middle_zdiff{n_repeat}_{maxrate}fr_ext{common_path}.eps'
    
    plt.figure(figsize=(5,5))
    plt.plot(sigs, ratios, 'o-')
    plt.xlabel('Stimuli size')
    plt.ylabel('Centre firing rate ratio')
    plt.title('Centre firing rate ratio vs. stimuli size')
    plt.savefig(save_pathr, dpi=600, format='eps')

    plt.figure(figsize=(5,5))
    plt.plot(sigs, diffs, 'o-')
    plt.xlabel('Stimuli size')
    plt.ylabel('Centre firing rate difference')
    plt.title('Centre firing rate ratio vs. stimuli size')
    plt.savefig(save_pathd, dpi=600, format='eps')

# exam whole field firing rate (receptive field)
def receptive_field_repeat3(param, n_repeat, plot=False, 
                            video0=False, video1=False, maxrate=1000, sig=2, sti_type='Uniform',
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
                     video0=False, video1=False, 
                     maxrate=1000, sti_type='Uniform',
                     save_load0=False, save_load1=False, 
                     le=64, li=32):
    # max_sig = ceil(31.5*sqrt(2))
    max_sig = ceil((le-1)/2)
    sigs = np.arange(0, max_sig + 1, 1)
    ratios = []
    diffs = []
    for sig in sigs:
        ratio, diff = receptive_field_repeat2(param, n_repeat, plot=plot, 
                                              video0=video0, video1=video1, 
                                              maxrate=maxrate, sig=sig, sti_type=sti_type,
                                              save_load0=save_load0, save_load1=save_load1, 
                                              le=le, li=li)
        ratios.append(ratio)
        diffs.append(diff)
    return ratios, diffs, sigs

# exam whole field different sig scane
def receptive_field3(param, n_repeat, plot=False, 
                     video0=False, video1=False, maxrate=1000, sti_type='Uniform',
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
fft_l = 1
fft_r = 100
# compute 1 area centre point LFP, and output FFT
def LFP_1area(param, maxrate=500, sig=5, dt=0.1, plot=True, video=True):
    ie_r_e1, ie_r_i1 = param
    common_path = f're{ie_r_e1:.4f}_ri{ie_r_i1:.4f}'
    save_path = f'{LFP_dir}/1area_FFT_{sig}_{common_path}.eps'
    result = compute.compute_1(comb=param, sti=True, maxrate=maxrate, sig=sig, sti_type='Uniform', video=video)
    LFP = result['data'].a1.ge.LFP
    freqs, power = mya.analyze_LFP_fft(LFP, dt=dt, plot=plot, save_path=save_path)
    return freqs, power

# compute 2 area take 1st layer's centre point LFP, and output FFT
def LFP_2area(param, maxrate=500, sig=5, dt=0.1, plot=True, video=True):
    ie_r_e1, ie_r_i1, ie_r_e2, ie_r_i2 = param
    # common_title = (rf'$\zeta^{{E1}}$: {ie_r_e1:.4f}, '
    #                 rf'$\zeta^{{I1}}$: {ie_r_i1:.4f}, '
    #                 rf'$\zeta^{{E2}}$: {ie_r_e2:.4f}, '
    #                 rf'$\zeta^{{I2}}$: {ie_r_i2:.4f}')
    common_path = f're1{ie_r_e1:.4f}_ri1{ie_r_i1:.4f}_re2{ie_r_e2:.4f}_ri2{ie_r_i2:.4f}'
    result = compute.compute_2(comb=param, sti=True, maxrate=maxrate, sig=sig, sti_type='Uniform', video=video)
    LFP = result['data'].a1.ge.LFP
    save_path = f'{LFP_dir}/2area_FFT_{sig}_{common_path}.eps'
    freqs, power = mya.analyze_LFP_fft(LFP, dt=dt, plot=plot, save_path=save_path)
    return freqs, power

# repeat computing 1 area FFT of LFP, output beta band and gamma band spectrum
def LFP_1area_repeat(param, n_repeat=64, maxrate=500, sig=5, sti_type='Uniform', dt=0.1, 
                     plot=True, video=True, save_load=False):
    ie_r_e1, ie_r_i1 = param
    common_path = f're{ie_r_e1:.4f}_ri{ie_r_i1:.4f}'
    save_path_beta = f'{LFP_dir}/beta_1area_FFT_{sig}_{common_path}_{n_repeat}.eps'
    save_path_gamma = f'{LFP_dir}/gamma_1area_FFT_{sig}_{common_path}_{n_repeat}.eps'
    save_path = f'{LFP_dir}/whole_1area_FFT_{sig}_{common_path}_{n_repeat}.eps'
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
    if plot:
    # whole
        plt.figure(figsize=(6,4))
        plt.loglog(freqs, mean_power, label='Mean Power')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum without feedback')
        plt.grid(True)
        plt.xlim(fft_l, fft_r)
        plt.legend()
        x_min, x_max = plt.xlim()
        mask = (freqs >= x_min) & (freqs <= x_max)
        if np.any(mask):
            plt.ylim(np.min(mean_power[mask]), np.max(mean_power[mask]))
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    # beta
        plt.figure(figsize=(6,4))
        plt.loglog(freqs, mean_power, label='Mean Power')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum (beta) without feedback')
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
        plt.figure(figsize=(6,4))
        plt.loglog(freqs, mean_power, label='Mean Power')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum (gamma) without feedback')
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
def LFP_2area_repeat(param, n_repeat=64, maxrate=500, sig=5, sti_type='Uniform', dt=0.1, 
                     plot=True, video=True, save_load=False):
    ie_r_e1, ie_r_i1, ie_r_e2, ie_r_i2 = param
    common_path = f're1{ie_r_e1:.4f}_ri1{ie_r_i1:.4f}_re2{ie_r_e2:.4f}_ri2{ie_r_i2:.4f}'
    save_path_beta = f'{LFP_dir}/beta_2area_FFT_{sig}_{common_path}_{n_repeat}.eps'
    save_path_gamma = f'{LFP_dir}/gamma_2area_FFT_{sig}_{common_path}_{n_repeat}.eps'
    save_path = f'{LFP_dir}/whole_2area_FFT_{sig}_{common_path}_{n_repeat}.eps'
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
    if plot:
    # whole
        plt.figure(figsize=(6,4))
        plt.loglog(freqs, mean_power, label='Mean Power')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum with feedback')
        plt.grid(True)
        plt.xlim(fft_l, fft_r)
        plt.legend()
        x_min, x_max = plt.xlim()
        mask = (freqs >= x_min) & (freqs <= x_max)
        if np.any(mask):
            plt.ylim(np.min(mean_power[mask]), np.max(mean_power[mask]))
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    # beta
        plt.figure(figsize=(6,4))
        plt.loglog(freqs, mean_power, label='Mean Power')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum (beta) with feedback')
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
        plt.figure(figsize=(6,4))
        plt.loglog(freqs, mean_power, label='Mean Power')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum (gamma) with feedback')
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
    save_path = f'{LFP_dir}/whole_diff_FFT_{sig}_{common_path}_{n_repeat}.eps'
    freqs1, mean_power1 = LFP_1area_repeat(param=param1, n_repeat=n_repeat, maxrate=maxrate, sig=sig,
                                           sti_type=sti_type, dt=dt, plot=plot, video=video, save_load=save_load)
    freqs2, mean_power2 = LFP_2area_repeat(param=param2, n_repeat=n_repeat, maxrate=maxrate, sig=sig,
                                           sti_type=sti_type, dt=dt, plot=plot, video=video, save_load=save_load)
    freqs_diff = freqs2
    mean_power_diff = mean_power2-mean_power1

    if plot:
    # whole
        plt.figure(figsize=(6,4))
        plt.plot(freqs_diff, mean_power_diff, label='Mean Power')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum difference')
        plt.grid(True)
        plt.xlim(fft_l, fft_r)
        plt.legend()
        x_min, x_max = plt.xlim()
        mask = (freqs_diff >= x_min) & (freqs_diff <= x_max)
        if np.any(mask):
            plt.ylim(np.min(mean_power_diff[mask]), np.max(mean_power_diff[mask]))
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    # beta
        plt.figure(figsize=(6,4))
        plt.plot(freqs_diff, mean_power_diff, label='Mean Power')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum (beta) difference')
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
        plt.figure(figsize=(6,4))
        plt.plot(freqs_diff, mean_power_diff, label='Mean Power')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum (gamma) difference')
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
                         sti_type='Uniform', plot=True, video=True, save_load=False):
    
    ie_r_e1, ie_r_i1, ie_r_e2, ie_r_i2 = param2
    common_path1 = f're1{ie_r_e1:.4f}_ri1{ie_r_i1:.4f}'
    common_path2 = f're1{ie_r_e1:.4f}_ri1{ie_r_i1:.4f}_re2{ie_r_e2:.4f}_ri2{ie_r_i2:.4f}'

    save_path_beta1 = f'{LFP_dir}/beta_1area_FFT_{common_path1}_{n_repeat}.eps'
    save_path_beta2 = f'{LFP_dir}/beta_2area_FFT_{common_path2}_{n_repeat}.eps'
    save_path_betad = f'{LFP_dir}/beta_diff_FFT_{common_path2}_{n_repeat}.eps'
    save_path_gamma1 = f'{LFP_dir}/gamma_1area_FFT_{common_path1}_{n_repeat}.eps'
    save_path_gamma2 = f'{LFP_dir}/gamma_2area_FFT_{common_path2}_{n_repeat}.eps'
    save_path_gammad = f'{LFP_dir}/gamma_diff_FFT_{common_path2}_{n_repeat}.eps'
    save_path1 = f'{LFP_dir}/whole_1area_FFT_{common_path1}_{n_repeat}.eps'
    save_path2 = f'{LFP_dir}/whole_2area_FFT_{common_path2}_{n_repeat}.eps'
    save_pathd = f'{LFP_dir}/whole_diff_FFT_{common_path2}_{n_repeat}.eps'

    results_1area = []
    results_2area = []
    results_diff = []

    for sig in sigs:
        freqs1, mean_power1, freqs2, mean_power2, freqs_diff, mean_power_diff = LFP_diff_repeat(
            param1=param1, param2=param2, n_repeat=n_repeat, maxrate=maxrate, sig=sig, dt=dt, 
            sti_type=sti_type, plot=plot, video=video, save_load=save_load
            )
        results_1area.append((sig, freqs1, mean_power1))
        results_2area.append((sig, freqs2, mean_power2))
        results_diff.append((sig, freqs1, mean_power_diff))

    # 1area
    if plot:
    # whole
        plt.figure(figsize=(6,4))
        for sig, freqs, power in results_1area:
            plt.loglog(freqs, power, label=f'sig={sig}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum without feedback')
        plt.grid(True)
        plt.xlim(fft_l, fft_r)
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
        if save_path1:
            plt.savefig(save_path1, dpi=300, bbox_inches='tight')
        plt.close()
    # beta
        plt.figure(figsize=(6,4))
        for sig, freqs, power in results_1area:
            plt.loglog(freqs, power, label=f'sig={sig}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum (beta) without feedback')
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
            plt.loglog(freqs, power, label=f'sig={sig}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum (gamma) without feedback')
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
    # whole
        plt.figure(figsize=(6,4))
        for sig, freqs, power in results_2area:
            plt.loglog(freqs, power, label=f'sig={sig}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum with feedback')
        plt.grid(True)
        plt.xlim(fft_l, fft_r)
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
        if save_path2:
            plt.savefig(save_path2, dpi=300, bbox_inches='tight')
        plt.close()
    # beta
        plt.figure(figsize=(6,4))
        for sig, freqs, power in results_2area:
            plt.loglog(freqs, power, label=f'sig={sig}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum (beta) with feedback')
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
            plt.loglog(freqs, power, label=f'sig={sig}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum (gamma) with feedback')
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
    # whole
        plt.figure(figsize=(6,4))
        for sig, freqs, power in results_diff:
            plt.plot(freqs, power, label=f'sig={sig}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum difference')
        plt.grid(True)
        plt.xlim(fft_l, fft_r)
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
        if save_pathd:
            plt.savefig(save_pathd, dpi=300, bbox_inches='tight')
        plt.close()
    # beta
        plt.figure(figsize=(6,4))
        for sig, freqs, power in results_diff:
            plt.plot(freqs, power, label=f'sig={sig}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Mean LFP FFT Spectrum (beta) difference')
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
        plt.title('Mean LFP FFT Spectrum (gamma) difference')
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
    # 1st pair (alpha <= 1.3)
    # param12 = (2.449990451446889, 1.795670364314891, 2.399131446733395, 1.8512390285440765)
    # 2nd pair(more d(r_rf), but alpha<=1.5)
    # 右下边缘，rf最小，alpha~1.5 - 双峰 gamma peak
    # param1  = (2.501407742047704, 1.8147028535939709)
    # 左上边缘，rf最大，alpha~1.5 - 没有 gamma peak
    param2  = (2.425126038006674, 1.927524600435643)
    # param12 = (2.501407742047704, 1.8147028535939709, 2.425126038006674, 1.927524600435643)
    # critical zone 右上角的点 - gamma peak 小
    # param1 = (2.67,2.03)
    # critical zone 左下角的点 - gamma peak 正常
    param1 = (2.22, 1.64)
    # 中心点 - gamma peak 较小
    # param1 = (2.4331,1.8447)
    param12 = (2.22, 1.64, 2.425126038006674, 1.927524600435643)

    # 故意写反看病态beta
    # param1  = (1.8147028535939709, 2.501407742047704)
    # param2  = (1.927524600435643, 2.425126038006674)
    # param12 = (1.8147028535939709, 2.501407742047704, 1.927524600435643, 2.425126038006674)
    # shuzheng 的参数
    # param1  = (2.3641, 1.9706)
    # param2  = (1.9313, 1.5709)
    # param12 = (2.3641, 1.9706, 1.9313, 1.5709)
    # shuzheng 的2nd area参数
    # param2 = (1.9313, 1.5709)
    # 我的第一层和黄的第二层
    # param1  = (2.501407742047704, 1.8147028535939709)
    # param2 = (1.9313, 1.5709)
    # param12 = (2.501407742047704, 1.8147028535939709, 1.9313, 1.5709)
    # 超过右下角的第一层，我的第二层
    # param1  = (2.6, 1.7)
    # param2  = (2.425126038006674, 1.927524600435643)
    # param12 = (2.6, 1.7, 2.425126038006674, 1.927524600435643)
    # 算一下看看有没有明显病态（显然是要画动画）
    # compute.compute_1(comb=param, video=True)
    # LFP_1area(param=param)
    # LFP_1area_repeat(param=param, n_repeat=64)

    # change scale
    le=64
    li=32
    # sti_type='Gaussian''Uniform''Annulus'
    # compute.compute_1(comb=param1, seed=10,sti=False,maxrate=500,sig=5,
    #                   sti_type='Uniform',video=True,le=int(le),li=int(li))
    # compute.compute_2(comb=param12,seed=10,sti=False,maxrate=500,sig=5,
    #                   sti_type='Uniform',video=True,le=int(le),li=int(li))

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
    ## distribution search
    # range_path = f'{state_dir}/critical_ellipse.file'
    # result = find_receptive_field_distribution_in_range(n_repeat=64, 
    #                                                     range_path=range_path, 
    #                                                     maxrate=1000, 
    #                                                     n_sample=1000)
    #%% draw 3d distribution
    # plot_rf_landscape_3d(1000,fit=False)

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

    #%% draw receptive field2
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

    #%% 1.5内的最大最小r_rf
    # param1  = (2.501407742047704, 1.8147028535939709)
    # param2  = (2.425126038006674, 1.927524600435643)
    # param12 = (2.501407742047704, 1.8147028535939709, 2.425126038006674, 1.927524600435643)
    # pick_parameters_and_repeat_compute(param=param1, video=True)
    # pick_parameters_and_repeat_compute(param=param2, video=True)
    # pick_parameters_and_repeat_compute2(param=param12,
    #                                     n_repeat=128,
    #                                     video=True)
    

    #%% LFP
    # def draw_LFP_FFT_2area():
    #     param1 = (1.795670364314891, 2.449990451446889)
    #     param2 = (1.795670364314891, 2.449990451446889, 1.8512390285440765, 2.399131446733395)
    #     sig=0
    #     LFP_1area(param=param1,maxrate=500,sig=sig,dt=0.1,plot=True)
    #     LFP_2area(param=param2,maxrate=500,sig=sig,dt=0.1,plot=True)
    # # draw_LFP_FFT_2area()
    def draw_LFP_FFT_1area_repeat(n_repeat=64,sig=0):
        # critical zone 右上角的点 - gamma peak 小
        # param = (2.67,2.03)
        # critical zone 左下角的点 - gamma peak 正常
        # param = (2.22, 1.64)
        # 右下边缘，rf最小，alpha~1.5 - 双峰 gamma peak
        # param  = (2.501407742047704, 1.8147028535939709)
        # 左上边缘，rf最大，alpha~1.5 - 没有 gamma peak
        # param  = (2.425126038006674, 1.927524600435643)
        # 中心点 - gamma peak 较小
        param = (2.4331,1.8447)
        # param = (1.795670364314891, 2.449990451446889)
        LFP_1area_repeat(param=param,n_repeat=n_repeat,maxrate=500,sig=sig,dt=0.1,plot=True,video=True,save_load=False)
    def draw_LFP_FFT_2area_repeat(n_repeat=64,sig=0):
        param = (1.795670364314891, 2.449990451446889, 1.8512390285440765, 2.399131446733395)
        LFP_2area_repeat(param=param,n_repeat=n_repeat,maxrate=500,sig=sig,dt=0.1,plot=True,video=True,save_load=False)
    # draw_LFP_FFT_1area_repeat()
    # draw_LFP_FFT_2area_repeat()

    # def draw_LFP_FFT_diff_repeat(n_repeat=128):
    #     param1 = (1.795670364314891, 2.449990451446889)
    #     param2 = (1.795670364314891, 2.449990451446889, 1.8512390285440765, 2.399131446733395)
    #     for sig in [0,5,10,15,20,25]:
    #         LFP_diff_repeat(param1=param1, param2=param2, n_repeat=n_repeat, sig=sig)
    # draw_LFP_FFT_diff_repeat()

    # param1 = (1.795670364314891, 2.449990451446889)
    # param2 = (1.795670364314891, 2.449990451446889, 1.8512390285440765, 2.399131446733395)
    # draw_LFP_FFT_compare(param1=param1, param2=param2)

    #%% alpha<1.5
    draw_LFP_FFT_compare(param1=param1, param2=param12, n_repeat=1024, maxrate=500, sti_type='Uniform')
    
    # print('computing start')
    # draw_receptive_field2(param=param1, n_repeat=64, le=le,li=li)
    # print('set 1 executed')
    # send_email.send_email('set 1 executed', 'set 1 executed')
    # draw_receptive_field2(param=param2, n_repeat=64, le=le,li=li)
    # print('set 2 executed')
    # send_email.send_email('set 2 executed', 'set 2 executed')




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
