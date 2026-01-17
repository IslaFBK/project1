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
from myscript.ie_search.batch_repeat import draw_statistical_MSD_pdx
from myscript.ie_search.batch_repeat import batch_repeat
from myscript.ie_search.batch_repeat import batch_repeat2
import myscript.ie_search.load_repeat as load_repeat
import myscript.ie_search.critical_states_search as search
import myscript.send_email as send_email
import myscript.ie_search.compute_general as compute

import logging
logging.getLogger('brian2').setLevel(logging.WARNING)

# plt.rcParams.update({
#     "text.usetex": True,  # 启用 LaTeX 渲染
#     "font.family": "serif",  # 使用衬线字体（类似 LaTeX 默认）
#     "font.serif": ["Times New Roman"]  # 指定字体
# })
def set_journal_style():
    plt.rcParams.update({
        # Font
        "text.usetex": True,  # 启用 LaTeX 渲染
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        
        # Font sizes
        # "axes.labelsize": 8,
        # "axes.titlesize": 10,
        # "xtick.labelsize": 7,
        # "ytick.labelsize": 7,
        # "legend.fontsize": 7,
        # "figure.titlesize": 10,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 9,
        
        # Axes & ticks
        # "axes.linewidth": 0.8,
        # "xtick.major.width": 0.8,
        # "ytick.major.width": 0.8,
        # "xtick.major.size": 3,
        # "ytick.major.size": 3,
        # "xtick.direction": "in",
        # "ytick.direction": "in",
        "axes.linewidth": 1,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "in",
        "ytick.direction": "in",

        # Save
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    })
set_journal_style()

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

def msd_pdx_1(param,n_repeat=64,stim_dura=1000,window=15,
              video=False,save_load=False,delta_gk=1,
              data_root=None,root_path=None,
              cmpt=True,save_data=True,plot=True,
              msd_path=None,pdx_path=None,msd_pdx_path=None):
    '''
    重复计算单层, 画MSD和PDX
    
    :param param: ie_ratio
    :param n_repeat: 重复计算数
    :param stim_dura: 稳态时长
    :param window: firing rate 时间窗宽度(ms)
    :param video: 是否画图
    :param save_load: 每组计算是否存元数据
    :param delta_gk: 使用1层delta_gk还是2层delta_gk
    :param data_root: MSD_PDX数据的保存根路径
    :param root_path: MSD_PDX图的保存根路径
    :param cmpt: 是否计算
    :param save_data: 是否保存MSD_PDX数据
    :param plot: 是否画图
    :param msd_path: MSD图路径
    :param pdx_path: PDX图路径
    :param msd_pdx_path: MSD_PDX结合图路径
    '''
    ie_r_e1, ie_r_i1 = param
    # common title & path
    common_path = f're{ie_r_e1:.4f}_ri{ie_r_i1:.4f}'
    # create data root path:
    if data_root is None:
        data_root = state_dir
    # create graph root path:
    if root_path is None:
        root_path = f'{graph_dir}/MSD_PDX/'
    
    Path(data_root).mkdir(parents=True, exist_ok=True)
    Path(root_path).mkdir(parents=True, exist_ok=True)
    data_path = f'{data_root}/1MSDPDX_{common_path}_{n_repeat}_{delta_gk}.file'
    if cmpt: # 是否计算
        # compute 1:
        if video:
            results = Parallel(n_jobs=-1)(
                delayed(compute.compute_1_general)(
                    comb=param,seed=i,index=i,sti=False,video=(i==0),save_load=save_load,
                    window=window,stim_dura=stim_dura,delta_gk=delta_gk
                )
                for i in range(n_repeat)
            )
        else:
            results = Parallel(n_jobs=-1)(
                delayed(compute.compute_1_general)(
                    comb=param,seed=i,index=i,sti=False,video=False,save_load=save_load,
                    window=window,stim_dura=stim_dura,delta_gk=delta_gk
                )
                for i in range(n_repeat)
            )
        # 合并msd
        msds = np.stack([r['msd'] for r in results])
        jump_interval = results[0]['jump_interval']
        msd_mean = np.mean(msds, axis=0)
        msd_std = np.std(msds, axis=0)
        # 合并pdx
        all_pdx = np.concatenate([r['pdx'] for r in results])
        if save_data:
            # 存msd & pdx
            msd_pdx_info = {
                'results':results,
                'msds':msds,
                'jump_interval':jump_interval,
                'msd_mean':msd_mean,
                'msd_std':msd_std,
                'all_pdx':all_pdx
            }
            with open(data_path, 'wb') as file:
                pickle.dump(msd_pdx_info,file)
    else:
        if os.path.exists(data_path):
            with open(data_path,'rb') as file:
                msd_pdx_info = pickle.load(file)
            results = msd_pdx_info['results']
            msds = msd_pdx_info['msds']
            jump_interval = msd_pdx_info['jump_interval']
            msd_mean = msd_pdx_info['msd_mean']
            msd_std = msd_pdx_info['msd_std']
            all_pdx = msd_pdx_info['all_pdx']
        else:
            msg = (
                f"Required data file not found: {data_path}. "
                "Run with cmpt=True to generate it or provide a valid data_path."
            )
            send_email.send_email('Required path missing', msg)
            raise FileNotFoundError(msg)
    if plot:
        # 画图
        if msd_path is None:
            msd_path = f'{root_path}/1MSD_{common_path}_{n_repeat}_{delta_gk}.svg'
        if pdx_path is None:
            pdx_path = f'{root_path}/1PDX_{common_path}_{n_repeat}_{delta_gk}.svg'
        if msd_pdx_path is None:
            msd_pdx_path = f'{root_path}/1MSDPDX_{common_path}_{n_repeat}_{delta_gk}.svg'

        draw_statistical_MSD_pdx(jump_interval=jump_interval,
                                 msd_mean=msd_mean,
                                 msd_std=msd_std,
                                 all_pdx=all_pdx,
                                 save_path_MSD=msd_path,
                                 save_path_pdx=pdx_path,
                                 save_path_combined=msd_pdx_path)

def msd_pdx_2(param,n_repeat=64,stim_dura=1000,window=15,
              video=False,save_load=False,
              data_root=None,root_path=None,
              cmpt=True,save_data=True,plot=True,
              msd_path=None,pdx_path=None,msd_pdx_path=None,
              w_12_e=2.4,w_12_i=2.4,w_21_e=2.4,w_21_i=2.4):
    '''
    重复计算双层, 画MSD和PDX
    
    :param param: ie_ratio
    :param n_repeat: 重复计算数
    :param stim_dura: 稳态时长
    :param window: firing rate 时间窗宽度(ms)
    :param video: 是否画图
    :param save_load: 每组计算是否存元数据
    :param data_root: MSD_PDX数据的保存根路径
    :param root_path: MSD_PDX图的保存根路径
    :param cmpt: 是否计算
    :param save_data: 是否保存MSD_PDX数据
    :param plot: 是否画图
    :param msd_path: MSD图路径
    :param pdx_path: PDX图路径
    :param msd_pdx_path: MSD_PDX结合图路径
    '''
    ie_r_e1, ie_r_i1, ie_r_e2, ie_r_i2 = param
    # common title & path
    common_path = f're1{ie_r_e1:.4f}_ri1{ie_r_i1:.4f}_re2{ie_r_e2:.4f}_ri2{ie_r_i2:.4f}'
    # create data root path:
    if data_root is None:
        data_root = state_dir
    # create graph root path:
    if root_path is None:
        root_path = f'{graph_dir}/MSD_PDX/'
    
    Path(data_root).mkdir(parents=True, exist_ok=True)
    Path(root_path).mkdir(parents=True, exist_ok=True)
    data_path = f'{data_root}/2MSDPDX_{common_path}_{n_repeat}.file'
    if cmpt:
        # compute 2:
        if video:
            results = Parallel(n_jobs=-1)(
                delayed(compute.compute_2_general)(
                    comb=param,seed=i,index=i,sti=False,video=(i==0),save_load=save_load,
                    window=window,stim_dura=stim_dura,
                    w_12_e=w_12_e,w_12_i=w_12_i,w_21_e=w_21_e,w_21_i=w_21_i
                )
                for i in range(n_repeat)
            )
        else:
            results = Parallel(n_jobs=-1)(
                delayed(compute.compute_2_general)(
                    comb=param,seed=i,index=i,sti=False,video=False,save_load=save_load,
                    window=window,stim_dura=stim_dura,
                    w_12_e=w_12_e,w_12_i=w_12_i,w_21_e=w_21_e,w_21_i=w_21_i
                )
                for i in range(n_repeat)
            )
        # Area 1
        # 合并msd
        msds1 = np.stack([r['msd1'] for r in results])
        jump_interval1 = results[0]['jump_interval1']
        msd_mean1 = np.mean(msds1, axis=0)
        msd_std1 = np.std(msds1, axis=0)
        # 合并pdx
        all_pdx1 = np.concatenate([r['pdx1'] for r in results])

        # Area 2
        # 合并msd
        msds2 = np.stack([r['msd2'] for r in results])
        jump_interval2 = results[0]['jump_interval2']
        msd_mean2 = np.mean(msds2, axis=0)
        msd_std2 = np.std(msds2, axis=0)
        # 合并pdx
        all_pdx2 = np.concatenate([r['pdx2'] for r in results])
        
        if save_data:
            # 存msd & pdx
            msd_pdx_info = {
                'results':results,
                'msds1':msds1,
                'jump_interval1':jump_interval1,
                'msd_mean1':msd_mean1,
                'msd_std1':msd_std1,
                'all_pdx1':all_pdx1,
                'msds2':msds2,
                'jump_interval2':jump_interval2,
                'msd_mean2':msd_mean2,
                'msd_std2':msd_std2,
                'all_pdx2':all_pdx2
            }
            with open(data_path, 'wb') as file:
                pickle.dump(msd_pdx_info,file)
    else:
        if os.path.exists(data_path):
            with open(data_path,'rb') as file:
                msd_pdx_info = pickle.load(file)
            results = msd_pdx_info['results']
            # Area 1
            msds1 = msd_pdx_info['msds1']
            jump_interval1 = msd_pdx_info['jump_interval1']
            msd_mean1 = msd_pdx_info['msd_mean1']
            msd_std1 = msd_pdx_info['msd_std1']
            all_pdx1 = msd_pdx_info['all_pdx1']
            # Area 2
            msds2 = msd_pdx_info['msds2']
            jump_interval2 = msd_pdx_info['jump_interval2']
            msd_mean2 = msd_pdx_info['msd_mean2']
            msd_std2 = msd_pdx_info['msd_std2']
            all_pdx2 = msd_pdx_info['all_pdx2']
        else:
            msg = (
                f"Required data file not found: {data_path}. "
                "Run with cmpt=True to generate it or provide a valid data_path."
            )
            send_email.send_email('Required path missing', msg)
            raise FileNotFoundError(msg)
    if plot:
        # 画图
        if msd_path is None:
            msd_path = f'{root_path}/2MSD_{common_path}_{n_repeat}'
        if pdx_path is None:
            pdx_path = f'{root_path}/2PDX_{common_path}_{n_repeat}'
        if msd_pdx_path is None:
            msd_pdx_path = f'{root_path}/2MSDPDX_{common_path}_{n_repeat}'

        # Area 1
        draw_statistical_MSD_pdx(jump_interval=jump_interval1,
                                 msd_mean=msd_mean1,
                                 msd_std=msd_std1,
                                 all_pdx=all_pdx1,
                                 save_path_MSD=f'{msd_path}_1.svg',
                                 save_path_pdx=f'{pdx_path}_1.svg',
                                 save_path_combined=f'{msd_pdx_path}_1.svg')
        # Area 2
        draw_statistical_MSD_pdx(jump_interval=jump_interval2,
                                 msd_mean=msd_mean2,
                                 msd_std=msd_std2,
                                 all_pdx=all_pdx2,
                                 save_path_MSD=f'{msd_path}_2.svg',
                                 save_path_pdx=f'{pdx_path}_2.svg',
                                 save_path_combined=f'{msd_pdx_path}_2.svg')

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

def evalution_search(compute=False, repeat_MSD=False, conf_level=0.99, delta_gk=1, 
                     remove_outlier=True):
    if compute:
        # 初始参数栅格
        initial_param = {
            'ie_r_e1': np.linspace(1.9, 2.6, 8),
            'ie_r_i1': np.linspace(1.5, 2.2, 8)
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
            n_child=5,
            delta_gk=delta_gk
        )
        # save
        print('saving')
        with open(f'{state_dir}/evolution{delta_gk}.file', 'wb') as file:
            pickle.dump(history, file)

    # load
    print('loading')
    with open(f'{state_dir}/evolution{delta_gk}.file', 'rb') as file:
        history = pickle.load(file)

    # print(len(history))

    # draw
    print('drawing')
    save_path = f'{graph_dir}/evaluation{delta_gk}.svg'
    ellipse_info = search.plot_evolution_history(history=history,
                                                 save_path=save_path,
                                                 remove_outlier=remove_outlier,
                                                 plot_hull=True,
                                                 plot_ellipse=True,
                                                 conf_level=conf_level)
    # 保存椭圆（及凸包）边界信息
    with open(f'{state_dir}/critical_ellipse{delta_gk}.file', 'wb') as file:
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

def sample_hull_uniform(hull_vertices, n_samples=1000):
    """基于凸包顶点进行均匀采样"""
    triangles = []
    areas = []
    # 三角剖分：以第一个顶点为公共顶点
    for i in range(1, len(hull_vertices)-1):
        tri = np.array([hull_vertices[0], hull_vertices[i], hull_vertices[i+1]])
        triangles.append(tri)
        # 计算三角形面积
        area = 0.5 * np.abs(
            (tri[1,0]-tri[0,0])*(tri[2,1]-tri[0,1]) - (tri[1,1]-tri[0,1])*(tri[2,0]-tri[0,0])
        )
        areas.append(area)
    
    # 面积加权采样
    areas = np.array(areas)
    area_weights = areas / areas.sum()
    tri_indices = np.random.choice(len(triangles), size=n_samples, p=area_weights)
    
    # 三角形内均匀采样
    samples = []
    for idx in tri_indices:
        tri = triangles[idx]
        a, b = np.random.rand(2)
        if a + b > 1:
            a, b = 1 - a, 1 - b
        sample = a * tri[0] + b * tri[1] + (1 - a - b) * tri[2]
        samples.append(sample)
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
                           save_load0=False, save_load1=False,
                           delta_gk=1):
    
    if video0:
        result0 = Parallel(n_jobs=-1)(
            delayed(compute.compute_1_general)(comb=param,seed=i,index=i,sti=False, 
                                               video=(i==0),save_load=save_load0,
                                               delta_gk=delta_gk)
            for i in range(n_repeat)
        )
    else:
        result0 = Parallel(n_jobs=-1)(
            delayed(compute.compute_1_general)(comb=param,seed=i,index=i,sti=False, 
                                               video=False,save_load=save_load0,
                                               delta_gk=delta_gk)
            for i in range(n_repeat)
        )
    if video1:
        result1 = Parallel(n_jobs=-1)(
            delayed(compute.compute_1_general)(comb=param,seed=i,index=i,sti=True, 
                                               maxrate=maxrate,
                                               video=(i==0),save_load=save_load1,
                                               delta_gk=delta_gk)
            for i in range(n_repeat)
        )
    else:
        result1 = Parallel(n_jobs=-1)(
            delayed(compute.compute_1_general)(comb=param,seed=i,index=i,sti=True, 
                                               maxrate=maxrate,
                                               video=False,save_load=save_load1,
                                               delta_gk=delta_gk)
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

    save_path = f'{recfield_dir}/{n_repeat}_{maxrate}fr_ext-dist{common_path}_{delta_gk}.svg'
    data_path = f'{state_dir}/{n_repeat}_{maxrate}fr_ext{common_path}_{delta_gk}.file'
    r_rf = mya.receptive_field(spk_rate0=spk_rate0_mean,
                               spk_rate1=spk_rate1_mean,
                               save_path=save_path,
                               data_path=data_path,
                               plot=plot)
    return r_rf

# compute receptive field radius and alpha with repeat realizaiton
def rf_and_alpha_repeat(param, n_repeat, plot=False,
                        video0=False, video1=False, maxrate=1000,
                        save_load0=False, save_load1=False,
                        delta_gk=1):
    # without stimuli
    if video0:
        result0 = Parallel(n_jobs=-1)(
            delayed(compute.compute_1_general)(comb=param, seed=i, index=i, sti=False, 
                                               video=(i==0), save_load=save_load0, delta_gk=delta_gk)
            for i in range(n_repeat)
        )
    else:
        result0 = Parallel(n_jobs=-1)(
            delayed(compute.compute_1_general)(comb=param, seed=i, index=i, sti=False, 
                                               video=False, save_load=save_load0, delta_gk=delta_gk)
            for i in range(n_repeat)
        )
    # with stimuli
    if video1:
        result1 = Parallel(n_jobs=-1)(
            delayed(compute.compute_1_general)(comb=param, seed=i, index=i, sti=True, maxrate=maxrate,
                                               video=(i==0), save_load=save_load1, delta_gk=delta_gk)
            for i in range(n_repeat)
        )
    else:
        result1 = Parallel(n_jobs=-1)(
            delayed(compute.compute_1_general)(comb=param, seed=i, index=i, sti=True, maxrate=maxrate,
                                               video=False, save_load=save_load1, delta_gk=delta_gk)
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

    save_path = f'{recfield_dir}/{n_repeat}_{maxrate}fr_ext-dist{common_path}_{delta_gk}.png'
    data_path = f'{state_dir}/{n_repeat}_{maxrate}fr_ext{common_path}_{delta_gk}.file'
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
                                               n_sample=1000, fit=False, delta_gk=1, 
                                               sample_type='Ellipse'):
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

    # 提取凸包核心信息
    hull_vertices = ellipse_info['hull_vertices']
    hull_boundary_closed = ellipse_info['hull_boundary_closed']
    # filtered_points = ellipse_info['filtered_critical_points']
    # hull = ellipse_info['hull_object']

    if sample_type == 'Ellipse':
        # 椭圆内采样参数
        params = sample_in_ellipse(mean, cov, conf_level, n_sample)
        params = [tuple(p) for p in params]
    elif sample_type == 'Hull':
        # 椭圆内采样参数
        params = sample_hull_uniform(hull_vertices, n_sample)
        params = [tuple(p) for p in params]
    else:
        # 抛出 ValueError，明确提示合法取值和当前错误值
        raise ValueError(
            f"无效的采样类型 '{sample_type}'!"
            f"合法的采样类型仅支持：'Ellipse' 或 'Hull'。"
        )

    # 尝试读取已有历史
    rf_history_path = f'{state_dir}/rf_landscape_{maxrate}_{delta_gk}.file'
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

        # 由于已修改critical_states_search.plot_evolution_history的xy轴，这里可以改回来了
        param_tuple = (param[0], param[1]) 
        try:
            field = rf_and_alpha_repeat(param=param_tuple, 
                                        n_repeat=n_repeat, 
                                        maxrate=maxrate, 
                                        plot=False,
                                        delta_gk=delta_gk)
            r_rf = field['r_rf']
            alpha = field['alpha']
            critical = field['critical']
        except Exception as e:
            print(f"参数 {param_tuple} 计算失败: {e}")
            send_email.send_email('Error', f"参数 {param_tuple} 计算失败: {e}")
            continue
        
        # 修改1: 当alpha>1.5时跳过当前参数
        if alpha > 1.5:
            print(f"参数 {param_tuple} alpha={alpha:.3f}>1.5,跳过")
            send_email.send_email('Skip', f"参数 {param_tuple} alpha={alpha:.3f}>1.5,跳过")
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
        with open(f'{state_dir}/rf_landscape_{maxrate}_{delta_gk}.file', 'wb') as file:
            pickle.dump(r_rf_history, file)

        # 画地形图
        x = [info['param'][0] for info in r_rf_history]
        y = [info['param'][1] for info in r_rf_history]
        z_rf = [info['r_rf'] for info in r_rf_history]
        z_alpha = [info['alpha'] for info in r_rf_history]
        if sample_type == 'Ellipse':
            # 创建椭圆对象
            ellipse1 = Ellipse(xy=(mean[0], mean[1]), width=width, height=height, angle=theta, 
                                edgecolor='blue', facecolor='none', lw=2, 
                                label='Ellipse Boundary', zorder=4)
            ellipse2 = Ellipse(xy=(mean[0], mean[1]), width=width, height=height, angle=theta, 
                                edgecolor='blue', facecolor='none', lw=2, 
                                label='Ellipse Boundary', zorder=4)

        fig, axs = plt.subplots(1, 2, figsize=(7, 3.5))
        # 左：r_rf
        sc1 = axs[0].scatter(x, y, c=z_rf, cmap='viridis', s=60)
        if sample_type == 'Ellipse':
            axs[0].add_patch(ellipse1)
        elif sample_type == 'Hull':
            axs[0].plot(hull_boundary_closed[:, 0], hull_boundary_closed[:, 1], 
                        'r-', linewidth=2, label='Convex Hull')
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
        if sample_type == 'Ellipse':
            axs[1].add_patch(ellipse2)
        elif sample_type == 'Hull':
            axs[1].plot(hull_boundary_closed[:, 0], hull_boundary_closed[:, 1], 
                        'r-', linewidth=2, label='Convex Hull')
        axs[1].set_xlabel(r'$\zeta^{\rm E}$', fontsize=10)
        axs[1].set_ylabel(r'$\zeta^{\rm I}$', fontsize=10)
        axs[1].set_title(r'$\alpha$', fontsize=11)
        axs[1].tick_params(axis='both', labelsize=10)
        cbar2 = plt.colorbar(sc2, ax=axs[1])
        cbar2.set_label(r'$\alpha$', fontsize=10)
        cbar2.ax.tick_params(labelsize=10)
        # axs[1].legend(fontsize=9)
        plt.tight_layout(pad=1.0)
        plt.savefig(f'{graph_dir}/rf_landscape_{maxrate}_{delta_gk}.svg', dpi=300)
        plt.close()

        # # 画3维地形图（已淘汰，现在用matlab画3d图）
        # # 提取有效数据
        # x, y, z = [], [], []
        # x_bad, y_bad = [], []
        # for info in r_rf_history:
        #     param = info['param']
        #     r_rf = info['r_rf']
        #     # 自动舍弃无效点
        #     if r_rf is None or not isinstance(r_rf, (int, float)) or not (r_rf == r_rf):  # 排除None和NaN
        #         x_bad.append(param[0])
        #         y_bad.append(param[1])
        #         continue
        #     x.append(param[0])
        #     y.append(param[1])
        #     z.append(r_rf)
        
        # # 转为numpy数组，避免幂运算报错
        # x = np.array(x)
        # y = np.array(y)
        # z = np.array(z)

        # fig = plt.figure(figsize=(8,6))
        # ax = fig.add_subplot(111, projection='3d')
        # sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=40)

        # if fit and len(x)>6:
        #     # 曲面拟合
        #     coeffs = fit_quadratic_surface(x, y, z)

        #     # 计算拟合值和R²
        #     X_design = np.column_stack([x**2, y**2, x*y, x, y, np.ones_like(x)])
        #     z_fit = X_design @ coeffs
        #     ss_res = np.sum((z - z_fit) ** 2)
        #     ss_tot = np.sum((z - np.mean(z)) ** 2)
        #     r2 = 1 - ss_res / ss_tot

        #     # 生成网格用于画曲面
        #     x_grid = np.linspace(np.min(x), np.max(x), 50)
        #     y_grid = np.linspace(np.min(y), np.max(y), 50)
        #     X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        #     Z_grid = (coeffs[0]*X_grid**2 + coeffs[1]*Y_grid**2 + coeffs[2]*X_grid*Y_grid +
        #             coeffs[3]*X_grid + coeffs[4]*Y_grid + coeffs[5])

        #     # 画三维地形图
        #     surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='coolwarm', alpha=0.5, linewidth=0, antialiased=True)
        #     ax.set_title(f'Receptive Field 3D Landscape with Quadratic Surface Fit, $R^2$={r2:.3f}')
        # else:
        #     ax.set_title('Receptive Field 3D Landscape')

        # # 获取当前z轴范围
        # if len(z) > 0:
        #     zmin, zmax = np.min(z), np.max(z)
        #     dz = zmax - zmin
        #     z_bad = np.full(len(x_bad), zmin - 0.05*dz if dz > 0 else zmin - 1)
        #     # 在最低面下方一点画无效点
        #     ax.scatter(x_bad, y_bad, z_bad, c='red', marker='x', s=50, label='Invalid r_rf')
        #     # 可选：加图例
        #     ax.legend()
        
        # ax.set_xlabel(r'$\zeta^{\rm E}$')
        # ax.set_ylabel(r'$\zeta^{\rm I}$')
        # ax.set_zlabel('Receptive Field')
        # fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10, label='Receptive Field')
        # plt.tight_layout()
        # plt.savefig(f'{graph_dir}/rf_landscape_3d_{maxrate}_{delta_gk}.png', dpi=300)
        # # 俯视视角
        # ax.view_init(elev=90, azim=-90)
        # plt.tight_layout()
        # plt.savefig(f'{graph_dir}/rf_landscape_3d_{maxrate}_{delta_gk}_top.png', dpi=300)

        # plt.close()

        # 发邮件报告进度
        send_email.send_email(
            f'Progress: Complete {loop_num} in {loop_total}',
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
# 已过时，n_sample意义不大现已替换为maxrate索引，而且3d图现在会在matlab里画
def plot_rf_landscape_3d(n_sample, fit=True, delta_gk=1):
    # 读取历史文件
    rf_history_path = f'{state_dir}/rf_landscape_{n_sample}_{delta_gk}.file'
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

def load_and_draw_receptive_field(param, maxrate=5000, n_repeat=64, delta_gk=1):
    ie_r_e1, ie_r_i1 = param
    common_path = f're{ie_r_e1:.4f}_ri{ie_r_i1:.4f}'
    save_path = f'{recfield_dir}/{n_repeat}_{maxrate}fr_ext-dist{common_path}_{delta_gk}.png'
    data_path = f'{state_dir}/{n_repeat}_{maxrate}fr_ext{common_path}_{delta_gk}.file'
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
                            save_load0=False, save_load1=False, le=64, li=32,delta_gk=1):
    
    if video0:
        result0 = Parallel(n_jobs=-1)(
            delayed(compute.compute_1_general)(comb=param, seed=i, index=i, sti=False,
                                               video=(i==0), save_load=save_load0,
                                               le=le, li=li,delta_gk=delta_gk)
            for i in range(n_repeat)
        )
    else:
        result0 = Parallel(n_jobs=-1)(
            delayed(compute.compute_1_general)(comb=param, seed=i, index=i, sti=False,
                                               video=False, save_load=save_load0,
                                               le=le, li=li,delta_gk=delta_gk)
            for i in range(n_repeat)
        )
    if video1:
        result1 = Parallel(n_jobs=-1)(
            delayed(compute.compute_1_general)(comb=param, seed=i, index=i, sti=True,
                                               maxrate=maxrate, sig=sig, sti_type=sti_type,
                                               video=(i==0), save_load=save_load1,
                                               le=le, li=li,delta_gk=delta_gk)
            for i in range(n_repeat)
        )
    else:
        result1 = Parallel(n_jobs=-1)(
            delayed(compute.compute_1_general)(comb=param, seed=i, index=i, sti=True,
                                               maxrate=maxrate, sig=sig, sti_type=sti_type,
                                               video=False, save_load=save_load1,
                                               le=le, li=li,delta_gk=delta_gk)
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

    save_pathr = f'{recfield_dir}/middle_zratio{n_repeat}_{maxrate}fr_ext{common_path}.svg'
    save_pathd = f'{recfield_dir}/middle_zdiff{n_repeat}_{maxrate}fr_ext{common_path}.svg'
    
    plt.figure(figsize=(5,5))
    plt.plot(sigs, ratios, 'o-')
    plt.xlabel('Stimuli size')
    plt.ylabel('Centre firing rate ratio')
    plt.title('Centre firing rate ratio vs. stimuli size')
    plt.savefig(save_pathr, dpi=600, format='svg')

    plt.figure(figsize=(5,5))
    plt.plot(sigs, diffs, 'o-')
    plt.xlabel('Stimuli size')
    plt.ylabel('Centre firing rate difference')
    plt.title('Centre firing rate ratio vs. stimuli size')
    plt.savefig(save_pathd, dpi=600, format='svg')

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

# 画单组LFP FFT, 可mean可单算例
def draw_LFP_FFT(freqs, power_mean, power_std, 
                 save_path, save_path_beta, save_path_gama, 
                 plotlog='loglog', std_plot=False):
    '''
    draw_LFP_FFT 的 Docstring
    
    :param freqs: LFP的集合
    :param power_mean:
    :param power_std:
    :param save_path: 全波段集合图保存路径
    :param save_path_beta: beta波段集合图保存路径
    :param save_path_gama: gamma波段集合图保存路径
    :param plotlog: 可以取'loglog','semilogx','semilogy','linear'
    '''
    def _plot_specturm(freqs, power_mean, power_std, plotlog, 
                       x_lim, title, save_file, figsize=(2,2), std_plot=False):
        plt.figure(figsize=figsize)
        # plot specturm:
        power_mean=power_mean*1e-9
        if plotlog=='loglog':
            line = plt.loglog(freqs, power_mean, label='Mean Power')[0]
        elif plotlog=='semilogx':
            line = plt.semilogx(freqs, power_mean, label='Mean Power')[0]
        elif plotlog=='semilogy':
            line = plt.semilogy(freqs, power_mean, label='Mean Power')[0]
        else:
            line = plt.plot(freqs, power_mean, label='Mean Power')[0]
        
        if power_std is not None and std_plot:
            line_color = line.get_color()
            # cut upper lower boundary
            if plotlog=='loglog' or 'semilogy':
                y_min = np.maximum(power_mean-power_std, 0)
            else:
                y_min = power_mean-power_std
            y_max = power_mean + power_std
            plt.fill_between(freqs, y_min, y_max, color=line_color, alpha=0.3)

        # 图表样式
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD (a.u.)')
        ax = plt.gca()
        # 1. 设置x轴主刻度字体
        for label in ax.get_xticklabels():
            label.set_family('Arial')  # 仅修改字体，不触碰标签文本和刻度位置

        # 2. 设置y轴主刻度字体（消除你遇到的警告）
        for label in ax.get_yticklabels():
            label.set_family('Arial')  # 替代原有的ax.set_yticklabels()

        # 可选：设置次刻度字体（若有对数刻度，避免遗漏）
        for label in ax.get_xminorticklabels():
            label.set_family('Arial')
        for label in ax.get_yminorticklabels():
            label.set_family('Arial')
        # plt.title(title)
        # 网格样式
        plt.grid(True, which='both', ls='-', alpha=0.2)
        plt.xlim(x_lim)
        # plt.legend()
        # y轴范围
        x_min, x_max = plt.xlim()
        mask = (freqs >= x_min) & (freqs <= x_max)
        if np.any(mask):
            if std_plot:
                y_min_data = np.min(power_mean[mask]-power_std[mask])
                y_max_data = np.max(power_mean[mask]+power_std[mask])
            else:
                y_min_data = np.min(power_mean[mask])
                y_max_data = np.max(power_mean[mask])
            # 5% margin
            y_margin = (y_max_data-y_min_data)*0.05/2*0
            plt.ylim(y_min_data-y_margin, y_max_data+y_margin)
        # save
        if save_file:
            plt.savefig(save_file, dpi=600, bbox_inches='tight')
        plt.close()
    
    # three frequency band
    fft_bands = [
        # (x_lim, title, save_path)
        ((1, 100), 'Mean LFP FFT Spectrum', save_path),
        ((15, 30), 'Mean LFP FFT Spectrum (beta)',  save_path_beta),
        ((30, 80), 'Mean LFP FFT Spectrum (gamma)', save_path_gama)
    ]
    for x_lim, title, save_file in fft_bands:
        _plot_specturm(freqs, power_mean, power_std, plotlog, 
                       x_lim, title, save_file, std_plot=std_plot)

# 多组数据compare在一个图里，LFPs代表多组LFP对比图
def draw_LFP_FFTs(results, save_path, save_path_beta, save_path_gama, 
                  plotlog='loglog', std_plot=True):
    '''
    draw_LFP_FFTs 的 Docstring
    
    :param results: LFP的集合
    :param save_path: 全波段集合图保存路径
    :param save_path_beta: beta波段集合图保存路径
    :param save_path_gama: gamma波段集合图保存路径
    :param plotlog: 可以取'loglog','semilogx','semilogy','linear'
    '''
    def _plot_multiple_spectra(results, plotlog, x_lim, title, save_file, figsize=(2,2)):
        plt.figure(figsize=figsize)
        # figsize原来是(6,4)
        # loop plot multiple spectra
        for sig, freqs, power_mean, power_std in results:
            power_mean=power_mean*1e-9
            if plotlog == 'loglog':
                line = plt.loglog(freqs, power_mean, label=f'sig={sig}')[0]
            elif plotlog == 'semilogx':
                line = plt.semilogx(freqs, power_mean, label=f'sig={sig}')[0]
            elif plotlog == 'semilogy':
                line = plt.semilogy(freqs, power_mean, label=f'sig={sig}')[0]
            elif plotlog == 'linear':  # 补充linear分支，与文档字符串对应
                line = plt.plot(freqs, power_mean, label=f'sig={sig}')[0]
            
            if power_std is not None and std_plot:
                line_color = line.get_color()
                # cut upper lower boundary
                if plotlog=='loglog' or 'semilogy':
                    y_min = np.maximum(power_mean-power_std, 0)
                else:
                    y_min = power_mean-power_std
                y_max = power_mean + power_std
                plt.fill_between(freqs, y_min, y_max, color=line_color, alpha=0.3)
        
        # 图表样式设置
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD (a.u.)')
        ax = plt.gca()
        # 1. 设置x轴主刻度字体
        for label in ax.get_xticklabels():
            label.set_family('Arial')  # 仅修改字体，不触碰标签文本和刻度位置

        # 2. 设置y轴主刻度字体（消除你遇到的警告）
        for label in ax.get_yticklabels():
            label.set_family('Arial')  # 替代原有的ax.set_yticklabels()

        # 可选：设置次刻度字体（若有对数刻度，避免遗漏）
        for label in ax.get_xminorticklabels():
            label.set_family('Arial')
        for label in ax.get_yminorticklabels():
            label.set_family('Arial')
        # plt.title(title)
        # 对数坐标适配的网格（显示主次网格，提升可读性）
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.xlim(x_lim)
        # plt.legend()
        
        # 计算所有数据在x轴范围内的y值，统一设置y轴范围（添加边距）
        x_min, x_max = plt.xlim()
        all_masked_power = []
        all_masked_std = []
        for sig, freqs, power, std in results:
            power=power*1e-9
            mask = (freqs >= x_min) & (freqs <= x_max)
            if np.any(mask):
                all_masked_power.append(power[mask])
                if std is not None and std_plot:
                    all_masked_std.append(std[mask])
        
        if all_masked_power:
            all_masked_power = np.concatenate(all_masked_power)
            if all_masked_std is not None and std_plot:
                all_masked_std = np.concatenate(all_masked_std)
                y_min_data = np.min(all_masked_power-all_masked_std)
                y_max_data = np.max(all_masked_power+all_masked_std)
            else:
                y_min_data = np.min(all_masked_power)
                y_max_data = np.max(all_masked_power)
            # 添加5%边距，避免图线紧贴坐标轴
            y_margin = (y_max_data - y_min_data) * 0.05/2*0
            plt.ylim(y_min_data - y_margin, y_max_data + y_margin)
        
        # 保存图片（仅当save_file有效时）
        if save_file:
            plt.savefig(save_file, dpi=600, bbox_inches='tight')
        plt.close()
    
    # 定义三个波段的核心参数（统一管理，便于后续修改）
    fft_bands = [
        # (x_lim, title, save_path)
        ((1, 100), 'Mean LFP FFT Spectrum', save_path),
        ((15, 30), 'Mean LFP FFT Spectrum (beta)',  save_path_beta),
        ((30, 80), 'Mean LFP FFT Spectrum (gamma)', save_path_gama)
    ]

    # 循环绘制三个波段的多组对比图（复用辅助函数，消除冗余）
    for x_lim, title, save_file in fft_bands:
        _plot_multiple_spectra(results, plotlog, x_lim, title, save_file)

# compute 1 area centre point LFP, and output FFT
def LFP_1area(param, maxrate=500, sig=5, dt=0.1, plot=True, video=True):
    ie_r_e1, ie_r_i1 = param
    common_path = f're{ie_r_e1:.4f}_ri{ie_r_i1:.4f}'
    save_path = f'{LFP_dir}/1area_FFT_{sig}_{common_path}.svg'
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
    save_path = f'{LFP_dir}/2area_FFT_{sig}_{common_path}.svg'
    freqs, power = mya.analyze_LFP_fft(LFP, dt=dt, plot=plot, save_path=save_path)
    return freqs, power

# repeat computing 1 area FFT of LFP, output beta band and gamma band spectrum
def LFP_1area_repeat(param, n_repeat=64, maxrate=500, sig=5, dt=0.1, 
                     plot=True, video=True,stim_dura=10000,std_plot=False,
                     save_load=False,save_path_video=None,save_lfp=True,
                     save_path_beta=None,save_path_gama=None,save_path=None,
                     save_path_root=LFP_dir,lfp_data_path=None,cmpt=True,
                     sti=True,sti_type='Uniform',delta_gk=1):
    ie_r_e1, ie_r_i1 = param
    common_path = f're{ie_r_e1:.4f}_ri{ie_r_i1:.4f}'
    if save_path_beta is None:
        save_path_beta = f'{save_path_root}/beta_1FFT_{maxrate}_{sti_type}_{sig}_{common_path}_{n_repeat}_{stim_dura}_{delta_gk}.svg'
    if save_path_gama is None:
        save_path_gama = f'{save_path_root}/gama_1FFT_{maxrate}_{sti_type}_{sig}_{common_path}_{n_repeat}_{stim_dura}_{delta_gk}.svg'
    if save_path is None:
        save_path =      f'{save_path_root}/full_1FFT_{maxrate}_{sti_type}_{sig}_{common_path}_{n_repeat}_{stim_dura}_{delta_gk}.svg'
    if cmpt:
        if video:
            results = Parallel(n_jobs=-1)(
                delayed(compute.compute_1_general)(
                    comb=param,seed=i,index=i,sti=sti,maxrate=maxrate,sig=sig,delta_gk=delta_gk,
                    sti_type=sti_type,video=(i==0),save_load=save_load,stim_dura=stim_dura,
                    save_path_video=save_path_video
                    )
                for i in range(n_repeat)
            )
        else:
            results = Parallel(n_jobs=-1)(
                delayed(compute.compute_1_general)(
                    comb=param,seed=i,index=i,sti=sti,maxrate=maxrate,sig=sig,delta_gk=delta_gk,
                    sti_type=sti_type,video=False,save_load=save_load,stim_dura=stim_dura,
                    save_path_video=save_path_video
                    )
                for i in range(n_repeat)
            )
        # 提取所有LFP
        LFP_list = [r['data'].a1.ge.LFP for r in results]
        # LFP_list = [r['LFP_cut'] for r in results]
        # 计算所有频谱
        fft_results = [mya.analyze_LFP_fft(LFP, dt=dt, plot=False) for LFP in LFP_list]
        freqs = fft_results[0][0]
        powers = np.array([fr[1] for fr in fft_results])
        power_mean = np.mean(powers, axis=0)
        power_std = np.std(powers, axis=0)
        
        LFP_results = {
            'freqs': freqs,
            'powers': powers,
            'power_mean': power_mean,
            'power_std': power_std
        }
        if lfp_data_path is None:
            lfp_data_path = f'{state_dir}/1FFT_{maxrate}_{sti_type}_{sig}_{common_path}_{n_repeat}_{stim_dura}_{delta_gk}.file'
        
        if save_lfp:
            with open(lfp_data_path, 'wb') as file:
                pickle.dump(LFP_results,file)

    else: # load data
        if lfp_data_path is None:
            lfp_data_path = f'{state_dir}/1FFT_{maxrate}_{sti_type}_{sig}_{common_path}_{n_repeat}_{stim_dura}_{delta_gk}.file'
        with open(lfp_data_path, 'rb') as file:
            LFP_results = pickle.load(file)
        freqs = LFP_results['freqs']
        powers = LFP_results['powers']
        power_mean = LFP_results['power_mean']
        power_std = LFP_results['power_std']
        
    # 画平均频谱
    if plot:
        draw_LFP_FFT(freqs, power_mean, power_std, 
                    save_path, save_path_beta, save_path_gama, 
                    plotlog='loglog',std_plot=std_plot)
   
    return LFP_results

# repeat computing 2 area FFT of LFP, output beta band and gamma band spectrum(已修改，添加第二份LFP)
def LFP_2area_repeat(param, n_repeat=64, maxrate=500, sig=5, dt=0.1, 
                     plot=True, plot12=False, video=True,stim_dura=10000,
                     save_load=False, save_path_video=None,save_lfp=True,
                     w_12_e=None,w_12_i=None,w_21_e=None,w_21_i=None,
                     save_path_beta=None,save_path_gama=None,save_path=None,
                     save_path_root=LFP_dir,lfp_data_path=None,cmpt=True,
                     sti=True, top_sti=False, sti_type='Uniform', 
                     adapt=False, adapt_type='Gaussian', std_plot=False,
                     new_delta_gk_2=0.5, chg_adapt_range=7):
    ie_r_e1, ie_r_i1, ie_r_e2, ie_r_i2 = param
    common_path = f're1{ie_r_e1:.4f}_ri1{ie_r_i1:.4f}_re2{ie_r_e2:.4f}_ri2{ie_r_i2:.4f}'
    if adapt and top_sti:
        topdown = 'adapt_stim2'
    elif adapt:
        topdown = 'adapt'
    elif top_sti:
        topdown = 'stim2'
    else:
        topdown = 'silnc'
    if save_path_beta is None:
        save_path_beta = f'{save_path_root}/beta_2FFT_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{sig}_{common_path}_{n_repeat}_{stim_dura}'
    if save_path_gama is None:
        save_path_gama = f'{save_path_root}/gama_2FFT_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{sig}_{common_path}_{n_repeat}_{stim_dura}'
    if save_path is None:
        save_path =      f'{save_path_root}/full_2FFT_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{sig}_{common_path}_{n_repeat}_{stim_dura}'
    if cmpt:
        if video:
            results = Parallel(n_jobs=-1)(
                delayed(compute.compute_2_general)(
                    comb=param, seed=i, index=i, sti=sti, maxrate=maxrate, 
                    sig=sig, sti_type=sti_type, video=(i==0), save_load=save_load,
                    save_path_video=save_path_video,stim_dura=stim_dura,
                    w_12_e=w_12_e,w_12_i=w_12_i,w_21_e=w_21_e,w_21_i=w_21_i,
                    top_sti=top_sti, adapt=adapt, adapt_type=adapt_type,
                    new_delta_gk_2=new_delta_gk_2, chg_adapt_range=chg_adapt_range
                    )
                for i in range(n_repeat)
            )
        else:
            results = Parallel(n_jobs=-1)(
                delayed(compute.compute_2_general)(
                    comb=param, seed=i, index=i, sti=sti, maxrate=maxrate, 
                    sig=sig, sti_type=sti_type, video=False, save_load=save_load,
                    save_path_video=save_path_video,stim_dura=stim_dura,
                    w_12_e=w_12_e,w_12_i=w_12_i,w_21_e=w_21_e,w_21_i=w_21_i,
                    top_sti=top_sti, adapt=adapt, adapt_type=adapt_type,
                    new_delta_gk_2=new_delta_gk_2, chg_adapt_range=chg_adapt_range
                    )
                for i in range(n_repeat)
            )
        # 提取所有LFP
        LFP1_list = [r['data'].a1.ge.LFP for r in results]
        LFP2_list = [r['data'].a2.ge.LFP for r in results]
        # LFP1_list = [r['LFP1_cut'] for r in results]
        # LFP2_list = [r['LFP2_cut'] for r in results]
        # 计算所有频谱
        fft1_results = [mya.analyze_LFP_fft(LFP, dt=dt, plot=False) for LFP in LFP1_list]
        freqs1 = fft1_results[0][0]
        powers1 = np.array([fr[1] for fr in fft1_results])
        power_mean1 = np.mean(powers1, axis=0)
        power_std1 = np.std(powers1, axis=0)

        fft2_results = [mya.analyze_LFP_fft(LFP, dt=dt, plot=False) for LFP in LFP2_list]
        freqs2 = fft2_results[0][0]
        powers2 = np.array([fr[1] for fr in fft2_results])
        power_mean2 = np.mean(powers2, axis=0)
        power_std2 = np.std(powers2, axis=0)

        LFP_results = {
            'freqs1': freqs1,
            'powers1': powers1,
            'power_mean1': power_mean1,
            'power_std1': power_std1,
            'freqs2': freqs2,
            'powers2': powers2,
            'power_mean2': power_mean2,
            'power_std2': power_std2,
        }
        if lfp_data_path is None:
            lfp_data_path = f'{state_dir}/2FFT_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{sig}_w{w_12_e}_{w_12_i}_{w_21_e}_{w_21_i}_{common_path}_{new_delta_gk_2}_{n_repeat}_{stim_dura}.file'
        
        if save_lfp:
            with open(lfp_data_path, 'wb') as file:
                pickle.dump(LFP_results,file)
    else: # load data
        if lfp_data_path is None:
            lfp_data_path = f'{state_dir}/2FFT_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{sig}_w{w_12_e}_{w_12_i}_{w_21_e}_{w_21_i}_{common_path}_{new_delta_gk_2}_{n_repeat}_{stim_dura}.file'
        with open(lfp_data_path, 'rb') as file:
            LFP_results = pickle.load(file)
        freqs1 = LFP_results['freqs1']
        powers1 = LFP_results['powers1']
        power_mean1 = LFP_results['power_mean1']
        power_std1 = LFP_results['power_std1']
        freqs2 = LFP_results['freqs2']
        powers2 = LFP_results['powers2']
        power_mean2 = LFP_results['power_mean2']
        power_std2 = LFP_results['power_std2']

    # 画平均频谱
    if plot:
        draw_LFP_FFT(freqs1, power_mean1, power_std1, 
                    save_path=f'{save_path}_1.svg',
                    save_path_beta=f'{save_path_beta}_1.svg',
                    save_path_gama=f'{save_path_gama}_1.svg',
                    plotlog='loglog',std_plot=std_plot)

    ## 第二层LFP
    # 画平均频谱
    if plot12:
        draw_LFP_FFT(freqs2, power_mean2, power_std2, 
                    save_path=f'{save_path}_2.svg',
                    save_path_beta=f'{save_path_beta}_2.svg',
                    save_path_gama=f'{save_path_gama}_2.svg',
                    plotlog='loglog')
        
    return LFP_results

# exam middle point LFP (FFT) (如果画出diff，强制画出1，2的FFT) 有topdown的 - 没topdown的
def LFP_diff_repeat(param1, param2, n_repeat=64, maxrate=500, sig=5, dt=0.1,
                    plot=True, video=True,stim_dura=10000,cmpt=True,
                    save_load=False,save_path_video=None,save_lfp=True,
                    w_12_e=None,w_12_i=None,w_21_e=None,w_21_i=None,
                    save_path_beta=None,save_path_gama=None,save_path=None,
                    save_path_root=LFP_dir,lfp_data_path=None,
                    sti=True, top_sti=False, sti_type='Uniform',
                    adapt=False, adapt_type='Gaussian', std_plot=False,
                    new_delta_gk_2=0.5,chg_adapt_range=7,delta_gk=1):
    ie_r_e1, ie_r_i1, ie_r_e2, ie_r_i2 = param2
    common_path = f're1{ie_r_e1:.4f}_ri1{ie_r_i1:.4f}_re2{ie_r_e2:.4f}_ri2{ie_r_i2:.4f}'
    if adapt and top_sti:
        topdown = 'adapt_stim2'
    elif adapt:
        topdown = 'adapt'
    elif top_sti:
        topdown = 'stim2'
    else:
        topdown = 'silnc'
    if save_path_beta is None:
        save_path_beta = f'{save_path_root}/beta_dFFT_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{sig}_{common_path}_{n_repeat}_{stim_dura}.svg'
    if save_path_gama is None:
        save_path_gama = f'{save_path_root}/gama_dFFT_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{sig}_{common_path}_{n_repeat}_{stim_dura}.svg'
    if save_path is None:
        save_path =      f'{save_path_root}/full_dFFT_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{sig}_{common_path}_{n_repeat}_{stim_dura}.svg'
    fft1 = LFP_1area_repeat(
        param=param1,n_repeat=n_repeat,maxrate=maxrate,sig=sig,dt=dt,
        plot=plot,video=video,stim_dura=stim_dura,cmpt=cmpt,
        save_load=save_load,save_path_video=save_path_video,save_lfp=save_lfp,
        save_path_beta=None,save_path_gama=None,save_path=None,
        save_path_root=save_path_root,lfp_data_path=lfp_data_path,
        sti=sti,sti_type=sti_type,delta_gk=delta_gk,std_plot=std_plot
        )
    fft2 = LFP_2area_repeat(
        param=param2,n_repeat=n_repeat,maxrate=maxrate,sig=sig,dt=dt,
        plot=plot,video=video,stim_dura=stim_dura,cmpt=cmpt,
        save_load=save_load,save_path_video=save_path_video,save_lfp=save_lfp,
        w_12_e=w_12_e,w_12_i=w_12_i,w_21_e=w_21_e,w_21_i=w_21_i,
        save_path_beta=None,save_path_gama=None,save_path=None,
        save_path_root=save_path_root,lfp_data_path=lfp_data_path,
        sti=sti,top_sti=top_sti,sti_type=sti_type,
        adapt=adapt,adapt_type=adapt_type,std_plot=std_plot,
        new_delta_gk_2=new_delta_gk_2,
        chg_adapt_range=chg_adapt_range
        )
    freqs1 = fft1['freqs']
    powers1 = fft1['powers']
    power_mean1 = fft1['power_mean']
    power_std1 = fft1['power_std']
    freqs2 = fft2['freqs1']
    powers2 = fft2['powers1']
    power_mean2 = fft2['power_mean1']
    power_std2 = fft2['power_std1']
    powers_diff = powers2 - powers1
    freqs_diff = freqs2
    power_mean_diff = np.mean(powers_diff, axis=0)
    power_std_diff = np.std(powers_diff, axis=0)

    if plot:
        draw_LFP_FFT(freqs_diff, power_mean_diff, power_std_diff, 
                     save_path, save_path_beta, save_path_gama,
                     plotlog='semilogx',std_plot=std_plot)
        
    LFP_results = {
        'freqs1': freqs1,
        'powers1': powers1,
        'power_mean1': power_mean1, 
        'power_std1': power_std1,
        'freqs2': freqs2,
        'powers2': powers2,
        'power_mean2': power_mean2,
        'power_std2': power_std2,
        'freqs_diff': freqs_diff,
        'power_mean_diff': power_mean_diff,
        'power_std_diff': power_std_diff
    }

    return LFP_results

# 1,2area and diff, compare different sig (used for bottom-up)
def draw_LFP_FFT_compare(param1, param2, n_repeat=64, maxrate=500,cmpt=True,
                         sigs=[0,5,10,15,20,25], dt=0.1,std_plot=False,
                         plot=True, plot_sub=True,video=False,stim_dura=10000,
                         save_load=False,save_path_video=None,save_lfp=True,
                         w_12_e=None,w_12_i=None,w_21_e=None,w_21_i=None,
                         save_path_beta1=None,save_path_gama1=None,save_path1=None,
                         save_path_beta2=None,save_path_gama2=None,save_path2=None,
                         save_path_betad=None,save_path_gamad=None,save_pathd=None,
                         save_path_root=LFP_dir,lfp_data_path=None,
                         sti=True, top_sti=False, sti_type='Uniform', 
                         adapt=False, adapt_type='Gaussian',
                         new_delta_gk_2=0.5):
    
    ie_r_e1, ie_r_i1, ie_r_e2, ie_r_i2 = param2
    common_path1 = f're1{ie_r_e1:.4f}_ri1{ie_r_i1:.4f}'
    common_path2 = f're1{ie_r_e1:.4f}_ri1{ie_r_i1:.4f}_re2{ie_r_e2:.4f}_ri2{ie_r_i2:.4f}'

    if save_path_beta1 is None:
        save_path_beta1 = f'{save_path_root}/beta_1FFT_{maxrate}_{sti_type}T_{common_path1}_{n_repeat}.svg'
    if save_path_beta2 is None:
        save_path_beta2 = f'{save_path_root}/beta_2FFT_{maxrate}_{sti_type}_{common_path2}_{n_repeat}.svg'
    if save_path_betad is None:
        save_path_betad = f'{save_path_root}/beta_dFFT_{maxrate}_{sti_type}_{common_path2}_{n_repeat}.svg'
    if save_path_gama1 is None:
        save_path_gama1 = f'{save_path_root}/gama_1FFT_{maxrate}_{sti_type}_{common_path1}_{n_repeat}.svg'
    if save_path_gama2 is None:
        save_path_gama2 = f'{save_path_root}/gama_2FFT_{maxrate}_{sti_type}_{common_path2}_{n_repeat}.svg'
    if save_path_gamad is None:
        save_path_gamad = f'{save_path_root}/gama_dFFT_{maxrate}_{sti_type}_{common_path2}_{n_repeat}.svg'
    if save_path1 is None:
        save_path1 = f'{save_path_root}/full_1FFT_{maxrate}_{sti_type}_{common_path1}_{n_repeat}.svg'
    if save_path2 is None:
        save_path2 = f'{save_path_root}/full_2FFT_{maxrate}_{sti_type}_{common_path2}_{n_repeat}.svg'
    if save_pathd is None:
        save_pathd = f'{save_path_root}/full_dFFT_{maxrate}_{sti_type}_{common_path2}_{n_repeat}.svg'

    results_1area = []
    results_2area = []
    results_diff = []

    for sig in sigs:
        results = LFP_diff_repeat(
            param1=param1,param2=param2,n_repeat=n_repeat,maxrate=maxrate,sig=sig,dt=dt, 
            plot=plot_sub,video=video,stim_dura=stim_dura,cmpt=cmpt,
            save_load=save_load,save_path_video=save_path_video,save_lfp=save_lfp,
            w_12_e=w_12_e,w_12_i=w_12_i,w_21_e=w_21_e,w_21_i=w_21_i,
            save_path_root=f'{save_path_root}/sub',lfp_data_path=lfp_data_path,
            sti=sti,top_sti=top_sti,sti_type=sti_type,
            adapt=adapt,adapt_type=adapt_type,std_plot=std_plot,
            new_delta_gk_2=new_delta_gk_2,chg_adapt_range=sig
            )
        freqs1 = results['freqs1']
        power_mean1 = results['power_mean1']
        power_std1 = results['power_std1']
        power_mean2 = results['power_mean2']
        power_std2 = results['power_std2']
        power_mean_diff = results['power_mean_diff']
        power_std_diff = results['power_std_diff']
        results_1area.append((sig, freqs1, power_mean1, power_std1))
        results_2area.append((sig, freqs1, power_mean2, power_std2))
        results_diff.append((sig, freqs1, power_mean_diff, power_std_diff))

    if plot:
        # 1 area:
        draw_LFP_FFTs(results=results_1area,
                      save_path=save_path1,
                      save_path_beta=save_path_beta1,
                      save_path_gama=save_path_gama1,
                      plotlog='loglog',
                      std_plot=std_plot)
        # 2 area:
        draw_LFP_FFTs(results=results_2area,
                      save_path=save_path2,
                      save_path_beta=save_path_beta2,
                      save_path_gama=save_path_gama2,
                      plotlog='loglog',
                      std_plot=std_plot)
        # difference:
        draw_LFP_FFTs(results=results_diff,
                      save_path=save_pathd,
                      save_path_beta=save_path_betad,
                      save_path_gama=save_path_gamad,
                      plotlog='semilogx',
                      std_plot=std_plot)

# prediction interaction compare with spontaneous LFP
def LFP_prediction_repeat(param,n_repeat=64,maxrate=500,sig=5,dt=0.1,
                          plot=True,video=True,stim_dura=10000,cmpt=True,
                          save_load=False,save_path_video=None,save_lfp=True,
                          w_12_e=None,w_12_i=None,w_21_e=None,w_21_i=None,
                          save_path=None,save_path_beta=None,save_path_gama=None,
                          save_path_root=LFP_dir,lfp_data_path=None,
                          sti=True,top_sti=False,sti_type='Uniform',
                          adapt=False,adapt_type='Gaussian',std_plot=False,
                          new_delta_gk_2=0.5,chg_adapt_range=7):
    
    ie_r_e1, ie_r_i1, ie_r_e2, ie_r_i2 = param
    common_path = f're1{ie_r_e1:.4f}_ri1{ie_r_i1:.4f}_re2{ie_r_e2:.4f}_ri2{ie_r_i2:.4f}'
    if adapt and top_sti:
        topdown = 'adapt_stim2'
    elif adapt:
        topdown = 'adapt'
    elif top_sti:
        topdown = 'stim2'
    else:
        topdown = 'silnc'
    if save_path_beta is None:
        save_path_beta = f'{save_path_root}/beta_pred_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{sig}_{common_path}_{n_repeat}_{stim_dura}'
    if save_path_gama is None:
        save_path_gama = f'{save_path_root}/gama_pred_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{sig}_{common_path}_{n_repeat}_{stim_dura}'
    if save_path is None:
        save_path      = f'{save_path_root}/full_pred_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{sig}_{common_path}_{n_repeat}_{stim_dura}'

    # prediction type depends on function input
    fft = LFP_2area_repeat(param=param,n_repeat=n_repeat,maxrate=maxrate,sig=sig,dt=dt,
                           plot=plot,video=video,stim_dura=stim_dura,save_lfp=save_lfp,
                           save_load=save_load,save_path_video=save_path_video,
                           w_12_e=w_12_e,w_12_i=w_12_i,w_21_e=w_21_e,w_21_i=w_21_i,
                           save_path_beta=save_path_beta,
                           save_path_gama=save_path_gama,
                           save_path=save_path,cmpt=cmpt,
                           save_path_root=save_path_root,lfp_data_path=lfp_data_path,
                           sti=sti,top_sti=top_sti,sti_type=sti_type,
                           adapt=adapt, adapt_type=adapt_type,std_plot=std_plot,
                           new_delta_gk_2=new_delta_gk_2,
                           chg_adapt_range=chg_adapt_range)

    return fft



# different prediction interaction compare with spontaneous LFP
# 因为要算spon的而被取代, sig=0时就是spon, 所以用上面那个就好
def LFP_diff_prediction_repeat(param,n_repeat=64,maxrate=500,dt=0.1,sig=5,
                               plot=True,video=True,stim_dura=10000,cmpt=True,
                               save_load=False,save_path_video=None,save_lfp=True,
                               w_12_e=None,w_12_i=None,w_21_e=None,w_21_i=None,
                               save_path_betas=None,save_path_gamas=None,save_paths=None,
                               save_path_betap=None,save_path_gamap=None,save_pathp=None,
                               save_path_betad=None,save_path_gamad=None,save_pathd=None,
                               save_path_root=LFP_dir,lfp_data_path=None,
                               sti=True,top_sti=False,sti_type='Uniform',
                               adapt=False,adapt_type='Gaussian',std_plot=False,
                               new_delta_gk_2=0.5,chg_adapt_range=7):
    
    ie_r_e1, ie_r_i1, ie_r_e2, ie_r_i2 = param
    common_path = f're1{ie_r_e1:.4f}_ri1{ie_r_i1:.4f}_re2{ie_r_e2:.4f}_ri2{ie_r_i2:.4f}'
    if adapt and top_sti:
        topdown = 'adapt_stim2'
    elif adapt:
        topdown = 'adapt'
    elif top_sti:
        topdown = 'stim2'
    else:
        topdown = 'silnc'
    if save_path_betas is None:
        save_path_betas = f'{save_path_root}/beta_spon_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{sig}_{common_path}_{n_repeat}_{stim_dura}.svg'
    if save_path_betap is None:
        save_path_betap = f'{save_path_root}/beta_pred_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{sig}_{common_path}_{n_repeat}_{stim_dura}.svg'
    if save_path_betad is None:
        save_path_betad = f'{save_path_root}/beta_diff_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{sig}_{common_path}_{n_repeat}_{stim_dura}.svg'
    if save_path_gamas is None:
        save_path_gamas = f'{save_path_root}/gama_spon_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{sig}_{common_path}_{n_repeat}_{stim_dura}.svg'
    if save_path_gamap is None:
        save_path_gamap = f'{save_path_root}/gama_pred_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{sig}_{common_path}_{n_repeat}_{stim_dura}.svg'
    if save_path_gamad is None:
        save_path_gamad = f'{save_path_root}/gama_diff_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{sig}_{common_path}_{n_repeat}_{stim_dura}.svg'
    if save_paths is None:
        save_paths      = f'{save_path_root}/full_spon_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{sig}_{common_path}_{n_repeat}_{stim_dura}.svg'
    if save_pathp is None:
        save_pathp      = f'{save_path_root}/full_pred_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{sig}_{common_path}_{n_repeat}_{stim_dura}.svg'
    if save_pathd is None:
        save_pathd      = f'{save_path_root}/full_diff_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{sig}_{common_path}_{n_repeat}_{stim_dura}.svg'
        
    # spontaneous
    fft1 = LFP_2area_repeat(param=param,n_repeat=n_repeat,maxrate=maxrate,sig=sig,dt=dt,
                            plot=plot,video=video,stim_dura=stim_dura,save_lfp=save_lfp,
                            save_load=save_load,save_path_video=save_path_video,
                            w_12_e=w_12_e,w_12_i=w_12_i,w_21_e=w_21_e,w_21_i=w_21_i,
                            save_path_beta=save_path_betas,
                            save_path_gama=save_path_gamas,
                            save_path=save_paths,cmpt=cmpt,
                            save_path_root=save_path_root,lfp_data_path=lfp_data_path,
                            sti=False,top_sti=False,sti_type=sti_type,
                            adapt=False,adapt_type=adapt_type,std_plot=std_plot,
                            new_delta_gk_2=new_delta_gk_2,
                            chg_adapt_range=chg_adapt_range)
    # prediction type depends on function input
    fft2 = LFP_2area_repeat(param=param,n_repeat=n_repeat,maxrate=maxrate,sig=sig,dt=dt,
                            plot=plot,video=video,stim_dura=stim_dura,save_lfp=save_lfp,
                            save_load=save_load,save_path_video=save_path_video,
                            w_12_e=w_12_e,w_12_i=w_12_i,w_21_e=w_21_e,w_21_i=w_21_i,
                            save_path_beta=save_path_betap,
                            save_path_gama=save_path_gamap,
                            save_path=save_pathp,cmpt=cmpt,
                            sti=sti,top_sti=top_sti,sti_type=sti_type,
                            adapt=adapt,adapt_type=adapt_type,std_plot=std_plot,
                            new_delta_gk_2=new_delta_gk_2,
                            chg_adapt_range=chg_adapt_range)
    freqs1 = fft1['freqs1']
    powers1 = fft1['powers1']
    power_mean1 = fft1['power_mean1']
    power_std1 = fft1['power_std1']
    freqs2 = fft2['freqs1']
    powers2 = fft2['powers1']
    power_mean2 = fft2['power_mean1']
    power_std2 = fft2['power_std1']
    freqsd = freqs1
    powersd = power_mean2-power_mean1
    power_meand = np.mean(powersd, axis=0)
    power_stdd = np.std(powersd, axis=0)

    if plot:
        draw_LFP_FFT(freqs=freqs1,
                     power_mean=power_meand,
                     power_std=power_stdd,
                     save_path=save_pathd,
                     save_path_beta=save_path_betad,
                     save_path_gama=save_path_gamad,
                     std_plot=std_plot,)
    
    LFP_results = {
        'freqs1': freqs1,
        'power_mean1': power_mean1,
        'power_std1': power_std1,
        'freqs2': freqs2,
        'power_mean2': power_mean2,
        'power_std2': power_std2,
        'freqsd': freqsd,
        'power_meand': power_meand,
        'power_stdd': power_stdd
    }

    return LFP_results

# predicted LFPs under different prediction interaction
# sub_plot控制每个sig的子图，sub_path系列表示每个sig子图的path
def LFPs_prediction_repeat(param,n_repeat=64,maxrate=500,dt=0.1,sigs=[0,5,10,15,20,25],
                           plot=True,plot_sub=False,video=True,stim_dura=10000,cmpt=True,
                           save_load=False,save_path_video=None,save_lfp=True,
                           w_12_e=None,w_12_i=None,w_21_e=None,w_21_i=None,
                           save_path_beta=None,save_path_gama=None,save_path=None,
                           sub_path_beta=None,sub_path_gamma=None,sub_path=None,
                           save_path_root=LFP_dir,sub_path_root=f'{LFP_dir}/sub',
                           sti=True,top_sti=False,sti_type='Uniform',std_plot=False,
                           adapt=False,adapt_type='Gaussian',lfp_data_path=None,
                           new_delta_gk_2=0.5, save_LFPs=True):
        
    ie_r_e1, ie_r_i1, ie_r_e2, ie_r_i2 = param
    common_path = f're1{ie_r_e1:.4f}_ri1{ie_r_i1:.4f}_re2{ie_r_e2:.4f}_ri2{ie_r_i2:.4f}'

    if save_path_beta is None:
        save_path_beta = f'{save_path_root}/beta_pred_Compr_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{common_path}_{n_repeat}_{stim_dura}.svg'
    if save_path_gama is None:
        save_path_gama = f'{save_path_root}/gama_pred_Compr_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{common_path}_{n_repeat}_{stim_dura}.svg'
    if save_path is None:
        save_path      = f'{save_path_root}/full_pred_Compr_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{common_path}_{n_repeat}_{stim_dura}.svg'

    results = []

    for sig in sigs:
        fft=LFP_prediction_repeat(
            param=param, n_repeat=n_repeat,maxrate=maxrate,dt=dt, 
            plot=plot_sub, video=video,stim_dura=stim_dura,cmpt=cmpt,
            save_load=save_load,save_path_video=save_path_video,save_lfp=save_lfp,
            w_12_e=w_12_e,w_12_i=w_12_i,w_21_e=w_21_e,w_21_i=w_21_i,
            save_path=sub_path,save_path_beta=sub_path_beta,save_path_gama=sub_path_gamma,
            save_path_root=sub_path_root,lfp_data_path=lfp_data_path,
            sti=sti,top_sti=top_sti,sti_type=sti_type,sig=sig,
            adapt=adapt,adapt_type=adapt_type,std_plot=std_plot,
            new_delta_gk_2=new_delta_gk_2,chg_adapt_range=sig
            )
        freqs = fft['freqs1']
        powers = fft['powers1']
        power_mean = fft['power_mean1']
        power_std = fft['power_std1']
        results.append((sig, freqs, power_mean, power_std))
    
    if adapt and top_sti:
        topdown = 'adapt_stim2'
    elif adapt:
        topdown = 'adapt'
    elif top_sti:
        topdown = 'stim2'
    else:
        topdown = 'silnc'

    if save_LFPs:
        LFPs_results = {
            'results': results
        }
        LFPs_path = (
            f'{data_dir}/LFPs_pred_{maxrate}_{sti_type}_{adapt_type}_{topdown}_w{w_12_e}_{w_12_i}_{w_21_e}_{w_21_i}_{common_path}_{new_delta_gk_2}_{n_repeat}_{stim_dura}.file'
            )
        with open(LFPs_path, 'wb') as file:
            pickle.dump(LFPs_results, file)

    if plot:
        # prediction
        draw_LFP_FFTs(results=results,save_path=save_path,
                      save_path_beta=save_path_beta,
                      save_path_gama=save_path_gama,
                      plotlog='loglog',std_plot=std_plot,)

# spontaneous, predicted, and thier difference LFPs under different prediction interaction
# sub_plot控制每个sig的子图，sub_path系列表示每个sig子图的path
# 对比topdown和spontanous需要算spon所以无用, 上面那个是算不同sig的, 可替代(有时间改成所有sig-sig(0)的)
def LFPs_diff_prediction_repeat(param,n_repeat=64,maxrate=500,dt=0.1,sigs=[0,5,10,15,20,25],
                                plot=True,plot_sub=False,video=True,stim_dura=10000,cmpt=True,
                                save_load=False,save_path_video=None,save_lfp=True,
                                w_12_e=None,w_12_i=None,w_21_e=None,w_21_i=None,
                                save_path_betas=None,save_path_gamas=None,save_paths=None,
                                save_path_betap=None,save_path_gamap=None,save_pathp=None,
                                save_path_betad=None,save_path_gamad=None,save_pathd=None,
                                sub_path_betas=None,sub_path_gammas=None,sub_paths=None,
                                sub_path_betap=None,sub_path_gammap=None,sub_pathp=None,
                                sub_path_betad=None,sub_path_gammad=None,sub_pathd=None,
                                save_path_root=LFP_dir,sub_path_root=f'{LFP_dir}/sub',
                                sti=True,top_sti=False,sti_type='Uniform',std_plot=False,
                                adapt=False,adapt_type='Gaussian',lfp_data_path=None,
                                new_delta_gk_2=0.5, save_LFPs=True):
        
    ie_r_e1, ie_r_i1, ie_r_e2, ie_r_i2 = param
    common_path = f're1{ie_r_e1:.4f}_ri1{ie_r_i1:.4f}_re2{ie_r_e2:.4f}_ri2{ie_r_i2:.4f}'
    if adapt and top_sti:
        topdown = 'adapt_stim2'
    elif adapt:
        topdown = 'adapt'
    elif top_sti:
        topdown = 'stim2'
    else:
        topdown = 'silnc'
    if save_path_betas is None:
        save_path_betas = f'{save_path_root}/beta_spon_Compr_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{common_path}_{n_repeat}_{stim_dura}.svg'
    if save_path_betap is None:
        save_path_betap = f'{save_path_root}/beta_pred_Compr_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{common_path}_{n_repeat}_{stim_dura}.svg'
    if save_path_betad is None:
        save_path_betad = f'{save_path_root}/beta_diff_Compr_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{common_path}_{n_repeat}_{stim_dura}.svg'
    if save_path_gamas is None:
        save_path_gamas = f'{save_path_root}/gama_spon_Compr_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{common_path}_{n_repeat}_{stim_dura}.svg'
    if save_path_gamap is None:
        save_path_gamap = f'{save_path_root}/gama_pred_Compr_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{common_path}_{n_repeat}_{stim_dura}.svg'
    if save_path_gamad is None:
        save_path_gamad = f'{save_path_root}/gama_diff_Compr_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{common_path}_{n_repeat}_{stim_dura}.svg'
    if save_paths is None:
        save_paths      = f'{save_path_root}/full_spon_Compr_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{common_path}_{n_repeat}_{stim_dura}.svg'
    if save_pathp is None:
        save_pathp      = f'{save_path_root}/full_pred_Compr_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{common_path}_{n_repeat}_{stim_dura}.svg'
    if save_pathd is None:
        save_pathd      = f'{save_path_root}/full_diff_Compr_{maxrate}_{sti_type}_{adapt_type}_{topdown}_{common_path}_{n_repeat}_{stim_dura}.svg'

    results_spon = []
    results_pred = []
    results_diff = []

    for sig in sigs:
        fft = LFP_diff_prediction_repeat(
            param=param,n_repeat=n_repeat,maxrate=maxrate,dt=dt, 
            plot=plot_sub,video=video,stim_dura=stim_dura,cmpt=cmpt,
            save_load=save_load,save_path_video=save_path_video,save_lfp=save_lfp,
            w_12_e=w_12_e,w_12_i=w_12_i,w_21_e=w_21_e,w_21_i=w_21_i,
            save_paths=sub_paths,save_path_betas=sub_path_betas,save_path_gamas=sub_path_gammas,
            save_pathp=sub_pathp,save_path_betap=sub_path_betap,save_path_gamap=sub_path_gammap,
            save_pathd=sub_pathd,save_path_betad=sub_path_betad,save_path_gamad=sub_path_gammad,
            save_path_root=sub_path_root,lfp_data_path=lfp_data_path,
            sti=sti,top_sti=top_sti,sti_type=sti_type,sig=sig,
            adapt=adapt,adapt_type=adapt_type,std_plot=std_plot,
            new_delta_gk_2=new_delta_gk_2, chg_adapt_range=sig
            )
        freqs1 = fft['freqs1']
        power_mean1 = fft['power_mean1']
        power_std1 = fft['power_std1']
        freqs2 = fft['freqs2']
        power_mean2 = fft['power_mean2']
        power_std2 = fft['power_std2']
        freqsd = fft['freqsd']
        power_meand = fft['power_meand']
        power_stdd = fft['power_stdd']
        results_spon.append((sig, freqs1, power_mean1, power_std1))
        results_pred.append((sig, freqs2, power_mean2, power_std2))
        results_diff.append((sig, freqsd, power_meand, power_stdd))
    

    if save_LFPs:
        LFPs_results = {
            'results_spon': results_spon,
            'results_pred': results_pred,
            'results_diff': results_diff
        }
        LFPs_path = (
            f'{data_dir}/LFPs_predd_{maxrate}_{sti_type}_{adapt_type}_{topdown}_w{w_12_e}_{w_12_i}_{w_21_e}_{w_21_i}_{common_path}_{new_delta_gk_2}_{n_repeat}_{stim_dura}.file'
            )
        with open(LFPs_path, 'wb') as file:
            pickle.dump(LFPs_results, file)

    if plot:
        # spontateous
        draw_LFP_FFTs(results=results_spon,save_path=save_paths,
                      save_path_beta=save_path_betas,
                      save_path_gama=save_path_gamas,
                      plotlog='loglog',std_plot=std_plot,)
        # prediction
        draw_LFP_FFTs(results=results_pred,save_path=save_pathp,
                      save_path_beta=save_path_betap,
                      save_path_gama=save_path_gamap,
                      plotlog='loglog',std_plot=std_plot,)
        # difference
        draw_LFP_FFTs(results=results_diff,save_path=save_pathd,
                      save_path_beta=save_path_betad,
                      save_path_gama=save_path_gamad,
                      plotlog='semilogx',std_plot=std_plot,)

#%% Execution area
try:
    send_email.send_email('Begin Running', 'ie_search.main running')
    #%% test
    # param = (1.8512390285440765, 2.399131446733395)
    # 1st pair (alpha <= 1.3)
    # param12 = (2.449990451446889, 1.795670364314891, 2.399131446733395, 1.8512390285440765)
    # 2nd pair(more d(r_rf), but alpha<=1.5)
    # critical zone 右上角的点 - gamma peak 小
    param_ne = (2.67,2.03)
    # critical zone 左下角的点 - gamma peak 正常
    param_sw = (2.22, 1.64)
    # 右下边缘，rf最小，alpha~1.5 - 双峰 gamma peak
    param_se  = (2.501407742047704, 1.8147028535939709)
    # 左上边缘，rf最大，alpha~1.5 - 没有 gamma peak
    param_nw  = (2.425126038006674, 1.927524600435643)
    # 中心点 - gamma peak 较小
    # param1 = (2.4331,1.8447)
    # param12 = (2.22, 1.64, 2.425126038006674, 1.927524600435643)

    # 第一层临界域内找参数
    def vary_ie_ratio(dx=0,dy=0):
        # critical zone 右上角的点 - gamma peak 小
        param_ne = (2.67,2.03)
        # critical zone 左下角的点 - gamma peak 正常
        param_sw = (2.22, 1.64)
        # 右下边缘，rf最小，alpha~1.5 - 双峰 gamma peak
        param_se  = (2.501407742047704, 1.8147028535939709)
        # 左上边缘，rf最大，alpha~1.5 - 没有 gamma peak
        param_nw  = (2.425126038006674, 1.927524600435643)
        # 使用 numpy 数组做向量运算
        p_ne = np.array(param_ne, dtype=float)
        p_sw = np.array(param_sw, dtype=float)
        p_se = np.array(param_se, dtype=float)
        p_nw = np.array(param_nw, dtype=float)
        param_c0 = (p_ne + p_sw + p_se + p_nw) / 4.0
        param_vec_hrz = (p_se - p_nw) / 2.0
        param_vec_vtc = (p_sw - p_ne) / 2.0
        # dx朝右下, dy朝左下
        param = tuple(param_c0 + param_vec_hrz*dx + param_vec_vtc*dy)
        return param
    
    #%% 单层挑参数算数据、视频
    def compute_data():
        # 第一层参数:
        param_area1 = vary_ie_ratio(dx=0,dy=1)
        # 第二层参数:
        param_area2 = (1.84138, 1.57448)
        # 双层参数组合:
        param_area12 = param_area1 + param_area2

        ## 单层算数据,输出视频
        # 哪一层
        for delta_gk in (1, 2):
            # delta_gk=1
            if delta_gk == 1:
                param=param_area1
            elif delta_gk == 2:
                param=param_area2
            ie_r_e1, ie_r_i1 = param
            common_path = f're{ie_r_e1:.4f}_ri{ie_r_i1:.4f}'
            # 激励相关
            sti=False
            sti_type='Uniform'
            sig=5
            maxrate=1000
            stim_dura=1000
            window=10
            if sti:
                input=f'on{maxrate}_{sti_type}_{sig}'
            else:
                input='off'
            
            data_path=f"{data_dir}/1data_{common_path}_{input}_{delta_gk}_win{window}.file"
            video_path=None
            result = compute.compute_1_general(comb=param,stim_dura=stim_dura,
                                               sti=sti,maxrate=maxrate,sti_type=sti_type,
                                               video=True,save_path_video=video_path,
                                               save_load=True,save_path_data=data_path,
                                               window=window,delta_gk=delta_gk,sig=sig)
    def conpute_data2():
        # 第一层参数:
        param_area1 = vary_ie_ratio(dx=0,dy=1)
        # 第二层参数:
        param_area2 = (1.84138, 1.57448)
        # 双层参数组合:
        param_area12 = param_area1 + param_area2

        ## 双层算数据,输出视频
        ie_r_e1, ie_r_i1, ie_r_e2, ie_r_i2 = param_area12
        common_path = f're1{ie_r_e1:.4f}_ri1{ie_r_i1:.4f}_re2{ie_r_e2:.4f}_ri2{ie_r_i2:.4f}'
        # 激励相关
        sti=False
        sti_type='Uniform'
        sig=5
        maxrate=1000
        stim_dura=1000
        window=10
        
        adapt = False
        top_sti = False
        adapt_type = 'Uniform'

        if sti:
            input=f'on{maxrate}_{sti_type}_{sig}'
        else:
            input='off'

        if adapt and top_sti:
            topdown = 'adapt_stim2'
        elif adapt:
            topdown = 'adapt'
        elif top_sti:
            topdown = 'stim2'
        else:
            topdown = 'silnc'

        data_path=f"{data_dir}/2data_{common_path}_{input}_{topdown}_win{window}.file"
        video_path=None
        result = compute.compute_2_general(comb=param_area12,stim_dura=stim_dura,
                                           sti=sti,maxrate=maxrate,sti_type=sti_type,
                                           adapt=adapt,adapt_type=adapt_type,top_sti=top_sti,
                                           video=True,save_path_video=video_path,
                                           save_load=True,save_path_data=data_path,
                                           window=window,sig=sig,chg_adapt_range=sig)
        

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
    # evalution_search(compute=False, conf_level=0.99, 
    #                  delta_gk=2, remove_outlier=False)

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
    # # 邻近的两个点，1000hz下第一个rf=17.66,第二个rf=15.35。试试2000hz，5000hz
    # param1 = (1.8999,1.6314)
    # param2 = (1.9009,1.6124)
    # maxrate=100
    # receptive_field_repeat(param=param1, n_repeat=64, maxrate=maxrate, plot=True, video1=True)
    # receptive_field_repeat(param=param2, n_repeat=64, maxrate=maxrate, plot=True, video1=True)

    #%% search receptive field
    # result = find_max_min_receptive_field(n_repeat=64, maxrate=1000)
    # distribution search
    def distribution_search():
        # 第一层
        # maxrate = 1000
        # delta_gk = 1
        # 第二层
        # maxrate = 2000
        # delta_gk = 2
        delta_gk=2
        range_path = f'{state_dir}/critical_ellipse{delta_gk}.file'
        result = find_receptive_field_distribution_in_range(n_repeat=64, 
                                                            range_path=range_path, 
                                                            maxrate=2000, 
                                                            n_sample=300,
                                                            delta_gk=delta_gk,
                                                            sample_type='Hull')

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

    #     save_pathr = f'{recfield_dir}/field_zratio{n_repeat}_{maxrate}fr_ext{common_path}.svg'
    #     save_pathd = f'{recfield_dir}/field_zdiff{n_repeat}_{maxrate}fr_ext{common_path}.svg'
        
    #     plt.figure(figsize=(5,5))
    #     plt.plot(sigs, ratios, 'o-')
    #     plt.xlabel('Stimuli size')
    #     plt.ylabel('Whole field mean firing rate ratio')
    #     plt.title('Whole field mean firing rate ratio vs. stimuli size')
    #     plt.savefig(save_pathr, dpi=600, format='svg')

    #     plt.figure(figsize=(5,5))
    #     plt.plot(sigs, diffs, 'o-')
    #     plt.xlabel('Stimuli size')
    #     plt.ylabel('Whole field mean firing rate difference')
    #     plt.title('Whole field mean firing rate difference vs. stimuli size')
    #     plt.savefig(save_pathd, dpi=600, format='svg')
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

    def draw_LFP_FFT_1area_repeat(n_repeat=64,sig=0,dx1=0.0,dy1=0.0,
                                  save_path_beta=None,
                                  save_path_gama=None,
                                  save_path=None):
        param=vary_ie_ratio(dx=dx1,dy=dy1)
        LFP_1area_repeat(param=param,n_repeat=n_repeat,maxrate=500,sig=sig,dt=0.1,
                         plot=True,video=True,save_load=False,cmpt=True,
                         save_path=save_path,
                         save_path_beta=save_path_beta,
                         save_path_gama=save_path_gama)
    # 这个还是认为第二层可以用第一层参数空间，已被淘汰
    def draw_LFP_FFT_2area_repeat(n_repeat=64,sig=0,dx1=0.0,dy1=0.0,dx2=0.0,dy2=0.0,
                                  w_12_e=None,w_12_i=None,w_21_e=None,w_21_i=None,
                                  save_path_beta=None,
                                  save_path_gama=None,
                                  save_path=None):
        param1=vary_ie_ratio(dx=dx1,dy=dy1)
        param2=vary_ie_ratio(dx=dx2,dy=dy2)
        param12=param1+param2
        LFP_2area_repeat(param=param12,n_repeat=n_repeat,maxrate=500,sig=sig,dt=0.1,
                         plot=True,video=True,save_load=False,
                         w_12_e=w_12_e,w_12_i=w_12_i,w_21_e=w_21_e,w_21_i=w_21_i,
                         save_path=save_path,cmpt=True,
                         save_path_beta=save_path_beta,
                         save_path_gama=save_path_gama)

    ## 第一层由dxdy指定，第二层直接指定的双层LFP计算，且画出第二层LFP
    def draw_LFP_FFT_2area_repeat2(n_repeat=64,sig=0,dx=0.0,dy=0.0,param2=None,
                                  w_12_e=None,w_12_i=None,w_21_e=None,w_21_i=None,
                                  save_path_beta=None,
                                  save_path_gama=None,
                                  save_path=None):
        param1=vary_ie_ratio(dx=dx,dy=dy)
        param2=param2
        param12=param1+param2
        LFP_2area_repeat(param=param12,n_repeat=n_repeat,maxrate=500,sig=sig,dt=0.1,
                         plot=True,plot12=True,video=True,save_load=False,
                         w_12_e=w_12_e,w_12_i=w_12_i,w_21_e=w_21_e,w_21_i=w_21_i,
                         save_path=save_path,cmpt=True,
                         save_path_beta=save_path_beta,
                         save_path_gama=save_path_gama)
    ## 第一层由dxdy指定，第二层直接指定的双层LFP计算
    # dx=0
    # dy=1
    # # param2 = (1.81273,1.53026)
    # param2 = (1.84138, 1.57448)
    # for w in [2.2, 2.3, 2.4, 2.5]:
    #     w_12_e=w
    #     w_12_i=w
    #     w_21_e=w
    #     w_21_i=w
    #     temp_dir=f'./{LFP_dir}/r{dx}_{dy}_{param2}w{w}'
    #     # Path(temp_dir1).mkdir(parents=True, exist_ok=True)
    #     Path(temp_dir).mkdir(parents=True, exist_ok=True)
    #     path=f'./{temp_dir}/r{dx}_{dy}_{param2}w{w}'
    #     draw_LFP_FFT_2area_repeat2(dx=dx,dy=dy,param2=param2,
    #                             w_12_e=w_12_e,w_12_i=w_12_i,w_21_e=w_21_e,w_21_i=w_21_i,
    #                             save_path=f'./{path}_whole.svg',
    #                             save_path_beta=f'./{path}_beta.svg',
    #                             save_path_gama=f'./{path}_gamma.svg')

    ## 这里在第一层椭圆临界域内尝试计算特殊点的LFP，就连第二层也在第一层临界域内取值
    # dx1=0.0
    # dy1=1.0
    # dx2=-1.0 # 0,-1,1
    # dy2=-1.0 # 0,1,-1
    # w_12_e=2.7
    # w_12_i=2.7
    # w_21_e=2.7
    # w_21_i=2.7
    # for dy2 in [0.0, 1.0, -1.0]:
    #     for dx2 in [0.0, -1.0, 1.0]:
    #         temp_dir1=f'./{LFP_dir}/r{dx1}_{dy1}'
    #         temp_dir2=f'./{LFP_dir}/r{dx1}_{dy1}_{dx2}_{dy2}w{w_12_e}_{w_12_i}_{w_21_e}_{w_21_i}'
    #         # Path(temp_dir1).mkdir(parents=True, exist_ok=True)
    #         Path(temp_dir2).mkdir(parents=True, exist_ok=True)
    #         path_1=f'./{temp_dir1}/r{dx1}_{dy1}'
    #         path_2=f'./{temp_dir2}/r{dx1}_{dy1}_{dx2}_{dy2}w{w_12_e}_{w_12_i}_{w_21_e}_{w_21_i}'
    #         # draw_LFP_FFT_1area_repeat(dx1=dx1,dy1=dy1,
    #         #                           save_path=f'./{path_1}_whole.svg',
    #         #                           save_path_beta=f'./{path_1}_beta.svg',
    #         #                           save_path_gama=f'./{path_1}_gamma.svg')
    #         draw_LFP_FFT_2area_repeat(dx1=dx1,dy1=dy1,dx2=dx2,dy2=dy2,
    #                                 w_12_e=w_12_e,w_12_i=w_12_i,w_21_e=w_21_e,w_21_i=w_21_i,
    #                                 save_path=f'./{path_2}_whole.svg',
    #                                 save_path_beta=f'./{path_2}_beta.svg',
    #                                 save_path_gama=f'./{path_2}_gamma.svg')

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
    # param1=vary_ie_ratio(dx=-0.2,dy=1)
    # param2=vary_ie_ratio(dx=-0.2,dy=1)
    # param12 = param1+param2
    # draw_LFP_FFT_compare(param1=param1, param2=param12, 
    #                      n_repeat=64, maxrate=500, sti_type='Uniform')
    
    # print('computing start')
    # draw_receptive_field2(param=param1, n_repeat=64, le=le,li=li)
    # print('set 1 executed')
    # send_email.send_email('set 1 executed', 'set 1 executed')
    # draw_receptive_field2(param=param2, n_repeat=64, le=le,li=li)
    # print('set 2 executed')
    # send_email.send_email('set 2 executed', 'set 2 executed')

    #%% plot trajectory
    # # area 1 parameter
    # param = vary_ie_ratio(dx=0,dy=1)
    # # 忘了是啥
    # param = (1.899,1.6314)
    # area 2 parameter
    # param= (1.84138, 1.57448)
    # result = compute.compute_1_general(comb=param,video=True,window=1,stim_dura=1000,delta_gk=2)
    # data = result['data']
    # centre = data.a1.ge.centre_mass.centre
    # save_path_trajectory = f"{graph_dir}/Levy_trajecotry2.svg"
    # conti = mya.unwrap_periodic_path(centre=centre)
    # mya.plot_trajectory(data=conti,title='Levy package trajectory',save_path=save_path_trajectory)

    #%% new comparable lfp fft (vary weight and check fft) (bottom up)
    def bottom_up_LFP_compare():
        param1=vary_ie_ratio(dx=0,dy=1)
        param2=(1.84138, 1.57448)
        param12=param1+param2
        maxrate=1000
        sti_type = 'Uniform'
        # w=2.4
        w_12_e=2.4
        w_12_i=2.4
        w_21_e=2.4
        w_21_i=2.4
        # w_12_e=0.0
        # w_12_i=0.0
        # w_21_e=3.0
        # w_21_i=3.0
        # w=(w_12_e,w_12_i,w_21_e,w_21_i)
        n_repeat=128
        stim_dura=10000

        ie_r_e1, ie_r_i1, ie_r_e2, ie_r_i2 = param12
        common_path1 = f're1{ie_r_e1:.4f}_ri1{ie_r_i1:.4f}'
        common_path2 = f're1{ie_r_e1:.4f}_ri1{ie_r_i1:.4f}_re2{ie_r_e2:.4f}_ri2{ie_r_i2:.4f}'

        # bottom_up表示只有前馈，top_down表示只有反馈
        temp_dir=f'./{LFP_dir}/bottomup_{maxrate}_{sti_type}_w{w_12_e}_{w_12_i}_{w_21_e}_{w_21_i}_{n_repeat}_{stim_dura}'
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        Path(f'{temp_dir}/sub').mkdir(parents=True, exist_ok=True)

        # path_beta1=f'./{temp_dir}/beta_1_{common_path1}_{n_repeat}.svg'
        # path_gama1=f'./{temp_dir}/gama_1_{common_path1}_{n_repeat}.svg'
        # path_full1=f'./{temp_dir}/full_1_{common_path1}_{n_repeat}.svg'
        # path_beta2=f'./{temp_dir}/beta_2_{common_path2}_{n_repeat}.svg'
        # path_gama2=f'./{temp_dir}/gama_2_{common_path2}_{n_repeat}.svg'
        # path_full2=f'./{temp_dir}/full_2_{common_path2}_{n_repeat}.svg'
        # path_betad=f'./{temp_dir}/beta_d_{common_path2}_{n_repeat}.svg'
        # path_gamad=f'./{temp_dir}/gama_d_{common_path2}_{n_repeat}.svg'
        # path_fulld=f'./{temp_dir}/full_d_{common_path2}_{n_repeat}.svg'

        # draw_LFP_FFT_compare这个函数就是默认bottom up的
        draw_LFP_FFT_compare(
            param1=param1,param2=param12,n_repeat=n_repeat,maxrate=maxrate,
            w_12_e=w_12_e,w_12_i=w_12_i,w_21_e=w_21_e,w_21_i=w_21_i,
            save_path_root=temp_dir,std_plot=False,cmpt=False,
            sti=True,top_sti=False,sti_type=sti_type,stim_dura=stim_dura
            # save_path_beta1=path_beta1,save_path_gama1=path_gama1,save_path1=path_full1,
            # save_path_beta2=path_beta2,save_path_gama2=path_gama2,save_path2=path_full2,
            # save_path_betad=path_betad,save_path_gamad=path_gamad,save_pathd=path_fulld
            )

    #%% test adaptation and stimulus 2
    # param1=vary_ie_ratio(dx=0,dy=1)
    # param2=(1.84138, 1.57448)
    # param12=param1+param2
    # maxrate=1000
    # w=2.4
    # n_repeat=128
    # compute.compute_2_general(comb=param12, sti=False, maxrate=500, adapt=False, top_sti=True, 
    #                           sig=5, sti_type='Uniform', adapt_type='Gaussian',
    #                           video=True, stim_dura=2000, new_delta_gk_2=0.5,
    #                           chg_adapt_range=5)

    #%% LFP FFT under top-down interaction
    # param1=vary_ie_ratio(dx=0,dy=1)
    # param2=(1.84138, 1.57448)
    # param12=param1+param2
    # maxrate=1000
    # n_repeat=128
    # w=2.6
    
    # ie_r_e1, ie_r_i1, ie_r_e2, ie_r_i2 = param12
    # common_path1 = f're1{ie_r_e1:.4f}_ri1{ie_r_i1:.4f}'
    # common_path2 = f're1{ie_r_e1:.4f}_ri1{ie_r_i1:.4f}_re2{ie_r_e2:.4f}_ri2{ie_r_i2:.4f}'

    # temp_dir_adapt=f'./{LFP_dir}/adapt_rate{maxrate}_w{w}_re{n_repeat}'
    # Path(temp_dir_adapt).mkdir(parents=True, exist_ok=True)
    # temp_dir_stim2=f'./{LFP_dir}/stim2_rate{maxrate}_w{w}_re{n_repeat}'
    # Path(temp_dir_stim2).mkdir(parents=True, exist_ok=True)
    # temp_dir_spont=f'./{LFP_dir}/spont_rate{maxrate}_w{w}_re{n_repeat}'
    # Path(temp_dir_spont).mkdir(parents=True, exist_ok=True)

    # path_beta_adapt=f'./{temp_dir_adapt}/beta_{common_path2}_{n_repeat}.svg'
    # path_gama_adapt=f'./{temp_dir_adapt}/gama_{common_path2}_{n_repeat}.svg'
    # path_full_adapt=f'./{temp_dir_adapt}/full_{common_path2}_{n_repeat}.svg'
    # path_beta_stim2=f'./{temp_dir_stim2}/beta_{common_path2}_{n_repeat}.svg'
    # path_gama_stim2=f'./{temp_dir_stim2}/gama_{common_path2}_{n_repeat}.svg'
    # path_full_stim2=f'./{temp_dir_stim2}/full_{common_path2}_{n_repeat}.svg'
    # path_beta_spont=f'./{temp_dir_spont}/beta_{common_path2}_{n_repeat}.svg'
    # path_gama_spont=f'./{temp_dir_spont}/gama_{common_path2}_{n_repeat}.svg'
    # path_full_spont=f'./{temp_dir_spont}/full_{common_path2}_{n_repeat}.svg'
    # # adaptation
    # LFP_2area_repeat(param=param12,n_repeat=n_repeat,maxrate=maxrate,
    #                  sti=False,top_sti=False,sti_type='Uniform',
    #                  adapt=True,adapt_type='Gaussian',
    #                  new_delta_gk_2=0.5,chg_adapt_range=5,
    #                  w_12_e=w,w_12_i=w,w_21_e=w,w_21_i=w,
    #                  save_path=path_full_adapt,
    #                  save_path_beta=path_beta_adapt,
    #                  save_path_gama=path_gama_adapt)
    # # stimulus
    # LFP_2area_repeat(param=param12,n_repeat=n_repeat,maxrate=maxrate,
    #                  sti=False,top_sti=True,sti_type='Uniform',
    #                  adapt=False,adapt_type='Gaussian',
    #                  new_delta_gk_2=0.5,chg_adapt_range=5,
    #                  w_12_e=w,w_12_i=w,w_21_e=w,w_21_i=w,
    #                  save_path=path_full_stim2,
    #                  save_path_beta=path_beta_stim2,
    #                  save_path_gama=path_gama_stim2)
    # # spontaneous
    # LFP_2area_repeat(param=param12,n_repeat=n_repeat,maxrate=maxrate,
    #                  sti=False,top_sti=False,sti_type='Uniform',
    #                  adapt=False,adapt_type='Gaussian',
    #                  new_delta_gk_2=0.5,chg_adapt_range=5,
    #                  w_12_e=w,w_12_i=w,w_21_e=w,w_21_i=w,
    #                  save_path=path_full_spont,
    #                  save_path_beta=path_beta_spont,
    #                  save_path_gama=path_gama_spont)

    #%% LFP FFT under different type and different size top-down interaction
    def top_down_LFP_compare():
        param1=vary_ie_ratio(dx=0,dy=1)
        param2=(1.84138, 1.57448)
        param12=param1+param2
        maxrate=1000
        new_delta_gk_2=0.5
        adapt_type = 'Uniform'
        sti_type = 'Uniform'
        n_repeat=128
        w=2.8
        w_12_e=2.4
        w_12_i=2.4
        w_21_e=2.4
        w_21_i=2.4
        
        
        ie_r_e1, ie_r_i1, ie_r_e2, ie_r_i2 = param12
        common_path = f're1{ie_r_e1:.4f}_ri1{ie_r_i1:.4f}_re2{ie_r_e2:.4f}_ri2{ie_r_i2:.4f}'

        temp_dir_adapt=f'./{LFP_dir}/compr_adapt{new_delta_gk_2}_w{w}_re{n_repeat}'
        Path(temp_dir_adapt).mkdir(parents=True, exist_ok=True)
        temp_dir_stim2=f'./{LFP_dir}/compr_stim2{maxrate}_w{w}_re{n_repeat}'
        Path(temp_dir_stim2).mkdir(parents=True, exist_ok=True)

        sub_temp_dir_adapt=f'./{LFP_dir}/compr_adapt{new_delta_gk_2}_{adapt_type}_w{w_12_e}_{w_12_i}_{w_21_e}_{w_21_i}_re{n_repeat}/sub'
        Path(sub_temp_dir_adapt).mkdir(parents=True, exist_ok=True)
        sub_temp_dir_stim2=f'./{LFP_dir}/compr_stim2{maxrate}_{sti_type}_w{w_12_e}_{w_12_i}_{w_21_e}_{w_21_i}_re{n_repeat}/sub'
        Path(sub_temp_dir_stim2).mkdir(parents=True, exist_ok=True)

        # adaptation
        LFPs_prediction_repeat(param=param12,n_repeat=n_repeat,maxrate=maxrate,
                               plot=True,plot_sub=True,video=True,save_load=False,
                               w_12_e=w,w_12_i=w,w_21_e=w,w_21_i=w,cmpt=True,
                               save_path_root=temp_dir_adapt,sub_path_root=sub_temp_dir_adapt,
                               sti=False,top_sti=False,sti_type=sti_type,
                               adapt=True,adapt_type=adapt_type,
                               new_delta_gk_2=new_delta_gk_2,save_LFPs=True)
        
        # stimulus
        LFPs_prediction_repeat(param=param12,n_repeat=n_repeat,maxrate=maxrate,
                               plot=True,plot_sub=True,video=True,save_load=False,
                               w_12_e=w,w_12_i=w,w_21_e=w,w_21_i=w,cmpt=True,
                               save_path_root=temp_dir_stim2,sub_path_root=sub_temp_dir_stim2,
                               sti=False,top_sti=True,sti_type=sti_type,
                               adapt=False,adapt_type=adapt_type,
                               new_delta_gk_2=new_delta_gk_2,save_LFPs=True)

    #%% repeat 2 area computation recetive field
    # param1=vary_ie_ratio(dx=0,dy=1)
    # param2=(1.84138, 1.57448)
    # param=param1+param2
    # n_repeat=128
    # stim_dura=1000
    # root_path=None
    # # 单层 area 1
    # msd_pdx_1(param=param1,n_repeat=n_repeat,stim_dura=stim_dura,window=15,
    #           video=True,save_load=False,delta_gk=1,
    #           data_root=None,root_path=root_path,
    #           cmpt=True,save_data=True,plot=True,
    #           msd_path=None,pdx_path=None,msd_pdx_path=None)
    # # 单层 area 2
    # msd_pdx_1(param=param2,n_repeat=n_repeat,stim_dura=stim_dura,window=15,
    #           video=True,save_load=False,delta_gk=2,
    #           data_root=None,root_path=root_path,
    #           cmpt=True,save_data=True,plot=True,
    #           msd_path=None,pdx_path=None,msd_pdx_path=None)
    # # 双层
    # w_12_e=2.4
    # w_12_i=2.4
    # w_21_e=2.4
    # w_21_i=2.4
    # msd_pdx_2(param=param,n_repeat=n_repeat,stim_dura=stim_dura,window=15,
    #           video=True,save_load=False,
    #           data_root=None,root_path=root_path,
    #           cmpt=True,save_data=True,plot=True,
    #           msd_path=None,pdx_path=None,msd_pdx_path=None,
    #           w_12_e=w_12_e,w_12_i=w_12_i,w_21_e=w_21_e,w_21_i=w_21_i)

    bottom_up_LFP_compare()

    send_email.send_email('code executed - server 1', 'ie_search.main accomplished')
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
