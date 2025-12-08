import brian2.numpy_ as np
from brian2.only import *
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull
from scipy.stats import chi2
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
from myscript.ie_search.compute_MSD_pdx import compute_MSD_pdx_repeat_and_packet_exist
from myscript.ie_search.compute_general import compute_1
from myscript.ie_search.compute_general import compute_2
from myscript.ie_search.compute_general import compute_1_general
from myscript.ie_search.compute_general import compute_2_general
import myscript.ie_search.utils as utils
import myscript.ie_search.batch_repeat as batch_repeat
import myscript.ie_search.load_repeat as load_repeat

import logging
logging.getLogger('brian2').setLevel(logging.WARNING)

initial_param = {
    'ie_r_e1': np.linspace(1.8, 2.5, 8),
    'ie_r_i1': np.linspace(1.8, 2.5, 8)
}
# generate looping parameter combinations
parents = list(itertools.product(*initial_param.values()))
# get total looping number
loop_total = len(parents)
def eval_func(comb, index=1, delta_gk=1):
    # result = compute_MSD_pdx(comb=comb, index=index, video=False)
    result = compute_1_general(comb=comb, index=index, video=False, delta_gk=delta_gk)
    msd = result['msd']
    jump_interval = result['jump_interval']
    pdx = result['pdx']
    spk_rate = result['spk_rate']
    centre = result['centre']
    exist = utils.wave_packet_exist(spk_rate=spk_rate, centre=centre, r=0.6)
    motion_critical, info = utils.is_critical_state(msd=msd, 
                                                    jump_interval=jump_interval, 
                                                    pdx=pdx)
    alpha = info['alpha']
    critical = exist and motion_critical
    return critical, alpha
# 似乎被淘汰了，因为重复计算数量爆炸，后续没有使用
def eval_func_repeat(comb, n_repeat=64):
    result = compute_MSD_pdx_repeat_and_packet_exist(param=comb, n_repeat=n_repeat)
    msd = result['msd_mean']
    jump_interval = result['jump_interval']
    pdx = result['pdx']
    packet_exist = result['packet_exist']
    # packet_true_count = result['packet_true_count']
    motion_critical, info = utils.is_critical_state(msd=msd,
                                                    jump_interval=jump_interval,
                                                    pdx=pdx)
    alpha = info['alpha']
    critical = packet_exist and motion_critical
    return critical, alpha

def generate_children(parent, r, n_child):
    '''
    generate n_child children for a parent (Gaussian perturbation)
    parent: parent parameter
    r: 3\sigma
    n_child: number of offspring
    '''
    parent = np.array(parent)
    return [tuple(parent + r/3 * np.random.randn(*parent.shape)) for _ in range(n_child)]

def evolve_search(initial_params, eval_func, r0=1.0, k=0.2, max_gen=10, n_child=5, n_jobs=-1, 
                  max_children_per_gen=1000, delta_gk=1):
    parents = [{'param': param, 'alpha': None, 'n_child_': n_child, 'critical': None, 'parent_alpha': None} 
               for param in initial_params]
    history = []
    for gen in range(max_gen):
        print(f'Generation {gen}, parent num: {len(parents)}')
        # 1. 并行评估所有父代
        results = Parallel(n_jobs=n_jobs)(
            delayed(eval_func)(comb=parent['param'], delta_gk=delta_gk)
            for parent in parents
        )
        # 2. 更新父代的alpha和critical
        for parent, (critical, alpha) in zip(parents, results):
            parent['critical'] = critical
            parent['alpha'] = alpha
        # 3. 只保留critical父代
        critical_parents = [p for p in parents if p['critical']]
        gen_info = [dict(p) for p in parents]  # 记录本代所有父代信息
        if not critical_parents:
            print('No critical parents left, evolution stops.')
            history.append(gen_info)
            break
        # 4. 评估n_child_调整（必须在生成子代前！）
        for parent in critical_parents:
            if gen == 0 or parent['parent_alpha'] is None:
                parent['n_child_'] = n_child  # 第0代或无parent_alpha，直接用初始值
            else:
                if parent['alpha'] < parent['parent_alpha']:
                    parent['n_child_'] = min(parent['n_child_'] + 1, 10)
                else:
                    parent['n_child_'] = max(parent['n_child_'] - 1, 1)
        # 5. 生成子代
        r = r0 * np.exp(-k * gen)
        children = []
        for parent in critical_parents:
            children.extend([
                {
                    'param': child,
                    'parent_param': parent['param'],
                    'parent_alpha': parent['alpha'],
                    'n_child_': parent['n_child_'],
                    'critical': None,
                    'alpha': None
                }
                for child in generate_children(parent['param'], r, parent['n_child_'])
            ])
        # 限制每代子代总数
        if len(children) > max_children_per_gen:
            idxs = np.random.choice(len(children), max_children_per_gen, replace=False)
            children = [children[i] for i in idxs]
        parents = children
        history.append(gen_info)
    return history

# 似乎被淘汰了，因为重复计算数量爆炸，后续没有使用
def evolve_search_repeat(initial_params, eval_func, r0=1.0, k=0.2, max_gen=10, n_child=5, n_jobs=-1, 
                         max_children_per_gen=1000):
    parents = [{'param': param, 'alpha': None, 'n_child_': n_child, 'critical': None, 'parent_alpha': None} 
               for param in initial_params]
    history = []
    for gen in range(max_gen):
        print(f'Generation {gen}, parent num: {len(parents)}')
        # 1. 并行评估所有父代
        results = [
            eval_func_repeat(parent['param'])
            for parent in parents
        ]
        # 2. 更新父代的alpha和critical
        for parent, (critical, alpha) in zip(parents, results):
            parent['critical'] = critical
            parent['alpha'] = alpha
        # 3. 只保留critical父代
        critical_parents = [p for p in parents if p['critical']]
        gen_info = [dict(p) for p in parents]  # 记录本代所有父代信息
        if not critical_parents:
            print('No critical parents left, evolution stops.')
            history.append(gen_info)
            break
        # 4. 评估n_child_调整（必须在生成子代前！）
        for parent in critical_parents:
            if gen == 0 or parent['parent_alpha'] is None:
                parent['n_child_'] = n_child  # 第0代或无parent_alpha，直接用初始值
            else:
                if parent['alpha'] < parent['parent_alpha']:
                    parent['n_child_'] = min(parent['n_child_'] + 1, 10)
                else:
                    parent['n_child_'] = max(parent['n_child_'] - 1, 1)
        # 5. 生成子代
        r = r0 * np.exp(-k * gen)
        children = []
        for parent in critical_parents:
            children.extend([
                {
                    'param': child,
                    'parent_param': parent['param'],
                    'parent_alpha': parent['alpha'],
                    'n_child_': parent['n_child_'],
                    'critical': None,
                    'alpha': None
                }
                for child in generate_children(parent['param'], r, parent['n_child_'])
            ])
        # 限制每代子代总数
        if len(children) > max_children_per_gen:
            idxs = np.random.choice(len(children), max_children_per_gen, replace=False)
            children = [children[i] for i in idxs]
        parents = children
        history.append(gen_info)
    return history

def plot_evolution_history(history, save_path, plot_hull=False, plot_ellipse=True, conf_level=0.99):
    plt.figure(figsize=(7, 3.5))
    generations = len(history)
    cmap = plt.get_cmap('viridis', generations)
    all_x, all_y, all_alpha, all_gen, all_critical = [], [], [], [], []
    for gen, gen_info in enumerate(history):
        for entry in gen_info:
            ie_r_e1, ie_r_i1 = entry['param']
            all_x.append(ie_r_e1)
            all_y.append(ie_r_i1)
            all_alpha.append(entry['alpha'])
            all_gen.append(gen)
            all_critical.append(entry['critical'])
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_alpha = np.array(all_alpha)
    all_gen = np.array(all_gen)
    all_critical = np.array(all_critical)

    mask_critical = all_critical == True

    # # 只画critical==True的点，颜色按代数
    # plt.scatter(all_x[mask_critical], all_y[mask_critical], 
    #             c=all_gen[mask_critical], cmap=cmap, s=60, edgecolors='k', label='Critical')
    # # 也可画出非critical点（灰色/透明）
    # plt.scatter(all_x[~mask_critical], all_y[~mask_critical], 
    #             c='lightgray', s=30, alpha=0.5, label='Non-critical')

    # plt.xlabel(r'$\zeta^{\rm I}$')
    # plt.ylabel(r'$\zeta^{\rm E}$')
    # plt.title('Evolutionary Search Trajectory')
    # cbar = plt.colorbar(ticks=range(generations), label='Generation')
    # cbar.set_label('Generation')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.close()

    # 计算凸包边界
    if plot_hull and np.sum(mask_critical) >= 3:
        critical_points = np.column_stack([all_x[mask_critical], all_y[mask_critical]])

        # 不剔除离群点
        hull = ConvexHull(critical_points)

        boundary = critical_points[hull.vertices]
        boundary_closed = np.vstack([boundary, boundary[0]])  # 闭合曲线

    # 计算椭圆边界
    if plot_ellipse and np.sum(mask_critical) >= 3:
        critical_points = np.column_stack([all_x[mask_critical], all_y[mask_critical]])
        mean = np.mean(critical_points, axis=0)
        cov = np.cov(critical_points, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        width, height = 2 * np.sqrt(vals * chi2.ppf(conf_level, df=2))

    # 子图1：颜色表示代数
    ax1 = plt.subplot(1, 2, 1)
    cmap_gen = plt.get_cmap('viridis', generations)
    ax1.scatter(
        all_x[~mask_critical], all_y[~mask_critical],
        c='lightgray', s=30, alpha=0.5, label='Non-critical', zorder=1
    )
    norm_gen = plt.Normalize(vmin=0, vmax=generations-1)
    sc1 = ax1.scatter(
        all_x[mask_critical], all_y[mask_critical],
        c=all_gen[mask_critical],
        cmap=cmap_gen,
        norm=norm_gen,
        s=60, edgecolors='k', label='Critical', zorder=2
    )
    # 画凸包边界
    if plot_hull and np.sum(mask_critical) >= 3:
        ax1.plot(boundary_closed[:,0], boundary_closed[:,1], 'r-', linewidth=2, 
                 label='Critical Region Boundary', zorder=3)
    # 画椭圆边界
    if plot_ellipse and np.sum(mask_critical) >= 3:
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta, edgecolor='blue', facecolor='none', lw=2, label='Ellipse Boundary', zorder=4)
        ax1.add_patch(ellipse)
    ax1.set_xlabel(r'$\zeta^{\rm E}$', fontsize=10)
    ax1.set_ylabel(r'$\zeta^{\rm I}$', fontsize=10)
    ax1.set_title('Generation (Critical States)', fontsize=11)
    ax1.tick_params(axis='both', labelsize=10)
    cbar1 = plt.colorbar(sc1, ax=ax1, ticks=range(generations))
    cbar1.set_label('Generation', fontsize=10)
    cbar1.ax.tick_params(labelsize=10)
    cbar1.set_ticks(range(generations))
    cbar1.set_ticklabels([str(i) for i in range(generations)])
    ax1.legend(fontsize=9)

    # 子图2：颜色表示alpha
    ax2 = plt.subplot(1, 2, 2)
    ax2.scatter(all_x[~mask_critical], all_y[~mask_critical], 
                c='lightgray', s=30, alpha=0.5, label='Non-critical')
    if np.any(mask_critical):
        norm_alpha = plt.Normalize(vmin=np.nanmin(all_alpha[mask_critical]), 
                                   vmax=np.nanmax(all_alpha[mask_critical]))
        cmap_alpha = plt.get_cmap('plasma')
        sc2 = ax2.scatter(all_x[mask_critical], all_y[mask_critical], 
                          c=all_alpha[mask_critical], cmap=cmap_alpha, norm=norm_alpha, 
                          s=60, edgecolors='k', label='Critical')
        cbar2 = plt.colorbar(sc2, ax=ax2)
        cbar2.set_label('Alpha', fontsize=10)
        cbar2.ax.tick_params(labelsize=10)
        # 画凸包边界
        if plot_hull and np.sum(mask_critical) >= 3:
            ax2.plot(boundary_closed[:,0], boundary_closed[:,1], 'r-', linewidth=2, 
                     label='Critical Region Boundary')
        # 画椭圆边界
        if plot_ellipse and np.sum(mask_critical) >= 3:
            ellipse2 = Ellipse(xy=mean, width=width, height=height, angle=theta, edgecolor='blue', facecolor='none', lw=2, label='Ellipse Boundary')
            ax2.add_patch(ellipse2)
    ax2.set_xlabel(r'$\zeta^{\rm E}$', fontsize=10)
    ax2.set_ylabel(r'$\zeta^{\rm I}$', fontsize=10)
    ax2.set_title('Alpha (Critical States)', fontsize=11)
    ax2.tick_params(axis='both', labelsize=10)
    ax2.legend(fontsize=9)

    plt.tight_layout(pad=1.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 椭圆参数
    ellipse_info = {
        'mean': mean,         # 均值 (中心)
        'cov': cov,           # 协方差矩阵
        'conf_level': conf_level
    }
    return ellipse_info