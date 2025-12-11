import brian2.numpy_ as np
from brian2.only import *
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull, KDTree
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

def remove_outliers_critical(points, k=3, distance_percentile=99.99):
    """
    仅对critical点剔除离群点（近邻数≤2判定为离群）
    参数：
        points: critical点集 (N, 2)
        k: 近邻数阈值（统计前k个近邻）
        distance_percentile: 距离阈值分位数（过滤过远的近邻）
    返回：
        filtered_points: 清洗后的critical点集
        outlier_mask: 离群点掩码（True=离群，False=保留）
    """
    if len(points) <= k:
        return points, np.zeros(len(points), dtype=bool)  # 点数过少，不剔除
    
    # 构建KD树计算近邻
    kd_tree = KDTree(points)
    distances, _ = kd_tree.query(points, k=k+1)  # 前k+1个（含自身）
    
    # 排除自身，统计有效近邻数
    neighbor_distances = distances[:, 1:]  # 去掉自身的距离
    dist_threshold = np.percentile(neighbor_distances, distance_percentile)
    neighbor_counts = np.sum(neighbor_distances <= dist_threshold, axis=1)
    
    # 离群点：近邻数≤2
    outlier_mask = neighbor_counts <= 2
    filtered_points = points[~outlier_mask]
    
    return filtered_points, outlier_mask

# def remove_outliers_critical(points, k=5, sigma=1.5, min_neighbors=3):
#     """
#     优化版离群点剔除：基于局部密度自适应判定
#     参数：
#         points: critical点集 (N, 2)
#         k: 计算局部密度的近邻数（建议5-10，越大越稳健）
#         sigma: 离群判定的标准差倍数（越小越严格，建议1.0-2.0）
#         min_neighbors: 最小近邻数阈值（原逻辑的2→建议3）
#     返回：
#         filtered_points: 清洗后的点集
#         outlier_mask: 离群点掩码（True=离群）
#     """
#     if len(points) <= k:
#         return points, np.zeros(len(points), dtype=bool)
    
#     # 1. 构建KD树，计算每个点的前k个近邻（不含自身）的距离
#     kd_tree = KDTree(points)
#     # query参数说明：k=k+1 → 取前k+1个（含自身），后续剔除自身
#     distances, _ = kd_tree.query(points, k=k+1)
#     neighbor_distances = distances[:, 1:]  # 剔除自身，保留前k个近邻的距离
    
#     # 2. 计算局部密度特征
#     # 每个点的k近邻平均距离（局部密度的反向指标：值越大，局部越稀疏）
#     avg_neighbor_dist = np.mean(neighbor_distances, axis=1)
#     # 全局平均距离 + 标准差（用于判定离群）
#     global_mean = np.mean(avg_neighbor_dist)
#     global_std = np.std(avg_neighbor_dist)
#     # 离群距离阈值：全局均值 + sigma倍标准差（自适应局部密度）
#     dist_threshold = global_mean + sigma * global_std
    
#     # 3. 双重判定离群点
#     # 条件1：局部平均距离超过阈值（局部极稀疏）
#     sparse_mask = avg_neighbor_dist > dist_threshold
#     # 条件2：有效近邻数（距离≤局部中位数）不足min_neighbors
#     local_median = np.median(neighbor_distances, axis=1)  # 每个点的近邻距离中位数（更稳健）
#     # local_median shape is (N,), neighbor_distances is (N, k)
#     # reshape local_median to (N, 1) so broadcasting compares each row to its median
#     neighbor_counts = np.sum(neighbor_distances <= local_median[:, None], axis=1)
#     few_neighbor_mask = neighbor_counts < min_neighbors
    
#     # 最终离群点：满足任一条件
#     outlier_mask = sparse_mask | few_neighbor_mask
#     filtered_points = points[~outlier_mask]
    
#     # 调试信息（可选）
#     print(f"局部平均距离统计 - 均值: {global_mean:.4f}, 标准差: {global_std:.4f}, 阈值: {dist_threshold:.4f}")
#     print(f"稀疏点数量: {np.sum(sparse_mask)}, 近邻不足点数量: {np.sum(few_neighbor_mask)}, 总离群点: {np.sum(outlier_mask)}")
    
#     return filtered_points, outlier_mask

def plot_evolution_history(history, save_path, remove_outlier=True, 
                           plot_hull=False, plot_ellipse=True, conf_level=0.99):
    plt.figure(figsize=(7, 3.5))
    generations = len(history)
    cmap = plt.get_cmap('viridis', generations)

    # 1. 提取所有点数据
    all_x, all_y, all_alpha, all_gen, all_critical = [], [], [], [], []
    for gen, gen_info in enumerate(history):
        for entry in gen_info:
            ie_r_e1, ie_r_i1 = entry['param']
            all_x.append(ie_r_e1)
            all_y.append(ie_r_i1)
            all_alpha.append(entry['alpha'])
            all_gen.append(gen)
            all_critical.append(entry['critical'])

    # 转为numpy数组
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_alpha = np.array(all_alpha)
    all_gen = np.array(all_gen)
    all_critical = np.array(all_critical, dtype=bool)

    # 2. 筛选critical点（核心分析对象）
    mask_critical = all_critical == True
    critical_points_original = np.column_stack([all_x[mask_critical], all_y[mask_critical]])

    # 3. 剔除critical点中的离群点
    if remove_outlier and len(critical_points_original) >= 3:
        critical_points, outlier_mask = remove_outliers_critical(critical_points_original)
        # 同步更新all_*系列数组的mask（用于可视化离群点）
        critical_idx = np.where(mask_critical)[0]  # critical点的全局索引
        outlier_global_idx = critical_idx[outlier_mask]  # 离群点的全局索引
        # 构建全局离群点掩码（仅critical中的离群点）
        global_outlier_mask = np.zeros_like(all_critical, dtype=bool)
        global_outlier_mask[outlier_global_idx] = True
    else:
        critical_points = critical_points_original
        global_outlier_mask = np.zeros_like(all_critical, dtype=bool)
    
    # 4. 初始化凸包/椭圆参数
    boundary_closed = None
    hull = None  # 保存凸包对象
    hull_vertices = None  # 凸包顶点坐标
    hull_area = None  # 凸包面积
    hull_vertex_indices = None  # 凸包顶点在filtered_critical_points中的索引
    mean = None
    cov = None
    theta = None
    width = None
    height = None

    # 5. 计算凸包（基于清洗后的critical点）
    if plot_hull and len(critical_points) >= 3:
        hull = ConvexHull(critical_points)
        hull_vertices = critical_points[hull.vertices]  # 凸包顶点坐标 (n, 2)
        hull_vertex_indices = hull.vertices  # 凸包顶点在critical_points中的索引
        hull_area = hull.area  # 凸包面积（二维）
        boundary = critical_points[hull.vertices]
        boundary_closed = np.vstack([boundary, boundary[0]])  # 闭合凸包

    # 6. 计算鲁棒椭圆（基于清洗后的critical点）
    if plot_ellipse and len(critical_points) >= 3:

        # critical_points = np.column_stack([all_x[mask_critical], all_y[mask_critical]])

        ## 直接通过协方差计算椭圆
        # mean = np.mean(critical_points, axis=0)
        # cov = np.cov(critical_points, rowvar=False)
        # vals, vecs = np.linalg.eigh(cov)
        # order = vals.argsort()[::-1]
        # vals, vecs = vals[order], vecs[:, order]
        # theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        # width, height = 2 * np.sqrt(vals * chi2.ppf(conf_level, df=2))

        ## 鲁棒椭圆
        # 步骤1：初始均值和协方差
        mean_x = np.mean(critical_points[:, 0])
        mean_y = np.mean(critical_points[:, 1])
        cov = np.cov(critical_points, rowvar=False)

        # 步骤2：计算马氏距离
        inv_cov = np.linalg.inv(cov)
        centered = critical_points - [mean_x, mean_y]
        mahalanobis_dist = np.sqrt(np.sum(centered @ inv_cov * centered, axis=1))

        # 步骤3：去除离群点
        threshold = np.sqrt(chi2.ppf(conf_level, 2))
        inlier_mask = mahalanobis_dist <= threshold
        inlier_points = critical_points[inlier_mask]

        # 如果去除离群点后点数不足，则使用所有点
        if len(inlier_points) < 3:
            inlier_points = critical_points

        # 步骤4：重新计算均值和协方差
        mean = np.mean(inlier_points, axis=0)
        cov = np.cov(inlier_points, rowvar=False)

        # 步骤5：计算椭圆参数
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]

        # 旋转角度（以度为单位）
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        # 半轴长度乘以缩放因子s（根据置信水平）
        s = np.sqrt(chi2.ppf(conf_level, 2))
        width, height = 2 * np.sqrt(vals) * s

    # 7. 子图1：颜色表示代数
    ax1 = plt.subplot(1, 2, 1)
    cmap_gen = plt.get_cmap('viridis', generations)
    
    # 绘制非critical点（浅灰色）
    ax1.scatter(
        all_x[~mask_critical], all_y[~mask_critical],
        c='lightgray', s=30, alpha=0.5, label='Non-critical', zorder=1
    )
    
    # 绘制critical中的离群点（红色叉号，突出显示）
    if remove_outlier and np.any(global_outlier_mask):
        ax1.scatter(
            all_x[global_outlier_mask], all_y[global_outlier_mask],
            c='red', s=50, alpha=0.8, marker='x', label='Critical Outliers', zorder=3
        )
    
    # 绘制清洗后的critical点（按代数着色）
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
                 label='Critical Region Boundary', zorder=4)
    
    # 画椭圆边界
    if plot_ellipse and np.sum(mask_critical) >= 3:
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta, 
                          edgecolor='blue', facecolor='none', lw=2, 
                          label='Ellipse Boundary', zorder=5)
        ax1.add_patch(ellipse)
    
    ax1.set_xlabel(r'$\zeta^{\rm E}$', fontsize=10)
    ax1.set_ylabel(r'$\zeta^{\rm I}$', fontsize=10)
    ax1.set_title('Generation (Critical States)', fontsize=11)
    ax1.tick_params(axis='both', labelsize=10)
    
    # 颜色条（代数）
    cbar1 = plt.colorbar(sc1, ax=ax1, ticks=range(generations))
    cbar1.set_label('Generation', fontsize=10)
    cbar1.ax.tick_params(labelsize=10)
    cbar1.set_ticks(range(generations))
    cbar1.set_ticklabels([str(i) for i in range(generations)])
    # ax1.legend(fontsize=9) # 老是遮挡，就注释了

    # 8. 子图2：颜色表示alpha
    ax2 = plt.subplot(1, 2, 2)
    
    # 绘制非critical点
    ax2.scatter(all_x[~mask_critical], all_y[~mask_critical], 
                c='lightgray', s=30, alpha=0.5, label='Non-critical', zorder=1)
    
    # 绘制critical中的离群点
    if remove_outlier and np.any(global_outlier_mask):
        ax2.scatter(
            all_x[global_outlier_mask], all_y[global_outlier_mask],
            c='red', s=50, alpha=0.8, marker='x', label='Critical Outliers', zorder=3
        )
    
    # 绘制清洗后的critical点（按alpha着色）
    if np.any(mask_critical):
        norm_alpha = plt.Normalize(vmin=np.nanmin(all_alpha[mask_critical]), 
                                   vmax=np.nanmax(all_alpha[mask_critical]))
        cmap_alpha = plt.get_cmap('plasma')
        sc2 = ax2.scatter(all_x[mask_critical], all_y[mask_critical], 
                          c=all_alpha[mask_critical], cmap=cmap_alpha, norm=norm_alpha, 
                          s=60, edgecolors='k', label='Critical', zorder=2)
        
        # 颜色条（alpha）
        cbar2 = plt.colorbar(sc2, ax=ax2)
        cbar2.set_label('Alpha', fontsize=10)
        cbar2.ax.tick_params(labelsize=10)

        # 画凸包边界
        if plot_hull and np.sum(mask_critical) >= 3:
            ax2.plot(boundary_closed[:,0], boundary_closed[:,1], 'r-', linewidth=2, 
                     label='Critical Region Boundary', zorder=4)
            
        # 画椭圆边界
        if plot_ellipse and np.sum(mask_critical) >= 3:
            ellipse2 = Ellipse(xy=mean, width=width, height=height, angle=theta, 
                               edgecolor='blue', facecolor='none', lw=2, 
                               label='Ellipse Boundary', zorder=5)
            ax2.add_patch(ellipse2)
    
    ax2.set_xlabel(r'$\zeta^{\rm E}$', fontsize=10)
    ax2.set_ylabel(r'$\zeta^{\rm I}$', fontsize=10)
    ax2.set_title('Alpha (Critical States)', fontsize=11)
    ax2.tick_params(axis='both', labelsize=10)
    # ax2.legend(fontsize=9) # 老是遮挡，就注释了

    # 9. 保存图片
    plt.tight_layout(pad=1.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 10. 返回参数（新增凸包信息）
    ellipse_info = {
        # 椭圆相关（原有）
        'mean': mean,                  # 椭圆中心 (2,)
        'cov': cov,                    # 椭圆协方差矩阵 (2,2)
        'conf_level': conf_level,      # 椭圆置信水平
        'theta': theta,                # 椭圆旋转角度（度）
        'width': width,                # 椭圆宽度（长轴）
        'height': height,              # 椭圆高度（短轴）
        # 数据过滤相关（原有）
        'filtered_critical_points': critical_points,  # 清洗后的critical点 (n,2)
        'outlier_mask': global_outlier_mask,           # 全局离群点掩码 (N,)
        # 凸包相关（新增）
        'hull_exists': plot_hull and len(critical_points)>=3,  # 是否计算了凸包
        'hull_vertices': hull_vertices,                  # 凸包顶点坐标 (m,2)
        'hull_vertex_indices': hull_vertex_indices,      # 凸包顶点在filtered_critical_points中的索引 (m,)
        'hull_area': hull_area,                          # 凸包面积（二维）
        'hull_boundary_closed': boundary_closed,         # 闭合的凸包边界坐标 (m+1,2)
        'hull_object': hull                              # 原始ConvexHull对象（可调用所有原生方法）
    }
    return ellipse_info