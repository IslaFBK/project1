import sys
import pickle
import itertools
import gc
import os
import brian2.numpy_ as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from scipy.spatial import ConvexHull, Delaunay, KDTree
from scipy.stats import chi2
from scipy.stats import linregress
from typing import Optional, Tuple, Union, List, Dict
from sklearn.linear_model import LinearRegression
from levy import fit_levy, levy
from analysis import mydata
from analysis import firing_rate_analysis as fra
from analysis import my_analysis as mya
from myscript.ie_search.batch_repeat import draw_statistical_MSD_pdx
import myscript.ie_search.critical_states_search as search

# plt.rcParams.update({
#     "text.usetex": True,  # 启用 LaTeX 渲染
#     "font.family": "serif",  # 使用衬线字体（类似 LaTeX 默认）
#     "font.serif": ["Times New Roman"]  # 指定字体
# })
def set_journal_style():
    plt.rcParams.update({
        # Font
        "text.usetex": True,  # 启用 LaTeX 渲染, 启用后ticks没法使用Arial
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        # 关键：LaTeX预编译指令，强制无衬线+加载Arial
        "text.latex.preamble": r"""
            \usepackage[utf8]{inputenc}
            \usepackage[T1]{fontenc}
            \usepackage{arevmath}  % 无衬线数学字体 (兼容Aria)
            \usepackage{sfmath}   % 强制数学符号用无衬线
            \renewcommand{\familydefault}{\sfdefault}  % 全局无衬线
            \usepackage{helvet}   % LaTeX的Arial兼容包 (helvet对应Helvetica/Arial)
            \renewcommand{\sfdefault}{phv}  % 指定helvet的字体族为phv (Arial/Helvetica)
        """,
        
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        # 强制ticks字体继承无衬线
        "xtick.labelbottom": True,
        "ytick.labelleft": True,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 9,
        
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
BASE_FIG_WIDTH = 3.35  # inch（单栏）
'''
图尺寸(inch): 
单栏3.35
双栏4.49-7.0
方图3.5*3.5
字号:
title 10-11
axis label 8-9
tick label 7-8
legend 7-8
'''

#%% OS operation
# test if data_dir exists, if not, create one.
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
elite_graph_dir = f'{graph_dir}/elite_graph/'
Path(elite_graph_dir).mkdir(parents=True, exist_ok=True)

def vary_ie_ratio(dx=0,dy=0):
    '''
    第一层临界域给相对坐标，生成参数
    
    :param dx: 临界域内x轴, 向右下
    :param dy: 临界域内y轴, 向左下
    '''
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

def data_analysis(data_path,transient=3000,stim_dura=1000,window=15,sti=False,sig=5,width=64):
    '''load data from disk'''
    data_load = mydata.mydata()
    data_load.load(data_path)
    transient=3000
    start_time = transient  #data.a1.param.stim1.stim_on[first_stim,0] - 300
    end_time = transient+stim_dura+window   # transient + stim_dura + window
    data_load.a1.ge.get_spike_rate(start_time=start_time,
                                   end_time=end_time,
                                   sample_interval=1,
                                   n_neuron = data_load.a1.param.Ne,
                                   window = window)
    spk_rate = data_load.a1.ge.spk_rate.spk_rate
    frames = data_load.a1.ge.spk_rate.spk_rate.shape[2]
    data_load.a1.ge.get_centre_mass()
    centre = data_load.a1.ge.centre_mass.centre
    centre_ind = data_load.a1.ge.centre_mass.centre_ind
    data_load.a1.ge.overlap_centreandspike()
    stim_on_off = data_load.a1.param.stim1.stim_on-start_time
    stim_on_off = stim_on_off[stim_on_off[:,0]>=0]
    stim = None
    if sti:
        stim = [[[[(width-1)/2,(width-1)/2]], 
                [stim_on_off], 
                [[sig]*stim_on_off.shape[0]]]]
    jump_interval = np.linspace(1, np.min([stim_dura,1000]), np.min([stim_dura,100]))
    data_load.a1.ge.get_MSD(start_time=start_time,
                            end_time=end_time,
                            sample_interval=1,
                            n_neuron = data_load.a1.param.Ne,
                            window = window,
                            dt = 0.1,
                            slide_interval=1,
                            jump_interval=jump_interval,
                            fit_stableDist='pylevy')
    msd = data_load.a1.ge.MSD.MSD
    jump_interval = data_load.a1.ge.MSD.jump_interval
    pdx = data_load.a1.ge.centre_mass.jump_size[:,1]
    return {
        'spk_rate': spk_rate,
        'frames': frames,
        'centre': centre,
        'centre_ind': centre_ind,
        'stim_on_off': stim_on_off,
        'stim': stim,
        'jump_interval': jump_interval,
        'msd': msd,
        'pdx': pdx
    }

# unwrap periodic trajectory
def unwrap_periodic_path(centre, width=64):
    '''
    对单个点做周期边界处理, 返回延拓环面最短路径
    
    :param centre: 轨迹序列, 大小[N,2]
    :param width: 平直环面宽度
    '''
    geodesic = centre.copy().astype(float)
    
    # 处理x坐标
    dx = np.diff(geodesic[:, 0])
    dx = np.mod(dx + width//2, width) - width//2  # 处理跳变
    geodesic[1:, 0] = geodesic[0, 0] + np.cumsum(dx)
    
    # 处理y坐标
    dy = np.diff(geodesic[:, 1])
    dy = np.mod(dy + width//2, width) - width//2
    geodesic[1:, 1] = geodesic[0, 1] + np.cumsum(dy)
    
    return geodesic

def split_path(centre, width=64.0, eps=1e-12):
    # centre: (N,2) 的轨迹点，已经在 [0,width)×[0,width) 的基本胞元内
    centre = np.asarray(centre, float)
    N = len(centre)

    segments = []   # 存放最终用于绘图的线段，每个元素是 [p_start, p_end]
    times = []      # 每个线段对应的归一化时间（用于渐变色）

    for i in range(N - 1):
        p0 = centre[i]       # 当前点（基本胞元内）
        p1 = centre[i + 1]   # 下一点（基本胞元内）

        # 1. 在环面上选取“最短延拓位移”
        delta = (p1 - p0 + width / 2) % width - width / 2

        # 延拓空间中的终点（不做 mod）
        p1_ext = p0 + delta

        # 该段在整条轨迹中的“时间标签”
        t_norm = i / (N - 1)

        # 2. 收集参数方程中的事件参数
        ts = [(0.0, None), (1.0, None)]   # 参数区间的起点和终点必然保留

        for d in (0, 1):  # d=0 → x 方向，d=1 → y 方向
            # 如果该方向位移为 0，则永远不会碰到对应边界
            if abs(delta[d]) < eps:
                continue

            # 解 p0[d] + t*delta[d] = 0
            t0 = (0.0 - p0[d]) / delta[d]

            # 解 p0[d] + t*delta[d] = width
            tL = (width - p0[d]) / delta[d]

            # 只保留真正发生在 (0,1) 内的穿越事件
            # eps 用于避免数值上“刚好在端点”的假事件
            if eps < t0 < 1 - eps:
                ts.append((t0, (d, 0.0)))
            if eps < tL < 1 - eps:
                ts.append((tL, (d, width)))

        # 对所有事件参数排序，并去重
        # 排序后，相邻 t 构成的区间内，线段不再穿越任何边界
        ts.sort(key=lambda x: x[0])

        # 3. 按参数区间逐段切割直线
        for (t_a, info_a), (t_b, info_b) in zip(ts[:-1], ts[1:]):
            # 延拓空间中的子段起点
            pa = p0 + t_a * delta
            # 延拓空间中的子段终点
            pb = p0 + t_b * delta

            # 4. 映射回基本胞元
            pa_draw = pa % width
            pb_draw = pb % width
            # 强制把边界点放边界上，防止取余导致的上、右边界缺失问题
            if info_a is not None:
                d, val = info_a
                # 跨界映射到对侧
                pa_draw[d] = width - val
            if info_b is not None:
                d, val = info_b
                pb_draw[d] = val
            # 特殊处理起点和终点
            if t_a == 0:
                pa_draw = p0
            if t_b == 1:
                pb_draw = p1
            
            # 5. 数值与几何安全性过滤
            if np.linalg.norm(pa_draw - pb_draw) < eps:
                continue

            # 6. 保存该可绘制线段
            segments.append([pa_draw, pb_draw])
            times.append(t_norm)

    return np.asarray(segments), np.asarray(times)

def plot_trajectory(
        data: np.ndarray,
        title: str = None,
        axis: bool = False,
        save_path: str = None,
        cmap: str = "plasma",
        linewidth: float = 2.0,
        show_colorbar: bool = False,
        scalebar_len: float = None,
        scalebar_label: str = None
):
    """
    绘制二维轨迹，线条随时间呈现渐变色，支持环面边界接续效果。
    参数:
        data: Nx2 数组 (x, y)
        title: 图标题
        save_path: 保存路径 (若为 None 则不保存)
        cmap: colormap 名称
        linewidth: 线宽
        show_colorbar: 是否显示颜色条（表示时间方向）
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("data must be an (N,2) array")

    fig, ax = plt.subplots(figsize=(2, 2))

    # 构造线段并按时间着色
    points = data.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)  # (N-1,2,2)
    t = np.linspace(0.0, 1.0, len(segments))
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    lc = LineCollection(segments, array=t, cmap=plt.get_cmap(cmap), 
                        norm=norm, linewidths=linewidth)
    ax.add_collection(lc)

    # 设置范围，使 x 和 y 轴比例一致并留白
    x_min, x_max = data[:, 0].min(), data[:, 0].max()
    y_min, y_max = data[:, 1].min(), data[:, 1].max()
    max_range = max(x_max - x_min, y_max - y_min) * 1.1
    ax.set_xlim((x_min + x_max - max_range) / 2, (x_min + x_max + max_range) / 2)
    ax.set_ylim((y_min + y_max - max_range) / 2, (y_min + y_max + max_range) / 2)

    # 右上边框取消
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)

    if axis:
        # ax.set_title(title, pad=20) # , fontsize=14
        ax.set_xlabel('Horizontal Position')
        ax.set_ylabel('Vertical Position')
    else:
        ax.tick_params(axis='both', which='both', length=0)  # 隐藏轴须（length=0）
        ax.set_xticks([])  # 清空x轴刻度数字
        ax.set_yticks([])  # 清空y轴刻度数字
    ax.set_aspect('equal', adjustable='box')

    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, label='Normalized Time', shrink=0.8)
        cbar.ax.tick_params(labelsize=8)

    # ---------- 标度尺（scale bar） ----------
    if scalebar_len is not None:
        # 自动在四个角落候选区域中选择点最少的地方绘制标度尺
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        xrange = x1 - x0
        yrange = y1 - y0
        pad_frac = 0.05
        tick_h = 0.02 * yrange
        sbw = float(scalebar_len)
        # 四个候选锚点（右下、左下、左上、右上）
        candidates = {
            'rb': (x1 - pad_frac * xrange - sbw, y0 + pad_frac * yrange),
            'lb': (x0 + pad_frac * xrange, y0 + pad_frac * yrange),
            'lt': (x0 + pad_frac * xrange, y1 - pad_frac * yrange - tick_h * 2),
            'rt': (x1 - pad_frac * xrange - sbw, y1 - pad_frac * yrange - tick_h * 2),
        }
        # 评价每个候选区域内轨迹点数，选最空的
        rect_w = sbw + pad_frac * xrange
        rect_h = tick_h * 3
        pts = np.asarray(data, float)
        counts = {}
        for k, (sx, sy) in candidates.items():
            xlo = sx - pad_frac * xrange / 2
            xhi = sx + rect_w + pad_frac * xrange / 2
            ylo = sy - pad_frac * yrange / 2
            yhi = sy + rect_h + pad_frac * yrange / 2
            # 限定在轴范围内
            xlo, xhi = max(xlo, x0), min(xhi, x1)
            ylo, yhi = max(ylo, y0), min(yhi, y1)
            if pts.size == 0:
                counts[k] = 0
            else:
                counts[k] = int(np.sum((pts[:, 0] >= xlo) & (pts[:, 0] <= xhi) &
                                        (pts[:, 1] >= ylo) & (pts[:, 1] <= yhi)))
        best = min(counts, key=counts.get)
        sb_x, sb_y = candidates[best]
        shiftx = 0.5
        # 绘制标度尺
        ax.plot([sb_x + shiftx, sb_x + sbw + shiftx], [sb_y, sb_y], color='k', lw=2)
        ax.plot([sb_x + shiftx, sb_x + shiftx], [sb_y - tick_h/2, sb_y + tick_h/2], color='k', lw=2)
        ax.plot([sb_x + sbw + shiftx, sb_x + sbw + shiftx], [sb_y - tick_h/2, sb_y + tick_h/2], 
                color='k', lw=2)
        label = scalebar_label if scalebar_label is not None else f'{scalebar_len}'
        ax.text(
            sb_x + sbw/2 + shiftx, sb_y + 1.5 * tick_h, 
            label,
            ha='center', va='bottom', 
            # fontsize=10, 
            color='k',
            bbox=dict(facecolor='white', alpha=0.0, edgecolor='none')
            )

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)

def plot_torus_trajectory(
        data: np.ndarray,
        width: float = 64,
        title: str = None,
        axis: bool = False,
        save_path: str = None,
        cmap: str = "plasma",
        linewidth: float = 2.0,
        show_colorbar: bool = False
):
    """
    绘制环面轨迹，线条随时间呈现渐变色，支持环面边界接续效果。
    参数:
        data: Nx2 数组 (x, y)
        title: 图标题
        save_path: 保存路径 (若为 None 则不保存)
        cmap: colormap 名称
        linewidth: 线宽
        show_colorbar: 是否显示颜色条（表示时间方向）
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("data must be an (N,2) array")
    
    segments, segment_times = split_path(centre=data, width=width)

    # 初始化画布
    fig, ax = plt.subplots(figsize=(2, 2))

    # 环面边界正方形
    # ax.add_patch(plt.Rectangle((0, 0), width, width, 
    #                           fill=False, color='black', linewidth=1.5, label='Torus Boundary'))
    
    # 构造线段并按时间着色
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    lc = LineCollection(segments, array=segment_times, cmap=plt.get_cmap(cmap), 
                        norm=norm, linewidths=linewidth)
    ax.add_collection(lc)

    # 设置范围，使 x 和 y 轴比例一致并留白
    x_min, x_max = data[:, 0].min(), data[:, 0].max()
    y_min, y_max = data[:, 1].min(), data[:, 1].max()
    max_range = max(x_max - x_min, y_max - y_min) * 1.1
    # ax.set_xlim((x_min + x_max - max_range) / 2, (x_min + x_max + max_range) / 2)
    # ax.set_ylim((y_min + y_max - max_range) / 2, (y_min + y_max + max_range) / 2)
    # ax.set_xlim((-0.1*width) / 2, (2.1*width) / 2)
    # ax.set_ylim((-0.1*width) / 2, (2.1*width) / 2)
    ax.set_xlim(0, width)
    ax.set_ylim(0, width)

    if axis:
        # ax.set_title(title, pad=20) # , fontsize=14
        ax.set_xlabel('Horizontal Position')
        ax.set_ylabel('Vertical Position')
    else:
        ax.tick_params(axis='both', which='both', length=0)  # 隐藏轴须（length=0）
        ax.set_xticks([])  # 清空x轴刻度数字
        ax.set_yticks([])  # 清空y轴刻度数字
    ax.set_aspect('equal', adjustable='box')

    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, label='Normalized Time', shrink=0.8)
        cbar.ax.tick_params(labelsize=8)

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)

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
    return best_end, best_r2

def get_flexible_bins(data, bin_width=2):
    # 计算 min_val 和 max_val，确保是奇数且覆盖数据范围
    min_val = np.floor((data.min() - 1) / bin_width) * bin_width + 1
    max_val = np.ceil((data.max() + 1) / bin_width) * bin_width - 1
    
    # 强制包含 [-1,1)
    min_val = min(min_val, -1)
    max_val = max(max_val, 1)
    
    return np.arange(min_val, max_val + bin_width, bin_width)

def draw_statistical_MSD_pdx(jump_interval, 
                             msd_mean, 
                             msd_std, 
                             all_pdx, 
                             save_path_MSD, 
                             save_path_pdx,
                             save_path_combined):
    # MSD
    log_jump_interval = np.log10(jump_interval)
    log_msd = np.log10(msd_mean)
    end, r2 = find_best_linear_region(log_jump_interval, log_msd, min_points=5)
    x_fit = log_jump_interval[:end]
    y_fit = log_msd[:end]
    model = LinearRegression().fit(x_fit.reshape(-1,1), y_fit)
    y_pred = model.predict(x_fit.reshape(-1,1))
    slope = model.coef_[0]
    slope_str = f'{slope:.2f}'
    r2_str = f'{r2:.2f}'
    plt.figure(figsize=(6, 6))
    plt.plot(jump_interval, msd_mean, color="#000000")
    n = int(len(x_fit) * 0.7)
    plt.plot(10**x_fit[:n], 2*10**y_pred[:n], 'r--', label='Linear Fit')
    plt.fill_between(jump_interval, msd_mean-msd_std, msd_mean+msd_std, color='gray', alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\tau$ (ms)', fontsize=16)
    plt.ylabel('MSD (gridpoint$^2$)', fontsize=16)
    plt.text(
        0.1, 0.95,
        rf'$\tau^{{{slope_str}}}$',
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    plt.text(
        0.1, 0.90,
        rf'$R^2:{{{r2_str}}}$',
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    plt.savefig(save_path_MSD, dpi=600, bbox_inches='tight')
    plt.close()

    # pdx
    plt.figure(figsize=(6, 6))
    bins = get_flexible_bins(all_pdx, bin_width=2)
    plt.hist(
        all_pdx,
        bins=bins,
        density=True,
        alpha=0.5,
        label='histogram',
        rwidth=0.8,
        color="#000000"
    )
    params, nll = fit_levy(all_pdx)
    alpha, beta, mu, sigma = params.get()
    x = np.linspace(all_pdx.min(), all_pdx.max(), 200)
    pdx_fit = levy(x, alpha, beta, mu, sigma)
    plt.plot(x, pdx_fit, 'r-', label='Levy fit')
    plt.xlabel(r'$\Delta$ x (gridpoint)', fontsize=16)
    plt.ylabel('Probability density', fontsize=16)
    plt.text(
        0.75, 0.95,
        rf'$\alpha: {alpha:.2f}$',
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    plt.text(
        0.75, 0.90,
        rf'$\beta:{beta:.2f}$',
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    plt.text(
        0.75, 0.85,
        rf'$\mu: {mu:.2f}$',
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    plt.text(
        0.75, 0.80,
        rf'$\sigma: {sigma:.2f}$',
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    plt.xlim(-45, 45)
    plt.legend()
    plt.savefig(save_path_pdx, dpi=600, bbox_inches='tight')
    plt.close()

    # combined graph
    fig, ax = plt.subplots(1,2, figsize=(12, 6))
    # MSD
    img_msd = mpimg.imread(save_path_MSD)
    ax[0].imshow(img_msd)
    ax[0].axis('off')
    # pdx
    img_pdx = mpimg.imread(save_path_pdx)
    ax[1].imshow(img_pdx)
    ax[1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path_combined, dpi=600, bbox_inches='tight')
    plt.close()

def stack_msd_pdx(results):
    '''
    并行compute的results的msd和pdx堆叠
    
    :param results: 多个compute函数并行的返回值, 含有msd和pdx
    '''
    # 假设results里每个元素有msd，jump_interval，pdx
    msds = np.stack([r['msd'] for r in results])
    jump_interval = results[0]['jump_interval']
    msd_mean = np.mean(msds, axis=0)
    msd_std = np.std(msds, axis=0)
    # pdx合并
    all_pdx = np.concatenate([r['pdx'] for r in results])
    return {
        'msds': msds,
        'jump_interval': jump_interval,
        'msd_mean': msd_mean,
        'msd_std': msd_std,
        'all_pdx': all_pdx
    }

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

def evolution_plot(delta_gk=1,remove_outlier=True,
                   plot_hull=True,plot_ellipse=True,conf_level=0.99):
    # load
    with open(f'{state_dir}/evolution{delta_gk}.file', 'rb') as file:
        history = pickle.load(file)

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

    generations = len(history)
    # 7. 子图1：颜色表示代数
    fig1, ax1 = plt.subplots(figsize=(2, 2))
    cmap_gen = plt.get_cmap('viridis', generations)
    
    # 绘制非critical点（浅灰色）
    ax1.scatter(
        all_x[~mask_critical], all_y[~mask_critical],
        c='lightgray', s=20, alpha=0.5, label='Non-critical', zorder=1
    )
    
    # 绘制critical中的离群点（红色叉号，突出显示）
    if remove_outlier and np.any(global_outlier_mask):
        ax1.scatter(
            all_x[global_outlier_mask], all_y[global_outlier_mask],
            c='red', s=30, alpha=0.8, marker='x', label='Critical Outliers', zorder=3
        )
    
    # 绘制清洗后的critical点（按代数着色）
    norm_gen = plt.Normalize(vmin=0, vmax=generations-1)
    sc1 = ax1.scatter(
        all_x[mask_critical], all_y[mask_critical],
        c=all_gen[mask_critical],
        cmap=cmap_gen,
        norm=norm_gen,
        s=20, edgecolors='k', label='Critical', zorder=2
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
    
    ax1.set_xlabel(r'$\zeta^{\rm E}$') # , fontsize=10
    ax1.set_ylabel(r'$\zeta^{\rm I}$') # , fontsize=10
    # ax1.set_title('Generation (Critical States)', fontsize=11)
    ax1.tick_params(axis='both') # , labelsize=10
    
    # 颜色条（代数）
    cbar1 = plt.colorbar(sc1, ax=ax1, ticks=range(generations))
    cbar1.set_label('Generation') # , fontsize=10
    cbar1.ax.tick_params(labelsize=8)
    cbar1.set_ticks(range(generations))
    cbar1.set_ticklabels([str(i) for i in range(generations)])

    # draw
    save_path = f'{elite_graph_dir}/evaluation_gen{delta_gk}.svg'
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

    # 8. 子图2：颜色表示alpha
    fig2, ax2 = plt.subplots(figsize=(2, 2))
    
    # 绘制非critical点
    ax2.scatter(all_x[~mask_critical], all_y[~mask_critical], 
                c='lightgray', s=20, alpha=0.5, label='Non-critical', zorder=1)
    
    # 绘制critical中的离群点
    if remove_outlier and np.any(global_outlier_mask):
        ax2.scatter(
            all_x[global_outlier_mask], all_y[global_outlier_mask],
            c='red', s=30, alpha=0.8, marker='x', label='Critical Outliers', zorder=3
        )
    
    # 绘制清洗后的critical点（按alpha着色）
    if np.any(mask_critical):
        norm_alpha = plt.Normalize(vmin=np.nanmin(all_alpha[mask_critical]), 
                                   vmax=np.nanmax(all_alpha[mask_critical]))
        cmap_alpha = plt.get_cmap('plasma')
        sc2 = ax2.scatter(all_x[mask_critical], all_y[mask_critical], 
                          c=all_alpha[mask_critical], cmap=cmap_alpha, norm=norm_alpha, 
                          s=20, edgecolors='k', label='Critical', zorder=2)
        
        # 颜色条（alpha）
        cbar2 = plt.colorbar(sc2, ax=ax2)
        cbar2.set_label(r'$\alpha$') # , fontsize=10
        cbar2.ax.tick_params(labelsize=8)

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
    
    ax2.set_xlabel(r'$\zeta^{\rm E}$') # , fontsize=10
    ax2.set_ylabel(r'$\zeta^{\rm I}$') # , fontsize=10
    # ax2.set_title('Alpha (Critical States)') # , fontsize=11
    ax2.tick_params(axis='both') # , labelsize=10

    # draw
    save_path = f'{elite_graph_dir}/evaluation_alpha{delta_gk}.svg'
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

def rf_alpha_distribution_plot(maxrate=1000, delta_gk=1, sample_type='Ellipse'):
    '''
    :maxrate: 1000 or 2000
    :delta_gk: 1 or 2
    :sample_type: Ellipse or Hull
    '''
    # 读取椭圆参数
    range_path = f'{state_dir}/critical_ellipse{delta_gk}.file'
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

    # 尝试读取已有历史
    rf_history_path = f'{state_dir}/rf_landscape_{maxrate}_{delta_gk}.file'
    r_rf_history = []
    computed_params = set()
    if os.path.exists(rf_history_path):
        with open(rf_history_path, 'rb') as file:
            r_rf_history = pickle.load(file)

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
        
    # r_rf
    fig1, ax1 = plt.subplots(figsize=(2, 2))
    # 1. 打包x、y、z_rf为元组列表
    rf_data_pack = list(zip(x, y, z_rf))
    # 2. 按z_rf（第3个元素，索引=2）进行升序排序
    rf_data_pack_sorted = sorted(rf_data_pack, key=lambda item: item[2])
    # 3. 解包排序后的数据，还原为单独的列表
    x_rf_sorted, y_rf_sorted, z_rf_sorted = zip(*rf_data_pack_sorted)
    # scattering
    sc1 = ax1.scatter(x_rf_sorted, y_rf_sorted, c=z_rf_sorted, cmap='viridis', s=20)
    if sample_type == 'Ellipse':
        ax1.add_patch(ellipse1)
    elif sample_type == 'Hull':
        ax1.plot(hull_boundary_closed[:, 0], hull_boundary_closed[:, 1], 
                    'r-', linewidth=2, label='Convex Hull')
    ax1.set_xlabel(r'$\zeta^{\rm E}$') # , fontsize=10
    ax1.set_ylabel(r'$\zeta^{\rm I}$') # , fontsize=10
    # ax1.set_title('Receptive Field Radius') # , fontsize=11
    ax1.tick_params(axis='both') # , labelsize=10
    cbar1 = plt.colorbar(sc1, ax=ax1)
    # 使用MaxNLocator，指定整数刻度（推荐，自动适配数据范围，保证宽度统一）
    cbar1.ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # 水平色条用xaxis
    cbar1.ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # 垂直色条用yaxis（你的场景是垂直，核心生效）
    cbar1.set_label('Receptive Field') # , fontsize=10
    cbar1.ax.tick_params(labelsize=8)
    plt.savefig(
        f'{elite_graph_dir}/rf_landscape_rf_{maxrate}_{delta_gk}.svg',
        dpi=600,bbox_inches='tight'
        )
    plt.close(fig1)

    # alpha
    fig2, ax2 = plt.subplots(figsize=(2, 2))
    # 1. 打包x、y、z_alpha为元组列表
    alpha_data_pack = list(zip(x, y, z_alpha))
    # 2. 按z_alpha（第3个元素，索引=2）进行升序排序
    alpha_data_pack_sorted = sorted(alpha_data_pack, key=lambda item: item[2])
    # 3. 解包排序后的数据，还原为单独的列表
    x_alpha_sorted, y_alpha_sorted, z_alpha_sorted = zip(*alpha_data_pack_sorted)
    # scattering
    sc2 = ax2.scatter(x_alpha_sorted, y_alpha_sorted, c=z_alpha_sorted, cmap='plasma', s=20)
    if sample_type == 'Ellipse':
        ax2.add_patch(ellipse2)
    elif sample_type == 'Hull':
        ax2.plot(hull_boundary_closed[:, 0], hull_boundary_closed[:, 1], 
                    'r-', linewidth=2, label='Convex Hull')
    ax2.set_xlabel(r'$\zeta^{\rm E}$') # , fontsize=10
    ax2.set_ylabel(r'$\zeta^{\rm I}$') # , fontsize=10
    # ax2.set_title(r'$\alpha$') # , fontsize=11
    ax2.tick_params(axis='both') # , labelsize=10
    cbar2 = plt.colorbar(sc2, ax=ax2)
    # 使用MaxNLocator，指定整数刻度（推荐，自动适配数据范围，保证宽度统一）
    cbar2.ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # 水平色条用xaxis
    cbar2.ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # 垂直色条用yaxis（你的场景是垂直，核心生效）
    cbar2.set_label(r'$\alpha$') # , fontsize=10
    cbar2.ax.tick_params(labelsize=8)
    plt.savefig(
        f'{elite_graph_dir}/rf_landscape_alpha_{maxrate}_{delta_gk}.svg',
        dpi=600,bbox_inches='tight'
        )
    plt.close(fig2)

#%% Execution part ########################################################
def draw_trajectory():
    # 载入数据
    # 第一层参数:
    param_area1 = vary_ie_ratio(dx=0,dy=1)
    # 第二层参数:
    param_area2 = (1.84138, 1.57448)
    # 双层参数组合:
    param_area12 = param_area1 + param_area2
    ## 单层读数据,输出轨迹
    # 哪一层
    delta_gk=2
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
    window=5
    width=64
    if sti:
        input=f'on{maxrate}_{sti_type}_{sig}'
    else:
        input='off'

    data_path=f"{data_dir}/1data_{common_path}_{input}_{delta_gk}_win{window}.file"
    video_path=None

    results = data_analysis(data_path=data_path,
                            transient=3000,
                            stim_dura=stim_dura,
                            window=window,
                            sti=sti,
                            sig=sig,
                            width=width)
    centre=results['centre']


    # 解缠绕轨迹
    unwrapped_trajectory = unwrap_periodic_path(centre=centre, width=width)

    save_path1=f'{elite_graph_dir}/trajectory_ctnu_win{window}_{delta_gk}.svg'
    save_path2=f'{elite_graph_dir}/trajectory_toru_win{window}_{delta_gk}.svg'
    # 绘制普通轨迹
    plot_trajectory(
        data=unwrapped_trajectory,
        save_path=save_path1,
        axis=True,
        cmap='plasma',
        linewidth=2.0,
        # scalebar_len=100,
        # scalebar_label='100 unit',
        show_colorbar=False
    )

    # 绘制轨迹（带环面接续效果）
    plot_torus_trajectory(
        data=centre,
        width=width,
        axis=True,
        save_path=save_path2,
        cmap='plasma',
        linewidth=2.0,
        show_colorbar=False
    )

def draw_spk_rate_distribution_spon():
    # 载入数据
    # 第一层参数:
    param_area1 = vary_ie_ratio(dx=0,dy=1)
    # 第二层参数:
    param_area2 = (1.84138, 1.57448)
    # 双层参数组合:
    param_area12 = param_area1 + param_area2
    
    param = param_area1
    ie_r_e1, ie_r_i1 = param
    # common title & path
    common_path = f're{ie_r_e1:.4f}_ri{ie_r_i1:.4f}'
    n_repeat = 128
    delta_gk = 1
    # create data root path:
    data_root = state_dir
    # create graph root path:
    root_path = elite_graph_dir
    
    data_path = f'{data_root}/1MSDPDX_{common_path}_{n_repeat}_{delta_gk}.file'

    with open(data_path,'rb') as file:
        msd_pdx_info = pickle.load(file)
    results = msd_pdx_info['results']
    spk_rates = results['spk_rate']
    print(f'result: {spk_rates}')
    spk_rate = np.stack([r['spk_rate'] for r in results], axis=0)
    tmean_fr = np.mean(spk_rate, axis=2)
    # ===================== 核心绘图部分（新增）=====================
    # 1. 创建画布（可选，便于调整尺寸）
    plt.figure(figsize=(8, 7))  # 宽8英寸，高7英寸，适配64×64数组
    
    # 2. 绘制二维热力图（核心：imshow() 展示64×64数组）
    # cmap指定颜色映射方案，viridis是matplotlib默认优质配色，也可改为jet、gray等
    im = plt.imshow(tmean_fr, cmap='viridis')
    
    # 3. 添加颜色条（显示数值与颜色的对应关系，必备）
    plt.colorbar(im, label='Spike Rate (Hz)')  # label标注颜色条含义
    
    # 5. 可选：设置坐标轴刻度（对应64×64网格的索引，0到63）
    plt.xticks(np.arange(0, 65, step=8))  # 每8个点标注一次刻度，避免拥挤
    plt.yticks(np.arange(0, 65, step=8))
    
    # 6. 可选：保存图片（推荐保存为png，清晰无失真；如需矢量图可保存为pdf）
    save_path = f'{root_path}/spk_rate_distribution_{delta_gk}.pdf'
    plt.savefig(save_path, dpi=600, bbox_inches='tight')  # dpi=300高清分辨率

#%% draw spk rate distribution in spontaneous condition
# draw_spk_rate_distribution_spon()
#%% draw trajectory
# draw_trajectory()
#%% draw rf and alpha distribution
# 第一层
# maxrate = 1000
# delta_gk = 1
# 第二层
# maxrate = 2000
# delta_gk = 2
# rf_alpha_distribution_plot(maxrate=2000, delta_gk=2, sample_type='Hull')
#%% evolution plot
evolution_plot(delta_gk=2,remove_outlier=False)