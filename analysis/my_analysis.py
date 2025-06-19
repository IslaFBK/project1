# -*- coding: utf-8 -*-
"""
Created on Sun June  1 00:45:55 2025

@author: jianing liu
"""
import numpy as np
from scipy.sparse import csc_matrix
from scipy.stats import sem
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import connection as cn
from scipy.stats import linregress
from matplotlib.colors import Normalize
from typing import Optional, Tuple, Union, List, Dict

plt.rcParams.update({
    "text.usetex": True,  # 启用 LaTeX 渲染
    "font.family": "serif",  # 使用衬线字体（类似 LaTeX 默认）
    "font.serif": ["Times New Roman"]  # 指定字体
})

def unwrap_periodic_path(centre, width=64):
    
    continuous_centre = centre.copy().astype(float)
    
    # 处理x坐标
    dx = np.diff(continuous_centre[:, 0])
    dx = np.mod(dx + width//2, width) - width//2  # 处理跳变
    continuous_centre[1:, 0] = continuous_centre[0, 0] + np.cumsum(dx)
    
    # 处理y坐标
    dy = np.diff(continuous_centre[:, 1])
    dy = np.mod(dy + width//2, width) - width//2
    continuous_centre[1:, 1] = continuous_centre[0, 1] + np.cumsum(dy)
    
    return continuous_centre

def plot_trajectory(
        data: np.ndarray,
        title: str = None,
        save_path: str = None
):
    plt.figure(figsize=(6, 6))
    plt.plot(data[:,0],data[:,1])
    # get range, take maximum as general boundary
    x_min, x_max = data[:, 0].min(), data[:, 0].max()
    y_min, y_max = data[:, 1].min(), data[:, 1].max()
    max_range = max(x_max - x_min, y_max - y_min) * 1.1  # take 10% edge
    # set the same range
    plt.xlim((x_min + x_max - max_range) / 2, (x_min + x_max + max_range) / 2)
    plt.ylim((y_min + y_max - max_range) / 2, (y_min + y_max + max_range) / 2)
    # plot
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Horizontal Position')
    plt.ylabel('Vertical Position')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return None

def check_jump_power_law(
    data: np.ndarray,
    tail_fraction: float = 0.2,
    plot: bool = True,
    save_path: str = None,
    title: str = None,
    num_bins: int = 50
) -> Tuple[float, float, np.ndarray, int]:
    # 数据预处理
    data = data[data > 0]  # 移除0值（对数坐标需要）
    sorted_data = np.sort(data)

    # # 计算概率密度直方图(对数均匀采样)
    # bins = np.logspace(np.log10(min(data)), np.log10(max(data)), num=num_bins)
    # hist, bin_edges = np.histogram(sorted_data, bins=bins, density=True)
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 计算概率密度直方图(均匀采样)
    hist, bin_edges = np.histogram(sorted_data, bins=num_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    if len(sorted_data) == 0:
        # 幂律拟合（使用尾部数据）
        tail_start = int((1 - tail_fraction) * len(bin_centers))
        x_fit = np.log(bin_centers[tail_start:])
        y_fit = np.log(hist[tail_start:])
        tail_points = len(x_fit)
    else:
        # 幂律拟合（使用尾部数据）(排除0概率)
        tail_start = int((1 - tail_fraction) * len(bin_centers))
        valid_mask = hist[tail_start:] > 0
        x_fit = np.log(bin_centers[tail_start:][valid_mask])
        y_fit = np.log(hist[tail_start:][valid_mask])
        tail_points = np.count_nonzero(valid_mask)
    
    # 线性回归
    slope, intercept, r_value, _, _ = linregress(x_fit, y_fit)
    alpha = -slope
    r_squared = r_value**2
    
    # 生成拟合曲线
    y_pred = np.exp(intercept) * bin_centers[tail_start:]**slope
    
    if plot:
        plt.figure(figsize=(6, 6))
        
        # 绘制原始数据（概率密度）
        plt.loglog(bin_centers, hist, 'o',
                 markersize=5,
                 color="#000000",
                 label=f'Empirical Density (n={len(data)})')
        
        # 绘制拟合线
        plt.loglog(bin_centers[tail_start:], y_pred, '--',
                 color='#d7191c',
                 linewidth=2,
                 label=('Power Law Fit'
                        '\n'
                        rf'($\alpha$={alpha:.2f}, $R^2$={r_squared:.2f})'))
        
        # 标记尾部起始点
        plt.axvline(x=bin_centers[tail_start],
                  color="#000000",
                  linestyle=':',
                  label=f'Tail Start (top {tail_fraction*100:.0f}\%)')
        
        plt.xlabel('Jump Distance (log)', fontsize=12)
        plt.ylabel('Probability Density (log)', fontsize=12)
        plt.title(title, fontsize=14, pad=20)
        plt.legend(fontsize=10, loc='upper left')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.close()
    
    # 返回拟合结果（对齐原始数据点）
    full_pred = np.concatenate([np.nan*np.ones(tail_start), y_pred])
    return alpha, r_squared, np.column_stack((bin_centers, hist, full_pred)), tail_points

def check_coactive_power_law(
    spk_rate_obj,
    tail_fraction: float = 0.2,
    plot: bool = True,
    save_path: Optional[str] = None,
    title: str = None,
    min_active: int = 1
) -> Tuple[float, float, np.ndarray, int]:
    """
    分析同时放电神经元数量的幂律分布
    
    参数:
        spk_rate_obj: data_load.a1.ge.spk_rate 对象
        tail_fraction: 用于拟合的尾部数据比例(0-1)
        plot: 是否绘制图表
        save_path: 图片保存路径
        N_e_ext: 外部E神经元编号(用于标题)
        N_i_ext: 外部I神经元编号(用于标题)
        min_active: 最小放电神经元数量阈值
        
    返回:
        alpha: 幂律指数
        r_squared: 拟合优度R²
        fit_data: 拟合数据数组[x值, 概率密度, 拟合值]
    """
    # 获取放电率数据并计算同时放电数量
    spk_rate = spk_rate_obj.spk_rate
    coactive_counts = np.sum(spk_rate > 0, axis=0)
    coactive_counts = coactive_counts[coactive_counts >= min_active]  # 应用阈值
    
    # 计算概率密度直方图
    hist, bin_edges = np.histogram(coactive_counts, bins='auto', density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    if len(coactive_counts) == 0:
        # 幂律拟合（使用尾部数据）
        tail_start = int((1 - tail_fraction) * len(bin_centers))
        x_fit = np.log(bin_centers[tail_start:])
        y_fit = np.log(hist[tail_start:])
        tail_points = len(x_fit)
    else:
        # 幂律拟合（使用尾部数据）（排除0概率）
        tail_start = int((1 - tail_fraction) * len(bin_centers))
        valid_mask = hist[tail_start:] > 0
        x_fit = np.log(bin_centers[tail_start:][valid_mask])
        y_fit = np.log(hist[tail_start:][valid_mask])
        tail_points = np.count_nonzero(valid_mask)
    
    # 线性回归
    slope, intercept, r_value, _, _ = linregress(x_fit, y_fit)
    alpha = -slope
    r_squared = r_value**2
    
    # 生成拟合曲线
    y_pred = np.exp(intercept) * bin_centers[tail_start:]**slope
    
    if plot:
        # plt.figure(figsize=(12, 5)) # for 2 subfig
        plt.figure(figsize=(6, 6))
        
        plt.loglog(bin_centers, hist, 'o', 
                 markersize=5,
                 color="#000000",
                 label=f'Empirical Density (n={len(coactive_counts)})')
        
        plt.loglog(bin_centers[tail_start:], y_pred, '--',
                    color='#d7191c',
                    linewidth=2,
                    label=('Power Law Fit'
                        '\n'
                        rf'$\alpha$={alpha:.2f}, $R^2$={r_squared:.2f}'))
        plt.axvline(x=bin_centers[tail_start], 
                    color="#000000",
                    linestyle=':',
                    label=f'Tail Start (top {tail_fraction*100:.0f}\%)')
        
        plt.xlabel('Number of Coactive Neurons (log)')
        plt.ylabel('Probability Density (log)')
        plt.title(title, fontsize=14, pad=20)
        plt.legend(fontsize=10, loc='upper left')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.close()
    
    # 返回拟合结果
    full_pred = np.concatenate([np.nan*np.ones(tail_start), y_pred])
    return alpha, r_squared, np.column_stack((bin_centers, hist, full_pred)), tail_points

class CriticalityAnalyzer:
    def __init__(self):
        self.params: List[Dict[str, float]] = [] # store all parameter combinations
        self.states: List[Dict[str, float]] = [] # store all states

    def states_collector(self, params: Dict[str, float], states: Dict[str, float]) -> "CriticalityAnalyzer":
        self.params.append(params.copy())
        self.states.append(states.copy())
        return self
    
    def _filter_data(
            self,
            fixed_params: Dict[str, float],
            tolerance: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
        """
        根据固定参数筛选数据
        Args:
            fixed_params: 需要固定的参数及其目标值(如 {'num_ie': 400, 'num_ii': 400})
            tolerance: 浮点数比较容差
        Returns:
            筛选后的x, y值数组和对应的状态数据
        """
        valid_indices = [
            i for i, p in enumerate(self.params)
            if all(np.abs(p[k] - v) < tolerance for k, v in fixed_params.items())
        ]
        return valid_indices
    
    def plot_phase_diagrams(
            self,
            param_x: str,
            param_y: str,
            state_vars: List[str],
            fixed_params: Optional[Dict[str, float]] = None,
            subplot_kwargs: Optional[Dict[str, Union[str, Normalize]]] = None,
            save_path: Optional[str] = None,
            figsize: tuple = (6,6)
    ) -> None:
        """
        支持固定其他参数的相图绘制
        Args:
            fixed_params: 需要固定的参数(如 {'num_ie': 400, 'num_ii': 400})
            subplot_kwargs: 每个state_var的绘图参数, 格式为:
                {
                    'alpha_jump': {'cmap': 'viridis', 'norm': Normalize(1, 3), 'title': 'Alpha Jump'},
                    ...
                }
        """
        # 参数处理
        fixed_params = fixed_params or {}
        subplot_kwargs = subplot_kwargs or {}
        
        # 筛选数据
        valid_indices = self._filter_data(fixed_params)
        if not valid_indices:
            raise ValueError(f"No data matches fixed params: {fixed_params}")
        
        # 转换为NumPy数组
        x_vals = np.array([self.params[i][param_x] for i in valid_indices])
        y_vals = np.array([self.params[i][param_y] for i in valid_indices])
        
        # 计算子图布局
        n = len(state_vars)
        ncols = min(2, n)
        nrows = int(np.ceil(n / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)

        # 固定参数标题
        fixed_title = ", ".join([f"{k}={v}" for k, v in fixed_params.items()])
        fig.suptitle(
            f"Phase Diagrams: {param_x} vs {param_y} | Fixed: {fixed_title}",
            fontsize=14,
            y=1.02
        )
        
        # 绘制每个状态变量
        axs = axs.ravel() if isinstance(axs, np.ndarray) else [axs]
        for i, var in enumerate(state_vars):
            c_vals = np.array([self.states[idx][var] for idx in valid_indices])
            kwargs = subplot_kwargs.get(var, {})
            
            # 默认参数
            cmap = kwargs.get('cmap', 'viridis')
            norm = kwargs.get('norm', Normalize(np.nanmin(c_vals), np.nanmax(c_vals)))
            title = kwargs.get('title', var)
            
            # 绘制散点图
            sc = axs[i].scatter(x_vals, y_vals, c=c_vals, cmap=cmap, norm=norm, s=80)
            axs[i].set_title(title, fontsize=12)
            plt.colorbar(sc, ax=axs[i])
            
            # 坐标轴标签
            if i >= (nrows-1)*ncols:
                axs[i].set_xlabel(param_x, fontsize=10)
            if i % ncols == 0:
                axs[i].set_ylabel(param_y, fontsize=10)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    