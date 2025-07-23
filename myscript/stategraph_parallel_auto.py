import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import re
from pathlib import Path
import pickle
import itertools
'''
3D slice of phase graph
Not indeed parallel yet, but can be used to visualize the state space of a system
'''
def state_graph(data_states, dim_ext, arrange, output_dir="output_plots"):
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 提取数据和参数
    params_list = data_states['params']
    states_list = data_states['states']
    
    # 参数维度设置
    param_dims = ['ee', 'ei', 'ie', 'ii']
    if dim_ext not in param_dims:
        raise ValueError(f"dim_ext must be one of {param_dims}")
    
    fixed_dims = [dim for dim in param_dims if dim != dim_ext]
    
    # 收集有效数据
    valid_data = []
    for i in range(min(len(params_list), len(states_list))):
        if states_list[i] is not None:
            valid_data.append({
                'params': params_list[i],
                'states': states_list[i]
            })
    
    if not valid_data:
        raise ValueError("No valid data points found")
    
    # 准备数据数组
    state_keys = list(valid_data[0]['states'].keys())
    param_values = {dim: np.array([d['params'][f'num_{dim}'] for d in valid_data]) for dim in param_dims}
    state_values = {
        key: np.array([
            d['states'][key] if d['states'][key] is not None else np.nan
            for d in valid_data
        ])
        for key in state_keys
    }
    unique_values = {dim: np.unique(param_values[dim]) for dim in param_dims}
    
    # 为每个参数值创建切片图
    for current_val in unique_values[dim_ext]:
        # 为每个状态变量创建图形
        for state_key in state_keys:
            fig = plt.figure(figsize=(16, 12))
        axes = []
        for i in range(4):
            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            axes.append(ax)

        for idx, state_key in enumerate(state_keys):
            vmin = np.nanpercentile(state_values[state_key], 2)
            vmax = np.nanpercentile(state_values[state_key], 99)
            ax = axes[idx]
            # 筛选数据
            mask = np.isclose(param_values[dim_ext], current_val)
            # 绘制散点图
            sc = ax.scatter(
                param_values[fixed_dims[0]][mask],
                param_values[fixed_dims[1]][mask],
                param_values[fixed_dims[2]][mask],
                c=state_values[state_key][mask],
                cmap='viridis',
                vmin=vmin if state_key in ['alpha_jump', 'alpha_spike'] else None,
                vmax=vmax if state_key in ['alpha_jump', 'alpha_spike'] else None
            )
            ax.set_xlabel(fixed_dims[0])
            ax.set_ylabel(fixed_dims[1])
            ax.set_zlabel(fixed_dims[2])
            ax.set_title(f"{state_key} at {dim_ext}={current_val}")
            fig.colorbar(sc, ax=ax, label=state_key)

            # 设置轴范围
            for dim, set_lim in zip(fixed_dims, [ax.set_xlim, ax.set_ylim, ax.set_zlim]):
                set_lim(arrange[f"num_{dim}"][0], arrange[f"num_{dim}"][1])

        # 保存合成图
        filename = f"{output_dir}/allstates_{dim_ext}_{current_val}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"所有图片已保存到 {output_dir} 目录")

# 使用示例
# load phase data and rebuild Analyzer object
root_dir = 'parallel'
data_dir = 'parallel/raw_data/'
graph_dir = 'parallel/graph/'
video_dir = 'parallel/vedio/'
state_dir = 'parallel/state/'
# automatically identify looping parameters
path_list = os.listdir(data_dir)
# filter out non-data files
params_list = []
pattern = r'EE(\d+)_EI(\d+)_IE(\d+)_II(\d+)'
# extract parameters from file names
for path in path_list:
    match = re.search(pattern, path)
    if match:
        params = {
            'num_ee': int(match.group(1)),
            'num_ei': int(match.group(2)),
            'num_ie': int(match.group(3)),
            'num_ii': int(match.group(4))
        }
        params_list.append(params)

# generate looping parameter combinations
loop_combinations = [
    (np.int64(p['num_ee']), np.int64(p['num_ei']), np.int64(p['num_ie']), np.int64(p['num_ii']))
    for p in params_list
]
# get total looping number
loop_total = len(loop_combinations)

''' load phase data '''
with open(f"{state_dir}auto_states.file", 'rb') as file:
    data_states = pickle.load(file)

start = time.perf_counter()
# run
state_graph(data_states, 
            'ee', 
            {'num_ee': [90, 300], 
             'num_ei': [90, 400], 
             'num_ie': [90, 200], 
             'num_ii': [90, 200]},
             output_dir=f'{state_dir}auto_states')
print(f'total time elapsed: {np.round((time.perf_counter() - start)/60,2)} min')
