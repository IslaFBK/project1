import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import os
from pathlib import Path
import pickle
import itertools
'''
3D slice of phase graph
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
    state_values = {key: np.array([d['states'][key] for d in valid_data]) for key in state_keys}
    unique_values = {dim: np.unique(param_values[dim]) for dim in param_dims}
    
    # 为每个参数值创建切片图
    for current_val in unique_values[dim_ext]:
        # 为每个状态变量创建图形
        for state_key in state_keys:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 筛选数据
            mask = np.isclose(param_values[dim_ext], current_val)
            
            # 绘制散点图
            sc = ax.scatter(
                param_values[fixed_dims[0]][mask],
                param_values[fixed_dims[1]][mask],
                param_values[fixed_dims[2]][mask],
                c=state_values[state_key][mask],
                cmap='viridis'
            )
            
            # 设置图形属性
            ax.set_xlabel(fixed_dims[0])
            ax.set_ylabel(fixed_dims[1])
            ax.set_zlabel(fixed_dims[2])
            ax.set_title(f"{state_key} at {dim_ext}={current_val}")
            fig.colorbar(sc, ax=ax, label=state_key)
            
            # 设置轴范围
            for dim, set_lim in zip(fixed_dims, [ax.set_xlim, ax.set_ylim, ax.set_zlim]):
                set_lim(arrange[f"num_{dim}"][0], arrange[f"num_{dim}"][1])
            
            # 保存图形
            filename = f"{output_dir}/{state_key}_{dim_ext}_{current_val}.png"
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
# appoint looping parameters
params_loop = {
    'num_ee': np.arange(100, 400+1, 50),
    'num_ei': np.arange(100, 400+1, 50),
    'num_ie': np.arange(100, 400+1, 50),
    'num_ii': np.arange(100, 400+1, 50)
}
# generate looping parameter combinations
loop_combinations = list(itertools.product(*params_loop.values()))
# get total looping number
loop_total = len(loop_combinations)
# common path for states
states_path = ''.join(
    f"{k.replace('num_','')}({v[0]}_{v[-1]})"
    for k, v in sorted(params_loop.items())
)
''' load phase data '''
with open(f"{state_dir}data_{loop_total}_states_{states_path}.file", 'rb') as file:
    data_states = pickle.load(file)

# print("data_states 的类型:", type(data_states))
# print("data_states 的键:", data_states.keys())
# print("\nparams 的类型:", type(data_states['params']))
# print("params 的第一个元素:", data_states['params'][0])  # 假设是列表
# print("params 的第一个元素的类型:", type(data_states['params'][0]))
# print("\nstates 的类型:", type(data_states['states']))
# print("states 的第一个元素:", data_states['states'][0])  # 假设是列表
# print("states 的第一个元素的类型:", type(data_states['states'][0]))

# run
state_graph(data_states, 
            'ii', 
            {'num_ee': [100, 400], 
             'num_ei': [100, 400], 
             'num_ie': [100, 400], 
             'num_ii': [100, 400]},
             output_dir=state_dir)