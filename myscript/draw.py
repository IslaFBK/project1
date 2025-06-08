import brian2.numpy_ as np
import matplotlib.pyplot as plt
from brian2.only import *
import time
import pickle
import itertools
from analysis import firing_rate_analysis as fra
from analysis import my_analysis as mya
from joblib import Parallel, delayed

plt.rcParams.update({
    "text.usetex": True,  # 启用 LaTeX 渲染
    "font.family": "serif",  # 使用衬线字体（类似 LaTeX 默认）
    "font.serif": ["Times New Roman"]  # 指定字体
})

data_dir = 'parallel/raw_data/'
graph_dir = 'parallel/graph/'
vedio_dir = 'parallel/vedio/'

start = time.perf_counter()
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
# save states
states_path = ''.join(
    f"{k.replace('num_','')}({v[0]}_{v[-1]})"
    for k, v in sorted(params_loop.items())
)
# load phase data and rebuild Analyzer object
''' load phase data '''
with open(f"{data_dir}data_{loop_total}_states_{states_path}.file", 'rb') as file:
    data_states = pickle.load(file)
Analyzer = mya.CriticalityAnalyzer()
Analyzer.params = data_states['params']
Analyzer.states = data_states['states']
# draw phase graph
fixed_params = {'num_ie': 200, 'num_ii': 300}
fixed_path = "_".join(f"{k}{v}" for k, v in sorted(fixed_params.items()))
Analyzer.plot_phase_diagrams(
    param_x='num_ee',
    param_y='num_ei',
    state_vars=['alpha_jump', 'r2_jump', 'alpha_spike', 'r2_spike'],
    fixed_params=fixed_params,  # 固定这两个参数
    save_path=f'./parallel/states_{loop_total}_{fixed_path}.png'
)
print(f'Phase graph of {loop_total} states saved to ./parallel/{fixed_path}')