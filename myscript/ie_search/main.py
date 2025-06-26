import brian2.numpy_ as np
from brian2.only import *
import matplotlib.pyplot as plt
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
from critical_states_search import compute_MSD_pdx
from state_evaluator import is_critical_state
from utils import is_param_near
from batch_repeat import batch_repeat


plt.rcParams.update({
    "text.usetex": True,  # 启用 LaTeX 渲染
    "font.family": "serif",  # 使用衬线字体（类似 LaTeX 默认）
    "font.serif": ["Times New Roman"]  # 指定字体
})

#%%
prefs.codegen.target = 'cython'

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
vedio_dir = f'{root_dir}/vedio/'
Path(vedio_dir).mkdir(parents=True, exist_ok=True)
state_dir = f'{root_dir}/state/'
Path(state_dir).mkdir(parents=True, exist_ok=True)
MSD_dir = f'./{graph_dir}/MSD/'
Path(MSD_dir).mkdir(parents=True, exist_ok=True)
pdx_dir = f'./{graph_dir}/pdx/'
Path(pdx_dir).mkdir(parents=True, exist_ok=True)
combined_dir = f'./{graph_dir}/combined'
Path(combined_dir).mkdir(parents=True, exist_ok=True)

#%%
# common title & path
param = (1.8, 2.4)
ie_r_e1, ie_r_i1 = param
n_repeat = 128
common_title = rf'$\zeta^{{E}}$: {ie_r_e1:.4f}, $\zeta^{{I}}$: {ie_r_i1:.4f}'
common_path = f're{ie_r_e1:.4f}_ri{ie_r_i1:.4f}'

save_path_MSD = f'{MSD_dir}/{common_path}_{n_repeat}.png'
save_path_pdx = f'{pdx_dir}/{common_path}_{n_repeat}.png'
save_path_combined = f'{combined_dir}/{common_path}_{n_repeat}.png'

batch_repeat(
    param=param,
    n_repeat=n_repeat,
    save_path_MSD=save_path_MSD,
    save_path_pdx=save_path_pdx,
    save_path_combined=save_path_combined
)