import pickle
import numpy as np

state_file = r'f:\ljn\project1\parallel\state\auto_states.file'
with open(state_file, 'rb') as f:
    data_states = pickle.load(f)

params_list = data_states['params']
states_list = data_states['states']

# 只考虑alpha_jump在target_alpha±0.1范围内的
target_alpha = 2
alpha_tol = 0.1

candidates = []
for p, s in zip(params_list, states_list):
    if s is not None \
    and s.get('alpha_jump') is not None \
    and abs(s['alpha_jump'] - target_alpha) <= alpha_tol:
        candidates.append((p, s))

if not candidates:
    print("没有找到alpha_jump在2附近的参数组")
else:
    # 找r2_jump最接近1的
    best = max(candidates, key=lambda x: x[1]['r2_jump'])
    print("最优参数组：")
    print(best[0])
    print("对应状态：")
    print(best[1])