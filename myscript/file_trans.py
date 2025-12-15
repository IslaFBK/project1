import pickle
import numpy as np
from scipy.io import savemat

state_dir = 'ie_ratio_2/state/'

#%% 3d dots
n_sample = 1000
delta_gk = 2
rf_history_path = f'{state_dir}/rf_landscape_{n_sample}_{delta_gk}.file'
with open(rf_history_path, 'rb') as file:
    r_rf_history = pickle.load(file)

x = []
y = []
z1 = []
z2 = []
for info in r_rf_history:
    param = info['param']
    r_rf = info['r_rf']
    alpha = info['alpha']
    if r_rf is None or not isinstance(r_rf, (int, float)) or not (r_rf == r_rf):
        continue
    x.append(param[0])
    y.append(param[1])
    z1.append(r_rf)
    z2.append(alpha)

x = np.array(x)
y = np.array(y)
z1 = np.array(z1)
z2 = np.array(z2)

savemat(f'{state_dir}/rf_alpha_landscape_{n_sample}_{delta_gk}.mat', {'x': x, 'y': y, 'z1': z1, 'z2': z2})

#%% 椭圆信息转化
# ellipse_path = f'{state_dir}/critical_ellipse.file'
# with open(ellipse_path, 'rb') as f:
#     ellipse_info = pickle.load(f)

# mean = ellipse_info['mean']
# cov = ellipse_info['cov']
# conf_level = ellipse_info.get('conf_level', 0.99)

# savemat(f'{state_dir}/critical_ellipse.mat', {
#     'mean': mean,
#     'cov': cov,
#     'conf_level': conf_level
# })