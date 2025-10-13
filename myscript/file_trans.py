import pickle
import numpy as np
from scipy.io import savemat

n_sample = 1000
state_dir = 'ie_ratio_2/state/'
rf_history_path = f'{state_dir}/rf_landscape_{n_sample}.file'
with open(rf_history_path, 'rb') as file:
    r_rf_history = pickle.load(file)

x = []
y = []
z = []
for info in r_rf_history:
    param = info[0]['param']
    r_rf = info[0]['r_rf_result'][0]['r_rf']
    if r_rf is None or not isinstance(r_rf, (int, float)) or not (r_rf == r_rf):
        continue
    x.append(param[0])
    y.append(param[1])
    z.append(r_rf)

x = np.array(x)
y = np.array(y)
z = np.array(z)

savemat(f'{state_dir}/rf_landscape_{n_sample}.mat', {'x': x, 'y': y, 'z': z})