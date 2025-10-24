import brian2.numpy_ as np
from sklearn.linear_model import LinearRegression
from levy import fit_levy
import os
import pickle

import logging
logging.getLogger('brian2').setLevel(logging.WARNING)

'''
tools
'''

def wrapTo2Pi(angle):
    #positiveinput = (angle > 0)
    angle = np.mod(angle, 2*np.pi)
    #angle[(angle==0) & positiveinput] = 2*np.pi
    return angle

def wrapToPi(angle):
    select = (angle < -np.pi) | (angle > np.pi) 
    angle[select] = wrapTo2Pi(angle[select] + np.pi) - np.pi
    return angle

def periodic_distance(i, j, cx, cy, N):
    '''
    i, j: lattice point coordinates
    cx, cy: centre coordinates
    N: lattice number
    '''
    # # map to complex unit circle
    # theta_i = i / N * 2 * np.pi
    # theta_j = j / N * 2 * np.pi
    # theta_cx = cx / N * 2 * np.pi
    # theta_cy = cy / N * 2 * np.pi
    # # unit circle coordinates
    # xi, yi = np.cos(theta_i), np.sin(theta_i)
    # xj, yj = np.cos(theta_j), np.sin(theta_j)
    # xcx, ycx = np.cos(theta_cx), np.sin(theta_cx)
    # xcy, ycy = np.cos(theta_cy), np.sin(theta_cy)
    # # Euclidean distance
    # dx = np.sqrt((xi - xcx)**2 + (yi - ycy)**2)
    # dy = np.sqrt((xj - xcy)**2 + (yj - ycy)**2)
    # return np.sqrt(dx**2 + dy**2)
    dx = np.minimum(np.abs(i - cx), N - np.abs(i - cx))
    dy = np.minimum(np.abs(j - cy), N - np.abs(j - cy))
    return np.sqrt(dx**2 + dy**2)

def wave_packet_exist(spk_rate, centre, r=0.6, epsilon=0.1, min_ratio=0.5):
    '''
    Determine whether wave packet exist
    spk_rate: (Nx, Ny, T) firing rate
    centre: (T, 2) centre coordinate, [:, 0] for vertical, [:, 1] for horizontal
    r: the ratio of the maximum diameter of the circle to the length of side
    epsilon: the maximal rate of firing rate out the circle to the global firing rate
    return: exist or not
    '''
    Nx, Ny, T = spk_rate.shape
    L = Nx
    max_radius = r * L / 2

    # generate lattice coordinates
    x = np.arange(Nx)
    y = np.arange(Ny)
    xx, yy = np.meshgrid(x, y, indexing='ij') # (Nx, Ny)

    count = 0
    for t in range(T):
        cx, cy = centre[t]
        # periodical boundary
        dist = periodic_distance(xx, yy, cx, cy, L)
        mask = dist <= max_radius
    
        # determine 1
        global_count = np.sum(spk_rate)
        circle_count = np.sum(spk_rate[:, :, t][mask])
        if circle_count > (1 - epsilon) * global_count:
            count += 1
            continue
        
        # determine 2
        max_val = spk_rate[:, :, t].max()
        global_max_count = np.sum(spk_rate[:, :, t] == max_val)
        circle_max_count = np.sum(spk_rate[:, :, t][mask] == max_val)
        if global_max_count > 0 and circle_max_count > (1 - epsilon) * global_max_count:
            count +=1
        
    return count >= min_ratio * T

def is_critical_state(msd, jump_interval, pdx, alpha_range=(1.00, 1.50), msd_r2_thresh=0.99):
    # MSD fit
    log_time = np.log10(jump_interval)
    log_msd = np.log10(msd)
    min_points = 5
    n = len(log_time)
    best_r2 = -np.inf
    best_end = min_points
    for end in range(min_points, n+1):
        x = log_time[:end].reshape(-1,1)
        y = log_msd[:end]
        model = LinearRegression().fit(x,y)
        r2 = model.score(x,y)
        if r2 > best_r2:
            best_r2 = r2
            best_end = end
    slope = model.coef_[0]
    # pdx heavy-tail fit
    params, nll = fit_levy(pdx)
    alpha, beta, mu, sigma = params.get()
    # judgement
    is_linear = best_r2 > msd_r2_thresh
    is_heavy_tail = alpha_range[0] < alpha < alpha_range[1]
    return is_linear and is_heavy_tail, dict(r2=best_r2, 
                                             slope=slope, 
                                             end=best_end, 
                                             alpha=alpha, 
                                             beta=beta, 
                                             mu=mu, 
                                             sigma=sigma)

def is_param_near(param, param_list, tol=0.05):
    # param: (ie_r_e1, ie_r_i1)
    for p in param_list:
        if np.linalg.norm(np.array(param) - np.array(p)) < tol:
            return True
    return False

# 删掉alpha > 1.5 的数据
def filter_alpha_data(state_dir, n_sample=1000, alpha_threshold=1.5):
    """
    处理已生成的数据文件，删除alpha>1.5的数据点
    
    参数:
    n_sample: 样本数量，用于构建文件名
    alpha_threshold: alpha阈值，默认1.5
    """
    rf_history_path = f'{state_dir}/rf_landscape_{n_sample}.file'
    
    if not os.path.exists(rf_history_path):
        print(f"文件 {rf_history_path} 不存在")
        return
    
    # 读取数据
    with open(rf_history_path, 'rb') as file:
        r_rf_history = pickle.load(file)
    
    # 过滤数据
    original_count = len(r_rf_history)
    # filtered_history = [info for info in r_rf_history if info.get('alpha', 2) <= alpha_threshold]
    filtered_history = [info for info in r_rf_history if info['alpha'] <= alpha_threshold]
    filtered_count = len(filtered_history)
    
    print(f"原始数据点数量: {original_count}")
    print(f"过滤后数据点数量: {filtered_count}")
    print(f"删除数据点数量: {original_count - filtered_count}")
    
    # 保存过滤后的数据（覆盖原文件）
    with open(rf_history_path, 'wb') as file:
        pickle.dump(filtered_history, file)
    
    print(f"已保存过滤后的数据到 {rf_history_path}")
    
    # 打印被删除的参数信息
    removed_params = [info['param'] for info in r_rf_history if info.get('alpha', 0) > alpha_threshold]
    if removed_params:
        print("被删除的参数:")
        for param in removed_params:
            print(f"  {param}")
    
    return filtered_history