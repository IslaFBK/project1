from joblib import Parallel, delayed
import brian2.numpy_ as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from levy import fit_levy, levy
import matplotlib.image as mpimg
from myscript.ie_search.compute_MSD_pdx import compute_MSD_pdx
from myscript.ie_search.compute_MSD_pdx import compute_MSD_pdx2

import logging
logging.getLogger('brian2').setLevel(logging.WARNING)

'''
rerun 60 times, draw MSD and pdx
'''

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
    n = int(len(x_fit) * 0.8)
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
    plt.savefig(save_path_MSD, dpi=300, bbox_inches='tight')
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
    plt.savefig(save_path_pdx, dpi=300, bbox_inches='tight')
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
    plt.savefig(save_path_combined, dpi=300, bbox_inches='tight')
    plt.close()

def batch_repeat(param, 
                 n_repeat=64, 
                 save_path_MSD:str=None, 
                 save_path_pdx:str=None,
                 save_path_combined:str=None,
                 video=False):
    # compute
    if video:
        results = Parallel(n_jobs=-1)(
            delayed(compute_MSD_pdx)(param, seed=i, index=i, video=(i==0))
            for i in range(n_repeat)
        )
    else:
        results = Parallel(n_jobs=-1)(
            delayed(compute_MSD_pdx)(param, seed=i, index=i, video=False)
            for i in range(n_repeat)
        )
    ie_r_e1, ie_r_i1 = param
    # common title & path
    common_title = rf'$\zeta^{{E}}$: {ie_r_e1:.4f}, $\zeta^{{I}}$: {ie_r_i1:.4f}'
    common_path = f're{ie_r_e1:.4f}_ri{ie_r_i1:.4f}'
    # 假设results里每个元素有msd，jump_interval，pdx
    msds = np.stack([r['msd'] for r in results])
    jump_interval = results[0]['jump_interval']
    msd_mean = np.mean(msds, axis=0)
    msd_std = np.std(msds, axis=0)
    # pdx合并
    all_pdx = np.concatenate([r['pdx'] for r in results])
    # 画图
    draw_statistical_MSD_pdx(jump_interval=jump_interval,
                             msd_mean=msd_mean,
                             msd_std=msd_std,
                             all_pdx=all_pdx,
                             save_path_MSD=save_path_MSD,
                             save_path_pdx=save_path_pdx,
                             save_path_combined=save_path_combined)

def batch_repeat2(param, 
                  n_repeat=64, 
                  save_path_MSD:str=None, 
                  save_path_pdx:str=None,
                  save_path_combined:str=None,
                  video=False):
    # compute
    if video:
        results = Parallel(n_jobs=-1)(
            delayed(compute_MSD_pdx2)(param, seed=i, index=i, video=(i==0))
            for i in range(n_repeat)
        )
    else:
        results = Parallel(n_jobs=-1)(
            delayed(compute_MSD_pdx2)(param, seed=i, index=i, video=False)
            for i in range(n_repeat)
        )
    ie_r_e1, ie_r_i1, ie_r_e2, ie_r_i2 = param
    # common title & path
    common_title = rf'$\zeta^{{E1}}$: {ie_r_e1:.4f}, $\zeta^{{I1}}$: {ie_r_i1:.4f},$\zeta^{{E2}}$: {ie_r_e2:.4f}, $\zeta^{{I2}}$: {ie_r_i2:.4f}'
    common_path = f're1{ie_r_e1:.4f}_ri1{ie_r_i1:.4f}_re2{ie_r_e2:.4f}_ri2{ie_r_i2:.4f}'
    # area 1
    # 假设results里每个元素有msd，jump_interval，pdx
    msds1 = np.stack([r['msd1'] for r in results])
    jump_interval1 = results[0]['jump_interval1']
    msd_mean1 = np.mean(msds1, axis=0)
    msd_std1 = np.std(msds1, axis=0)
    # pdx合并
    all_pdx1 = np.concatenate([r['pdx1'] for r in results])

    # area 2
    msds2 = np.stack([r['msd2'] for r in results])
    jump_interval2 = results[0]['jump_interval2']
    msd_mean2 = np.mean(msds2, axis=0)
    msd_std2 = np.std(msds2, axis=0)
    #
    all_pdx2 = np.concatenate([r['pdx2'] for r in results])

    # 画图  
    draw_statistical_MSD_pdx(jump_interval=jump_interval1,
                             msd_mean=msd_mean1,
                             msd_std=msd_std1,
                             all_pdx=all_pdx1,
                             save_path_MSD=f'{save_path_MSD}_1.png',
                             save_path_pdx=f'{save_path_pdx}_1.png',
                             save_path_combined=f'{save_path_combined}_1.png')
    
    draw_statistical_MSD_pdx(jump_interval=jump_interval2,
                             msd_mean=msd_mean2,
                             msd_std=msd_std2,
                             all_pdx=all_pdx2,
                             save_path_MSD=f'{save_path_MSD}_2.png',
                             save_path_pdx=f'{save_path_pdx}_2.png',
                             save_path_combined=f'{save_path_combined}_2.png')