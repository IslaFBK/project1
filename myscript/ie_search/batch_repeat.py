from joblib import Parallel, delayed
import brian2.numpy_ as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from levy import fit_levy, levy
import matplotlib.image as mpimg
from myscript.ie_search.compute_MSD_pdx import compute_MSD_pdx

import logging
logging.getLogger('brian2').setLevel(logging.WARNING)

'''
rerun 60 times, draw MSD and pdx
'''
def batch_repeat(param, 
                 n_repeat=60, 
                 save_path_MSD:str=None, 
                 save_path_pdx:str=None,
                 save_path_combined:str=None):
    # compute
    results = Parallel(n_jobs=-1)(
        delayed(compute_MSD_pdx)(param, seed=i, index=i)
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
    # MSD
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
    plt.plot(10**x_fit, 10**y_pred, 'r--', label='Linear Fit')
    plt.fill_between(jump_interval, msd_mean-msd_std, msd_mean+msd_std, color='gray', alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\tau$ (ms)', fontsize=16)
    plt.ylabel('MSD (gridpoint$^2$)', fontsize=16)
    plt.text(
        0.2, 0.8,
        rf'$\tau^{{{slope_str}}}\\ R^2:{{{r2_str}}}$',
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
    bins = np.arange(-41, 41 + 2, 2)
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

