from joblib import Parallel, delayed
import brian2.numpy_ as np
from brian2 import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from levy import fit_levy, levy
import matplotlib.image as mpimg
from pathlib import Path
import mydata
from connection import get_stim_scale
from myscript.ie_search.compute_MSD_pdx import compute_MSD_pdx
'''
rerun 60 times, draw MSD and pdx
'''
#%% OS operation
# test if data_dir exists, if not, create one.
# FileExistsError means if menu is create by other progress or thread, ignore it.
root_dir = 'ie_ratio_2/'
Path(root_dir).mkdir(parents=True, exist_ok=True)
data_dir = f'{root_dir}/raw_data/'
Path(data_dir).mkdir(parents=True, exist_ok=True)
def load_repeat(param, 
                 n_repeat=60, 
                 save_path_MSD:str=None, 
                 save_path_pdx:str=None,
                 save_path_combined:str=None):
    ie_r_e1, ie_r_i1 = param
    # common title & path
    common_title = rf'$\zeta^{{E}}$: {ie_r_e1:.4f}, $\zeta^{{I}}$: {ie_r_i1:.4f}'
    common_path = f're{ie_r_e1:.4f}_ri{ie_r_i1:.4f}'
    def load_MSD_pdx(param, seed):
        '''load data from disk'''
        data_load = mydata.mydata()
        data_load.load(f"{data_dir}data_{seed}.file")
        # reclaim time parameters
        '''stim 1; constant amplitude'''
        '''no attention''' # ?background?
        stim_dura = 1000 # ms duration of each stimulus presentation
        transient = 3000 # ms initial transient period; when add stimulus
        inter_time = 2000 # ms interval between trials without and with attention
        stim_scale_cls = get_stim_scale.get_stim_scale()
        stim_scale_cls.seed = seed # random seed
        n_StimAmp = 1
        n_perStimAmp = 1
        stim_amp_scale = np.ones(n_StimAmp*n_perStimAmp)
        for i in range(n_StimAmp):
            stim_amp_scale[i*n_perStimAmp:i*n_perStimAmp+n_perStimAmp] = 2**(i)
        stim_scale_cls.stim_amp_scale = stim_amp_scale
        stim_scale_cls.stim_dura = stim_dura
        stim_scale_cls.separate_dura = np.array([300,600])
        stim_scale_cls.get_scale()
        stim_scale_cls.n_StimAmp = n_StimAmp
        stim_scale_cls.n_perStimAmp = n_perStimAmp
        # concatenate
        init = np.zeros(transient//stim_scale_cls.dt_stim)
        stim_scale_cls.scale_stim = np.concatenate((init,stim_scale_cls.scale_stim))
        stim_scale_cls.stim_on += transient
        simu_time_tot = (stim_scale_cls.stim_on[-1,1] + 500)*ms
        simu_time1 = (stim_scale_cls.stim_on[n_StimAmp*n_perStimAmp-1,1] + round(inter_time/2))*ms
        simu_time2 = simu_time_tot - simu_time1
        #%% analysis
        start_time = transient - 500  #data.a1.param.stim1.stim_on[first_stim,0] - 300
        end_time = int(round(simu_time_tot/ms))   #data.a1.param.stim1.stim_on[last_stim,0] + 1500
        window = 15
        data_load.a1.ge.get_spike_rate(start_time=start_time,
                                    end_time=end_time,
                                    sample_interval=1,
                                    n_neuron = data_load.a1.param.Ne,
                                    window = window)
        data_load.a1.ge.get_centre_mass()
        data_load.a1.ge.overlap_centreandspike()
        frames = data_load.a1.ge.spk_rate.spk_rate.shape[2]
        stim_on_off = data_load.a1.param.stim1.stim_on-start_time
        stim_on_off = stim_on_off[stim_on_off[:,0]>=0]
        jump_interval = np.linspace(1, 1000, 100)
        data_load.a1.ge.get_MSD(start_time=start_time,
                                end_time=end_time,
                                sample_interval=1,
                                n_neuron = data_load.a1.param.Ne,
                                window = window,
                                dt = 0.1,
                                slide_interval=1,
                                jump_interval=jump_interval,
                                fit_stableDist='pylevy')
        msd = data_load.a1.ge.MSD.MSD
        jump_interval = data_load.a1.ge.MSD.jump_interval
        pdx = data_load.a1.ge.centre_mass.jump_size[:,1]
        return {
            'msd': msd,
            'jump_interval': jump_interval,
            'pdx': pdx
        }
    
    print('loading')
    # load
    results = Parallel(n_jobs=-1)(
        delayed(load_MSD_pdx)(param, seed)
        for seed in range(n_repeat)
    )
    print('loaded')
    # 假设results里每个元素有msd，jump_interval，pdx
    msds = np.stack([r['msd'] for r in results])
    jump_interval = results[0]['jump_interval']
    msd_mean = np.mean(msds, axis=0)
    msd_std = np.std(msds, axis=0)
    # pdx合并
    all_pdx = np.concatenate([r['pdx'] for r in results])
    # 画图
    print('drawing')
    # MSD
    print('drawing MSD')
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
    print('MSD finished')

    # pdx
    print('drawing pdx')
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
    print('pdx finished')

    # combined graph
    print('drawing combined graph')
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
    print('combined graph finished')