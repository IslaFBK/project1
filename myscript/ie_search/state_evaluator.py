import brian2.numpy_ as np
from sklearn.linear_model import LinearRegression
from levy import fit_levy
'''
Judgement of critical or not
'''
def is_critical_state(msd, jump_interval, pdx, alpha_range=(1.00, 1.30), msd_r2_thresh=0.99):
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
