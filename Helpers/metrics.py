import numpy as np

def calculate_waic(log_likelihood):
    """waic and lppd metric"""
    likelihood = np.exp(log_likelihood)
    mean_likelihood = np.mean(likelihood, axis=0)
    lppd = np.sum(np.log(mean_likelihood))
    var = np.var(log_likelihood, axis=0)
    p_waic = np.sum(var)
    waic = -2 * (lppd - p_waic)
    
    return {'lppd': lppd, 'waic': waic}