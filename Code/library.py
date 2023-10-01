import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

def CDF_Inverse(cdf_inv, n_samples):
    u = np.random.uniform(0, 1, n_samples)
    return cdf_inv(u)

def Rejection(f, g, inv_cdf, START, END, K, N):
    x = CDF_Inverse(inv_cdf, N)
    y = np.random.uniform(0, 1, N) * K * g(x)
    return x[y <= f(x)]

def optimize_envelope(f, START, END):
    def objective(params):
        mu, sigma = params
        x = np.linspace(START, END, 1000)
        g = norm.pdf(x, mu, sigma)
        K = np.max(f(x) / g)
        return K  

    init_params = [(START + END) / 2, (END - START) / 4]
    bounds = [(START, END), (1e-5, (END - START) / 2)]
    result = minimize(objective, init_params, bounds=bounds, method='L-BFGS-B')
    
    if not result.success:
        raise ValueError("Optimización no exitosa!")
    
    mu_opt, sigma_opt = result.x
    K_opt = result.fun
    
    return mu_opt, sigma_opt, K_opt


def direct_sampling(function, a, b, iters=1000):
    x = np.random.uniform(a, b, iters)
    y = function(x)
    estimate = np.mean(y) * (b - a)
    return estimate

def control_variates(function, a, b, iters=1000):
    x = np.random.uniform(0, 1, 1000)
    y = function(x)
    c = np.cov(x, y)[0, 1] / np.var(x)
    estimate = np.mean(y + c*(0.5 - np.mean(x))) 
    return estimate

def importance_sampling(function, a, b, iters=1000):
    x_samples = np.linspace(a, b, 10000)
    y_samples = function(x_samples)
    
    mu = np.mean(y_samples)
    std = np.std(y_samples)
    
    x = []
    while len(x) < iters:
        _x = np.random.normal(mu, std)
        if _x >= a and _x <= b:
            x.append(_x)
    x = np.array(x)

    y = function(x) / norm.pdf(x, mu, std)
    estimate = np.mean(y) * (b - a)
            
    return estimate

def stratified_sampling(function, a, b, iters=1000):
    estimate = 0
    strata = np.linspace(a, b, iters + 1)
    for i in range(iters):
        x = np.random.uniform(strata[i], strata[i + 1])
        estimate += function(x) / iters
    return estimate

def antithetic_sampling(function, a, b, iters=1000):
    estimate = 0
    for _ in range(iters // 2):
        u = np.random.uniform(a, b)
        estimate += (function(u) + function(b - (u - a))) / iters
    return estimate

