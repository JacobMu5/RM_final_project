import numpy as np

def run_single_simulation(config, seed, sim_id):
    """
    Executes a single simulation.
    """
    np.random.seed(seed)
    
    # 1. Generate Data
    dgp = config.dgp_class(**config.dgp_params)
    D, Y, W = dgp.sample(n_obs=config.sample_size, seed=seed)
    
    # 2. Fit Estimator
    estimator = config.estimator_class(
        **config.estimator_params,
        random_state=seed + 12345
    )
    estimator.fit(D, Y, W)
    
    # 3. Extract Metrics
    tau_hat = estimator.tau_hat
    se_hat = estimator.se_hat
    ci_lower, ci_upper = estimator.ci
    
    # Standard Metrics
    bias = tau_hat - dgp.tau
    coverage = (ci_lower <= dgp.tau <= ci_upper)
    ci_length = ci_upper - ci_lower
    
    # Advanced Metric: Spurious Correlation (Collider Check)
    spurious_corr = 0.0
    if config.dgp_params.get('include_collider'):
        try:
            cate_estimates = estimator.cate_estimates
            # If the model found heterogeneity, check if it correlates with the collider
            if cate_estimates is not None and np.std(cate_estimates) > 0:
                spurious_corr = np.corrcoef(cate_estimates, dgp.C)[0, 1]
        except:
            pass
    
    return {
        'scenario': config.name,
        'sim_id': sim_id,
        'tau_hat': tau_hat,
        'se_hat': se_hat,
        'bias': bias,
        'coverage': coverage,
        'ci_length': ci_length,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'spurious_corr': spurious_corr
    }