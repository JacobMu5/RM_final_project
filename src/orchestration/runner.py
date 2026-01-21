import numpy as np
import pandas as pd
import inspect


def run_single_simulation(config, seed, sim_id):
    """
    Executes a single simulation and returns quantitative metrics.
    """
    spurious_corr_mult = 0.0
    spurious_corr_linear = 0.0

    dgp, estimator = run_raw_simulation(config, seed)

    tau_hat = estimator.tau_hat
    se_hat = estimator.se_hat
    ci_lower, ci_upper = estimator.ci

    bias = tau_hat - dgp.tau
    coverage = (ci_lower <= dgp.tau <= ci_upper)
    ci_length = ci_upper - ci_lower

    cate_estimates = None
    spurious_corr_mult = 0.0
    if config.dgp_params.get('include_collider'):
        try:
            cate_estimates = estimator.cate_estimates
            if cate_estimates is not None and np.std(cate_estimates) > 0:
                spurious_corr_mult = np.corrcoef(cate_estimates, dgp.C)[0, 1]
        except Exception:
            pass

    if config.dgp_params.get('include_linear_collider', False):
        try:
            spurious_corr_linear = np.corrcoef(cate_estimates, dgp.C_linear)[0, 1]
        except Exception:
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
        'spurious_corr_mult': spurious_corr_mult,
        'spurious_corr_linear': spurious_corr_linear
    }


def run_raw_simulation(config, seed):
    """
    Runs a simulation and returns the raw DGP and Estimator objects.
    Used for diagnostics (Microscope View) where we need internal model states.
    """
    dgp_kwargs = config.dgp_params.copy()

    dgp_init_params = inspect.signature(config.dgp_class.__init__).parameters

    if 'confounding_strength' in dgp_init_params:
        dgp_kwargs['confounding_strength'] = 0.2

    if 'noise_std' in dgp_init_params:
        dgp_kwargs['noise_std'] = 1.0

    dgp = config.dgp_class(**dgp_kwargs)

    D, Y, W = dgp.sample(n_obs=config.sample_size, seed=seed)

    est_kwargs = config.estimator_params.copy()
    tuning_params = est_kwargs.pop('tuning_params', None)
    
    est_kwargs['random_state'] = seed

    estimator = config.estimator_class(**est_kwargs)
    
    if tuning_params is not None:
        estimator.tune(D, Y, W, **tuning_params)
    
    estimator.fit(D, Y, W)

    return dgp, estimator