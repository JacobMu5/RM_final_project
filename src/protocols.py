from typing import Protocol
import numpy as np

class DGP(Protocol):
    """Simple interface for DGPs."""
    def sample(self, n_obs, seed=None):
        ...
    
    @property
    def tau(self): ...

    @property
    def C(self): ...

class Estimator(Protocol):
    """Simple interface for Estimators."""
    def fit(self, D, Y, W): ...

    @property
    def tau_hat(self): ...

    @property
    def se_hat(self): ...

    @property
    def ci(self): ...

    @property
    def cate_estimates(self): ...