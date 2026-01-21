import numpy as np
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor

class EconMLEstimator:
    """
    Wrapper for EconML CausalForestDML.
    Estimates Heterogeneous Treatment Effects (CATE).
    """
    def __init__(self, n_estimators=200, random_state=42, rf_params=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.rf_params = rf_params if rf_params is not None else {}
        self._model = None
        self._tau_hat = None
        self._se_hat = None
        self._ci = None
        self._cate_estimates = None

    def fit(self, D, Y, W):
        """
        Fits the Causal Forest model.
        
        Args:
            D (np.array): Treatment assignments.
            Y (np.array): Outcome values.
            W (np.array): Confounding variables (Controls).
        """
        params = {
            'n_estimators': 100,
            'max_features': 0.33, 
            'min_samples_leaf': 3, 
            'min_samples_split': 10, 
            'n_jobs': 1
        }
        params.update(self.rf_params)
        
        est = CausalForestDML(
            model_y=RandomForestRegressor(**params),
            model_t=RandomForestRegressor(**params),
            n_estimators=self.n_estimators,
            discrete_treatment=False,
            random_state=self.random_state,
            n_jobs=1
        )
        

        est.fit(Y, D, X=W, W=None)
        
        self._model = est
        
        self._tau_hat = est.ate(W)
        
        self._cate_estimates = est.effect(W)
        
        te_pred_interval = est.ate_interval(W)
        self._ci = (te_pred_interval[0], te_pred_interval[1])
        
        self._se_hat = (self._ci[1] - self._tau_hat) / 1.96

    @property
    def tau_hat(self):
        return self._tau_hat

    @property
    def se_hat(self):
        return self._se_hat
        
    @property
    def ci(self):
        return self._ci

    @property
    def cate_estimates(self):
        return self._cate_estimates