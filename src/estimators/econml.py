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
        # We use CausalForestDML because we want to see if it finds "fake" heterogeneity
        # (i.e. if it thinks the effect varies based on the collider C)
        
        # Random Forest Parameters
        # Single-threaded (n_jobs=1) for inner models to allow efficient outer parallelization
        params = {
            'n_estimators': 100, # Internal RFs usually use fewer trees
            'max_features': 0.33, 
            'min_samples_leaf': 3, 
            'min_samples_split': 10, 
            'n_jobs': 1
        }
        params.update(self.rf_params)
        
        # Initialize CausalForestDML
        # This uses two Random Forests (model_y, model_t) to residualize Y and D,
        # and then runs a Causal Forest on the residuals to find Heterogeneous Effects.
        est = CausalForestDML(
            model_y=RandomForestRegressor(**params),
            model_t=RandomForestRegressor(**params),
            n_estimators=self.n_estimators,
            discrete_treatment=False,
            random_state=self.random_state,
            n_jobs=1
        )
        
        # Fit the model
        # We pass W as 'X' (Heterogeneity Features) because we want to inspect 
        # how the treatment effect varies with these controls.
        est.fit(Y, D, X=W, W=None)
        
        self._model = est
        
        # 1. Average Treatment Effect (ATE)
        # The overall effect averaged across the population
        self._tau_hat = est.ate(W)
        
        # 2. CATE (Conditional Average Treatment Effect)
        # The effect estimated for EACH INDIVIDUAL person.
        # This is what we plot in the "Microscope View".
        self._cate_estimates = est.effect(W)
        
        # 3. Confidence Interval for ATE
        te_pred_interval = est.ate_interval(W)
        self._ci = (te_pred_interval[0], te_pred_interval[1])
        
        # Back out standard error (approximate from CI width)
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