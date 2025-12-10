import numpy as np
import pandas as pd
from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.ensemble import RandomForestRegressor

class DoubleMLEstimator:
    """
    Wrapper for DoubleMLPLR (Partial Linear Regression).
    Assumes constant treatment effect.
    """
    def __init__(self, n_folds=5, n_trees=200, n_rep=9, random_state=42, rf_params=None):
        self.n_folds = n_folds
        self.n_trees = n_trees
        self.n_rep = n_rep
        self.random_state = random_state
        self.rf_params = rf_params if rf_params is not None else {}
        self._model = None
        self._tau_hat = None
        self._se_hat = None
        self._ci = None

    def fit(self, D, Y, W):
        """
        Fits the Double ML model.
        
        Args:
            D (np.array): Treatment assignments.
            Y (np.array): Outcome values.
            W (np.array): Confounding variables (Controls).
        """
        # Prepare data for DoubleML
        # We lump all controls (X and maybe C) into W
        n_obs, n_vars = W.shape
        w_cols = [f'W{i}' for i in range(n_vars)]
        
        df = pd.DataFrame(W, columns=w_cols)
        df['D'] = D
        df['Y'] = Y
        
        dml_data = DoubleMLData(df, 'Y', 'D', w_cols)
        
        # Random Forest Parameters
        # We use a single-threaded RF (n_jobs=1) because the outer simulation loop 
        # is already running in parallel. This prevents "over-subscription" of CPU cores.
        params = {
            'n_estimators': self.n_trees,
            'max_depth': None,
            'max_features': 0.33, 
            'min_samples_leaf': 3,
            'min_samples_split': 10,
            'n_jobs': 1, 
            'random_state': self.random_state
        }
        # Update with user-provided params
        params.update(self.rf_params)
        
        # Instantiate Nuisance Models (Random Forests)
        # model_y predicts Outcome Y from Controls W
        # model_t predicts Treatment D from Controls W
        model_y = RandomForestRegressor(**params)
        model_t = RandomForestRegressor(**params)
        
        # Initialize DoubleMLPLR (Partially Linear Regression)
        # We assume a constant treatment effect (theta is constant across individuals)
        dml_plr = DoubleMLPLR(
            dml_data, 
            model_y, 
            model_t, 
            n_folds=self.n_folds, 
            n_rep=self.n_rep
        )
        
        # Fit the model
        dml_plr.fit()
        
        # Store results
        self._model = dml_plr
        self._tau_hat = dml_plr.coef[0] # The estimated treatment effect
        self._se_hat = dml_plr.se[0]    # The standard error of the estimate
        
        # Get 95% Confidence Interval
        conf_int = dml_plr.confint().iloc[0]
        self._ci = (conf_int['2.5 %'], conf_int['97.5 %'])

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
        # DML PLR assumes constant effect, so no CATE heterogeneity
        return None