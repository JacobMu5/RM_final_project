import numpy as np
import pandas as pd
from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.ensemble import RandomForestRegressor
import optuna

class DoubleMLEstimator:
    """
    Wrapper for DoubleMLPLR (Partial Linear Regression).
    Assumes constant treatment effect.
    """
    def __init__(self, n_folds=5, n_trees=200, n_rep=9, random_state=42, rf_params=None, n_jobs=None):
        self.n_folds = n_folds
        self.n_trees = n_trees
        self.n_rep = n_rep
        self.random_state = random_state
        self.rf_params = rf_params if rf_params is not None else {}
        self.n_jobs = n_jobs
        self._model = None
        self._tau_hat = None
        self._se_hat = None
        self._ci = None

    def _prepare_model(self, D, Y, W):
        """
        Initializes the DoubleMLPLR object.
        """
        n_obs, n_vars = W.shape
        w_cols = [f'W{i}' for i in range(n_vars)]
        
        df = pd.DataFrame(W, columns=w_cols)
        df['D'] = D
        df['Y'] = Y
        
        dml_data = DoubleMLData(df, 'Y', 'D', w_cols)
        
        params = {
            'n_estimators': self.n_trees,
            'max_depth': None,
            'max_features': 0.33, 
            'min_samples_leaf': 3,
            'min_samples_split': 10,
            'n_jobs': self.n_jobs if self.n_jobs is not None else 1,
            'random_state': self.random_state
        }
        params.update(self.rf_params)
        
        model_y = RandomForestRegressor(**params)
        model_t = RandomForestRegressor(**params)
        
        self._model = DoubleMLPLR(
            dml_data, 
            model_y, 
            model_t, 
            n_folds=self.n_folds, 
            n_rep=self.n_rep
        )

    def tune(self, D, Y, W, param_space, n_trials=20, show_progress=False):
        """
        Modular tuning method using tune_ml_models.
        """
        self._prepare_model(D, Y, W)
        
        from optuna.samplers import TPESampler

        optuna_settings = {
            'n_trials': n_trials,
            'show_progress_bar': show_progress,
            'verbosity': optuna.logging.WARNING if show_progress else 0,
            'sampler': TPESampler(seed=self.random_state)
        }
        
        self._model.tune_ml_models(
            ml_param_space=param_space,
            optuna_settings=optuna_settings
        )
        return self._model.params

    def fit(self, D, Y, W):
        """
        Fits the Double ML model.
        """
        if self._model is None:
            self._prepare_model(D, Y, W)
        
        self._model.fit()
        
        self._tau_hat = self._model.coef[0]
        self._se_hat = self._model.se[0]
        
        conf_int = self._model.confint().iloc[0]
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