import numpy as np
import statsmodels.api as sm


class OLSEstimator:
    """
    Wrapper for Ordinary Least Squares (OLS) with optional robust standard errors.
    """

    def __init__(self, add_intercept: bool = True, robust_se: str | None = "HC3", random_state=None):
        self.add_intercept = add_intercept
        self.robust_se = robust_se
        self.random_state = random_state  # unused, kept for interface compatibility

        self._tau_hat = None
        self._se_hat = None
        self._ci = None
        self._cate_estimates = None
        self._model = None

    def fit(self, D, Y, W):
        Y = np.asarray(Y).reshape(-1)
        D = np.asarray(D).reshape(-1)

        if W is None:
            W = np.empty((Y.shape[0], 0))
        else:
            W = np.asarray(W)
            if W.ndim == 1:
                W = W.reshape(-1, 1)

        if not (len(Y) == len(D) == W.shape[0]):
            raise ValueError("Y, D, and W must have the same number of rows.")

        X = np.column_stack([D, W])

        if self.add_intercept:
            X = sm.add_constant(X, has_constant="add")

        res = sm.OLS(Y, X).fit()

        if self.robust_se:
            res = res.get_robustcov_results(cov_type=self.robust_se)

        d_idx = 1 if self.add_intercept else 0

        self._tau_hat = float(res.params[d_idx])
        self._se_hat = float(res.bse[d_idx])

        ci = (self._tau_hat - 1.96 * self._se_hat, self._tau_hat + 1.96 * self._se_hat)
        self._ci = (float(ci[0]), float(ci[1]))

        self._cate_estimates = None
        self._model = res

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
