import numpy as np
from scipy.linalg import toeplitz

class PLRCCDDHNR2018DGP:
    """
    Partially Linear Regression (PLR) Data Generating Process.

    Follows Chernozhukov et al. (2018) for the structural model.

    Attributes:
        n_features (int): Number of control variables.
        tau (float): Treatment effect.
        theta (float): Collider strength.
        include_collider (bool): Whether to include the multiplicative collider.
        include_linear_collider (bool): Whether to include the linear collider.
        noise_std (float): Standard deviation of noise terms.
    """

    def __init__(
        self,
        n_features=20,
        tau=1.0,
        theta=0.0,
        include_collider=False,
        include_linear_collider=False,
        noise_std=1.0,
    ):
        self.n_features = n_features
        self._tau = tau
        self.theta = theta
        self.include_collider = include_collider
        self.include_linear_collider = include_linear_collider
        self.noise_std = noise_std

        self._C = None
        self._C_linear = None

    @property
    def tau(self):
        """Returns experimental treatment effect."""
        return self._tau

    @property
    def C(self):
        """Returns the multiplicative collider variable."""
        if self._C is None:
            raise ValueError("Data has not been sampled yet.")
        return self._C

    @property
    def C_linear(self):
        """Returns the linear collider variable."""
        if self._C_linear is None:
            raise ValueError("Linear collider has not been sampled yet.")
        return self._C_linear

    def sample(self, n_obs, seed=None):
        """
        Generates a sample from the PLR DGP.

        Args:
            n_obs (int): Number of observations.
            seed (int, optional): Random seed.

        Returns:
            tuple: (D, Y, W) where D is treatment, Y is outcome, and W is controls.
        """
        if seed is not None:
            np.random.seed(seed)

        a_0 = 1.0
        a_1 = 0.25
        s_1 = 1.0
        b_0 = 1.0
        b_1 = 0.25
        s_2 = 1.0
        
        cov_mat = toeplitz([np.power(0.7, k) for k in range(self.n_features)])
        
        X = np.random.multivariate_normal(
            np.zeros(self.n_features),
            cov_mat,
            size=[n_obs,],
        )
        
        D = (
            a_0 * X[:, 0]
            + a_1 * np.divide(np.exp(X[:, 2]), 1 + np.exp(X[:, 2]))
            + s_1 * np.random.standard_normal(size=[n_obs,])
        )
        
        Y = (
            self._tau * D
            + b_0 * np.divide(np.exp(X[:, 0]), 1 + np.exp(X[:, 0]))
            + b_1 * X[:, 2]
            + s_2 * np.random.standard_normal(size=[n_obs,])
        )

        rng = np.random.default_rng(seed)

        self._C = self.theta * (D * Y) + rng.normal(0, self.noise_std, n_obs)

        self._C_linear = self.theta * (
            0.5 * D
            + 0.3 * Y
            + 0.1 * X[:, 0]
        ) + rng.normal(0, self.noise_std, n_obs)

        W = X
        if self.include_collider:
            W = np.column_stack([W, self._C])
        if self.include_linear_collider:
            W = np.column_stack([W, self._C_linear])

        return D, Y, W

    def get_nuisance_m(self, X):
        """Calculates true m(X) = E[D|X]."""
        a_0 = 1.0
        a_1 = 0.25
        return (
            a_0 * X[:, 0]
            + a_1 * np.divide(np.exp(X[:, 2]), 1 + np.exp(X[:, 2]))
        )

    def get_nuisance_g(self, X):
        """Calculates true g(X) = E[Y|X, D=0] (approx)."""
        b_0 = 1.0
        b_1 = 0.25
        return (
            b_0 * np.divide(np.exp(X[:, 0]), 1 + np.exp(X[:, 0]))
            + b_1 * X[:, 2]
        )