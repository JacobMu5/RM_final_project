import numpy as np

class TreeFriendlyDGP:
    """Docstring for TreeFriendlyDGP

    Designed to be better learnable by Random Forests (Decision Trees).
    Partially build on Fuhr et al. (2024)
    """
    def __init__(
        self, 
        n_features=4,
        tau=1.0,
        alpha_u=0.0,
        gamma_u=0.0, 
        theta=0.0,
        include_collider=False,
        include_linear_collider=False,
        noise_std=1.0, 
        confounding_strength=0.2,
    ):
        """
        Initialize the TreeFriendly DGP.

        Args:
            n_features (int): Number of features (X).
            tau (float): Constant treatment effect.
            alpha_u (float): Strength of effect of Hidden Confounder on Treatment.
            gamma_u (float): Strength of effect of Hidden Confounder on Outcome.
            theta (float): Collider strength (effect of D*Y on C).
            include_collider (bool): If True, returns Collider C as a feature in W.
            include_linear_collider (bool): Whether to include the linear collider.
            noise_std (float): Standard deviation of Gaussian noise.
            confounding_strength (float): Coefficient for the nuisance function g(X).
        """
        self.n_features = n_features
        self._tau = tau
        self.alpha_u = alpha_u
        self.gamma_u = gamma_u
        self.theta = theta
        self.include_collider = include_collider
        self.include_linear_collider = include_linear_collider
        self.noise_std = noise_std
        self.confounding_strength = confounding_strength
        
        self._C = None
        self._C_linear = None


    @property
    def tau(self):
        return self._tau

    @property
    def C(self):
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
        Generate a sample from the DGP.

        Args:
            n_obs (int): Number of observations.
            seed (int, optional): Random seed.

        Returns:
            tuple: (D, Y, W) where:
                D (np.ndarray): Treatment vector.
                Y (np.ndarray): Outcome vector.
                W (np.ndarray): Covariates matrix (includes C if include_collider=True).
        """
        rng = np.random.default_rng(seed)
        
        U = rng.normal(0, 1, n_obs)
        
        if self.n_features < 4:
            raise ValueError("TreeFriendlyDGP requires at least 4 features.")
            
        X = rng.normal(0, 1, (n_obs, self.n_features))
        
        g_X = self.confounding_strength * (0.1 * (X[:, 0]**3) + 0.5 * np.sin(X[:, 1]))
        
        D = g_X + self.alpha_u * U + rng.normal(0, self.noise_std, n_obs)
        
        Y = self._tau * D + g_X + self.gamma_u * U + rng.normal(0, self.noise_std, n_obs)
        
        self._C = self.theta * (D * Y) + rng.normal(0, self.noise_std, n_obs)
        self._C_linear = self.theta * (0.5 * D + 0.3 * Y + 0.1 * X[:, 0]) + rng.normal(0, self.noise_std, n_obs)
        
        W = X
        if self.include_collider:
            W = np.column_stack([W, self._C])
        if self.include_linear_collider:
            W = np.column_stack([W, self._C_linear])
            
        return D, Y, W