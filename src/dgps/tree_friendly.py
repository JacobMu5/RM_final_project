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
            noise_std (float): Standard deviation of Gaussian noise.
            confounding_strength (float): Coefficient for the nuisance function g(X).
        """
        self.n_features = n_features
        self._tau = tau
        self.alpha_u = alpha_u
        self.gamma_u = gamma_u
        self.theta = theta
        self.include_collider = include_collider
        self.noise_std = noise_std
        self.confounding_strength = confounding_strength
        
        self._C = None


    @property
    def tau(self):
        return self._tau

    @property
    def C(self):
        if self._C is None:
            raise ValueError("Data has not been sampled yet.")
        return self._C
    
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
        
        # 1. Generate Confounder U (Unobserved) - here not used since alpha, gamma = 0
        U = rng.normal(0, 1, n_obs)
        
        # 2. Generate Covariates X
        # We need at least 4 features for the paper's specification
        if self.n_features < 4:
            raise ValueError("TreeFriendlyDGP requires at least 4 features.")
            
        X = rng.normal(0, 1, (n_obs, self.n_features))
        
        # Construct nuisance function g(X) using the first 4 features
        # We use the "Reduced Cubic-Sin" specification confirmed to have good coverage.
        
        # g(X) = confounding_strength * (0.1 * X1^3 + 0.5 * sin(X2))
        g_X = self.confounding_strength * (0.1 * (X[:, 0]**3) + 0.5 * np.sin(X[:, 1]))
        
        # 3. Generate Treatment D
        # D = g(X) + alpha*U + noise
        D = g_X + self.alpha_u * U + rng.normal(0, self.noise_std, n_obs)
        
        # 4. Generate Outcome Y
        # Y = tau*D + g(X) + gamma*U + noise
        Y = self._tau * D + g_X + self.gamma_u * U + rng.normal(0, self.noise_std, n_obs)
        
        # 5. Generate Collider C
        # C = theta * (D * Y) + noise
        self._C = self.theta * (D * Y) + rng.normal(0, self.noise_std, n_obs)
        
        if self.include_collider:
            W = np.column_stack([X, self._C])
        else:
            W = X
            
        return D, Y, W