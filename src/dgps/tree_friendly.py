import numpy as np

class TreeFriendlyDGP:
    """Docstring for TreeFriendlyDGP

    Designed to be better learnable by Random Forests (Decision Trees).
    Partially build on Fuhr et al. (2024)
    """
    def __init__(
            self,
            n_features = 4,
            tau = 1.0,
            alpha_u = 0.0,
            gamma_u = 0.0,
            theta = 0.0,
            include_collider = False,
            noise_std = 1.0
            confounding = 0.2,
    )