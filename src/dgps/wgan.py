"""WGAN-GP Data Generating Process using official ds-wgan package."""

import numpy as np
import pandas as pd
import torch
import wgan
from doubleml.datasets import fetch_401K
from pathlib import Path
from functools import partial


class WGANDGP:
    """Wasserstein GAN-GP for generating synthetic 401(k) data.
    
    Implements conditional WGAN-GP following Athey et al. (2021) with:
    - Bayesian network factorization: P(X|D) and P(Y|X,D)
    - WGAN-GP hyperparameters
    - Early stopping
    """
    
    def __init__(self, theta=0.0, include_collider=False, include_linear_collider=False, print_every=1000, wgan_epochs=10000):
        """Initialize WGAN DGP.
        
        Args:
            theta: Collider strength coefficient
            include_collider: Whether to add bad control C.
            include_linear_collider: Whether to add linear collider.
            print_every: Print frequency.
            wgan_epochs: Maximum training epochs.
        """
        self.theta = theta
        self.include_collider = include_collider
        self.include_linear_collider = include_linear_collider
        self.print_every = print_every
        self.wgan_epochs = wgan_epochs
        
        self.base_path = Path("trained_models")
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.path_gx = self.base_path / "wgan_gx.pth"
        self.path_gy = self.base_path / "wgan_gy.pth"
        self.device = torch.device('cpu')
        
        data = fetch_401K(return_type='DataFrame')
        self.df = data
        self.feature_names = [c for c in data.columns if c not in ['net_tfa', 'e401', 'p401']]
        self.X_cols = self.feature_names
        self.D_cols = ['e401']
        self.Y_cols = ['net_tfa']
        
        self.binary_vars = ['marr', 'twoearn', 'pira', 'hown', 'db']
        self.continuous_vars = [c for c in self.X_cols if c not in self.binary_vars]
        
        self._C = None
        self._C_linear = None
        
        self._prepare_data_wrappers()
        self._load_or_train()
        
        self.tau = self._calculate_oracle_ate(n_obs=1_000_000)

    @property
    def C(self):
        """Returns the multiplicative collider variable."""
        return self._C

    @property
    def C_linear(self):
        """Returns the linear collider variable."""
        return self._C_linear

    def _prepare_data_wrappers(self):
        """Setup DataWrappers and Specifications for G_X and G_Y."""
        x_lower_bounds = {c: max(0, self.df[c].min()) for c in self.continuous_vars}
        AdamGAN = partial(torch.optim.Adam, betas=(0.0, 0.9))
        
        gx_df = self.df[self.X_cols + self.D_cols].copy()
        self.dw_gx = wgan.DataWrapper(
            gx_df, continuous_vars=self.continuous_vars, categorical_vars=self.binary_vars,
            context_vars=self.D_cols, continuous_lower_bounds=x_lower_bounds
        )
        self.spec_gx = wgan.Specifications(
            self.dw_gx, optimizer=AdamGAN, batch_size=128, max_epochs=self.wgan_epochs,
            critic_d_hidden=[128, 128, 128], generator_d_hidden=[128, 128, 128],
            test_set_size=32, critic_dropout=0, generator_dropout=0,
            critic_gp_factor=10, critic_steps=15, critic_lr=1e-4, generator_lr=1e-4,
            device=self.device, print_every=self.print_every
        )
        self.gen_gx = wgan.Generator(self.spec_gx)
        self.crit_gx = wgan.Critic(self.spec_gx)

        gy_df = self.df[self.Y_cols + self.X_cols + self.D_cols].copy()
        self.dw_gy = wgan.DataWrapper(
            gy_df, continuous_vars=self.Y_cols, categorical_vars=[],
            context_vars=self.X_cols + self.D_cols,
            continuous_lower_bounds={'net_tfa': self.df['net_tfa'].min()}
        )
        self.spec_gy = wgan.Specifications(
            self.dw_gy, optimizer=AdamGAN, batch_size=128, max_epochs=self.wgan_epochs,
            critic_d_hidden=[128, 128, 128], generator_d_hidden=[128, 128, 128],
            test_set_size=32, critic_dropout=0, generator_dropout=0,
            critic_gp_factor=10, critic_steps=15, critic_lr=1e-4, generator_lr=1e-4,
            device=self.device, print_every=self.print_every
        )
        self.gen_gy = wgan.Generator(self.spec_gy)
        self.crit_gy = wgan.Critic(self.spec_gy)

    def _train_with_patience(self, gen, crit, x, ctx, spec, patience=2000, min_epochs=1000):
        """Train with early stopping monitoring WD on test set."""
        import torch.utils.data as D
        s = spec.settings
        opt_gen = s['optimizer'](gen.parameters(), lr=s['generator_lr'])
        opt_crit = s['optimizer'](crit.parameters(), lr=s['critic_lr'])
        
        train, test = D.random_split(D.TensorDataset(x, ctx), 
                                     [x.size(0) - s['test_set_size'], s['test_set_size']])
        train_loader = D.DataLoader(train, s['batch_size'], shuffle=True)
        test_loader = D.DataLoader(test, s['batch_size'])
        
        best_wd_abs, counter, best_ep, best_state = float('inf'), 0, 0, None
        step = 1
        
        for ep in range(s['max_epochs']):
            wd_tr, n = 0, 0
            for xb, cb in train_loader:
                xb, cb = xb.to(s['device']), cb.to(s['device'])
                if step % s['critic_steps'] == 0:
                    gen.zero_grad()
                    (-crit(gen(cb), cb).mean()).backward()
                    opt_gen.step()
                else:
                    crit.zero_grad()
                    xf = gen(cb)
                    wd = crit(xb, cb).mean() - crit(xf, cb).mean()
                    loss = -wd + s['critic_gp_factor'] * crit.gradient_penalty(xb, xf, cb)
                    loss.backward()
                    opt_crit.step()
                    wd_tr += wd.item()
                    n += 1
                step += 1
            
            with torch.no_grad():
                wd_te = sum((crit(xb.to(s['device']), cb.to(s['device'])).mean() - 
                             crit(gen(cb.to(s['device'])), cb.to(s['device'])).mean()).item()
                            for xb, cb in test_loader) / len(test_loader)
            
            if ep % 100 == 0:
                print(f"Epoch {ep}: Test WD = {wd_te:.4f}")
            
            if ep >= min_epochs:
                # Early Stopping Logic: Minimize Absolute WD (Convergence to 0)
                if abs(wd_te) < best_wd_abs:
                    best_wd_abs, best_ep, counter = abs(wd_te), ep, 0
                    best_state = {k: v.cpu().clone() for k, v in gen.state_dict().items()}
                else:
                    counter += 1
                    if counter >= patience:
                        # Restore best model
                        print(f"Early stopping triggered at epoch {ep}. Best epoch: {best_ep} (Abs WD={best_wd_abs:.4f})")
                        if best_state:
                            gen.load_state_dict(best_state)
                        return ep
        
        return s['max_epochs'] - 1

    def _load_or_train(self):
        """Load existing models or train new ones."""
        if self.path_gx.exists() and self.path_gy.exists():
            self.gen_gx.load_state_dict(torch.load(self.path_gx))
            self.gen_gy.load_state_dict(torch.load(self.path_gy))
        else:
            x_tr, ctx_tr = self.dw_gx.preprocess(self.df)
            ep_gx = self._train_with_patience(self.gen_gx, self.crit_gx, x_tr, ctx_tr, 
                                              self.spec_gx, 2000, 1000)
            torch.save(self.gen_gx.state_dict(), self.path_gx)
            
            y_tr, ctx_tr_y = self.dw_gy.preprocess(self.df)
            ep_gy = self._train_with_patience(self.gen_gy, self.crit_gy, y_tr, ctx_tr_y,
                                              self.spec_gy, 2000, 1000)
            torch.save(self.gen_gy.state_dict(), self.path_gy)

    def sample(self, n_obs, seed=None):
        """Generate synthetic samples.
        
        Args:
            n_obs: Number of observations to generate
            seed: Random seed for reproducibility
            
        Returns:
            tuple: (D, Y, W) where D is treatment, Y is outcome, W is covariates
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        indices = np.random.choice(len(self.df), n_obs, replace=True)
        D_real = self.df[self.D_cols].iloc[indices].values
        
        temp_df = pd.DataFrame(0.0, index=np.arange(n_obs), columns=self.df.columns)
        for i, col in enumerate(self.D_cols):
            temp_df[col] = D_real[:, i]
        _, ctx_gx = self.dw_gx.preprocess(temp_df)
        
        with torch.no_grad():
            x_fake_df = self.dw_gx.deprocess(self.gen_gx(ctx_gx), ctx_gx)
        
        for c in self.Y_cols:
            x_fake_df[c] = 0.0
        _, ctx_gy = self.dw_gy.preprocess(x_fake_df)
        
        with torch.no_grad():
            y_fake_df = self.dw_gy.deprocess(self.gen_gy(ctx_gy), ctx_gy)
        
        Y_syn = y_fake_df['net_tfa'].values
        W_syn_base = x_fake_df[self.X_cols].values
        
        self._C = self.theta * (D_real.flatten() * Y_syn) + np.random.normal(0, 1, n_obs)
        
        try:
            X0 = W_syn_base[:, 0]
        except IndexError:
            X0 = np.zeros(n_obs)
            
        self._C_linear = self.theta * (0.5 * D_real.flatten() + 0.3 * Y_syn + 0.1 * X0) + np.random.normal(0, 1, n_obs)
        
        W_syn = W_syn_base
        if self.include_collider:
            W_syn = np.column_stack([W_syn, self._C])
        
        if self.include_linear_collider:
            W_syn = np.column_stack([W_syn, self._C_linear])
        
        return D_real.flatten(), Y_syn, W_syn

    def _calculate_oracle_ate(self, n_obs=1_000_000):
        """Calculates the True ATE using 1M Monte Carlo samples."""
        indices = np.random.choice(len(self.df), n_obs, replace=True)
        D_context = self.df[self.D_cols].iloc[indices].copy().reset_index(drop=True)
        
        temp_df_x = pd.DataFrame(0.0, index=np.arange(n_obs), columns=self.df.columns)
        temp_df_x[self.D_cols] = D_context
        
        _, ctx_gx = self.dw_gx.preprocess(temp_df_x)
        
        with torch.no_grad():
            x_fake_df = self.dw_gx.deprocess(self.gen_gx(ctx_gx), ctx_gx)
        
        X_generated = x_fake_df[self.X_cols]
        
        df_d1 = pd.concat([X_generated, pd.DataFrame(1, index=np.arange(n_obs), columns=self.D_cols)], axis=1)
        for c in self.Y_cols: df_d1[c] = 0.0
        _, ctx_gy_d1 = self.dw_gy.preprocess(df_d1)
        
        df_d0 = pd.concat([X_generated, pd.DataFrame(0, index=np.arange(n_obs), columns=self.D_cols)], axis=1)
        for c in self.Y_cols: df_d0[c] = 0.0
        _, ctx_gy_d0 = self.dw_gy.preprocess(df_d0)
        
        with torch.no_grad():
            y1_fake_df = self.dw_gy.deprocess(self.gen_gy(ctx_gy_d1), ctx_gy_d1)
            y0_fake_df = self.dw_gy.deprocess(self.gen_gy(ctx_gy_d0), ctx_gy_d0)
            
        y1 = y1_fake_df['net_tfa'].values
        y0 = y0_fake_df['net_tfa'].values
        
        return np.mean(y1 - y0)
