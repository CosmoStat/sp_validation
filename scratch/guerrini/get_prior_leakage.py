"""Converted from cosmo_val/get_prior_leakage.ipynb (personal exploratory analysis).

Hardcoded paths point to the original author's (guerrini) machine and are
preserved as-is.
"""

# %%
import numpy as np

# %%
base_dir = 'output/rho_tau_stats/'
version = 'SP_v1.4.5_leak_corr'

# %%
samples_leakage = np.load(f'{base_dir}/samples_{version}.npy')

# %%
mean_samples = np.mean(samples_leakage, axis=0)

cov_samples = np.cov(samples_leakage.T)

# %%
prior_psf = {
    'bin1': {
        'mean': mean_samples,
        'cov': cov_samples,
    }
}

# %%
np.save(f'{base_dir}/prior_psf_sys_{version}.npy', prior_psf)
