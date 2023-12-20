from cosmosis.datablock import names, option_section
from cosmosis.datablock.cosmosis_py import lib
from astropy.io import fits
import numpy as np

#This file should be added to your cosmosis_standard_library following the path shear/xi_sys/xi_sys_psf.py
def setup(options):
    filename = options.get_string(option_section, 'data_file')
    data = fits.open(filename)
    rho_stats_name = options.get_string(option_section, 'rho_stats_name')
    samples_path = options.get_string(option_section,'samples')

    samples = np.load(samples_path)
    mean = np.mean(samples, axis=0)
    cov = np.cov(samples.T)

    rho_stats = data[rho_stats_name].data

    return mean, cov, rho_stats

def execute(block, config):

    mean, cov, rho_stats = config

    alpha, beta, eta = np.random.multivariate_normal(mean, cov)
    block['xi_sys', 'alpha'], block['xi_sys', 'beta'], block['xi_sys', 'eta'] = alpha, beta, eta

    xi_sys_p = (
        alpha**2*rho_stats["rho_0_p"]
        + beta**2*rho_stats["rho_1_p"]
        + eta**2*rho_stats["rho_3_p"]
        + 2*alpha*beta*rho_stats["rho_2_p"]
        + 2*beta*eta*rho_stats["rho_4_p"]
        + 2*alpha*eta*rho_stats["rho_5_p"]
    )

    xi_sys_m = (
        alpha**2*rho_stats["rho_0_m"]
        + beta**2*rho_stats["rho_1_m"]
        + eta**2*rho_stats["rho_3_m"]
        + 2*alpha*beta*rho_stats["rho_2_m"]
        + 2*beta*eta*rho_stats["rho_4_m"]
        + 2*alpha*eta*rho_stats["rho_5_m"]
    )

    block['xi_sys', 'xi_sys_vec'] = np.concatenate([xi_sys_p, xi_sys_m])

    return 0