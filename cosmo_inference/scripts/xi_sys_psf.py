from cosmosis.datablock import names, option_section
from cosmosis.datablock.cosmosis_py import lib
from astropy.io import fits
import numpy as np

#This file should be added to your cosmosis_standard_library following the path shear/xi_sys/xi_sys_psf.py
def setup(options):
    filename = options.get_string(option_section, 'data_file')
    xi_p_name = options.get_string(option_section, 'xi_plus_name')
    xi_m_name = options.get_string(option_section, 'xi_minus_name')
    data = fits.open(filename)
    rho_stats_name = options.get_string(option_section, 'rho_stats_name')
    samples_path = options.get_string(option_section,'samples')

    samples = np.load(samples_path)
    mean = np.mean(samples, axis=0)
    cov = np.cov(samples.T)

    rho_stats = data[rho_stats_name].data

    range_xi_p= options[option_section, 'angle_range_XI_PLUS_1_1']
    range_xi_m= options[option_section, 'angle_range_XI_MINUS_1_1']

    mask_xi_p = (data[xi_p_name].data['ANG'] > range_xi_p[0]) & (data[xi_p_name].data['ANG'] < range_xi_p[1])
    mask_xi_m = (data[xi_m_name].data['ANG'] > range_xi_m[0]) & (data[xi_m_name].data['ANG'] < range_xi_m[1])

    return mean, cov, rho_stats, mask_xi_p, mask_xi_m

def execute(block, config):

    mean, cov, rho_stats, mask_xi_p, mask_xi_m = config

    alpha, beta, eta = np.random.multivariate_normal(mean, cov)
    block['xi_sys', 'alpha'], block['xi_sys', 'beta'], block['xi_sys', 'eta'] = alpha, beta, eta

    xi_sys_p = (
        alpha**2*rho_stats["rho_0_p"][mask_xi_p]
        + beta**2*rho_stats["rho_1_p"][mask_xi_p]
        + eta**2*rho_stats["rho_3_p"][mask_xi_p]
        + 2*alpha*beta*rho_stats["rho_2_p"][mask_xi_p]
        + 2*beta*eta*rho_stats["rho_4_p"][mask_xi_p]
        + 2*alpha*eta*rho_stats["rho_5_p"][mask_xi_p]
    )

    xi_sys_m = (
        alpha**2*rho_stats["rho_0_m"][mask_xi_m]
        + beta**2*rho_stats["rho_1_m"][mask_xi_m]
        + eta**2*rho_stats["rho_3_m"][mask_xi_m]
        + 2*alpha*beta*rho_stats["rho_2_m"][mask_xi_m]
        + 2*beta*eta*rho_stats["rho_4_m"][mask_xi_m]
        + 2*alpha*eta*rho_stats["rho_5_m"][mask_xi_m]
    )

    block['xi_sys', 'xi_sys_vec'] = np.concatenate([xi_sys_p, xi_sys_m])

    return 0