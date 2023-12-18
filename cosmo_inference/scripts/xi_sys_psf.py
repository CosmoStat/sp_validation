from cosmosis.datablock import names, option_section
from astropy.io import fits
import numpy as np

#This file should be added to your cosmosis_standard_library following the path shear/xi_sys/xi_sys_psf.py
def setup(options):
    filename = options.get_string(option_section, 'data_file')
    data = fits.open(filename)
    rho_stats_name = options.get_string(option_section, 'rho_stats_name')
    xi_plus_name = options.get_string(option_section, 'xi_plus_name')
    xi_minus_name = options.get_string(option_section, 'xi_minus_name')
    samples_path = options.get_string(option_section,'samples')
    return data, filename, rho_stats_name, xi_plus_name, xi_minus_name, samples_path

def execute(block, config):

    data, filename, rho_stats_name, xi_plus_name, xi_minus_name, samples_path = config

    samples = np.load(samples_path)
    mean = np.mean(samples, axis=0)
    cov = np.cov(samples.T)

    alpha, beta, eta = np.random.multivariate_normal(mean, cov)
    #alpha, beta, eta = block[cosmo, 'alpha'], block[cosmo, 'beta'], block[cosmo, 'eta']

    xi_p, xi_m = data[xi_plus_name], data[xi_minus_name]

    rho_stats = data[rho_stats_name].data

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

    xi_p.data["VALUE"] = xi_p.data["RAW_VALUE"]+xi_sys_p #SG: will add xi_sys for each sample. We need to keep in memory the value of the xi_sys
    xi_m.data["VALUE"] = xi_m.data["RAW_VALUE"]+xi_sys_m

    data.writeto(filename, overwrite=True)

    return 0