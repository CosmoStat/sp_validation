"""Converted from cosmo_val/plot_comparison.ipynb (personal exploratory analysis).

Hardcoded paths point to the original author's (guerrini) machine and are
preserved as-is.
"""

# %%
import os

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import healpy as hp
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],  # You can change this font if you prefer
    "axes.labelsize": 14,  # Adjust as needed
    "axes.titlesize": 16,  # Adjust as needed
    "xtick.labelsize": 14,  # Adjust as needed
    "ytick.labelsize": 14,  # Adjust as needed
    "legend.fontsize": 12,  # Adjust as needed
    "figure.figsize": (15, 8),  # Adjust as needed
    "figure.dpi": 600  # Adjust as needed
})
from shear_psf_leakage.rho_tau_stat import RhoStat, TauStat, PSFErrorFit

import treecorr

from getdist import plots, MCSamples

from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
    ipython.run_line_magic("matplotlib", "inline")

# %%
base_dir = '/n17data/mkilbing/astro/data/CFIS/v1.0/ShapePipe/'
path_gal = base_dir + 'v1.4.x/unions_shapepipe_cut_struc_2024_v1.4.2.fits'
path_psf = base_dir + 'unions_shapepipe_psf_2024_v1.4.1.fits'

cat_gal = fits.getdata(path_gal)
cat_psf = fits.getdata(path_psf)

# %%
ra = cat_gal['RA']
dec = cat_gal['DEC']

theta = (90 - dec) * np.pi / 180
phi = ra * np.pi / 180

nside = 2**12
print(f"nside: {nside}")
pix = hp.ang2pix(nside, theta, phi)

n_map = np.zeros(hp.nside2npix(nside))

unique_pix, idx, idx_pix = np.unique(pix, return_index=True, return_inverse=True)

n_map[unique_pix] += np.bincount(idx_pix, weights=cat_gal['w_iv'])

mask = (n_map != 0)

# %%
area = np.sum(mask)*hp.nside2pixarea(nside, degrees=True)
print(f'Area: {area} deg^2')

n_eff_gal = 1/(area*60*60)*(np.sum(cat_gal['w_iv']))**2/np.sum(cat_gal['w_iv']**2)
print(f'Effective number of galaxies: {n_eff_gal}')

n_eff_psf = 1/(area*60*60)*(len(cat_psf))**2/np.sum(len(cat_psf))
print(f'Effective number of PSFs: {n_eff_psf}')

sigma_e = np.sqrt(
    0.5*(np.sum((cat_gal['e1']*cat_gal['w_iv'])**2) + np.sum((cat_gal['e2']*cat_gal['w_iv'])**2)) / np.sum(cat_gal['w_iv']**2)
)
print(f'Intrinsic shape noise: {sigma_e}')

# %%
mask_path = '/n17data/mkilbing/astro/data/CFIS/v1.0/Lensfit/masks/CFIS3500_THELI_mask_hp_8192.fits.gz'
mask_theli = hp.read_map(mask_path)

nside = hp.npix2nside(len(mask_theli))
area = np.sum(mask_theli == 0) * hp.nside2pixarea(nside, degrees=True)
print(f'Area: {area} deg^2')

# %% [markdown]
# ## Plot comparison between catalogs

# %%
#versions = ['SP_v1.3_LFmask_8k', 'SP_v1.3_LFmask_8k_no_alpha', 'SP_v1.4-P1+3', 'SP_v1.4-P1+3_no_alpha', 'DES']
#versions = ['SP_v1.3_LFmask_8k', 'SP_v1.3_LFmask_8k_no_alpha', 'SP_v1.4.1', 'SP_v1.4.1_noleakage', 'DES']
versions = ['SP_v1.4.1', 'SP_v1.4.1_noleakage', 'SP_v1.4.5', 'SP_v1.4.5_leak_corr']
#colors = ['blue', 'blueviolet', 'brown', 'black', 'orange']
#colors = ['blue', 'blueviolet', 'black', 'brown', 'orange']
colors = ['brown', 'black', 'blue', 'midnightblue', 'green', 'olive']
root_dir = '/home/guerrini/sp_validation/cosmo_val/output/'

#Inference of the xi_sys parameters
sep_units = 'arcmin'
coord_units = 'degrees'
theta_min = 0.1
theta_max = 250
nbins = 20


TreeCorrConfig_xi = {
    'ra_units': coord_units,
    'dec_units': coord_units,
    'min_sep': theta_min,
    'max_sep': theta_max,
    'sep_units': sep_units,
    'nbins': nbins,
    'var_method':'jackknife',
}

rho_stats_handler = RhoStat(
    output='.',
    treecorr_config=TreeCorrConfig_xi,
    verbose=True
)

tau_stats_handler = TauStat(
    catalogs=rho_stats_handler.catalogs,
    output='.',
    treecorr_config=TreeCorrConfig_xi,
    verbose=True
    )

psf_fitter = PSFErrorFit(rho_stats_handler, tau_stats_handler, root_dir+'/rho_tau_stats/')

#plot of the xi_sys
plt.figure()

for ver, color in zip(versions, colors):
    print("Plotting version:", ver)
    sample = np.load(root_dir+'rho_tau_stats/samples_'+ver+'.npy')
    psf_fitter.load_rho_stat('rho_stats_'+ver+'.fits')
    quant = 0.84
    quantiles = [1-quant, quant]
    nbins = psf_fitter.rho_stat_handler._treecorr_config['nbins']
    xi_psf_sys_samples = np.array([]).reshape(0, nbins)
    for i in range(len(sample)):
        xi_psf_sys = psf_fitter.compute_xi_psf_sys(sample[i])
        xi_psf_sys_samples = np.vstack((xi_psf_sys_samples, xi_psf_sys))

    xi_psf_sys_mean = np.mean(xi_psf_sys_samples, axis=0)
    xi_psf_sys_quant = np.quantile(xi_psf_sys_samples, quantiles, axis=0)
    theta = psf_fitter.rho_stat_handler.rho_stats['theta']
    plt.plot(theta, xi_psf_sys_mean, c=color, label=r'$\xi_{\rm sys}$ '+ver)
    plt.plot(theta, xi_psf_sys_quant[0], c=color, ls='--')
    plt.plot(theta, xi_psf_sys_quant[1], c=color, ls='--')
    plt.fill_between(theta, xi_psf_sys_quant[0], xi_psf_sys_quant[1], color=color, alpha=0.1)

    if os.path.exists(root_dir+'/xi_plus_'+ver+'.fits'):
        xi_plus = fits.getdata(root_dir+'/xi_plus_'+ver+'.fits')
        plt.plot(xi_plus['ANG'], xi_plus['VALUE'], c=color, ls='-.', label=r"$\xi_+(\vartheta)$ "+ver)
    else:
        raise ValueError("File not found")

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\theta$ (arcmin)')
plt.ylabel(r'$\xi_{+}$')
plt.legend()
plt.title("Systematic error comparison at 68\% level")
plt.savefig('Plots/xi_sys_lq_comparison.png', dpi=600)
plt.show()

# %%
#versions = ['SP_v1.3_LFmask_8k', 'SP_v1.3_LFmask_8k_no_alpha', 'SP_v1.4-P1+3', 'SP_v1.4-P1+3_no_alpha', 'DES']
#colors = ['blue', 'blueviolet', 'brown', 'black', 'orange']
markers = ['o', 'h', 'd','x', '*', 's']

#plot of the xi_sys
plt.figure()

offset = 0.02


for idx, (ver, color, marker) in enumerate(zip(versions, colors, markers)):
    print("Plotting version:", ver)
    sample = np.load(root_dir+'rho_tau_stats/samples_'+ver+'.npy')
    psf_fitter.load_rho_stat('rho_stats_'+ver+'.fits')
    quant = 0.84
    quantiles = [1-quant, quant]
    nbins = psf_fitter.rho_stat_handler._treecorr_config['nbins']
    xi_psf_sys_samples = np.array([]).reshape(0, nbins)
    for i in range(len(sample)):
        xi_psf_sys = psf_fitter.compute_xi_psf_sys(sample[i])
        xi_psf_sys_samples = np.vstack((xi_psf_sys_samples, xi_psf_sys))

    xi_psf_sys_mean = np.mean(xi_psf_sys_samples, axis=0)
    xi_psf_sys_quant = np.quantile(xi_psf_sys_samples, quantiles, axis=0)
    theta = psf_fitter.rho_stat_handler.rho_stats['theta']

    if os.path.exists(root_dir+'/xi_plus_'+ver+'.fits'):
        xi_plus = fits.getdata(root_dir+'/xi_plus_'+ver+'.fits')
        xip = xi_plus['VALUE']
    else:
        raise ValueError("File not found")

    ratio_mean = xi_psf_sys_mean/xip
    ratio_quant = xi_psf_sys_quant/xip

    jittered_theta = theta * (1+idx * offset)

    plt.errorbar(jittered_theta, ratio_mean, yerr=[ratio_mean-ratio_quant[0], ratio_quant[1]-ratio_mean], c=color, label=ver, fmt=marker, capsize=5)


threshold = 0.05
plt.fill_between([0.1, 250], [-threshold, -threshold], [threshold, threshold], color='black', alpha=0.1)
plt.plot([0.1, 250], [threshold, threshold], c='black', ls='--', label=str(threshold*100)+'\% level')
plt.plot([0.1, 250], [-threshold, -threshold], c='k', ls='--')
plt.xscale('log')
plt.xlabel(r'$\theta$ (arcmin)')
plt.ylabel(r'$\xi_{\rm sys}/\xi_{+}$')
plt.legend()
plt.title("Systematic error comparison at 68\% level")
plt.savefig('Plots/ratio_lq_comparison.png', dpi=600)
plt.show()

# %%
ratio_mean_des = ratio_mean

# %%
#versions = ['SP_v1.3_LFmask_8k', 'SP_v1.3_LFmask_8k_no_alpha', 'SP_v1.4-P1+3', 'SP_v1.4-P1+3_no_alpha']
#colors = ['blue', 'blueviolet', 'brown', 'black']
#markers = ['o', 'x', 'h', 'd']

#plot of the xi_sys
plt.figure()

for idx, (ver, color, marker) in enumerate(zip(versions, colors, markers)):
    print("Plotting version:", ver)
    sample = np.load(root_dir+'rho_tau_stats/samples_'+ver+'.npy')
    psf_fitter.load_rho_stat('rho_stats_'+ver+'.fits')
    quant = 0.84
    quantiles = [1-quant, quant]
    nbins = psf_fitter.rho_stat_handler._treecorr_config['nbins']
    xi_psf_sys_samples = np.array([]).reshape(0, nbins)
    for i in range(len(sample)):
        xi_psf_sys = psf_fitter.compute_xi_psf_sys(sample[i])
        xi_psf_sys_samples = np.vstack((xi_psf_sys_samples, xi_psf_sys))

    xi_psf_sys_mean = np.mean(xi_psf_sys_samples, axis=0)
    xi_psf_sys_quant = np.quantile(xi_psf_sys_samples, quantiles, axis=0)
    theta = psf_fitter.rho_stat_handler.rho_stats['theta']

    if os.path.exists(root_dir+'/xi_plus_'+ver+'.fits'):
        xi_plus = fits.getdata(root_dir+'/xi_plus_'+ver+'.fits')
        xip = xi_plus['VALUE']
    else:
        raise ValueError("File not found")

    ratio_mean = (xi_psf_sys_mean/xip)/np.abs(ratio_mean_des)
    ratio_quant = (xi_psf_sys_quant/xip)/np.abs(ratio_mean_des)

    jittered_theta = theta * (1+idx * offset)

    plt.errorbar(jittered_theta, ratio_mean, yerr=[ratio_mean-ratio_quant[0], ratio_quant[1]-ratio_mean], c=color, label=ver, fmt=marker, capsize=5)

threshold = 2
plt.fill_between([0.1, 250], [-threshold, -threshold], [threshold, threshold], color='black', alpha=0.1)
plt.plot([0.1, 250], [threshold, threshold], c='black', ls='--', label=str(threshold)+'X DES level')
plt.plot([0.1, 250], [-threshold, -threshold], c='k', ls='--')
plt.xscale('log')
plt.xlabel(r'$\theta$ (arcmin)')
plt.ylabel(r'$\frac{\xi_{\rm sys}/\xi_{+}}{[\xi_{\rm sys}/\xi_{+}]_{\rm DES}}$')
plt.legend()
plt.title("Systematic error comparison at 68\% level")
plt.savefig('Plots/ratio_DES.png', dpi=600)
plt.show()

# %% [markdown]
# ## Reproduce plot axel

# %% [markdown]
# ### Compare rho-stats

# %%
base_dir = '/sps/euclid/Users/sguerrin/results/cov_mat/'

""" #versions = [
    'DES',
    'SP_v0.0',
    'SP_v1.3_LFmask_8k_SN7',
    'SP_v1.3_LFmask_8k_wcs',
    'SP_v1.4-P1+3'
]

#colors = [
    'orange',
    'cyan',
    'g',
    'blue',
    'brown'
] """

filenames_rho = [
    base_dir+'/'+version+'/rho_stats_'+version+'.fits' for version in versions
]

filenames_tau = [
    base_dir+'/'+version+'/tau_stats_'+version+'.fits' for version in versions
]

# %%
#versions = ['SP_v1.3_LFmask_8k', 'SP_v1.3_LFmask_8k_no_alpha', 'SP_v1.4-P1+3', 'SP_v1.4-P1+3_no_alpha', 'DES']
#colors = ['blue', 'blueviolet', 'brown', 'black', 'orange']
#markers = ['o', 'x', 'h', 'd', '*']

#plot of the xi_sys
plt.figure()

for idx, (ver, color, marker) in enumerate(zip(versions, colors, markers)):
    print("Plotting version:", ver)
    sample = np.load(root_dir+'rho_tau_stats/samples_'+ver+'.npy')
    psf_fitter.load_rho_stat('rho_stats_'+ver+'.fits')
    quant = 0.84
    quantiles = [1-quant, quant]
    nbins = psf_fitter.rho_stat_handler._treecorr_config['nbins']
    xi_psf_sys_samples = np.array([]).reshape(0, nbins)
    for i in range(len(sample)):
        xi_psf_sys = psf_fitter.compute_xi_psf_sys(sample[i])
        xi_psf_sys_samples = np.vstack((xi_psf_sys_samples, xi_psf_sys))

    xi_psf_sys_mean = np.mean(xi_psf_sys_samples, axis=0)
    xi_psf_sys_quant = np.quantile(xi_psf_sys_samples, quantiles, axis=0)
    theta = psf_fitter.rho_stat_handler.rho_stats['theta']

    if os.path.exists(root_dir+'/xi_pm_'+ver+'.txt'):
        xi_plus = fits.getdata(root_dir+'/xi_plus_'+ver+'.fits')
        xip = xi_plus['VALUE']
        gg = treecorr.GGCorrelation(TreeCorrConfig_xi)
        gg.read(root_dir+'/xi_pm_'+ver+'.txt')
        varxip = gg.varxip

    ratio_mean = xi_psf_sys_mean/np.sqrt(varxip)
    ratio_quant = xi_psf_sys_quant/np.sqrt(varxip)
    jittered_theta = theta * (1+idx * offset)

    plt.errorbar(jittered_theta, ratio_mean, yerr=[ratio_mean-ratio_quant[0], ratio_quant[1]-ratio_mean], c=color, label=ver, fmt=marker, capsize=5)

threshold = 0.50
plt.fill_between([0.1, 250], [-threshold, -threshold], [threshold, threshold], color='black', alpha=0.1)
plt.plot([0.1, 250], [threshold, threshold], c='black', ls='--', label=str(threshold*100)+'\% level')
plt.plot([0.1, 250], [-threshold, -threshold], c='k', ls='--')
plt.xscale('log')
plt.xlabel(r'$\theta$ (arcmin)')
plt.ylabel(r'$\xi_{\rm sys}/\sigma_{\xi_+}$')
plt.legend()
plt.title("Systematic error comparison at 68\% level")
plt.savefig('Plots/ratio_varxip_lq_comparison.png', dpi=600)
plt.show()

# %% [markdown]
# ### Plot sigma_xi_+

# %%
plt.figure(figsize=(15, 15))

for ver, color in zip(versions, colors):
    gg = treecorr.GGCorrelation(TreeCorrConfig_xi)
    gg.read(root_dir+'/xi_pm_'+ver+'.txt')
    plt.plot(gg.meanr, np.sqrt(gg.varxip), label=ver, color=color)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\theta$ (arcmin)')
plt.ylabel(r'$\sigma_{\xi_+}$')
plt.legend()
plt.savefig('Plots/sigma_xip_comparison.png', dpi=600)
plt.show()

# %% [markdown]
# ### Plot xi_+

# %%
plt.figure(figsize=(10, 10))

for ver, color in zip(versions, colors):
    gg = treecorr.GGCorrelation(TreeCorrConfig_xi)
    gg.read(root_dir+'/xi_pm_'+ver+'.txt')
    plt.errorbar(gg.meanr, gg.xip, yerr=np.sqrt(gg.varxip), label=ver, color=color, capsize=2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\theta$ (arcmin)')
plt.ylabel(r'$\xi_+$')
plt.title("2-Point correlation functions")
plt.legend()
plt.savefig('./Plots/xi_p.png', dpi=600)
plt.show()

# %%
offset = 0.02
output_folder = "/n09data/guerrini/output_chains/"
add_xi_sys = True

plt.figure(figsize=(15, 15))

plt.subplot(211)

for idx, (ver, color) in enumerate(zip(versions, colors)):
    gg = treecorr.GGCorrelation(TreeCorrConfig_xi)
    gg.read(root_dir+'/xi_pm_'+ver+'.txt')
    jittered_theta = gg.meanr * (1+idx * offset)
    plt.errorbar(jittered_theta, gg.xip, yerr=np.sqrt(gg.varxip), label=ver, color=color, fmt='o', capsize=2, markersize=2)

#Line to guide the eye
root = "SP_v1.4.5_A"
theta = np.loadtxt(output_folder + 'best_fit/{}/shear_xi_plus/theta.txt'.format(root))
theta_arcmin = theta * 180 * 60 / np.pi
shear_xi_plus = np.loadtxt(output_folder + 'best_fit/{}/shear_xi_plus/bin_1_1.txt'.format(root))
shear_xi_minus = np.loadtxt(output_folder + 'best_fit/{}/shear_xi_minus/bin_1_1.txt'.format(root))
xi_sys_plus = np.loadtxt(output_folder + 'best_fit/{}/xi_sys/shear_xi_plus.txt'.format(root))
xi_sys_minus = np.loadtxt(output_folder + 'best_fit/{}/xi_sys/shear_xi_minus.txt'.format(root))
theta_xi_sys = np.loadtxt(output_folder + 'best_fit/{}/xi_sys/theta.txt'.format(root))
theta_xi_sys_arcmin = theta_xi_sys * 180 * 60 / np.pi

mask = (theta_arcmin > 0.1) & (theta_arcmin < 250)
xi_plus_model = shear_xi_plus[mask]
if add_xi_sys:
    xi_plus_model += np.interp(theta_arcmin[mask], theta_xi_sys_arcmin, xi_sys_plus)

plt.plot(theta_arcmin[mask], xi_plus_model, color=color, label=root, alpha=0.5)

plt.ylabel(r'$\xi_{+}$', fontsize=26)
plt.xscale('log')
plt.yscale('log')
plt.legend(loc="lower left", fontsize=8)

plt.subplot(212)

for idx, (ver, color) in enumerate(zip(versions, colors)):
    gg = treecorr.GGCorrelation(TreeCorrConfig_xi)
    gg.read(root_dir+'/xi_pm_'+ver+'.txt')
    jittered_theta = gg.meanr * (1+idx * offset)
    plt.errorbar(jittered_theta, gg.xim, yerr=np.sqrt(gg.varxim), label=ver, color=color, fmt='o', capsize=2, markersize=2)

#Line to guide the eye
mask = (theta_arcmin > 0.1) & (theta_arcmin < 250)
xi_minus_model = shear_xi_minus[mask]
if add_xi_sys:
    xi_minus_model += np.interp(theta_arcmin[mask], theta_xi_sys_arcmin, xi_sys_minus)

plt.plot(theta_arcmin[mask], xi_minus_model, color=color, label=root, alpha=0.5)

plt.xlabel(r'$\theta$ [arcmin]', fontsize=26)
plt.ylabel(r'$\xi_{-}$', fontsize=26)
plt.xscale('log')
plt.yscale('log')
plt.legend(loc="lower left", fontsize=8)

plt.show()


# %%

#Inference of the xi_sys parameters
sep_units = 'arcmin'
coord_units = 'degrees'
theta_min = 0.1
theta_max = 250
nbins = 20


TreeCorrConfig_xi = {
    'ra_units': coord_units,
    'dec_units': coord_units,
    'min_sep': theta_min,
    'max_sep': theta_max,
    'sep_units': sep_units,
    'nbins': nbins,
    'var_method':'shot',
}

colors = ['brown', 'black', 'blue', 'midnightblue', 'green', 'olive', 'magenta']

# %%
offset = 0.005
output_folder = "/n09data/guerrini/output_chains/"
add_xi_sys = True

cat_gal = fits.getdata('/n17data/mkilbing/astro/data/CFIS/v1.0/ShapePipe/unions_shapepipe_2024_v1.4.1.fits')

plt.figure(figsize=(15, 15))

plt.subplot(211)

for i in tqdm(range(1, 8)):
    mask = cat_gal['patch'] != i
    gg = treecorr.GGCorrelation(TreeCorrConfig_xi)
    cat = treecorr.Catalog(ra=cat_gal['ra'][mask], dec=cat_gal['Dec'][mask], g1=cat_gal['e1'][mask], g2=cat_gal['e2'][mask], w=cat_gal['w'][mask], ra_units='deg', dec_units='deg')
    gg.process(cat)
    jittered_theta = gg.meanr * (1+(i-1) * offset)

    plt.subplot(211)
    plt.errorbar(jittered_theta, gg.xip, yerr=np.sqrt(gg.varxip), label=f"P{i}", color=colors[i-1], fmt='o', capsize=2, markersize=2)

    plt.subplot(212)
    plt.errorbar(jittered_theta, gg.xim, yerr=np.sqrt(gg.varxim), label=f"P{i}", color=colors[i-1], fmt='o', capsize=2, markersize=2)

plt.subplot(211)
#Line to guide the eye
root = "SP_v1.4.5_A"
theta = np.loadtxt(output_folder + 'best_fit/{}/shear_xi_plus/theta.txt'.format(root))
theta_arcmin = theta * 180 * 60 / np.pi
shear_xi_plus = np.loadtxt(output_folder + 'best_fit/{}/shear_xi_plus/bin_1_1.txt'.format(root))
shear_xi_minus = np.loadtxt(output_folder + 'best_fit/{}/shear_xi_minus/bin_1_1.txt'.format(root))
xi_sys_plus = np.loadtxt(output_folder + 'best_fit/{}/xi_sys/shear_xi_plus.txt'.format(root))
xi_sys_minus = np.loadtxt(output_folder + 'best_fit/{}/xi_sys/shear_xi_minus.txt'.format(root))
theta_xi_sys = np.loadtxt(output_folder + 'best_fit/{}/xi_sys/theta.txt'.format(root))
theta_xi_sys_arcmin = theta_xi_sys * 180 * 60 / np.pi

mask = (theta_arcmin > 0.1) & (theta_arcmin < 250)
xi_plus_model = shear_xi_plus[mask]
if add_xi_sys:
    xi_plus_model += np.interp(theta_arcmin[mask], theta_xi_sys_arcmin, xi_sys_plus)

plt.plot(theta_arcmin[mask], xi_plus_model, color='blue', label=root, alpha=0.5)

plt.ylabel(r'$\xi_{+}$', fontsize=26)
plt.xscale('log')
plt.yscale('log')
plt.legend(loc="lower left", fontsize=8)

plt.subplot(212)

#Line to guide the eye
mask = (theta_arcmin > 0.1) & (theta_arcmin < 250)
xi_minus_model = shear_xi_minus[mask]
if add_xi_sys:
    xi_minus_model += np.interp(theta_arcmin[mask], theta_xi_sys_arcmin, xi_sys_minus)

plt.plot(theta_arcmin[mask], xi_minus_model, color='blue', label=root, alpha=0.5)

plt.xlabel(r'$\theta$ [arcmin]', fontsize=26)
plt.ylabel(r'$\xi_{-}$', fontsize=26)
plt.xscale('log')
plt.yscale('log')
plt.legend(loc="lower left", fontsize=8)

plt.show()

# %%
cat_gal = fits.getdata('/n17data/mkilbing/astro/data/CFIS/v1.0/ShapePipe/unions_shapepipe_2024_v1.4.1.fits')

# %%
cat_gal['patch']

# %%
theta_min = 0.3
theta_max = 200
nbins = 500
npatch = 25

TreeCorrConfig_map = {
        'ra_units': coord_units,
        'dec_units': coord_units,
        'max_sep': str(theta_max),
        'min_sep': str(theta_min),
        'sep_units': sep_units,
        'nbins': nbins,
        'var_method':'jackknife',
}

n_bins_map = 15
theta_map = np.geomspace(theta_min * 5, theta_max / 2, n_bins_map)

# %%
gg = treecorr.GGCorrelation(TreeCorrConfig_map)

#plot of the xi_sys
plt.figure()

for ver, color in zip(versions, colors):
    print("Plotting version:", ver)
    gg.read(root_dir+f'xi_for_map2_{ver}.txt')

    mapsq, mapsq_im, mxsq, mxsq_im, varmapsq = gg.calculateMapSq(
        R=theta_map,
        m2_uform='Schneider',
    )
    plt.errorbar(theta_map, mapsq, yerr=np.sqrt(varmapsq), c=color, label=r'$\langle M^2_{\rm ap} \rangle$ '+ver, capsize=2)
    plt.errorbar(theta_map, np.abs(mxsq), yerr=np.sqrt(varmapsq), c=color, ls='--', label=r'$\langle M^2_\times \rangle$ '+ver, capsize=2)

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\theta$ (arcmin)')
plt.ylabel(r'dispersion')
plt.title('Mass-aperture')
plt.legend()
plt.savefig('Plots/map_sq_comparison.png', dpi=600)
plt.show()
