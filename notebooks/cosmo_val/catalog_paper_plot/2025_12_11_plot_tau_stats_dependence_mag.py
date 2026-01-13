# %%
import IPython

ipython = IPython.get_ipython()

if ipython is not None:
    ipython.run_line_magic("load_ext",  "autoreload")
    ipython.run_line_magic("autoreload",  "2")

import os
import copy
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.io import fits
import treecorr

plt.style.use(
    "./matplotlib_config/paper.mplstyle"
)

plt.rcParams["text.usetex"] = True

sns.set_palette("husl")

if ipython is not None:
    ipython.run_line_magic("matplotlib",  "inline")

# %%
path_cat_gal = "/n17data/UNIONS/WL/v1.4.x/v1.4.6/unions_shapepipe_cut_struc_2024_v1.4.6.fits"
path_cat_stars = "/n17data/UNIONS/WL/v1.4.x//full_starcat-0000000.fits"

cat_gal = fits.getdata(path_cat_gal)
cat_star = fits.getdata(path_cat_stars)

# %%
mag = cat_gal['MAG']

quantiles_mag = np.quantile(mag, [0.33, 0.66])
# %%
plt.figure()

plt.hist(mag, bins=50, histtype='step', density=True, color='C0')

plt.xlabel("Magnitude")
plt.ylabel("Density")

plt.show()

# %%
faint_objects = mag > quantiles_mag[1]
medium_objects = (mag <= quantiles_mag[1]) & (mag > quantiles_mag[0])
bright_objects = mag <= quantiles_mag[0]

# %%
treecorr_config = {
    'min_sep': 1.0,
    'max_sep': 250.0,
    'nbins': 20,
    'sep_units': 'arcmin',
    'var_method': 'jackknife',
}

tc_psf = treecorr.Catalog(
    ra=cat_star['RA'],
    dec=cat_star['DEC'],
    g1=cat_star['E1_PSF_HSM'],
    g2=cat_star['E2_PSF_HSM'],
    ra_units='degrees',
    dec_units='degrees',
    npatch=100
)

patch_centers = tc_psf.patch_centers
print("PSF catalog done")

tc_faint = treecorr.Catalog(
    ra=cat_gal['RA'][faint_objects],
    dec=cat_gal['DEC'][faint_objects],
    g1=cat_gal['e1'][faint_objects],
    g2=cat_gal['e2'][faint_objects],
    ra_units='degrees',
    dec_units='degrees',
    patch_centers=patch_centers
)
print("Faint catalog done")

tc_medium = treecorr.Catalog(
    ra=cat_gal['RA'][medium_objects],
    dec=cat_gal['DEC'][medium_objects],
    g1=cat_gal['e1'][medium_objects],
    g2=cat_gal['e2'][medium_objects],
    ra_units='degrees',
    dec_units='degrees',
    patch_centers=patch_centers
)
print("Medium catalog done")

tc_bright = treecorr.Catalog(
    ra=cat_gal['RA'][bright_objects],
    dec=cat_gal['DEC'][bright_objects],
    g1=cat_gal['e1'][bright_objects],
    g2=cat_gal['e2'][bright_objects],
    ra_units='degrees',
    dec_units='degrees',
    patch_centers=patch_centers
)
print("Bright catalog done")



# %%
tau_faint = treecorr.GGCorrelation(**treecorr_config)
tau_faint.process(tc_faint, tc_psf)
print("Faint done")

tau_medium = treecorr.GGCorrelation(**treecorr_config)
tau_medium.process(tc_medium, tc_psf)
print("Medium done")

tau_bright = treecorr.GGCorrelation(**treecorr_config)
tau_bright.process(tc_bright, tc_psf)
print("Bright done")

# %%
plt.figure()

plt.errorbar(
    tau_faint.meanr,
    tau_faint.xip,
    yerr=np.sqrt(tau_faint.varxip),
    fmt='o',
    linestyle='-',
    label='Faint galaxies',
    capsize=2,
    color='C0'
)

plt.errorbar(
    tau_medium.meanr,
    tau_medium.xip,
    yerr=np.sqrt(tau_medium.varxip),
    fmt='o',
    linestyle='-',
    label='Medium galaxies',
    capsize=2,
    color='C1'
)

plt.errorbar(
    tau_bright.meanr,
    tau_bright.xip,
    yerr=np.sqrt(tau_bright.varxip),
    fmt='o',
    linestyle='-',
    label='Bright galaxies',
    capsize=2,
    color='C2'
)

plt.xscale('log')

plt.xlabel(r'$\theta$ [arcmin]')
plt.ylabel(r'$\tau_{0,+}(\theta)$')
plt.legend(fontsize=12)

plt.show()

# %%
#Load tau_stats of the whole sample
path_tau_stats = "/home/guerrini/sp_validation/notebooks/cosmo_val/output/rho_tau_stats/tau_stats_SP_v1.4.6.fits"

tau_stats = fits.getdata(path_tau_stats)

# %%
plt.figure()

plt.errorbar(
    tau_faint.meanr,
    tau_faint.xip,
    yerr=np.sqrt(tau_faint.varxip),
    fmt='o',
    linestyle='-',
    label=rf'Faint galaxies, $m > {quantiles_mag[1]:.2f}$',
    capsize=2,
    color='C0'
)

plt.errorbar(
    tau_medium.meanr,
    tau_medium.xip,
    yerr=np.sqrt(tau_medium.varxip),
    fmt='o',
    linestyle='-',
    label=rf'Intermediate galaxies, ${quantiles_mag[0]:.2f} < m < {quantiles_mag[1]:.2f}$',
    capsize=2,
    color='C1'
)

plt.errorbar(
    tau_bright.meanr,
    tau_bright.xip,
    yerr=np.sqrt(tau_bright.varxip),
    fmt='o',
    linestyle='-',
    label=rf'Bright galaxies, $m < {quantiles_mag[0]:.2f}$',
    capsize=2,
    color='C2'
)

plt.errorbar(
    tau_stats['theta'],
    tau_stats['tau_0_p'],
    yerr=np.sqrt(tau_stats['vartau_0_p']),
    fmt='o',
    linestyle='-',
    label='All galaxies',
    capsize=2,
    color='black'
)

plt.xscale('log')

plt.xlabel(r'$\vartheta$ [arcmin]')
plt.ylabel(r'$\tau_{0,+}(\vartheta)$')
plt.legend(fontsize=10)

plt.savefig("./plots/plot_tau_stats_dependence_mag.pdf")
plt.show()

# %%
very_faint_objects = mag > 24
print(np.sum(very_faint_objects))

# %%
tc_very_faint = treecorr.Catalog(
    ra=cat_gal['RA'][very_faint_objects],
    dec=cat_gal['DEC'][very_faint_objects],
    g1=cat_gal['e1'][very_faint_objects],
    g2=cat_gal['e2'][very_faint_objects],
    ra_units='degrees',
    dec_units='degrees',
    patch_centers=patch_centers
)

tau_very_faint = treecorr.GGCorrelation(**treecorr_config)
tau_very_faint.process(tc_very_faint, tc_psf)
print("Very faint done")

# %%
plt.figure()

plt.errorbar(
    tau_very_faint.meanr,
    tau_very_faint.xip,
    yerr=np.sqrt(tau_very_faint.varxip),
    fmt='o',
    linestyle='-',
    label='Very faint galaxies',
    capsize=2,
    color='C3'
)

plt.xscale('log')

plt.show()

# %%
mag_thr = [23, 23.5, 23.75, 24, 24.25]

tau_output_list = []
for mag_cut in mag_thr:
    objects = mag > mag_cut
    print(f"Mag > {mag_cut}: {np.sum(objects)} galaxies")

    tc_object = treecorr.Catalog(
        ra=cat_gal['RA'][objects],
        dec=cat_gal['DEC'][objects],
        g1=cat_gal['e1'][objects],
        g2=cat_gal['e2'][objects],
        ra_units='degrees',
        dec_units='degrees',
        patch_centers=patch_centers
    )

    tau_object = treecorr.GGCorrelation(**treecorr_config)
    tau_object.process(tc_object, tc_psf)
    print(f"Mag > {mag_cut} done")

    tau_output_list.append(copy.deepcopy(tau_object))

# %%
plt.figure()

for i, mag_cut in enumerate(mag_thr):
    tau_object = tau_output_list[i]

    objects = mag > mag_cut

    percent_object = 100 * np.sum(objects) / len(mag)
    print(f"Mag > {mag_cut}: {percent_object:.2f} % of galaxies")

    plt.errorbar(
        tau_object.meanr,
        tau_object.xip,
        yerr=np.sqrt(tau_object.varxip),
        fmt='o',
        linestyle='-',
        label=rf'$m > {mag_cut}, {percent_object:.2f} \%$ galaxies',
        capsize=2,
        color=f'C{i}'
    )

plt.errorbar(
    tau_stats['theta'],
    tau_stats['tau_0_p'],
    yerr=np.sqrt(tau_stats['vartau_0_p']),
    fmt='o',
    linestyle='-',
    label='All galaxies',
    capsize=2,
    color='black'
)

plt.xscale('log')
plt.xlabel(r'$\vartheta$ [arcmin]')
plt.ylabel(r'$\tau_{0,+}(\vartheta)$')

formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-2, 2))
formatter.set_scientific(True)
plt.gca().yaxis.set_major_formatter(formatter)
plt.ylim(-1.6e-4, 0.3e-4)
plt.legend(
    fontsize=10,
    ncol=2,
)

plt.savefig("./plots/plot_tau_stats_dependence_mag_threshold.pdf")
plt.show()

# %%
