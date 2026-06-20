# %%
import IPython

ipython = IPython.get_ipython()

# enable autoreload for interactive sessions
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

import numpy as np
import statsmodels.api as sm
from astropy.io import fits
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.style.use(
    "/home/guerrini/matplotlib_config/paper.mplstyle"
)

plt.rcParams["text.usetex"] = True
sns.set_palette("husl")

#Matplotlib inline if in jupyter
if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

path_cat = "/n17data/UNIONS/WL/v1.4.x/v1.4.6.3/unions_shapepipe_cut_struc_2024_v1.4.6.3.fits"

num_bins = 20
# %%
cat_gal = fits.getdata(path_cat)

# %%
psf_size = "NGMIX_Tpsf_NOSHEAR"
gal_size = "NGMIX_T_NOSHEAR"
size_ratio = cat_gal[psf_size] / (cat_gal[gal_size] + cat_gal[psf_size])

df_gal = pd.DataFrame(
    np.array([
        np.array(cat_gal['e1'], dtype=np.float64),
        np.array(cat_gal['e2'], dtype=np.float64),
        np.array(cat_gal['e1_PSF'], dtype=np.float64),
        np.array(cat_gal['e2_PSF'], dtype=np.float64),
        np.array(size_ratio, dtype=np.float64),
        np.array(cat_gal['snr'], dtype=np.float64),
        np.array(cat_gal['w_des'], dtype=np.float64),
    ]).T, columns=['e1', 'e2', 'e1_PSF', 'e2_PSF', 'size_ratio', 'snr', 'w']
)

# %%
del size_ratio

# %%
n_bins_snr = num_bins
n_bins_size = num_bins

#Create logarithmic bins in size and SNR
df_gal.loc[:, 'bin_R'] = pd.qcut(df_gal['size_ratio'], n_bins_size, labels=False, retbins=False)

#initialize bin snr
df_gal.loc[:, 'bin_snr'] = -999

for ibin_r in range(n_bins_size):
    mask_binr = df_gal['bin_R'].values == ibin_r

    df_gal.loc[mask_binr, 'bin_snr'] = pd.qcut(
        df_gal.loc[mask_binr, 'snr'], 
        n_bins_snr, 
        labels=False, 
        retbins=False
    )

#Group by bin
df_gal_grouped = df_gal.groupby(['bin_R', 'bin_snr'])
ngroups = df_gal_grouped.ngroups
print(f"Number of groups: {ngroups}")

# %%
# Measure leakage per bins
alpha_df = pd.DataFrame(
    0.,
    index=np.arange(ngroups),
    columns=['R', 'SNR', 'alpha_1', 'alpha_2', 'alpha_1_err', 'alpha_2_err']
)

i_group = 0
for name, group in df_gal_grouped:

    #Save weighted average
    alpha_df.loc[i_group, 'R'] = np.average(group['size_ratio'], weights=group['w'])
    alpha_df.loc[i_group, 'SNR'] = np.average(group['snr'], weights=group['w'])

    #Fit linear model to compute alpha
    e1_out = np.array(group['e1'])
    e2_out = np.array(group['e2'])
    weight_out = np.array(group['w'])
    e1_PSF = np.array(group['e1_PSF'])
    e2_PSF = np.array(group['e2_PSF'])
    del group

    #Fit e1
    mod_wls = sm.WLS(e1_out, sm.add_constant(e1_PSF), weights=weight_out)
    try:
        res_wls = mod_wls.fit()
    except:
        raise RunTimeError("Linear regression fit for PSF leakage failed")
    alpha_df.loc[i_group, 'alpha_1'] = res_wls.params[1]
    alpha_df.loc[i_group, 'alpha_1_err'] = np.sqrt(res_wls.cov_params()[1, 1])
    del res_wls, mod_wls

    #Fit e2
    mod_wls = sm.WLS(e2_out, sm.add_constant(e2_PSF), weights=weight_out)
    res_wls = mod_wls.fit()
    alpha_df.loc[i_group, 'alpha_2'] = res_wls.params[1]
    alpha_df.loc[i_group, 'alpha_2_err'] = np.sqrt(res_wls.cov_params()[1, 1])
    del weight_out, res_wls, mod_wls

    i_group += 1

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

SNR_bins_val = alpha_df["SNR"].values
R_bins_val = alpha_df["R"].values

alpha = 0.5 * (alpha_df['alpha_1'].values + alpha_df['alpha_2'].values)
vmin, vmax = np.min(alpha), np.max(alpha)
vmin, vmax = -np.max([np.abs(vmin), np.abs(vmax)]), np.max([np.abs(vmin), np.abs(vmax)])

cset1 = ax.scatter(SNR_bins_val, R_bins_val, c=alpha, cmap='coolwarm', vmin=vmin, vmax=vmax)
ax.set_xlabel("SNR", fontsize=15)
ax.set_ylabel(r"$\mathcal{R}$", fontsize=15)
ax.set_xscale('log')
ax.set_ylim(max(R_bins_val)+0.01, min(R_bins_val)-0.01)
cbar = fig.colorbar(cset1, ax=ax)
cbar.set_label(r"$\alpha$", fontsize=15)

plt.savefig("./plots/alpha_leakage_bin.png", dpi=300)
plt.show()
# %%
