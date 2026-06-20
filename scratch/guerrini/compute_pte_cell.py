"""Converted from cosmo_val/compute_pte_cell.ipynb (personal exploratory analysis).

Computes chi^2 / PTE statistics on pseudo-C_ell data vectors and compares
covariance matrices across ShapePipe versions. Hardcoded paths point to the
original author's (guerrini) machine and are preserved as-is.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import treecorr
from sp_validation.cosmo_val import CosmologyValidation
from sp_validation.basic import chi2_and_pte
import scipy.stats as stats
import camb

from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

# %%
cv = CosmologyValidation(
    versions=["SP_v1.4.6.3_A", "SP_v1.4.6.3_A_leak_corr"],
    npatch=100,
    nrandom_cell=25,
    cell_method="catalog"
)

# %%
cv.plot_pseudo_cl() #Loads c_ells in cv object

# %%
header = (
        "| Root | $\chi^2$ (EB) | $\chi^2$ (EB) / dof | p-val (EB)| $\chi^2$ (BB) | $\chi^2$ (BB) / dof | p-val (BB) |\n"
        "|------|----------------|------------|---------------|------------|------------------|--------------|\n"
    )

rows = []

for key in cv._pseudo_cls.keys():
    print(f"Key: {key}")
    row = f"| `{key}` "
    print("Metrics for EB pseudo cells:")
    data_vector = cv._pseudo_cls[key]["pseudo_cl"]["EB"]
    cov = cv._pseudo_cls[key]["cov"]["COVAR_EB_EB"].data
    chi2, reduced_chi2, pte = chi2_and_pte(data_vector, cov, verbose=True)
    row += f"| {chi2:.4f} | {reduced_chi2:.4f} | {pte:.4f} "

    print("Metrics for BB pseudo cells:")
    data_vector = cv._pseudo_cls[key]["pseudo_cl"]["BB"]
    cov = cv._pseudo_cls[key]["cov"]["COVAR_BB_BB"].data
    chi2, reduced_chi2, pte = chi2_and_pte(data_vector, cov, verbose=True)
    row += f"| {chi2:.4f} | {reduced_chi2:.4f} | {pte:.4f} |"
    rows.append(row)

# Display in Jupyter
print(header + "\n".join(rows))

# %%
redshift_distr = np.loadtxt('/n17data/mkilbing/astro/data/CFIS/v1.0/nz/dndz_SP_A.txt')
z, dndz = redshift_distr[:, 0], redshift_distr[:, 1]

# %%
#Compute theory Cl'set
h = 0.7
Oc = 0.25
Ob = 0.05

pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2,
                                    NonLinear=camb.model.NonLinear_both, WantTransfer=True)

nside = 1024
lmax = 2*nside

#getthe expected cl's from CAMB
pars.min_l = 1
pars.set_for_lmax(lmax)
pars.SourceWindows = [
    camb.sources.SplinedSourceWindow(z=z, W=dndz, source_type='lensing')
]
theory_cls = camb.get_results(pars).get_source_cls_dict(lmax=lmax, raw_cl=True)

# %%
best_fit_axel = np.load('/n17data/guinot/CFIS_3500/CFIS_1.4.5/inference/plots_nautilus_SP_v1_full_model_no_B_IA_opti_scale/best_model_cell_nautilus_SP_v1_full_model_no_B_IA_opti_scale.npy')

# %%
ell, cl_best_fit = best_fit_axel[0], best_fit_axel[1]

# %%
import healpy as hp
import matplotlib.scale as mscale
from sp_validation.rho_tau import SquareRootScale

mscale.register_scale(SquareRootScale)

nside = 1024
lmax = 2*nside
l = np.arange(lmax+1)
pw = hp.pixwin(nside, lmax=lmax)

pseudo_cl_glass = cv._pseudo_cls['SP_v1.4.5.A']['pseudo_cl']
cov_cl_glass = cv._pseudo_cls['SP_v1.4.5.A']['cov']

ell_eff = pseudo_cl_glass['ELL']

fig, ax = plt.subplots(figsize=(8, 8))

ax.errorbar(ell_eff, ell_eff*pseudo_cl_glass['EE'], yerr=ell_eff*np.sqrt(np.diag(cov_cl_glass["COVAR_EE_EE"].data)), fmt='o', capsize=2, markersize=4, label='EE')
ax.errorbar(ell_eff, ell_eff*pseudo_cl_glass['BB'], yerr=ell_eff*np.sqrt(np.diag(cov_cl_glass["COVAR_BB_BB"].data)), fmt='o', capsize=2, markersize=4, label='BB')
#ax.plot(l, l*theory_cls['W1xW1'], label=r'$C_\ell$ theory', c='k', ls='--')
ax.plot(ell, ell*cl_best_fit, label=r'$C_\ell$ best fit', c='k', ls='--')

ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\ell C_\ell$')

ax.set_xlim(ell_eff.min()-10, ell_eff.max()+100)
ax.set_xscale('squareroot')
ax.set_xticks(np.array([100, 400, 900, 1600]))
ax.minorticks_on()
ax.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
ax.xaxis.set_ticks(minor_ticks, minor=True)

ax.set_yscale('log')

plt.legend()
plt.show()

# %% [markdown]
# ## SP_v1.4.6.3

# %%
#PTE EB SP_v1.4.6.3
data_vector = cv._pseudo_cls["SP_v1.4.6.3_A"]["pseudo_cl"]["BB"][:]
cov = cv._pseudo_cls["SP_v1.4.6.3_A"]["cov"]["COVAR_BB_BB"].data[:, :]

# %%
cv._pseudo_cls["SP_v1.4.6.3_A"]

# %%
chi2_and_pte(data_vector, cov, verbose=True)

# %%
#PTE BB SP_v1.4.6.3
from astropy.io import fits
cosmo_val_outdir = "/home/guerrini/sp_validation/cosmo_val/output/"
data_vector = fits.getdata(cosmo_val_outdir+"/pseudo_cl_SP_v1.4.6.3_B_leak_corr.fits")["BB"][12:-3]
cov = fits.open(cosmo_val_outdir+"/pseudo_cl_cov_SP_v1.4.6.3_B_leak_corr.fits")["COVAR_BB_BB"].data[12:-3, 12:-3]

# %%
path_gaussian_sims = "/home/guerrini/sp_validation/cosmo_val/harmonic_covariance_gaussian_sims_v1463_B/"
for i in range(10_000):
    sim_cl = np.load(
        path_gaussian_sims + f"sample_{i}.npz"
    )
    if i == 0:
        sim_dv = sim_cl['cl_all'][1]
    else:
        sim_dv = np.vstack((sim_dv, sim_cl['cl_all'][1]))

cov_gaussian = np.cov(sim_dv.T)


# %%
data_vector_gaussian = fits.getdata(cosmo_val_outdir+"/pseudo_cl_SP_v1.4.6.3_A_leak_corr.fits")["EB"]
chi2_and_pte(data_vector_gaussian[12:-4], cov_gaussian[12:-4, 12:-4], verbose=True)

# %%
chi2_and_pte(data_vector, cov_gaussian, verbose=True)

# %%
chi2_and_pte(data_vector, cov, verbose=True)

# %% [markdown]
# ## SP_v1.4.5_leak_corr

# %%
#PTE EB SP_v1.4.5
data_vector = cv._pseudo_cls["SP_v1.4.5_intermediate"]["pseudo_cl"]["EB"]
cov = cv._pseudo_cls["SP_v1.4.5_intermediate"]["cov"]["COVAR_EB_EB"].data

# %%
chi2_and_pte(data_vector, cov, verbose=True)

# %%
# PTE BB SP_v1.4.5_leak_corr
data_vector = cv._pseudo_cls["SP_v1.4.5_intermediate"]["pseudo_cl"]["BB"][:]
cov = cv._pseudo_cls["SP_v1.4.5_intermediate"]["cov"]["COVAR_BB_BB"].data[:, :]

# %%
chi2_and_pte(data_vector, cov, verbose=True)

# %% [markdown]
# ## SP_v1.4.1

# %%
#PTE EB SP_v1.4.5
data_vector = cv._pseudo_cls["SP_v1.4.5_bright"]["pseudo_cl"]["EB"]
cov = cv._pseudo_cls["SP_v1.4.5_bright"]["cov"]["COVAR_EB_EB"].data

# %%
chi2_and_pte(data_vector, cov, verbose=True)

# %%
#PTE BB SP_v1.4.5
data_vector = cv._pseudo_cls["SP_v1.4.5_bright"]["pseudo_cl"]["BB"][:]
cov = cv._pseudo_cls["SP_v1.4.5_bright"]["cov"]["COVAR_BB_BB"].data[:, :]

# %%
chi2_and_pte(data_vector, cov, verbose=True)

# %%
chi2_and_pte(data_vector[1:-12], cov[1:-12, 1:-12], verbose=True)

# %%
cov_141 = cv._pseudo_cls["SP_v1.4.1"]["cov"]["COVAR_BB_BB"].data
cov_145 = cv._pseudo_cls["SP_v1.4.5"]["cov"]["COVAR_BB_BB"].data

data_vector_sim = np.load("/home/guerrini/random_stuff/cl.npy")
cov_141_sim = np.load("/home/guerrini/random_stuff/cov_cl.npy")
cov_145_sim = np.load("/home/guerrini/random_stuff/cov_cl_1.4.5.npy")

corr_141 = np.array(
    [[1/(cov_141[i,i]*cov_141[j,j]) for i in range(32)] for j in range(32)]
)
corr_141 = cov_141*np.sqrt(corr_141)

corr_145 = np.array(
    [[1/(cov_145[i,i]*cov_145[j,j]) for i in range(32)] for j in range(32)]
)
corr_145 = cov_145*np.sqrt(corr_145)

# %%
ell = cv._pseudo_cls["SP_v1.4.1"]["pseudo_cl"]["ELL"]

plt.figure()

plt.errorbar(ell, ell*data_vector, yerr=ell*np.sqrt(np.diag(cov_141)), fmt='o', label='SP_v1.4.1', capsize=2)
plt.errorbar(ell, ell*data_vector_sim[3], yerr=ell*np.sqrt(np.diag(cov_141_sim[3])), fmt='o', label='SP_v1.4.1_sim', capsize=2)


plt.show()

# %%
data_vector_sim.shape

# %%
plt.figure()

plt.subplot(121)

plt.imshow(corr_141, cmap='seismic')
plt.colorbar()

plt.subplot(122)

plt.imshow(corr_145, cmap='seismic')
plt.colorbar()

plt.tight_layout()
plt.show()

# %%
plt.figure()

plt.plot(np.diag(cov_141), label="SP_v1.4.1")
#plt.plot(np.diag(cov_145), label="SP_v1.4.5")
plt.plot(np.diag(cov_141_sim[3]), label="SP_v1.4.1 sim")
#plt.plot(np.diag(cov_145_sim[3]), label="SP_v1.4.5 sim")

plt.legend()
plt.xlabel("ell")
plt.ylabel("Covariance")

plt.yscale("log")

plt.show()

# %%
cov_141_sim[3].shape

# %%
n_sims = 8000
n_dv = 32
hartlap_factor = (n_sims-n_dv-2)/(n_sims-1)
print(hartlap_factor)

# %%
data_vector @ np.linalg.inv(cov_141_sim[3]) @ data_vector
