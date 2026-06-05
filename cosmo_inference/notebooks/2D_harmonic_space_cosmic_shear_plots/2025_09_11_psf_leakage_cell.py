# %%
import os

# Trick to plot with tex
os.environ["LD_LIBRARY_PATH"] = ""
os.environ["CONDA_PREFIX"] = "/home/guerrini/.conda/envs/sp_validation_3.11"

from IPython import get_ipython

ipython = get_ipython()

# enable autoreload for interactive sessions
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import scale as mscale
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import healpy as hp
import pymaster as nmt

from sp_validation.utils_cosmo_val import SquareRootScale

mscale.register_scale(SquareRootScale)

plt.style.use(
    '/home/guerrini/matplotlib_config/paper.mplstyle'
)
plt.rcParams['text.usetex'] = True

sns.set_palette("Dark2")

if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

# %%
#Import galaxy and star catalogs
cat_gal = fits.getdata(
    "/n17data/UNIONS/WL/v1.4.x/v1.4.6.3/unions_shapepipe_cut_struc_2024_v1.4.6.3.fits"
)

cat_star = fits.getdata(
    "/n17data/UNIONS/WL/v1.4.x/unions_shapepipe_psf_2024_v1.4.a.fits"
)

glass_root_dir = "/n09data/guerrini/glass_mock_v1.4.6.3_v2/results/"

# %%
def get_shape_noise(e1, e2, w):
    shape_noise = 0.5 * np.sum(w**2 * (e1**2 + e2**2)) / np.sum(w**2)
    return np.sqrt(shape_noise)

# %%
#Define namaster binning
lmin = 8
lmax = 2048
b_lmax = lmax - 1

n_ells = 32

ells = np.arange(lmin, lmax+1)

start = np.power(lmin, 1/2)
stop = np.power(lmax, 1/2)
bins_ell = np.power(np.linspace(start, stop, 33), 2)

bpws = np.digitize(ells.astype(float), bins_ell) - 1
bpws[0] = 0
bpws[-1] = 31

b = nmt.NmtBin(ells=ells, bpws=bpws, lmax=b_lmax)


# %%
ell_eff = b.get_effective_ells()

#Create namaster fields
f_gal_corrected = nmt.NmtFieldCatalog(
    positions=[cat_gal['RA'], cat_gal['DEC']],
    weights=cat_gal['w_des'],
    field=[cat_gal['e1_leak_corrected'], -cat_gal['e2_leak_corrected']],
    lmax=b_lmax,
    lmax_mask=b_lmax,
    spin=2,
    lonlat=True
)

f_gal = nmt.NmtFieldCatalog(
    positions=[cat_gal['RA'], cat_gal['DEC']],
    weights=cat_gal['w_des'],
    field=[cat_gal['e1'], -cat_gal['e2']],
    lmax=b_lmax,
    lmax_mask=b_lmax,
    spin=2,
    lonlat=True
)

f_psf = nmt.NmtFieldCatalog(
    positions=[cat_star['RA'], cat_star['DEC']],
    weights=np.ones_like(cat_star['RA']),
    field=[cat_star['E1_PSF_HSM'], -cat_star['E2_PSF_HSM']],
    lmax=b_lmax,
    lmax_mask=b_lmax,
    spin=2,
    lonlat=True
)

# %%
#Compute rho_0

wsp = nmt.NmtWorkspace.from_fields(f_psf, f_psf, b)

rho_cl = nmt.compute_coupled_cell(f_psf, f_psf)
rho_cl = wsp.decouple_cell(rho_cl)

# %%
#Compute tau_0
wsp = nmt.NmtWorkspace.from_fields(f_gal, f_psf, b)

tau_cl = nmt.compute_coupled_cell(f_gal, f_psf)
tau_cl = wsp.decouple_cell(tau_cl)

wsp = nmt.NmtWorkspace.from_fields(f_gal_corrected, f_psf, b)

tau_cl_corrected = nmt.compute_coupled_cell(f_gal_corrected, f_psf)
tau_cl_corrected = wsp.decouple_cell(tau_cl_corrected)

# %%
#Compute cell

wsp = nmt.NmtWorkspace.from_fields(f_gal, f_gal, b)

cell_cl = nmt.compute_coupled_cell(f_gal, f_gal)
cell_cl = wsp.decouple_cell(cell_cl)

wsp = nmt.NmtWorkspace.from_fields(f_gal_corrected, f_gal_corrected, b)
cell_cl_corrected = nmt.compute_coupled_cell(f_gal_corrected, f_gal_corrected)
cell_cl_corrected = wsp.decouple_cell(cell_cl_corrected)

# %%
# Compute the covariance for rho_0 and tau_0 with iNKA
def project_cat_map(cat, e1_col, e2_col, nside=1024, w_col=None):
    ra = cat['RA']
    dec = cat['DEC']
    e1 = cat[e1_col]
    e2 = cat[e2_col]

    if w_col is None:
        weights = np.ones_like(ra)
    else:
        weights = cat[w_col]
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    pix_indices = hp.ang2pix(nside, theta, phi)
    unique_pix, idx, idx_rep = np.unique(pix_indices, return_index=True, return_inverse=True)
    count_map = np.zeros(hp.nside2npix(nside))
    count_map[unique_pix] = np.bincount(idx_rep, weights=weights)

    e1_map = np.zeros(hp.nside2npix(nside))
    e2_map = np.zeros(hp.nside2npix(nside))
    e1_map[unique_pix] = np.bincount(idx_rep, weights=weights * e1) / count_map[unique_pix]
    e2_map[unique_pix] = np.bincount(idx_rep, weights=weights * e2) / count_map[unique_pix]
    return count_map, e1_map, e2_map

def get_field_catalog(cat, e1_col, e2_col, nside, w_col=None):
    count_map, e1_map, e2_map = project_cat_map(cat, e1_col, e2_col, nside=nside, w_col=w_col)

    field = nmt.NmtField(
        mask=count_map,
        maps=[e1_map, -e2_map],
        lmax=b_lmax,
        lmax_mask=b_lmax,
    )
    return field

def get_workspace(field1, field2):
    wsp = nmt.NmtWorkspace.from_fields(field1, field2, b)
    return wsp

def get_cell(field1, field2, wsp, unbin=False, coupled=False):
    cl_coupled = nmt.compute_coupled_cell(field1, field2)
    if coupled:
        return cl_coupled
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    if unbin:
        cl_decoupled = b.unbin_cell(cl_decoupled)
        lowest_ell = b.get_ell_list(0)[0]
        for i in range(4):
            cl_decoupled[i, :lowest_ell] = cl_decoupled[i, lowest_ell]
    return cl_decoupled

def compute_iNKA_covariance(field1, field2, coupled=False):
        
    cw = nmt.NmtCovarianceWorkspace.from_fields(field1, field2, field1, field2)

    wsp = get_workspace(field1, field2)
    if coupled:
        cl_a1b1 = get_cell(field1, field1, get_workspace(field1, field1), unbin=False, coupled=True) / np.mean(field1.mask**2)
        cl_a1b2 = get_cell(field1, field2, wsp, unbin=False, coupled=True) / np.mean(field1.mask * field2.mask)
        cl_a2b2 = get_cell(field2, field2, get_workspace(field2, field2), unbin=False, coupled=True) / np.mean(field2.mask**2)
    else:
        cl_a1b1 = get_cell(field1, field1, get_workspace(field1, field1), unbin=True)
        cl_a1b2 = get_cell(field1, field2, wsp, unbin=True)
        cl_a2b2 = get_cell(field2, field2, get_workspace(field2, field2), unbin=True)

    cov = nmt.gaussian_covariance(cw, 2, 2, 2, 2,
                                 cl_a1b1,
                                 cl_a1b2,
                                 cl_a1b2,
                                 cl_a2b2,
                                 wsp, wb=wsp).reshape([n_ells, 4, n_ells, 4])

    return cov
    

# %%
field_psf_map = get_field_catalog(cat_star, 'E1_PSF_HSM', 'E2_PSF_HSM', nside=1024)
field_gal_map = get_field_catalog(cat_gal, 'e1', 'e2', nside=1024, w_col='w_des')
field_gal_corrected_map = get_field_catalog(cat_gal, 'e1_leak_corrected', 'e2_leak_corrected', nside=1024, w_col='w_des')

# %%
cov_rho_0 = compute_iNKA_covariance(field_psf_map, field_psf_map, coupled=True)
print("Computed cov_rho_0")
cov_tau_0 = compute_iNKA_covariance(field_gal_map, field_psf_map)
print("Computed cov_tau_0")
cov_tau_0_corrected = compute_iNKA_covariance(field_gal_corrected_map, field_psf_map)
print("Computed cov_tau_0_corrected")

# %%
#Correct the diagonal of cov_rho
for i in range(4):
    for j in range(4):
        for k in range(32):
            cov_rho_0[k, i, k, j] = np.abs(cov_rho_0[k, i, k, j])
# %%
# Get covariance of rho_0 and tau_0
n_sims = 350

rho_0_cls = np.array([]).reshape((0, 32*4))
tau_0_cls = np.array([]).reshape((0, 32*4))
for i in range(n_sims):
    index_sim = str(i+1).zfill(5)
    rho_0 = np.load(glass_root_dir + f"rho_cl_glass_mock_{index_sim}_4096.npy")[1:].reshape((32*4,))
    tau_0 = np.load(glass_root_dir + f"tau_cl_glass_mock_{index_sim}_4096.npy")[1:].reshape((32*4,))
    rho_0_cls = np.vstack((rho_0_cls, rho_0))
    tau_0_cls = np.vstack((tau_0_cls, tau_0))

cov_rho_0_sim = np.cov(rho_0_cls.T)
cov_tau_0_sim = np.cov(tau_0_cls.T)

# %%
def cov_to_corr(cov):
    diag = np.sqrt(np.diag(cov))
    corr = cov / np.outer(diag, diag)
    return corr

# %%
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 3))
plt.subplots_adjust(wspace=0.3)

im0 = ax0.imshow(cov_to_corr(cov_tau_0[:, 0,:,0]), vmin=-1, vmax=1, cmap='coolwarm')
ax0.set_title(r"Covariance $\tau_0$ iNKA")
divider = make_axes_locatable(ax0)
cax0 = divider.append_axes("right", size="5%", pad=0.1)
cbar0 = fig.colorbar(im0, cax=cax0)

im1 = ax1.imshow(cov_to_corr(cov_tau_0_sim[:32,:32]), vmin=-1, vmax=1, cmap='coolwarm')
ax1.set_title(r"Covariance $\tau_0$")
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("right", size="5%", pad=0.1)
cbar1 = fig.colorbar(im1, cax=cax1)

im2 = ax2.imshow(cov_to_corr(cov_rho_0[:, 0,:,0]), vmin=-1, vmax=1, cmap='coolwarm')
ax2.set_title(r"Covariance $\rho_0$ iNKA")
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("right", size="5%", pad=0.1)
cbar2 = fig.colorbar(im2, cax=cax2)

plt.savefig("cov_rho0_tau0.png", dpi=300)
plt.show()

# %%
plt.figure()

plt.plot(ell_eff, np.sqrt(np.diag(cov_tau_0[:, 0, :, 0])), label="iNKA")
plt.plot(ell_eff, np.sqrt(np.diag(cov_tau_0_sim[:32, :32])), label="Simulations")
plt.plot(ell_eff, np.sqrt(np.diag(cov_tau_0_corrected[:, 0, :, 0])), label="iNKA corrected")

plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
plt.xlabel(r"$\ell$")
plt.yscale('log')
plt.ylabel(r"$\sigma(C_\ell^{\tau_0})$")
plt.legend()

plt.savefig("./plots/sigma_tau0.png", dpi=300)
plt.show()

# %%
cov_rho_0_ee = cov_rho_0[:, 0, :, 0]
cov_tau_0_ee = cov_tau_0[:, 0, :, 0]
cov_rho_0_bb = cov_rho_0[:, 3, :, 3]
cov_tau_0_bb = cov_tau_0[:, 3, :, 3]

# %%
plt.figure()

plt.errorbar(ell_eff, ell_eff*rho_cl[0], yerr=ell_eff*np.sqrt(np.abs(cov_rho_0_ee.diagonal())), label=r"$\rho_0$ EE", fmt='o', capsize=2)
plt.errorbar(ell_eff, ell_eff*rho_cl[3], yerr=ell_eff*np.sqrt(np.abs(cov_rho_0_bb.diagonal())), label=r"$\rho_0$ BB", fmt='o', capsize=2)

plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
#plt.xlim(100, 2048)
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\ell C^{\rho_0}_\ell$")
plt.legend()
plt.savefig("rho_0_cl.png", dpi=300)

plt.show()
# %%
# Paper plot tau_0
plt.figure(figsize=(10, 6))

offset = 0.15

ell_widths = np.diff(ell_eff)
ell_widths = np.append(ell_widths, ell_widths[-1])

list_offset = []

for i in range(4):
    # Better jittering: symmetric around original ell values
    jitter_fraction = (i - (4 - 1) / 2) * offset
    jittered_ell = ell_eff + jitter_fraction * ell_widths
    list_offset.append(jittered_ell)


plt.errorbar(list_offset[0], ell_eff*tau_cl[0], yerr=ell_eff*np.sqrt(cov_tau_0_ee.diagonal()), label=r"$\tau_0$ EE", fmt='o', capsize=2)
plt.errorbar(list_offset[1], ell_eff*tau_cl_corrected[0], yerr=ell_eff*np.sqrt(cov_tau_0_ee.diagonal()), label=r"$\tau_0$ corrected EE", fmt='o', capsize=2)
plt.errorbar(list_offset[2], ell_eff*tau_cl[3], yerr=ell_eff*np.sqrt(cov_tau_0_bb.diagonal()), label=r"$\tau_0$ BB", fmt='o', capsize=2)
plt.errorbar(list_offset[3], ell_eff*tau_cl_corrected[3], yerr=ell_eff*np.sqrt(cov_tau_0_bb.diagonal()), label=r"$\tau_0$ corrected BB", fmt='o', capsize=2)

plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)

plt.axhline(0, c='k', ls='--', alpha=0.7)

plt.tick_params(axis='both', which='major', labelsize=16)

plt.gca().yaxis.get_offset_text().set_visible(False)


plt.xlabel(r"Multipole $\ell$", fontsize=20)
plt.ylabel(r"$\ell C^{\tau_0}_\ell \times 10^{-7}$", fontsize=20)
plt.xlim(1, 2048)
plt.ylim(-5e-7, 5e-7)
plt.legend(fontsize=16)
plt.savefig("./plots/tau_0_cl.png", dpi=300)
#Save pdf
plt.savefig("./plots/tau_0_cl.pdf")

plt.show()

# %%
tau_cl_samples = np.random.multivariate_normal(
    mean=tau_cl[0],
    cov=cov_tau_0_ee,
    size=10000
)
tau_cl_corrected_samples = np.random.multivariate_normal(
    mean=tau_cl_corrected[0],
    cov=cov_tau_0_ee,
    size=10000
)

alpha_ell_samples = tau_cl_samples / rho_cl[0]
alpha_ell_corrected_samples = tau_cl_corrected_samples / rho_cl[0]
alpha_ell_mean = np.mean(alpha_ell_samples, axis=0)
alpha_ell_std = np.std(alpha_ell_samples, axis=0)
alpha_ell_corrected_mean = np.mean(alpha_ell_corrected_samples, axis=0)
alpha_ell_corrected_std = np.std(alpha_ell_corrected_samples, axis=0)

# %%
plt.figure()

offset = 0.3

ell_widths = np.diff(ell_eff)
ell_widths = np.append(ell_widths, ell_widths[-1])  # repeat last width

list_offset = []
for i in range(2):
    jitter_fraction = (i - (2 - 1) / 2) * offset
    jiterred_ell = ell_eff + jitter_fraction * ell_widths
    list_offset.append(jiterred_ell)

plt.errorbar(list_offset[0], tau_cl[0]/rho_cl[0], yerr=alpha_ell_std, label="Without object-wise leakage correction", fmt='o', capsize=2)
plt.errorbar(list_offset[1], tau_cl_corrected[0]/rho_cl[0], yerr=alpha_ell_corrected_std, label="With object-wise leakage correction", fmt='o', capsize=2)

plt.xscale('squareroot')

plt.axhline(0., color='black', linestyle='--', alpha=0.6)
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
plt.xlabel(r"$\ell$", fontsize=12)
plt.ylabel(r"$\alpha_\ell$", fontsize=12)
plt.ylim(-0.1, 0.1)
plt.legend(fontsize=10)
plt.savefig("./plots/alpha_ell.png", dpi=300)
# Save PDF
plt.savefig("./plots/alpha_ell.pdf")
plt.show()

# %%
plt.figure()

plt.plot(ell_eff, ell_eff*cell_cl[0], label=r"$C^{EE}_\ell$ uncorrected")
plt.plot(ell_eff, ell_eff*tau_cl[0]**2/rho_cl[0], label="Leakage bias uncorrected")
plt.plot(ell_eff, ell_eff*cell_cl_corrected[0], label=r"$C^{EE}_\ell$ corrected")
plt.plot(ell_eff, ell_eff*tau_cl_corrected[0]**2/rho_cl[0], label="Leakage bias corrected")

plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\ell C_\ell$")
plt.legend()
plt.savefig("ee_cl_leakage_bias.png", dpi=300)
plt.show()

# %%
plt.figure()

plt.plot(ell_eff, ell_eff*cell_cl[0], label=r"$C^{EE}_\ell$ uncorrected")
#plt.plot(ell_eff, ell_eff*tau_cl[0]**2/rho_cl[0], label="Leakage bias uncorrected")
plt.plot(ell_eff, ell_eff*cell_cl_corrected[0], label=r"$C^{EE}_\ell$ corrected")
#plt.plot(ell_eff, ell_eff*tau_cl_corrected[0]**2/rho_cl[0], label="Leakage bias corrected")

plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\ell C_\ell$")
plt.legend()
plt.savefig("ee_cl.png", dpi=300)
plt.show()

# %%
plt.figure()

plt.plot(ell_eff, ell_eff*cell_cl[3], label=r"$C^{BB}_\ell$ uncorrected")
#plt.plot(ell_eff, ell_eff*tau_cl[0]**2/rho_cl[0], label="Leakage bias uncorrected")
plt.plot(ell_eff, ell_eff*cell_cl_corrected[3], label=r"$C^{BB}_\ell$ corrected")
#plt.plot(ell_eff, ell_eff*tau_cl_corrected[0]**2/rho_cl[0], label="Leakage bias corrected")

plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\ell C_\ell$")
plt.legend()
plt.savefig("bb_cl.png", dpi=300)
plt.show()

# %%
plt.figure()

plt.plot(ell_eff, cell_cl_corrected[0]/cell_cl[0] - 1, label=r"Ratio corrected/uncorrected EE")
plt.plot(ell_eff, cell_cl_corrected[3]/cell_cl[3] - 1, label=r"Ratio corrected/uncorrected BB")


plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\Delta C_\ell / C_\ell$")
plt.legend()
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
plt.savefig("ratio_ee_cl.png", dpi=300)
plt.show()


# %%
plt.figure()

plt.plot(ell_eff, ell_eff*cell_cl_corrected[0], label=r"$C^{EE}_\ell$")
plt.plot(ell_eff, ell_eff*tau_cl_corrected[0]**2/rho_cl[0], label="Leakage bias")

plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\ell C_\ell$")
plt.legend()
plt.show()

# %%
leakage_bias_samples = tau_cl_samples**2 / rho_cl[0]
leakage_bias_corrected_samples = tau_cl_corrected_samples**2 / rho_cl[0]

leakage_bias_mean = np.mean(leakage_bias_samples, axis=0)
leakage_bias_std = np.std(leakage_bias_samples, axis=0)
leakage_bias_corrected_mean = np.mean(leakage_bias_corrected_samples, axis=0)
leakage_bias_corrected_std = np.std(leakage_bias_corrected_samples, axis=0)

# %%
plt.figure()

leakage_bias = tau_cl[0]**2/rho_cl[0]
leakage_bias_corrected = tau_cl_corrected[0]**2/rho_cl[0]

plt.errorbar(list_offset[0], leakage_bias/cell_cl[0], yerr=leakage_bias_std/cell_cl[0], label=r"Without object-wise leakage correction", fmt='o', markersize=3, capsize=2)
plt.errorbar(list_offset[1], leakage_bias_corrected/cell_cl_corrected[0], yerr=leakage_bias_corrected_std/cell_cl_corrected[0], label="With object-wise leakage correction", fmt='o', markersize=3, capsize=2)

threshold = 0.05
plt.fill_between(np.arange(7, 2050), -threshold, threshold, color='gray', alpha=0.3, label=r'5\% threshold')
plt.xscale('squareroot')

plt.axhline(threshold, color='black', linestyle='--', alpha=0.6)
plt.axhline(-threshold, color='black', linestyle='--', alpha=0.6)
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
plt.xlim(ell_eff[0], ell_eff[-1])
plt.xlabel(r"Multipole $\ell$", fontsize=16)
plt.xlim(7, 2050)
plt.ylabel(r"$C_\ell^{\rm sys} / C_\ell$", fontsize=16)
plt.ylim(-0.052, 0.152)
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
plt.legend(loc="upper center", fontsize=12)
plt.savefig("./plots/leakage_bias_fraction_ee.png", dpi=300)
#Save PDF
plt.savefig("./plots/leakage_bias_fraction_ee.pdf")
plt.show()

# %%
tau_samples_bb = np.random.multivariate_normal(
    mean=tau_cl[3],
    cov=cov_tau_0_bb,
    size=10000
)
tau_corrected_samples_bb = np.random.multivariate_normal(
    mean=tau_cl_corrected[3],
    cov=cov_tau_0_bb,
    size=10000
)
leakage_bias_samples_bb = tau_samples_bb**2 / rho_cl[3]
leakage_bias_corrected_samples_bb = tau_corrected_samples_bb**2 / rho_cl[3]
leakage_bias_mean_bb = np.mean(leakage_bias_samples_bb, axis=0)
leakage_bias_std_bb = np.std(leakage_bias_samples_bb, axis=0)
leakage_bias_corrected_mean_bb = np.mean(leakage_bias_corrected_samples_bb, axis=0)
leakage_bias_corrected_std_bb = np.std(leakage_bias_corrected_samples_bb, axis=0)

# %%
plt.figure()

leakage_bias = tau_cl[3]**2/rho_cl[3]
leakage_bias_corrected = tau_cl_corrected[3]**2/rho_cl[3]

plt.errorbar(list_offset[0], leakage_bias, yerr=leakage_bias_std_bb, label=r"Without object-wise leakage correction", fmt='o', markersize=3, capsize=2)
plt.errorbar(list_offset[1], leakage_bias_corrected, yerr=leakage_bias_corrected_std_bb, label="With object-wise leakage correction", fmt='o', markersize=3, capsize=2)

plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
plt.xlabel(r"$\ell$")
plt.ylabel(r"$C_\ell^{\rm sys, BB}$")
plt.legend()
plt.savefig("./plots/leakage_bias_bb.png", dpi=300)
plt.ylim(-0.02e-8, 0.02e-8)
plt.show()

# %%
plt.figure()

plt.plot(ell_eff, ell_eff*cell_cl[3], label=r"$C^{BB}_\ell$ uncorrected")
plt.plot(ell_eff, ell_eff*tau_cl[3]**2/rho_cl[3], label="Leakage bias uncorrected")
plt.plot(ell_eff, ell_eff*cell_cl_corrected[3], label=r"$C^{BB}_\ell$ corrected")
plt.plot(ell_eff, ell_eff*tau_cl_corrected[3]**2/rho_cl[3], label="Leakage bias corrected")

plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\ell C_\ell$")
plt.legend()
plt.savefig("bb_cl_leakage_bias.png", dpi=300)
plt.show()
# %%
plt.figure()

leakage_bias = tau_cl[3]**2/rho_cl[3]
leakage_bias_corrected = tau_cl_corrected[3]**2/rho_cl[3]

plt.plot(ell_eff, leakage_bias/cell_cl[3], label=r"Uncorrected")
plt.plot(ell_eff, leakage_bias_corrected/cell_cl_corrected[3], label="Corrected")

plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
plt.xlabel(r"$\ell$")
plt.ylabel(r"$C_\ell^{\rm sys, BB} / C_\ell^\mathrm{BB}$")
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
plt.legend()
plt.savefig("leakage_bias_fraction_bb.png", dpi=300)
plt.show()

# %%
#Plot signal with errorbars
pseudo_cl = fits.getdata("/home/guerrini/sp_validation/notebooks/cosmo_val/output/pseudo_cl_SP_v1.4.5_leak_corr.fits")
cov_cl = fits.open("/home/guerrini/sp_validation/notebooks/cosmo_val/output/pseudo_cl_cov_SP_v1.4.5_leak_corr.fits")

plt.figure()

plt.errorbar(ell_eff, ell_eff*pseudo_cl['EE'], yerr=ell_eff*np.sqrt(np.diag(cov_cl["COVAR_EE_EE"].data)), label=r"$C^{EE}_\ell$ SP_v1.4.5 corrected", fmt='o', markersize=3, capsize=2)

plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\ell C_\ell$")
plt.legend()
plt.title("EE Pseudo-Cl")
plt.savefig("ee_cl_with_errors.png", dpi=300)

plt.show()
# %%
plt.figure()

plt.errorbar(ell_eff, ell_eff*pseudo_cl['BB'], yerr=ell_eff*np.sqrt(np.diag(cov_cl["COVAR_BB_BB"].data)), label=r"$C^{BB}_\ell$ SP_v1.4.5 corrected", fmt='o', markersize=3, capsize=2)
plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\ell C_\ell$")
plt.legend()
plt.axhline(0, color='black', linestyle='--')
plt.title("BB Pseudo-Cl")
plt.savefig("bb_cl_with_errors.png", dpi=300)

plt.show()
# %%
