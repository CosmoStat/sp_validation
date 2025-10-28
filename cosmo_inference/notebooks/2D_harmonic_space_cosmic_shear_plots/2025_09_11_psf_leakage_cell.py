# %%
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
plt.rcParams['text.usetex'] = False

sns.set_palette("husl")

if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

# %%
#Import galaxy and star catalogs
cat_gal = fits.getdata(
    "/n17data/UNIONS/WL/v1.4.x/v1.4.6/unions_shapepipe_cut_struc_2024_v1.4.6.fits"
)

cat_star = fits.getdata(
    "/n17data/UNIONS/WL/v1.4.x/unions_shapepipe_psf_2024_v1.4.a.fits"
)

glass_root_dir = "/n09data/guerrini/glass_mock_v1.4.6/results/"

# %%
#Define namaster binning
lmin = 8
lmax = 2048
b_lmax = lmax - 1

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

cov_rho_0 = np.cov(rho_0_cls.T)
cov_tau_0 = np.cov(tau_0_cls.T)

# %%
def cov_to_corr(cov):
    diag = np.sqrt(np.diag(cov))
    corr = cov / np.outer(diag, diag)
    return corr

# %%
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 3))
plt.subplots_adjust(wspace=0.3)

im0 = ax0.imshow(cov_to_corr(cov_rho_0), vmin=-1, vmax=1, cmap='coolwarm')
ax0.set_title(r"Covariance $\rho_0$")
divider = make_axes_locatable(ax0)
cax0 = divider.append_axes("right", size="5%", pad=0.1)
cbar0 = fig.colorbar(im0, cax=cax0)

im1 = ax1.imshow(cov_to_corr(cov_tau_0), vmin=-1, vmax=1, cmap='coolwarm')
ax1.set_title(r"Covariance $\tau_0$")
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("right", size="5%", pad=0.1)
cbar1 = fig.colorbar(im1, cax=cax1)

plt.savefig("cov_rho0_tau0.png", dpi=300)
plt.show()

# %%
cov_rho_0_ee = cov_rho_0[0:32, 0:32]
cov_tau_0_ee = cov_tau_0[0:32, 0:32]
cov_rho_0_bb = cov_rho_0[96:128, 96:128]
cov_tau_0_bb = cov_tau_0[96:128, 96:128]

# %%
plt.figure()

plt.errorbar(ell_eff, ell_eff*rho_cl[0], yerr=ell_eff*np.sqrt(cov_rho_0_ee.diagonal()), label=r"$\rho_0$ EE", fmt='o', capsize=2)
plt.errorbar(ell_eff, ell_eff*rho_cl[3], yerr=ell_eff*np.sqrt(cov_rho_0_bb.diagonal()), label=r"$\rho_0$ BB", fmt='o', capsize=2)

plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\ell C^{\rho_0}_\ell$")
plt.legend()
plt.savefig("rho_0_cl.png", dpi=300)

plt.show()
# %%
plt.figure()

offset = 2

list_offset = [ell_eff + idx * offset for idx in range(4)]

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
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\ell C^{\tau_0}_\ell$")
plt.legend()
plt.savefig("tau_0_cl.png", dpi=300)

plt.show()

# %%
plt.figure()

plt.plot(ell_eff, tau_cl[0]/rho_cl[0], label="Without object-wise leakage correction")
plt.plot(ell_eff, tau_cl_corrected[0]/rho_cl[0], label="With object-wise leakage correction")

plt.xscale('squareroot')

plt.axhline(0., color='black', linestyle='--', alpha=0.6)
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\alpha_\ell$")
plt.ylim(-0.1, 0.1)
plt.legend()
plt.savefig("alpha_ell.png", dpi=300)
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
plt.figure()

leakage_bias = tau_cl[0]**2/rho_cl[0]
leakage_bias_corrected = tau_cl_corrected[0]**2/rho_cl[0]

plt.plot(ell_eff, leakage_bias/cell_cl[0], label=r"Uncorrected")
plt.plot(ell_eff, leakage_bias_corrected/cell_cl_corrected[0], label="Corrected")

threshold = 0.05
plt.fill_between(ell_eff, -threshold, threshold, color='gray', alpha=0.3, label='5% threshold')
plt.xscale('squareroot')

plt.axhline(threshold, color='black', linestyle='--', alpha=0.6)
plt.axhline(-threshold, color='black', linestyle='--', alpha=0.6)
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
plt.xlim(ell_eff[0], ell_eff[-1])
plt.xlabel(r"$\ell$")
plt.ylabel(r"$C_\ell^{\rm sys} / C_\ell$")
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
plt.legend()
plt.savefig("leakage_bias_fraction_ee.png", dpi=300)
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
