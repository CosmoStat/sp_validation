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
    "/n17data/UNIONS/WL/v1.4.x/v1.4.5/unions_shapepipe_cut_struc_2024_v1.4.5.fits"
)

cat_star = fits.getdata(
    "/n17data/UNIONS/WL/v1.4.x/unions_shapepipe_psf_2024_v1.4.a.fits"
)

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
plt.figure()

plt.plot(ell_eff, ell_eff*rho_cl[0], label=r"$\rho_0$ EE")
plt.plot(ell_eff, ell_eff*rho_cl[3], label=r"$\rho_0$ BB")

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

plt.plot(ell_eff, ell_eff*tau_cl[0], label=r"$\tau_0$ EE")
plt.plot(ell_eff, ell_eff*tau_cl_corrected[0], label=r"$\tau_0$ corrected EE")
plt.plot(ell_eff, ell_eff*tau_cl[3], label=r"$\tau_0$ BB")
plt.plot(ell_eff, ell_eff*tau_cl_corrected[3], label=r"$\tau_0$ corrected BB")

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

plt.plot(ell_eff, tau_cl[0]/rho_cl[0], label="Uncorrected")
plt.plot(ell_eff, tau_cl_corrected[0]/rho_cl[0], label="Corrected")

plt.xscale('squareroot')
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

plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
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
