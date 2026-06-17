"""Converted from cosmo_val/one_covariance.ipynb (personal exploratory analysis).

Builds healpix masks and noise maps from ShapePipe catalogues, computes
Gaussian C_ell covariances with NaMaster, and compares them against
OneCovariance and cosmo_val outputs. Hardcoded paths point to the original
author's (guerrini) machine and are preserved as-is.
"""

# %%
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

import matplotlib.scale as mscale
from sp_validation.rho_tau import SquareRootScale
from sp_validation.basic import corr_from_cov, cov_from_one_covariance

mscale.register_scale(SquareRootScale)

# %%
redshift_distribution = np.loadtxt('/n17data/mkilbing/astro/data/CFIS/v1.0/nz/dndz_SP_A.txt')

# %%
z, dndz = redshift_distribution[:, 0], redshift_distribution[:, 1]

# %%
plt.figure()

plt.plot(z, dndz)

plt.show()

# %%
# Define the header
header = "# # binstart, density\n" \
         "# dndz_SP_A"

np.savetxt("/home/guerrini/OneCovariance/input/redshift_distribution/dndz_SP_A.asc", redshift_distribution, header=header, fmt="%.18e", comments='')

# %%
#Get healpix mask
path_cat = "/n17data/guinot/CFIS_3500_cat/catalogues_SPv1_v1.4.5/shapepipe_SPv1.fits"
cat_gal = fits.getdata(path_cat)

# %%
import healpy as hp

nside = 1024

ra = cat_gal['ra']
dec = cat_gal['dec']
w = cat_gal['w']

theta = (90. - dec) * np.pi / 180.
phi = ra * np.pi / 180.
pix = hp.ang2pix(nside, theta, phi)

unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)
n_gal = np.zeros(hp.nside2npix(nside))
n_gal[unique_pix] = np.bincount(idx_rep, weights=w)

# %%
hp.mollview(n_gal)

# %%
mask = n_gal > 0

# %%
area = np.sum(mask) * hp.nside2pixarea(nside, degrees=True)

print(f"Area: {area} deg^2")

# %%
n_eff = 1/(area*3600)*(np.sum(cat_gal['w']))**2/np.sum(cat_gal['w']**2)
n_eff

# %%
shape_noise = 0.5*(np.sum(cat_gal['w']**2*cat_gal['g1']**2)/np.sum(cat_gal['w']**2) + np.sum(cat_gal['w']**2*cat_gal['g2']**2)/np.sum(cat_gal['w']**2))
np.sqrt(shape_noise)

# %%
cov_one_cov = np.genfromtxt("/home/guerrini/OneCovariance/output/covariance_list_3x2pt_pure_Cell.dat")
cov_cosmo_val = fits.open("./output/pseudo_cl_cov_SP_v1.4.5.A.fits")["COVAR_EE_EE"].data

# %%
cov_one_cov

# %%
hp.write_map("/home/guerrini/OneCovariance/input/mask/healpix_mask_SPv1.fits", mask, overwrite=True)

# %%
corr_cosmo_val = corr_from_cov(cov_cosmo_val)

# %%
start = np.power(8, 1/2)
end = np.power(2048, 1/2)
bins_ell = np.power(np.linspace(start, end, 33), 2)

bins_ell = 0.5 * (bins_ell[1:] + bins_ell[:-1])

# %%
plt.figure()

plt.imshow(corr_cosmo_val, cmap='seismic', vmin=-1, vmax=1)
plt.colorbar(label='Correlation Coefficient')

step = len(bins_ell) // 4
plt.xticks(ticks=np.arange(0, len(bins_ell), step), labels=np.round(bins_ell[::step], 1))
plt.yticks(ticks=np.arange(0, len(bins_ell), step), labels=np.round(bins_ell[::step], 1))

plt.title('Gaussian Covariance Matrix Cell')
plt.show()


# %%
cov_matrix_one_cov_gaussian = cov_from_one_covariance(cov_one_cov, gaussian=True)
cov_matrix_one_cov_pure = cov_from_one_covariance(cov_one_cov, gaussian=False)

corr_one_cov_gaussian = corr_from_cov(cov_matrix_one_cov_gaussian)
corr_one_cov_pure = corr_from_cov(cov_matrix_one_cov_pure)

# %%
plt.figure()

plt.subplot(121)

plt.imshow(corr_one_cov_gaussian, cmap='seismic', vmin=-1, vmax=1)
plt.colorbar(label='Correlation Coefficient')

step = len(bins_ell) // 4
plt.xticks(ticks=np.arange(0, len(bins_ell), step), labels=np.round(bins_ell[::step], 1))
plt.yticks(ticks=np.arange(0, len(bins_ell), step), labels=np.round(bins_ell[::step], 1))

plt.title('Gaussian (OneCov)')

plt.subplot(122)

plt.imshow(corr_one_cov_pure, cmap='seismic', vmin=-1, vmax=1)
plt.colorbar(label='Correlation Coefficient')

step = len(bins_ell) // 4
plt.xticks(ticks=np.arange(0, len(bins_ell), step), labels=np.round(bins_ell[::step], 1))
plt.yticks(ticks=np.arange(0, len(bins_ell), step), labels=np.round(bins_ell[::step], 1))

plt.title('G+NG (OneCov)')

plt.tight_layout()
plt.savefig("Plots/corr_onecovariance.png")
plt.show()

# %%
plt.figure()

plt.plot(bins_ell, np.diag(cov_matrix_one_cov_gaussian), label='Gaussian Covariance (OneCovariance)')
plt.plot(bins_ell, np.diag(cov_matrix_one_cov_pure), label='Pure Covariance (OneCovariance)')
plt.plot(bins_ell, np.diag(cov_cosmo_val), label='Cosmo Val Covariance')

plt.xlabel('Multipole (ell)')
plt.ylabel('Covariance')
plt.title('Diagonal of Covariance Matrices')

plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.grid()

plt.savefig("Plots/diag_cov_onecovariance.png")
plt.show()

# %%
#mask comparison
path_145 = "/n17data/mkilbing/astro/data/CFIS/v1.0/ShapePipe/v1.4.x/v1.4.5/unions_shapepipe_cut_struc_2024_v1.4.5.fits"
cat_145 = fits.getdata(path_145)

# %%
nside = 1024

ra = cat_145['ra']
dec = cat_145['dec']
w = cat_145['w_des']

theta = (90. - dec) * np.pi / 180.
phi = ra * np.pi / 180.
pix = hp.ang2pix(nside, theta, phi)

unique_pix, idx, idx_rep_145 = np.unique(pix, return_index=True, return_inverse=True)
n_gal_145 = np.zeros(hp.nside2npix(nside))
n_gal_145[unique_pix] = np.bincount(idx_rep_145, weights=w)

# %%
hp.mollview(n_gal_145)

# %%
n_gal_145[~mask] = 0

# %%
hp.mollview(n_gal_145)

# %%
rot_angle = np.random.rand(len(cat_145))*2*np.pi
e1_rot = cat_145['e1'] * np.cos(rot_angle) + cat_145['e2'] * np.sin(rot_angle)
e2_rot = -cat_145['e1'] * np.sin(rot_angle) + cat_145['e2'] * np.cos(rot_angle)

# %%
noise_map_e1 = np.zeros(hp.nside2npix(nside))
noise_map_e2 = np.zeros(hp.nside2npix(nside))

noise_map_e1[unique_pix] += np.bincount(idx_rep_145, weights=e1_rot*cat_145['w_des'])
noise_map_e2[unique_pix] += np.bincount(idx_rep_145, weights=e2_rot*cat_145['w_des'])
noise_map_e1[~mask] = 0
noise_map_e2[~mask] = 0
noise_map_e1[mask] /= n_gal[mask]
noise_map_e2[mask] /= n_gal[mask]

# %%
lmax = 2*nside
path_redshift_distr = "/n17data/mkilbing/astro/data/CFIS/v1.0/nz/dndz_SP_A.txt"
pw = hp.pixwin(nside, lmax=lmax)


# %%
import camb
from astropy.cosmology import Planck18

planck = Planck18

h = planck.H0.value/100
Om = planck.Om0
Ob = planck.Ob0
Oc = Om - Ob
ns = 0.965
As = 2.1e-9
m_nu = 0.06
w = -1

pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2, ns=ns, mnu=m_nu, w=w, As=As, WantTransfer=True, NonLinear=camb.model.NonLinear_both)
Onu = pars.omeganu
Oc = Om - Ob - Onu
pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2, ns=ns, mnu=m_nu, w=w, As=As, WantTransfer=True, NonLinear=camb.model.NonLinear_both)

z, dndz = np.loadtxt(path_redshift_distr, unpack=True)

#getthe expected cl's from CAMB
pars.min_l = 1
pars.set_for_lmax(lmax)
pars.SourceWindows = [
    camb.sources.SplinedSourceWindow(z=z, W=dndz, source_type='lensing')
]
theory_cls = camb.get_results(pars).get_source_cls_dict(lmax=lmax, raw_cl=True)

fiducial_cl = theory_cls["W1xW1"] * pw**2


# %%
import pymaster as nmt

f = nmt.NmtField(mask=mask, maps=[noise_map_e1, noise_map_e2], lmax=lmax)


b_lmax = lmax - 1

lmin = 8
lmax = 2048

start = np.power(lmin, 1/2)
end = np.power(lmax, 1/2)
bins_ell = np.power(np.linspace(start, end, 33), 2)

ells = np.arange(lmin, lmax+1)

bpws = np.digitize(ells.astype(float), bins_ell) - 1
bpws[0] = 0
bpws[-1] = 32-1

b = nmt.NmtBin(ells=ells, bpws=bpws, lmax=lmax)


wsp = nmt.NmtWorkspace.from_fields(f, f, b)

cl_noise = nmt.compute_coupled_cell(f, f)
cl_noise = wsp.decouple_cell(cl_noise)

# %%
fiducial_cl = np.array([fiducial_cl, 0.*fiducial_cl, 0.*fiducial_cl, 0.*fiducial_cl])+ np.mean(cl_noise, axis=1, keepdims=True)

# %%
cw = nmt.NmtCovarianceWorkspace.from_fields(f, f, f, f)

covar_22_22 = nmt.gaussian_covariance(cw, 2, 2, 2, 2,
                                        fiducial_cl,
                                        fiducial_cl,
                                        fiducial_cl,
                                        fiducial_cl,
                                        wsp, wb=wsp).reshape([32, 4, 32, 4])

# %%
plt.figure()

bins_ell_mid = 0.5 * (bins_ell[1:] + bins_ell[:-1])

cov_cosmo_val_145 = fits.open("./output/pseudo_cl_cov_SP_v1.4.5.fits")["COVAR_EE_EE"].data

plt.plot(bins_ell_mid, np.diag(cov_cosmo_val_145), label='SP_v1.4.5')
plt.plot(bins_ell_mid, np.diag(covar_22_22[:, 0, :, 0]), label="SP_v1.4.5 w mask SP_v1.4.5.A")

plt.yscale('log')
plt.xscale('log')
plt.legend()

plt.xlabel('Multipole (ell)')
plt.ylabel('Covariance')
plt.savefig("Plots/mask_effect_covariance.png")

plt.show()

# %%
plt.figure()

plt.plot(bins_ell_mid, np.diag(cov_cosmo_val_145), label='SP_v1.4.5')
plt.plot(bins_ell_mid, np.diag(cov_cosmo_val), label="SP_v1.4.5.A")

plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.xlabel('Multipole (ell)')
plt.ylabel('Covariance')
plt.savefig("Plots/covariance_axel_comparison.png")

plt.show()
