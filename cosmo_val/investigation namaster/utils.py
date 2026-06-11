from IPython import get_ipython

ipython = get_ipython()

# enable autoreload for interactive sessions
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

import numpy as np
from astropy.io import fits
from astropy.cosmology import Planck18
import matplotlib.pyplot as plt
from matplotlib import scale as mscale
import seaborn as sns
from tqdm import tqdm
import healpy as hp
import pymaster as nmt
import camb

from sp_validation.utils_cosmo_val import SquareRootScale
from sp_validation.cosmo_val import CosmologyValidation
from sp_validation.cosmology import get_theo_c_ell, get_cosmo
from sp_validation.rho_tau import get_params_rho_tau

mscale.register_scale(SquareRootScale)

plt.style.use(
    '/home/guerrini/matplotlib_config/paper.mplstyle'
)
plt.rcParams['text.usetex'] = False

sns.set_palette("husl")

if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

def load_cat(path_cat, w_col='w_des'):
    cat_gal = fits.getdata(path_cat)
    if w_col is None:
        w = np.ones_like(cat_gal['RA'])
    else:
        w = cat_gal[w_col]
    return cat_gal['RA'], cat_gal['DEC'], cat_gal['e1'], cat_gal['e2'], w

def load_cat_des(path_cat):
    cat_gal = fits.getdata(path_cat)
    ra = cat_gal['RA']
    dec = cat_gal['DEC']
    e1 = cat_gal['E1']
    e2 = cat_gal['E2']
    weight = cat_gal['w']

    e1 = (e1 - np.average(e1, weights=weight)) / np.average(cat_gal['R11'], weights=weight)
    e2 = (e2 - np.average(e2, weights=weight)) / np.average(cat_gal['R22'], weights=weight)
    return ra, dec, e1, e2, weight

def load_redshift_distr(path_nz):
    z, dndz = np.loadtxt(path_nz, unpack=True)
    return z, dndz

def load_cl_measurement(path_cl):
    cl_data = fits.getdata(path_cl)
    cl_array = np.array([
        cl_data['EE'].data,
        cl_data['EB'].data,
        cl_data['EB'].data,
        cl_data['BB'].data
    ])
    return cl_array

def get_binning(lmin, lmax, num_bins):
    b_lmax = lmax - 1

    ells = np.arange(lmin, lmax+1)

    start = np.power(lmin, 1/2)
    end = np.power(lmax, 1/2)
    bins_ell = np.power(np.linspace(start, end, num_bins +1), 2)

    bpws = np.digitize(ells.astype(float), bins_ell) - 1
    bpws[0] = 0
    bpws[-1] = num_bins - 1

    b = nmt.NmtBin(ells=ells, bpws=bpws, lmax=b_lmax)

    return b

def get_theory(lmax, z, nz):
    ell = np.arange(1, lmax + 1)
    cl_theo = get_theo_c_ell(
        ell=ell,
        z=z,
        nz=nz,
        backend="camb",
        cosmo=get_cosmo()
    )
    return cl_theo

def get_n_gal_map(nside, ra, dec, w):

    theta = (90. - dec) * np.pi / 180.
    phi = ra * np.pi / 180.
    pix = hp.ang2pix(nside, theta, phi)

    unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)
    n_gal = np.zeros(hp.nside2npix(nside))
    n_gal[unique_pix] = np.bincount(idx_rep, weights=w)
    return n_gal, unique_pix, idx, idx_rep

def apply_random_rotations(e1, e2):
    np.random.seed()
    rot_angle = np.random.rand(len(e1))*2*np.pi
    e1_out = e1*np.cos(rot_angle) + e2*np.sin(rot_angle)
    e2_out = -e1*np.sin(rot_angle) + e2*np.cos(rot_angle)
    return e1_out, e2_out

def get_gaussian_real(nside, mask, n_gal, unique_pix, idx_rep, e1, e2, w):
    e1_rot, e2_rot = apply_random_rotations(e1, e2)

    noise_map_e1 = np.zeros(hp.nside2npix(nside))
    noise_map_e2 = np.zeros(hp.nside2npix(nside))

    noise_map_e1[unique_pix] += np.bincount(idx_rep, weights=e1_rot * w)
    noise_map_e2[unique_pix] += np.bincount(idx_rep, weights=e2_rot * w)
    noise_map_e1[mask] /= n_gal[mask]
    noise_map_e2[mask] /= n_gal[mask]

    noise_map = noise_map_e1 + 1j * noise_map_e2
    return noise_map
    
def get_sample(noise_map, mask, lmax, b, wsp=None):

    f = nmt.NmtField(mask=mask, maps=[noise_map.real, noise_map.imag], lmax=lmax)
    if wsp is None:
        wsp  = nmt.NmtWorkspace.from_fields(f, f, b)

    cl_noise_coupled = nmt.compute_coupled_cell(f, f)
    cl_noise = wsp.decouple_cell(cl_noise_coupled)

    return cl_noise, f, wsp, cl_noise_coupled

def get_field_and_workspace_cat(ra, dec, e1, e2, lmax, b, w=None):
    if w is None:
        w = np.ones_like(e1)

    ra[ra < 0] += 360

    f_all = nmt.NmtFieldCatalog(
        positions=[ra, dec],
        weights = w,
        field=[e1, -e2],
        lmax = lmax,
        lmax_mask = lmax,
        spin=2,
        lonlat=True
    )
    
    wsp = nmt.NmtWorkspace.from_fields(f_all, f_all, b)

    return f_all, wsp

def get_field_and_workspace_map(mask, lmax, b):
    nside = hp.npix2nside(len(mask))
    f = nmt.NmtField(mask=mask, maps=[np.zeros(hp.nside2npix(nside)), np.zeros(hp.nside2npix(nside))], lmax=lmax)
    wsp  = nmt.NmtWorkspace.from_fields(f, f, b)

    return f, wsp

def get_covariance_workspace(f):
    cw = nmt.NmtCovarianceWorkspace.from_fields(f, f, f, f)
    return cw

def get_covariance(cl_array, cw, wsp, num_bins):
    covar_22_22 = nmt.gaussian_covariance(cw, 2, 2, 2, 2,
                                          cl_array,
                                          cl_array,
                                          cl_array,
                                          cl_array,
                                          wsp, wb=wsp).reshape((num_bins, 4, num_bins, 4))
    
    return covar_22_22

def get_covariance_from_glass(root, n_sim, num_bins, lmax, type):
    if type == 'cat':
        name_file = 'cl_glass_mock_'
        shape_output = num_bins
    elif type == 'cat_coupled':
        name_file = 'cl_coupled_glass_mock_'
        shape_output = lmax
    elif type == 'map':
        name_file = 'cl_map_glass_mock_'
        shape_output = num_bins
    elif type == 'map_coupled':
        name_file = 'cl_coupled_map_glass_mock_'
        shape_output = lmax
    else:
        raise ValueError("Type must be one of 'cat', 'cat_coupled', 'map', 'map_coupled'")
    cls = np.array([]).reshape((0, shape_output))
    for i in tqdm(range(1, n_sim+1)):
        path_cl = f'{root}/{name_file}{i:05d}_4096.npy'
        cl_sim = np.load(path_cl)
        cls = np.vstack((cls, cl_sim[1]))
    return np.cov(cls.T), cls