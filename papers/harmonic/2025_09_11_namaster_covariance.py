# %%
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

from sp_validation.rho_tau import SquareRootScale
from sp_validation.cosmo_val import CosmologyValidation
from sp_validation.rho_tau import get_params_rho_tau

mscale.register_scale(SquareRootScale)

plt.style.use(
    '/home/guerrini/matplotlib_config/paper.mplstyle'
)
plt.rcParams['text.usetex'] = False

sns.set_palette("husl")

if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

# %%
cat_gal = fits.getdata(
    "/n17data/UNIONS/WL/v1.4.x/v1.4.5/unions_shapepipe_cut_struc_2024_v1.4.5.fits"
)
# %%
nside = 1024
nside_mask = 4096

lmax = 2*nside
path_redshift_distr = "/n23data1/n06data/lgoh/scratch/UNIONS/cosmo_inference/data/SP_v1.4.5_A/nz_SP_v1.4.5_A.txt"
pw = hp.pixwin(nside, lmax=lmax)
# %%
planck = Planck18
h = planck.H0.value/100
Om = planck.Om0
Ob = planck.Ob0
Oc = Om - Ob
ns = 0.965
As = 2.1e-9
m_nu = 0.06
w = -1
print(h, Om, Ob)

pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2, ns=ns, mnu=m_nu, w=w, As=As, WantTransfer=True, NonLinear=camb.model.NonLinear_both)
Onu = pars.omeganu
Oc = Om - Ob - Onu
pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2, ns=ns, mnu=m_nu, w=w, As=As, WantTransfer=True, NonLinear=camb.model.NonLinear_both)
print(camb.get_results(pars).get_sigma8())
z, dndz = np.loadtxt(path_redshift_distr, unpack=True)

#getthe expected cl's from CAMB
pars.min_l = 1
pars.set_for_lmax(lmax)
pars.SourceWindows = [
    camb.sources.SplinedSourceWindow(z=z, W=dndz, source_type='lensing')
]
theory_cls = camb.get_results(pars).get_source_cls_dict(lmax=lmax, raw_cl=True)

fiducial_cl = theory_cls["W1xW1"]
# %%
lmin = 8
lmax = 2*nside
b_lmax = lmax - 1

ells = np.arange(lmin, lmax+1)
start = np.power(lmin, 1/2)
stop = np.power(lmax, 1/2)
bins_ell = np.power(np.linspace(start, stop, 32+1), 2)

bpws = np.digitize(ells.astype(float), bins_ell) -1
bpws[0] = 0
bpws[-1] = 31

b = nmt.NmtBin(ells=ells, bpws=bpws, lmax=b_lmax)
# %%
cv = CosmologyValidation(
    versions=["SP_v1.4.5_leak_corr"],
    data_base_dir="/n17data/mkilbing/astro/data/",
    catalog_config="/home/guerrini/sp_validation/cosmo_val/cat_config.yaml"
)
# %%
params = get_params_rho_tau(cv.cc["SP_v1.4.5_leak_corr"], survey="SP_v1.4.5_leak_corr")

# %%
print("Collecting maps...")
n_gal, unique_pix, idx, idx_rep = cv.get_n_gal_map(params, nside, cat_gal)
mask = n_gal > 0

print("Computing noise...")
cl_noise, f, wsp = cv.get_sample(params, nside, b_lmax, b, cat_gal, n_gal, mask, unique_pix, idx_rep)

fiducial_cl = np.array([fiducial_cl, 0.*fiducial_cl, 0.*fiducial_cl, 0.*fiducial_cl])+ np.mean(cl_noise, axis=1, keepdims=True)
# %%
mask_8192 = hp.read_map("/n09data/guerrini/glass_mock/mask_v1.4.5_nside8192.fits")

# %%
cw = nmt.NmtCovarianceWorkspace.from_fields(f, f, f, f)
# %%
covar_22_22 = nmt.gaussian_covariance(cw, 2, 2, 2, 2,
                                    fiducial_cl,
                                    fiducial_cl,
                                    fiducial_cl,
                                    fiducial_cl,
                                    wsp, wb=wsp).reshape([32, 4, 32, 4])
# %%
def get_cov_from_one_cov(cov_one_cov, gaussian=True):
    """
    Returns a numpy array with the covariance matrix from the OneCovariance output.
    """

    n_bins = np.sqrt(cov_one_cov.shape[0]).astype(int)
    cov = np.zeros((n_bins, n_bins))

    index_value = 10 if gaussian else 9
    for i in range(n_bins):
        for j in range(n_bins):
            cov[i, j] = cov_one_cov[i * n_bins + j, index_value]
    
    return cov

# %%
covar_EE_EE = covar_22_22[:, 0, :, 0]
cov_one_cov = np.genfromtxt("/home/guerrini/OneCovariance/output/covariance_list_3x2pt_pure_Cell.dat")
ell = b.get_effective_ells()

plt.figure()

plt.plot(ell, np.diag(covar_EE_EE), label="NaMaster")
plt.plot(ell, np.diag(get_cov_from_one_cov(cov_one_cov, gaussian=True)), label="OneCovariance")
plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
plt.yscale('log')
plt.legend()

plt.show()
# %%
plt.figure()

plt.plot(ell, np.sqrt(np.diag(covar_EE_EE)/np.diag(get_cov_from_one_cov(cov_one_cov, gaussian=True))), label="NaMaster/OneCovariance")
plt.xscale('squareroot')
plt.xticks(np.array([100, 400, 900, 1600]))
plt.minorticks_on()
plt.tick_params(axis='x', which='minor', length=2, width=0.8)
minor_ticks = [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 21)]
plt.xticks(minor_ticks, minor=True)
plt.legend()

plt.show()

# %%
