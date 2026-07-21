# %%
from pathlib import Path

import IPython
import numpy as np
import yaml
from astropy.io import fits
from astropy.table import Table

from sp_validation.cosmo_val import CosmologyValidation
from sp_validation.rho_tau import get_params_rho_tau

ipython = IPython.get_ipython()

# %%
NSIDE = 64
SEED = 1234
N_ELL_BINS = 8

rng = np.random.default_rng(SEED)
version = "TestCatalog"

cat_dir = Path("./catalog")
nz_dir = Path("./nz")
output_dir = Path("./output_test")

for d in (cat_dir, nz_dir, output_dir):
    d.mkdir(exist_ok=True)

n_gal = 5000
ra = rng.uniform(10.0, 30.0, n_gal)
dec = rng.uniform(10.0, 30.0, n_gal)
e1 = rng.normal(0, 0.25, n_gal)
e2 = rng.normal(0, 0.25, n_gal)
w = rng.uniform(0.5, 1.0, n_gal)
tomo_bin_id = rng.integers(1, 3, n_gal)  # Create a two bin catalogue
Table(
    {"RA": ra, "Dec": dec, "e1": e1, "e2": e2, "w": w, "tomo_bin_id": tomo_bin_id}
).write(cat_dir / "shear.fits", overwrite=True)

shear_cfg = {
    "path": "shear.fits",
    "w_col": "w",
    "e1_col": "e1",
    "e2_col": "e2",
    "R": 1.0,
    "e1_col_corrected": "e1",
    "e2_col_corrected": "e2",
    "ra_col": "RA",
    "dec_col": "Dec",
    "tomo_bin_col": "tomo_bin_id",
}

psf_cfg = {
    "path": "shear.fits",
    "ra_col": "RA",
    "dec_col": "Dec",
    "e1_PSF_col": "e1",
    "e2_PSF_col": "e2",
    "e1_star_col": "e1",
    "e2_star_col": "e2",
    "PSF_size": "w",
    "star_size": "w",
    "PSF_flag": "w",
    "star_flag": "w",
}
config_data = {
    "nz": {"subdir": str(nz_dir), "dndz": {"blind": "A", "path": "dndz"}},
    "paths": {"output": str(output_dir)},
    version: {
        "subdir": str(cat_dir),
        "pipeline": "SP",
        "shear": shear_cfg,
        "star": {**psf_cfg},
        "psf": psf_cfg,
    },
}

config_path = Path("./config.yaml")
config_path.write_text(yaml.dump(config_data, sort_keys=False))

# %%
cv = CosmologyValidation(
    versions=[version],
    catalog_config=config_path,
    output_dir=str(output_dir),
    nside=NSIDE,
    binning="powspace",
    power=0.5,
    n_ell_bins=N_ELL_BINS,
    pol_factor=-1,
)
cv._test_version = version

ver = cv._test_version
params = get_params_rho_tau(cv.cc[ver], survey=ver)
cat_gal = fits.getdata(cv.cc[ver]["shear"]["path"])

cat_gal_tomo_bin_1 = cat_gal[cat_gal[params["tomo_bin_col"]] == 1]
cat_gal_tomo_bin_2 = cat_gal[cat_gal[params["tomo_bin_col"]] == 2]
n_gal_map_a = cv.get_n_gal_map(params, NSIDE, cat_gal_tomo_bin_1)
shear_map_a_e1, shear_map_a_e2 = cv.get_shear_map(params, NSIDE, cat_gal_tomo_bin_1)
n_gal_map_b = cv.get_n_gal_map(params, NSIDE, cat_gal_tomo_bin_2)
shear_map_b_e1, shear_map_b_e2 = cv.get_shear_map(params, NSIDE, cat_gal_tomo_bin_2)

shear_map_a = shear_map_a_e1 + 1j * shear_map_a_e2
shear_map_b = shear_map_b_e1 + 1j * shear_map_b_e2


# %%
ell_eff, cl_all, wsp = cv.get_pseudo_cls_map(
    shear_map_a, n_gal_map_a, shear_map_b=shear_map_b, mask_b=n_gal_map_b
)

# %%
# Get the catalogue output
ver = cv._test_version
cv._pseudo_cls = {
    ver: {
        "tomo_bin_1_tomo_bin_1": {},
        "tomo_bin_1_tomo_bin_2": {},
        "tomo_bin_2_tomo_bin_2": {},
    }
}
out_path = cv._output_path(f"pseudo_cl_cat_{ver}.fits")
tomo_bin_ids, tomo_bin_pairs = cv._get_tomo_bins(ver)
for tomo_bin_a, tomo_bin_b in tomo_bin_pairs:
    out_path = cv._output_path(f"pseudo_cl_cat_{ver}_{tomo_bin_a}_{tomo_bin_b}.fits")
    cv.calculate_pseudo_cl_catalog(
        ver, out_path, tomo_bin_a=tomo_bin_a, tomo_bin_b=tomo_bin_b
    )


# %%
np.savez("./test_cl_catalog", cv._pseudo_cls[ver])
# %%
test_result = np.load("./test_cl_catalog.npz", allow_pickle=True)
