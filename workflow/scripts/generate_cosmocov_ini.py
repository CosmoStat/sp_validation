"""Generate a CosmoCov ``.ini`` for one (version, blind, grid, flavour, mask).

Cosmology comes from the frozen ``planck18.json`` snapshot, survey parameters
(area, n_eff, sigma_e) from the catalog config's per-version ``cov_th``, and
n(z) from ``workflow/common.build_redshift_path``. The footprint mask power
spectrum is passed explicitly (empty string for the unmasked variant).

    python generate_cosmocov_ini.py \
        --version SP_v1.4.6.3_leak_corr --blind A \
        --planck18-json <cosmology_snapshot>/planck18.json \
        --cat-config <cosmo_val/cat_config.yaml> \
        --min-sep 0.5 --max-sep 300.0 --nbins 1000 --gaussian g \
        --mask-cls <mask_cls_footprint_nside_4096_norm.txt> \
        --out-ini <path/to/covariance.ini>
"""

import argparse
import importlib.util
import json
import os
import sys

import yaml


def _load_workflow_common():
    """Load ``workflow/common.py`` (this script also runs outside Snakemake)."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "common.py"
    )
    spec = importlib.util.spec_from_file_location("workflow_common", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


common = _load_workflow_common()


INI_TEMPLATE = """\
#
# Cosmological parameters
#
Omega_m : {Omega_m}
Omega_v : {Omega_v}
sigma_8 : {sigma_8}
n_spec : {n_s}
w0 : -1
wa : 0
omb : {Omega_b}
h0 : {h}


# Survey and galaxy parameters
#
# area in degrees
# n_gal,lens_n_gal in gals/arcmin^2

area : {area}
sourcephotoz : multihisto
lensphotoz : multihisto
source_tomobins : 1
lens_tomobins : 1
sigma_e : {sigma_e}
source_n_gal : {n_e}
lens_n_gal : {n_e}


shear_REDSHIFT_FILE : {nz}
clustering_REDSHIFT_FILE : {nz}
c_footprint_file : {mask}


# IA parameters
IA : 1
A_ia : 0.0
eta_ia : 0.0


# Covariance parameters
#
# tmin,tmax in arcminutes
tmin : {min_sep}
tmax : {max_sep}
ntheta : {nbins}
ng : {ng}
cng : {ng}


outdir : ./
filename : cov_tmp
ss : true
ls : false
ll : false
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--version", required=True)
    ap.add_argument("--blind", default="A")
    ap.add_argument("--planck18-json", required=True)
    ap.add_argument("--cat-config", required=True)
    ap.add_argument("--min-sep", required=True, help="tmin arcmin (string, e.g. 0.5)")
    ap.add_argument("--max-sep", required=True, help="tmax arcmin (string, e.g. 300.0)")
    ap.add_argument("--nbins", required=True, help="ntheta (string, e.g. 1000)")
    ap.add_argument("--gaussian", required=True, choices=["g", "ng"])
    ap.add_argument(
        "--mask-cls", default="", help="footprint mask Cl path ('' = unmasked)"
    )
    ap.add_argument("--out-ini", required=True)
    a = ap.parse_args(argv)

    with open(a.planck18_json) as f:
        cosmo = json.load(f)
    with open(a.cat_config) as f:
        cat_config = yaml.safe_load(f)

    cov_th = cat_config[common.base_version(a.version)]["cov_th"]

    ng_value = "1" if a.gaussian == "ng" else "0"

    ini = INI_TEMPLATE.format(
        Omega_m=cosmo["Omega_m"],
        Omega_v=cosmo["Omega_v"],
        sigma_8=cosmo["sigma_8"],
        n_s=cosmo["n_s"],
        Omega_b=cosmo["Omega_b"],
        h=cosmo["h"],
        area=cov_th["A"],
        sigma_e=cov_th["sigma_e"],
        n_e=cov_th["n_e"],
        nz=common.build_redshift_path(a.version, a.blind),
        mask=a.mask_cls,
        min_sep=a.min_sep,
        max_sep=a.max_sep,
        nbins=a.nbins,
        ng=ng_value,
    )

    os.makedirs(os.path.dirname(os.path.abspath(a.out_ini)), exist_ok=True)
    with open(a.out_ini, "w") as f:
        f.write(ini)
    print(f"Wrote {a.out_ini}")


if __name__ == "__main__":
    main()
