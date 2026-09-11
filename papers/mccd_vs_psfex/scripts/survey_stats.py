"""Survey statistics (A, n_e, sigma_e, n_psf) for the cov_th block of cat_config.

Same estimators as papers/bmodes/scripts/update_survey_stats.py:
A from the binary HEALPix footprint mask, n_eff = (sum w)^2 / (A sum w^2),
sigma_e = sqrt(sum w^2 |e|^2 / sum w^2). n_psf is the density of PSF-catalogue
stars inside the mask. Also reports the fraction of galaxies (and stars) that
fall inside the mask, as a check that mask and catalogue belong together.

Usage:
    python survey_stats.py --cat-config cat_config.yaml SP_v1.3.6.3 SP_v1.4.6.3
"""

import argparse
import os

import healpy as hp
import numpy as np
import yaml
from astropy.io import fits


def resolve(cc, ver, path):
    return path if os.path.isabs(path) else os.path.join(cc[ver]["subdir"], path)


def stats(cc, ver, e_cols=("e1", "e2"), chunk=2_000_000):
    mask = hp.read_map(cc[ver]["mask"], dtype=np.float64)
    nside = hp.get_nside(mask)
    area_deg2 = float(mask.sum() * hp.nside2pixarea(nside, degrees=True))

    shear = cc[ver]["shear"]
    sum_w = sum_w2 = sum_w2_e2 = 0.0
    n_gal = n_in = 0
    with fits.open(resolve(cc, ver, shear["path"]), memmap=True) as hdul:
        data = hdul[1].data
        for start in range(0, len(data), chunk):
            d = data[start : start + chunk]
            w = np.asarray(d[shear["w_col"]], dtype=np.float64)
            e1 = np.asarray(d[e_cols[0]], dtype=np.float64)
            e2 = np.asarray(d[e_cols[1]], dtype=np.float64)
            sum_w += w.sum()
            sum_w2 += (w**2).sum()
            sum_w2_e2 += (w**2 * (e1**2 + e2**2)).sum()
            pix = hp.ang2pix(nside, d["RA"], d["Dec"], lonlat=True)
            n_in += int((mask[pix] > 0).sum())
            n_gal += len(d)

    psf = cc[ver]["psf"]
    with fits.open(resolve(cc, ver, psf["path"]), memmap=True) as hdul:
        d = hdul[psf.get("hdu", 1)].data
        pix = hp.ang2pix(nside, d["RA"], d["DEC"], lonlat=True)
        inside = mask[pix] > 0
        ok = (d["FLAG_PSF_HSM"] == 0) & (d["FLAG_STAR_HSM"] == 0)
        n_star, n_star_in, n_star_ok_in = len(d), int(inside.sum()), int((inside & ok).sum())

    area_arcmin2 = area_deg2 * 3600
    return dict(
        nside=nside,
        A=area_deg2,
        n_e=sum_w**2 / (area_arcmin2 * sum_w2),
        sigma_e=float(np.sqrt(sum_w2_e2 / sum_w2)),
        n_psf_all=n_star / area_arcmin2,
        n_psf_in=n_star_in / area_arcmin2,
        n_psf_ok_in=n_star_ok_in / area_arcmin2,
        n_gal=n_gal,
        frac_gal_in_mask=n_in / n_gal,
        frac_star_in_mask=n_star_in / n_star,
        cov_th_config=cc[ver].get("cov_th"),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat-config", default="cat_config.yaml")
    ap.add_argument("versions", nargs="+")
    a = ap.parse_args()
    with open(a.cat_config) as f:
        cc = yaml.safe_load(f)
    for ver in a.versions:
        res = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in stats(cc, ver).items()}
        print(ver, yaml.safe_dump(res, sort_keys=False), flush=True)
