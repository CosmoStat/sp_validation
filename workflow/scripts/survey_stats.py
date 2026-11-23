"""Survey numbers of the v2.0.0 calibrated sample on its current bins and masks.

Reads the release's HATS calibrated catalogue (TOM_BIN_ID as delivered), its two
footprints and its response record (catalogs/response.json). Each sample's sigma_e is
that of its own calibrated shear, as cosmo_val measures it: the whole sample's
R^-1 R e, a bin's R_k^-1 R e, minus the sample's weighted mean
(sp_validation.response.calibrated_shear). Writes, in the layout of the release's
catalogs/survey*.json:
- survey.json: whole-sample and per-bin n_eff and sigma_e on the full footprint;
- survey_tomo.json: per-bin n_eff and sigma_e on the tomographic footprint, plus the
  analytic/true shape-noise variance ratio at nside 256 on both areas.
Checks that the tomographic mask is the occupied nside-4096 pixels of the tiles that hold
a bin 1-4 object, and that the full mask is the occupied pixels of the whole sample.

usage: survey_stats.py <release dir> <outdir>
"""
import glob, json, pathlib, sys
from concurrent.futures import ThreadPoolExecutor

import healpy as hp
import numpy as np
import pyarrow.parquet as pq
from sp_validation.response import calibrated_shear, sample_response
from sp_validation.survey import effective_survey_stats, ellipticity_dispersion

REL, OUT = map(pathlib.Path, sys.argv[1:3])
NSIDE = 4096
COLS = ["RA", "Dec", "e1", "e2", "w_des", "TOM_BIN_ID", "TILE_KEY"]

files = sorted(glob.glob(str(REL / "catalogs/calibrated/dataset/**/*.parquet"), recursive=True))
with ThreadPoolExecutor(32) as ex:
    parts = list(ex.map(lambda f: pq.read_table(f, columns=COLS), files))
col = {c: np.concatenate([p[c].to_numpy() for p in parts]) for c in COLS}
del parts
ra, dec = col["RA"].astype(float), col["Dec"].astype(float)
e1, e2, w = (col[k].astype(float) for k in ("e1", "e2", "w_des"))
tb = col["TOM_BIN_ID"].astype(np.int16)
tile = col["TILE_KEY"]
print("rows", len(tb), flush=True)
record = json.loads((REL / "catalogs/response.json").read_text())

pix = hp.ang2pix(NSIDE, ra, dec, lonlat=True)
pixarea = hp.nside2pixarea(NSIDE, degrees=True)
binned = tb > 0
tomo_tiles = np.unique(tile[binned])
in_tomo_tiles = np.isin(tile, tomo_tiles)

checks = {}
for name, sel in [("full", slice(None)), ("tomo", in_tomo_tiles)]:
    fname = {"full": "footprint-nside4096-ring.fits",
             "tomo": "footprint-tomo-nside4096-ring.fits"}[name]
    m = hp.read_map(REL / "masks" / fname, dtype=np.uint8, nest=False)
    occupied = np.unique(pix[sel])
    on_disk = np.flatnonzero(m > 0)
    checks[name] = {"file": fname, "pixels_on_disk": int(len(on_disk)),
                    "pixels_occupied": int(len(occupied)),
                    "equal": bool(np.array_equal(on_disk, occupied)),
                    "area_deg2": float(len(on_disk) * pixarea)}
print(json.dumps(checks, indent=1), flush=True)
assert all(c["equal"] for c in checks.values()), "a footprint is not the sample's occupied pixels"
full_area, tomo_area = checks["full"]["area_deg2"], checks["tomo"]["area_deg2"]


def stats(sel, area, b=None):
    """n_eff and sigma_e of one sample: the whole catalogue (b None) or bin b."""
    response = sample_response(record) if b is None else sample_response(record, "TOM_BIN_ID", b)
    g1, g2 = calibrated_shear(np.array([e1[sel], e2[sel]]), w[sel], response, record["R"])
    s = effective_survey_stats(g1, g2, w[sel], area)
    s["sigma_e_per_component"] = float(ellipticity_dispersion(g1, g2, w[sel]))
    return {k: (float(v) if np.ndim(v) == 0 else np.asarray(v).tolist()) for k, v in s.items()}


survey = {"rows": int(len(tb)), "area_deg2": full_area, "nside": NSIDE,
          "bin_counts": {str(b): int((tb == b).sum()) for b in range(5)}, "samples": {}}
for name, sel, b in [("all", slice(None), None)] + [(f"bin{b}", tb == b, b) for b in (1, 2, 3, 4)]:
    survey["samples"][name] = stats(sel, full_area, b)
    survey["samples"][name]["weight_fraction"] = float(w[sel].sum() / w.sum())
(OUT / "survey.json").write_text(json.dumps(survey, indent=1) + "\n")

p256 = hp.ang2pix(256, ra, dec, lonlat=True)
a256 = hp.nside2pixarea(256, degrees=True)


def noise_ratio(sel, area_deg2):
    """Analytic (uniform, area_deg2) over pixel-resolved shape-noise variance."""
    ww = w[sel]
    r1 = np.bincount(p256[sel], ww, minlength=hp.nside2npix(256)) / a256
    r2 = np.bincount(p256[sel], ww**2, minlength=hp.nside2npix(256)) / a256
    true = (r2**2).sum() * a256 / ((r1**2).sum() * a256) ** 2
    return float(area_deg2 * (ww**2).sum() ** 2 / ww.sum() ** 4 / true)


tomo = {"nside": NSIDE, "area_deg2": tomo_area, "full_area_deg2": full_area,
        "tiles": {"all": int(len(np.unique(tile))), "tomographic": int(len(tomo_tiles))},
        "binned_weight_in_footprint": float(w[binned & in_tomo_tiles].sum() / w[binned].sum()),
        "samples": {}}
for b in (1, 2, 3, 4):
    sel = tb == b
    s = stats(sel, tomo_area, b)
    s["noise_ratio_nside256"] = {"full_area": noise_ratio(sel, full_area),
                                 "tomographic_area": noise_ratio(sel, tomo_area)}
    tomo["samples"][f"bin{b}"] = s
(OUT / "survey_tomo.json").write_text(json.dumps(tomo, indent=1) + "\n")
print(json.dumps(survey, indent=1))
print(json.dumps(tomo, indent=1))
