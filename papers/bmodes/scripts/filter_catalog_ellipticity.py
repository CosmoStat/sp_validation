"""Filter a shear catalog by ellipticity magnitude.

Applies |e| < e_max using leak-corrected ellipticity columns,
writes filtered FITS, and computes survey properties (n_eff, sigma_e)
using the parent version's area.
"""

import json
import sys

import numpy as np
from astropy.io import fits

sys.stdout = (
    sys.stdout if hasattr(sys, "ps1") else open(sys.stdout.fileno(), "w", buffering=1)
)
sys.stderr = (
    sys.stderr if hasattr(sys, "ps1") else open(sys.stderr.fileno(), "w", buffering=1)
)

from snakemake.script import snakemake  # noqa: E402

input_path = snakemake.input["catalog"]
output_fits = snakemake.output["catalog"]
output_props = snakemake.output["survey_props"]

e_max = float(snakemake.params["e_max"])
e1_col = snakemake.params["e1_col"]
e2_col = snakemake.params["e2_col"]
w_col = snakemake.params["w_col"]
parent_area_deg2 = float(snakemake.params["parent_area_deg2"])

print(f"Loading catalog: {input_path}")
with fits.open(input_path, memmap=True) as hdul:
    data = hdul[1].data
    n_total = len(data)

    e1 = np.asarray(data[e1_col], dtype=np.float32)
    e2 = np.asarray(data[e2_col], dtype=np.float32)
    e_mag = np.sqrt(e1**2 + e2**2)
    del e1, e2

    mask = e_mag < e_max
    del e_mag
    n_kept = int(np.sum(mask))
    print(
        f"Ellipticity cut |e| < {e_max}: {n_kept}/{n_total} galaxies kept ({100 * n_kept / n_total:.1f}%)"
    )

    # Compute survey properties from masked columns (memory-efficient)
    w = np.asarray(data[w_col][mask], dtype=np.float64)
    e1_filt = np.asarray(data[e1_col][mask], dtype=np.float64)
    e2_filt = np.asarray(data[e2_col][mask], dtype=np.float64)

    sum_w = float(np.sum(w))
    sum_w2 = float(np.sum(w**2))
    sum_w2_e2 = float(np.sum(w**2 * (e1_filt**2 + e2_filt**2)))
    del w, e1_filt, e2_filt

    filtered = data[mask]
    del mask

    print(f"Writing filtered catalog: {output_fits}")
    fits.writeto(output_fits, filtered, overwrite=True)
    del filtered

area_arcmin2 = parent_area_deg2 * 3600.0
n_eff = sum_w**2 / (area_arcmin2 * sum_w2) if sum_w2 > 0 else 0.0
sigma_e = np.sqrt(sum_w2_e2 / sum_w2) if sum_w2 > 0 else 0.0

props = {
    "parent_area_deg2": parent_area_deg2,
    "n_eff": n_eff,
    "sigma_e": sigma_e,
    "n_total": n_total,
    "n_kept": n_kept,
    "fraction_kept": n_kept / n_total,
    "e_max": e_max,
}

print(f"Survey properties: n_eff={n_eff:.3f} gal/arcmin², sigma_e={sigma_e:.4f}")
print(f"  (parent n_eff and sigma_e should be compared to assess impact)")

with open(output_props, "w") as f:
    json.dump(props, f, indent=2)

print("Done.")
