"""Stage-1 scalar response curve with paired spatial jackknife errors.

Public v1.4.6.3 or unblinded grids only. No cosmological signals are measured.
"""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

from sp_validation.response_diagnostics import scalar_pair_response

PUBLIC = Path(
    "/n17data/UNIONS/WL/v1.4.x/v1.4.6.3/unions_shapepipe_cut_struc_2024_v1.4.6.3.fits"
)
RELEASE = Path("/automnt/n17data/cdaley/unions/code/sp_validation/cosmo_val/output")


def run(sample, output, bin_slop):
    start = time.monotonic()
    if sample == "public":
        with fits.open(PUBLIC, memmap=True) as f:
            d = f[1].data
            ra, dec, w = [np.array(d[k], dtype=float) for k in ("RA", "Dec", "w_des")]
            r = 0.5 * (d["R_g11"].astype(float) + d["R_g22"].astype(float))
        # Existing release patch centres; globals and pair products delete the
        # same patches. Their covariance is included before forming the ratio.
        centers = str(RELEASE / "SP_v1.4.6.3_patches_npatch=100.dat")
        result = scalar_pair_response(
            ra, dec, w, r, patch_centers=centers, bin_slop=bin_slop
        )
    else:
        from round2_weights import BASE, load

        gm, tiles = load(int(sample), "1z2z", "des")
        # Read the cut coordinates only; selection and row order checked.
        with fits.open(
            BASE / f"grid_{sample}/1z2z_grid_{sample}/shape_catalog_cut_ngmix.fits"
        ) as f:
            ra, dec = [np.array(f[1].data[k], dtype=float) for k in ("RA", "Dec")]
            for i in (1, 2):
                np.testing.assert_allclose(
                    f[1].data[f"e{i}_uncal"],
                    gm.ns[f"g{i}"][gm.mask_dict["ns"]],
                    atol=1e-12,
                )
        idx = gm.mask_dict["ns"]
        w = gm.branch_weights["ns"][idx]
        assert len(ra) == len(w)
        patch = np.unique(tiles, return_inverse=True)[1][idx]
        r = 0.5 * (gm.R11 + gm.R22)
        result = scalar_pair_response(ra, dec, w, r, patch=patch, bin_slop=bin_slop)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / f"pairs_{sample}.npz", **result)
    summary = dict(
        sample=sample,
        n=len(w),
        seconds=time.monotonic() - start,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        bin_slop=bin_slop,
        max_fraction=float(np.max(abs(result["fraction"]))),
        min_fraction=float(np.min(result["fraction"])),
        max_signed_fraction=float(np.max(result["fraction"])),
        npatch=len(result["deleted"]),
        theta=result["theta"].tolist(),
        fraction=result["fraction"].tolist(),
        error=result["error"].tolist(),
    )
    if sample == "public":
        path = (
            RELEASE / "SP_v1.4.6.3_xi_minsep=1.0_maxsep=250.0_nbins=20_npatch=100.txt"
        )
        xi = np.loadtxt(path, skiprows=3, max_rows=20)
        # Global-calibrated xi -> pair-calibrated xi: multiply by 1/(1+f).
        shift = -result["fraction"] / (1 + result["fraction"])
        summary.update(
            xi_source=str(path),
            fractional_xi_shift=shift.tolist(),
            xip_shift_over_sigma=(shift * xi[:, 3] / xi[:, 7]).tolist(),
            xim_shift_over_sigma=(shift * xi[:, 4] / xi[:, 8]).tolist(),
            xip_shift_over_sigma_error=(
                result["error"]
                / (1 + result["fraction"]) ** 2
                * abs(xi[:, 3] / xi[:, 7])
            ).tolist(),
            xim_shift_over_sigma_error=(
                result["error"]
                / (1 + result["fraction"]) ** 2
                * abs(xi[:, 4] / xi[:, 8])
            ).tolist(),
        )
    (output / f"pairs_{sample}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", choices=["public", "1", "2", "3", "4"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--bin-slop",
        type=float,
        help="Default: 0.05 on the full public release, exact bins (0) on grids",
    )
    a = p.parse_args()
    slop = (
        a.bin_slop if a.bin_slop is not None else (0.05 if a.sample == "public" else 0)
    )
    run(a.sample, a.output, slop)
