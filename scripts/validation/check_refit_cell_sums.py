"""Compare all grid refit-W tables, responses and image-sim m/c to row rescans.

Run in the science container on an allocated node. Inputs are unblinded grids.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from round2_weights import BASE, EXTERNAL, covariance, load

from sp_validation import response
from sp_validation.des_jackknife import accumulate


def check(grid):
    maxima = dict(
        W=0.0,
        shear=0.0,
        selection=0.0,
        total=0.0,
        joint=0.0,
        c_e=0.0,
        c_g=0.0,
        covariance=0.0,
        mc=0.0,
        mc_covariance=0.0,
    )
    deletes = [dict(), dict()]
    for name in ("1z2z", *EXTERNAL):
        gm, groups = load(grid, name, "des", min_count=3)
        acc = accumulate(gm, groups)
        labels, fast, cov = acc.jackknife()
        ref_labels, ref, ref_cov = gm.response_jackknife(
            groups, refit_w=True, reference=True
        )
        np.testing.assert_array_equal(labels, ref_labels)
        maxima["covariance"] = max(
            maxima["covariance"], float(abs(cov - ref_cov).max())
        )
        for key in fast:
            diff = float(abs(fast[key] - ref[key]).max())
            maxima[key] = max(maxima[key], diff)
            np.testing.assert_allclose(fast[key], ref[key], atol=1e-12, rtol=0)
        ns = gm.mask_dict["ns"]
        snr = (gm.ns["flux"] / gm.ns["flux_err"])[ns]
        size = (gm.ns["T"] / gm.ns["Tpsf"])[ns]
        e = np.array(gm.branches["ns"])[:, ns]
        r = 0.5 * (gm.R11 + gm.R22)
        for k, label in enumerate(labels):
            keep = groups[ns] != label
            table = response.fit_w(
                snr[keep],
                size[keep],
                e[:, keep],
                r[keep],
                gm.W[0],
                gm.W[1],
                min_count=3,
            )
            diff = float(abs(acc.deleted_weight(k) - table[2]).max())
            maxima["W"] = max(maxima["W"], diff)
            np.testing.assert_allclose(
                acc.deleted_weight(k), table[2], atol=1e-12, rtol=0
            )
        for d, results in zip(deletes, (fast, ref)):
            d[name] = results["c_g"]
        print(grid, name, maxima, flush=True)
    with open(BASE / f"grid_{grid}/results/m_bias_config.yaml") as f:
        gin = yaml.safe_load(f)["shear_amplitude"]

    def mc(values):
        result = []
        for i, (p, m) in enumerate((("1p2z", "1m2z"), ("1z2p", "1z2m"))):
            result.extend(
                (
                    (values[p][:, i] - values[m][:, i]) / (2 * gin) - 1,
                    (values[p][:, i] + values[m][:, i]) / 2,
                )
            )
        return np.array(result).T

    fast, ref = map(mc, deletes)
    maxima["mc"] = float(abs(fast - ref).max())
    maxima["mc_covariance"] = float(abs(covariance(fast) - covariance(ref)).max())
    np.testing.assert_allclose(fast, ref, atol=1e-12, rtol=0)
    assert max(maxima.values()) <= 1e-12, maxima
    return maxima


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {str(grid): check(grid) for grid in range(1, 5)}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
