"""Tile-jackknife error of the difference in m between two response
weightings, e.g. iv_d - iv. Both share the same objects and shear mean, so
the error of the difference is much smaller than either error alone.

Runs ImageSimMBias on both schemes' m_bias_config.yaml and records the
leave-one-tile-out samples of m that _jackknife_std receives.

Usage: python diff_jk.py <out dir> <scheme a> <scheme b> [...]
Pairs are (a, b): prints m_b - m_a per weight column, method and component,
and writes them to <out dir>/diff_<b>_<a>.yaml.
"""

import os
import sys

import numpy as np
import yaml

from sp_validation import image_sims


def samples(out_root, scheme):
    """Per-tile m samples: {(method, w_col, comp): array}."""
    with open(f"{out_root}/{scheme}/m_bias_config.yaml") as f:
        cfg = yaml.safe_load(f)
    mb = image_sims.ImageSimMBias(cfg)
    mb.load_catalogs(verbose=False)
    got = []
    orig = image_sims._jackknife_std
    image_sims._jackknife_std = lambda x: (got.append(np.array(x)), orig(x))[1]
    try:
        out = {}
        for method in ("default", "joint"):
            got.clear()
            for name_p, name_m, comp in mb.pairs:
                res = mb._m_c_pair(name_p, name_m, comp, verbose=False, method=method)
                # _jackknife_pair calls m then c for each w_col in order
                for k, w_col in enumerate(mb.w_cols):
                    out[(method, w_col, comp + 1)] = (
                        res[w_col]["m"],
                        got[-2 * len(mb.w_cols) + 2 * k],
                    )
        return out
    finally:
        image_sims._jackknife_std = orig


def main():
    out_root = sys.argv[1]
    pairs = list(zip(sys.argv[2::2], sys.argv[3::2]))
    for a, b in pairs:
        sa, sb = samples(out_root, a), samples(out_root, b)
        res = {}
        for key in sa:
            method, w_col, comp = key
            d = sb[key][0] - sa[key][0]
            err = image_sims._jackknife_std(sb[key][1] - sa[key][1])
            res.setdefault(method, {}).setdefault(w_col, {})[f"dm{comp}"] = float(d)
            res[method][w_col][f"dm{comp}_err_jk"] = float(err)
            print(f"{b} - {a}  {method:7s} {w_col:6s} m{comp}: {d:+.5f} +/- {err:.5f}")
        with open(os.path.join(out_root, f"diff_{b}_{a}.yaml"), "w") as f:
            yaml.dump(res, f, default_flow_style=False)


if __name__ == "__main__":
    main()
