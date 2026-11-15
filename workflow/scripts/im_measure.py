#!/usr/bin/env python
"""Image-simulation m and c under every weighting, in S/N bins, with each
simulation's response and the sample's (S/N, size) distribution.

Each simulation is read from its comprehensive catalogue and preselected with
its ``config_mask.yaml`` (the ``dat`` cuts), exactly as the calibration step
does; the metacal cuts and the DES table product come from the same file.
Per simulation, :func:`sp_validation.response.accumulate` reduces the rows to
per-tile sums once with ``u = 1`` and once with inverse-variance ``u``, and
once per S/N bin (the metacal cuts with that bin's S/N limits, on each
branch's own S/N). Every m and c then comes from
:func:`~sp_validation.response.shear_bias` with the tile-deleted covariance of
:func:`~sp_validation.response.shear_bias_jackknife`:

- ``des``: the DES table product (the ``im_mbias`` number);
- ``unit``: unit weights;
- ``iv``: inverse-variance weights (``u`` from each branch's own errors);
- ``snr_bins``: the DES table, per S/N bin.

Per simulation the record also holds the response (shear, selection, total)
and the weighted mean ellipticity before calibration (``c_e``) under the DES
table, with jackknife errors. The ``--distributions`` npz holds, for the
reference simulation, the no-shear selection's counts and summed shear
response on the DES table's (S/N, size) cells, and finer one-dimensional
histograms of S/N = FLUX/FLUX_ERR and T/T_PSF_RECONV of the preselected and
the metacal-selected rows, to compare with the data's.
"""

import argparse

import numpy as np
import yaml

from sp_validation import catalog_builders, response

SNR_HIST = np.logspace(0, 3, 61)
SIZE_HIST = np.logspace(-1, 1.5, 51)


def read(catalogue, mask_config):
    """Rows, preselection, tiles and config of one simulation."""
    obj = catalog_builders.CalibrateCat()
    config = obj.read_config_set_params(mask_config)
    obj._params["input_path"] = catalogue
    obj._params["verbose"] = False
    rows = obj.read_cat(load_into_memory=True)
    masks, _ = catalog_builders.get_masks_from_config(config, rows)
    pre = catalog_builders.Mask.from_list(masks)._mask
    tiles = np.char.strip(np.asarray(rows["TILE_ID"]).astype(str))
    return rows, pre, tiles, config


def bias(pairs, g_in, table):
    """m, c and their jackknife errors and covariance."""
    values = response.shear_bias(pairs, g_in, table)
    labels, _, cov = response.shear_bias_jackknife(pairs, g_in, table)
    out = {k: float(v) for k, v in values.items()}
    out.update({f"{k}_err": float(e) for k, e in zip(values, np.sqrt(np.diag(cov)))})
    out["covariance"] = cov.tolist()
    out["n_tiles"] = len(labels)
    return out


def describe(sums, table):
    """A simulation's response and mean ellipticity, with jackknife errors."""
    est = response.estimate(sums, table)
    _, deleted, _ = response.jackknife(sums, table)
    k = len(sums.labels)
    out = {}
    for key in ("shear", "selection", "total", "c_e"):
        d = deleted[key]
        err = np.sqrt((k - 1) / k * ((d - d.mean(axis=0)) ** 2).sum(axis=0))
        out[f"R_{key}" if key != "c_e" else "c_e"] = np.asarray(est[key]).tolist()
        out[f"R_{key}_err" if key != "c_e" else "c_e_err"] = err.tolist()
    out["counts"] = est["counts"]
    return out


def histograms(rows, pre, cuts):
    """S/N and size histograms of the preselected and the selected rows."""
    ns = response.branch_columns(rows, "ns")
    sel = pre & response.select(ns, cuts)
    out = {"snr_hist_edges": SNR_HIST, "size_hist_edges": SIZE_HIST}
    for name, rows_in in (("preselected", pre), ("selected", sel)):
        out[f"snr_hist_{name}"] = np.histogram(ns["snr"][rows_in], SNR_HIST)[0]
        out[f"size_hist_{name}"] = np.histogram(ns["size"][rows_in], SIZE_HIST)[0]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--catalogues", nargs="+", required=True,
                   help="comprehensive catalogue per simulation, in branch order")
    p.add_argument("--masks", nargs="+", required=True,
                   help="config_mask.yaml per simulation, same order")
    p.add_argument("--snr-edges", nargs="+", type=float, required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--distributions", required=True)
    args = p.parse_args()

    with open(args.manifest) as f:
        manifest = yaml.safe_load(f)
    branches = list(manifest["branches"])
    g_in = manifest["shear_amplitude"]

    data = {
        b: read(cat, mask)
        for b, cat, mask in zip(branches, args.catalogues, args.masks, strict=True)
    }
    labels = np.unique(np.concatenate([d[2][d[1]] for d in data.values()]))
    metacal = data[branches[0]][3]["metacal"]
    table = response.read_table(metacal["weight_table"])
    edges = table[:2]
    unit = response.unit_table(edges)
    cuts = response.Cuts.from_config(metacal)
    step = metacal.get("step", 0.01)

    def sums(b, cuts=cuts, sigma_e=None):
        rows, pre, tiles, _ = data[b]
        return response.accumulate(
            rows, cuts, edges, preselection=pre, groups=tiles, labels=labels,
            sigma_e=sigma_e, step=step,
        )

    has_errors = all(
        f"NGMIX_G1_ERR_{s}" in data[b][0].dtype.names
        for b in branches for s in ("1P", "1M", "2P", "2M")
    )
    u1 = {b: sums(b) for b in branches}
    iv = (
        {b: sums(b, sigma_e=metacal["sigma_eps_prior"]) for b in branches}
        if has_errors else None
    )

    def pairs(s):
        return {p["component"]: (s[p["plus"]], s[p["minus"]]) for p in manifest["pairs"]}

    results = {
        "shear_amplitude": g_in,
        "weight_table": metacal["weight_table"],
        "cuts": cuts._asdict(),
        "m": {"des": bias(pairs(u1), g_in, table), "unit": bias(pairs(u1), g_in, unit)},
        "simulations": {b: describe(u1[b], table) for b in branches},
    }
    if iv is not None:
        results["m"]["iv"] = bias(pairs(iv), g_in, unit)

    bins = []
    for lo, hi in zip(args.snr_edges[:-1], args.snr_edges[1:]):
        binned = {b: sums(b, cuts._replace(snr_min=lo, snr_max=hi)) for b in branches}
        bins.append({"snr_min": lo, "snr_max": hi,
                     **bias(pairs(binned), g_in, table),
                     "counts_ns": {b: binned[b].total().arrays["count"][0].sum().item()
                                   for b in branches}})
    results["m"]["snr_bins"] = bins

    with open(args.output, "w") as f:
        yaml.safe_dump(results, f, sort_keys=False)

    ref = manifest["reference"]
    total = u1[ref].total().arrays
    shape = (len(edges[0]) - 1, len(edges[1]) - 1)
    count = total["count"][0, 0].reshape(shape)
    rg = total["rg"][:, :, 0].reshape(2, 2, *shape)
    rows, pre, _, _ = data[ref]
    np.savez(
        args.distributions,
        simulation=ref,
        snr_edges=edges[0],
        size_edges=edges[1],
        count=count,
        rg_sum=rg,
        R_shear=np.divide(rg, count, out=np.full_like(rg, np.nan), where=count > 0),
        **histograms(rows, pre, cuts),
    )
    for key in ("des", "unit", "iv"):
        if key in results["m"]:
            r = results["m"][key]
            print(f"{key:5s} m1={r['m1']:+.4f}+-{r['m1_err']:.4f} "
                  f"m2={r['m2']:+.4f}+-{r['m2_err']:.4f} "
                  f"c1={r['c1']:+.5f}+-{r['c1_err']:.5f} "
                  f"c2={r['c2']:+.5f}+-{r['c2_err']:.5f}")


if __name__ == "__main__":
    main()
