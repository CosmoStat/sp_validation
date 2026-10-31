"""Full-population, fixed-v2-table m with fixed parent occupancy multipliers.

Run on allocated nodes only. Use shared loading and response/JK kernels.
Anchor multipliers to no-external-shear detections and transfer by same-tile
sky position, never NUMBER (a detection sequence, not a cross-branch ID).
External-only detections extend the anchor catalogue on first appearance;
no population is restricted to matched objects. The anchor cell and multiplier
are then frozen across external/artificial branches and tile deletions.
SNR terciles instead use branch-specific SNR selection, including its R_sel.
"""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import inputs
import matplotlib.pyplot as plt
import numpy as np
import yaml
from inputs import EXTERNAL, covariance, load, table_weights
from scipy.spatial import cKDTree

from sp_validation import calibration, response


def read_grid(grid, name):
    original = calibration.metacal_from_config
    meta = {}

    def capture(data, pre, cfg):
        # Retain reference coordinates even for detections excluded by preselection.
        for k in (
            "RA",
            "Dec",
            "TILE_ID",
            "NGMIX_FLUX_NOSHEAR",
            "NGMIX_FLUX_ERR_NOSHEAR",
            "NGMIX_T_NOSHEAR",
            "NGMIX_T_PSF_RECONV_NOSHEAR",
        ):
            meta[k] = np.asarray(data[k]).copy()
        meta["pre"] = np.flatnonzero(pre)
        return original(data, pre, cfg)

    with patch.object(calibration, "metacal_from_config", capture):
        gm, tiles = load(grid, name, "des", min_count=3)
    return gm, tiles, meta


def xyz(ra, dec):
    r, d = np.deg2rad(ra), np.deg2rad(dec)
    return np.column_stack((np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)))


def index(snr, size, table):
    return response.cell_index(snr, table[0]) * table[2].shape[1] + response.cell_index(
        size, table[1]
    )


def meta_cells(meta, table):
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = meta["NGMIX_FLUX_NOSHEAR"] / meta["NGMIX_FLUX_ERR_NOSHEAR"]
        size = meta["NGMIX_T_NOSHEAR"] / meta["NGMIX_T_PSF_RECONV_NOSHEAR"]
    return index(snr, size, table), np.isfinite(snr) & np.isfinite(size)


def measure(gm, tiles, w, snr_range=None):
    masks, rg = gm.mask_dict, gm.R_shear
    if snr_range is not None:
        lo, hi = snr_range
        masks = {}
        for b, rows in gm.mask_dict.items():
            v = getattr(gm, b)
            s = v["flux"][rows] / v["flux_err"][rows]
            masks[b] = rows[(s > lo) & (s <= hi)]
        pos = np.searchsorted(gm.mask_dict["ns"], masks["ns"])
        np.testing.assert_array_equal(gm.mask_dict["ns"][pos], masks["ns"])
        rg = gm.R_shear[:, :, pos]
    point = response.responses(gm.branches, masks, w, rg, gm._step)
    labels, jk, _ = response.jackknife(gm.branches, masks, w, rg, gm._step, tiles)
    return point["c_g"], jk["c_g"], labels, {b: len(v) for b, v in masks.items()}


def mc(values, gin):
    out = []
    for i, (p, m) in enumerate((("1p2z", "1m2z"), ("1z2p", "1z2m"))):
        out.extend(
            (
                (values[p][..., i] - values[m][..., i]) / (2 * gin) - 1,
                (values[p][..., i] + values[m][..., i]) / 2,
            )
        )
    return np.array(out).T


def package(value, deleted, labels):
    cov = covariance(deleted)
    return dict(
        value=np.asarray(value).tolist(),
        error=np.sqrt(np.diag(cov)).tolist(),
        covariance=cov.tolist(),
        deleted=deleted.tolist(),
        labels=labels.tolist(),
    )


def combine(rows):
    v = np.array([r["value"] for r in rows])
    e = np.array([r["error"] for r in rows])
    w = e**-2
    a = w / w.sum(0)
    cov = sum(
        np.array(r["covariance"]) * a[i, :, None] * a[i, None, :]
        for i, r in enumerate(rows)
    )
    return dict(
        value=np.sum(a * v, 0).tolist(),
        error=np.sqrt(np.diag(cov)).tolist(),
        covariance=cov.tolist(),
        weights=a.tolist(),
    )


def run(output, production):
    path = output / "results.json"
    previous = json.loads(path.read_text())
    table, _ = calibration.load_des_weight_product(inputs.PRODUCT)
    es, er, wd, dc = table
    pd = dc / dc.sum()
    # Interpolate the W_COUNT CDF uniformly in log SNR inside occupied cells.
    cdf = np.r_[0, np.cumsum(pd.sum(1))]
    edges = np.exp(np.interp([0, 1 / 3, 2 / 3, 1], cdf, np.log(es)))
    ranges = list(zip(edges[:-1], edges[1:]))
    result = dict(
        order=["m1", "c1", "m2", "c2"],
        product=inputs.PRODUCT,
        method="Fixed ns-external anchor-cell multipliers; same-tile sky transfer at production match radius, external-only detections extend anchors; no matching restriction. Fixed across artificial/external shears and JK deletions.",
        tercile_method="Original (not occupancy-reweighted) populations; branch-specific SNR cuts with recalculated shear and selection responses. Data W_COUNT terciles, log-uniform interpolation within SNR bins.",
        snr_tercile_edges=edges.tolist(),
        grids={},
    )
    for grid in range(1, 5):
        gm0, _, meta0 = read_grid(grid, "1z2z")
        cell0, valid0 = meta_cells(meta0, table)
        count = np.array(previous["grids"][str(grid)]["count"])
        factor = pd / (count / count.sum())
        ns = gm0.mask_dict["ns"]
        np.testing.assert_array_equal(
            np.bincount(cell0[meta0["pre"]][ns], minlength=400).reshape(wd.shape), count
        )
        del gm0
        # Keep every initial raw detection as a positional anchor. Invalid raw
        # coordinates in SNR/size get neutral occupancy, not a fabricated cell.
        f0 = np.where(valid0, factor.ravel()[cell0], 1.0)
        anchor = {}
        tile0 = np.char.strip(meta0["TILE_ID"].astype(str))
        for t in np.unique(tile0):
            use = tile0 == t
            anchor[t] = (
                xyz(meta0["RA"][use], meta0["Dec"][use]),
                f0[use],
                np.zeros(use.sum(), bool),
            )
        cfg = yaml.safe_load(
            (inputs.BASE / f"grid_{grid}/results/m_bias_config.yaml").read_text()
        )
        radius = 2 * np.sin(np.deg2rad(cfg["match_radius_deg"]) / 2)
        means, deletes = {}, {}
        counts, transfers = {}, {}
        all_labels = None
        for name in EXTERNAL:
            gm, tiles, meta = read_grid(grid, name)
            idx, valid = meta_cells(meta, table)
            fallback = np.where(valid, factor.ravel()[idx], 1.0)
            parent_factor = np.ones(len(meta["RA"]))
            target_tile = np.char.strip(meta["TILE_ID"].astype(str))
            distance = np.zeros(len(parent_factor))
            fresh = np.zeros(len(parent_factor), bool)
            external_origin = np.zeros(len(parent_factor), bool)
            for t in np.unique(target_tile):
                rows = np.flatnonzero(target_tile == t)
                pos = xyz(meta["RA"][rows], meta["Dec"][rows])
                ap, af, external_anchor = anchor[t]
                dist, nearest = cKDTree(ap).query(pos, workers=1)
                matched = dist <= radius
                parent_factor[rows] = np.where(matched, af[nearest], fallback[rows])
                distance[rows] = dist
                fresh[rows] = ~matched
                external_origin[rows] = ~matched | external_anchor[nearest]
                if (~matched).any():
                    anchor[t] = (
                        np.concatenate((ap, pos[~matched])),
                        np.r_[af, fallback[rows[~matched]]],
                        np.r_[external_anchor, np.ones((~matched).sum(), bool)],
                    )
            density = parent_factor[meta["pre"]]
            w = table_weights(gm, table)
            policies = dict(
                original=(w, None),
                occupancy_matched=({b: v * density for b, v in w.items()}, None),
            )
            # Sensitivity to the tiny population without any ns-external anchor.
            neutral_density = np.where(external_origin[meta["pre"]], 1.0, density)
            policies["external_only_neutral_check"] = (
                {b: v * neutral_density for b, v in w.items()},
                None,
            )
            # Diagnostic only: quantify the consequence of re-evaluating density
            # at each external branch's ns SNR instead of freezing object factors.
            policies["external_ns_lookup_diagnostic"] = (
                {b: v * fallback[meta["pre"]] for b, v in w.items()},
                None,
            )
            for i, limits in enumerate(ranges):
                policies[f"tercile_{i + 1}"] = (w, limits)
            for policy, (weights, limits) in policies.items():
                point, jk, labels, n = measure(gm, tiles, weights, limits)
                if all_labels is None:
                    all_labels = labels
                np.testing.assert_array_equal(labels, all_labels)
                means.setdefault(policy, {})[name] = point
                deletes.setdefault(policy, {})[name] = jk
                counts.setdefault(policy, {})[name] = n
            selected_raw = meta["pre"][gm.mask_dict["ns"]]
            transfers[name] = dict(
                new_anchors=int(fresh.sum()),
                parent=len(fresh),
                selected_new_anchors=int(fresh[selected_raw].sum()),
                selected=len(selected_raw),
                selected_external_origin_DES_weight_fraction=float(
                    w["ns"][gm.mask_dict["ns"]][external_origin[selected_raw]].sum()
                    / w["ns"][gm.mask_dict["ns"]].sum()
                ),
                selected_new_anchor_DES_weight_fraction=float(
                    w["ns"][gm.mask_dict["ns"]][fresh[selected_raw]].sum()
                    / w["ns"][gm.mask_dict["ns"]].sum()
                ),
                selected_match_99_percentile_arcsec=float(
                    np.rad2deg(
                        2 * np.arcsin(np.quantile(distance[selected_raw], 0.99) / 2)
                    )
                    * 3600
                ),
            )
            print("loaded", grid, name, transfers[name], flush=True)
        row = {
            k: package(
                mc(v, cfg["shear_amplitude"]),
                mc(deletes[k], cfg["shear_amplitude"]),
                all_labels,
            )
            for k, v in means.items()
        }
        for k in ("occupancy_matched", "external_ns_lookup_diagnostic"):
            row[k + "_minus_original"] = package(
                np.array(row[k]["value"]) - row["original"]["value"],
                np.array(row[k]["deleted"]) - row["original"]["deleted"],
                all_labels,
            )
        row["external_only_neutral_minus_matched"] = package(
            np.array(row["external_only_neutral_check"]["value"])
            - row["occupancy_matched"]["value"],
            np.array(row["external_only_neutral_check"]["deleted"])
            - row["occupancy_matched"]["deleted"],
            all_labels,
        )
        # Within-grid paired tercile contrast, including cross-tercile covariance.
        row["high_minus_low_tercile"] = package(
            np.array(row["tercile_3"]["value"]) - row["tercile_1"]["value"],
            np.array(row["tercile_3"]["deleted"]) - row["tercile_1"]["deleted"],
            all_labels,
        )
        prod = json.loads((production / f"production_{grid}.json").read_text())
        p = prod["weights"]["w_des"]
        np.testing.assert_array_equal(all_labels, prod["jackknife_tiles"])
        np.testing.assert_allclose(
            row["original"]["value"],
            [p[k] for k in result["order"]],
            atol=1e-12,
            rtol=0,
        )
        np.testing.assert_allclose(
            row["original"]["deleted"], p["mc_jackknife"], atol=1e-12, rtol=0
        )
        row["counts"] = counts
        row["anchor_transfer"] = transfers
        result["grids"][str(grid)] = row
        print(
            "GRID",
            grid,
            "matched",
            row["occupancy_matched"]["value"],
            "delta",
            row["occupancy_matched_minus_original"]["value"],
            flush=True,
        )
    keys = list(means) + [
        "occupancy_matched_minus_original",
        "external_ns_lookup_diagnostic_minus_original",
        "external_only_neutral_minus_matched",
        "high_minus_low_tercile",
    ]
    result["means"] = {
        k: combine([r[k] for r in result["grids"].values()]) for k in keys
    }
    # Shift of original-weighted grid mean: same grid weights for both policies,
    # with the covariance of paired deletions, not independent mean errors.
    a = np.array(result["means"]["original"]["weights"])
    ds = [r["occupancy_matched_minus_original"] for r in result["grids"].values()]
    value = np.sum(a * np.array([r["value"] for r in ds]), axis=0)
    cov = sum(
        np.array(r["covariance"]) * a[i, :, None] * a[i, None, :]
        for i, r in enumerate(ds)
    )
    result["means"]["shift_at_original_grid_weights"] = dict(
        value=value.tolist(),
        error=np.sqrt(np.diag(cov)).tolist(),
        covariance=cov.tolist(),
    )
    result["combination_note"] = (
        "Component-wise inverse variance, assuming independent grids. Paired differences combined using their own inverse variances; also save shift at common original grid weights. W and occupancy factors held fixed in JK; no data-table or density-estimation uncertainty included."
    )
    previous["residual_m_occupancy"] = result
    path.write_text(json.dumps(previous, indent=2))
    (output / "residual_m_occupancy.json").write_text(json.dumps(result, indent=2))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for j, component in enumerate((0, 2)):
        ax = axes[0, j]
        for k, label, offset in (
            ("original", "Original", -0.07),
            ("occupancy_matched", "Data occupancy", 0.07),
        ):
            vals = [r[k]["value"][component] * 1000 for r in result["grids"].values()]
            err = [r[k]["error"][component] * 1000 for r in result["grids"].values()]
            ax.errorbar(
                np.arange(1, 5) + offset,
                vals,
                yerr=err,
                fmt="o",
                capsize=3,
                label=label,
            )
        ax.axhline(0, color="gray", ls="--")
        ax.set(xlabel="Grid", ylabel=f"m{j + 1} × 1000", xticks=range(1, 5))
        ax.legend()
        ax = axes[1, j]
        for grid, row in result["grids"].items():
            vals = [row[f"tercile_{i}"]["value"][component] * 1000 for i in (1, 2, 3)]
            err = [row[f"tercile_{i}"]["error"][component] * 1000 for i in (1, 2, 3)]
            ax.errorbar(
                np.arange(1, 4) + (int(grid) - 2.5) * 0.035,
                vals,
                yerr=err,
                fmt="o-",
                capsize=2,
                alpha=0.65,
                label=f"Grid {grid}",
            )
        vals = [
            result["means"][f"tercile_{i}"]["value"][component] * 1000
            for i in (1, 2, 3)
        ]
        err = [
            result["means"][f"tercile_{i}"]["error"][component] * 1000
            for i in (1, 2, 3)
        ]
        ax.errorbar(
            range(1, 4),
            vals,
            yerr=err,
            fmt="ks-",
            capsize=4,
            lw=2,
            label="IV grid mean",
        )
        ax.axhline(0, color="gray", ls="--")
        ax.set(
            xlabel="SNR tercile (data-occupancy boundaries)",
            ylabel=f"m{j + 1} × 1000",
            xticks=range(1, 4),
        )
        ax.legend(fontsize=8)
    fig.suptitle("Fixed v2 DES table; full populations; shared-tile jackknife")
    fig.savefig(output / "residual_m_occupancy.png", dpi=180)
    plt.close(fig)
    for grid, row in result["grids"].items():
        print("\nRESULT GRID", grid)
        for k in keys:
            print(
                k,
                "value",
                np.round(np.array(row[k]["value"]) * 1000, 4),
                "err",
                np.round(np.array(row[k]["error"]) * 1000, 4),
            )
    print("\nMEANS")
    for k, v in result["means"].items():
        print(
            k,
            "value",
            np.round(np.array(v["value"]) * 1000, 4),
            "err",
            np.round(np.array(v["error"]) * 1000, 4),
        )
    print("SNR tercile boundaries", edges)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    inputs.add_arguments(p)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--production", type=Path, required=True)
    args = p.parse_args()
    inputs.configure(args)
    run(args.output, args.production)
