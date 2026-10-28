"""Measure and check total-response recombination closure on simulation grids."""

import argparse
from pathlib import Path

import inputs
import numpy as np
import yaml

from sp_validation import (
    calibration,
    catalog_builders,
    image_sim_config,
    io,
    response,
    sim_m_snr,
)


def measure(gm, tiles, edges):
    """Measure each bin and return its means, responses and deletions."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        masks = sim_m_snr.snr_masks(gm, lo, hi)
        pos = np.searchsorted(gm.mask_dict["ns"], masks["ns"])
        rg = gm.R_shear[:, :, pos]
        point = response.responses(gm.branches, masks, gm.branch_weights, rg, gm._step)
        labels, jk, _ = response.jackknife(
            gm.branches, masks, gm.branch_weights, rg, gm._step, tiles
        )
        ns = masks["ns"]
        w = gm.branch_weights["ns"][ns]
        _, groups = np.unique(tiles, return_inverse=True)
        tile_weights = np.bincount(groups[ns], weights=w, minlength=len(labels))
        rows.append((point, jk, w.sum(), tile_weights.sum() - tile_weights))
    return rows, labels


def run_grid(grid, output, edges):
    """Write response ingredients for each bin and the full SNR range."""
    cats = {}
    for name in inputs.EXTERNAL:
        directory = inputs.BASE / f"grid_{grid}" / f"{name}_grid_{grid}"
        cfg = yaml.safe_load((directory / "config_mask.yaml").read_text())
        cfg = image_sim_config.calibration_config(
            cfg, ["w_des"], source="external", product=inputs.PRODUCT
        )
        data = io.read_catalogue(directory / "shape_catalog_comprehensive_ngmix.fits")
        masks, _ = catalog_builders.get_masks_from_config(cfg, data, verbose=False)
        pre = catalog_builders.Mask.from_list(
            masks, label="combined", verbose=False
        )._mask
        gm = calibration.metacal_from_config(data, pre, cfg["metacal"])
        cats[name] = (gm, np.char.strip(np.asarray(data["TILE_ID"][pre]).astype(str)))
    cfg = yaml.safe_load(
        (inputs.BASE / f"grid_{grid}/results/m_bias_config.yaml").read_text()
    )
    amplitude = cfg["shear_amplitude"]
    output.mkdir(parents=True, exist_ok=True)
    for label, boundaries in (("default", edges), ("full", [edges[0], edges[-1]])):
        results = {}
        labels = None
        for name, (gm, tiles) in cats.items():
            results[name], current = measure(gm, tiles, boundaries)
            if labels is not None and not np.array_equal(labels, current):
                raise ValueError("External branches must share jackknife tiles")
            labels = current
        value, deleted = [], []
        for b in range(len(boundaries) - 1):
            v, d = [], []
            for c, (plus, minus) in enumerate(sim_m_snr.PAIRS):
                a, z = results[plus][b], results[minus][b]
                v.extend(
                    (
                        (a[0]["c_g"][c] - z[0]["c_g"][c]) / (2 * amplitude) - 1,
                        (a[0]["c_g"][c] + z[0]["c_g"][c]) / 2,
                    )
                )
                d.extend(
                    (
                        (a[1]["c_g"][:, c] - z[1]["c_g"][:, c]) / (2 * amplitude) - 1,
                        (a[1]["c_g"][:, c] + z[1]["c_g"][:, c]) / 2,
                    )
                )
            value.append(v)
            deleted.append(np.stack(d, axis=1))
        deleted = np.stack(deleted, axis=1)
        delta = deleted.reshape(len(labels), -1).copy()
        delta -= delta.mean(axis=0)
        out = dict(
            edges=boundaries,
            labels=labels,
            value=np.array(value),
            deleted=deleted,
            covariance=(len(labels) - 1) / len(labels) * delta.T @ delta,
        )
        for field in ("shear", "selection", "total"):
            out[field] = np.array(
                [[row[0][field] for row in results[name]] for name in inputs.EXTERNAL]
            )
            out[field + "_jk"] = np.array(
                [[row[1][field] for row in results[name]] for name in inputs.EXTERNAL]
            )
        out["sum_w"] = np.array(
            [[row[2] for row in results[name]] for name in inputs.EXTERNAL]
        )
        out["sum_w_jk"] = np.array(
            [[row[3] for row in results[name]] for name in inputs.EXTERNAL]
        )
        np.savez_compressed(output / f"review_{label}_{grid}.npz", **out)
        print(grid, label, flush=True)


def check(output):
    """Compare total-share bin m with direct full m using paired deletions."""
    rows, full = [], []
    for g in range(1, 5):
        with np.load(output / f"review_default_{g}.npz") as f:
            rows.append({k: f[k].copy() for k in f.files})
        with np.load(output / f"review_full_{g}.npz") as f:
            full.append({k: f[k].copy() for k in f.files})
    fw = sim_m_snr.combine_grids(full)["grid_weights"][:, [0, 2]]
    differences, cov = [], np.zeros((2, 2))
    for i, (p, f) in enumerate(zip(rows, full, strict=True)):
        if not np.array_equal(p["labels"], f["labels"]):
            raise ValueError("Mismatched deletion tiles")
        delta = []
        draws = []
        for c, (ip, im) in enumerate(((0, 1), (2, 3))):
            h = (p["sum_w"][[ip, im]] * p["total"][[ip, im], :, c, c]).mean(axis=0)
            hj = (
                (p["sum_w_jk"][[ip, im]] * p["total_jk"][[ip, im], :, :, c, c])
                .mean(axis=0)
                .T
            )
            h /= h.sum()
            hj /= hj.sum(axis=1)[:, None]
            delta.append(h @ p["value"][:, 2 * c] - f["value"][0, 2 * c])
            draws.append(
                np.sum(hj * p["deleted"][:, :, 2 * c], axis=1)
                - f["deleted"][:, 0, 2 * c]
            )
        differences.append(delta)
        draw = np.stack(draws, axis=1) * fw[i]
        draw -= draw.mean(axis=0)
        cov += (len(draw) - 1) / len(draw) * draw.T @ draw
    print(
        "total-share minus direct full m × 1000:",
        np.sum(np.array(differences) * fw, axis=0) * 1000,
    )
    print("paired shared-tile errors × 1000:", np.sqrt(np.diag(cov)) * 1000)
    print("Diagonal composition is approximate; this is a closure diagnostic.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    inputs.add_arguments(p)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--grid", type=int)
    p.add_argument(
        "--edges", type=float, nargs="+", default=sim_m_snr.DEFAULT_SNR_EDGES
    )
    args = p.parse_args()
    inputs.configure(args)
    if args.grid is None:
        check(args.output)
    else:
        run_grid(args.grid, args.output, args.edges)
