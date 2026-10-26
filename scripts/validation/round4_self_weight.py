"""Self-weighting toy and independent-table case-d grid tests; allocated node only."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits
from round2_pairs import PUBLIC
from round2_weights import BASE, EXTERNAL, calibrated_mean, covariance, load
from round3_sparse import training

from sp_validation import response


def toy(output):
    """Antithetic external shears, independent cells, unit response, no selection.

    Every trial has 400 equally occupied cells. Artificial metacal shifts leave
    cell membership fixed, hence split response is exactly the identity.
    """
    rng = np.random.default_rng(42504)
    result = {}
    for n in (3, 10, 30, 100, 300, 1000):
        draws = []
        for _ in range(200):
            e = rng.normal(0, 0.25, (2, 400, n))
            independent = rng.normal(0, 0.25, e.shape)
            estimates = []
            for source in (e, independent):
                means = []
                for g in (-0.025, 0.025):
                    train = source.copy()
                    train[0] += g
                    w = 1 / np.mean(train**2, axis=(0, 2))
                    means.append(np.average(e[0].mean(axis=1) + g, weights=w))
                estimates.append((means[1] - means[0]) / 0.05 - 1)
            draws.append(estimates)
        draws = np.array(draws)
        result[str(n)] = dict(
            self_m=float(draws[:, 0].mean()),
            external_m=float(draws[:, 1].mean()),
            sem=(draws.std(axis=0, ddof=1) / np.sqrt(len(draws))).tolist(),
            trials=len(draws),
            cells=400,
        )
        print("toy", n, result[str(n)], flush=True)
    (output / "toy.json").write_text(json.dumps(result, indent=2))


def public_table(output):
    """Recover actual released cell values, testing both PSF-size definitions."""
    es, er = np.geomspace(10, 500, 21), np.geomspace(0.707, 3, 21)
    summaries = {}
    with fits.open(PUBLIC, memmap=True) as f:
        print("public extensions", [h.name for h in f], flush=True)
        print("public columns", f[1].columns.names, flush=True)
        for size_col in ("NGMIX_Tpsf_NOSHEAR", "NGMIX_T_PSF_RECONV_NOSHEAR"):
            lo, hi = np.full(400, np.inf), np.full(400, -np.inf)
            count, sums = np.zeros(400), np.zeros(400)
            if size_col not in f[1].columns.names:
                continue
            for start in range(0, len(f[1].data), 500000):
                d = f[1].data[start : start + 500000]
                idx = response.cell_index(d["snr"], es) * 20 + response.cell_index(
                    d["NGMIX_T_NOSHEAR"] / d[size_col], er
                )
                w = np.asarray(d["w_des"], float)
                np.minimum.at(lo, idx, w)
                np.maximum.at(hi, idx, w)
                count += np.bincount(idx, minlength=400)
                sums += np.bincount(idx, weights=w, minlength=400)
            occupied = count > 0
            spread = (hi[occupied] - lo[occupied]) / hi[occupied]
            summaries[size_col] = dict(
                max_relative_spread=float(spread.max()),
                nonconstant_cells=int(np.sum(spread > 1e-6)),
            )
            print(size_col, summaries[size_col], flush=True)
            if spread.max() < 1e-6:
                values = np.divide(sums, count, out=np.zeros(400), where=count > 0)
                np.savez(
                    output / "public_table.npz",
                    edges_s=es,
                    edges_r=er,
                    values=values.reshape(20, 20),
                    count=count.reshape(20, 20),
                    size_column=size_col,
                )
    (output / "public_compatibility.json").write_text(json.dumps(summaries, indent=2))


def fit(samples, edges):
    return response.fit_w(
        *[
            np.concatenate([s[i] for s in samples], axis=1 if i == 2 else 0)
            for i in range(4)
        ],
        *edges,
        min_count=3,
    )


def weights(gm, table):
    return {
        b: response.evaluate_w(
            table,
            getattr(gm, b)["flux"] / getattr(gm, b)["flux_err"],
            getattr(gm, b)["T"] / getattr(gm, b)["Tpsf"],
        )
        for b in response.BRANCHES
    }


def grid_run(grid, output):
    with open(BASE / f"grid_{grid}/results/m_bias_config.yaml") as f:
        gin = yaml.safe_load(f)["shear_amplitude"]
    means, deletes = {}, {}
    public = None
    if (output / "public_table.npz").exists():
        with np.load(output / "public_table.npz") as d:
            if str(d["size_column"]) == "NGMIX_Tpsf_NOSHEAR":
                public = tuple(d[k] for k in ("edges_s", "edges_r", "values", "count"))
    for name in EXTERNAL:
        gm, tiles = load(grid, name, "des", min_count=3)
        own = training(gm)
        others = []
        for other in range(1, 5):
            if other != grid:
                other_gm, _ = load(other, name, "des", min_count=3)
                others.append(training(other_gm))
                del other_gm
        edges = gm.W[:2]
        tables = dict(a=gm.W, b=fit(others, edges), c=fit([own] + others, edges))
        if public is not None:
            tables["d"] = public
        for policy, table in tables.items():
            w = weights(gm, table)
            means.setdefault(policy, {})[name] = calibrated_mean(gm, w)
            if policy in ("b", "d"):
                labels, jk, _ = response.jackknife(
                    gm.branches, gm.mask_dict, w, gm.R_shear, gm._step, tiles
                )
                deleted = jk["c_g"]
            else:
                labels = np.unique(tiles)
                deleted = []
                for label in labels:
                    keep = tiles != label
                    selected = keep[gm.mask_dict["ns"]]
                    retained = [
                        v[:, selected] if i == 2 else v[selected]
                        for i, v in enumerate(own)
                    ]
                    sample = [retained] + (others if policy == "c" else [])
                    deleted.append(
                        calibrated_mean(gm, weights(gm, fit(sample, edges)), keep)
                    )
                deleted = np.array(deleted)
            deletes.setdefault(policy, {})[name] = deleted
            print(grid, name, policy, flush=True)

    def mc(values):
        out = []
        for i, (p, m) in enumerate((("1p2z", "1m2z"), ("1z2p", "1z2m"))):
            out.extend(
                [
                    (values[p][..., i] - values[m][..., i]) / (2 * gin) - 1,
                    (values[p][..., i] + values[m][..., i]) / 2,
                ]
            )
        return np.array(out).T

    result = {}
    for policy in means:
        value, jk = mc(means[policy]), mc(deletes[policy])
        cov = covariance(jk)
        result[policy] = dict(
            value=value.tolist(),
            error=np.sqrt(np.diag(cov)).tolist(),
            covariance=cov.tolist(),
            deleted=jk.tolist(),
            labels=labels.tolist(),
        )
    diff = np.array(result["a"]["deleted"]) - np.array(result["b"]["deleted"])
    result["a_minus_b"] = dict(
        value=(
            np.array(result["a"]["value"]) - np.array(result["b"]["value"])
        ).tolist(),
        error=np.sqrt(np.diag(covariance(diff))).tolist(),
        covariance=covariance(diff).tolist(),
    )
    (output / f"grid_{grid}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task", choices=["toy", "public", "1", "2", "3", "4"], required=True
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.task == "toy":
        toy(a.output)
    elif a.task == "public":
        public_table(a.output)
    else:
        grid_run(int(a.task), a.output)
