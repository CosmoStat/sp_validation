"""Sparse DES table occupancy, sampling noise and fixed/refit-W threshold ladder.

Unblinded simulations and public v1.4.6.3 only; run on an allocated node.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits
from round2_pairs import PUBLIC
from round2_weights import BASE, EXTERNAL, calibrated_mean, covariance, load

from sp_validation import response

LADDER = (0, 3, 5, 10, 30, 100, 300, 1000)


def occupancy(count, weight_sum):
    count, weight_sum = count.ravel(), weight_sum.ravel()
    return dict(
        count=count.reshape(20, 20).tolist(),
        total=int(count.sum()),
        occupied=int(np.count_nonzero(count)),
        quantiles=np.quantile(count[count > 0], [0, 0.1, 0.5, 0.9, 1]).tolist(),
        thresholds={
            str(n): dict(
                cells=int(np.sum(count < n)),
                objects=int(count[count < n].sum()),
                object_fraction=float(count[count < n].sum() / count.sum()),
                weight_fraction=float(weight_sum[count < n].sum() / weight_sum.sum()),
            )
            for n in LADDER
        },
    )


def training(gm):
    ma = gm.mask_dict["ns"]
    snr = (gm.ns["flux"] / gm.ns["flux_err"])[ma]
    size = (gm.ns["T"] / gm.ns["Tpsf"])[ma]
    e = np.asarray(gm.branches["ns"])[:, ma]
    r = 0.5 * (gm.R11 + gm.R22)
    return snr, size, e, r


def sampling_noise(gm):
    """Within-cell repeated subsamples, excluding genuinely sparse parent cells.

    Hold parent cell fixed, draw objects with replacement at each count, and
    measure SD(W)/mean(W). This includes the nonlinear ratio's small-N tail.
    """
    snr, size, e, r = training(gm)
    idx = response.cell_index(snr, gm.W[0]) * 20 + response.cell_index(size, gm.W[1])
    y = np.mean(e**2, axis=0)
    rng = np.random.default_rng(425)
    result = {}
    cells = [np.flatnonzero(idx == k) for k in range(400)]
    cells = [rows for rows in cells if len(rows) >= 100]
    if not cells:
        raise ValueError("No parent cells with at least 100 objects")
    for n in (2, 3, 10, 30, 100, 300, 1000):
        frac = []
        for rows in cells:
            draws = rows[rng.integers(len(rows), size=(300, n))]
            w = np.mean(r[draws], axis=1) ** 2 / np.mean(y[draws], axis=1)
            frac.append(float(w.std(ddof=1) / w.mean()))
        result[str(n)] = dict(
            cells=len(cells),
            median=float(np.median(frac)),
            p90=float(np.quantile(frac, 0.9)),
        )
    return result


def grid_run(grid, output):
    gm0, _ = load(grid, "1z2z", "des", min_count=0)
    occ = {}
    count = gm0.W[3]
    occ["1z2z"] = occupancy(count, count * gm0.W[2])
    noise = sampling_noise(gm0)
    fragility = {}
    del gm0
    with open(BASE / f"grid_{grid}/results/m_bias_config.yaml") as f:
        gin = yaml.safe_load(f)["shear_amplitude"]
    means = {n: {} for n in LADDER}
    deletes = {n: {"fixed": {}, "refit": {}} for n in LADDER}
    for name in EXTERNAL:
        gm, tiles = load(grid, name, "des", min_count=0)
        occ[name] = occupancy(gm.W[3], gm.W[3] * gm.W[2])
        for n in LADDER:
            gm._des_min_count = n
            gm._prepare_weights()
            if grid == 4 and name == "1z2m":
                snr, size, e, r = training(gm)
                keep = tiles[gm.mask_dict["ns"]] != "275.282"
                table = response.fit_w(
                    snr[keep],
                    size[keep],
                    e[:, keep],
                    r[keep],
                    gm.W[0],
                    gm.W[1],
                    min_count=n,
                )
                if n == LADDER[0]:
                    raw = response.fit_w(
                        snr[keep], size[keep], e[:, keep], r[keep], gm.W[0], gm.W[1]
                    )
                    cell = np.unravel_index(np.argmax(raw[2]), raw[2].shape)
                fragility[str(n)] = dict(
                    cell=list(map(int, cell)),
                    full_count=int(gm.W[3][cell]),
                    deleted_count=int(table[3][cell]),
                    full_w=float(gm.W[2][cell]),
                    deleted_w=float(table[2][cell]),
                )
            means[n][name] = calibrated_mean(gm, gm.branch_weights)
            for policy in ("fixed", "refit"):
                labels, jk, _ = gm.response_jackknife(tiles, refit_w=policy == "refit")
                deletes[n][policy][name] = jk["c_g"]
            print(grid, name, n, flush=True)

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

    results = {}
    for n in LADDER:
        results[str(n)] = dict(value=mc(means[n]).tolist())
        for policy in ("fixed", "refit"):
            jk = mc(deletes[n][policy])
            cov = covariance(jk)
            results[str(n)][policy] = dict(
                error=np.sqrt(np.diag(cov)).tolist(),
                covariance=cov.tolist(),
                deleted=jk.tolist(),
                labels=labels.tolist(),
            )
    result = dict(
        occupancy=occ, sampling_noise=noise, ladder=results, grid4_deletion=fragility
    )
    (output / f"sparse_grid_{grid}.json").write_text(json.dumps(result, indent=2))


def public_run(output):
    # Match the grids' fixed 20x20 log edges; public selection is the released cut.
    es, er = np.geomspace(10, 500, 21), np.geomspace(0.707, 3, 21)
    sums = np.zeros((4, 400))  # count, sum r, sum |e|^2/2, released weight
    with fits.open(PUBLIC, memmap=True) as f:
        for start in range(0, len(f[1].data), 500000):
            d = f[1].data[start : start + 500000]
            snr = np.asarray(d["snr"], float)
            size = d["NGMIX_T_NOSHEAR"] / d["NGMIX_Tpsf_NOSHEAR"]
            r = 0.5 * (d["R_g11"] + d["R_g22"])
            y = 0.5 * (d["e1_uncal"] ** 2 + d["e2_uncal"] ** 2)
            assert np.all(np.isfinite(snr + size + r + y))
            idx = response.cell_index(snr, es) * 20 + response.cell_index(size, er)
            for i, v in enumerate((None, r, y, d["w_des"])):
                sums[i] += np.bincount(idx, weights=v, minlength=400)
    count, rs, ys, released = sums
    w = np.divide(rs**2, count * ys, out=np.zeros(400), where=count * ys > 0)
    result = dict(
        edges_s=es.tolist(),
        edges_r=er.tolist(),
        reconstructed=occupancy(count, count * w),
        released=occupancy(count, released),
    )
    (output / "sparse_public.json").write_text(json.dumps(result, indent=2))
    print("public", result["reconstructed"]["quantiles"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", required=True, choices=["public", "1", "2", "3", "4"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--counts", type=int, nargs="+", default=LADDER)
    a = p.parse_args()
    LADDER = tuple(a.counts)
    a.output.mkdir(parents=True, exist_ok=True)
    if a.sample == "public":
        public_run(a.output)
    else:
        grid_run(int(a.sample), a.output)
