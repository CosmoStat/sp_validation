"""Summarize external-table production measurements; run on allocated nodes."""

import json
from pathlib import Path

import numpy as np
from round2_weights import covariance
from v2_data_table import OUT, PRODUCT

DOC = Path("docs/validation/response")
rows = {}
for g in range(1, 5):
    prod = json.loads((OUT / f"production_{g}.json").read_text())
    p = prod["weights"]["w_des"]
    b = json.loads((DOC / f"independent_grid_{g}.json").read_text())["b"]
    labels = prod["jackknife_tiles"]
    np.testing.assert_array_equal(labels, b["labels"])
    value = np.array([p[k] for k in ("m1", "c1", "m2", "c2")])
    deleted = np.array(p["mc_jackknife"])
    delta = deleted - np.array(b["deleted"])
    rows[str(g)] = dict(
        value=value.tolist(),
        error=[p[k + "_err"] for k in ("m1", "c1", "m2", "c2")],
        covariance=p["mc_covariance"],
        deleted=deleted.tolist(),
        labels=labels,
        b=b,
        difference=dict(
            value=(value - np.array(b["value"])).tolist(),
            error=np.sqrt(np.diag(covariance(delta))).tolist(),
            covariance=covariance(delta).tolist(),
            deleted=delta.tolist(),
        ),
        clipping=json.loads((OUT / f"clipping_{g}.json").read_text()),
    )
    print(
        g,
        "d",
        value[[0, 2]] * 1000,
        np.array(rows[str(g)]["error"])[[0, 2]] * 1000,
        "diff",
        np.array(rows[str(g)]["difference"]["value"])[[0, 2]] * 1000,
        np.array(rows[str(g)]["difference"]["error"])[[0, 2]] * 1000,
    )
means = {}
for policy in ("d", "b", "difference"):
    vals = np.array(
        [r["value"] if policy == "d" else r[policy]["value"] for r in rows.values()]
    )
    errors = np.array(
        [r["error"] if policy == "d" else r[policy]["error"] for r in rows.values()]
    )
    w = errors**-2
    means[policy] = dict(
        value=(np.sum(w * vals, axis=0) / w.sum(axis=0)).tolist(),
        error=np.sqrt(1 / w.sum(axis=0)).tolist(),
    )
    print(policy, means[policy])
for g, r in rows.items():
    c = r["clipping"]
    print(
        "clip",
        g,
        "selected",
        sum(x["outside"] for a in c.values() for x in a.values()),
        "parent",
        sum(x["parent_outside"] for a in c.values() for x in a.values())
        / sum(x["parent"] for a in c.values() for x in a.values()),
    )
(DOC / "v2_data_table.json").write_text(
    json.dumps(
        dict(
            product=PRODUCT,
            calibration_commit="d533a078ed6998683e92e6f24191d0baa8419fa9",
            sha256="1e551a3645e98d5c3037f2d2175cfa1c7675b8d804e19ca13bf4f178de3ff016",
            grids=rows,
            means=means,
        ),
        indent=2,
    )
)
