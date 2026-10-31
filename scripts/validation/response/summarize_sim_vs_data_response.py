"""Print direct-response diagnostics from sim_vs_data_response.py (allocated node)."""

import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
d = json.loads((root / "results.json").read_text())
es, er = np.array(d["edges_snr"]), np.array(d["edges_size"])
pd = np.array(d["data_count"], float)
pd /= pd.sum()
print("Data marginals: fractions in first/last five bins, SNR then size")
print(
    pd.sum(1)[:5].sum(), pd.sum(1)[-5:].sum(), pd.sum(0)[:5].sum(), pd.sum(0)[-5:].sum()
)
for grid, g in d["grids"].items():
    print("\nGrid", grid, "N", g["selected"])
    print("policy  R_shear11/22  R_sel11/22  means(shear,selection)")
    for policy, r in g["responses"].items():
        print(
            policy,
            np.round(r["shear"]["diagonal"], 6),
            np.round(r["selection"]["diagonal"], 6),
            round(r["shear"]["mean_diagonal"], 6),
            round(r["selection"]["mean_diagonal"], 6),
        )
    print("summaries", g["summaries"])
    ps = np.array(g["sim_fraction"])
    print(
        "Sim marginals first/last five bins",
        ps.sum(1)[:5].sum(),
        ps.sum(1)[-5:].sum(),
        ps.sum(0)[:5].sum(),
        ps.sum(0)[-5:].sum(),
    )
    print("R_snr", np.round(g["mean_R_snr"], 3))
    print("R_size", np.round(g["mean_R_size"], 3))
    print(
        "matched mean responses",
        {
            k: [round(v[z]["mean_diagonal"], 6) for z in ("shear", "selection")]
            for k, v in g["occupancy_matched_responses"].items()
        },
    )
print("Bin5 edges", es[5], er[5], "last5", es[-6], er[-6])
