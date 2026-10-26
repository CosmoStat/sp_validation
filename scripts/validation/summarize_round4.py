"""Summarize round-four tables and check the self-built round-three baseline."""

import argparse
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("root", type=Path)
p.add_argument("docs", type=Path)
a = p.parse_args()
checks = {}
lines = [
    "| Grid | Component | (a) Self | (b) Other three | (c) Pooled four | (d) Release |",
    "| --- | --- | ---: | ---: | ---: | --- |",
]
diff = ["| Grid | Δm1 (a−b) | Δm2 (a−b) |", "| --- | ---: | ---: |"]
for grid in range(1, 5):
    result = json.loads((a.root / f"grid_{grid}.json").read_text())
    baseline = json.loads((a.docs / f"sparse_grid_{grid}.json").read_text())["ladder"][
        "3"
    ]
    checks[str(grid)] = dict(
        value_max_abs=float(
            np.max(np.abs(np.array(result["a"]["value"]) - baseline["value"]))
        ),
        deleted_max_abs=float(
            np.max(
                np.abs(np.array(result["a"]["deleted"]) - baseline["refit"]["deleted"])
            )
        ),
    )
    assert checks[str(grid)]["value_max_abs"] < 1e-13
    assert checks[str(grid)]["deleted_max_abs"] < 1e-13
    (a.docs / f"independent_grid_{grid}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    for j, comp in ((0, "m1"), (2, "m2")):
        cells = [
            f"{result[k]['value'][j] * 1000:+.2f} ± {result[k]['error'][j] * 1000:.2f}"
            for k in ("a", "b", "c")
        ]
        lines.append(f"| {grid} | {comp} | " + " | ".join(cells) + " | Not verified |")
    v = result["a_minus_b"]
    diff.append(
        f"| {grid} | {v['value'][0] * 1000:+.2f} ± {v['error'][0] * 1000:.2f} | "
        f"{v['value'][2] * 1000:+.2f} ± {v['error'][2] * 1000:.2f} |"
    )
(a.root / "baseline_checks.json").write_text(json.dumps(checks, indent=2))
(a.root / "tables.md").write_text("\n".join(lines + [""] + diff) + "\n")
for name in ("toy", "public_compatibility"):
    (a.docs / f"self_weight_{name}.json").write_text(
        (a.root / f"{name}.json").read_text() + "\n"
    )
print("\n".join(lines + [""] + diff))
print("baseline checks", checks)
