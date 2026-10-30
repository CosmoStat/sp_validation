"""Check all 20 case-d products against Martin's reference measurements."""

import argparse
import json
from pathlib import Path


def summarize(root, output):
    output.mkdir(parents=True, exist_ok=True)
    results = [
        json.loads((root / f"grid_{i}/validation.json").read_text())
        for i in range(1, 5)
    ]
    summary = dict(
        products=sum(len(r["branches"]) for r in results),
        max_abs_R_difference=max(
            b["max_abs_R_difference"] for r in results for b in r["branches"].values()
        ),
        max_relative_W_difference=max(
            b["max_relative_W_difference"]
            for r in results
            for b in r["branches"].values()
        ),
        max_paired_m_difference=max(
            abs(v["new_paired"] - v["martin_paired"])
            for r in results
            for v in r["m"].values()
        ),
    )
    assert summary["products"] == 20
    for key, value in summary.items():
        if key != "products":
            assert value <= 1e-14, (key, value)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summarize(args.root, args.output)
