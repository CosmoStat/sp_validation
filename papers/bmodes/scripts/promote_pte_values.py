"""Promote select headline PTE values from nested evidence.json carriers into
a flat, {astra:value}-queryable CSV table.

Six PTE numbers quoted in the B-modes paper prose (footnote, Results, and
Discussion sections) live inside nested-dict `evidence.json` files belonging
to `type: figure` / `type: data` ASTRA outputs
(``null_tests.config_space_pte_evidence`` and
``cosebis.cosebis_harmonic_modes``). MySTRA's ``{astra:value col= where=}``
role can only read a flat CSV/JSON table, not a nested dict — see
``work/transpile-map.md`` §0 in the ASTRA reproduction. This script reads
those two *already-materialized* evidence.json files (it does not recompute
anything) and writes the requested cells out as a small CSV table with one
row per macro.

A sixth macro (``harmCosebisPteSixThreeFull``, the harmonic-space full-range
COSEBI PTE) has no local carrier at all: the root `fig_harmonic_config_cosebis`
recipe only runs `--angular-range fiducial`, so no evidence.json anywhere in
the reproduction tree contains that number. It is intentionally NOT written
here — promoting a value that was not read from a materialized artifact would
defeat the point of the exercise. The row schema below already carries a
`cut` column so that value can slot in as an additional row once the
harmonic full-range COSEBI materialization lands (a separate, larger compute
task), without any restructuring of this table or its consumers.

    python promote_pte_values.py \
        --config-space-evidence /path/to/config_space_pte_evidence/evidence.json \
        --cosebis-evidence /path/to/cosebis_harmonic_modes/evidence.json \
        --out <output_dir>
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

# (macro, path-spec) — path-spec is a tuple of keys to walk into the source
# evidence dict. `source` selects which --*-evidence file to read from.
ROWS = [
    {
        "macro": "configPteSixThreeCombined",
        "source": "config_space",
        "version": "SP_v1.4.6.3_leak_corr",
        "statistic": "combined",
        "cut": "fiducial",
        "path": (
            "versions",
            "SP_v1.4.6.3_leak_corr",
            "combined_stats",
            "pte_at_fiducial",
        ),
    },
    {
        "macro": "configPteEightXim",
        "source": "config_space",
        "version": "SP_v1.4.8_leak_corr",
        "statistic": "xim",
        "cut": "fiducial",
        "path": ("versions", "SP_v1.4.8_leak_corr", "xim_stats", "pte_at_fiducial"),
    },
    {
        "macro": "harmCosebisPteSixThreeFid",
        "source": "cosebis",
        "version": "SP_v1.4.6.3_leak_corr",
        "statistic": "cosebis_harmonic",
        "cut": "fiducial",
        "path": ("harmonic_b_mode_ptes", "SP_v1.4.6.3_leak_corr", "pte"),
    },
    {
        "macro": "cfgCosebisPteSixThreeFid",
        "source": "cosebis",
        "version": "SP_v1.4.6.3_leak_corr",
        "statistic": "cosebis_config",
        "cut": "fiducial",
        "path": ("config_b_mode_ptes", "SP_v1.4.6.3_leak_corr", "pte"),
    },
    {
        "macro": "cfgCosebisPteSixThreeFull",
        "source": "config_space",
        "version": "SP_v1.4.6.3_leak_corr",
        "statistic": "cosebis_config",
        "cut": "full_range",
        "path": (
            "versions",
            "SP_v1.4.6.3_leak_corr",
            "cosebis_stats",
            "pte_at_full_range",
        ),
    },
]


def _walk(d, path):
    for key in path:
        d = d[key]
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-space-evidence", required=True)
    parser.add_argument("--cosebis-evidence", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sources = {
        "config_space": json.load(open(args.config_space_evidence))["evidence"],
        "cosebis": json.load(open(args.cosebis_evidence))["evidence"],
    }
    source_files = {
        "config_space": args.config_space_evidence,
        "cosebis": args.cosebis_evidence,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "pte_promoted_values.csv"
    fieldnames = ["macro", "version", "statistic", "cut", "pte", "source_evidence"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ROWS:
            pte = _walk(sources[row["source"]], row["path"])
            writer.writerow(
                {
                    "macro": row["macro"],
                    "version": row["version"],
                    "statistic": row["statistic"],
                    "cut": row["cut"],
                    "pte": pte,
                    "source_evidence": source_files[row["source"]],
                }
            )

    evidence = {
        "spec_id": "pte_promoted_values",
        "spec_path": "papers/bmodes/scripts/promote_pte_values.py",
        "generated": datetime.now(timezone.utc).isoformat(),
        "evidence": {
            "type": "value_promotion",
            "description": (
                "Flattens five headline paper PTE values out of the nested "
                "config_space_pte_evidence and cosebis_harmonic_modes "
                "evidence.json carriers into a queryable table. See "
                "work/transpile-map.md §0 for the value-binding gap this "
                "closes, and the module docstring for the sixth macro "
                "(harmCosebisPteSixThreeFull) intentionally left unpromoted."
            ),
            "source_evidence": source_files,
            "rows_written": len(ROWS),
        },
        "output": {"table": "pte_promoted_values.csv"},
    }
    with open(out_dir / "evidence.json", "w") as f:
        json.dump(evidence, f, indent=2)

    print(f"Wrote {csv_path} ({len(ROWS)} rows)")


if __name__ == "__main__":
    main()
