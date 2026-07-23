"""Emit a flat harmonic-vs-config COSEBI PTE comparison table.

Numbers quoted in the B-modes paper compare the COSEBIs B-mode PTE computed
two ways — harmonic space (pseudo-Cℓ bandpower fit) and config space
(real-space COSEBIs coefficients) — across catalog version and angular-range
cut. Those values live inside nested-dict `evidence.json` carriers belonging
to `type: figure` / `type: data` ASTRA outputs
(``null_tests.config_space_pte_evidence``, ``cosebis.cosebis_harmonic_modes``,
and ``cosebis.cosebis_harmonic_modes_full``). MySTRA's ``{astra:value col=
where=}`` role can only read a flat CSV/JSON table, not a nested dict — see
``work/transpile-map.md`` §0 in the ASTRA reproduction. This script reads
those *already-materialized* evidence.json files (it does not recompute
anything) and writes the requested cells out as a flat CSV table, one row per
(version, path, cut) triple.

Config-space PTEs are computed empirically (bootstrap resampling), so they
carry no chi2/dof — those columns are left blank for `path=config` rows.

    python emit_cosebis_harmonic_config.py \
        --config-space-evidence /path/to/config_space_pte_evidence/evidence.json \
        --cosebis-evidence /path/to/cosebis_harmonic_modes/evidence.json \
        --cosebis-evidence-full /path/to/cosebis_harmonic_modes_full/evidence.json \
        --out <output_dir>
"""

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def _sig3(x):
    """Format a float to 3 significant figures; scientific for |x|<1e-3 or >=1e4, else fixed."""
    if x == "" or x is None:
        return ""
    x = float(x)
    if x == 0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    if e < -3 or e >= 4:
        return f"{x:.2e}"  # mantissa carries 3 sig figs, e.g. 7.37e-10
    return f"{x:.{max(0, 2 - e)}f}"  # 3 sig figs fixed, e.g. 0.938, 0.00461


# Full leak-corrected version string -> CSV slug, in the paper's version order.
_VERSION_SLUGS = {
    "SP_v1.4.5_leak_corr": "initial",
    "SP_v1.4.6.3_leak_corr": "size_cuts",
    "SP_v1.4.8_leak_corr": "masked",
    "SP_v1.4.11.3_leak_corr": "relaxed_flags",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-space-evidence", required=True)
    parser.add_argument("--cosebis-evidence", required=True)
    parser.add_argument("--cosebis-evidence-full", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config_evidence = json.load(open(args.config_space_evidence))["evidence"]
    harmonic_fid_evidence = json.load(open(args.cosebis_evidence))["evidence"]
    harmonic_full_evidence = json.load(open(args.cosebis_evidence_full))["evidence"]

    config_versions = config_evidence.get("versions", {})
    harmonic_fid_ptes = harmonic_fid_evidence.get("harmonic_b_mode_ptes", {})
    harmonic_full_ptes = harmonic_full_evidence.get("harmonic_b_mode_ptes", {})

    rows = []
    for version_full, slug in _VERSION_SLUGS.items():
        # Harmonic space: fiducial and full-range come from separate evidence
        # carriers (--angular-range fiducial vs full).
        for cut, harm_ptes in [
            ("fiducial", harmonic_fid_ptes),
            ("full_range", harmonic_full_ptes),
        ]:
            harm = harm_ptes.get(version_full, {})
            rows.append(
                {
                    "version": slug,
                    "path": "harmonic",
                    "cut": cut,
                    "pte": _sig3(harm.get("pte")),
                    "chi2": _sig3(harm.get("chi2")),
                    "dof": harm.get("dof"),
                }
            )

        # Config space: single evidence carrier, both cuts as separate keys.
        cfg = config_versions.get(version_full, {}).get("cosebis_stats", {})
        for cut, pte_key in [
            ("fiducial", "pte_at_fiducial"),
            ("full_range", "pte_at_full_range"),
        ]:
            rows.append(
                {
                    "version": slug,
                    "path": "config",
                    "cut": cut,
                    "pte": _sig3(cfg.get(pte_key)),
                    "chi2": "",
                    "dof": "",
                }
            )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "cosebis_harmonic_config.csv"
    fieldnames = ["version", "path", "cut", "pte", "chi2", "dof"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    evidence = {
        "spec_id": "cosebis_harmonic_config",
        "spec_path": "papers/bmodes/scripts/emit_cosebis_harmonic_config.py",
        "generated": datetime.now(timezone.utc).isoformat(),
        "evidence": {
            "type": "value_promotion",
            "description": (
                "Flattens the harmonic-vs-config COSEBI B-mode PTE comparison "
                "(chi2/dof where available) out of the nested "
                "config_space_pte_evidence, cosebis_harmonic_modes, and "
                "cosebis_harmonic_modes_full evidence.json carriers into a "
                "queryable table. See work/transpile-map.md §0/§2d for the "
                "value-binding gap this closes."
            ),
            "source_evidence": {
                "config_space": args.config_space_evidence,
                "cosebis": args.cosebis_evidence,
                "cosebis_full": args.cosebis_evidence_full,
            },
            "rows_written": len(rows),
        },
        "output": {"table": "cosebis_harmonic_config.csv"},
    }
    with open(out_dir / "evidence.json", "w") as f:
        json.dump(evidence, f, indent=2)

    print(f"Wrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
