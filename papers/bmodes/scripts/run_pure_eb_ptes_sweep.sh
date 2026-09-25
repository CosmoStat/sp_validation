#!/usr/bin/env bash
# Pure E/B PTE-matrix sweep over non-fiducial versions (Design B, Tier-2).
#
# Loops the non-fiducial version list (resolved via sweep_versions.py) and runs
# calculate_pure_eb_ptes.py once per version — the same χ² PTE-matrix compute the
# fiducial pure_eb_pte_per_cut recipe calls. Each version's gathered semi-analytic
# NPZ (data vectors + MC covariance) is read from the pure_eb_sweep output dir by
# absolute path — lc does not wire cross-output deps, so the driver reads the
# upstream sweep directly; run pure_eb_sweep before this. Each version emits the
# canonical ``{ver}_{blind}_pure_eb_ptes.npz`` — the exact name
# config_space_pte_matrices.py reconstructs from --pte-intermediate-dir — straight
# into --out. Serial over versions; each version is fast (206-pair grid, ~seconds).
#
# Usage:
#   run_pure_eb_ptes_sweep.sh --config <config.yaml> --cat-config <cat_config.yaml> \
#     --pure-eb-sweep-dir <pure_eb/.../pure_eb_sweep> \
#     --out <output_dir> [--blind A] [--versions "v1 v2 ..."]
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/container_env.sh"

CONFIG=""; CATCONFIG=""; PUREEBSWEEP=""; OUT=""; BLIND="A"; VERSIONS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --cat-config) CATCONFIG="$2"; shift 2;;
    --pure-eb-sweep-dir) PUREEBSWEEP="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --blind) BLIND="$2"; shift 2;;
    --versions) VERSIONS="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

mkdir -p "$OUT"

VERSIONS=$(sweep_versions "$CONFIG")

for ver in $VERSIONS; do
  pureeb="$PUREEBSWEEP/${ver}_${BLIND}_pure_eb_semianalytic.npz"
  [ -f "$pureeb" ] || { echo "MISSING upstream input for $ver: $pureeb" >&2; exit 1; }
  echo "[pure_eb_ptes_sweep] $ver"
  spv_python "$PSCRIPTS/calculate_pure_eb_ptes.py" \
    --version "$ver" --blind "$BLIND" \
    --pure-eb-data "$pureeb" --n-samples 2000 --out "$OUT"
  echo "[pure_eb_ptes_sweep] $ver -> $OUT/${ver}_${BLIND}_pure_eb_ptes.npz"
done
echo "[pure_eb_ptes_sweep] done -> $OUT"
