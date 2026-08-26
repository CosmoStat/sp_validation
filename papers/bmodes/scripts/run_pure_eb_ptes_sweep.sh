#!/usr/bin/env bash
# Pure E/B PTE-matrix sweep over non-fiducial versions (Design B, Tier-2).
#
# Loops the non-fiducial version list (resolved via sweep_versions.py) and runs
# calculate_pure_eb_ptes.py once per version — the same χ² PTE-matrix compute the
# fiducial pure_eb_pte_per_cut recipe calls. Each version's gathered semi-analytic
# NPZ is read from the pure_eb_sweep output dir and its Gaussian integration
# covariance from the cov_sweep output dir (both by absolute path — lc does not
# wire cross-output deps, so the driver reads upstream sweeps directly; run
# pure_eb_sweep + cov_sweep before this). Each version emits the canonical
# ``{ver}_{blind}_pure_eb_ptes.npz`` — the exact name config_space_pte_matrices.py
# reconstructs from --pte-intermediate-dir — straight into --out. Serial over
# versions; each version is fast (206-pair grid, ~seconds).
#
# Usage:
#   run_pure_eb_ptes_sweep.sh --config <config.yaml> --cat-config <cat_config.yaml> \
#     --pure-eb-sweep-dir <pure_eb/.../pure_eb_sweep> \
#     --cov-sweep-dir <covariance/.../cov_sweep> \
#     --out <output_dir> [--blind A] [--versions "v1 v2 ..."]
set -euo pipefail

WT=/n17data/cdaley/unions/code/sp_validation.worktrees/repro-paper-ii-astra
# The image Snakemake pulled, resolved from the workflow's one declaration
# of it (workflow/common.py CONTAINER_URI + the candide profile's prefix).
CONTAINER=$($WT/workflow/scripts/container_path.py)
SRC=$WT/src
WSCRIPTS=$WT/workflow/scripts
PSCRIPTS=$WT/papers/bmodes/scripts
BIND=/home,/scratch,/automnt,/n17data,/n23data1,/n09data

CONFIG=""; CATCONFIG=""; PUREEBSWEEP=""; COVSWEEP=""; OUT=""; BLIND="A"; VERSIONS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --cat-config) CATCONFIG="$2"; shift 2;;
    --pure-eb-sweep-dir) PUREEBSWEEP="$2"; shift 2;;
    --cov-sweep-dir) COVSWEEP="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --blind) BLIND="$2"; shift 2;;
    --versions) VERSIONS="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

mkdir -p "$OUT"

if [ -z "$VERSIONS" ]; then
  VERSIONS=$(apptainer exec --bind "$BIND" --env PYTHONPATH="$SRC" "$CONTAINER" \
    /usr/local/bin/python "$PSCRIPTS/sweep_versions.py" --config "$CONFIG")
fi

for ver in $VERSIONS; do
  pureeb="$PUREEBSWEEP/${ver}_${BLIND}_pure_eb_semianalytic.npz"
  covbase="covariance_${ver}_${BLIND}_g_minsep=0.5_maxsep=300.0_nbins=1000_masked"
  covint="$COVSWEEP/$covbase/${covbase}_processed.txt"
  for f in "$pureeb" "$covint"; do
    [ -f "$f" ] || { echo "MISSING upstream input for $ver: $f" >&2; exit 1; }
  done
  echo "[pure_eb_ptes_sweep] $ver"
  apptainer exec --bind "$BIND" --env PYTHONPATH="$SRC" "$CONTAINER" \
    /usr/local/bin/python "$PSCRIPTS/calculate_pure_eb_ptes.py" \
    --version "$ver" --blind "$BLIND" \
    --pure-eb-data "$pureeb" --cov-integration "$covint" \
    --npatch 1 --n-samples 2000 --out "$OUT"
  echo "[pure_eb_ptes_sweep] $ver -> $OUT/${ver}_${BLIND}_pure_eb_ptes.npz"
done
echo "[pure_eb_ptes_sweep] done -> $OUT"
