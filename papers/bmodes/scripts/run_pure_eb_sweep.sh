#!/usr/bin/env bash
# Pure E/B semi-analytic covariance sweep over non-fiducial versions (Design B).
#
# Loops the non-fiducial version list (resolved via sweep_versions.py) and runs
# the same run_pure_eb_semianalytic.sh scatter-gather the fiducial
# pure_eb_semianalytic_data recipe calls, once per version. Each version's
# reporting+integration ξ± is read from the xi_sweep output dir and its Gaussian
# integration covariance from the cov_sweep output dir (both by absolute path —
# lc does not wire cross-output deps, so the driver reads upstream sweeps
# directly; run xi_sweep + cov_sweep before this). The gathered
# ``{ver}_{blind}_pure_eb_semianalytic.npz`` — already the canonical name
# pure_eb_version_comparison (--results-dir) reads — is moved into --out; the
# per-version MC chunks stage in a scratch subdir that is removed after gather.
#
# Usage:
#   run_pure_eb_sweep.sh --config <config.yaml> --cat-config <cat_config.yaml> \
#     --xi-sweep-dir <two_point/.../xi_sweep> \
#     --cov-sweep-dir <covariance/.../cov_sweep> \
#     --out <output_dir> [--blind A] [--versions "v1 v2 ..."]
set -euo pipefail

CONTAINER=/n17data/cdaley/containers/snakemake-sif/current.sif
WT=/n17data/cdaley/unions/code/sp_validation.worktrees/repro-paper-ii-astra
SRC=$WT/src
WSCRIPTS=$WT/workflow/scripts
PSCRIPTS=$WT/papers/bmodes/scripts
BIND=/home,/scratch,/automnt,/n17data,/n23data1,/n09data

CONFIG=""; CATCONFIG=""; XISWEEP=""; COVSWEEP=""; OUT=""; BLIND="A"; VERSIONS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --cat-config) CATCONFIG="$2"; shift 2;;
    --xi-sweep-dir) XISWEEP="$2"; shift 2;;
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
  xirep="$XISWEEP/${ver}_xi_minsep=1.0_maxsep=250.0_nbins=20_npatch=1.txt"
  xiint="$XISWEEP/${ver}_xi_minsep=0.5_maxsep=300.0_nbins=1000_npatch=1.txt"
  covbase="covariance_${ver}_${BLIND}_g_minsep=0.5_maxsep=300.0_nbins=1000_masked"
  covint="$COVSWEEP/$covbase/${covbase}_processed.txt"
  for f in "$xirep" "$xiint" "$covint"; do
    [ -f "$f" ] || { echo "MISSING upstream input for $ver: $f" >&2; exit 1; }
  done
  echo "[pure_eb_sweep] $ver"
  stage="$OUT/_stage_${ver}"
  bash "$PSCRIPTS/run_pure_eb_semianalytic.sh" \
    --version "$ver" --blind "$BLIND" --cat-config "$CATCONFIG" \
    --xi-reporting "$xirep" --xi-integration "$xiint" --cov-integration "$covint" \
    --out "$stage"
  mv "$stage/${ver}_${BLIND}_pure_eb_semianalytic.npz" \
     "$OUT/${ver}_${BLIND}_pure_eb_semianalytic.npz"
  rm -rf "$stage"
  echo "[pure_eb_sweep] $ver -> $OUT/${ver}_${BLIND}_pure_eb_semianalytic.npz"
done
echo "[pure_eb_sweep] done -> $OUT"
