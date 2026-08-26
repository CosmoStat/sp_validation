#!/usr/bin/env bash
# CosmoCov integration-grid covariance sweep over non-fiducial versions (Design B).
#
# Loops the non-fiducial version list (resolved via sweep_versions.py) and runs
# the same run_cosmocov_chain.sh chain the fiducial cov_integration_g recipe
# calls, once per version, on the 1000-bin integration grid (Gaussian-only,
# masked, blind A). Each version's processed matrix lands in a per-version subdir
# named canonically so cosebis_version_comparison (--cov-dir) and the pure E/B
# sweep can reconstruct it:
#
#   <out>/covariance_<ver>_A_g_minsep=0.5_maxsep=300.0_nbins=1000_masked/
#         covariance_<ver>_A_g_minsep=0.5_maxsep=300.0_nbins=1000_masked_processed.txt
#
# run_cosmocov_chain.sh writes covariance_processed.txt into --out; this driver
# renames it to the {base}_processed.txt the plotter/pure_eb sweep expect.
#
# Mask per version: v1.4.8 uses the star-halo footprint power spectrum, every
# other version uses the standard footprint (mirrors covariance.smk
# get_mask_cls_path). Covariance is recomputed per variant — it is NOT
# correction-invariant (masked v1.4.8 differs; uncorrected σ_e differs slightly).
#
# Usage:
#   run_cov_sweep.sh --config <config.yaml> --cat-config <cat_config.yaml> \
#     --planck18-json <cosmology_snapshot>/planck18.json \
#     --mask-base <cosmo_inference/data/mask> --out <output_dir> \
#     [--blind A] [--versions "v1 v2 ..."]
set -euo pipefail

CONTAINER=/n17data/cdaley/containers/snakemake-sif/current.sif
WT=/n17data/cdaley/unions/code/sp_validation.worktrees/repro-paper-ii-astra
SRC=$WT/src
WSCRIPTS=$WT/workflow/scripts
PSCRIPTS=$WT/papers/bmodes/scripts
BIND=/home,/scratch,/automnt,/n17data,/n23data1,/n09data

CONFIG=""; CATCONFIG=""; PLANCK18=""; MASKBASE=""; OUT=""; BLIND="A"; VERSIONS=""
MINSEP=0.5; MAXSEP=300.0; NBINS=1000
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --cat-config) CATCONFIG="$2"; shift 2;;
    --planck18-json) PLANCK18="$2"; shift 2;;
    --mask-base) MASKBASE="$2"; shift 2;;
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
  base="covariance_${ver}_${BLIND}_g_minsep=${MINSEP}_maxsep=${MAXSEP}_nbins=${NBINS}_masked"
  # Version dir key: strip SP_, _leak_corr, _ecutNN (mirrors get_mask_cls_path).
  vdir=$(echo "$ver" | sed -e 's/_leak_corr//' -e 's/^SP_//' -e 's/_ecut[0-9]*//')
  if [ "$vdir" = "v1.4.8" ]; then
    mask="$MASKBASE/mask_cls_footprint_starhalo_nside_4096_norm.txt"
  else
    mask="$MASKBASE/mask_cls_footprint_nside_4096_norm.txt"
  fi
  echo "[cov_sweep] $ver (mask: $(basename "$mask"))"
  bash "$WSCRIPTS/run_cosmocov_chain.sh" \
    --version "$ver" --blind "$BLIND" \
    --min-sep "$MINSEP" --max-sep "$MAXSEP" --nbins "$NBINS" --gaussian g \
    --planck18-json "$PLANCK18" --cat-config "$CATCONFIG" \
    --mask-cls "$mask" --out "$OUT/$base"
  mv "$OUT/$base/covariance_processed.txt" "$OUT/$base/${base}_processed.txt"
  echo "[cov_sweep] $ver -> $OUT/$base/${base}_processed.txt"
done
echo "[cov_sweep] done -> $OUT"
