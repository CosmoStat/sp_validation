#!/usr/bin/env bash
# CosmoCov covariance chain (lc-native, container:none recipe).
#
# Faithful port of covariance_ini -> covariance_cosmocov (x3 blocks) ->
# covariance_cat -> covariance_process. The CosmoCov C++ binary runs on the
# bare host (module load gcc/intelpython/openmpi, as in the original
# container:None rule); the .ini generation and cosmocov_process step run inside
# the sp_validation apptainer container. The 3 shear-shear blocks (++,--,+-) are
# independent and run in parallel.
#
# Usage:
#   run_cosmocov_chain.sh --version SP_v1.4.6.3_leak_corr --blind A \
#     --min-sep 0.5 --max-sep 300.0 --nbins 1000 --gaussian g \
#     --planck18-json <cosmology_snapshot>/planck18.json \
#     --cat-config <cat_config.yaml> --mask-cls <mask_cls...norm.txt> \
#     --out <output_dir>
set -euo pipefail

CONTAINER=/n17data/cdaley/containers/containers/
WT=/n17data/cdaley/unions/code/sp_validation.worktrees/repro-paper-ii-astra
SRC=$WT/src
BIND=/home,/scratch,/automnt,/n17data,/n23data1,/n09data
COSMOCOV=/n23data1/n06data/lgoh/scratch/UNIONS/CosmoCov/covs/cov

VERSION=""; BLIND="A"; MINSEP=""; MAXSEP=""; NBINS=""; GAUSSIAN=""
PLANCK18=""; CATCONFIG=""; MASKCLS=""; OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --version) VERSION="$2"; shift 2;;
    --blind) BLIND="$2"; shift 2;;
    --min-sep) MINSEP="$2"; shift 2;;
    --max-sep) MAXSEP="$2"; shift 2;;
    --nbins) NBINS="$2"; shift 2;;
    --gaussian) GAUSSIAN="$2"; shift 2;;
    --planck18-json) PLANCK18="$2"; shift 2;;
    --cat-config) CATCONFIG="$2"; shift 2;;
    --mask-cls) MASKCLS="$2"; shift 2;;
    --cosmocov) COSMOCOV="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

mkdir -p "$OUT"
INI="$OUT/covariance.ini"

echo "[cosmocov] generating .ini"
apptainer exec --bind "$BIND" --env PYTHONPATH="$SRC" "$CONTAINER" \
  /usr/local/bin/python "$WT/workflow/scripts/generate_cosmocov_ini.py" \
  --version "$VERSION" --blind "$BLIND" \
  --planck18-json "$PLANCK18" --cat-config "$CATCONFIG" \
  --min-sep "$MINSEP" --max-sep "$MAXSEP" --nbins "$NBINS" --gaussian "$GAUSSIAN" \
  --mask-cls "$MASKCLS" --out-ini "$INI"

echo "[cosmocov] loading modules + running 3 blocks (parallel)"
source /etc/profile.d/modules.sh
module unload gcc 2>/dev/null || true; module load gcc
module unload intelpython 2>/dev/null || true; module load intelpython/3-2024.1.0
module load openmpi

cd "$OUT"
# BLOCK_PAIRS = [("++","1"), ("--","2"), ("+-","3")] — one CosmoCov invocation per block
for idx in 1 2 3; do
  ( "$COSMOCOV" "$idx" "$INI" > "$OUT/cosmocov_block_${idx}.log" 2>&1 ) &
done
wait

# Concatenate blocks in BLOCK_PAIRS order (++, --, +-) — as covariance_cat does
CAT="$OUT/covariance.txt"
: > "$CAT"
for pm_idx in "++:1" "--:2" "+-:3"; do
  pm="${pm_idx%%:*}"; idx="${pm_idx##*:}"
  blk="$OUT/cov_tmp_ssss_${pm}_cov_Ntheta${NBINS}_Ntomo1_${idx}"
  [ -f "$blk" ] || { echo "MISSING block $blk (see cosmocov_block_${idx}.log)" >&2; exit 1; }
  cat "$blk" >> "$CAT"
done
echo "[cosmocov] concatenated -> $CAT"

echo "[cosmocov] processing (positive-definite check, G/G+NG extract, QA plot)"
apptainer exec --bind "$BIND" --env PYTHONPATH="$SRC" "$CONTAINER" \
  /usr/local/bin/python "$WT/cosmo_inference/scripts/cosmocov_process.py" \
  "$CAT" "$OUT/covariance_processed"
echo "[cosmocov] done -> $OUT/covariance_processed.txt (+_g.txt, +_plot.pdf)"
