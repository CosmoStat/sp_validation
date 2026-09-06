#!/usr/bin/env bash
# CosmoCov covariance chain (lc-native, container:none recipe).
#
# covariance_ini -> covariance_cosmocov (x3 blocks) -> covariance_cat ->
# covariance_process. The CosmoCov C++ binary runs on the bare host (module load
# gcc/intelpython/openmpi); the .ini generation and cosmocov_process steps run
# inside the sp_validation apptainer container. The 3 shear-shear blocks
# (++,--,+-) are independent and run in parallel.
#
# Usage:
#   run_cosmocov_chain.sh --version SP_v1.4.6.3_leak_corr --blind A \
#     --min-sep 0.5 --max-sep 300.0 --nbins 1000 --gaussian g \
#     --planck18-json <cosmology_snapshot>/planck18.json \
#     --cat-config <cat_config.yaml> --mask-cls <mask_cls...norm.txt> \
#     --out <output_dir>
#
# The checkout is this script's own (workflow/scripts/../..). Deployment paths
# come from the environment, with the current candide values as defaults:
#   SPV_CONTAINER  apptainer image        (default /n17data/cdaley/containers/containers/)
#   SPV_BIND       apptainer --bind list
#   COSMOCOV       CosmoCov `cov` binary  (also settable with --cosmocov)
set -euo pipefail

WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC=$WT/src
CONTAINER=${SPV_CONTAINER:-/n17data/cdaley/containers/containers/}
BIND=${SPV_BIND:-/home,/scratch,/automnt,/n17data,/n23data1,/n09data}
COSMOCOV=${COSMOCOV:-/n23data1/n06data/lgoh/scratch/UNIONS/CosmoCov/covs/cov}

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
# Absolutize OUT before the `cd "$OUT"` below (the CosmoCov binary writes its
# blocks into cwd): every other OUT-relative path would otherwise re-resolve
# against the new cwd and double-nest. lc templates {output} as a
# project-relative path, so both relative and absolute --out must work.
OUT="$(cd "$OUT" && pwd)"
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
# One CosmoCov invocation per block; see common.py BLOCK_PAIRS.
for idx in 1 2 3; do
  ( "$COSMOCOV" "$idx" "$INI" > "$OUT/cosmocov_block_${idx}.log" 2>&1 ) &
done
wait

# Concatenate blocks in BLOCK_PAIRS order (++, --, +-), as covariance_cat does.
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
