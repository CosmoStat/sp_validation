#!/usr/bin/env bash
# Pure E/B semi-analytic covariance driver (lc-native, container:none recipe).
#
# Runs the 20 independent MC chunks in parallel (each a fresh apptainer-exec
# process, deterministic seed 42+chunk_id) throttled to the node core count,
# then gathers them into the per-(version,blind) semianalytic .npz. Bit-exact
# to the paper's scatter-gather (same per-chunk seeds/order); no nested dask.
#
# Usage:
#   run_pure_eb_semianalytic.sh --version SP_v1.4.6.3_leak_corr --blind A \
#     --cat-config <cat_config.yaml> \
#     --xi-reporting <xi 20-bin .txt> --xi-integration <xi 1000-bin .txt> \
#     --cov-integration <cov ..._processed.txt> \
#     --out <output_dir> [--n-chunks 20] [--n-samples 2000] [--nproc 16]
set -euo pipefail

WT=/n17data/cdaley/unions/code/sp_validation.worktrees/repro-paper-ii-astra
SRC=$WT/src
# This user's own image, at the canonical path `spv-container pull` writes to
# (see workflow/README.md). SPV_CONTAINER overrides it here and everywhere else.
CONTAINER=${SPV_CONTAINER:-$HOME/.cache/sp_validation/sp_validation.sif}
SCRIPTS=/n17data/cdaley/unions/code/sp_validation.worktrees/repro-paper-ii-astra/papers/bmodes/scripts
BIND=/home,/scratch,/automnt,/n17data,/n23data1,/n09data

VERSION=""; BLIND="A"; CATCONFIG=""; XIREP=""; XIINT=""; COVINT=""; OUT=""
NCHUNKS=20; NSAMPLES=2000; NPROC="${SLURM_CPUS_PER_TASK:-16}"
MINSEP=1.0; MAXSEP=250.0; NBINS=20
MINSEPINT=0.5; MAXSEPINT=300.0; NBINSINT=1000; NPATCH=1

while [ $# -gt 0 ]; do
  case "$1" in
    --version) VERSION="$2"; shift 2;;
    --blind) BLIND="$2"; shift 2;;
    --cat-config) CATCONFIG="$2"; shift 2;;
    --xi-reporting) XIREP="$2"; shift 2;;
    --xi-integration) XIINT="$2"; shift 2;;
    --cov-integration) COVINT="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --n-chunks) NCHUNKS="$2"; shift 2;;
    --n-samples) NSAMPLES="$2"; shift 2;;
    --nproc) NPROC="$2"; shift 2;;
    --min-sep) MINSEP="$2"; shift 2;;
    --max-sep) MAXSEP="$2"; shift 2;;
    --nbins) NBINS="$2"; shift 2;;
    --min-sep-int) MINSEPINT="$2"; shift 2;;
    --max-sep-int) MAXSEPINT="$2"; shift 2;;
    --nbins-int) NBINSINT="$2"; shift 2;;
    --npatch) NPATCH="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

mkdir -p "$OUT/chunks"

echo "[pure_eb] $NCHUNKS chunks, $NSAMPLES samples, nproc=$NPROC, version=$VERSION blind=$BLIND"
for i in $(seq 0 $((NCHUNKS-1))); do
  (
    apptainer exec --bind "$BIND" --env PYTHONPATH="$SRC" \
      --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
      --env NUMBA_NUM_THREADS=1 --env NUMEXPR_NUM_THREADS=1 --env VECLIB_MAXIMUM_THREADS=1 \
      "$CONTAINER" \
      /usr/local/bin/python "$SCRIPTS/precompute_pure_eb_chunk.py" \
      --chunk-id "$i" --n-chunks "$NCHUNKS" --n-samples "$NSAMPLES" \
      --version "$VERSION" --blind "$BLIND" --cat-config "$CATCONFIG" \
      --xi-reporting "$XIREP" --xi-integration "$XIINT" --cov-integration "$COVINT" \
      --min-sep "$MINSEP" --max-sep "$MAXSEP" --nbins "$NBINS" \
      --min-sep-int "$MINSEPINT" --max-sep-int "$MAXSEPINT" --nbins-int "$NBINSINT" \
      --npatch "$NPATCH" --out "$OUT/chunks"
  ) > "$OUT/chunks/chunk_${i}.log" 2>&1 &
  while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do sleep 1; done
done
wait

# Verify all chunks landed
missing=0
for i in $(seq 0 $((NCHUNKS-1))); do
  [ -f "$OUT/chunks/pure_eb_chunk_${i}.npz" ] || { echo "MISSING chunk $i (see $OUT/chunks/chunk_${i}.log)" >&2; missing=1; }
done
[ "$missing" -eq 0 ] || { echo "[pure_eb] chunk failures — aborting gather" >&2; exit 1; }
echo "[pure_eb] all $NCHUNKS chunks done; gathering"

apptainer exec --bind "$BIND" --env PYTHONPATH="$SRC" "$CONTAINER" \
  /usr/local/bin/python "$SCRIPTS/gather_pure_eb_chunks.py" \
  --version "$VERSION" --blind "$BLIND" \
  --xi-reporting "$XIREP" --xi-integration "$XIINT" \
  --chunks-dir "$OUT/chunks" \
  --min-sep "$MINSEP" --max-sep "$MAXSEP" --nbins "$NBINS" \
  --min-sep-int "$MINSEPINT" --max-sep-int "$MAXSEPINT" --nbins-int "$NBINSINT" \
  --npatch "$NPATCH" --out "$OUT"
echo "[pure_eb] done -> $OUT"
