#!/bin/bash
# Drive seeds START..END through one local Snakemake inside the current allocation.
set -euo pipefail
R=$(readlink -f /project/rrg-kyi/cdaley/unions/analyses/mock-ensemble-sampled)
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
module load apptainer
export COSMO_VAL=$R/cosmo_val COSMO_INFERENCE=$R/cosmo_inference
export SPV_CONTAINER=/project/rrg-kyi/cdaley/unions/analyses/mock-ensemble-tomo/inputs/containers/sp_validation-ddd58fe0.sif
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1 APPTAINERENV_MKL_NUM_THREADS=1
export APPTAINERENV_TMPDIR=$R/tmp APPTAINERENV_MPLCONFIGDIR=$R/cache/mpl
cd "$R"
args=()
for ((s=${START:-1}; s<=${END:-1}; s++)); do
  printf -v tag '%05d' "$s"
  args+=("$R/seeds/MOCK_${tag}.json")
done
if [ -n "${TARGETS:-}" ]; then read -r -a args <<< "$TARGETS"; fi
"$HOME/sp/snakemake-env/bin/snakemake" \
  --profile "$REPO/workflow/profiles/default" --workflow-profile none \
  -s "$HERE/Snakefile" --configfile "$R/config.yaml" \
  --cores "${SLURM_CPUS_PER_TASK:-48}" --resources mem_mb="${MEM_MB:-180000}" \
  --set-resources xi:mem_mb=110000 pseudo_cl:mem_mb=48000 cv_cosebis:mem_mb=12000 \
  --set-threads xi=24 pseudo_cl=24 cv_cosebis=1 \
  --apptainer-args '--cleanenv --bind /project,/home,/scratch' \
  --rerun-incomplete --keep-going --printshellcmds "${args[@]}" "$@"
