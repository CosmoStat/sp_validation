#!/bin/bash
set -euo pipefail
R=/project/6001537/cdaley/mock-ensemble-tomo
REPO=/project/6001537/cdaley/sp/sp_validation-mockens
HERE=$REPO/scratch/cdaley/mock_ensemble_tomo
export COSMO_VAL=$R/cosmo_val COSMO_INFERENCE=$R/cosmo_inference
export SPV_CONTAINER=/project/6001537/cdaley/containers/sp_validation-ddd58fe0.sif
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export APPTAINERENV_OMP_NUM_THREADS=24 APPTAINERENV_OPENBLAS_NUM_THREADS=1 APPTAINERENV_MKL_NUM_THREADS=1
export APPTAINERENV_TMPDIR=$R/tmp APPTAINERENV_MPLCONFIGDIR=$R/cache/mpl
export PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/apptainer/1.3.4/bin:$PATH
cd "$R"
args=()
for ((s=${START:-1}; s<=${END:-1}; s++)); do
  printf -v v 'MOCK_%05d' "$s"
  args+=("$R/cosmo_val/${v}_bmode_summary_tomo.json")
done
if [ "${START:-1}" = 1 ]; then
  args+=("$R/geometry_audit.json" "$R/cosmo_val/MOCK_00001_clean_bmode_summary_tomo.json")
fi
if [ -n "${TARGETS:-}" ]; then
  read -r -a args <<< "$TARGETS"
fi
exec /home/cdaley/sp/snakemake-env/bin/snakemake \
  --profile "$REPO/workflow/profiles/default" --workflow-profile none \
  -s "$HERE/Snakefile" --configfile "$R/config.yaml" \
  --cores "${SLURM_CPUS_PER_TASK:-72}" --resources mem_mb="${MEM_MB:-340000}" \
  --set-resources xi:mem_mb=128000 pseudo_cl:mem_mb=48000 \
  --set-threads xi=24 pseudo_cl=24 cv_cosebis=24 \
  --apptainer-args '--cleanenv --bind /project,/home' \
  --rerun-incomplete --printshellcmds "${args[@]}" "$@"
