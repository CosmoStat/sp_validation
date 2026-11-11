#!/bin/bash
#SBATCH --account=def-sfabbro
#SBATCH --job-name=glassmock-v2
#SBATCH --cpus-per-task=64
#SBATCH --mem=360G
#SBATCH --time=1-00:00:00
#SBATCH --output=/project/def-mjhudson/cdaley/glass-mock-v2/cvpaint/logs/suite-%j.log
# The whole v2 painted-mock suite in one allocation: paint, registry, geometry audit,
# cosmo_val_all (host Snakemake, local jobs inside the image), then every validation stage.
# Each stage is skipped when its products exist, so a resubmission resumes.
set -uo pipefail
HERE=/project/def-mjhudson/cdaley/sp/sp_validation-glassv2/scratch/cdaley/glass_mock_cvpaint/nibi
source "$HERE/env.sh"
L=$CVPAINT_ROOT/logs
echo "start $(date -Is) on $(hostname), $SLURM_CPUS_PER_TASK cores, $(git -C "$REPO" rev-parse --short HEAD)"
for f in painted painted_psfleak painted_addB; do
  [ -f "$CVPAINT_MOCK/$f.h5" ] || { bash "$HERE/paint.sh" || exit 1; break; }
done
echo "paint ok $(date -Is)"
[ -f "$CVPAINT_ROOT/cat_config.yaml" ] || spv env REPO="$REPO" SOURCE_CAT_CONFIG="$SOURCE_CAT_CONFIG" SIF="$SIF" python "$HERE/prepare.py" || exit 1
[ -f "$CVPAINT_ROOT/geometry_audit.json" ] || spv python -u "$REPO/scratch/cdaley/glass_mock_cvpaint/audit_geometry.py" > "$L/audit_geometry.log" 2>&1 || { echo "geometry audit failed"; exit 1; }
echo "registry + audit ok $(date -Is)"
if [ "${PREP_ONLY:-0}" = 1 ]; then
  CVPAINT_THREADS=$SLURM_CPUS_PER_TASK bash "$HERE/run.sh" -n --rerun-incomplete cosmo_val_all > "$L/dryrun.log" 2>&1
  echo "dry run rc=$? $(date -Is)"; exit 0
fi
CVPAINT_THREADS=$SLURM_CPUS_PER_TASK bash "$HERE/run.sh" --rerun-incomplete cosmo_val_all > "$L/all.log" 2>&1
rc=$?; echo "cosmo_val_all rc=$rc $(date -Is)"
[ $rc -eq 0 ] && git -C "$REPO" rev-parse HEAD > "$CVPAINT_ROOT/workflow_complete"
mkdir -p "$CVPAINT_ROOT/figures"
for part in harmonic additive blind_xi xi decompositions psf scan; do
  bash "$HERE/analyze.sh" $part > "$L/validate_$part.log" 2>&1
  prc=$?; echo "validate $part rc=$prc $(date -Is)"
  [ $prc -eq 0 ] || rc=1
done
[ $rc -eq 0 ] && touch "$CVPAINT_ROOT/validation_complete"
echo "done $(date -Is)"
