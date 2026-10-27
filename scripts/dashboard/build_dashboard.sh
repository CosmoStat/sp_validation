#!/bin/bash
# Build every cosmo_val dashboard figure for one catalogue snapshot through Slurm.
#
# Submits the catalogue-level products (build_products.sbatch) and the
# position-shear products (build_correlations.sbatch) in parallel, then a render
# job that runs once both succeed. With --after, all of it waits for that job
# (e.g. the catalogue extraction) to finish successfully.
#
# Usage, from the repository root on a login node:
#   scripts/dashboard/build_dashboard.sh CATALOGUE_DIR RUN_DIR VERSION OUT_DIR \
#       [--gaia GAIA_FITS] [--after JOBID]
# Products go to OUT_DIR/products, figures and figures.json to OUT_DIR/figures.
# ACCOUNT (default def-sfabbro) and SPV_IMAGE (default the pinned SIF) are read
# from the environment.
set -euo pipefail
catdir=$(realpath -m "${1:?catalogue directory}")
rundir=$(realpath -m "${2:?cosmo_val run directory}")
version=${3:?version}
out=$(realpath -m "${4:?output directory}")
shift 4
gaia=""
after=""
while [ $# -gt 0 ]; do
    case $1 in
        --gaia) gaia=$(realpath "$2"); shift 2 ;;
        --after) after=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

root=$(git rev-parse --show-toplevel)
cd "$root"
account=${ACCOUNT:-def-sfabbro}
export SPV_IMAGE=${SPV_IMAGE:-/project/6001537/cdaley/containers/sp_validation-ddd58fe0.sif}
mkdir -p "$out/products" "$out/figures" "$out/logs"
dep=()
[ -n "$after" ] && dep=(--dependency="afterok:$after")

products=$(sbatch --parsable --account="$account" "${dep[@]}" \
    --output="$out/logs/%x-%j.log" --export=ALL \
    scripts/dashboard/build_products.sbatch "$catdir" "$version" "$out/products")
correlations=$(sbatch --parsable --account="$account" "${dep[@]}" \
    --output="$out/logs/%x-%j.log" \
    --export=ALL,CATALOGUE_DIR="$catdir",VERSION="$version",OUT_DIR="$out/products",GAIA="$gaia",SIF="$SPV_IMAGE",SRC="$root/src" \
    scripts/dashboard/build_correlations.sbatch)
render=$(sbatch --parsable --account="$account" --job-name=dashboard-render \
    --dependency="afterok:$products:$correlations" \
    --time=00:30:00 --cpus-per-task=2 --mem=8G \
    --output="$out/logs/%x-%j.log" \
    --wrap="module load apptainer && apptainer exec --bind /scratch,/project,/home \
--env PYTHONPATH=$root/src $SPV_IMAGE python -m sp_validation.dashboard.render \
--run-dir $rundir --version $version --maps-dir $out/products \
--correlations-dir $out/products --out $out/figures")
echo "products $products  correlations $correlations  render $render"
