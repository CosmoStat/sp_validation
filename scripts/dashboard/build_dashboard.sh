#!/bin/bash
# Build every cosmo_val dashboard figure for one catalogue snapshot through Slurm.
#
# Submits the catalogue-level products (build_products.sbatch), the
# position-shear products (build_correlations.sbatch) and the COSEBIs
# scale-cut PTE grid (from the run's ξ± and covariance) in parallel, then a
# render job that runs once all succeed. A grid already in OUT_DIR/products is
# kept. With --after, all of it waits for that job
# (e.g. the catalogue extraction) to finish successfully.
#
# Every cosmological product is checked against the blind its version is
# declared under in the run's catalogue config: --cat-config, else RUN_DIR's
# cat_config.yaml. With neither, the script refuses to start.
#
# Usage, from the repository root on a login node:
#   scripts/dashboard/build_dashboard.sh CATALOGUE_DIR RUN_DIR VERSION OUT_DIR \
#       [--gaia GAIA_FITS] [--after JOBID] [--cat-config CAT_CONFIG]
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
catconfig=""
while [ $# -gt 0 ]; do
    case $1 in
        --gaia) gaia=$(realpath "$2"); shift 2 ;;
        --after) after=$2; shift 2 ;;
        --cat-config) catconfig=$(realpath "$2"); shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

root=$(git rev-parse --show-toplevel)
cd "$root"
catconfig=${catconfig:-$rundir/cat_config.yaml}
[ -e "$catconfig" ] || {
    echo "no catalogue config at $catconfig: pass --cat-config with the config the run declared its blinds in" >&2
    exit 2
}
echo "blinds declared by $catconfig"
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
grid="$out/products/cosebis_grid_$version.npz"
after_grid=""
if [ ! -e "$grid" ]; then
    # #425 full grid, 210 cuts: see the cosebis-grid log for timing.
    cosebis=$(sbatch --parsable --account="$account" --job-name=dashboard-cosebis-grid \
        --time=01:30:00 --cpus-per-task=16 --mem=32G \
        --output="$out/logs/%x-%j.log" \
        --wrap="module load apptainer && apptainer exec --bind /scratch,/project,/home \
--env PYTHONPATH=$root/src $SPV_IMAGE python -m sp_validation.dashboard.cosebis_grid \
--run-dir $rundir --version $version --cat-config $catconfig --out $grid")
    after_grid=":$cosebis"
fi
render=$(sbatch --parsable --account="$account" --job-name=dashboard-render \
    --dependency="afterok:$products:$correlations$after_grid" \
    --time=00:30:00 --cpus-per-task=2 --mem=8G \
    --output="$out/logs/%x-%j.log" \
    --wrap="module load apptainer && apptainer exec --bind /scratch,/project,/home \
--env PYTHONPATH=$root/src $SPV_IMAGE python -m sp_validation.dashboard.render \
--run-dir $rundir --version $version --cat-config $catconfig --maps-dir $out/products \
--correlations-dir $out/products --out $out/figures")
echo "products $products  correlations $correlations  cosebis-grid ${cosebis:-kept}  render $render"
