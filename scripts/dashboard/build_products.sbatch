#!/bin/bash
#SBATCH --job-name=dashboard-products
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00

# Usage: sbatch build_products.sbatch CATALOGUE_DIR VERSION PRODUCT_DIR
# SPV_IMAGE must name the sp_validation Apptainer image. Run from the checkout.
# #425, 10,094 tiles: galaxy + star pass 3m42s, 7.45 GiB peak RSS
# (Slurm step 25103890.71, 4 CPUs). Linear 20,500-tile extrapolation:
# ~7m31s, ~15.1 GiB; quantile buffers scale with galaxy count.
# Comprehensive pass: 59m15s, 2.99 GiB peak RSS (step 25103890.23,
# 2 CPUs); linear 20,500-tile extrapolation ~2h00m20s, with streaming
# memory approximately constant. Wall time is sensitive to filesystem I/O.
set -euo pipefail
catdir=$(realpath "${1:?catalogue directory}")
version=${2:?version}
out=$(realpath -m "${3:?product directory}")
root=$(git rev-parse --show-toplevel)
: "${SPV_IMAGE:?set SPV_IMAGE to the container image}"
module load apptainer
mkdir -p "$out"
stage=$(mktemp -d "$out/.build.XXXXXXXX")
trap 'rm -rf "$stage"' EXIT
run() {
    apptainer exec --bind /scratch,/project,/home \
        --env "PYTHONPATH=$root/src" "$SPV_IMAGE" python "$@"
}
galaxies="$catdir/shape_catalog_cut_ngmix.fits"
expected=$(run -c 'import fitsio,sys; print(fitsio.FITS(sys.argv[1])[1].get_nrows())' "$galaxies")
run -m sp_validation.dashboard.field_maps \
    --galaxies "$galaxies" --stars "$catdir/stars.hdf5" \
    --version "$version" --out-dir "$stage"
run -m sp_validation.dashboard.selection_hist \
    --catalogue "$catdir/shape_catalog_comprehensive_ngmix.hdf5" \
    --config "$catdir/config_mask.yaml" --expected "$expected" --version "$version" \
    --out "$stage/selection_hist_${version}.npz"
mv "$stage"/*.npz "$out/"
