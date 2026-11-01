# Nibi site for the painted-mock suite on the v2 footprint; sourced by every script here.
export R=/project/def-mjhudson/cdaley/glass-mock-v2
export REPO=/project/def-mjhudson/cdaley/sp/sp_validation-glassv2
export SIF=/project/6001537/cdaley/containers/sp_validation-ddd58fe0.sif
export SNAKEMAKE=/home/cdaley/sp/snakemake-env/bin/snakemake
export INPUT=/project/def-mjhudson/cdaley/exports/glass-mock-input/full2-r425_selected_shapes.hdf5
export SOURCE_CAT_CONFIG=/scratch/cdaley/cosmo-val-v2/run-int1028/cat_config.yaml
export CVPAINT_ROOT=$R/cvpaint CVPAINT_MOCK=$R/mocks/seed_00001 CVPAINT_BLINDS=$R/blinds
export TMPDIR=$R/tmp MPLCONFIGDIR=$R/.mpl XDG_CACHE_HOME=$R/cache
export BINDS=/project,/scratch,/home
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$CVPAINT_ROOT/logs"
module load apptainer >/dev/null 2>&1 || true
# One command inside the image, with this checkout's src/ and single-threaded BLAS.
spv() {
  apptainer exec --cleanenv --bind "$BINDS" "$SIF" env PYTHONPATH="$REPO/src" \
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
    MPLCONFIGDIR="$MPLCONFIGDIR" TMPDIR="$TMPDIR" PYTHONUNBUFFERED=1 \
    CVPAINT_ROOT="$CVPAINT_ROOT" CVPAINT_MOCK="$CVPAINT_MOCK" CVPAINT_BLINDS="$CVPAINT_BLINDS" "$@"
}
