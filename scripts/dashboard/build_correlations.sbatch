#!/bin/bash
#SBATCH --job-name=sp-correlations
#SBATCH --account=def-sfabbro
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=%x-%j.log
#
# Catalogue-paper position-shear and alpha(theta) products
# (sp_validation.dashboard.correlations), one measurement per step so each
# process holds only what it needs.
#
# Measured on the #425 snapshot (10,094 tiles, 73.0M galaxies, 5.76M valid-HSM
# star detections), 8 cores; see MEASURED below. Pass the variables with
# --export, e.g.
#   sbatch --export=ALL,CATALOGUE_DIR=.../run-r425/catalogue,VERSION=SP_v2.0,\
#     OUT_DIR=.../catpaper/products,GAIA=.../gaia_dr3_G20_ruwe1p4_unions20k.fits \
#     scripts/dashboard/build_correlations.sbatch
# Stage Gaia beforehand on a login node (compute nodes may lack internet):
#   uv run scripts/dashboard/stage_gaia.py --tiles <tile lists> --out <stem>
#
# Measured at 10k tiles (#425, 8 cores; wall time incl. a 45 s catalogue read
# per step; peak RSS of the Python process):
#   stars + tiles   4 min    38 GB   (4.77M unique stars as lenses; tiles 6 s)
#   lambda          2.5 min  15 GB   (7.3M galaxy lenses, 7.3M randoms)
#   alpha           7 min    44 GB   (rho0 + tau0 for all and 3 magnitude bins)
#   gaia            6 min    39 GB   (7.23M Gaia stars, 7.3M randoms)
#   total          ~20 min
# Extrapolated to 20.5k tiles (~2x galaxies, stars, Gaia stars and randoms;
# TreeCorr NG/GG time scales roughly with the lens and source counts):
# ~40-50 min and ~75-90 GB peak (alpha, stars), hence --mem=96G and 1.5 h.

set -euo pipefail
: "${CATALOGUE_DIR:?}" "${VERSION:?}" "${OUT_DIR:?}"
SIF=${SIF:-/project/6001537/cdaley/containers/sp_validation-ddd58fe0.sif}
SRC=${SRC:-$SLURM_SUBMIT_DIR/src}  # submit from the repository root
module load apptainer

run() {
    apptainer exec --bind /scratch,/project,/home --env PYTHONPATH="$SRC" "$SIF" \
        python -m sp_validation.dashboard.correlations \
        --catalogue-dir "$CATALOGUE_DIR" --version "$VERSION" --out-dir "$OUT_DIR" \
        --threads "$SLURM_CPUS_PER_TASK" "$@"
}

run --only stars tiles
run --only lambda
run --only alpha
if [ -n "${GAIA:-}" ]; then
    run --only gaia --gaia "$GAIA"
fi
