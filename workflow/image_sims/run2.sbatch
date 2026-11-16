#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --output=/scratch/cdaley/imsim/logs/run2/%x-%j.out
# Fir image-simulation run 2, after ShapePipe: m (im_results), the data checks
# and paired estimators (/scratch/cdaley/imsim/work-proc), the exposure
# background-rms check, figures, and the export of the small products.
# From the repository root:
#
#   sbatch -c 32 --mem=192G -t 6:00:00 -J im2-measure workflow/image_sims/run2.sbatch
#
# Writes /scratch/cdaley/imsim/export-grid2/DONE, or appends to its FAILED.
set -euo pipefail
I=/scratch/cdaley/imsim
EXPORT=$I/export-grid2
trap 'echo "$(date) ${SLURM_JOB_NAME} ${SLURM_JOB_ID} failed at line $LINENO" >> $EXPORT/FAILED' ERR
mkdir -p $EXPORT
module load apptainer
SIF=/project/rrg-kyi/cdaley/unions/containers/sp_validation-develop.sif
K=$SLURM_CPUS_PER_TASK
M=$I/measure-grid2
W=$I/work-proc
TRUTH=$I/sims/1z2z_grid_2/catalogues/input/truth.fits
TABLE=$(awk '/weight_table:/{print $2}' workflow/image_sims/fir_grid2.yaml)
py() { apptainer exec --bind /scratch,/project,/home $SIF \
         env OMP_NUM_THREADS=1 GRID=grid_2 PYTHONPATH=$PWD/src python "$@"; }
echo "start $(date)"

$I/env/snakemake/bin/snakemake --profile workflow/profiles/fir -s workflow/image_sims/Snakefile \
    -j $K im_results --configfile workflow/image_sims/fir_grid2.yaml

mkdir -p $M/diagnostics $M/figures
py $W/diagnose.py $I/products $I/sims $M/diagnostics
py $W/paired.py $M $TRUTH $M/diagnostics/paired
py $W/m_truth_cut.py $M $TRUTH $M/diagnostics/m_truth_cut.yaml 24.0 24.5 25.0 25.5
py $W/tables.py $M/results $TABLE $M/diagnostics
py $W/figures.py $M/results $TABLE $M/figures
py $W/paired_fig.py $M/results $M/diagnostics/paired.txt $M/figures/m_paired.png \
    "Image-sim run 2 (150 interior tiles around 253.246): m per S/N and input-magnitude bin, tile jackknife"
py $W/rms_check.py $I/sims/1z2z_grid_2 $I/sims/grid_2_shared/ccds.parquet \
    $I/sp-fir/workflow/config/cfis_image_sims $M/diagnostics/rms_check 0 $K

cp -r $M/results $M/figures $M/diagnostics $EXPORT/
cp $I/staging/g2-253-246/READY.json $I/staging/g2-253-246/tiles.txt \
   $I/staging/g2-253-246/exposures.txt $I/sims/grid_2_shared/ccds.parquet $EXPORT/
rm -f $EXPORT/diagnostics/match_1z2z.npz $EXPORT/diagnostics/psf_used_vs_truth.npz
echo "$(date) run 2 measured; results in $M, copies here" > $EXPORT/DONE
echo "done $(date)"
