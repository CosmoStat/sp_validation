# SACC → CosmoSIS inference

The shared workflow converts an analysis SACC to CosmoSIS 2pt-FITS, renders a pipeline INI from explicit template/values/priors inputs, samples under MPI, and writes a weighted JSON summary.
It never reads a catalogue or a blind record.
A concealed SACC stays concealed; its `blind` stamp is recorded in the input provenance.

```bash
snakemake --profile workflow/profiles/default -j 32 \
  -s workflow/inference/Snakefile inference_fiducial --configfile run.yaml
```

Run Snakemake on the host, from the real checkout path.
Its jobs run in the CI dependency image and import the launched checkout.
The standalone entry point needs only the config below; `workflow/Snakefile` includes the same rules for composition with measurement workflows.
`inference_fiducial` gathers all named runs, not just preparation products.
A single run can be requested as `<output_dir>/<run>/summary.json`.

```yaml
inference:
  output_dir: /path/to/run/inference
  runs:
    mock_00001:
      sacc: /path/to/MOCK_00001_clean_tomo.sacc
      covariance: /path/to/current_reference_tomo.sacc  # optional replacement
      n_bins: 4
      template: cosmo_inference/cosmosis_config/templates/cosmosis_pipeline_A_ia.ini
      values: /path/to/mock_values.ini
      priors: /path/to/mock_priors.ini
      threads: 32
      sampler: nautilus
      sampler_settings:
        n_live: 300
        n_eff: 1500
        resume: T
      scale_cuts:
        XI_PLUS: [12.0, 83.0]
        XI_MINUS: [12.0, 83.0]
      overrides:
        pipeline:
          debug: F
          timing: F
```

Each run produces `data.fits`, `input.json`, `pipeline.ini`, `chain.txt`, `cosmosis.log`, and `summary.json`.
Keep values and priors outside the code: every sampled bin needs its own m and redshift-shift entries.
The example sampler settings are economical mock-validation settings, not a convergence recommendation for a final analysis.
The summary uses Nautilus log weights (or equal weights for unweighted chains) and reports means, standard deviations and central 68% intervals.
MCMC burn-in must be handled by the sampler configuration before summarization.

## Inputs and covariance

Conversion writes all ξ+ pairs, then all ξ− pairs, in numeric upper-triangle order, with 1-indexed BIN1/BIN2 and one NZ column per source bin.
The covariance follows exactly the same SACC-index permutation, including cross-pair terms.
Integration-grid and unrelated diagnostic points are excluded.
A covariance reference contributes only its covariance, never its means or n(z).
Reference rows are matched by statistic, tracer pair, grid and nominal angle (or measured angle if no nominal bin is stored).
This allows a corrected data covariance to be used on a clean mock with identical nominal bins.

The direct `sacc_to_twopoint_fits` converter retains the historical mixed ξ/Cℓ file layout and single-bin bytes.
CSL's legacy reader cannot associate two separate covariances with such a mixed file, so inference preparation writes only the requested likelihood statistic.
The default is `statistic: xi`.
`statistic: cell` selects a CELL_EE-only likelihood product, but the SAMPLE representation does not propagate SACC bandpower windows: a harmonic inference run must validate its window treatment separately.

For the legacy single-bin PSF-leakage pipeline, add `rho_stats` and `tau_stats` sidecar FITS paths and select `cosmosis_pipeline_A_psf.ini`.
Both sidecars are required; RHO_STATS carries variances absent from the SACC.
The τ covariance comes from the SACC, including τ0–τ2 correlations.
Tomographic PSF sidecars are not supported by that legacy model.
The old ξ-file preparation CLI `scripts/cosmosis_fitting.py` is retired; its FITS builders survive only as a frozen byte-compatibility test oracle.

## Runtime

CosmoSIS and the UNIONS-WL CSL fork (`b7b1552a`) ship in the CI image at `/opt/cosmosis-standard-library`.
The renderer resolves `CSL_DIR` explicitly, and the chain rule evaluates `python -m cosmosis.configure` before launch so compiled modules can find `libcosmosis.so`.
Run sampling as `mpiexec -n N cosmosis --mpi`, never `--smp`.
PolyChord requires its optional sampler package; Nautilus is available in the current image.
Use resumable settings for long runs.

Each chain rule runs on one allocated node and disables Open MPI's attempt to rediscover Slurm, because the image has no `srun` binary.
For manual single-node launches, use the same environment settings:

```bash
export APPTAINERENV_PRTE_MCA_ras='^slurm'
export APPTAINERENV_PRTE_MCA_plm='^slurm'
```

Set `inference.mpi_args: "--bind-to none"` when running several chains concurrently inside the allocation.
Never use this local-launch setting to span unallocated nodes.
The allocation, its CPU limit, and Snakemake's `--cores` remain the resource limits.
