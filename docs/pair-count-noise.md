# Pair-count shape noise

Non-tomographic `cosmo_val` workflows use measured effective pairs for OneCovariance's real-space Gaussian shape-noise term by default.
Set `covariance.shape_noise: analytic` to recover the uniform-area approximation.
Other workflows retain their analytic default; tomographic pair-count noise is not implemented and an explicit request fails.
This changes neither blinding nor the mixed, sample-variance, non-Gaussian or SSC terms.

## Cache and normalization

`covariance_pair_counts` reads only the configured RA, Dec and weight columns of the same selected shear catalogue as `xi`.
It counts on each requested covariance grid, using TreeCorr NN correlations with weights `w` and `w²`, spherical coordinates, logarithmic bins and `min_top=6`, matching `xi`.
The cache is independent of cosmology and contains no shapes, signals or blind records.
The reader projects fields directly from the lazy catalogue chunks, including compound HDF5 datasets, rather than materialising rows containing shape fields.
It is a separate rule so covariance-only reruns need no shape measurement or access to a blind.
Its `threads`, memory and runtime can be overridden just like `xi` (large catalogues may need more than the default 40 GB).
For the full v2 catalogue, use `--set-resources covariance_pair_counts:mem_mb=128000 covariance_pair_counts:runtime=1440`, alongside the run's existing resource overrides; the two NN passes can peak above the ξ rule's memory usage.

For unordered auto pairs in bin b, the NPZ stores:

- `weight`: W = Σ w_i w_j.
- `weight_squared`: Q = Σ (w_i w_j)².
- `npairs`: the raw unordered count, plus exact bin edges and nominal centres.
- `ordered_effective_pairs`: 2 W²/Q.

With independent isotropic shapes of constant per-component dispersion s, the pure-noise variance of either ξ component is `2 s⁴ Q/W²`.
The total two-component RMS is σ_e = √2 s, so the same variance is `σ_e⁴/(2 N_eff)`, where `N_eff = W²/Q` is **unordered**.
Mixing the two dispersion conventions gives a factor-of-four error.
`cov_th.sigma_e` stores the per-component dispersion s, which OneCovariance takes directly.

## OneCovariance interface

Verified against OneCovariance `c8755d6de43aadaa1cd4fe132ecf1c1d534e61f8`, the source used by the v2 run with the `ddd58fe0` container.
The external checkout is selected by `tools.onecov_executable`; the image supplies its dependencies, not the source.
`onecov/cov_input.py` reads `[tabulated inputs files]` keys `npair_directory` and `npair_mm_file` (first column θ in arcmin, last column counts).
`onecov/cov_setup.py:get_npair` accepts a matching grid without rebinning and otherwise falls back to an unsafe-for-effective-counts sum.
The workflow therefore writes **nominal geometric bin centres**, not measured mean separations, and counts separately on every covariance grid.
Empty or non-finite pair bins fail instead of silently restoring analytic noise.

OneCovariance's analytic auto count is `A_arcmin² n_eff² π(θ_hi²−θ_lo²)`, which counts **ordered** pairs.
At `onecov/cov_theta_space.py` lines 1288–1297, the two Kronecker contractions and the `/0.5` yield `4 s⁴/N_ordered` for a single source bin.
The two-column `.txt` thus supplies `N_ordered = 2 W²/Q`, giving exactly `2 s⁴ Q/W²`.
No area rescaling or post-run whole-covariance scaling is applied.

The normalization test checks the uniform-field expectation analytically, including non-unit equal weights and both dispersion conventions.
A four-object positions-only catalogue checks TreeCorr's unordered convention against explicit pair enumeration.
HDF5 tests forbid whole-dataset or shape-column reads for single, chunked and joined layouts.
DAG tests verify the cache-to-ini dependency, analytic opt-out, dispersion boundary and tomographic default/refusal behavior.

## Existing products

Existing covariance files do not encode the noise prescription in their filenames.
When migrating a completed run, use a fresh output tree or explicitly rerun `covariance_ini` and its descendants; do not interpret an old cached matrix as measured-pair noise.
Do not sum fine-grid effective counts to obtain coarse-grid counts: sum W and Q separately before forming W²/Q, or count directly on that grid as this rule does.
Pair noise includes the observed spatial selection and weight variation; it is not an independent mask-only simulation.
Mask-dependent mixed and sample-variance corrections remain separate validation work.
