# Response, weights and additive bias

The applied estimator follows Martin Kilbinger's **Separate** definition on slide S1 (6 October 2026, PDF p14, [shared deck](https://docs.google.com/presentation/d/1BdPlRCf742AcpfJVrAe_c3-OOp1ARiwuU9RMNflb0U8/edit)).
DES weights follow **case d** in `/n23data1/mkilbing/imsim_wdes_response/wdes_response.py`, functions `w_table`, `eval_table`, `responses`.
The common-parent spatial deletion is ported from Martin's `response_jackknife` in commit `b268fd64` (`feat/spv-launcher`); it now uses sufficient group sums and includes c and the response-element covariance.
Neither split nor joint on aligned metacal rows includes detection response.

For branch b, define A_b(x) = sum(S_b w_b x) / sum(S_b w_b).
Every branch has its own selection and denominator.
The artificial metacal step h defaults to 0.01; it is distinct from the manifest's externally injected simulation shear (0.025 for these grids).

| Quantity | Definition |
| --- | --- |
| W per cell | mean(trace R_g/2)^2 / mean((e1²+e2²)/2), unweighted raw no-shear moments |
| DES weights | Fit W once on S_ns; evaluate that same table at each branch's own SNR and T/Tpsf |
| IV weights | 1/(2 sigma_eps² + G1_ERR² + G2_ERR²), using each branch's own measured errors |
| R_shear | A_ns((e_j+ − e_j−)/(2h)), all four elements |
| R_sel | (A_j+(e_ns) − A_j−(e_ns))/(2h); includes cuts, weight changes and cell migration |
| R_total / R_joint | Apply R_shear + R_sel; save joint = (A_j+(e_j+) − A_j−(e_j−))/(2h) as a diagnostic |
| c_e / c_g | Raw c_e = A_ns(e); calibrated c_g = inverse(R_total) c_e |
| Applied shear | inverse(R_total) e − c_g; omit c_g subtraction for constant-shear simulations |

## Configuration and compatibility

```yaml
metacal:
  weight_scheme: des
  w_col: w_des
  des_num_bins: 20
  additive_correction: true  # false for constant-shear simulations
```

The remaining cuts and sigma_eps_prior are unchanged in `mask_v2.0.yaml`.
`metacal_from_config(data, preselection, config['metacal'])` is the common entry point.
A mismatched `w_col` raises an error.
New cut products declare `WGHTSCH`, `RAPPLIED`, `MCSTEP`, `WPOLICY` and `IVPOLICY`.
Consumers must use the matching weight and `R: 1`; the catalogue already received the full matrix.
The configured catalogue reader and image-simulation reader refuse a second R or a different numerator weight on these products.
External/historical catalogues without these metadata keep their existing interpretation.

At the Python API, `global_R_weight='w'` is deprecated and maps to `weight_scheme='iv'`, including squared G_ERR and weighted selection means.
Old versioned release configurations, identified by the absence of `weight_scheme` at the config entry point, explicitly use the legacy response/c and unsquared-IV policy rather than reinterpret a historical release.
Extraction carries all ten branch G_ERR columns and writes corrected `w_iv` using the public variance helper; a legacy calibration intentionally reconstructs the historical IV internally.
Modern IV rejects inputs missing sheared-branch errors: regenerate from the tile-final catalogues, whose ngmix errors are already present.
The merge allow-list must retain those columns too.
The optional direct-calibration tail of `extract_info.py` refuses because it lacks a shared weight/selection configuration; use the comprehensive product and calibration driver.

DES cells are right-closed, including internal boundaries.
Finite coordinates outside the table range use the nearest edge cell; empty/invalid cells or nonfinite coordinates get zero weight.
This deliberately differs from the historical pandas fallback that sent every invalid/out-of-range training coordinate to the last cell.
With strictly interior no-shear-selected training rows the definitions agree with Martin's table.
A row selected only on a sheared branch still receives a fixed-table weight.

## Stored diagnostics and uncertainties

`save_response_product` adds W_DES, W_COUNT, W_EDGES, R_JK and R_COV extensions when spatial groups are supplied.
R_JK deletes each shared TILE_ID from every branch and recomputes the selected population and its denominator.
R_COV uses (K−1)/K times the sum of outer products of centered delete-one estimates.
Its ordering is row-major R_shear, R_sel, R_total, R_joint, c_e, c_g (20 elements).
The primary uncertainty is conditional on **fixed W**, matching Martin's case-d comparison.
Whether to refit W in each spatial deletion remains an open question for Martin; this PR does not implement that policy.
The returned c_err is a weighted error of the raw mean (legacy mode keeps historical scatter); correlated response/c uncertainty belongs in R_COV.

`response_diagnostics.plot_response` plots weighted/unweighted per-object R_g, spatial delete-one R_sel/total/joint, split−joint differences, and patch-mean R_g maps.
R_sel is an ensemble derivative, never a per-object response.
`pair_products` measures the angle-binned, pair-weighted products of all four per-object R_g elements and compares them to global weighted products.
This is **stage 1 of #23**: it omits selection/weight derivatives and cannot certify total two-point response adequacy.
No Appendix-A mixed derivative, A14 normalization or cosmological signal is calculated by this diagnostic.

The reproducible unblinded-grid validation is `scripts/validation/validate_response.py`.
It matches Martin's cuts, table and paired weighting first, then reports the unpaired full-population estimator and spatial errors with R re-estimated for each deletion.
Squared IV changes exported IV weights but not pure-DES R or m.
Residual m from simulations remains a separate result, not an extra correction silently applied to catalogue e.

## Review questions

1. IV uses each branch's measured shape variances; the [four-grid measurement](../validation/response/README.md#exact-inverse-variance-mode) shows that freezing them biases m upward by 0.23–0.36 percentage points.
2. What scale-dependent/anisotropic calibration budget should the complete two-point response meet?
3. Is W a deliberately fixed calibration object, or should errors include a W refit in each spatial sample?
4. Is there a retained config/result identifying the weights and selection behind the historical m = −0.057?
