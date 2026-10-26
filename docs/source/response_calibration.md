# Response, weights and additive bias

The applied estimator follows Martin Kilbinger's **Separate** definition on slide S1 (6 October 2026, PDF p14, [shared deck](https://docs.google.com/presentation/d/1BdPlRCf742AcpfJVrAe_c3-OOp1ARiwuU9RMNflb0U8/edit)).
DES weights follow **case d** in `/n23data1/mkilbing/imsim_wdes_response/wdes_response.py`, functions `w_table`, `eval_table`, `responses`.
Neither split nor joint response on aligned metacal rows includes detection response.

For branch b, define A_b(x) = sum(S_b w_b x) / sum(S_b w_b).
Every branch has its own selection and denominator.
The artificial metacal step h defaults to 0.01; it is distinct from the manifest's externally injected simulation shear.

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
  des_min_count: 3
  additive_correction: true  # false for constant-shear simulations
```

`metacal_from_config(data, preselection, config['metacal'])` is the common entry point.
A mismatched `w_col` raises an error.
Cut products declare `WGHTSCH`, `RAPPLIED`, `MCSTEP`, `WPOLICY` and `IVPOLICY`.
Consumers must use the matching weight and `R: 1`; the catalogue already received the full matrix.
The configured catalogue reader and image-simulation reader refuse a second R or a different numerator weight on these products.
External catalogues without these metadata keep their existing interpretation.

At the Python API, `global_R_weight='w'` is deprecated and maps to `weight_scheme='iv'`, including squared G_ERR and weighted selection means.
Versioned release configurations without `weight_scheme` explicitly use the legacy response/c and unsquared-IV policy.
Extraction carries all ten branch G_ERR columns and writes corrected `w_iv` using the public variance helper; legacy calibration reconstructs the historical IV internally.
Modern IV requires sheared-branch errors: regenerate incomplete inputs from the tile-final catalogues and retain those columns in the merge allow-list.
The optional direct-calibration tail of `extract_info.py` refuses because it lacks a shared weight/selection configuration; use the comprehensive product and calibration driver.

DES cells are right-closed, including internal boundaries.
Finite coordinates outside the table range use the nearest edge cell; nonfinite coordinates get zero weight.
Below `des_min_count`, training moments are pooled over the smallest clipped concentric square reaching that count (or the entire table if necessary).
Counts remain the original, unpooled counts; unpooled empty/invalid cells get zero weight.
A row selected only on a sheared branch still receives a fixed-table weight.

## Spatial uncertainties and simulation calibration

`save_response_product` adds W_DES, W_COUNT, W_EDGES, R_JK and R_COV extensions when spatial groups are supplied.
R_JK deletes each shared TILE_ID from every branch and recomputes the selected population and its denominator.
R_COV uses (K−1)/K times the sum of outer products of centered delete-one estimates.
Its ordering is row-major R_shear, R_sel, R_total, R_joint, c_e, c_g (20 elements).
For a self-fitted DES table, the writer refits W on each retained no-shear sample, evaluates it on all five artificial branches and stores the calibrated deleted means in `R_JK.c_g1,c_g2`.
`response_jackknife(groups, refit_w=False)` supplies the conditional fixed-W comparison.
The returned c_err is a weighted error of the raw mean (legacy mode keeps historical scatter); correlated response/c uncertainty belongs in R_COV.

Image simulations use a named, independent DES data-table product, declared by `image_sim_config.calibration_config` and the image-simulation workflow configuration.
The product contains W, counts and bin edges, with its name recorded in the calibrated catalogue metadata.
Self-fitted weights correlate with the measured shapes: the deterministic test in `test_des_table_independence.py` gives m = −1/N_cell for self-fitting and zero for independent training.
An external table stays fixed during spatial deletions; its training uncertainty is not included.
The production `ImageSimMBias` estimator uses full selected populations with separate branch weights and shared-tile response-recalibrating jackknife covariance for (m1,c1,m2,c2).
Modern products reject pair matching; symmetric pair weights omit a numerator contribution retained in the response and are diagnostic only.
Residual m remains a separate result, not an extra correction applied silently to catalogue e.

## Diagnostics and reproduction

`response_diagnostics.plot_response` plots per-object R_g, spatial delete-one R_sel/total/joint, split−joint differences and tile-mean R_g maps.
R_sel is an ensemble derivative, never a per-object response.
`pair_products` measures all 16 angle-binned, pair-weighted products of R_g elements.
`scalar_pair_response` measures <r_i r_j>_ww/<r>_w²−1 with r=trace(R_g)/2 and shared spatial deletions.
These shear-only diagnostics omit selection/weight derivatives and cannot certify total two-point response adequacy.

See `scripts/validation/response/README.md` for unblinded case-d reproduction, estimator comparison, independent-table residual m and occupancy-reweighted m.
No generated measurements or plots are stored in git.
