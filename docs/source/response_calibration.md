# Response, weights and additive bias

One module, {mod}`sp_validation.response`, defines the metacalibration response, the galaxy weights, the additive bias and the image-simulation multiplicative bias.
Every catalogue calibration, jackknife and image-simulation measurement is a few calls to it.
No detection response is inferred from aligned metacal branches.

## Definitions

For metacal branch b (no-shear `ns`, and `1p`, `1m`, `2p`, `2m` sheared by ±h along each component), let S_b be the selection made on that branch and w^(b) the weight evaluated on that branch.
Write A_b(x) = Σ_{S_b} w^(b) x / Σ_{S_b} w^(b): every branch mean has its own selection and its own denominator.
The metacal step h defaults to 0.01; it is distinct from the shear injected into image simulations.

| Quantity | Definition |
| --- | --- |
| Selection S_b | `flag == 0`, `snr_min < FLUX/FLUX_ERR < snr_max`, `size_min < size < size_max` on branch b, strict, finite coordinates; with `gal_size_corr_ell` the size cut uses size·(1−\|e\|²)/(1+\|e\|²) |
| Per-object R_g | (e_i^{j+} − e_i^{j−})/(2h), all four elements, on S_ns |
| DES table W | (Σ r)² / (n Σ \|e\|²/2) per (S/N, size) cell, unweighted, with r = (R_g,11 + R_g,22)/2, trained on S_ns |
| DES weights | W evaluated at each branch's own S/N = FLUX/FLUX_ERR and size = T/T_PSF_RECONV |
| IV weights | 1/(2σ_ε² + G1_ERR² + G2_ERR²), with each branch's own measured errors |
| R_shear | A_ns(R_g) |
| R_sel | (A_{j+}(e_ns) − A_{j−}(e_ns))/(2h); includes cuts, weight changes and cell migration |
| R_total | R_shear + R_sel, the applied response |
| R_joint | (A_{j+}(e_{j+}) − A_{j−}(e_{j−}))/(2h), a diagnostic |
| c_e, c_g | c_e = A_ns(e), raw; c_g = R_total⁻¹ c_e |
| Calibrated shear | R_total⁻¹ (e − c_e); R_total⁻¹ e for constant-shear simulations |

### Separate normalisation

For the selection and joint responses of component j:

```math
R^{\rm sel}_{ij} = \frac{1}{2h} \left[
  \frac{\sum_{S_{j+}} w^{(j+)} e_i^{(\rm ns)}}{\sum_{S_{j+}} w^{(j+)}}
  - \frac{\sum_{S_{j-}} w^{(j-)} e_i^{(\rm ns)}}{\sum_{S_{j-}} w^{(j-)}}
\right],
\qquad
R^{\rm joint}_{ij} = \frac{1}{2h} \left[
  \frac{\sum_{S_{j+}} w^{(j+)} e_i^{(j+)}}{\sum_{S_{j+}} w^{(j+)}}
  - \frac{\sum_{S_{j-}} w^{(j-)} e_i^{(j-)}}{\sum_{S_{j-}} w^{(j-)}}
\right].
```

The two denominators differ, since shear moves objects across the cuts and changes their weights.
R_sel differs from R_joint only in the numerator ellipticity: no-shear for R_sel, the branch's own for R_joint.

The estimator applied to data is the ratio ⟨e⟩_w = Σ w e / Σ w.
Its response is the derivative of the ratio, which has a normalisation term:

```math
\frac{\partial \langle e \rangle_w}{\partial g}
= \frac{1}{\sum w} \frac{\partial \sum w e}{\partial g}
- \langle e \rangle_w \, \frac{1}{\sum w} \frac{\partial \sum w}{\partial g}.
```

Finite differences of separately normalised means include both terms, so R is the response of the weighted mean it calibrates.
A common denominator for both branches drops the second term, and symmetric pair weights ½(w₊ + w₋) on matched objects remove the change of the weight from the numerator.

## One weight, one set of sums

Every weight is w = u × W[cell]: `cell` is the object's (S/N, size) cell on the branch being averaged, W a table on those cells, and u a per-object factor.
DES weights have u = 1 and the fitted table; IV weights have the inverse variance as u and the all-ones table (`unit_table`); unit weights have u = 1 and the all-ones table.
Every branch mean is therefore a table contracted with per-cell, u-weighted sums of the selected rows.

{class}`~sp_validation.response.Sums` holds those sums per spatial group (tile) and cell: for each branch Σu, the row count, Σu e_ns and Σu e_b over that branch's own selection in the cells of its own coordinates; Σu R_g on S_ns; and the table's unweighted training moments n, Σr and Σ|e|²/2 over S_ns rows whose shapes and response are finite.
Sums of disjoint rows add (`+`), so a catalogue read in chunks reduces to the same object as one held in memory; `accumulate` builds them from rows, and `save`/`load` keep them on disk.
`estimate(sums, table)` returns R_shear, R_sel, R_total, R_joint, c_e, c_g and each branch's row count and weight sum.

## The table

Cells are logarithmic between the S/N and size cuts, `des_num_bins` a side (`table_edges`), and right-closed; finite values outside the range fall in the nearest edge cell.
The table coordinate is always the uncorrected T/T_PSF_RECONV, also when the size cut is ellipticity-corrected.
A cell with fewer than `des_min_count` training objects takes the moments of the smallest square of cells centred on it, clipped at the table edges, that holds `des_min_count`, or of the whole table (`pool`); window moments are summed directly, never differenced from cumulative sums.
Training counts stay unpooled, and a cell whose pooled n Σ|e|² is zero gets zero weight.

`fit_table(sums, min_count)` fits a table on the training moments of the sums; `write_table` and `read_table` keep it as a FITS product (`W_DES`, `W_COUNT`, `W_EDGES`; header `WGHTSCH`, `WSNRCOL`, `WSIZDEF`, `DESNMIN`, `DESMERGE`).

## Spatial uncertainties

`jackknife(sums, table)` deletes each group from every branch and re-estimates, returning the deleted estimates and their covariance (K−1)/K Σ δδᵀ in the order R_shear, R_sel, R_total, R_joint (row-major), c_e, c_g (20 elements).
A table passed as a {class}`~sp_validation.response.Table` stays fixed; the rule `refit(min_count)` refits the table on each deletion's retained training moments, so the error includes the table's own noise.
A table independent of the catalogue (the data's table, applied to simulations) stays fixed: its training noise is not part of the simulations' error.

`additive_bias(e, w, groups)` gives c of rows held in memory with its shape-noise error √(Σw²(e−c)²)/Σw and its delete-one-group jackknife error; on the selected no-shear rows with their weights it equals c_e and its jackknife from the sums.

## Catalogue calibration

`calibrate_catalogue(rows, metacal, preselection=, groups=)` runs all of this for a catalogue in memory under a `metacal` config block:

```yaml
metacal:
  weight_scheme: des        # or iv; required
  des_num_bins: 20
  des_min_count: 3
  sigma_eps_prior: 0.34     # IV weights
  weight_table: /path.fits  # optional: weight with this DES table, not a fitted one
  additive_correction: true # false for constant-shear simulations
```

For rows from the catalogue reader (the five `NGMIX_*` metacal branches) and a boolean preselection `keep`:

```python
import yaml
from sp_validation import response

with open("config/calibration/mask_v2.0.yaml") as f:
    metacal = yaml.safe_load(f)["metacal"]
cal = response.calibrate_catalogue(rows, metacal, preselection=keep, groups=rows["TILE_ID"])

R, c_e = cal["result"]["total"], cal["result"]["c_e"]
e1, e2 = cal["e"]  # calibrated shapes of the selected rows, rows[cal["index"]]
w = cal["w"]       # their weights
labels, deleted, covariance = cal["jackknife"]  # tile jackknife of every estimate
```

`scripts/calibration/calibrate_comprehensive_cat.py` calls it grouped by `TILE_ID` and writes the cut catalogue (weight column `w_<scheme>`, calibrated `e1`, `e2`), and beside it the response record (`_response.json`, `response.record`), the sums (`_sums.npz`) and, for DES, the table (`_des_weights.fits`).
The catalogue header declares `WGHTSCH`, `RAPPLIED` and `MEANSUB`; the configured catalogue reader refuses another weight column or a second response for such a catalogue.

## Image-simulation m and c

For shear component j, a simulation sheared by +g and one by −g, each with its own sums, give ĝ± = (R±)⁻¹ ⟨e⟩_ns± with each simulation's own response and no additive subtraction, and

```math
m_j = \frac{\hat g_j^+ - \hat g_j^-}{2 g_{\rm in}} - 1,
\qquad
c_j = \frac{\hat g_j^+ + \hat g_j^-}{2}.
```

`shear_bias(pairs, g_in, table)` measures them; `shear_bias_jackknife` deletes the same tile from every simulation at once and returns the full (m1, c1, m2, c2) covariance.
Each mean uses the full selected population of its simulation with its own weights, as the data estimator does, so m includes the selection response of the weighted mean.

The simulations take the data's selection and the data's DES table, held fixed: `calibrate_catalogue` refuses a `weight_table` whose cells are not those of the selection's cuts and `des_num_bins`, since an object outside the table's range would otherwise enter every mean with an edge cell's weight.
A table fitted on the simulations' own shapes correlates the weights with the measured shapes and suppresses the shear: for N objects per cell, a self-fitted table gives m = −1/N where an independent one gives zero.
`scripts/compute_m_bias_image_sims.py` reads the simulations' `_sums.npz` and writes `m_bias_results.yaml` (`m1`, `m1_err`, …, `mc_covariance`, `covariance_order`); the image-sims workflow's `weight_table` key names the table (`self` and `unit` are diagnostics).
Residual m is a result, not a correction applied to catalogue shapes.

## Measuring a variant

Each variant is a different table or a different calibrating response on the same sums.
With `pairs = {0: (plus1, minus1), 1: (plus2, minus2)}` the simulations' sums and `table` the data's table:

```python
from sp_validation import response

# m split into its shear-response and selection parts
m_total = response.shear_bias(pairs, g_in, table)
m_shear = response.shear_bias(pairs, g_in, table, response="shear")
selection_part = {k: m_total[k] - m_shear[k] for k in m_total}

# m in an S/N range: zero the table outside it
snr_mid = 0.5 * (table.snr_edges[1:] + table.snr_edges[:-1])
inside = (snr_mid > 20) & (snr_mid < 50)
m_snr = response.shear_bias(pairs, g_in, table._replace(w=table.w * inside[:, None]))

# occupancy-reweighted m: multiply the table by a per-cell ratio
ratio = data_counts / sim_counts  # per cell, e.g. from the two tables' counts
m_occupancy = response.shear_bias(pairs, g_in, table._replace(w=table.w * ratio))

# unit weights (u = 1 sums), and IV weights (sums accumulated with sigma_e)
m_unit = response.shear_bias(pairs, g_in, response.unit_table(table[:2]))
m_iv = response.shear_bias(iv_pairs, g_in, response.unit_table(table[:2]))

# the error of any of them
labels, deleted, covariance = response.shear_bias_jackknife(pairs, g_in, table)
```

Zeroing the table selects each branch on its own S/N, so objects that migrate across the range edge under shear contribute to the selection response, as a cut would.

The response of a subsample, such as a tomographic bin, is the same estimator on the subsample's rows with the full sample's table held fixed, so the weights are those applied to the catalogue:

```python
sums_bin = response.accumulate(rows, cuts, table[:2], preselection=in_bin, groups=tiles)
result_bin = response.estimate(sums_bin, table)
labels, deleted, covariance = response.jackknife(sums_bin, table)
```

The preselection holds the subsample's membership fixed across the five metacal branches: the result includes the response of the metacal cuts and of the weights within the subsample, not the response of whatever assigned the subsample (photometric redshifts, magnitudes measured on the unsheared image).
