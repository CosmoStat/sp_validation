# Unblinded response validation

Run from the repository root inside the container on an allocated compute node.
Prefix each command with `srun --jobid=<id> -n 1 -c 8 --mem=45G spv-container exec` and set `PYTHONPATH=src`.
Choose an output directory outside git, e.g. `OUT="$PWD/response-results"`.
The commands below use the shared simulation and reference directories and the archived independent DES table by default; no personal scratch inputs are required.

## Case-d reproduction and estimator comparison

```bash
for g in 1 2 3 4; do
  python scripts/validation/response/validate_response.py --grid "$g" --output "$OUT/martin/grid_$g"
done
python scripts/validation/response/summarize_response.py "$OUT/martin" "$OUT/martin"
python scripts/validation/response/plot_response_m_estimators.py "$OUT/martin" "$OUT/martin"
```

The summary asserts agreement in R, W and paired m to 1e-14 on all 20 products, using Martin's unpooled cells (`N_min=0`).
The plot compares symmetric paired, matched branch-weighted and full-population estimators.

## Independent-table production residual m

```bash
for g in 1 2 3 4; do
  python scripts/validation/response/v2_data_table.py --grid "$g" --output "$OUT/production"
done
python scripts/validation/response/summarize_v2_table.py "$OUT/production"
```

## Shear-only and selection-response split

Measure full-matrix shear-only m and the selection-response contribution with identical numerators and shared-tile deletions, combining four grids with common total-m weights to preserve the split.

```bash
python scripts/validation/response/m_split.py --output "$OUT/m_split"
```

## SNR-resolved m and a data selection

Measure each grid with the production parent cuts and the named fixed data table.
Each metacal branch is selected using its own SNR in a bin, so its selection response includes bin-edge migration.
The default edges are 10, 15, 20, 30, 50, 100 and 500; override them with `--edges`.
Bins are left-open and right-closed, intersected with the production cuts.

```bash
for g in 1 2 3 4; do
  python scripts/validation/response/snr_m.py --grid "$g" --output "$OUT/snr"
done
python scripts/validation/response/summarize_snr_m.py "$OUT/snr" "$OUT/snr/m_snr.fits"
```

The FITS product records the weight scheme, table name and path, bin-major values `(m1,c1,m2,c2)`, and their full cross-bin covariance.
The per-grid NPZ files retain the shared-tile deletions; the JSON files retain branch counts.
Grids are combined independently, with inverse-variance weights per bin and component; the full covariance uses those same weights.
The table-training uncertainty and detection response are not included.

For a selected catalogue, `sim_m_snr.effective_m` accepts each bin's no-shear sum of weights W and its ensemble 2×2 total response R = R_shear + R_sel.
It returns m with shares proportional to W times the diagonal total response, the propagated simulation covariance and the shares.
This diagonal, linear approximation neglects matrix mixing and bin-fraction derivative terms; separately corrected bins combine differently.
Measure a new total response for every data sub-selection and test transfer of the simulation calibration to that selection.

The closure check measures all four grids and compares total-share recombination with direct full-population m, including paired tile-deletion errors:

```bash
for g in 1 2 3 4; do
  python scripts/validation/response/closure_snr_m.py --grid "$g" --output "$OUT/snr"
done
python scripts/validation/response/closure_snr_m.py --output "$OUT/snr"
```

For the halo-clear fiducial v2 data selection, `data_snr_hist_v2.json` contains 62,309,299 galaxies and measured per-bin shear, selection and total responses using the fixed `v2-d533a078` DES table.
The product is available as a companion file to the `where-does-425-s-residual-m` felt thread and [on the PR assets branch](https://raw.githubusercontent.com/CosmoStat/sp_validation/pr-assets/425-stack/data_snr_hist_v2.json); use its path for `--histogram`.
Its SHA-256 is `6a93f7232d64caf46d1fd5327d353d8d3e573fbcd4c78826d4d3d4d84ef91dc8`.
The measured data binning is `[lower, upper)`, whereas the simulations use `(lower, upper]`; the effect of objects exactly on internal edges has not been quantified.
Data histogram and response covariance, weight-table uncertainty and data-transfer uncertainty are not included.
The comparison uses paired simulation tile deletions for its difference from the full-population m.
The script also compares the no-shear-weighted mean of the bin response matrices to the directly measured full-sample response; they need not agree exactly because selection responses are separately normalised.

```bash
python scripts/validation/response/data_snr_m.py \
  --product-snr "$OUT/snr/m_snr.fits" \
  --histogram /path/to/data_snr_hist_v2.json \
  --production "$OUT/production" --output "$OUT/snr"
```

## FLAGS comparison

The following reproduces the production path with the FLAGS parent cut kept and removed, then combines per-grid paired jackknife differences:

```bash
for g in 1 2 3 4; do
  python scripts/validation/response/flags_comparison.py --grid "$g" --output "$OUT/flags"
done
python scripts/validation/response/summarize_flags.py "$OUT/flags"
```

## Simulation/data occupancy and response

```bash
python scripts/validation/response/sim_vs_data_response.py --output "$OUT/occupancy"
python scripts/validation/response/summarize_sim_vs_data_response.py "$OUT/occupancy"
python scripts/validation/response/occupancy_residual_m.py --output "$OUT/occupancy" --production "$OUT/production"
```

Measurement scripts accept `--base` (Martin's comprehensive grids), `--reference` (his calibrated case-d products), `--product` and `--product-name` (independent DES data table).
Their `--help` shows site-specific defaults; the table defaults to the named `v2-d533a078` FITS product at `/n17data/cdaley/unions/validation/2026-10-26_response-425/products/des_weight_product-d533a078.fits`.
That directory's `INDEX.md` records the table's provenance and checksum; use `--product /path/to/des_weight_product-d533a078.fits` to override its location.
FITS, JSON, NPZ and PNG outputs all go under the supplied output directories; no archived review-round outputs are required.
The estimator plot optionally accepts `--production "$OUT/production"` after the production summary exists.
