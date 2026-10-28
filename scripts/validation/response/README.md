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

For a selected catalogue, `sim_m_snr.effective_m` accepts its no-shear SNR, weight and per-object diagonal responses, or a `(bins, 2)` histogram of sums of `w * R_ii`.
It returns the response-weighted m, its propagated simulation covariance and the shares.
This is a response-shear-weighted approximation when per-object `R_g` is supplied: ensemble selection response is not a per-object quantity and is not included in those data shares.
It also assumes that the binned simulation m transfers to the data selection.

For the archived v2 cell product, the following supplies a provisional estimate by interpolating the weighted response sums uniformly in log SNR within each table cell; the data histogram is not exact on the default bin edges.
The output comparison uses paired tile deletions for its difference from the full-population m.

```bash
python scripts/validation/response/data_snr_m.py \
  --product-snr "$OUT/snr/m_snr.fits" \
  --cells /n17data/cdaley/unions/validation/2026-10-26_response-425/response/cell_response-d533a078.npz \
  --production "$OUT/production" --output "$OUT/snr"
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
