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
