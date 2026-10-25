"""Summarize validate_response.py outputs; run plotting on a compute node."""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize(root, output):
    output.mkdir(parents=True, exist_ok=True)
    results = [
        json.loads((root / f"grid_{i}/validation.json").read_text())
        for i in range(1, 5)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for comp, ax in enumerate(axes, 1):
        x = [r["grid"] for r in results]
        y = [r["m"][f"m{comp}"] for r in results]
        ax.plot(
            x,
            [v["martin_paired"] for v in y],
            "o",
            ms=9,
            mfc="none",
            label="Martin paired case d",
        )
        ax.plot(x, [v["new_paired"] for v in y], "x", label="New paired reproduction")
        ax.errorbar(
            [i + 0.08 for i in x],
            [v["new_unpaired"] for v in y],
            yerr=[v["unpaired_fixed_W_tile_jk_error"] for v in y],
            fmt=".",
            capsize=3,
            label="New unpaired, fixed-W tile JK",
        )
        ax.axhline(0, color="k", lw=0.7)
        ax.set(xlabel="Simulation grid", ylabel=f"Residual m{comp}", xticks=x)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "m_comparison.png", dpi=160)
    plt.close(fig)
    for name in (
        "response_distributions",
        "response_jackknife",
        "response_map",
        "split_joint",
    ):
        shutil.copy(root / f"grid_1/1z2z_grid_1/{name}.png", output)
    shutil.copy(root / "grid_1/pair_response.png", output)
    for r in results:
        shutil.copy(
            root / f"grid_{r['grid']}/validation.json",
            output / f"grid_{r['grid']}.json",
        )
    rows = []
    for r in results:
        a, b = r["m"]["m1"], r["m"]["m2"]
        rows.append(
            f"| {r['grid']} | {a['martin_paired']:+.6f}, {b['martin_paired']:+.6f} | "
            f"{a['new_paired']:+.6f}, {b['new_paired']:+.6f} | "
            f"{a['new_unpaired']:+.6f} ± {a['unpaired_fixed_W_tile_jk_error']:.6f} | "
            f"{b['new_unpaired']:+.6f} ± {b['unpaired_fixed_W_tile_jk_error']:.6f} |"
        )
    table = "\n".join(
        [
            "| Grid | Martin paired (m1, m2) | New paired (m1, m2) | New unpaired m1 | New unpaired m2 |",
            "| --- | --- | --- | --- | --- |",
        ]
        + rows
    )
    max_r = max(
        b["max_abs_R_difference"] for r in results for b in r["branches"].values()
    )
    max_w = max(
        b["max_relative_W_difference"] for r in results for b in r["branches"].values()
    )
    max_m = max(
        abs(v["new_paired"] - v["martin_paired"])
        for r in results
        for v in r["m"].values()
    )
    perf = "\n".join(
        f"| {r['grid']} | {r['seconds']:.1f} | {r['max_rss_mib']:.0f} | "
        f"{100 * r['stage1']['max_diagonal_fraction']:.2f}% |"
        for r in results
    )
    text = f"""# DES case-d validation

These are observed results on Martin's four unblinded image-simulation grids.
Each grid contains five external-shear branches and 39 shared tile labels.
The injected shear is 0.025 from the existing manifest-derived m config; the metacal step is 0.01.
The measured code revision is recorded in each linked JSON file.

{table}

The paired reproduction matches cuts (including FLAGS≤2), 20×20 table, right-closed interior cells, pair matching radius and symmetric pair weights.
Across all 20 external-shear products, max |R_new−R_Martin| = {max_r:.3g}, max relative W difference = {max_w:.3g}, and max paired-m difference = {max_m:.3g}.
The comparison uses `/n23data1/mkilbing/imsim_wdes_response/grid_*/des_d/results/m_bias_results.yaml`, specifically `weights.w_des`, not its unweighted top-level headline.

The unpaired estimator deliberately uses each external branch's full selected population, with its own weights and response.
Its errors delete the same tile on both external branches and recalibrate R for each deletion, conditional on the fixed W table.
These errors are not conditional fixed-response object-bootstrap errors.
The paired comparison remains diagnostic because matching changes the population without recalibrating its selection response.
Squaring IV errors changes exported w_iv but cannot change pure-DES W/R/m; IV proxies are not used.
No mean shear is subtracted from these constant-shear simulations.
These runs validate agreement with Martin's implementation, not a new fiducial residual-m calibration for blinded v2 data.

![Residual m](m_comparison.png)

## Response distributions and spatial diagnostics

![Per-object R_g](response_distributions.png)
![Delete-one R_sel, split and joint](response_jackknife.png)
![Split minus joint](split_joint.png)
![Tile-mean R_g map](response_map.png)

## Two-point response: stage 1 only

The pair diagnostic measures all 16 products of R_g elements with DES pair weights in eight coarse bins from 1–120 arcmin, using the no-external-shear simulation branch.
The map above uses the same sample's mean R_g across tiles.
The first panel shows the diagonal pair products relative to global weighted products; the second shows products involving off-diagonal elements minus their global products.
The two diagonal products are shown separately, without imposing isotropy.

![Shear-only pair/global comparison, grid 1](pair_response.png)

| Grid | Wall time [s] | Peak RSS [MiB] | Max diagonal pair/global deviation |
| --- | --- | --- | --- |
{perf}

Wall time/RSS include five calibrations, fixed-W tile covariance, plotting, scalar pair counting and matching on four allocated cores.
Per-branch timings are in [grid 1](grid_1.json), [grid 2](grid_2.json), [grid 3](grid_3.json), [grid 4](grid_4.json).
The original comprehensive catalogues are read in their existing locations; no catalogue is committed here.

**No total-response adequacy verdict is justified.**
The measured scale dependence is shear-only; selection/weight derivatives and a calibration budget agreed with Martin are still missing.
This implements stage 1 and leaves #23 open.
No Appendix-A mixed derivatives or blinded v2 signal were measured.

## Reproduction

On allocation 1037389, from this checkout:

```bash
TMPDIR=/n17data/cdaley/tmp MPLCONFIGDIR=/n17data/cdaley/tmp/mpl \\
srun --jobid=1037389 -n 1 -c 4 apptainer exec --bind /automnt,/n17data,/n23data1 \\
  --env PYTHONPATH=$PWD/src ~/.cache/sp_validation/sp_validation.sif \\
  python scripts/validation/validate_response.py --grid 1 \\
  --output /n17data/cdaley/scratch/response-pr/grid_1
```

Repeat for grids 2–4, then run `scripts/validation/summarize_response.py` on a compute node.
Fixed-W covariance is primary here; whether W should be refit in each spatial deletion remains a review question, not an implemented alternative.
"""
    (output / "README.md").write_text(text)
    print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summarize(args.root, args.output)
