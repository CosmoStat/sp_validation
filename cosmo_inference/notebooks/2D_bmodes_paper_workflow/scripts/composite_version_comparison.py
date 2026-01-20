# %%
"""Composite version comparison panels for appendix figures.

Combines three pre-rendered PNG panels horizontally into a single figure.
Used for version comparison composites (Pure E/B xip, xim, and Cl).
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython import get_ipython


ipython = get_ipython()

if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
    ipython.run_line_magic("matplotlib", "inline")


def _load_snakemake():
    if hasattr(sys, "ps1"):
        from snakemake_helpers import snakemake_interactive

        return snakemake_interactive(
            "results/claims/pte_version_comparison_pure_eb_xip/trace.auto.json",
            str(Path.cwd()),
        )
    from snakemake.script import snakemake

    return snakemake


snakemake = _load_snakemake()


def composite_horizontal(panel_paths, output_path, panel_width=2.41, panel_height=2.41, dpi=300):
    """Combine panels horizontally into a single figure.

    Parameters
    ----------
    panel_paths : list of str/Path
        Paths to PNG images to combine (left to right).
    output_path : str/Path
        Output composite figure path.
    panel_width : float
        Width of each panel in inches.
    panel_height : float
        Height of each panel in inches.
    dpi : int
        Output resolution.

    Returns
    -------
    dict
        Composite metadata for trace.
    """
    n_panels = len(panel_paths)
    total_width = panel_width * n_panels

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(total_width, panel_height),
        constrained_layout=True
    )

    # Handle single panel case
    if n_panels == 1:
        axes = [axes]

    for ax, panel_path in zip(axes, panel_paths):
        img = mpimg.imread(str(panel_path))
        ax.imshow(img)
        ax.axis("off")

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "n_panels": n_panels,
        "dimensions_inches": [total_width, panel_height],
        "source_panels": [str(p) for p in panel_paths],
    }


def main():
    params = snakemake.params
    composite_type = params.composite_type

    trace_path = Path(snakemake.output.trace)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    output_path = Path(snakemake.output.composite_figure)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect input panel paths
    input_panels = list(snakemake.input.panels)

    print(f"Compositing {len(input_panels)} panels for {composite_type}")
    for i, panel in enumerate(input_panels):
        print(f"  Panel {i+1}: {panel}")

    # Create composite
    composite_meta = composite_horizontal(
        input_panels,
        output_path,
        panel_width=2.41,
        panel_height=2.41,
        dpi=300,
    )

    print(f"Saved composite to {output_path}")

    # Copy to paper figures directory if specified
    paper_fig_path = snakemake.output.get("paper_figure", None)
    if paper_fig_path:
        shutil.copy2(output_path, paper_fig_path)
        print(f"Copied to paper figures: {paper_fig_path}")

    # Write trace
    trace = {
        "claim": params.claim,
        "implementation_plan": params.implementation_plan,
        "evidence": {
            "composite_type": composite_type,
            "n_panels": composite_meta["n_panels"],
            "dimensions_inches": composite_meta["dimensions_inches"],
            "source_panels": composite_meta["source_panels"],
            "creation_timestamp": datetime.now().isoformat(),
        },
        "artifact_paths": {
            "composite_figure": str(output_path),
        },
        "parameters": {
            "kind": params.kind,
            "composite_type": composite_type,
        },
        "depends_on": params.depends_on,
        "ai_review": None,
    }

    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=2)
    print(f"Saved trace to {trace_path}")


if __name__ == "__main__":
    main()
