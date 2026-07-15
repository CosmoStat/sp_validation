#!/usr/bin/env python
"""Compute multiplicative and additive shear bias from image simulations.

Usage:
    compute_m_bias_image_sims.py -c config.yaml [-v] [--cumulative] [--n_tiles N]
"""

import argparse
import os
import sys

# Configure matplotlib for non-interactive backend
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sp_validation.image_sims import ImageSimMBias


def to_python(obj):
    """Recursively cast numpy scalars/arrays to plain Python types.

    ``ImageSimMBias.run`` returns a nested dict of numpy floats; dumping those
    to YAML with ``yaml.dump`` writes opaque ``!!python/object`` binary tags. A
    recursive pass down the results tree (dicts, lists, arrays, scalars) leaves
    a clean, human-readable, ``safe_load``-able document.
    """
    if isinstance(obj, dict):
        return {key: to_python(val) for key, val in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(val) for val in obj]
    if isinstance(obj, np.ndarray):
        return float(obj.item()) if obj.size == 1 else obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    return obj


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-c", "--config", required=True, help="config YAML file")
    p.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    p.add_argument(
        "--cumulative",
        action="store_true",
        default=True,
        help="track convergence as tiles accumulate (default: True)",
    )
    p.add_argument(
        "--n_tiles", type=int, help="number of tiles (auto-detected if not given)"
    )
    return p.parse_args()


def get_n_tiles(grids_dir, num):
    """Detect number of tiles from final_cat HDF5 files."""
    try:
        import h5py

        # Count tiles in first sim's final_cat
        for sim in ["1z2z_grid", "1m2z_grid", "1p2z_grid", "1z2m_grid", "1z2p_grid"]:
            sim_name = f"{sim}_{num}"
            final_cat = os.path.join(grids_dir, sim_name, f"final_cat_{sim_name}.hdf5")
            if os.path.isfile(final_cat):
                with h5py.File(final_cat, "r") as hf:
                    if "patches" in hf:
                        n_tiles = sum(
                            1 for patch in hf["patches"] for _ in hf[f"patches/{patch}"]
                        )
                        return n_tiles
    except Exception:
        pass
    return None


def update_cumulative_file(cumulative_path, n_tiles, results):
    """Update the cumulative m/c bias tracking file.

    Writes ``results`` under the ``n_tiles`` key, *overwriting* an existing
    entry for that count.  The earlier behaviour silently skipped when the key
    was already present, which meant a re-run against fixed catalogues left the
    old (possibly wrong) number in place -- a stale value masquerading as
    current.  A fresh run is the authority for its tile count, so it overwrites.

    Returns ``True`` when a new key was added, ``False`` when an existing entry
    was overwritten (the file is written either way).
    """
    if os.path.isfile(cumulative_path):
        with open(cumulative_path) as f:
            try:
                cumulative = yaml.safe_load(f) or {}
            except yaml.YAMLError:
                # Legacy file written before the to_python cleanup: it carries
                # numpy python-object tags that safe_load rejects. Load it
                # unsafely, then the to_python pass on write heals it in place.
                f.seek(0)
                cumulative = yaml.unsafe_load(f) or {}
    else:
        cumulative = {}

    is_new = str(n_tiles) not in cumulative
    cumulative[str(n_tiles)] = results
    with open(cumulative_path, "w") as f:
        yaml.dump(to_python(cumulative), f, default_flow_style=False)
    return is_new


def plot_convergence(cumulative_path, diagnostics_dir):
    """Create convergence plots: m/c vs n_tiles and errors vs n_tiles."""
    os.makedirs(diagnostics_dir, exist_ok=True)

    try:
        with open(cumulative_path) as f:
            cumulative = yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: could not read cumulative file {cumulative_path}: {e}")
        return

    if not cumulative:
        print("No cumulative data yet, skipping plots")
        return

    # Sort by n_tiles
    n_tiles_list = sorted([int(k) for k in cumulative.keys()])
    m1_vals = []
    m1_err_vals = []
    c1_vals = []
    c1_err_vals = []
    m2_vals = []
    m2_err_vals = []
    c2_vals = []
    c2_err_vals = []

    for n in n_tiles_list:
        res = cumulative[str(n)]
        m1_vals.append(res["m1"])
        m1_err_vals.append(res["m1_err"])
        c1_vals.append(res["c1"])
        c1_err_vals.append(res["c1_err"])
        m2_vals.append(res["m2"])
        m2_err_vals.append(res["m2_err"])
        c2_vals.append(res["c2"])
        c2_err_vals.append(res["c2_err"])

    n_tiles_str = (
        f"n_tiles = {n_tiles_list}"
        if len(n_tiles_list) > 1
        else f"n_tiles = {n_tiles_list[0]}"
    )

    # Plot 1: m and c with error bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"m and c convergence ({n_tiles_str})", fontsize=12)

    ax1.errorbar(
        n_tiles_list, m1_vals, yerr=m1_err_vals, fmt="o-", label="m1", capsize=5
    )
    ax1.errorbar(
        n_tiles_list, m2_vals, yerr=m2_err_vals, fmt="s-", label="m2", capsize=5
    )
    ax1.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax1.set_xlabel("Number of tiles")
    ax1.set_ylabel("Multiplicative bias m")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.errorbar(
        n_tiles_list, c1_vals, yerr=c1_err_vals, fmt="o-", label="c1", capsize=5
    )
    ax2.errorbar(
        n_tiles_list, c2_vals, yerr=c2_err_vals, fmt="s-", label="c2", capsize=5
    )
    ax2.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax2.set_xlabel("Number of tiles")
    ax2.set_ylabel("Additive bias c")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot1_path = os.path.join(diagnostics_dir, "mbias_convergence.png")
    plt.savefig(plot1_path, dpi=150)
    plt.close()
    print(f"Saved convergence plot to {plot1_path}")

    # Plot 2: error bars only
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Error convergence ({n_tiles_str})", fontsize=12)

    ax1.errorbar(
        n_tiles_list,
        [0] * len(n_tiles_list),
        yerr=m1_err_vals,
        fmt="o-",
        label="m1 error",
        capsize=5,
        alpha=0.7,
    )
    ax1.errorbar(
        n_tiles_list,
        [0] * len(n_tiles_list),
        yerr=m2_err_vals,
        fmt="s-",
        label="m2 error",
        capsize=5,
        alpha=0.7,
    )
    ax1.set_xlabel("Number of tiles")
    ax1.set_ylabel("Multiplicative bias error")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2.errorbar(
        n_tiles_list,
        [0] * len(n_tiles_list),
        yerr=c1_err_vals,
        fmt="o-",
        label="c1 error",
        capsize=5,
        alpha=0.7,
    )
    ax2.errorbar(
        n_tiles_list,
        [0] * len(n_tiles_list),
        yerr=c2_err_vals,
        fmt="s-",
        label="c2 error",
        capsize=5,
        alpha=0.7,
    )
    ax2.set_xlabel("Number of tiles")
    ax2.set_ylabel("Additive bias error")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plot2_path = os.path.join(diagnostics_dir, "mbias_errors.png")
    plt.savefig(plot2_path, dpi=150)
    plt.close()
    print(f"Saved errors plot to {plot2_path}")


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Config: {args.config}")
    print(f"Grids : {config['grids_dir']}")
    print(f"Run   : grid_{config['num']}")
    print(f"g_in  : ±{config['shear_amplitude']}")
    print()

    # Auto-detect n_tiles if --cumulative
    if args.cumulative and not args.n_tiles:
        n_tiles = get_n_tiles(config["grids_dir"], config["num"])
        if n_tiles:
            args.n_tiles = n_tiles
            print(f"Auto-detected {n_tiles} tiles")

    mb = ImageSimMBias(config)

    print("Loading catalogues...")
    mb.load_catalogs(verbose=args.verbose)

    # ``run`` returns a document with the primary scheme's m/c mirrored at the
    # top level plus a per-scheme ``weights`` block. Cast the whole tree to
    # plain Python floats so the YAML/text output is human-readable (raw numpy
    # scalars serialise as !!python/object binary).
    results = to_python(mb.run(verbose=True))

    print()
    print("=" * 40)
    print("  Results")
    print("=" * 40)
    for scheme, res in results["weights"].items():
        print(f"  weights: {scheme}")
        print(f"    m1 = {res['m1']:+.4f} +-{res['m1_err']:.4f}")
        print(f"    c1 = {res['c1']:+.4f} +-{res['c1_err']:.4f}")
        print(f"    m2 = {res['m2']:+.4f} +-{res['m2_err']:.4f}")
        print(f"    c2 = {res['c2']:+.4f} +-{res['c2_err']:.4f}")
    print("=" * 40)

    # Cumulative tracking
    if args.cumulative:
        results_dir = config.get(
            "diagnostics_dir", config.get("results_dir", "results")
        )
        os.makedirs(results_dir, exist_ok=True)
    else:
        results_dir = None

    # Output path: in results dir if cumulative, else from config or current dir
    if results_dir:
        out_path = os.path.join(results_dir, "m_bias_results.yaml")
    else:
        out_path = config.get("output_path", "m_bias_results.yaml")

    # A result file describes itself: the provenance block the rule assembled
    # (manifest hash, both repos' branch+commit, container sif + GHCR revision)
    # rides verbatim from the config into the output yaml.  It is appended as a
    # separate top-level key, so the numeric m/c fields serialise byte-for-byte
    # as before -- the reproduction gate sees only the added `provenance:` block.
    output = dict(results)
    if "provenance" in config:
        output["provenance"] = config["provenance"]

    with open(out_path, "w") as f:
        yaml.dump(output, f, default_flow_style=False)
    print(f"Results written to {out_path}")

    # Also write to text file for readability
    if results_dir:
        txt_path = os.path.join(results_dir, "m_bias_results.txt")
    else:
        txt_path = config.get("output_path", "m_bias_results.yaml").replace(
            ".yaml", ".txt"
        )

    with open(txt_path, "w") as f:
        f.write("Multiplicative and additive shear bias from image simulations\n")
        f.write("=" * 60 + "\n")
        for scheme, res in results["weights"].items():
            f.write(f"\nweights: {scheme}\n")
            f.write(f"  m1 = {res['m1']:+.6f} ± {res['m1_err']:.6f}\n")
            f.write(f"  c1 = {res['c1']:+.6f} ± {res['c1_err']:.6f}\n")
            f.write(f"  m2 = {res['m2']:+.6f} ± {res['m2_err']:.6f}\n")
            f.write(f"  c2 = {res['c2']:+.6f} ± {res['c2_err']:.6f}\n")
        f.write(
            "\nErrors computed via bootstrap resampling "
            f"(n={config['n_bootstrap']} resamples)\n"
        )
    print(f"Results written to {txt_path}")

    if args.cumulative:
        cumulative_path = os.path.join(results_dir, "mbias_cumulative.yaml")
        if args.n_tiles:
            added = update_cumulative_file(cumulative_path, args.n_tiles, results)
            verb = "Added" if added else "Overwrote"
            print(f"\n{verb} n_tiles={args.n_tiles} in {cumulative_path}")
            # Regenerate plots after every update (an overwrite can shift the
            # curve, so the plots must track it -- not just fresh additions).
            try:
                plot_convergence(cumulative_path, results_dir)
            except Exception as e:
                print(
                    f"Warning: could not generate convergence plots: {e}",
                    file=sys.stderr,
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
