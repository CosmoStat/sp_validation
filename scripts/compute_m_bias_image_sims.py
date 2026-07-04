#!/usr/bin/env python
"""Compute multiplicative and additive shear bias from image simulations.

Usage:
    compute_m_bias_image_sims.py -c config.yaml [-v] [--cumulative] [--n_tiles N]
"""

import sys
import os
import argparse
import yaml
import numpy as np

# Configure matplotlib for non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

from sp_validation.image_sims import ImageSimMBias


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-c", "--config", required=True, help="config YAML file")
    p.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    p.add_argument("--cumulative", action="store_true", default=True,
                   help="track convergence as tiles accumulate (default: True)")
    p.add_argument("--n_tiles", type=int, help="number of tiles (auto-detected if not given)")
    return p.parse_args()


def get_n_tiles(grids_dir, num):
    """Detect number of tiles from final_cat HDF5 files."""
    try:
        import h5py
        # Count tiles in first sim's final_cat
        for sim in ['1m2z_grid', '1p2z_grid', '1z2m_grid', '1z2p_grid']:
            sim_name = f"{sim}_{num}"
            final_cat = os.path.join(grids_dir, sim_name, f"final_cat_{sim_name}.hdf5")
            if os.path.isfile(final_cat):
                with h5py.File(final_cat, 'r') as hf:
                    if 'patches' in hf:
                        n_tiles = sum(1 for patch in hf['patches']
                                     for _ in hf[f'patches/{patch}'])
                        return n_tiles
    except Exception:
        pass
    return None


def update_cumulative_file(cumulative_path, n_tiles, results):
    """Update cumulative m/c bias tracking file."""
    if os.path.isfile(cumulative_path):
        with open(cumulative_path) as f:
            cumulative = yaml.safe_load(f) or {}
    else:
        cumulative = {}

    # Check if this n_tiles already exists
    if str(n_tiles) in cumulative:
        return False

    # Convert numpy types to Python floats for clean YAML
    clean_results = {}
    for key, val in results.items():
        if isinstance(val, np.ndarray):
            clean_results[key] = float(val.item()) if val.size == 1 else val.tolist()
        elif isinstance(val, (np.integer, np.floating)):
            clean_results[key] = float(val)
        else:
            clean_results[key] = val

    cumulative[str(n_tiles)] = clean_results
    with open(cumulative_path, 'w') as f:
        yaml.dump(cumulative, f, default_flow_style=False)
    return True


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
        m1_vals.append(res['m1'])
        m1_err_vals.append(res['m1_err'])
        c1_vals.append(res['c1'])
        c1_err_vals.append(res['c1_err'])
        m2_vals.append(res['m2'])
        m2_err_vals.append(res['m2_err'])
        c2_vals.append(res['c2'])
        c2_err_vals.append(res['c2_err'])

    n_tiles_str = f"n_tiles = {n_tiles_list}" if len(n_tiles_list) > 1 else f"n_tiles = {n_tiles_list[0]}"

    # Plot 1: m and c with error bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"m and c convergence ({n_tiles_str})", fontsize=12)

    ax1.errorbar(n_tiles_list, m1_vals, yerr=m1_err_vals, fmt='o-', label='m1', capsize=5)
    ax1.errorbar(n_tiles_list, m2_vals, yerr=m2_err_vals, fmt='s-', label='m2', capsize=5)
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Number of tiles')
    ax1.set_ylabel('Multiplicative bias m')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.errorbar(n_tiles_list, c1_vals, yerr=c1_err_vals, fmt='o-', label='c1', capsize=5)
    ax2.errorbar(n_tiles_list, c2_vals, yerr=c2_err_vals, fmt='s-', label='c2', capsize=5)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Number of tiles')
    ax2.set_ylabel('Additive bias c')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot1_path = os.path.join(diagnostics_dir, 'mbias_convergence.png')
    plt.savefig(plot1_path, dpi=150)
    plt.close()
    print(f"Saved convergence plot to {plot1_path}")

    # Plot 2: error bars only
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Error convergence ({n_tiles_str})", fontsize=12)

    ax1.errorbar(n_tiles_list, [0]*len(n_tiles_list), yerr=m1_err_vals,
                 fmt='o-', label='m1 error', capsize=5, alpha=0.7)
    ax1.errorbar(n_tiles_list, [0]*len(n_tiles_list), yerr=m2_err_vals,
                 fmt='s-', label='m2 error', capsize=5, alpha=0.7)
    ax1.set_xlabel('Number of tiles')
    ax1.set_ylabel('Multiplicative bias error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2.errorbar(n_tiles_list, [0]*len(n_tiles_list), yerr=c1_err_vals,
                 fmt='o-', label='c1 error', capsize=5, alpha=0.7)
    ax2.errorbar(n_tiles_list, [0]*len(n_tiles_list), yerr=c2_err_vals,
                 fmt='s-', label='c2 error', capsize=5, alpha=0.7)
    ax2.set_xlabel('Number of tiles')
    ax2.set_ylabel('Additive bias error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plot2_path = os.path.join(diagnostics_dir, 'mbias_errors.png')
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
        n_tiles = get_n_tiles(config['grids_dir'], config['num'])
        if n_tiles:
            args.n_tiles = n_tiles
            print(f"Auto-detected {n_tiles} tiles")

    mb = ImageSimMBias(config)

    print("Loading catalogues...")
    mb.load_catalogs(verbose=args.verbose)

    results = mb.run(verbose=True)

    print()
    print("=" * 40)
    print("  Results")
    print("=" * 40)
    print(f"  m1 = {results['m1']:+.4f} +-{results['m1_err']:.4f}")
    print(f"  c1 = {results['c1']:+.4f} +-{results['c1_err']:.4f}")
    print(f"  m2 = {results['m2']:+.4f} +-{results['m2_err']:.4f}")
    print(f"  c2 = {results['c2']:+.4f} +-{results['c2_err']:.4f}")
    print("=" * 40)

    # Cumulative tracking
    if args.cumulative:
        results_dir = config.get("diagnostics_dir", config.get("results_dir", "results"))
        os.makedirs(results_dir, exist_ok=True)
    else:
        results_dir = None

    # Output path: in results dir if cumulative, else from config or current dir
    if results_dir:
        out_path = os.path.join(results_dir, "m_bias_results.yaml")
    else:
        out_path = config.get("output_path", "m_bias_results.yaml")

    with open(out_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"Results written to {out_path}")

    # Also write to text file for readability
    if results_dir:
        txt_path = os.path.join(results_dir, "m_bias_results.txt")
    else:
        txt_path = config.get("output_path", "m_bias_results.yaml").replace(".yaml", ".txt")

    with open(txt_path, "w") as f:
        f.write("Multiplicative and additive shear bias from image simulations\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"m1 = {results['m1']:+.6f} ± {results['m1_err']:.6f}\n")
        f.write(f"c1 = {results['c1']:+.6f} ± {results['c1_err']:.6f}\n\n")
        f.write(f"m2 = {results['m2']:+.6f} ± {results['m2_err']:.6f}\n")
        f.write(f"c2 = {results['c2']:+.6f} ± {results['c2_err']:.6f}\n\n")
        f.write("Errors computed via bootstrap resampling (n=500 resamples)\n")
    print(f"Results written to {txt_path}")

    if args.cumulative:
        cumulative_path = os.path.join(results_dir, "mbias_cumulative.yaml")
        if args.n_tiles:
            added = update_cumulative_file(cumulative_path, args.n_tiles, results)
            if added:
                print(f"\nAdded n_tiles={args.n_tiles} to {cumulative_path}")
                # Generate plots after update
                try:
                    plot_convergence(cumulative_path, results_dir)
                except Exception as e:
                    print(f"Warning: could not generate convergence plots: {e}", file=sys.stderr)
            else:
                print(f"\nn_tiles={args.n_tiles} already in {cumulative_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
