#!/usr/bin/env python
"""Per-sim diagnostics for image simulation catalogues.

For each requested grid catalogue, produces:
  - footprint (RA/Dec scatter)
  - ellipticity histograms (e1, e2)
  - weight histogram
  - response matrix element histograms (R_g11, R_g22, R_g12, R_g21)
  - PSF leakage scatter (e1 vs e1_PSF, e2 vs e2_PSF)
  - additive bias (weighted mean e1, e2)

Shares the estimator's config schema (``sp_validation.image_sims``): the same
``grids_dir`` / ``num`` / ``catalog_name`` keys, the ``branches`` list (the sim
map, not a hard-coded five), and the same ``w_col`` weight semantics -- a column
name, or ``null`` for unit weights.  Reading ``w_col`` (rather than hard-coding
``w_des``) means the diagnostics never KeyError on a catalogue that lacks the
weight column, and they weight exactly as the m-bias run they accompany.

Usage:
    diagnostics_image_sims.py -c config.yaml [-v]
"""

import argparse
import os
import sys

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

# Conventional campaign layout, used only when the config carries no branch map
# -- the same fallback the estimator uses.
_DEFAULT_BRANCHES = ["1z2z", "1m2z", "1p2z", "1z2m", "1z2p"]


def load(path):
    with fits.open(path) as hdul:
        return {col.name: hdul[1].data[col.name].copy() for col in hdul[1].columns}


def weights(cat, w_col):
    """Per-object weights: the ``w_col`` column, or unit weights when null.

    Mirrors the estimator's ``w_col`` contract (image_sims._load_cat): ``None``
    -> every object unit weight (the no-weighting mode, #227).  Reading it here
    means the diagnostics never KeyError when the weight column is absent.
    """
    return cat[w_col].copy() if w_col else np.ones(len(cat["RA"]))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-c", "--config", required=True)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def savefig(fig, out_dir, name):
    path = f"{out_dir}/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_footprints(cats, colors, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, d in cats.items():
        ax.scatter(d["RA"], d["Dec"], s=1, alpha=0.4, label=name, color=colors[name])
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.legend(markerscale=5)
    ax.set_title("Footprint")
    return savefig(fig, out_dir, "footprint")


def plot_ellipticity(cats, colors, w_col, out_dir, nbins=100):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.linspace(-1.0, 1.0, nbins + 1)
    for name, d in cats.items():
        w = weights(d, w_col)
        for ax, col, label in zip(axs, ["e1", "e2"], [r"$e_1$", r"$e_2$"]):
            ax.hist(
                d[col],
                bins=bins,
                density=True,
                weights=w,
                histtype="step",
                label=name,
                color=colors[name],
            )
    for ax, label in zip(axs, [r"$e_1$", r"$e_2$"]):
        ax.set_xlabel(label)
        ax.set_ylabel("normalised count")
        ax.legend(fontsize=7)
    wlabel = w_col if w_col else "unit"
    fig.suptitle(f"Ellipticity histograms ({wlabel} weighted)")
    return savefig(fig, out_dir, "ellipticity_hist")


def plot_weights(cats, colors, w_col, out_dir, nbins=50):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, d in cats.items():
        ax.hist(
            weights(d, w_col),
            bins=nbins,
            density=True,
            histtype="step",
            label=name,
            color=colors[name],
        )
    wlabel = w_col if w_col else "unit"
    ax.set_xlabel(wlabel)
    ax.set_ylabel("normalised count")
    ax.legend()
    ax.set_title("Weight distribution")
    return savefig(fig, out_dir, "weight_hist")


def plot_response(cats, colors, out_dir, nbins=50):
    cols = ["R_g11", "R_g22", "R_g12", "R_g21"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    for ax, col in zip(axs.flat, cols):
        for name, d in cats.items():
            ax.hist(
                d[col],
                bins=nbins,
                range=(-1, 2),
                density=True,
                histtype="step",
                label=name,
                color=colors[name],
            )
        ax.set_xlim(-1, 2)
        ax.set_xlabel(col)
        ax.set_ylabel("normalised count")
        ax.legend(fontsize=7)
    fig.suptitle("Response matrix elements")
    fig.tight_layout()
    return savefig(fig, out_dir, "response_hist")


def plot_psf_leakage(cats, colors, out_dir):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for name, d in cats.items():
        for ax, eg, ep, label in zip(
            axs,
            ["e1", "e2"],
            ["e1_PSF", "e2_PSF"],
            [r"$e_1$", r"$e_2$"],
        ):
            ax.scatter(d[ep], d[eg], s=1, alpha=0.3, label=name, color=colors[name])
    for ax, xlab, ylab in zip(
        axs, [r"$e_1^{\rm PSF}$", r"$e_2^{\rm PSF}$"], [r"$e_1$", r"$e_2$"]
    ):
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.legend(markerscale=5, fontsize=7)
    fig.suptitle("Object-wise PSF leakage")
    return savefig(fig, out_dir, "psf_leakage")


def calculate_additive_bias(cats, w_col, verbose=True):
    print("\n--- Additive bias (weighted mean ellipticity) ---")
    results = {}
    for name, d in cats.items():
        w = weights(d, w_col)
        c1 = np.average(d["e1"], weights=w)
        c2 = np.average(d["e2"], weights=w)
        results[name] = (c1, c2)
        if verbose:
            print(f"  {name}:  c1 = {c1:+.5f}   c2 = {c2:+.5f}")
    return results


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Same config schema as the estimator: grids_dir (not base), num,
    # catalog_name, the branch map, and the w_col weight contract.
    grids_dir = config["grids_dir"]
    num = config["num"]
    cat_name = config.get("catalog_name", "shape_catalog_cut_ngmix.fits")
    branches = list(config.get("branches", _DEFAULT_BRANCHES))
    w_col = config["w_col"]  # required, like the estimator; null -> unit weights
    out_dir = config.get("diagnostics_dir", f"{grids_dir}/diagnostics")

    # Colour per branch from a palette, so any branch list plots (no hard-coded
    # five-branch colour map).
    palette = plt.get_cmap("tab10")
    colors = {name: palette(i % 10) for i, name in enumerate(branches)}

    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading catalogues from {grids_dir}...")
    cats = {}
    for name in branches:
        path = f"{grids_dir}/{name}_grid_{num}/{cat_name}"
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        cats[name] = load(path)
        if args.verbose:
            print(f"  {name}: {len(cats[name]['RA'])} objects")

    if not cats:
        print("No catalogues found, exiting.")
        return 1

    print(f"\nSaving plots to {out_dir}/")
    print(f"  footprint      -> {plot_footprints(cats, colors, out_dir)}")
    print(f"  ellipticity    -> {plot_ellipticity(cats, colors, w_col, out_dir)}")
    print(f"  weights        -> {plot_weights(cats, colors, w_col, out_dir)}")
    print(f"  response       -> {plot_response(cats, colors, out_dir)}")
    print(f"  PSF leakage    -> {plot_psf_leakage(cats, colors, out_dir)}")
    calculate_additive_bias(cats, w_col, verbose=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
