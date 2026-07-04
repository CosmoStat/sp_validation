#!/usr/bin/env python
"""Per-sim diagnostics for image simulation catalogues.

For each of the 5 sheared grid catalogues, produces:
  - footprint (RA/Dec scatter)
  - ellipticity histograms (e1, e2)
  - weight histogram (w_des)
  - response matrix element histograms (R_g11, R_g22, R_g12, R_g21)
  - PSF leakage scatter (e1 vs e1_PSF, e2 vs e2_PSF)
  - additive bias (weighted mean e1, e2)

Usage:
    diagnostics_image_sims.py -c config.yaml [-v]
"""

import sys
import argparse
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits


SIM_NAMES = ["1z2z", "1m2z", "1p2z", "1z2m", "1z2p"]
COLORS     = {"1z2z": "black", "1m2z": "C0", "1p2z": "C1", "1z2m": "C2", "1z2p": "C3"}


def load(path):
    with fits.open(path) as hdul:
        return {col.name: hdul[1].data[col.name].copy() for col in hdul[1].columns}


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


def plot_footprints(cats, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, d in cats.items():
        ax.scatter(d["RA"], d["Dec"], s=1, alpha=0.4, label=name, color=COLORS[name])
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.legend(markerscale=5)
    ax.set_title("Footprint")
    return savefig(fig, out_dir, "footprint")


def plot_ellipticity(cats, out_dir, nbins=100):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.linspace(-1.0, 1.0, nbins + 1)
    for name, d in cats.items():
        w = d["w_des"]
        for ax, col, label in zip(axs, ["e1", "e2"], [r"$e_1$", r"$e_2$"]):
            ax.hist(d[col], bins=bins, density=True, weights=w,
                    histtype="step", label=name, color=COLORS[name])
    for ax, label in zip(axs, [r"$e_1$", r"$e_2$"]):
        ax.set_xlabel(label)
        ax.set_ylabel("normalised count")
        ax.legend(fontsize=7)
    fig.suptitle("Ellipticity histograms (w_des weighted)")
    return savefig(fig, out_dir, "ellipticity_hist")


def plot_weights(cats, out_dir, nbins=50):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, d in cats.items():
        ax.hist(d["w_des"], bins=nbins, density=True,
                histtype="step", label=name, color=COLORS[name])
    ax.set_xlabel("w_des")
    ax.set_ylabel("normalised count")
    ax.legend()
    ax.set_title("Weight distribution")
    return savefig(fig, out_dir, "weight_hist")


def plot_response(cats, out_dir, nbins=50):
    cols = ["R_g11", "R_g22", "R_g12", "R_g21"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    for ax, col in zip(axs.flat, cols):
        for name, d in cats.items():
            ax.hist(d[col], bins=nbins, density=True,
                    histtype="step", label=name, color=COLORS[name])
        ax.set_xlabel(col)
        ax.set_ylabel("normalised count")
        ax.legend(fontsize=7)
    fig.suptitle("Response matrix elements")
    fig.tight_layout()
    return savefig(fig, out_dir, "response_hist")


def plot_psf_leakage(cats, out_dir):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for name, d in cats.items():
        for ax, eg, ep, label in zip(
            axs,
            ["e1", "e2"],
            ["e1_PSF", "e2_PSF"],
            [r"$e_1$", r"$e_2$"],
        ):
            ax.scatter(d[ep], d[eg], s=1, alpha=0.3, label=name, color=COLORS[name])
    for ax, xlab, ylab in zip(axs, [r"$e_1^{\rm PSF}$", r"$e_2^{\rm PSF}$"],
                                    [r"$e_1$", r"$e_2$"]):
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.legend(markerscale=5, fontsize=7)
    fig.suptitle("Object-wise PSF leakage")
    return savefig(fig, out_dir, "psf_leakage")


def calculate_additive_bias(cats, verbose=True):
    print("\n--- Additive bias (weighted mean ellipticity) ---")
    results = {}
    for name, d in cats.items():
        w = d["w_des"]
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

    grids_dir = config["grids_dir"]
    num       = config["num"]
    cat_name  = config.get("catalog_name", "shape_catalog_cut_ngmix.fits")
    out_dir   = config.get("diagnostics_dir", f"{grids_dir}/diagnostics")

    import os
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading catalogues from {grids_dir}...")
    cats = {}
    for name in SIM_NAMES:
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
    print(f"  footprint      -> {plot_footprints(cats, out_dir)}")
    print(f"  ellipticity    -> {plot_ellipticity(cats, out_dir)}")
    print(f"  weights        -> {plot_weights(cats, out_dir)}")
    print(f"  response       -> {plot_response(cats, out_dir)}")
    print(f"  PSF leakage    -> {plot_psf_leakage(cats, out_dir)}")
    calculate_additive_bias(cats, verbose=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
