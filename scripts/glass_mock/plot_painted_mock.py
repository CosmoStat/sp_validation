"""Known-answer figures for a catalogue-painted mock, plus a sign-control sky.

Input catalogue shapes are used only for one-point noise distributions.
No real-catalogue cosmological statistic is computed.
"""

import argparse
import json
from pathlib import Path

import h5py
import healpy as hp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sp_validation.glass_mock import convergence_to_shear, paint_shear
from sp_validation.grammar import materialise
from sp_validation.io import Catalogue
from sp_validation.pseudo_cl import get_pseudo_cls_map


def sign_control(output):
    """Compare correct and reversed e2 through the pipeline's NaMaster path."""
    nside, lmax = 32, 63
    alm = np.zeros(hp.Alm.getsize(lmax), dtype=complex)
    for ell, emm in ((12, 3), (19, 7), (24, 10), (30, 2)):
        alm[hp.Alm.getidx(lmax, ell, emm)] = (1 + 0.3j) * 0.01
    kappa = hp.alm2map(alm, nside, lmax=lmax)
    gamma1, gamma2 = convergence_to_shear(kappa, lmax)
    ra, dec = hp.pix2ang(nside, np.arange(len(kappa)), lonlat=True)
    painted = paint_shear(
        ra,
        dec,
        np.zeros(len(ra)),
        np.zeros(len(ra)),
        (np.zeros_like(kappa), gamma1, gamma2),
        np.random.default_rng(1),
    )
    shear = painted["e1"] + 1j * painted["e2"]
    results = []
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for sign, label, color in (
        (-1, "pipeline pol_factor=-1", "C0"),
        (1, "wrong-sign control", "C3"),
    ):
        ell, cls, _ = get_pseudo_cls_map(
            shear, np.ones(len(ra)), nside, "linear", pol_factor=sign, ell_step=8
        )
        results.append(float(np.sum(np.abs(cls[3])) / np.sum(cls[0])))
        ax.semilogy(
            ell, np.maximum(cls[0], 1e-24), "o-", color=color, label=f"EE, {label}"
        )
        ax.semilogy(
            ell,
            np.maximum(np.abs(cls[3]), 1e-24),
            "s--",
            color=color,
            label=f"|BB|, {label}",
        )
    ax.set(
        xlabel=r"$\ell$",
        ylabel=r"$C_\ell$",
        title="Pure-E mock: catalogue e2 = -HEALPix gamma2",
    )
    ax.legend(fontsize=8)
    fig.savefig(output / "sign_convention.png", dpi=160)
    plt.close(fig)
    return dict(correct_bb_over_ee=results[0], wrong_bb_over_ee=results[1])


def map_spectra(maps, output):
    spectra = np.load(maps / "spectra.npz")
    ell = spectra["ell"]
    edges = np.unique(np.r_[2, np.geomspace(8, min(8001, ell[-1] + 1), 40).astype(int)])

    def bin_spectrum(values):
        return np.array(
            [
                np.average(
                    values[(ell >= lo) & (ell < hi)],
                    weights=2 * ell[(ell >= lo) & (ell < hi)] + 1,
                )
                for lo, hi in zip(edges[:-1], edges[1:], strict=True)
            ]
        )

    centers = bin_spectrum(ell)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
        constrained_layout=True,
    )
    metrics = {}
    for field, color in (("kk", "C0"), ("ee", "C1")):
        theory = spectra[f"cl_{field}_pixel"]
        measured = spectra[f"realization_{field}"]
        unpixel = spectra[f"cl_{field}"]
        axes[0].loglog(
            ell[2:8001],
            theory[2:8001],
            color=color,
            label=f"{field.upper()} shell ensemble, pixels",
        )
        axes[0].loglog(
            ell[2:8001],
            unpixel[2:8001],
            color=color,
            ls=":",
            alpha=0.7,
            label=f"{field.upper()} ensemble, no pixels",
        )
        axes[0].loglog(
            centers,
            bin_spectrum(measured),
            "o",
            ms=3,
            color=color,
            label=f"{field.upper()} realization",
        )
        ratio = bin_spectrum(measured) / bin_spectrum(theory)
        axes[1].semilogx(centers, ratio, "o-", ms=3, color=color)
        lower, upper = min(100, max(2, ell[-1] // 4)), min(8000, ell[-1])
        choose = (ell >= lower) & (ell <= upper)
        achieved = spectra[f"cl_{field}_achieved_pixel"]
        axes[1].semilogx(
            centers,
            bin_spectrum(achieved) / bin_spectrum(theory),
            ls=":",
            color=color,
            alpha=0.8,
        )
        metrics[f"{field}_solver_weighted_fractional_error"] = float(
            np.average(
                (achieved[choose] - theory[choose]) / theory[choose],
                weights=2 * ell[choose] + 1,
            )
        )
        metrics[f"{field}_mean_ratio_ell{lower}_{upper}"] = float(
            np.average(measured[choose] / theory[choose], weights=2 * ell[choose] + 1)
        )
    axes[0].loglog(
        centers,
        bin_spectrum(spectra["realization_bb"]),
        "x",
        color="C3",
        label="BB realization",
    )
    axes[0].set(
        ylabel=r"$C_\ell$",
        title="Projected GLASS sky: full-sky realization vs shell ensemble",
    )
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].axhline(1, color="0.4", lw=1)
    axes[1].set(xlabel=r"$\ell$", ylabel="realization / ensemble", ylim=(0.5, 1.5))
    fig.savefig(output / "known_answer_spectra.png", dpi=160)
    plt.close(fig)
    return metrics


def reduced_shear_comparison(maps, output):
    """Measure gamma/(1-kappa) EE/BB relative to the same seed's pure gamma."""
    saved = np.load(maps / "spectra.npz")
    ell = saved["ell"]
    path = output / "reduced_shear_comparison.npz"
    if path.exists():
        with np.load(path) as cached:
            reduced = cached["reduced"]
    else:
        kappa = np.load(maps / "kappa.npy", mmap_mode="r")
        denominator = 1 - np.asarray(kappa, dtype=float)
        g1 = np.load(maps / "gamma1.npy", mmap_mode="r") / denominator
        g2 = np.load(maps / "gamma2.npy", mmap_mode="r") / denominator
        del denominator
        reduced = hp.anafast(
            [np.zeros(len(kappa)), g1, g2], lmax=int(ell[-1]), pol=True, iter=3
        )
        np.savez(
            path,
            ell=ell,
            reduced=reduced,
            gamma_ee=saved["realization_ee"],
            gamma_bb=saved["realization_bb"],
        )
    edges = np.arange(2, min(8000, int(ell[-1])) + 101, 100)
    centers, ee_ratio, bb_ratio, gamma_bb_ratio = [], [], [], []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (ell >= lower) & (ell < upper)
        weight = 2 * ell[selected] + 1
        gamma_ee = np.average(saved["realization_ee"][selected], weights=weight)
        reduced_ee = np.average(reduced[1, selected], weights=weight)
        centers.append(np.average(ell[selected], weights=weight))
        ee_ratio.append(reduced_ee / gamma_ee)
        bb_ratio.append(np.average(reduced[2, selected], weights=weight) / reduced_ee)
        gamma_bb_ratio.append(
            np.average(saved["realization_bb"][selected], weights=weight) / gamma_ee
        )
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
    axes[0].plot(centers, ee_ratio, color="C0")
    axes[0].axhline(1, color="0.5", ls="--")
    axes[0].set(
        ylabel="EE(reduced) / EE(gamma)",
        title="Seed 1: physical reduced-shear corrections, full sky",
    )
    axes[1].semilogy(centers, bb_ratio, label="reduced shear")
    axes[1].semilogy(centers, gamma_bb_ratio, label="gamma (transform floor)")
    axes[1].set(xlabel=r"$\ell$", ylabel="BB / EE")
    axes[1].legend()
    fig.savefig(output / "reduced_shear_effect.png", dpi=160)
    plt.close(fig)
    metrics = {}
    for target in (500, 1000, 2000, 4000, 7900):
        index = int(np.argmin(np.abs(np.asarray(centers) - target)))
        metrics[str(target)] = dict(
            ell=float(centers[index]),
            ee_ratio=float(ee_ratio[index]),
            bb_over_ee=float(bb_ratio[index]),
        )
    return metrics


def catalogue_noise(source_path, mock_path, maps, output):
    names = ["RA", "Dec", "e1", "e2", "w_des"]
    source_hist = np.zeros(100)
    noise_hist = np.zeros(100)
    mock_hist = np.zeros(100)
    edges = np.linspace(0, 2, 101)
    # Each row contains sum(w), sum(w*x), sum(w*x^2), for w and w^2.
    moments = {kind: np.zeros((2, 2, 3)) for kind in ("input", "noise", "mock")}
    samples = []
    offset = 0
    max_truth_error = 0.0
    max_amplitude_error = 0.0
    lensing = [
        np.load(maps / f"{name}.npy", mmap_mode="r")
        for name in ("kappa", "gamma1", "gamma2")
    ]
    nside = hp.npix2nside(len(lensing[0]))
    with Catalogue(source_path) as cat, h5py.File(mock_path) as painted:
        shear_choice = json.loads(painted.attrs["mock_metadata"])["shear"]
        for part in cat.chunks.values():
            for lo in range(0, len(part), 250_000):
                real = materialise(part[lo : lo + 250_000], names)
                mock = painted["data"][offset : offset + len(real)]
                for key in ("RA", "Dec", "w_des"):
                    if not np.array_equal(real[key], mock[key]):
                        raise ValueError(f"Painting changed {key}")
                weight = real["w_des"]
                for kind, pair in (
                    ("input", (real["e1"], real["e2"])),
                    ("noise", (mock["e1_noise"], mock["e2_noise"])),
                    ("mock", (mock["e1"], mock["e2"])),
                ):
                    for power in (1, 2):
                        w = weight**power
                        for comp, val in enumerate(pair):
                            moments[kind][power - 1, comp] += [
                                w.sum(),
                                np.sum(w * val),
                                np.sum(w * val**2),
                            ]
                true_amp = np.hypot(real["e1"], real["e2"])
                noise_amp = np.hypot(mock["e1_noise"], mock["e2_noise"])
                source_hist += np.histogram(true_amp, edges, weights=weight)[0]
                noise_hist += np.histogram(noise_amp, edges, weights=weight)[0]
                mock_hist += np.histogram(
                    np.hypot(mock["e1"], mock["e2"]), edges, weights=weight
                )[0]
                max_amplitude_error = max(
                    max_amplitude_error, np.max(np.abs(true_amp - noise_amp))
                )
                # Uniform deterministic sample across the full catalogue.
                select = np.flatnonzero((offset + np.arange(len(real))) % 100 == 0)
                selected = mock[select]
                pix = hp.ang2pix(nside, selected["RA"], selected["Dec"], lonlat=True)
                kap = np.asarray(lensing[0][pix], dtype=float)
                denominator = 1 - kap if shear_choice == "reduced" else 1
                g1 = np.asarray(lensing[1][pix], dtype=float) / denominator
                g2 = -np.asarray(lensing[2][pix], dtype=float) / denominator
                max_truth_error = max(
                    max_truth_error,
                    np.max(np.abs(g1 - selected["g1_true"])),
                    np.max(np.abs(g2 - selected["g2_true"])),
                )
                samples.append(
                    np.column_stack(
                        (
                            g1,
                            g2,
                            selected["g1_true"],
                            selected["g2_true"],
                            selected["e1"],
                            selected["e2"],
                            selected["e1_noise"],
                            selected["e2_noise"],
                            selected["w_des"],
                        )
                    )
                )
                offset += len(real)
    sample = np.concatenate(samples)
    sigmas = {}
    for kind, accum in moments.items():
        sigmas[kind] = {}
        for power in (1, 2):
            total, first, second = accum[power - 1].T
            sigmas[kind][f"weight_power_{power}"] = np.sqrt(
                second / total - (first / total) ** 2
            ).tolist()
    fig, ax = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    centers = (edges[:-1] + edges[1:]) / 2
    for label, hist in (
        ("input calibrated |e|", source_hist),
        ("rotated |e_noise|", noise_hist),
        ("lensed mock |e|", mock_hist),
    ):
        ax[0, 0].plot(centers, hist / np.sum(hist) / np.diff(edges), label=label)
    ax[0, 0].set(
        xlabel="ellipticity amplitude", ylabel="weighted density", xlim=(0, 1.5)
    )
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].hist(
        sample[:, 6],
        bins=100,
        range=(-1, 1),
        density=True,
        histtype="step",
        label="noise e1",
    )
    ax[0, 1].hist(
        sample[:, 7],
        bins=100,
        range=(-1, 1),
        density=True,
        histtype="step",
        label="noise e2",
    )
    ax[0, 1].set(xlabel="rotated component", ylabel="density")
    ax[0, 1].legend()
    for comp, color in ((0, "C0"), (1, "C1")):
        ax[0, 2].scatter(
            sample[::100, comp],
            sample[::100, comp + 2],
            s=3,
            color=color,
            label=f"g{comp + 1} vs map",
        )
    ax[0, 2].set(xlabel=f"{shear_choice} at positions", ylabel="saved g_true")
    ax[0, 2].legend()
    ax[1, 0].hexbin(
        sample[:, 0],
        sample[:, 4],
        gridsize=75,
        bins="log",
        mincnt=1,
        extent=(-0.1, 0.1, -1, 1),
        cmap="viridis",
    )
    ax[1, 0].set(xlabel="g1_true", ylabel="painted e1")
    bins = np.linspace(-0.05, 0.05, 16)
    for comp, color in ((0, "C0"), (1, "C1")):
        x, y, error = [], [], []
        for lower, upper in zip(bins[:-1], bins[1:], strict=True):
            selected = (sample[:, comp] >= lower) & (sample[:, comp] < upper)
            w, e = sample[selected, 8], sample[selected, comp + 4]
            if len(w) < 100:
                continue
            x.append(np.average(sample[selected, comp], weights=w))
            y.append(np.average(e, weights=w))
            error.append(np.sqrt(np.sum(w**2 * (e - y[-1]) ** 2)) / np.sum(w))
        ax[1, 1].errorbar(
            x, y, yerr=error, fmt="o", ms=3, color=color, label=f"component {comp + 1}"
        )
    ax[1, 1].plot([-0.05, 0.05], [-0.05, 0.05], color="0.4", ls="--")
    ax[1, 1].set(xlabel="g_true (conditional mean)", ylabel="weighted mean mock e")
    ax[1, 1].legend(fontsize=8)
    ax[1, 2].axis("off")
    text = [
        f"{offset:,} galaxies; positions / weights unchanged",
        "",
        "sigma_e per component, w^2 weighted:",
    ]
    for kind in ("input", "noise", "mock"):
        s1, s2 = sigmas[kind]["weight_power_2"]
        text.append(f"{kind:>5}: ({s1:.6f}, {s2:.6f})")
    text += [
        "",
        f"max |noise amplitude - input amplitude|: {max_amplitude_error:.2g}",
        f"max |saved g - sampled map g|: {max_truth_error:.2g}",
        f"conditional means: 1/{100} of catalogue",
    ]
    ax[1, 2].text(0, 1, "\n".join(text), va="top", fontsize=9, family="monospace")
    fig.suptitle("Catalogue-painted GLASS mock: geometry, signal and empirical noise")
    fig.savefig(output / "catalogue_noise.png", dpi=160)
    plt.close(fig)
    return dict(
        rows=offset,
        sigma_e=sigmas,
        max_amplitude_error=float(max_amplitude_error),
        sampled_max_truth_error=float(max_truth_error),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path)
    parser.add_argument("--catalogue")
    parser.add_argument("--mock")
    parser.add_argument("--reduced-comparison", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = {"sign": sign_control(args.output)}
    if args.maps:
        metrics["spectra"] = map_spectra(args.maps, args.output)
        if args.reduced_comparison:
            metrics["reduced_shear"] = reduced_shear_comparison(args.maps, args.output)
    if args.catalogue and args.mock:
        metrics["catalogue"] = catalogue_noise(
            args.catalogue, args.mock, args.maps, args.output
        )
    (args.output / "diagnostics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
