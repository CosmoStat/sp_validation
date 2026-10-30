"""Known-answer diagnostics for the four isolated painted seed-1 mocks.

Run inside the image on the allocation. Stages are restartable and do not
read data shears or any blind outside the explicitly checked mock registry.
"""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import scipy.linalg as la
from scipy.stats import chi2

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from astropy.io import fits

from sp_validation import blinding
from sp_validation import sacc_io as sio

ROOT = Path("/n17data/cdaley/scratch/glass-mock-cv/cvpaint")
OUT, FIG = ROOT / "output", ROOT / "figures"
MOCK = ROOT.parent / "mocks/v1.4.6.3/seed_00001"
U, B, P, A = "GLASS1_clean", "GLASS1_blinded", "GLASS1_psfleak", "GLASS1_addB"
REG = yaml.safe_load((ROOT / "cat_config.yaml").read_text())
assert (
    Path(REG["paths"]["blinds"]).resolve() == (ROOT.parent / "cvmock/blinds").resolve()
)
assert all(REG[v]["shear"]["path"].startswith(str(ROOT.parent)) for v in (U, B, P, A))
METRIC_PATH = ROOT / "metrics.json"
metrics = json.loads(METRIC_PATH.read_text()) if METRIC_PATH.exists() else {}
metrics.update(rejection_pte=0.01, cuts={"ell": [300, 1600], "theta_arcmin": [12, 83]})


def save():
    METRIC_PATH.write_text(json.dumps(metrics, indent=2, allow_nan=False))


def pte(y, cov):
    y, cov = np.asarray(y), np.asarray(cov)
    eig, vec = la.eigh(cov)
    keep = eig > max(eig.max(), 0) * 1e-12
    if not np.any(keep):
        raise ValueError("Covariance has no positive numerical rank")
    q = float(np.sum((vec[:, keep].T @ y) ** 2 / eig[keep]))
    n = int(keep.sum())
    return dict(
        chi2=q,
        dof=n,
        pte=float(chi2.sf(q, n)),
        size=len(y),
        min_eigenvalue=float(eig.min()),
    )


def block(s, typ):
    i = np.asarray(s.indices(typ), int)
    cov = None if s.covariance is None else s.covariance.dense[np.ix_(i, i)]
    return i, s.mean[i], cov


def coord(s, indices, tag):
    return np.array([s.data[i].tags[tag] for i in indices])


def equal(a, b, sigma=None):
    a, b = np.asarray(a), np.asarray(b)
    delta = np.abs(a - b)
    result = dict(
        bit_identical=bool(
            a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
        ),
        exactly_equal_values=bool(np.array_equal(a, b)),
        max_abs_difference=float(delta.max(initial=0)),
    )
    if sigma is not None:
        result["max_difference_over_sigma"] = float(
            np.max(delta / np.maximum(sigma, 1e-100), initial=0)
        )
    return result


def harmonic(version):
    s = sio.load(
        OUT / f"pseudo_cl_{version}_powspace_nbins=32.sacc", blind=REG[version]["blind"]
    )
    with fits.open(OUT / f"pseudo_cl_cov_{version}_powspace_nbins=32.fits") as hdus:
        cov = la.block_diag(*[hdus[f"COVAR_{t}_{t}"].data for t in ("EE", "BB", "EB")])
    s.add_covariance(cov, overwrite=True)
    return s


def xi(version, grid):
    stem = f"{version}_xi_minsep=" + (
        "1.0_maxsep=250.0_nbins=20"
        if grid == "reporting"
        else "0.08_maxsep=300.0_nbins=1000"
    )
    npatch = (
        100
        if grid == "reporting" and (OUT / (stem + "_npatch=100.sacc")).exists()
        else 1
    )
    return sio.load(
        OUT / (stem + f"_npatch={npatch}.sacc"), blind=REG[version]["blind"]
    )


def convolve(s, indices, ell, cl):
    window = s.get_bandpower_windows(indices)
    return np.asarray(window.weight).T @ np.interp(
        window.values, ell, cl, left=0, right=0
    )


def mock_parameter_shift(key="S8"):
    from smokescreen.datavector import draw_param_shifts

    record = blinding.open_blind("glassmock-test", REG).record()
    draw = draw_param_shifts(record["envelope"], record["seed"])
    return float(draw.get(key, 0.0))


def expected_shift(s):
    from smokescreen.datavector import draw_param_shifts

    record = blinding.open_blind("glassmock-test", REG).record()
    fiducial = record["fiducial"]
    draw = draw_param_shifts(record["envelope"], record["seed"])
    hidden = {k: v + draw.get(k, 0.0) for k, v in fiducial.items()}
    return blinding.shear(hidden, s) - blinding.shear(fiducial, s)


def shift_metrics(su, sb):
    actual = sb.mean - su.mean
    expected = expected_shift(su)
    result = dict(
        max_abs_error=float(np.max(np.abs(actual - expected))),
        matches_float_precision=bool(
            np.allclose(actual, expected, rtol=1e-11, atol=2e-20)
        ),
        mock_delta_S8=mock_parameter_shift(),
        mock_delta_Omega_m=mock_parameter_shift("Omega_m"),
        maximum_per_bin_shift_sigma=float(
            np.max(np.abs(expected) / np.sqrt(np.diag(su.covariance.dense)))
        ),
    )
    result["full_covariance"] = equal(su.covariance.dense, sb.covariance.dense)
    result["invariants"] = {}
    for typ in su.get_data_types():
        i, y, cov = block(su, typ)
        result["invariants"][typ] = dict(
            vector=equal(y, sb.mean[i], np.sqrt(np.diag(cov))),
            covariance=equal(cov, sb.covariance.dense[np.ix_(i, i)]),
        )
    active = np.concatenate(
        [
            su.indices(t)
            for t in (sio.XI_PLUS, sio.XI_MINUS, sio.CL_EE)
            if len(su.indices(t))
        ]
    ).astype(int)
    distance = pte(expected[active], su.covariance.dense[np.ix_(active, active)])
    result["joint_shift"] = dict(
        chi2=distance["chi2"],
        vector_mahalanobis_distance=float(np.sqrt(distance["chi2"])),
        size=len(active),
    )
    return result, expected


def blind_panel(ax, x, raw, sealed, expected, cov, label):
    sigma = np.sqrt(np.diag(cov))
    q = pte(expected, cov)["chi2"]
    ax.fill_between(
        x,
        -np.ones(len(x)),
        np.ones(len(x)),
        color="0.8",
        label="Expected marginal 1σ scatter",
    )
    ax.plot(x, expected / sigma, "-", label="Designed blind shift")
    ax.plot(x, (sealed - raw) / sigma, "o", ms=3, label="Measured blinded − clean")
    ax.axhline(0, color="k", lw=0.7)
    ax.set(
        xscale="log",
        ylabel="Shift / pipeline marginal σ",
        title=f"{label}: mock ΔS8 = {mock_parameter_shift():+.4f}\nΔχ² = {q:.2f}; max |shift| = {np.max(np.abs(expected) / sigma):.2f}σ",
    )
    ax.legend(fontsize=8)


def run_harmonic():
    su, sb = harmonic(U), harmonic(B)
    spectra = np.load(MOCK / "maps/spectra.npz")
    i, ee, cov = block(su, sio.CL_EE)
    x = coord(su, i, "ell")
    real = convolve(su, i, spectra["ell"], spectra["realization_ee"])
    target = convolve(su, i, spectra["ell"], spectra["cl_ee_pixel"])
    cut = (x >= 300) & (x <= 1600)
    metrics["harmonic_EE"] = {
        "realization_full": pte(ee - real, cov),
        "realization_fiducial": pte((ee - real)[cut], cov[np.ix_(cut, cut)]),
        "ensemble_full": pte(ee - target, cov),
        "ensemble_fiducial": pte((ee - target)[cut], cov[np.ix_(cut, cut)]),
        "fractional_residual": ((ee - real) / real).tolist(),
        "ell": x.tolist(),
        "multiplicative_fit_full": fit_amplitude(ee, real, cov),
        "multiplicative_fit_fiducial": fit_amplitude(
            ee[cut], real[cut], cov[np.ix_(cut, cut)]
        ),
        "multiplicative_fit_high_ell": fit_amplitude(
            ee[x > 1600], real[x > 1600], cov[np.ix_(x > 1600, x > 1600)]
        ),
    }
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    scale = x * (x + 1) / (2 * np.pi)
    axes[0].errorbar(
        x,
        scale * ee,
        yerr=scale * np.sqrt(np.diag(cov)),
        fmt="o",
        ms=3,
        label="Clean painted mock",
    )
    axes[0].plot(x, scale * real, label="Own full-sky realization → bandpower windows")
    axes[0].plot(
        x, scale * target, "--", label="GLASS shell ensemble → bandpower windows"
    )
    axes[0].set(
        ylabel="ℓ(ℓ+1) Cℓ EE / 2π",
        title="EE known-answer recovery (pixelized sky; no measurement-map pixel window)",
    )
    axes[0].legend(fontsize=8)
    axes[1].errorbar(
        x, (ee - real) / np.sqrt(np.diag(cov)), yerr=np.ones(len(x)), fmt="o", ms=3
    )
    axes[1].axhline(0, color="k", lw=0.7)
    axes[1].set(xscale="log", xlabel="Multipole ℓ", ylabel="Residual / marginal σ")
    for ax in axes:
        ax.axvspan(300, 1600, color="0.5", alpha=0.1)
    fig.tight_layout()
    fig.savefig(FIG / "ee_harmonic.png", dpi=170)
    plt.close(fig)
    metrics["harmonic_nulls"] = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, typ in zip(axes, (sio.CL_BB, sio.CL_EB)):
        ii, y, cc = block(su, typ)
        xx = coord(su, ii, "ell")
        keep = (xx >= 300) & (xx <= 1600)
        result = dict(full=pte(y, cc), fiducial=pte(y[keep], cc[np.ix_(keep, keep)]))
        metrics["harmonic_nulls"][typ] = result
        ax.errorbar(xx, y / np.sqrt(np.diag(cc)), yerr=np.ones(len(y)), fmt="o", ms=3)
        ax.axhline(0, color="k", lw=0.7)
        ax.set(
            xlabel="Multipole ℓ",
            ylabel="Signal / marginal σ",
            title=f"{typ}: full PTE={result['full']['pte']:.3g}",
        )
    fig.tight_layout()
    fig.savefig(FIG / "bmode_harmonic.png", dpi=170)
    plt.close(fig)
    result, shift = shift_metrics(su, sb)
    metrics["blind_harmonic"] = result
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    sigma = np.sqrt(np.diag(cov))
    axes[0].fill_between(
        x,
        (real - sigma) * scale,
        (real + sigma) * scale,
        color="0.8",
        label="Expected mock marginal 1σ scatter",
    )
    axes[0].plot(x, ee * scale, "o", ms=3, label="Clean mock")
    axes[0].plot(x, sb.mean[i] * scale, "s", ms=3, label="Blinded mock")
    axes[0].set(
        ylabel="ℓ(ℓ+1) Cℓ EE / 2π",
        title="Same painted mock, independently measured before/after concealment",
    )
    axes[0].legend(fontsize=8)
    blind_panel(axes[1], x, ee, sb.mean[i], shift[i], cov, "Cℓ EE")
    axes[1].set_xlabel("Multipole ℓ")
    fig.tight_layout()
    fig.savefig(FIG / "blinding_harmonic.png", dpi=170)
    plt.close(fig)
    np.savez(
        ROOT / "harmonic_reference.npz",
        ell=x,
        realized=real,
        target=target,
        covariance=cov,
    )
    save()


def fit_amplitude(y, template, cov):
    inv_t = la.solve(cov, template, assume_a="pos")
    error = float(1 / np.sqrt(template @ inv_t))
    amplitude = float((y @ inv_t) * error**2)
    residual = pte(y - amplitude * template, cov)
    residual["dof"] -= 1
    residual["pte"] = float(chi2.sf(residual["chi2"], residual["dof"]))
    return dict(
        amplitude=amplitude, error=error, snr=amplitude / error, residual=residual
    )


def run_additive():
    su, sa = harmonic(U), harmonic(A)
    i, clean, cov = block(su, sio.CL_BB)
    x = coord(su, i, "ell")
    _, injected, injected_cov = block(sa, sio.CL_BB)
    spec = np.load(MOCK / "painted_addB.h5.bmode_spectra.npz")
    realized = convolve(su, i, spec["ell"], spec["realization_bb"])
    target = convolve(su, i, spec["ell"], spec["target_bb"])
    a0 = 2.6e-11
    keep = (x >= 300) & (x <= 1600)
    metrics["additive_harmonic"] = dict(
        input_amplitude=a0,
        full_fit=fit_amplitude(injected - clean, realized / a0, cov),
        fiducial_fit=fit_amplitude(
            (injected - clean)[keep], realized[keep] / a0, cov[np.ix_(keep, keep)]
        ),
        injected_vector_null_full=pte(injected, injected_cov),
        injected_vector_null_fiducial=pte(
            injected[keep], injected_cov[np.ix_(keep, keep)]
        ),
        clean_vector_null=pte(clean, cov),
        difference_residual_under_single_pipeline_cov=pte(
            injected - clean - realized, cov
        ),
        uncertainty="Single clean-mock pipeline BB covariance; not a covariance estimate for the same-noise difference. The difference contains injection×noise and injection×baseline-B cross terms.",
    )
    for label in ("full_fit", "fiducial_fit"):
        metrics["additive_harmonic"][label]["expected_input_snr"] = (
            a0 / metrics["additive_harmonic"][label]["error"]
        )
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    scale = x * (x + 1) / (2 * np.pi)
    sig = np.sqrt(np.diag(cov))
    axes[0].errorbar(
        x,
        (injected - clean) * scale,
        yerr=sig * scale,
        fmt="o",
        ms=3,
        label="Injected − clean (identical shape noise)",
    )
    axes[0].plot(
        x, realized * scale, label="Realized full-sky injection → bandpower windows"
    )
    axes[0].plot(
        x, target * scale, "--", label="Ensemble injection target → bandpower windows"
    )
    fit = metrics["additive_harmonic"]["fiducial_fit"]
    axes[0].set(
        ylabel="ℓ(ℓ+1) ΔCℓ BB / 2π",
        title=f"Injected A={a0:.2g}; fitted A=({fit['amplitude'] / 1e-11:.2f}±{fit['error'] / 1e-11:.2f})×10⁻¹¹; S/N={fit['snr']:.2f} (300≤ℓ≤1600)",
    )
    axes[0].legend(fontsize=8)
    axes[1].errorbar(
        x, (injected - clean - realized) / sig, yerr=np.ones(len(x)), fmt="o", ms=3
    )
    axes[1].axhline(0, color="k", lw=0.7)
    axes[1].set(
        xscale="log", xlabel="Multipole ℓ", ylabel="Difference residual / clean σ"
    )
    fig.tight_layout()
    fig.savefig(FIG / "additive_harmonic.png", dpi=170)
    plt.close(fig)
    save()


def xi_covariance(version):
    paths = list((ROOT / "inference/data").rglob(f"*{version}*"))
    candidates = [
        p
        for p in paths
        if p.is_file()
        and "cov" in p.name.lower()
        and "nbins=20" in p.name
        and "ng" in p.name
        and p.suffix == ".dat"
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one reporting NG covariance, found {candidates}")
    return sio.read_onecov_matrix(candidates[0])


def run_blind_xi():
    su, sb = xi(U, "reporting"), xi(B, "reporting")
    su.add_covariance(xi_covariance(U), overwrite=True)
    sb.add_covariance(xi_covariance(B), overwrite=True)
    result, expected = shift_metrics(su, sb)
    metrics["blind_xi"] = result
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    for j, typ in enumerate((sio.XI_PLUS, sio.XI_MINUS)):
        i, y, cov = block(su, typ)
        theta = coord(su, i, "theta")
        sigma = np.sqrt(np.diag(cov))
        axes[0, j].fill_between(
            theta,
            theta * (y - sigma),
            theta * (y + sigma),
            color="0.8",
            label="Expected 1σ scatter about clean mock",
        )
        axes[0, j].plot(theta, theta * y, "o", ms=3, label="Clean mock")
        axes[0, j].plot(theta, theta * sb.mean[i], "s", ms=3, label="Blinded mock")
        axes[0, j].set(ylabel="θ ξ" + ("+" if j == 0 else "−"))
        axes[0, j].legend(fontsize=8)
        blind_panel(
            axes[1, j],
            theta,
            y,
            sb.mean[i],
            expected[i],
            cov,
            "ξ" + ("+" if j == 0 else "−"),
        )
        axes[1, j].set_xlabel("Separation θ [arcmin]")
    fig.suptitle(
        f"Mock ΔS8 = {result['mock_delta_S8']:+.4f}: joint ξ± shift Δχ² = {result['joint_shift']['chi2']:.2f}; max |shift| = {result['maximum_per_bin_shift_sigma']:.2f}σ"
    )
    fig.tight_layout()
    fig.savefig(FIG / "blinding_xi.png", dpi=170)
    plt.close(fig)
    save()


def run_xi():
    import pyccl as ccl
    from cs_util.cosmo import get_cosmo

    su, sb = xi(U, "reporting"), xi(B, "reporting")
    cov = xi_covariance(U)
    covb = xi_covariance(B)
    su.add_covariance(cov, overwrite=True)
    sb.add_covariance(covb, overwrite=True)
    fine = sio.xi_correlation(xi(U, "integration"), grid="integration")
    fine_edges = np.geomspace(0.08, 300, 1001)
    report_edges = np.geomspace(1, 250, 21)
    weights = []
    for lo, hi in zip(report_edges[:-1], report_edges[1:]):
        lower, upper = np.maximum(fine_edges[:-1], lo), np.minimum(fine_edges[1:], hi)
        frac = np.maximum(upper**2 - lower**2, 0) / (
            fine_edges[1:] ** 2 - fine_edges[:-1] ** 2
        )
        ww = frac * fine.weight
        weights.append(ww / ww.sum())
    weights = np.array(weights)
    spec = np.load(MOCK / "maps/spectra.npz")
    ell = spec["ell"][2:]
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    cosmo = get_cosmo(**config["cosmo_val"]["cosmo_params"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex="col")
    predictions, ensemble_predictions, theta_list = [], [], []
    for j, typ in enumerate((sio.XI_PLUS, sio.XI_MINUS)):
        i, y, cc = block(su, typ)
        theta = coord(su, i, "theta")
        theta_list.append(theta)
        kind = "GG+" if j == 0 else "GG-"
        real = weights @ ccl.correlation(
            cosmo,
            ell=ell,
            C_ell=spec["realization_ee"][2:],
            theta=fine.meanr / 60,
            type=kind,
        )
        target = weights @ ccl.correlation(
            cosmo,
            ell=ell,
            C_ell=spec["cl_ee_pixel"][2:],
            theta=fine.meanr / 60,
            type=kind,
        )
        predictions.append(real)
        ensemble_predictions.append(target)
        sig = np.sqrt(np.diag(cc))
        axes[0, j].errorbar(
            theta, theta * y, yerr=theta * sig, fmt="o", ms=3, label="Clean mock"
        )
        axes[0, j].plot(theta, theta * real, label="Own pixelized full-sky EE → ξ")
        axes[0, j].plot(theta, theta * target, "--", label="GLASS shell ensemble → ξ")
        axes[0, j].set(
            ylabel="θ ξ" + ("+" if j == 0 else "−"),
            title="Pair-weighted reporting-bin theory",
        )
        axes[0, j].legend(fontsize=8)
        axes[1, j].errorbar(
            theta, (y - real) / sig, yerr=np.ones(len(y)), fmt="o", ms=3
        )
        axes[1, j].axhline(0, color="k", lw=0.7)
        axes[1, j].set(
            xscale="log", xlabel="Separation θ [arcmin]", ylabel="Residual / marginal σ"
        )
        for ax in axes[:, j]:
            ax.axvspan(12, 83, color="0.5", alpha=0.1)
    prediction = np.concatenate(predictions)
    keep = np.concatenate([(t >= 12) & (t <= 83) for t in theta_list])
    ensemble_prediction = np.concatenate(ensemble_predictions)
    metrics["xi_EE"] = dict(
        full=pte(su.mean - prediction, cov),
        fiducial=pte((su.mean - prediction)[keep], cov[np.ix_(keep, keep)]),
        ensemble_full=pte(su.mean - ensemble_prediction, cov),
        ensemble_fiducial=pte(
            (su.mean - ensemble_prediction)[keep], cov[np.ix_(keep, keep)]
        ),
        theory_binning="Actual fine-grid weighted pairs and fractional annular overlap; CCL GG± transform, ell 2–8192; no extra pixel window.",
    )
    fig.tight_layout()
    fig.savefig(FIG / "ee_xi.png", dpi=170)
    plt.close(fig)
    result, expected = shift_metrics(su, sb)
    metrics["blind_xi"] = result
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex="col")
    for j, typ in enumerate((sio.XI_PLUS, sio.XI_MINUS)):
        i, y, cc = block(su, typ)
        theta = coord(su, i, "theta")
        sig = np.sqrt(np.diag(cc))
        real = predictions[j]
        axes[0, j].fill_between(
            theta,
            theta * (real - sig),
            theta * (real + sig),
            color="0.8",
            label="Expected marginal 1σ scatter",
        )
        axes[0, j].plot(theta, theta * y, "o", ms=3, label="Clean mock")
        axes[0, j].plot(theta, theta * sb.mean[i], "s", ms=3, label="Blinded mock")
        axes[0, j].set(ylabel="θ ξ" + ("+" if j == 0 else "−"))
        axes[0, j].legend(fontsize=8)
        blind_panel(
            axes[1, j],
            theta,
            y,
            sb.mean[i],
            expected[i],
            cc,
            "ξ" + ("+" if j == 0 else "−"),
        )
        axes[1, j].set_xlabel("Separation θ [arcmin]")
    fig.suptitle(
        f"Mock ΔS8 = {result['mock_delta_S8']:+.4f}: joint ξ± shift Δχ² = {result['joint_shift']['chi2']:.2f}; max |shift| = {result['maximum_per_bin_shift_sigma']:.2f}σ"
    )
    fig.tight_layout()
    fig.savefig(FIG / "blinding_xi.png", dpi=170)
    plt.close(fig)
    save()


def run_decompositions():
    metrics["decomposition_nulls"] = {}
    types = [sio.COSEBI_BB, sio.PURE_TYPES["xip_B"], sio.PURE_TYPES["xim_B"]]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, version in enumerate((U, A)):
        for col, typ in enumerate(types):
            suffix = "cosebis" if typ == sio.COSEBI_BB else "pure_eb"
            s = sio.load(OUT / f"{version}_{suffix}.sacc", blind="none")
            i, y, cov = block(s, typ)
            x = (
                np.arange(1, len(y) + 1)
                if suffix == "cosebis"
                else coord(s, i, "theta")
            )
            stat = pte(y, cov)
            checks = {"full": stat}
            if suffix == "cosebis":
                checks["first_6_modes"] = pte(y[:6], cov[:6, :6])
                label = f"B_n: PTE(6)={checks['first_6_modes']['pte']:.3g}, PTE(20)={stat['pte']:.3g}"
            else:
                keep = (x >= 12) & (x <= 83)
                checks["fiducial"] = pte(y[keep], cov[np.ix_(keep, keep)])
                label = f"{typ}: PTE(12–83′)={checks['fiducial']['pte']:.3g}"
            metrics["decomposition_nulls"].setdefault(version, {})[typ] = checks
            ax = axes[row, col]
            ax.errorbar(
                x, y / np.sqrt(np.diag(cov)), yerr=np.ones(len(x)), fmt="o", ms=3
            )
            ax.axhline(0, color="k", lw=0.7)
            ax.set(title=f"{version}\n{label}", ylabel="Signal / marginal σ")
            if row == 0:
                twin = sio.load(OUT / f"{B}_{suffix}.sacc", blind="glassmock-test")
                metrics.setdefault("blind_decompositions", {})[typ] = dict(
                    vector=equal(y, twin.mean[i], np.sqrt(np.diag(cov))),
                    covariance=equal(cov, twin.covariance.dense[np.ix_(i, i)]),
                )
    fig.tight_layout()
    fig.savefig(FIG / "bmode_decompositions.png", dpi=170)
    plt.close(fig)
    summary = OUT / "bmode_summary.json"
    if summary.exists():
        metrics["pipeline_bmode_summary"] = json.loads(summary.read_text())
    save()


def run_psf():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics["psf_fit"] = {}
    for j, version in enumerate((U, P)):
        path = next(OUT.glob(f"psf_fit_{version}_*.npz"))
        with np.load(path) as product:
            samples = product["flat_samples"]
        mean, std = samples.mean(axis=0), samples.std(axis=0)
        truth = np.array([0.0 if version == U else 0.1, 0.0, 0.0])
        metrics["psf_fit"][version] = {
            key: dict(
                mean=float(mu),
                std=float(sig),
                truth=float(t),
                offset_sigma=float((mu - t) / sig),
            )
            for key, mu, sig, t in zip(("alpha", "beta", "eta"), mean, std, truth)
        }
        metrics["psf_fit"][version]["joint_truth_test"] = pte(
            mean - truth, np.cov(samples, rowvar=False)
        )
        for k, ax in enumerate(axes):
            ax.errorbar(j, mean[k], yerr=std[k], fmt="o", label=version)
            ax.plot(j, truth[k], "kx", ms=7)
            ax.set(
                xticks=[0, 1],
                xticklabels=["clean", "α=0.1 injection"],
                title=["α", "β", "η"][k],
            )
    fig.tight_layout()
    fig.savefig(FIG / "psf_leakage.png", dpi=170)
    plt.close(fig)
    coefficients = OUT / "leakage_coefficients.json"
    if coefficients.exists():
        metrics["objectwise_leakage"] = json.loads(coefficients.read_text())

    # These are independently computed rho/tau parts, not copied caches.
    def psfpart(version):
        return sio.load(
            next((OUT / "rho_tau_stats").glob(f"rho_tau_{version}_*.sacc")),
            blind=REG[version]["blind"],
        )

    su, sb = psfpart(U), psfpart(B)
    metrics["independent_psf_invariance"] = {}
    for typ in su.get_data_types():
        i, y, cov = block(su, typ)
        metrics["independent_psf_invariance"][typ] = dict(
            vector=equal(y, sb.mean[i], np.sqrt(np.diag(cov))),
            covariance=equal(cov, sb.covariance.dense[np.ix_(i, i)]),
        )
    save()


def run_scan():
    from sp_validation.tests.blind_scan import assert_no_raw_values

    # The suite shares a root. This manifest includes every blinded-entry
    # product plus shared products, while excluding other entries' products.
    audit = ROOT / "blinded_tree"
    if audit.exists():
        shutil.rmtree(audit)
    audit.mkdir()
    manifest = []
    for path in OUT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(OUT)
        if any(v in str(relative) for v in (U, P, A)):
            continue
        dst = audit / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(path)
        manifest.append(str(relative))
    (ROOT / "blind_scan_manifest.json").write_text(json.dumps(manifest, indent=2))
    refs, expected = {}, {}
    for grid in ("reporting", "integration"):
        su, sb = xi(U, grid), xi(B, grid)
        refs[grid] = su.mean
        expected[grid] = sb.mean
    su, sb = harmonic(U), harmonic(B)
    i = su.indices(sio.CL_EE)
    refs["harmonic_EE"] = su.mean[i]
    expected["harmonic_EE"] = sb.mean[i]
    for suffix, typ in [
        ("cosebis", sio.COSEBI_EE),
        ("pure_eb", sio.PURE_TYPES["xip_E"]),
        ("pure_eb", sio.PURE_TYPES["xim_E"]),
    ]:
        su = sio.load(OUT / f"{U}_{suffix}.sacc", blind="none")
        sb = sio.load(OUT / f"{B}_{suffix}.sacc", blind="glassmock-test")
        i = su.indices(typ)
        refs[typ] = su.mean[i]
        expected[typ] = sb.mean[i]
    for name, raw in refs.items():
        valid = np.isfinite(raw) & (raw != 0)
        refs[name] = raw[valid]
        expected[name] = expected[name][valid]
    try:
        assert_no_raw_values(audit, refs, expected=expected)
    except AssertionError as error:
        metrics["blind_leak_gate"] = dict(
            pass_scan=False, error=str(error), files=len(manifest)
        )
        save()
        raise
    metrics["blind_leak_gate"] = dict(
        pass_scan=True,
        files=len(manifest),
        scope="Every blinded-entry product and shared output; manifest lists inclusions. Other mock-entry vectors are excluded. PNG pixels are not invertible by this gate; metadata is scanned.",
    )
    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stages",
        nargs="+",
        choices=[
            "harmonic",
            "additive",
            "blind_xi",
            "xi",
            "decompositions",
            "psf",
            "scan",
        ],
    )
    args = parser.parse_args()
    from importlib.metadata import PackageNotFoundError, version

    metrics["package_versions"] = {}
    for package in (
        "numpy",
        "scipy",
        "pyccl",
        "camb",
        "healpy",
        "treecorr",
        "pymaster",
        "sacc",
        "glass",
        "smokescreen",
    ):
        try:
            metrics["package_versions"][package] = version(package)
        except PackageNotFoundError:
            metrics["package_versions"][package] = "distribution not found"
    for stage in args.stages:
        globals()["run_" + stage]()
        print("Completed", stage, flush=True)
