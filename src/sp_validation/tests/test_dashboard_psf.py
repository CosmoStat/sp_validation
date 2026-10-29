"""Cached PSF figures: scaling, independent grids and missing variances."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from sp_validation.dashboard import products as P
from sp_validation.dashboard import psf


@pytest.fixture
def rt():
    theta = np.array([1.0, 10.0, 100.0])
    return {
        "theta": theta,
        "theta_tau": theta,
        "rho": {k: {"p": np.ones(3) * 1e-5} for k in range(6)},
        "tau": {k: {"p": np.array([1., 2., 3.]) * 1e-7,
                    "var_p": np.array([1e-16, -1e-16, 4e-16])}
                for k in (0, 2, 5)},
        "fit": {"theta": theta, "flat_samples": np.ones((8, 3)),
                "samples_plus": np.ones((8, 3)) * 1e-7},
    }


@pytest.mark.parametrize("overlay", [False, True])
def test_theta_tau_scaling_and_own_grid(rt, overlay, tmp_path, monkeypatch):
    corrected = {"theta_tau": rt["theta_tau"] * 1.001, "tau": rt["tau"]}
    saved = []
    monkeypatch.setattr(psf.S, "save", lambda fig, *args: saved.append(fig))
    psf.theta_tau(rt, tmp_path / "theta_tau.png", corrected if overlay else None)
    fig = saved[0]
    for ax, k in zip(fig.axes, (0, 2, 5)):
        lines = {line.get_label(): line for line in ax.lines}
        np.testing.assert_allclose(lines["raw"].get_ydata(),
                                   rt["theta_tau"] * rt["tau"][k]["p"])
        if overlay:
            np.testing.assert_allclose(lines["leakage-corrected"].get_xdata(),
                                       corrected["theta_tau"])
            np.testing.assert_allclose(lines["leakage-corrected"].get_ydata(),
                                       corrected["theta_tau"] * rt["tau"][k]["p"])
        # One bar segment per positive variance, not a fictitious zero bar.
        assert len(ax.containers[0].lines[2][0].get_segments()) == 2
    fig.savefig(tmp_path / "theta_tau.png")  # exercise mathtext/layout too
    plt.close(fig)


@pytest.mark.parametrize("sign", [1, -1])
def test_xi_comparison_uses_blinded_points_and_errors(rt, sign, tmp_path, monkeypatch):
    xi = {"theta": rt["theta"] * 1.001,
          "xip": np.array([1., 2., 3.]) * 1e-5 * sign,
          "sigma_xip": np.full(3, 1e-6)}
    saved = []
    monkeypatch.setattr(psf.S, "save", lambda fig, *args: saved.append(fig))
    psf.xi_sys_comparison(rt, tmp_path / "xi.png", xi=xi)
    fig = saved[0]
    ax = fig.axes[0]
    container = ax.containers[0]
    line = container.lines[0]
    np.testing.assert_allclose(line.get_xdata(), xi["theta"])
    np.testing.assert_allclose(line.get_ydata(), xi["xip"])
    segments = container.lines[2][0].get_segments()
    np.testing.assert_allclose([s[1, 1] - s[0, 1] for s in segments],
                               2 * xi["sigma_xip"])
    assert "blinded" in container.get_label()
    assert ax.get_yscale() == ("log" if sign > 0 else "symlog")
    fig.savefig(tmp_path / "xi.png")
    plt.close(fig)


def test_xi_comparison_without_overlay(rt, tmp_path):
    out = tmp_path / "no_xi.png"
    psf.xi_sys_comparison(rt, out)
    assert out.exists()


def test_xi_minus_native_units_at_zero_crossing(rt, tmp_path, monkeypatch):
    rt["fit"]["samples_minus"] = np.ones((8, 3)) * 1e-7
    xi = {"theta": rt["theta"] * 1.001,
          "xip": np.full(3, 1e-5),
          "xim": np.array([0.0, -2e-6, 3e-6]),
          "sigma_xim": np.full(3, 3e-6)}
    saved = []
    monkeypatch.setattr(psf.S, "save", lambda fig, *args: saved.append(fig))
    with np.errstate(divide="raise", invalid="raise"):
        psf.xi_sys_ratio(rt, xi, tmp_path / "xi_sys.png")
    fig = saved[0]
    plus, minus = fig.axes
    # The plus panel remains the original ratio with +/-10% reference lines.
    np.testing.assert_allclose(plus.lines[3].get_ydata(), np.full(3, 0.01))
    np.testing.assert_allclose(plus.lines[0].get_ydata(), [-0.1, -0.1])
    # Zero line then theta-scaled posterior median; no xi- denominator.
    np.testing.assert_allclose(minus.lines[1].get_ydata(),
                               rt["fit"]["theta"] * 1e-7)
    container = minus.containers[0]
    np.testing.assert_allclose(container.lines[0].get_ydata(),
                               xi["theta"] * xi["xim"])
    segments = container.lines[2][0].get_segments()
    np.testing.assert_allclose([s[1, 1] - s[0, 1] for s in segments],
                               2 * xi["theta"] * xi["sigma_xim"])
    assert minus.get_yscale() == "linear"
    fig.savefig(tmp_path / "xi_sys.png")
    plt.close(fig)


def test_corrected_tau_loader_needs_no_assembled_or_fit(tmp_path, monkeypatch):
    stats = tmp_path / "cosmo_val/rho_tau_stats"
    stats.mkdir(parents=True)
    part = stats / "rho_tau_V_tomo_bin_all_minsep=1.sacc"
    cov = stats / "cov_tau_V_tomo_bin_all_minsep=1_th.npy"
    part.touch()
    cov.touch()
    monkeypatch.setattr(P, "load_tau", lambda s, c: (s, c))
    assert P.load_corrected_tau(tmp_path, "V") == (part, cov)
