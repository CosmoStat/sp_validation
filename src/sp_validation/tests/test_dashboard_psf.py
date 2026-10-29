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


def test_corrected_tau_loader_needs_no_assembled_or_fit(tmp_path, monkeypatch):
    stats = tmp_path / "cosmo_val/rho_tau_stats"
    stats.mkdir(parents=True)
    part = stats / "rho_tau_V_tomo_bin_all_minsep=1.sacc"
    cov = stats / "cov_tau_V_tomo_bin_all_minsep=1_th.npy"
    part.touch()
    cov.touch()
    monkeypatch.setattr(P, "load_tau", lambda s, c: (s, c))
    assert P.load_corrected_tau(tmp_path, "V") == (part, cov)
