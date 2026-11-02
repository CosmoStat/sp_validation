"""Presentation regressions with synthetic arrays only."""

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sp_validation.cosmo_val import diagnostic_plots as plots


def test_xi_panels_use_constrained_layout(tmp_path, monkeypatch):
    figures = []
    monkeypatch.setattr(plt, "close", lambda fig: figures.append(fig))
    gg = SimpleNamespace(
        meanr=np.array([1.0, 10.0]),
        xip=np.array([1.0, -1.0]),
        xim=np.array([-1.0, 1.0]),
        varxip=np.ones(2),
        varxim=np.ones(2),
    )
    plots.plot_2pcf(
        {"toy": gg},
        {"toy": {"marker": "o", "colour": "C0"}},
        [tmp_path / "xi.png", tmp_path / "theta_xi.png"],
    )
    assert all(fig.get_constrained_layout() for fig in figures if hasattr(fig, "axes"))


def test_tau_panels_use_constrained_layout(tmp_path, monkeypatch):
    figures = []
    monkeypatch.setattr(plt, "close", lambda fig: figures.append(fig))
    table = {"theta": np.array([1.0, 10.0])}
    for i in (0, 2, 5):
        table[f"tau_{i}_p"] = np.zeros(2)
        table[f"vartau_{i}_p"] = np.ones(2)
    plots.plot_psf_stats(
        {"toy": table}, {"toy": {"colour": "C0"}}, tmp_path / "tau.png", "tau"
    )
    assert all(fig.get_constrained_layout() for fig in figures if hasattr(fig, "axes"))
