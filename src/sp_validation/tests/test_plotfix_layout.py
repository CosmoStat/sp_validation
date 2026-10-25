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
    gg = SimpleNamespace(meanr=np.array([1., 10.]), xip=np.array([1., -1.]),
                         xim=np.array([-1., 1.]), varxip=np.ones(2),
                         varxim=np.ones(2))
    plots.plot_2pcf({"toy": gg}, {"toy": {"marker": "o", "colour": "C0"}},
                    [tmp_path / "xi.png", tmp_path / "theta_xi.png"])
    assert all(fig.get_constrained_layout() for fig in figures if hasattr(fig, "axes"))
