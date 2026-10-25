"""Pseudo-Cl plots must preserve signs, including error bars."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sp_validation.cosmo_val.pseudo_cl import plot_pseudo_cl_spectrum


def test_signed_cell(tmp_path, monkeypatch):
    figures = []
    monkeypatch.setattr(plt, "close", lambda fig: figures.append(fig))
    plot_pseudo_cl_spectrum({"toy": {"ell": np.array([100., 400.]),
                            "cl": np.array([-1., 1.]), "cov": np.eye(2) * 4}},
                            "EB", tmp_path / "eb.png")
    fig = next(fig for fig in reversed(figures) if hasattr(fig, "axes"))
    assert fig.axes[1].get_yscale() == "symlog"
    lo, hi = fig.axes[1].get_ylim()
    assert lo < -3 and hi > 3
