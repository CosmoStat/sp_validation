"""Signed PSF contamination must remain visible."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sp_validation.cosmo_val import diagnostic_plots as plots


def test_signed_psf_summary(tmp_path, monkeypatch):
    from shear_psf_leakage import plots as external
    monkeypatch.setattr(external, "plot_contours", lambda *a, **kw: None)
    figures = []
    monkeypatch.setattr(plt, "close", lambda fig: figures.append(fig))
    data = {"theta": np.array([1., 10.]), "flat_samples": np.ones((10, 3))}
    for component in ("plus", "minus"):
        data[f"samples_{component}"] = np.array([[1., -1.]])
        data[f"mean_{component}"] = np.array([1., -1.])
        data[f"quantiles_{component}"] = np.array([[0., -2.], [2., 0.]])
    plots.plot_rho_tau_fits({"toy": data}, {"toy": {"colour": "C0"}},
                           tmp_path / "contours.png", tmp_path / "samples.png",
                           tmp_path / "summary.png")
    for fig in figures:
        if hasattr(fig, "axes"):
            assert all(ax.get_yscale() == "linear" for ax in fig.axes)
            assert all(ax.get_ylim()[0] < 0 for ax in fig.axes)
