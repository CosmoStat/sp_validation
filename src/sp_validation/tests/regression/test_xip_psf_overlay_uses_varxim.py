"""The xi+ vs PSF-systematic overlay must show xi+ uncertainties."""

from types import SimpleNamespace

import numpy as np
import pytest

from sp_validation.cosmo_val import real_space


def _make_validator(tmp_path):
    if not hasattr(real_space.RealSpaceMixin, "plot_2pcf"):
        pytest.skip("tomography branch removes plot_2pcf and its PSF overlay")

    class Validator(real_space.RealSpaceMixin):
        pass

    meanr = np.array([1.0, 3.0, 10.0, 30.0])
    gg = SimpleNamespace(
        meanr=meanr,
        npairs=np.full(4, 1e6),
        weight=np.full(4, 1e6),
        xip=np.array([4e-5, 2e-5, 1e-5, 5e-6]),
        xim=np.array([1e-5, 8e-6, 5e-6, 2e-6]),
        # Distinguish the variances by a factor of 100 (errors by a factor of 10).
        varxip=np.full(4, 1e-12),
        varxim=np.full(4, 1e-10),
    )
    validator = Validator()
    validator.versions = ["v1"]
    validator.cat_ggs = {"v1": gg}
    validator.cc = {"v1": {"ls": "-", "colour": "k"}}
    validator.treecorr_config = {"sep_units": "arcmin"}
    validator.theta_min_plot, validator.theta_max_plot = 0.5, 50.0
    psf = {"v1": {"mean": np.full(4, 1e-7), "var": np.full(4, 1e-16)}}
    validator._xi_psf_sys = psf
    validator.xi_psf_sys = psf
    validator._output_path = lambda name: str(tmp_path / name)
    validator.print_done = lambda msg: None
    # Supply the fixture's measurement; keep the plotting code unchanged.
    validator.calculate_2pcf = lambda ver, *a, **k: validator.cat_ggs[ver]
    return validator, gg


@pytest.mark.xfail(
    strict=True,
    reason="#393: xi+ PSF overlay uses xi- variance",
)
def test_xip_in_psf_sys_overlay_uses_sqrt_varxip_error_bars(tmp_path, monkeypatch):
    """Protect the xi_p_xi_psf_sys_{ver}.png overlay's xi+ error bars.

    The expected error is sqrt(varxip), TreeCorr's variance of xi+.
    The fixture sets varxim = 100 * varxip, so using sqrt(varxim) inflates
    the errors tenfold: 1e-5 instead of the correct 1e-6.
    The interactive overlay is reached only when xi_psf_sys exists on the
    object, so set that statistic directly and capture errorbar calls instead
    of reading the PNG.
    The tomography branch removes this plotting method and skips this test.
    """
    validator, gg = _make_validator(tmp_path)
    real_space.plt.switch_backend("Agg")
    calls = []
    original = real_space.plt.errorbar

    def spy(x, y, *args, **kwargs):
        calls.append((np.asarray(y), kwargs))
        return original(x, y, *args, **kwargs)

    monkeypatch.setattr(real_space.plt, "errorbar", spy)
    monkeypatch.setattr(real_space.cs_plots, "show", lambda *a, **k: None)
    initial_figures = set(real_space.plt.get_fignums())
    try:
        validator.plot_2pcf()

        xip_calls = [kw for _, kw in calls if kw.get("label") == r"$\xi_+$"]
        assert len(xip_calls) == 1, "xi+ vs xi_psf_sys overlay was not drawn"
        yerr = np.asarray(xip_calls[0]["yerr"])
        np.testing.assert_allclose(
            yerr,
            np.sqrt(gg.varxip),
            err_msg=(
                f"xi+ overlay errors {yerr} != sqrt(varxip) {np.sqrt(gg.varxip)}; "
                f"sqrt(varxim) = {np.sqrt(gg.varxim)}"
            ),
        )
    finally:
        for figure in set(real_space.plt.get_fignums()) - initial_figures:
            real_space.plt.close(figure)
