"""plot_pure_eb must accept the bin-keyed results from calculate_pure_eb."""

import inspect

import numpy as np
import pytest
import treecorr

pure_eb = pytest.importorskip(
    "sp_validation.cosmo_val.pure_eb",
    reason="PureEBMixin is required for the tomography branch's bin-keyed API",
)
if not hasattr(pure_eb, "PureEBMixin"):
    pytest.skip(
        "PureEBMixin is missing (tomography branch API)", allow_module_level=True
    )
PureEBMixin = pure_eb.PureEBMixin
BIN_KEY = "tomo_bin_all_tomo_bin_all"


def _synthetic_gg(min_sep, max_sep, nbins, npatch):
    rng = np.random.default_rng(1)
    n = 8000
    cat = treecorr.Catalog(
        ra=rng.uniform(150.0, 152.0, n),
        dec=rng.uniform(1.0, 3.0, n),
        g1=rng.normal(0, 0.2, n),
        g2=rng.normal(0, 0.2, n),
        ra_units="deg",
        dec_units="deg",
        npatch=npatch,
        rng=rng,
    )
    gg = treecorr.GGCorrelation(
        min_sep=min_sep,
        max_sep=max_sep,
        nbins=nbins,
        var_method="jackknife",
        sep_units="arcmin",
        bin_slop=0.0,
    )
    gg.process(cat, num_threads=2)
    return gg


class _TomoCV(PureEBMixin):
    """Minimal host supplying the tomography branch's documented 2PCF mapping."""

    versions = ["v"]
    npatch = 32
    integration = None
    treecorr_config = {"min_sep": 2.0, "max_sep": 60.0, "nbins": 3}

    def __init__(self, output):
        self.cc = {"paths": {"output": str(output)}}
        self._pure_eb_results = {}

    def print_start(self, *args, **kwargs):
        pass

    def _binning(self, min_sep=None, max_sep=None, nbins=None):
        return {
            "min_sep": min_sep or self.treecorr_config["min_sep"],
            "max_sep": max_sep or self.treecorr_config["max_sep"],
            "nbins": nbins or self.treecorr_config["nbins"],
        }

    def _integration_binning(self, min_sep, max_sep, nbins, *, default):
        return self._binning(
            default[0] if min_sep is None else min_sep,
            default[1] if max_sep is None else max_sep,
            default[2] if nbins is None else nbins,
        )

    def calculate_2pcf_version(self, version, npatch, compute_tomography, **config):
        return {
            BIN_KEY: _synthetic_gg(
                config["min_sep"], config["max_sep"], config["nbins"], npatch
            )
        }


@pytest.mark.slow
def test_plot_pure_eb_passes_one_bin_result_to_eb_statistics(tmp_path, monkeypatch):
    """The statistics step needs a per-bin result, not the outer bin mapping.

    On the tomography branch calculate_pure_eb returns {bin_key: result}, even
    for the single (all, all) bin. plot_pure_eb must pass that inner result
    (carrying gg, cov and xip_B) to calculate_eb_statistics. A finite combined
    B-mode PTE is the expected outcome of this random-shear TreeCorr fixture.
    The real calculation and statistics run; only the 2PCF source and figure
    writers are substituted. Develop's single-result API is unaffected and
    skips because it lacks the tomography branch's compute_tomography argument.
    The real transform's eager JIT and jackknife resampling take over 20 seconds.
    """
    if (
        "compute_tomography"
        not in inspect.signature(PureEBMixin.calculate_pure_eb).parameters
    ):
        pytest.skip("Bin-keyed calculate_pure_eb API exists only on tomography branch")

    for name in (
        "plot_integration_vs_reporting",
        "plot_pure_eb_correlations",
        "plot_pte_2d_heatmaps",
        "plot_eb_covariance_matrix",
        "save_pure_eb_results",
    ):
        if hasattr(pure_eb, name):
            monkeypatch.setattr(pure_eb, name, lambda *args, **kwargs: None)

    seen = []
    real_statistics = pure_eb.calculate_eb_statistics

    def observe_statistics(results, **kwargs):
        seen.append(sorted(results)[:6])
        return real_statistics(results, **kwargs)

    monkeypatch.setattr(pure_eb, "calculate_eb_statistics", observe_statistics)
    cv = _TomoCV(tmp_path)
    try:
        cv.plot_pure_eb(
            versions=["v"],
            min_sep_int=0.5,
            max_sep_int=200.0,
            nbins_int=60,
            npatch=32,
        )
    except KeyError as error:
        pytest.fail(
            f"Statistics received keys {seen} rather than one per-bin result: {error}"
        )
    assert set(cv._pure_eb_results["v"]) == {BIN_KEY}
    pte = cv._pure_eb_results["v"][BIN_KEY]["pte_matrices"]["combined"]
    assert np.isfinite(pte).any(), "No finite combined B-mode PTE produced"
