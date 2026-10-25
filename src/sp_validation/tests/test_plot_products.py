"""Small, catalogue-free products exercise the compute/render boundary."""

from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from sp_validation import b_modes
from sp_validation.cosmo_val import diagnostic_plots as plots
from sp_validation.cosmo_val.catalog_characterization import (
    CatalogCharacterizationMixin,
)

STYLES = {"synthetic": {"colour": "C0", "marker": "o"}}


@pytest.fixture(autouse=True)
def no_catalogues(monkeypatch):
    from sp_validation import io

    def forbidden(*args, **kwargs):
        raise AssertionError("A renderer attempted a catalogue read")

    monkeypatch.setattr(io, "open_entry", forbidden)
    yield
    plt.close("all")


def assert_figures(*paths):
    for path in paths:
        assert Path(path).stat().st_size > 100


def psf_product():
    rng = np.random.default_rng(42)
    data = {
        "theta": np.geomspace(1, 100, 4),
        "flat_samples": rng.normal(size=(1000, 3)),
    }
    for component in ("plus", "minus"):
        samples = np.exp(rng.normal(-10, 0.1, (100, 4)))
        data.update(
            {
                f"samples_{component}": samples,
                f"mean_{component}": samples.mean(axis=0),
                f"var_{component}": samples.var(axis=0),
                f"quantiles_{component}": np.quantile(samples, [0.16, 0.84], axis=0),
            }
        )
    return data


def test_cached_psf_handlers_do_not_initialize_catalogues(tmp_path):
    import importlib.util

    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "plot_test_runner", root / "workflow/scripts/cv_runner.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    cv = SimpleNamespace(
        cc={"paths": {"output": str(tmp_path)}},
        treecorr_config={"min_sep": 1.0, "max_sep": 100.0, "nbins": 4},
    )
    runner.prepare_cached_psf(cv)
    assert cv._rho_stat_handler.catalogs._output == str(tmp_path / "rho_tau_stats")
    assert cv._tau_stat_handler.catalogs is cv._rho_stat_handler.catalogs


def test_objectwise_compute_uses_regression_not_plot_driver(tmp_path, monkeypatch):
    from contextlib import nullcontext

    from sp_validation.cosmo_val.psf_systematics import PSFSystematicsMixin, leakage

    params = {k: SimpleNamespace(value=0.01, stderr=0.001) for k in ("a11", "a22")}
    calls = []

    def regress(*args, **kwargs):
        calls.append(kwargs)
        return params

    monkeypatch.setattr(leakage, "corr_2d", regress)
    monkeypatch.setattr(leakage, "save_to_file", lambda *args: None)
    obj = SimpleNamespace(
        get_out_base=lambda *args, **kwargs: str(tmp_path / "fit"),
        temporarily_read_data=lambda **kwargs: nullcontext(),
        get_ellipticity_weights=lambda: ([np.zeros(4), np.zeros(4)], np.ones(4)),
        _dat={"p1": np.zeros(4), "p2": np.zeros(4)},
        _params={"e1_PSF_col": "p1", "e2_PSF_col": "p2", "verbose": False},
        _stats_file=None,
    )
    cv = PSFSystematicsMixin()
    cv.print_cyan = lambda *args: None
    result = cv._objectwise_leakage_bin(obj, "synthetic", "all", True, "lin")
    assert len(calls) == 1
    assert result["aii_mean"].nominal_value == pytest.approx(0.01)


def test_weights_roundtrip_and_overwrite(tmp_path):
    path = tmp_path / "weights.npz"
    w = np.array([0.1, 0.2, 0.2, 0.5, 0.9])

    # This is exactly the compute seam; no matplotlib histogram in the renderer.
    class CV(CatalogCharacterizationMixin):
        versions = ["synthetic"]
        results = {
            "synthetic": SimpleNamespace(
                temporarily_read_data=lambda: __import__("contextlib").nullcontext()
            )
        }

        def _read_shear_cols(self, *args):
            return w

    counts, edges = CV().weight_histograms(nbins=3)["synthetic"]
    assert counts.sum() == pytest.approx(w.sum())
    np.savez(path, counts=counts, edges=edges)
    with np.load(path, allow_pickle=False) as cache:
        hist = {"synthetic": (cache["counts"], cache["edges"])}
    output = tmp_path / "weights.png"
    output.write_text("stale")
    plots.plot_weights(hist, STYLES, output)
    assert_figures(output)


def test_footprint_every_region_from_one_map(tmp_path, monkeypatch):
    from sp_validation.cosmo_val.diagnostic_products import (
        footprint_counts,
        load_footprint,
        save_footprint,
    )

    count_map = footprint_counts([150.0, 150.0, 20.0], [40.0, 40.0, 30.0], 4, 16)
    assert count_map[count_map.valid_pixels].sum() == 3
    cache = tmp_path / "map.npz"
    save_footprint(count_map, cache)
    loaded = load_footprint(cache)
    calls = []

    def draw(self, data, region, outpath):
        np.testing.assert_array_equal(data.valid_pixels, count_map.valid_pixels)
        calls.append(region)
        Path(outpath).write_bytes(b"rendered" * 100)

    monkeypatch.setattr(plots.FootprintPlotter, "plot_region", draw)
    outputs = {r: tmp_path / f"{r}.png" for r in plots.FootprintPlotter._regions}
    plots.plot_footprint(loaded, outputs)
    assert calls == list(plots.FootprintPlotter._regions.values())
    assert_figures(*outputs.values())


def test_scalar_diagnostics_and_covariance(tmp_path):
    coeff = {
        "synthetic": {
            "tomo_bin_all": {
                key: [0.001, 0.0001]
                for key in ("aii_mean", "alpha_mean", "alpha_1", "alpha_0")
            }
        }
    }
    leakage, covariance = tmp_path / "leakage.png", tmp_path / "covariance.pdf"
    plots.plot_objectwise_leakage(coeff, STYLES, leakage)
    plots.plot_covariance(np.eye(4), covariance)
    assert_figures(leakage, covariance)


def test_psf_fit_and_sealed_ratio_rendering(tmp_path):
    data = psf_product()
    path = tmp_path / "psf.npz"
    np.savez(path, **data)
    with np.load(path, allow_pickle=False) as cache:
        products = {"synthetic": dict(cache)}
    contours, samples, summary, ratio = [
        tmp_path / f"{x}.png" for x in ("contours", "samples", "summary", "ratio")
    ]
    plots.plot_rho_tau_fits(products, STYLES, contours, samples, summary)
    gg = SimpleNamespace(
        meanr=data["theta"],
        xip=np.ones(4) * 1e-4,
        xim=np.ones(4) * 1e-4,
        varxip=np.ones(4) * 1e-12,
        varxim=np.ones(4) * 1e-12,
    )
    plots.plot_ratio_xi_sys_xi(products, {"synthetic": gg}, STYLES, ratio)
    xi_paths = [tmp_path / f"xi{i}.png" for i in range(2)]
    plots.plot_2pcf({"synthetic": gg}, STYLES, xi_paths)
    assert_figures(contours, samples, summary, ratio, *xi_paths)


@pytest.mark.parametrize("kind,indices", [("rho", range(6)), ("tau", (0, 2, 5))])
def test_psf_overlay_arrays(tmp_path, kind, indices):
    table = {"theta": np.geomspace(1, 100, 4)}
    for i in indices:
        table[f"{kind}_{i}_p"] = np.ones(4) * 1e-6
        table[f"var{kind}_{i}_p"] = np.ones(4) * 1e-14
    output = tmp_path / f"{kind}.png"
    plots.plot_psf_stats(
        {"synthetic": table},
        STYLES,
        output,
        kind,
        [np.eye(12) * 1e-14] if kind == "tau" else [],
    )
    assert_figures(output)


def test_pure_eb_plot_cache_is_complete(tmp_path):
    n = 3
    edges = np.geomspace(1, 100, n + 1)
    theta = np.sqrt(edges[:-1] * edges[1:])
    result = dict(
        theta=theta,
        left_edges=edges[:-1],
        right_edges=edges[1:],
        theta_int=theta,
        xip_int=np.ones(n) * 1e-5,
        xim_int=np.ones(n) * 1e-5,
        xip=np.ones(n) * 1e-5,
        xim=np.ones(n) * 1e-5,
        var_xip=np.ones(n) * 1e-12,
        var_xim=np.ones(n) * 1e-12,
        cov=np.eye(6 * n) * 1e-12,
        npatch=None,
    )
    result.update({k: np.ones(n) * 1e-6 for k in b_modes._EB_KEYS})
    result = b_modes.calculate_eb_statistics(result)
    path = tmp_path / "sealed_eb.npz"
    parent = SimpleNamespace(metadata={"blind": "toy"})
    b_modes.save_pure_eb_results(result, path, derived_from=[parent])
    loaded = b_modes.load_pure_eb_results(path, blind="toy")
    with np.load(path, allow_pickle=False) as cache:
        assert cache["blind"].item() == "toy"
    assert loaded["npatch"] is None
    for key in ("theta_int", "xip_int", "xim_int", "xip", "var_xip", "cov"):
        np.testing.assert_array_equal(result[key], loaded[key])
    outputs = [tmp_path / f"eb{i}.png" for i in range(4)]
    b_modes.plot_integration_vs_reporting(loaded, outputs[0], "synthetic")
    b_modes.plot_pure_eb_correlations(loaded, outputs[1], "synthetic")
    b_modes.plot_pte_2d_heatmaps(loaded, "synthetic", outputs[2])
    b_modes.plot_eb_covariance_matrix(
        loaded["cov"], "analytic", outputs[3], "synthetic"
    )
    assert_figures(*outputs)


def test_cosebis_transform_caches_selected_bins(tmp_path, monkeypatch):
    from cosmo_numba.B_modes import cosebis

    class SyntheticKernel:
        def __init__(self, **kwargs):
            pass

        def cosebis_from_xipm(self, theta, xip, xim, **kwargs):
            return np.ones(2), np.zeros(2)

        def cosebis_covariance_from_xipm_covariance(self, theta, cov):
            return np.eye(4)

    monkeypatch.setattr(cosebis, "COSEBIS", SyntheticKernel)
    edges = b_modes.log_bin_edges(0.08, 300, 1000)
    theta = np.sqrt(edges[0] * edges[1])
    results = b_modes.cosebis_scan_from_xi(
        theta,
        np.zeros(1000),
        np.zeros(1000),
        np.eye(2000),
        *edges,
        nmodes=2,
        scale_cuts=[(12, 83)],
    )
    np.testing.assert_array_equal(results[12, 83]["bin_indices"], [609, 843])
    path = tmp_path / "cuts.npz"
    b_modes.save_cosebis_results(
        results,
        path,
        (12, 83),
        edges,
        derived_from=[SimpleNamespace(metadata={"blind": "toy"})],
    )
    loaded, _ = b_modes.load_cosebis_results(path, blind="toy")
    np.testing.assert_array_equal(loaded[12, 83]["bin_indices"], [609, 843])


def test_cosebis_heatmap_uses_stored_inward_bins(tmp_path, monkeypatch):
    # Nearest edges would put this at (0, 2); the transform selected (1, 2).
    result = {(1.1, 7.9): dict(chi2_E=4.0, pte_B=0.7, bin_indices=[1, 2])}
    edges = (np.array([1.0, 2.0, 4.0]), np.array([2.0, 4.0, 8.0]))
    b_modes.plot_cosebis_scale_cut_heatmap(
        result, edges, "synthetic", tmp_path / "cut.png", (1.1, 7.9)
    )
    ax = plt.gcf().axes[1]
    matrix = ax.images[0].get_array()
    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == pytest.approx(0.7)
    assert np.ma.count(matrix) == 1
    assert ax.patches[0].get_xy() == (0, 0)
    assert ax.get_xticklabels()[0].get_text() == "2.00"
    assert ax.get_yticklabels()[0].get_text() == "4.00"


def test_notebook_cosebis_cache_and_heatmap_use_integration_edges(
    tmp_path, monkeypatch
):
    from sp_validation.cosmo_val import cosebis as module

    class CV(module.CosebisMixin):
        versions = ["synthetic"]
        cc = {"paths": {"output": str(tmp_path)}}
        treecorr_config = {"npatch": 1}
        xi_parts = {
            ("synthetic", "integration"): SimpleNamespace(metadata={"blind": "toy"})
        }
        _cosebis_results = {}

        def _binning(self, *args):
            return dict(min_sep=1.0, max_sep=8.0, nbins=3)

    edges = b_modes.log_bin_edges(1.0, 8.0, 10)
    results = {
        cut: dict(
            En=np.ones(2),
            Bn=np.zeros(2),
            cov=np.eye(4),
            chi2_E=2.0,
            chi2_B=0.0,
            pte_B=1.0,
            bin_indices=indices,
        )
        for cut, indices in [((1.0, 8.0), [0, 10]), ((1.0, 4.0), [0, 6])]
    }
    monkeypatch.setattr(module, "plot_cosebis_modes", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "plot_cosebis_covariance_matrix", lambda *args: None)

    def heatmap(data, used_edges, *args, **kwargs):
        np.testing.assert_array_equal(used_edges, edges)

    monkeypatch.setattr(module, "plot_cosebis_scale_cut_heatmap", heatmap)
    CV().plot_cosebis(results=results, min_sep_int=1.0, max_sep_int=8.0, nbins_int=10)
    _, cached_edges = b_modes.load_cosebis_results(
        next(tmp_path.glob("*.npz")), blind="toy"
    )
    np.testing.assert_array_equal(cached_edges, edges)


def test_cosebis_preserves_all_cuts_for_plotting(tmp_path):
    edges = np.array([1.0, 2.0, 4.0, 8.0])
    results = {
        (lo, hi): dict(
            En=np.array([1.0, 2.0]),
            Bn=np.array([0.1, 0.2]),
            cov=np.eye(4),
            chi2_E=5.0,
            chi2_B=0.05,
            pte_B=0.9,
            bin_indices=np.array([i, j]),
        )
        for i, lo in enumerate(edges[:-1])
        for j, hi in enumerate(edges[i + 1 :], start=i + 1)
    }
    path = tmp_path / "sealed_cosebis.npz"
    parent = SimpleNamespace(metadata={"blind": "toy"})
    b_modes.save_cosebis_results(
        results, path, (1.0, 8.0), (edges[:-1], edges[1:]), derived_from=[parent]
    )
    loaded, loaded_edges = b_modes.load_cosebis_results(path, blind="toy")
    with np.load(path, allow_pickle=False) as cache:
        assert cache["blind"].item() == "toy"
    assert loaded.keys() == results.keys()
    np.testing.assert_array_equal(loaded_edges, (edges[:-1], edges[1:]))
    with np.load(path, allow_pickle=False) as cache:
        assert float(cache["pte_B"]) == 0.9  # Existing summary consumer.
    outputs = [tmp_path / f"cosebis{i}.png" for i in range(3)]
    b_modes.plot_cosebis_modes(loaded[1.0, 8.0], "synthetic", outputs[0])
    b_modes.plot_cosebis_covariance_matrix(
        loaded[1.0, 8.0], "synthetic", "analytic", outputs[1]
    )
    b_modes.plot_cosebis_scale_cut_heatmap(
        loaded, loaded_edges, "synthetic", outputs[2]
    )
    assert_figures(*outputs)
