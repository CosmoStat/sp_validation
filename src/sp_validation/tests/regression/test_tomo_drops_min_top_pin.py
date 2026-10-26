"""ξ± from CosmologyValidation must not depend on the host's CPU count."""

import multiprocessing

import numpy as np
import pytest
import treecorr
import yaml

from sp_validation.cosmo_val import CosmologyValidation


@pytest.fixture(autouse=True)
def _restore_treecorr_threads():
    previous = treecorr.get_omp_threads()
    try:
        yield
    finally:
        treecorr.set_omp_threads(previous)


def _cosmology_validation(tmp_path):
    """Minimal CosmologyValidation: only the constructor runs, no catalogue is read."""
    cfg = {
        "paths": {"output": str(tmp_path / "out")},
        "nz": {"subdir": str(tmp_path), "dndz": {"path": "nz.txt"}},
        "synthetic": {
            "blind": "none",
            "subdir": str(tmp_path),
            "shear": {"path": "cat.fits"},
        },
    }
    cfg_path = tmp_path / "cat_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return CosmologyValidation(
        ["synthetic"],
        catalog_config=str(cfg_path),
        npatch=8,
        theta_min=15.0,
        theta_max=70.0,
        nbins=6,
    )


def _synthetic_catalog(**kwargs):
    """4000 galaxies over ~4x4 deg with a spatially correlated shear field."""
    rng = np.random.default_rng(1234)
    n = 4000
    ra = rng.uniform(150.0, 154.0, n)
    dec = rng.uniform(0.0, 4.0, n)
    g1 = 0.02 * np.sin(np.radians(ra) * 90) + 0.25 * rng.standard_normal(n)
    g2 = 0.02 * np.cos(np.radians(dec) * 90) + 0.25 * rng.standard_normal(n)
    return treecorr.Catalog(
        ra=ra, dec=dec, g1=g1, g2=g2, ra_units="deg", dec_units="deg", **kwargs
    )


def _xi_on_node(cv, n_cpu, patch_centers, monkeypatch):
    """ξ± as calculate_2pcf_version builds it (instance npatch, jackknife, fixed
    patch centres, fresh Catalog), on a node with n_cpu CPUs."""
    monkeypatch.setattr(multiprocessing, "cpu_count", lambda: n_cpu)
    # Simulate the thread count seen by Field's default min_top calculation,
    # without actually launching 48 threads inside a two-core test allocation.
    # Leave min_top itself untouched: the missing pin must remain observable.
    monkeypatch.setattr(treecorr.field, "get_omp_threads", multiprocessing.cpu_count)
    config = {**cv._binning(), "var_method": "jackknife"}
    gg = treecorr.GGCorrelation(config)
    gg.process(_synthetic_catalog(patch_centers=patch_centers), num_threads=2)
    return gg


def test_xi_pm_identical_on_16_and_48_cpu_nodes(tmp_path, monkeypatch):
    """Protects the machine independence of ξ±. TreeCorr picks the depth of its root
    cells (min_top) as max(3, ceil(log2 n_threads)) when the config leaves it unset,
    and the thread count defaults to the node's CPU count; min_top decides which pairs
    bin_slop approximates, so ξ± changes between a 16-CPU node
    (min_top=4) and a 48-CPU node (min_top=6) once the catalogue is split into jackknife
    patches. The same catalogue, patch centres and CosmologyValidation settings must
    give the same ξ± on either node, to far below the jackknife σ (1e-6σ), which holds
    once the config pins min_top (and num_threads). This passes on develop,
    whose pin must survive the tomography merge. Only Field's view of the host
    thread count is simulated; the actual correlation runs on two cores.
    """
    cv = _cosmology_validation(tmp_path)
    patch_file = str(tmp_path / "patches.dat")
    treecorr.set_omp_threads(2)
    _synthetic_catalog(
        npatch=cv.npatch, rng=np.random.default_rng(99)
    ).write_patch_centers(patch_file)
    gg16 = _xi_on_node(cv, 16, patch_file, monkeypatch)
    gg48 = _xi_on_node(cv, 48, patch_file, monkeypatch)
    sigma = np.sqrt(np.concatenate([gg48.varxip, gg48.varxim]))
    shift = (
        np.abs(
            np.concatenate([gg16.xip, gg16.xim]) - np.concatenate([gg48.xip, gg48.xim])
        )
        / sigma
    )

    assert shift.max() < 1e-6, (
        f"ξ± moves by {shift.max():.3g}σ between a 16- and a 48-CPU node "
        f"(xip {shift[: cv.nbins].max():.3g}σ, xim {shift[cv.nbins :].max():.3g}σ); "
        f"treecorr_config min_top={cv.treecorr_config.get('min_top')}, "
        f"num_threads={cv.treecorr_config.get('num_threads')}"
    )
