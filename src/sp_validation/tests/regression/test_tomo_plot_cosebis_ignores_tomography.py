"""plot_cosebis(compute_tomography=True) must produce per-bin-pair COSEBIs."""

import inspect
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from astropy.io import fits

cosebis_mod = pytest.importorskip(
    "sp_validation.cosmo_val.cosebis",
    reason="COSEBIs module required by the tomography branch",
)
from sp_validation.cosmo_val import CosmologyValidation  # noqa: E402

VER = "SP_toy"
NBINS = 60


def _make_cv(tmp_path):
    rng = np.random.default_rng(11)
    n = 600
    t = np.zeros(
        n,
        dtype=[
            ("RA", "f8"),
            ("Dec", "f8"),
            ("e1", "f8"),
            ("e2", "f8"),
            ("w", "f8"),
            ("tomo", "i4"),
        ],
    )
    t["RA"] = 10 + rng.uniform(-0.5, 0.5, n)
    t["Dec"] = rng.uniform(-0.5, 0.5, n)
    t["tomo"] = np.repeat([1, 2], n // 2)
    # Opposite coherent e1 in the two bins: the merged catalogue nearly cancels,
    # so the all-galaxy COSEBIs differ strongly from any single bin pair.
    t["e1"] = np.where(t["tomo"] == 1, 0.1, -0.1) + rng.normal(0, 0.01, n)
    t["e2"] = rng.normal(0, 0.01, n)
    t["w"] = 1.0
    cat = tmp_path / "toy.fits"
    fits.BinTableHDU(t).writeto(cat)
    out = tmp_path / "out"
    out.mkdir()
    cfg = {
        VER: {
            "blind": "none",
            "subdir": str(tmp_path),
            "shear": {
                "path": str(cat),
                "e1_col": "e1",
                "e2_col": "e2",
                "w_col": "w",
                "R": 1.0,
                "tomo_bin_col": "tomo",
            },
        },
        "nz": {"subdir": str(tmp_path)},
        "paths": {"output": str(out)},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    cv = CosmologyValidation(
        [VER],
        catalog_config=str(cfg_path),
        output_dir=str(out),
        compute_tomography=True,
        npatch=4,
    )
    cv.treecorr_config["num_threads"] = 2
    # Hand the table straight in, bypassing the leakage-correction reader.
    cv._results = {
        VER: SimpleNamespace(dat_shear=t, temporarily_read_data=lambda: nullcontext())
    }
    cv._c1 = {VER: 0.0}
    cv._c2 = {VER: 0.0}
    # Each bin pair takes its own jackknife covariance; a single cov_path cannot
    # serve every pair.
    params = dict(
        version=VER,
        min_sep_int=1.0,
        max_sep_int=30.0,
        nbins_int=NBINS,
        npatch=4,
        nmodes=1,
    )
    return cv, params


@pytest.mark.slow
def test_plot_cosebis_compute_tomography_true_stores_bin_pair_results(
    tmp_path, monkeypatch
):
    """plot_cosebis exposes a ``compute_tomography`` switch and saves/stores the
    COSEBIs data vector and PTEs it computes. With ``compute_tomography=True`` on
    a two-bin catalogue the stored results must be the three tomographic bin
    pairs (1,1), (1,2), (2,2), identical to what ``calculate_cosebis(...,
    compute_tomography=True)`` returns for the same inputs; the merged
    all-galaxy E_n is the wrong answer. The fixture gives the two bins opposite
    coherent e1 so the merged catalogue's E_1 differs from every bin pair's by
    a large factor, making the substitution unmistakable."""
    if (
        "compute_tomography"
        not in inspect.signature(CosmologyValidation.plot_cosebis).parameters
    ):
        pytest.skip("Tomography branch adds plot_cosebis(compute_tomography=...)")
    # The first COSEBIs call can compile numba kernels; keep their cache local.
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "numba-cache"))
    # Plot rendering is irrelevant; keep genuine computation and result storage.
    monkeypatch.setattr(cosebis_mod, "plot_cosebis_modes", lambda *a, **k: None)
    monkeypatch.setattr(
        cosebis_mod, "plot_cosebis_covariance_matrix", lambda *a, **k: None
    )
    cv, params = _make_cv(tmp_path)

    expected = cv.calculate_cosebis(**params, compute_tomography=True)
    assert set(expected) == {
        "tomo_bin_1_tomo_bin_1",
        "tomo_bin_1_tomo_bin_2",
        "tomo_bin_2_tomo_bin_2",
    }

    cv.plot_cosebis(**params, compute_tomography=True)
    stored = cv._cosebis_results[VER]

    stored_keys = sorted(stored) if isinstance(stored, dict) else None
    assert isinstance(stored, dict) and set(expected) <= set(stored), (
        "plot_cosebis(compute_tomography=True) stored non-tomographic COSEBIs: "
        f"keys={stored_keys}, En={np.asarray(stored.get('En')).tolist()}; "
        "expected bin pairs with En="
        f"{ {k: v['En'].tolist() for k, v in expected.items()} }"
    )
    for k, v in expected.items():
        np.testing.assert_allclose(stored[k]["En"], v["En"], rtol=1e-10)
        products = list((tmp_path / "out").glob(f"*_{k}_data.npz"))
        assert len(products) == 1
        with np.load(products[0]) as product:
            np.testing.assert_allclose(product["En"], v["En"], rtol=1e-10)
            np.testing.assert_allclose(product["Bn"], v["Bn"], rtol=1e-10)
