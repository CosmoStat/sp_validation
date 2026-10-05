"""Two-pass measurement contracts, including unpatched single-pass use."""

import numpy as np
import treecorr

from sp_validation.correlation import measure_with_patches, process_gg


def test_unpatched_measurement_is_one_pass():
    cat = treecorr.Catalog(x=[0, 1, 2], y=[0, 0, 1], g1=[0.1] * 3, g2=[0.2] * 3)
    calls = []

    def measure(catalogs):
        calls.append(catalogs)
        return object()

    means, covariance = measure_with_patches(measure, {"a": cat})
    assert len(calls) == 1
    assert means is covariance


def test_covariance_draws_reuse_means_and_preserve_catalog_aliases():
    cat = treecorr.Catalog(
        x=[0, 1, 2, 3],
        y=[0, 0, 1, 1],
        g1=[0.1] * 4,
        g2=[0.2] * 4,
        npatch=2,
        rng=np.random.default_rng(1),
    )
    calls = []

    def measure(catalogs):
        assert catalogs["a"] is catalogs["b"]
        calls.append(catalogs["a"].npatch)
        return object()

    catalogs = {"a": cat, "b": cat}
    means, _ = measure_with_patches(measure, catalogs)
    reused, _ = measure_with_patches(measure, catalogs, means=means)
    assert calls == [2, 1, 2]
    assert reused is means


def test_unpatched_gg_matches_direct_measurement():
    rng = np.random.default_rng(1)
    cat = treecorr.Catalog(
        x=rng.random(500),
        y=rng.random(500),
        g1=rng.normal(size=500),
        g2=rng.normal(size=500),
    )
    config = dict(
        min_sep=0.01,
        max_sep=1,
        nbins=8,
        min_top=6,
        num_threads=1,
        bin_slop=0.1,
        var_method="shot",
    )
    direct = treecorr.GGCorrelation(config)
    direct.process(cat)
    measured = process_gg(config, cat)
    for name in ("xip", "xim", "meanr", "meanlogr", "weight", "npairs", "cov"):
        np.testing.assert_array_equal(getattr(measured, name), getattr(direct, name))
