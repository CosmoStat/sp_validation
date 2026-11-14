"""Tomographic ρ reuse honours the measurement metadata of the cache."""

import numpy as np

from sp_validation.correlation import write_measurement_metadata
from sp_validation.rho_tau import rho_cache_valid

CONFIG = {"min_sep": 1.0, "max_sep": 100.0, "nbins": 5, "sep_units": "arcmin"}


def _cache(tmp_path, config=None):
    rho = tmp_path / "rho_stats.fits"
    cov = tmp_path / "cov_rho_jk.npy"
    rho.write_bytes(b"")
    np.save(cov, np.eye(2))
    if config is not None:
        write_measurement_metadata(rho, config)
    return rho, cov


def test_rho_cache_without_metadata_is_remeasured(tmp_path):
    rho, cov = _cache(tmp_path)
    assert not rho_cache_valid(rho, cov, CONFIG)


def test_rho_cache_with_other_means_settings_is_remeasured(tmp_path):
    rho, cov = _cache(tmp_path, {**CONFIG, "nbins": 6})
    assert not rho_cache_valid(rho, cov, CONFIG)


def test_rho_cache_matching_is_reused_unless_forced(tmp_path):
    rho, cov = _cache(tmp_path, CONFIG)
    assert rho_cache_valid(rho, cov, CONFIG)
    assert not rho_cache_valid(rho, cov, CONFIG, force_run=True)
    cov.unlink()
    assert not rho_cache_valid(rho, cov, CONFIG)
