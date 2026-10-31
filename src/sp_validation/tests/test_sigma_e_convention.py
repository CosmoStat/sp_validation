"""Regression tests for the per-component shape-noise convention."""

import ast
from pathlib import Path

import numpy as np
import pytest
import yaml

from sp_validation.survey import effective_survey_stats, ellipticity_dispersion

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("version", "old_sigma"),
    [("SP_v1.4.6.3", 0.3783979556382064), ("DES", 0.29), ("SP_axel_v0.0", 0.3)],
)
def test_config_preserves_covariance_inputs(version, old_sigma):
    config = yaml.safe_load((ROOT / "cosmo_val/cat_config.yaml").read_text())
    entry = config[version]
    # Exercise the actual Snakemake parameter function without importing Snakemake.
    source = (ROOT / "workflow/rules/covariance.smk").read_text()
    function = source[source.index("def get_onecov_cat_params(") :]
    function = function[: function.index("\n\ndef ")]
    namespace = {"catalogue_entry": lambda _: entry}
    exec(compile(ast.parse(function), "covariance.smk", "exec"), namespace)
    sigma, density, _ = namespace["get_onecov_cat_params"](version)
    sigma, density = float(sigma), float(density)
    assert sigma == pytest.approx(old_sigma / np.sqrt(2))
    assert sigma**2 / density == pytest.approx(old_sigma**2 / (2 * density))
    # CosmoCov's summed-component input, when constructed from this value.
    assert sigma * np.sqrt(2) == pytest.approx(old_sigma)


def test_survey_stats_per_component():
    e1 = np.array([0.1, 0.4, -0.3])
    e2 = np.array([0.2, -0.1, 0.5])
    weights = np.array([1.0, 2.0, 3.0])
    stats = effective_survey_stats(e1, e2, weights, 1.0)
    assert stats["sigma_e"] == pytest.approx(ellipticity_dispersion(e1, e2, weights))
