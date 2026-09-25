"""Tests for the ξ± grid table in ``workflow/common.py``.

The table names the files the ``xi`` rule writes and the ones every consumer
asks for, so producer and consumer agree only if the tag is built from
canonical values. These tests pin that canonicalisation and the grid lookup.
"""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast


def _load_common():
    root = next(
        p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()
    )
    path = root / "workflow" / "common.py"
    spec = importlib.util.spec_from_file_location("wf_common_grids", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = _load_common()

# A cosmo_val block as YAML delivers it: integer-valued separations stay ints.
CONFIG = {
    "cosmo_val": {
        "theta_min": 1.0,
        "theta_max": 250.0,
        "nbins": 20,
        "npatch": 100,
        "integration": {"min_sep": 0.08, "max_sep": 300, "nbins": 1000},
        "cosebis": {"nmodes": 20, "scale_cuts": [[12, 83]]},
    }
}
FIDUCIAL = {
    "min_sep": 1.0,
    "max_sep": 250.0,
    "nbins": 20,
    "npatch": 1,
    "min_sep_int": 0.5,
    "max_sep_int": 300,
    "nbins_int": 1000,
}


def test_tag_is_built_from_canonical_values():
    """An integer YAML separation still names the file as a float.

    run_2pcf coerces separations with float() before TreeCorr writes, so a
    `max_sep: 300` that reached the tag as "300" would have the consumer ask
    for a path the producer never writes.
    """
    grids = common.xi_grids(CONFIG, FIDUCIAL)
    # Counts stay integers, so no "nbins=1000.0" creeps into a name.
    assert (
        common.grid_binning(grids["integration"])
        == "minsep=0.08_maxsep=300.0_nbins=1000_npatch=1"
    )


def test_grid_lookup_round_trips_through_the_tag():
    """Every grid's own binning resolves back to that grid.

    This is the producer/consumer contract: the rule resolves a job's grid from
    the wildcards its filename bound.
    """
    grids = common.xi_grids(CONFIG, FIDUCIAL)
    for name, grid in grids.items():
        binning = {key: grid[key] for key in common.XI_KEYS}
        assert common.grid_of(grids, binning) == name
        # Wildcards arrive as strings; the comparison is numeric.
        assert common.grid_of(grids, {k: str(v) for k, v in binning.items()}) == name


def test_one_integration_grid():
    """COSEBIs and pure-E/B share the integration grid; there is no third."""
    assert set(common.xi_grids(CONFIG, FIDUCIAL)) == {"reporting", "integration"}


@pytest.mark.parametrize(
    "npatch, cov", [(1, "diagonal"), ("1", "diagonal"), ("100", "jackknife")]
)
def test_covariance_mode_follows_the_patches(npatch, cov):
    """A patched part gets a jackknife block, an unpatched one the diagonal.

    Every part then carries variances: at npatch=1 TreeCorr's var_method is
    "shot", and its diagonal is the estimate it has. The rule passes the
    wildcard, a string.
    """
    assert common.patch_cov(npatch) == cov


def test_unnamed_binning_is_a_reporting_measurement():
    """The paper's convergence-check binning belongs to no named grid."""
    grids = common.xi_grids(CONFIG, FIDUCIAL)
    stray = {"min_sep": 1.0, "max_sep": 250.0, "nbins": 10000, "npatch": 1}
    assert common.grid_of(grids, stray) == "reporting"


def test_workflow_without_cosmo_val_falls_back_to_fiducial():
    """papers/bmodes carries no cosmo_val block; its grids come from FIDUCIAL."""
    grids = common.xi_grids({}, FIDUCIAL)
    assert grids["reporting"]["npatch"] == 1
    assert grids["integration"]["min_sep"] == 0.5
