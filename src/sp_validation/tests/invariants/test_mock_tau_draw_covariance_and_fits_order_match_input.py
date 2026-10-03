"""A seeded tau mock must preserve its full covariance and FITS component order.

The serialized plus columns match one statistic-major Gaussian draw; minus
columns are zero, and the mock ID alone determines a reproducible draw.
"""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import sp_validation

pytestmark = [pytest.mark.fast, pytest.mark.decision("mocks.mock_inference")]

REPO_ROOT = Path(sp_validation.__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT / "workflow" / "scripts" / "generate_glass_mock_rhotau_samples.py"
)


def _load_generate_samples_for_mock():
    """Execute the workflow script without writing bytecode beside its source."""
    namespace = {"__file__": str(SCRIPT_PATH), "__name__": "tau_mock_generator"}
    source = SCRIPT_PATH.read_bytes()
    exec(compile(source, str(SCRIPT_PATH), "exec"), namespace)
    return namespace["generate_samples_for_mock"]


def _read_tau_columns(path):
    with fits.open(path) as hdul:
        data = hdul[1].data
        return {name: np.array(data[name], copy=True) for name in hdul[1].columns.names}


def _plus_vector(columns):
    return np.concatenate([columns["tau_0_p"], columns["tau_2_p"], columns["tau_5_p"]])


def test_mock_tau_draw_covariance_and_fits_order_match_input(tmp_path):
    """The statistical false-alarm probability is 0 for these deterministic checks;
    same-seed comparison uses 1e-12 tolerances, and rank-one comparison allows 1e-7
    relative and 1e-9 absolute floating-point error.
    """
    generate = _load_generate_samples_for_mock()
    theta = np.array([5.0, 20.0])
    covariance = np.diag(np.arange(1.0, 7.0)) + 0.125 * np.ones((6, 6))
    header = {}

    reference = np.random.default_rng(7).multivariate_normal(np.zeros(6), covariance)
    first_dir = tmp_path / "first"
    generate(7, covariance, theta, header, first_dir)
    first = _read_tau_columns(first_dir / "00007" / "tau_stats_sampled.fits")
    sampled = _plus_vector(first)

    np.testing.assert_allclose(
        sampled,
        reference,
        rtol=1e-12,
        atol=1e-12,
        err_msg=(
            "tau-plus FITS columns must match the statistic-major N(0, C_tau) draw"
        ),
    )
    np.testing.assert_array_equal(first["theta"], theta)
    minus_columns = [first[f"tau_{stat}_m"] for stat in (0, 2, 5)]
    np.testing.assert_array_equal(np.concatenate(minus_columns), np.zeros(6))

    replay_dir = tmp_path / "replay"
    generate(7, covariance, theta, header, replay_dir)
    replay = _read_tau_columns(replay_dir / "00007" / "tau_stats_sampled.fits")
    assert first.keys() == replay.keys()
    for name in first:
        np.testing.assert_array_equal(replay[name], first[name])

    other_dir = tmp_path / "other-mock"
    generate(8, covariance, theta, header, other_dir)
    other = _read_tau_columns(other_dir / "00008" / "tau_stats_sampled.fits")
    assert not np.array_equal(_plus_vector(other), sampled), (
        "distinct mock IDs must not reuse the same tau draw"
    )

    rank_one_vector = np.arange(1.0, 7.0)
    rank_one_covariance = np.outer(rank_one_vector, rank_one_vector)
    rank_one_dir = tmp_path / "rank-one"
    generate(31, rank_one_covariance, theta, header, rank_one_dir)
    rank_one = _read_tau_columns(rank_one_dir / "00031" / "tau_stats_sampled.fits")
    ratios = _plus_vector(rank_one) / rank_one_vector
    np.testing.assert_allclose(
        ratios,
        np.full(6, ratios[0]),
        rtol=1e-7,
        atol=1e-9,
        err_msg="rank-one covariance must produce one shared scalar draw",
    )
