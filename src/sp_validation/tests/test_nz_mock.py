"""The SOM n(z) calibration on mock galaxies with known redshifts.

Small versions of rungs 0 and 1 of ``scripts/calibration/nz_mock_ladder.py``:
a 12² SOM, 100 groups, 3000 calibrators and 1e5 sources.
"""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

from sp_validation.nz import mock, pipeline

RECIPE = Path(__file__).parents[3] / "config/calibration/nz/v2.yaml"
N_CAL, N_SRC = 3000, 100_000
# Across 40 realisations at this size, rung 0's Δ⟨z⟩ has |mean| ≤ 0.0015,
# scatter 0.0036–0.0047 per bin and largest |Δ| 0.0117 of 160; seed 0, used
# here, gives 0.0108 in bin 2.
TOLERANCE = 0.015


def calibrate(tmp_path, rung, seed=0, shuffle=False):
    """Calibrated, true and naive ⟨z⟩ per bin for one small realisation.

    ``shuffle`` permutes the calibrators' spectroscopic redshifts, breaking
    their link to the photometry.
    """
    config = yaml.safe_load(RECIPE.read_text())
    config.pop("ensemble")
    config["som"].update(side=12, epochs=15)
    config["groups"] = 100
    cal, src, z = mock.draw(rung, seed, N_CAL, N_SRC)
    if shuffle:
        cal["z"] = np.random.default_rng(seed).permutation(cal["z"])
    out = tmp_path / f"rung{rung}-{seed}-{shuffle}"
    summary = pipeline.calibrate(config, cal, src, out)
    tom = pq.read_table(out / "tom_bin.parquet")["TOM_BIN_ID"].to_numpy()
    edges = np.array(summary["edges"])
    result = []
    for k, b in enumerate(summary["bins"]):
        sel = tom == k + 1
        in_bin = (cal["photoz"] >= edges[k]) & (cal["photoz"] < edges[k + 1])
        result.append(
            (
                b["mean"],
                np.average(z[sel], weights=src["weight"][sel]),
                cal["z"][in_bin].mean(),
            )
        )
    return np.array(result).T  # calibrated, truth, naive


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("nz_mock")
    return {
        "rung0": calibrate(tmp, 0),
        "rung1": calibrate(tmp, 1),
        "shuffled": calibrate(tmp, 0, shuffle=True),
    }


def test_representative_calibrators_recover_each_bin(runs):
    calibrated, truth, _ = runs["rung0"]
    assert np.all(abs(calibrated - truth) < TOLERANCE)


def test_magnitude_selected_spectra_hurt_the_naive_estimate_more(runs):
    """At this size rung 1's calibrated ⟨z⟩ is itself low by 0.004–0.008 on
    average (the r selection's residual inside groups), the naive one by
    0.007–0.019, so only the comparison is asserted."""
    calibrated, truth, naive = runs["rung1"]
    assert np.mean(abs(naive - truth)) > np.mean(abs(calibrated - truth))


def test_shuffled_spectra_fail_the_recovery_check(runs):
    """Negative test: the recovery check of rung 0 catches broken spectra."""
    calibrated, truth, _ = runs["shuffled"]
    assert np.any(abs(calibrated - truth) > TOLERANCE)
