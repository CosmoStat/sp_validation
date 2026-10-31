"""Geometry-only pair-count sharing is explicit, direct and opt-in."""

import pytest


def test_pair_count_reference_is_opt_in(toy):
    config = {"versions": ["clean", "blinded"]}
    assert toy.common.pair_count_reference(config, "blinded") == "blinded"
    config["covariance"] = {"pair_counts_reference": {"blinded": "clean"}}
    assert toy.common.pair_count_reference(config, "blinded") == "clean"
    assert toy.common.pair_count_reference(config, "clean") == "clean"


def test_pair_count_reference_rejects_missing_producer(toy):
    config = {
        "versions": ["blinded"],
        "covariance": {"pair_counts_reference": {"blinded": "missing"}},
    }
    with pytest.raises(ValueError, match="not a run version"):
        toy.common.pair_count_reference(config, "blinded")


def test_pair_count_reference_rejects_chains(toy):
    config = {
        "versions": ["clean", "blinded", "injected"],
        "covariance": {
            "pair_counts_reference": {"blinded": "injected", "injected": "clean"}
        },
    }
    with pytest.raises(ValueError, match="directly to a producer"):
        toy.common.pair_count_reference(config, "blinded")
