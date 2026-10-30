"""Integration covariance sharing never shares a measured signal."""

import copy

import pytest
from conftest import VERSIONS, parse_jobs


def setup_reference(toy, monkeypatch):
    entries = {
        name: {
            "cov_th": {"sigma_e": 0.3, "n_e": 5, "A": 100},
            "mask": "same.fits",
            "shear": {"redshift_path": "same.txt"},
        }
        for name in ("clean", "blinded")
    }
    monkeypatch.setattr(toy.common, "catalogue_entry", entries.__getitem__)
    config = {
        "versions": list(entries),
        "covariance": {
            "pair_counts_reference": {"blinded": "clean"},
            "integration_reference": {"blinded": "clean"},
        },
    }
    return config, entries


def test_decomposition_dag_reads_reference_not_own_covariance(toy):
    import json

    target = str(toy.cosmo_val / f"{VERSIONS[1]}_pure_eb.sacc")
    config = {
        "integration_reference": {VERSIONS[1]: VERSIONS[0]},
        "pair_counts_reference": {VERSIONS[1]: VERSIONS[0]},
    }
    result = toy.snakemake("-n", target, config=["covariance=" + json.dumps(config)])
    assert result.returncode == 0, result.stdout
    (job,) = [j for j in parse_jobs(result.stdout) if j.rule == "cv_pure_eb"]
    assert str(toy.covariances[VERSIONS[0], "g"]) in job.input
    assert str(toy.covariances[VERSIONS[1], "g"]) not in job.input
    # The signal still belongs to the requesting catalogue, never its reference.
    assert any(f"{VERSIONS[1]}_xi_" in p for p in job.input)


def test_reference_is_opt_in(toy, monkeypatch):
    config, _ = setup_reference(toy, monkeypatch)
    assert toy.common.integration_covariance_reference(config, "blinded") == "clean"
    del config["covariance"]["integration_reference"]
    assert toy.common.integration_covariance_reference(config, "blinded") == "blinded"


@pytest.mark.parametrize("key", ["cov_th", "mask", "shear"])
def test_reference_rejects_mismatched_inputs(toy, monkeypatch, key):
    config, entries = setup_reference(toy, monkeypatch)
    entries["blinded"][key] = copy.deepcopy(entries["blinded"][key])
    if key == "cov_th":
        entries["blinded"][key]["n_e"] = 6
    elif key == "shear":
        entries["blinded"][key]["redshift_path"] = "different.txt"
    else:
        entries["blinded"][key] = "different.fits"
    with pytest.raises(ValueError, match="requires identical"):
        toy.common.integration_covariance_reference(config, "blinded")


def test_reference_requires_shared_geometry(toy, monkeypatch):
    config, _ = setup_reference(toy, monkeypatch)
    config["covariance"].pop("pair_counts_reference")
    with pytest.raises(ValueError, match="shared pair-count"):
        toy.common.integration_covariance_reference(config, "blinded")


def test_reference_rejects_missing_and_chained_producers(toy, monkeypatch):
    config, _ = setup_reference(toy, monkeypatch)
    refs = config["covariance"]["integration_reference"]
    refs["blinded"] = "missing"
    with pytest.raises(ValueError, match="not a run version"):
        toy.common.integration_covariance_reference(config, "blinded")
    refs.update(blinded="clean", clean="blinded")
    with pytest.raises(ValueError, match="directly to a producer"):
        toy.common.integration_covariance_reference(config, "blinded")
