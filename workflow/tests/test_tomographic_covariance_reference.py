"""Only identical tomographic surveys may share analytic covariances."""
from copy import deepcopy

import pytest


def setup_reference(toy, monkeypatch, tmp_path):
    nz = tmp_path / "nz.txt"
    nz.write_text("0 1 2\n1 2 1\n")
    entry = dict(cov_th=dict(A=100, tomo=dict(A=80, mask="tomo.fits", n_e=[1, 2], sigma_e=[.3, .3])),
                 mask="full.fits", psf=dict(mask_veto="veto.hsp"),
                 shear=dict(tomo_bin_col="BIN", redshift_path=str(nz), w_col="w", R=1))
    entries = dict(mock=deepcopy(entry), data=deepcopy(entry))
    monkeypatch.setattr(toy.common, "CATALOG_CONFIG", entries)
    config = dict(versions=["mock"], covariance=dict(tomographic_reference=dict(mock="data")))
    return config, entries


def test_external_reference_is_opt_in(toy, monkeypatch, tmp_path):
    config, _ = setup_reference(toy, monkeypatch, tmp_path)
    assert toy.common.tomographic_covariance_reference(config, "mock") == "data"
    assert toy.common.tomographic_covariance_reference({}, "mock") == "mock"
    config["covariance"]["reference_roots"] = {"data": {"cosmo_val": str(tmp_path)}}
    path = toy.common.COSMO_VAL / "pseudo_cl_cov_data_tomo.fits"
    assert toy.common.reference_product(config, "data", str(path), "cosmo_val") == str(tmp_path / path.name)


@pytest.mark.parametrize("key", ["mask", "area", "nz", "bins", "weights", "veto"])
def test_reference_refuses_mismatched_survey(toy, monkeypatch, tmp_path, key):
    config, entries = setup_reference(toy, monkeypatch, tmp_path)
    entry = entries["mock"]
    if key == "mask":
        entry["mask"] = "different.fits"
    elif key == "area":
        entry["cov_th"]["tomo"]["A"] = 70
    elif key == "nz":
        other = tmp_path / "other.txt"
        other.write_text("0 2 1\n1 1 2\n")
        entry["shear"]["redshift_path"] = str(other)
    elif key == "bins":
        entry["shear"]["tomo_bin_col"] = "OTHER_BIN"
    elif key == "weights":
        entry["shear"]["w_col"] = "different_weight"
    elif key == "veto":
        entry["psf"]["mask_veto"] = "other.hsp"
    with pytest.raises(ValueError, match="requires identical"):
        toy.common.tomographic_covariance_reference(config, "mock")


def test_reference_refuses_missing_and_chained_producer(toy, monkeypatch, tmp_path):
    config, _ = setup_reference(toy, monkeypatch, tmp_path)
    refs = config["covariance"]["tomographic_reference"]
    refs["mock"] = "missing"
    with pytest.raises(ValueError, match="Unknown"):
        toy.common.tomographic_covariance_reference(config, "mock")
    refs.update(mock="data", data="third")
    with pytest.raises(ValueError, match="cannot chain"):
        toy.common.tomographic_covariance_reference(config, "mock")
