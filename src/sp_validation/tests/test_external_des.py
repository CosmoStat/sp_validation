"""Independent data-table policy for image-simulation calibration."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy.io import fits

from sp_validation import calibration, catalog_builders, response
from sp_validation.image_sim_config import calibration_config
from sp_validation.tests.test_consistent_response import catalogue


def make_product(tmp_path):
    data, _, _ = catalogue()
    gm = calibration.metacal(
        data,
        np.ones(len(data), bool),
        weight_scheme="des",
        size_corr_ell=False,
        des_num_bins=2,
    )
    path = tmp_path / "data-calibration.fits"
    fits.HDUList([fits.PrimaryHDU()]).writeto(path)
    calibration.save_response_product(path, gm)
    return dict(name="synthetic-data-calibration", path=str(path)), gm


def test_external_table_never_trains_on_sim_shapes_and_uses_fixed_jk(tmp_path):
    product, source = make_product(tmp_path)
    data, _, _ = catalogue()
    for suffix in ("NOSHEAR", "1P", "1M", "2P", "2M"):
        data[f"NGMIX_G1_{suffix}"][:20] += 0.3
    kwargs = dict(weight_scheme="des", size_corr_ell=False, des_weight_product=product)
    gm = calibration.metacal(data, np.ones(len(data), bool), **kwargs)
    # Product edges win, even when the sim's configured bin count differs.
    np.testing.assert_array_equal(gm.W[2], source.W[2])
    for b in response.BRANCHES:
        np.testing.assert_array_equal(gm.branch_weights[b], source.branch_weights[b])
    groups = np.repeat(np.arange(4), 20)
    labels, jk, cov = gm.response_jackknife(groups, refit_w=True)
    fixed = gm.response_jackknife(groups, refit_w=False)
    np.testing.assert_allclose(cov, fixed[2])
    for key in jk:
        np.testing.assert_allclose(jk[key], fixed[1][key])
    for j, label in enumerate(labels):
        keep = groups != label
        retained = calibration.metacal(data[keep], np.ones(keep.sum(), bool), **kwargs)
        np.testing.assert_allclose(jk["c_g"][j], retained.c_g, atol=1e-13)
        assert gm.refit_weights(keep) is gm.branch_weights
    path = tmp_path / "sim.fits"
    header = fits.Header()
    # Exercise the catalogue driver's config-to-header path with the named
    # product mapping, before attaching the response and weight extensions.
    catalog_builders.CalibrateCat().add_params_to_FITS_header(
        header, cm=dict(weight_scheme="des", des_weight_product=product)
    )
    fits.HDUList([fits.PrimaryHDU(header=header)]).writeto(path)
    calibration.save_response_product(path, gm, groups)
    with fits.open(path) as f:
        assert f[0].header["WPOLICY"] == "fixed-external"
        assert f[0].header["WSOURCE"] == "external"
        assert f[0].header["WPRODUCT"] == product["name"]
        np.testing.assert_allclose(f["R_JK"].data["c_g1"], jk["c_g"][:, 0])
        np.testing.assert_array_equal(f["W_DES"].data, source.W[2])


def test_reject_product_without_verified_coordinates(tmp_path):
    product, _ = make_product(tmp_path)
    with fits.open(product["path"], mode="update") as f:
        del f[0].header["WSIZDEF"]
    with pytest.raises(ValueError, match="coordinate metadata"):
        calibration.load_des_weight_product(product)


def test_staging_replaces_old_symlink_without_editing_shared_config(tmp_path):
    original = "metacal:\n  global_R_weight: null\n  additive_correction: true\n"
    source = tmp_path / "mask.yaml"
    source.write_text(original)
    staged = tmp_path / "config_mask.yaml"
    staged.symlink_to(source)
    script = (
        Path(__file__).resolve().parents[3]
        / "scripts/calibration/stage_image_sim_config.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(source),
            "--output",
            str(staged),
            "--w-cols",
            "w_des",
            "--product-name",
            "data",
            "--product-path",
            "/data.fits",
        ],
        check=True,
    )
    assert not staged.is_symlink()
    assert source.read_text() == original
    cm = yaml.safe_load(staged.read_text())["metacal"]
    assert cm["des_weight_product"] == dict(name="data", path="/data.fits")
    assert cm["weight_scheme"] == "des"


def test_image_sim_default_requires_named_external_product():
    original = dict(metacal=dict(global_R_weight=None, additive_correction=True))
    with pytest.raises(ValueError, match="require des_weight_product"):
        calibration_config(original, ["w_des"])
    product = dict(name="data-release", path="/calibration/data.fits")
    staged = calibration_config(original, ["w_des"], product=product)
    assert staged["metacal"]["des_weight_product"] == product
    assert staged["metacal"]["weight_scheme"] == "des"
    assert not staged["metacal"]["additive_correction"]
    assert "global_R_weight" not in staged["metacal"]
    assert "weight_scheme" not in original["metacal"]
    diagnostic = calibration_config(staged, ["w_des"], source="self")
    assert "des_weight_product" not in diagnostic["metacal"]
    assert calibration_config(original, ["w_iv"]) == original
