"""Synthetic tests of the SACC-to-CosmoSIS workflow boundary."""

import configparser
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from sp_validation import sacc_io as sio
from sp_validation.inference import prepare, render, summarize


def test_prepare_uses_reference_covariance_not_reference_signal(tmp_path):
    z = np.linspace(0.01, 2.01, 20)
    s = sio.new_sacc([(z, np.ones(20))], metadata={"blind": "mock"})
    theta = np.array([2.0, 4.0, 8.0])
    sio.add_xi(s, (0, 0), theta, theta, -theta, grid="reporting", theta_nom=theta)
    s.add_covariance(np.eye(6))
    original = tmp_path / "mock.sacc"
    s.save_fits(original)
    ref = s.copy()
    # Reference means and measured angles differ, nominal bins identify rows.
    for p in ref.data:
        p.value = 999.0
        p.tags["theta"] *= 1.001
    ref.add_covariance(np.eye(6) * 7, overwrite=True)
    reference = tmp_path / "covariance.sacc"
    ref.save_fits(reference)
    out = tmp_path / "out.fits"
    provenance = prepare(original, out, covariance=reference)
    assert provenance["blind"] == "mock"
    with fits.open(out) as hdus:
        np.testing.assert_array_equal(hdus["XI_PLUS"].data["VALUE"], theta)
        np.testing.assert_array_equal(hdus["XI_PLUS"].data["ANG"], theta)
        np.testing.assert_array_equal(hdus["COVMAT"].data, np.eye(6) * 7)
        np.testing.assert_array_equal(hdus["NZ_SOURCE"].data["BIN1"], np.ones(20))


def test_render_explicit_inputs_and_all_bin_cuts(tmp_path):
    template = tmp_path / "template.ini"
    template.write_text(
        "[pipeline]\nvalues = old\n[runtime]\nsampler = polychord\n"
        "[output]\nformat = text\n[2pt_like]\n"
        "angle_range_XI_PLUS_1_1 = 10.0 200.0\n"
    )
    out = tmp_path / "pipeline.ini"
    render(
        template,
        out,
        fits_path=tmp_path / "data.fits",
        values=tmp_path / "v.ini",
        priors=tmp_path / "p.ini",
        chain=tmp_path / "chain.txt",
        n_bins=4,
        settings={"n_live": 300},
        scale_cuts={"XI_PLUS": [12, 83]},
    )
    ini = configparser.ConfigParser(interpolation=None)
    ini.read(out)
    assert ini["pipeline"]["values"] == str(tmp_path / "v.ini")
    assert ini["pipeline"]["priors"] == str(tmp_path / "p.ini")
    assert ini["runtime"]["sampler"] == "nautilus"
    assert ini["nautilus"]["n_live"] == "300"
    assert ini["2pt_like"]["angle_range_XI_PLUS_3_4"] == "12.0 83.0"
    assert len([k for k in ini["2pt_like"] if k.startswith("angle_range")]) == 10


def test_psf_template_uses_likelihood_that_consumes_xi_sys(tmp_path):
    root = Path(__file__).resolve().parents[3]
    template = (
        root / "cosmo_inference/cosmosis_config/templates/cosmosis_pipeline_A_psf.ini"
    )
    out = tmp_path / "pipeline.ini"
    render(
        template,
        out,
        fits_path=tmp_path / "data.fits",
        values=tmp_path / "v.ini",
        priors=tmp_path / "p.ini",
        chain=tmp_path / "chain.txt",
    )
    ini = configparser.ConfigParser()
    ini.read(out)
    assert ini["2pt_like"]["file"].endswith("/likelihood/2pt/2pt_like_xi_sys.py")
    assert ini["2pt_like"].getboolean("add_xi_sys")


def test_weighted_summary(tmp_path):
    chain = tmp_path / "chain.txt"
    chain.write_text(
        "#cosmological_parameters--s_8 log_weight prior post\n"
        "0.7 0 0 0\n0.9 1.0986122886681098 0 0\n"
    )
    out = tmp_path / "summary.json"
    summarize(chain, out)
    result = json.loads(out.read_text())
    np.testing.assert_allclose(
        result["parameters"]["cosmological_parameters--s_8"]["mean"], 0.85
    )
    np.testing.assert_allclose(result["effective_samples"], 1.6)
