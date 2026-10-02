"""Generated real-data pipelines must preserve the published fiducial priors."""

import configparser
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import sp_validation

EXPECTED_M = -0.057
EXPECTED_SIGMA = 0.014
# Configuration-space paper, Table tab:inference_priors; alpha uses the
# fiducial chain's unrounded mean (the paper rounds it to 0.005).
OTHER_FIDUCIAL_PRIORS = {
    ("nofz_shifts", "bias_1"): (-0.0030, 0.018),
    ("psf_leakage_parameters", "alpha"): (0.0051, 0.0022),
    ("intrinsic_alignment_parameters", "a"): (0.83, 0.78),
    ("cosmological_parameters", "ombh2"): (0.02218, 0.00055),
}


def _repository_root():
    root = Path(__file__).resolve().parents[4]
    # Cross-checks keep this file but import another checkout via PYTHONPATH.
    # The generated pipeline must use that checkout's script and templates.
    source_root = Path(sp_validation.__file__).resolve().parents[2]
    if source_root != root and (source_root / "pyproject.toml").is_file():
        return source_root
    return root


def _load_module(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Dynamic imports must not leave __pycache__ beside repository scripts.
    with monkeypatch.context() as patch:
        patch.setattr(sys, "dont_write_bytecode", True)
        spec.loader.exec_module(module)
    return module


def _read_ini(path):
    # Generated templates can repeat unrelated options; last value wins.
    config = configparser.ConfigParser(interpolation=None, strict=False)
    with path.open() as stream:
        config.read_file(stream)
    return config


@pytest.fixture
def generated_pipeline(request, tmp_path, monkeypatch):
    inference_root = _repository_root() / "cosmo_inference"
    fitting = _load_module(
        "regression_cosmosis_fitting",
        inference_root / "scripts/cosmosis_fitting.py",
        monkeypatch,
    )
    argv = [
        "cosmosis_fitting.py",
        "--cosmosis-root",
        "synthetic",
        "--data-dir",
        str(tmp_path),
        "--nz-file",
        "unused_nz.txt",
        "--xi",
        "unused_xip.fits",
        "unused_xim.fits",
        "--template-dir",
        str(inference_root / "cosmosis_config/templates"),
        "--output-root",
        str(tmp_path),
    ]
    if request.param:
        argv.append("--use-rho-tau")
    with monkeypatch.context() as patch:
        patch.setattr(sys, "argv", argv)
        args = fitting.parse_args()
    # Normally derived by __main__; no actual catalogue or sampler is needed.
    args.output_config_dir = str(tmp_path / "configs")
    args.config_name_base = "synthetic"
    args.config_relative_fits = "unused_synthetic.fits"
    fitting.generate_cosmosis_config(args)
    pipeline = _read_ini(tmp_path / "configs/cosmosis_pipeline_synthetic.ini")
    prior_path = inference_root / pipeline["pipeline"]["priors"]
    return request.param, pipeline, _read_ini(prior_path), prior_path


@pytest.mark.parametrize(
    "generated_pipeline",
    [
        pytest.param(False, id="without_rho_tau"),
        pytest.param(
            True,
            id="with_rho_tau",
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="#379: rho/tau m prior is -0.0057, not -0.057",
            ),
        ),
    ],
    indirect=True,
)
def test_rho_tau_keeps_image_simulation_m_prior_and_xi_calibration(
    generated_pipeline, monkeypatch
):
    """PSF nuisance inference must not change image-simulation calibration.

    Paper II specifies m=-0.057 +/- 0.014. The CLI parser and config builder
    choose a prior that CosmoSIS parses, then the configured CSL shear_m_bias
    module acts on synthetic unit xi in one source bin. Its independent
    required gain is (1-0.057)**2=0.889249 for both xi signs, with or without
    rho/tau. The non-rho/tau control passes on develop and remains unmarked:
    it protects the correct calibration from a future tomography regression.
    """
    datablock = pytest.importorskip("cosmosis.datablock")
    prior_module = pytest.importorskip("cosmosis.runtime.prior")
    _, pipeline, priors, prior_path = generated_pipeline
    prior = prior_module.Prior.parse_prior(priors["shear_calibration_parameters"]["m1"])
    assert isinstance(prior, prior_module.GaussianPrior)
    assert "shear_m_bias" in pipeline["pipeline"]["modules"].split()
    csl_dir = os.environ.get("CSL_DIR")
    if not csl_dir:
        pytest.skip("CSL_DIR is required to execute the CosmoSIS shear_m_bias module")
    module_path = Path(
        pipeline["shear_m_bias"]["file"].replace("%(COSMOSIS_DIR)s", csl_dir)
    )
    if not module_path.is_file():
        pytest.skip(f"CosmoSIS standard-library module missing: {module_path}")
    shear_bias = _load_module("regression_shear_m_bias", module_path, monkeypatch)
    options = datablock.DataBlock()
    options[datablock.option_section, "m_per_bin"] = pipeline.getboolean(
        "shear_m_bias", "m_per_bin"
    )
    options[datablock.option_section, "cl_section"] = pipeline["shear_m_bias"][
        "cl_section"
    ]
    block = datablock.DataBlock()
    block["shear_calibration_parameters", "m1"] = prior.mu
    sections = pipeline["shear_m_bias"]["cl_section"].split()
    for section in sections:
        block[section, "nbin"] = 1
        block[section, "bin_1_1"] = np.array([1.0])
    assert shear_bias.execute(block, shear_bias.setup(options)) == 0
    calibrated_xi = [float(block[s, "bin_1_1"][0]) for s in sections]
    expected_xi = (1.0 + EXPECTED_M) ** 2
    assert prior.mu == pytest.approx(EXPECTED_M, abs=1e-12), (
        f"{prior_path.name}: m1 mean={prior.mu:g}, sigma={prior.sigma:g}; "
        f"expected mean={EXPECTED_M:g}, sigma={EXPECTED_SIGMA:g}; "
        f"unit xi becomes {calibrated_xi[0]:.8f}, expected {expected_xi:.8f}"
    )
    assert prior.sigma == pytest.approx(EXPECTED_SIGMA, abs=1e-12)
    assert calibrated_xi == pytest.approx([expected_xi] * len(calibrated_xi), abs=1e-12)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#379: generated priors differ from published fiducial",
)
@pytest.mark.parametrize(
    "generated_pipeline",
    [False, True],
    ids=["without_rho_tau", "with_rho_tau"],
    indirect=True,
)
def test_generated_pipeline_preserves_other_published_fiducial_priors(
    generated_pipeline,
):
    """Guard the other prior differences reproduced in the duplicate audit.

    The paper and fiducial chain give Delta-z=-0.0030 +/- 0.018,
    alpha=0.0051 +/- 0.0022, A_IA=0.83 +/- 0.78 and
    omega_b*h**2=0.02218 +/- 0.00055. Alpha uses the chain's unrounded mean
    and is present only in PSF fits. Follow the generated pipeline's priors
    reference, so this checks distributions actually handed to the sampler,
    not an unused template. No posterior fit or external data is needed.
    """
    use_rho_tau, _, priors, prior_path = generated_pipeline
    differences = []
    for (section, parameter), expected in OTHER_FIDUCIAL_PRIORS.items():
        if section == "psf_leakage_parameters" and not use_rho_tau:
            continue
        words = priors[section][parameter].split()
        assert words[0].lower() == "gaussian", f"Unexpected prior: {words}"
        observed = tuple(float(value) for value in words[1:])
        if observed != pytest.approx(expected, rel=1e-12, abs=1e-12):
            differences.append(
                f"[{section}] {parameter}: got {observed}, expected {expected}"
            )
    assert not differences, (
        f"Generated priors={prior_path.name} disagree with published fiducial:\n"
        + "\n".join(differences)
    )
