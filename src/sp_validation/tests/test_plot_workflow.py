"""Dry-run the real plotting DAG with caches present and no catalogues.

Set SPV_TEST_SNAKEMAKE to a host Snakemake executable if it isn't in the image.
The test never launches a workflow job or opens a real science product.
"""

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
PLOT_RULES = [
    "cosmo_val_plots",
    "cv_plot_rho_stats",
    "cv_plot_tau_stats",
    "cv_plot_rho_tau_fits",
    "cv_plot_footprints",
    "cv_plot_objectwise_leakage",
    "cv_plot_weights",
    "cv_plot_2pcf",
    "cv_ratio_xi_sys_xi",
    "cv_plot_pure_eb",
    "cv_plot_cosebis",
    "cv_plot_pseudo_cl",
    "covariance_plot",
]


@pytest.mark.parametrize("pseudo_cl", [False, True])
def test_cached_plot_dag_has_no_compute_jobs(tmp_path, monkeypatch, pseudo_cl):
    executable = os.environ.get("SPV_TEST_SNAKEMAKE") or shutil.which("snakemake")
    if not executable:
        pytest.skip("Snakemake not installed; set SPV_TEST_SNAKEMAKE")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COSMO_VAL", str(tmp_path / "products"))
    monkeypatch.setenv("COSMO_INFERENCE", str(tmp_path / "inference"))
    monkeypatch.setenv("PYTHONUNBUFFERED", "1")
    # common.configure expects this fixed relative cosmology file at parse time.
    cosmology = tmp_path / "results/cosmology/planck18.json"
    cosmology.parent.mkdir(parents=True)
    shutil.copy(ROOT / "papers/cosmo_val/results/cosmology/planck18.json", cosmology)
    config = yaml.safe_load((ROOT / "papers/cosmo_val/config/config.yaml").read_text())
    config["versions"] = [
        "SP_v1.4.6.3"
    ]  # Public fixture entry; no real input is opened.
    config["cosmo_val"]["include_pseudo_cl"] = pseudo_cl
    configfile = tmp_path / "config.yaml"
    configfile.write_text(yaml.safe_dump(config))

    spec = importlib.util.spec_from_file_location(
        "plot_test_common", ROOT / "workflow/common.py"
    )
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    # Set the path-helper context without importing Snakemake in the image;
    # the subprocess below runs the real configure() in the host environment.
    common.CATALOG_CONFIG = yaml.safe_load(Path(common.CAT_CONFIG).read_text())
    common.FIDUCIAL = config["fiducial"]
    common.DEFAULT_MASK_SUFFIX = (
        "_masked" if config["covariance"].get("default_masked", False) else ""
    )
    namespace = dict(vars(common), config=config)
    # Execute only the ordinary Python path helpers, before the first rule.
    twopoint = (ROOT / "workflow/rules/twopoint.smk").read_text().split("rule xi:")[0]
    exec(twopoint, namespace)
    diagnostics = (
        (ROOT / "workflow/rules/cosmo_val.smk")
        .read_text()
        .split("rule cv_plot_rho_stats:")[0]
    )
    exec(diagnostics, namespace)
    ns = namespace
    version = config["versions"][0]
    products = [
        *ns["CV_PSF_PRODUCTS"],
        *ns["CV_FOOTPRINT_MAPS"],
        str(common.COSMO_VAL / "weight_hist.npz"),
        str(common.COSMO_VAL / "leakage_coefficients.json"),
        common.cv_rho_stats(version, ns["CV_FIDUCIAL"]),
        common.cv_tau_stats(version, ns["CV_FIDUCIAL"]),
        *ns["cv_tau_covariances"](),
        ns["cv_xi_sacc"](version, "reporting"),
        ns["cv_pure_eb_npz"](version),
        ns["cv_cosebis_npz"](version),
        ns["cv_xi_cov"](version),
        ns["cv_xi_cov_integration"](version),
    ]
    if pseudo_cl:
        products += [
            ns["cv_pseudo_cl_analysis_sacc"](version),
            ns["cv_pseudo_cl_cov"](version),
        ]
    for product in products:
        path = Path(product)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    command = [
        executable,
        "-s",
        str(ROOT / "workflow/Snakefile"),
        "--configfile",
        str(configfile),
        "--cores",
        "1",
        "--dry-run",
        "--forceall",
        "--workflow-profile",
        "none",
        "--rerun-triggers",
        "mtime",
        "--allowed-rules",
        *PLOT_RULES,
        "--",
        "cosmo_val_plots",
    ]
    result = subprocess.run(command, text=True, capture_output=True, timeout=120)
    log = result.stdout + result.stderr
    assert result.returncode == 0, log
    assert "cv_plot_weights" in log
    assert "cv_plot_footprints" in log
    assert "cv_plot_pure_eb" in log
    assert "cv_plot_cosebis" in log
    assert ("rule cv_plot_pseudo_cl:" in log) == pseudo_cl
    for forbidden in (
        "cv_weights",
        "cv_footprints",
        "cv_rho_tau_fits",
        "cv_cosebis",
        "cv_pure_eb",
        "xi",
        "rho_tau_stats",
        "covariance_run",
    ):
        assert f"rule {forbidden}:" not in log
    # Existing PNGs must never hide a missing cache when allowed-rules prunes
    # a plot job. Populate *all* figures, then remove each cache in turn.
    figures = [
        *ns["CV_FOOTPRINT_FIGURES"],
        *ns["cv_pure_eb_figures"](version).values(),
        *ns["cv_cosebis_figures"](version).values(),
        *[
            str(common.COSMO_VAL / name)
            for name in (
                "weight_hist.png",
                "leakage_coefficients.png",
                "ratio_xi_sys_xi.png",
                "xi_pm_tomography_False.png",
                "xi_pm_theta_tomography_False.png",
                "rho_tau_stats/rho_stats.png",
                "rho_tau_stats/tau_stats.png",
                "rho_tau_stats/contours_tau_stat.png",
                "rho_tau_stats/xi_psf_sys_tomography_False_sample.png",
                "rho_tau_stats/xi_psf_sys_tomography_False_mean_and_std.png",
            )
        ],
        ns["cv_xi_cov"](version).removesuffix(".dat") + "_corr_plot.pdf",
        ns["cv_xi_cov_integration"](version).removesuffix(".dat") + "_corr_plot.pdf",
    ]
    if pseudo_cl:
        figures += [
            str(common.COSMO_VAL / f"cell_{name}.png") for name in ("ee", "eb", "bb")
        ]
    for figure in figures:
        path = Path(figure)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("existing figure")
    for product in products:
        path = Path(product)
        path.unlink()
        missing = subprocess.run(command, text=True, capture_output=True, timeout=120)
        log = missing.stdout + missing.stderr
        assert missing.returncode != 0, f"Missing cache {product} was ignored:\n{log}"
        assert "MissingInputException" in log
        assert "existing figure" == Path(figures[0]).read_text()
        path.touch()
