"""Trace real rule expressions, adapters and constructors without survey data.

Snakemake is not needed: only its params-expression and wildcard boundary is
reproduced. Expressions are read from the checkout rather than copied here.
A cosmology-factory spy records its inputs; construction stops before spectra
or covariance calculations. No harmonic settings or producer logic are patched.
"""

import copy
import importlib.util
import inspect
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from cs_util.cosmo import PLANCK18

import sp_validation
from sp_validation.cosmo_val import CosmologyValidation


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repo():
    root = Path(__file__).resolve().parents[4]
    # Follow the code under test when this file is run against another checkout.
    code_root = Path(sp_validation.__file__).resolve().parents[2]
    return root if code_root == root else code_root


def _rule_params(repo, producer, config, catalog, output):
    rule_path = repo / "workflow/rules/twopoint.smk"
    text = rule_path.read_text()
    rule = re.search(
        rf"^rule {producer}:\n(.*?)(?=^rule |\Z)", text, re.MULTILINE | re.DOTALL
    ).group(1)
    body = re.search(
        r"^    params:\n(.*?)(?=^    \w|\Z)", rule, re.MULTILINE | re.DOTALL
    ).group(1)
    common = _load_module(repo / "workflow/common.py", "_regression_common")
    # common.configure reads a generated JSON; use its canonical source if absent.
    planck_path = repo / common.COSMOLOGY_PARAMS
    planck = (
        json.loads(planck_path.read_text())
        if planck_path.exists()
        else copy.deepcopy(PLANCK18)
    )
    namespace = dict(
        vars(common),
        config=config,
        CAT_CONFIG=str(catalog),
        COSMO_VAL=output,
        PLANCK18=planck,
    )
    params = eval(compile("dict(\n" + body + "\n)", str(rule_path), "eval"), namespace)
    wildcards = SimpleNamespace(
        version="synthetic", blind="A", binning="powspace", nbins="4"
    )
    expanded = {}
    for key, value in params.items():
        if callable(value):
            value = value(wildcards)
        elif isinstance(value, str):
            value = value.format(**vars(wildcards))
        expanded[key] = value
    return expanded


def _build_producer_cv(repo, producer, config, monkeypatch, tmp_path):
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        yaml.safe_dump(
            {
                "nz": {"subdir": str(tmp_path)},
                "synthetic": {"subdir": str(tmp_path)},
                "paths": {"output": str(tmp_path)},
            }
        )
    )
    params = _rule_params(repo, producer, config, catalog, tmp_path)
    script = _load_module(
        repo / f"workflow/scripts/generate_{producer}.py",
        f"_regression_generate_{producer}",
    )
    core = inspect.getmodule(CosmologyValidation.__init__)
    monkeypatch.setattr(core, "get_cosmo", lambda **kwargs: copy.deepcopy(kwargs))
    built = []

    class ConstructorTraced(Exception):
        pass

    original_init = CosmologyValidation.__init__

    def record(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        built.append(self)
        raise ConstructorTraced

    monkeypatch.setattr(CosmologyValidation, "__init__", record)
    suffix = "sacc" if producer == "pseudo_cl" else "fits"
    smk = SimpleNamespace(
        params=params,
        output=SimpleNamespace(**{producer: str(tmp_path / f"{producer}.{suffix}")}),
    )
    with pytest.raises(ConstructorTraced):
        script._from_snakemake(smk)
    assert len(built) == 1, "Producer must construct one CV object"
    return built[0]


def _paper_config(repo):
    return yaml.safe_load((repo / "papers/cosmo_val/config/config.yaml").read_text())


@pytest.mark.xfail(
    strict=True,
    reason="#381: producers ignore harmonic overrides",
)
@pytest.mark.parametrize("producer", ["pseudo_cl", "pseudo_cl_cov"])
def test_pseudo_cl_producers_do_not_ignore_harmonic_overrides(
    repo, producer, monkeypatch, tmp_path
):
    """Both producers must retain the supplied cosmo_val harmonic settings.

    Distinct synthetic settings must appear on the actual CosmologyValidation
    object, which controls resolution, bandpowers, noise subtraction and randoms.
    Expected values come from the input config, not constructor defaults or a
    second implementation of the producer.
    """
    config = _paper_config(repo)
    default_pol = (
        inspect.signature(CosmologyValidation.__init__).parameters["pol_factor"].default
    )
    expected = dict(
        nside=8,
        power=0.75,
        cell_method="map",
        nrandom_cell=3,
        pol_factor=False if isinstance(default_pol, bool) else 1,
        noise_bias_method="randoms",
        fiducial_input_inka="decoupled",
        cell_seed=17,
    )
    config["cosmo_val"].update(expected)
    cv = _build_producer_cv(repo, producer, config, monkeypatch, tmp_path)
    differences = [
        f"{key}: actual={getattr(cv, key)!r}, expected={value!r}"
        for key, value in expected.items()
        if getattr(cv, key) != value
    ]
    assert not differences, f"{producer} ignores cosmo_val: " + "; ".join(differences)


@pytest.mark.xfail(
    strict=True,
    reason="#381: producers discard configured cosmology",
)
@pytest.mark.parametrize("producer", ["pseudo_cl", "pseudo_cl_cov"])
def test_pseudo_cl_producers_do_not_replace_configured_cosmology(
    repo, producer, monkeypatch, tmp_path
):
    """Preserve the covariance cosmology rather than a five-key Planck mapping.

    The exact cosmo_val.cosmo_params dictionary must reach get_cosmo: its native
    sig8/ns names and mnu/extra_params carry neutrino mass and CAMB settings.
    Equality to the input is the correct reference because no alternate
    cosmology was requested. This needs neither data nor theory computation.
    """
    config = _paper_config(repo)
    expected = copy.deepcopy(config["cosmo_val"]["cosmo_params"])
    cv = _build_producer_cv(repo, producer, config, monkeypatch, tmp_path)
    assert cv.cosmo == expected, (
        f"{producer} replaces cosmo_val.cosmo_params: "
        f"actual={json.dumps(cv.cosmo, sort_keys=True)}; "
        f"expected={json.dumps(expected, sort_keys=True)}"
    )
