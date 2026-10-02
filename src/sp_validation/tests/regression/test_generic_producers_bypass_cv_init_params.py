"""The pseudo-Cl and rho/tau producer rules must honour config['cosmo_val'].

The cosmo_val suite builds its diagnostics from ``common.cv_init_params(config)``,
so ``papers/cosmo_val/config/config.yaml::cosmo_val`` looks like the single
place that sets nside, ell binning power, the pseudo-Cl estimator and the rho/tau
covariance. The products that actually reach the SACC and the B-mode summary are
made by the generic rules ``pseudo_cl``, ``pseudo_cl_cov`` and ``rho_tau_stats``
(``workflow/rules/twopoint.smk``), whose scripts build their own
``CosmologyValidation``. These tests evaluate each rule's ``params:`` block
against a config whose ``cosmo_val`` values all differ from the hardcoded ones,
run the rule's script with an injected ``snakemake`` object, and capture the
constructor call (a spy stops execution there, so no catalogue is read). The
expected value of every key is the configured one by construction: a key that
lives in ``config['cosmo_val']`` and is forwarded to the diagnostics must reach
the producer of the same quantity too.
"""

import inspect
import io
import os
import re
import runpy
import sys
import types
from pathlib import Path

import pytest
import yaml

import sp_validation.cosmo_val as cv_pkg

REPO = Path(__file__).resolve().parents[4]
# When testing another checkout's code, use its matching workflow scripts too.
ACTIVE_REPO = Path(cv_pkg.__file__).resolve().parents[3]
if ACTIVE_REPO != REPO:
    REPO = ACTIVE_REPO
SCRIPTS = REPO / "workflow" / "scripts"
REAL_CV = cv_pkg.CosmologyValidation

# Every value differs from the hardcoded literal in the rule/script and from the
# CosmologyValidation default, so a value that reaches the constructor can only
# have come from the config.
PSEUDO_CL_OVERRIDES = dict(
    nside=512,
    power=0.25,
    cell_method="map",
    nrandom_cell=7,
    noise_bias_method="randoms",
    fiducial_input_inka="decoupled",
    cell_seed=1234,
    pol_factor=False,
)
RHO_TAU_OVERRIDES = dict(cov_estimate_method="jk", compute_cov_rho=False)


class _Captured(Exception):
    pass


class _Spy:
    calls = []

    def __init__(self, *args, **kwargs):
        bound = inspect.signature(REAL_CV.__init__).bind(None, *args, **kwargs)
        bound.apply_defaults()
        effective = dict(bound.arguments)
        effective.pop("self")
        _Spy.calls.append(effective)
        raise _Captured


class _Params(dict):
    def __getattr__(self, key):
        return self[key]


def _config():
    if not (REPO / "papers/cosmo_val/config/config.yaml").is_file():
        pytest.skip("Repository does not contain the cosmo_val paper workflow")
    config = yaml.safe_load(
        (REPO / "papers" / "cosmo_val" / "config" / "config.yaml").read_text()
    )
    config["cosmo_val"].update(PSEUDO_CL_OVERRIDES)
    config["cosmo_val"].update(RHO_TAU_OVERRIDES)
    return config


def _rule_params(rule, wildcards, config, tmp_path):
    """Evaluate a rule's ``params:`` block as Snakemake would for one job."""
    text = (REPO / "workflow" / "rules" / "twopoint.smk").read_text()
    block = re.search(rf"^rule {rule}:\n(.*?)(?=^\S)", text, re.M | re.S).group(1)
    body = re.search(r"^    params:\n((?:        .*\n|\s*\n)+)", block, re.M).group(1)
    common = runpy.run_path(str(REPO / "workflow" / "common.py"))
    namespace = dict(
        common=types.SimpleNamespace(**common),
        config=config,
        CAT_CONFIG=str(tmp_path / "cat_config.yaml"),
        COSMO_VAL=tmp_path,
        PLANCK18=None,
        CV_INIT=None,
    )
    raw = eval(f"dict(\n{body})", namespace)
    w = types.SimpleNamespace(**wildcards)
    params = _Params()
    for key, value in raw.items():
        if callable(value):
            value = value(w)
        elif isinstance(value, str):
            value = value.format(**wildcards)
        params[key] = value
    return params


@pytest.fixture
def spy(monkeypatch):
    _Spy.calls = []
    monkeypatch.setattr(cv_pkg, "CosmologyValidation", _Spy)
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setitem(
        sys.modules, "_spv_container", sys.modules.get("_spv_container")
    )
    monkeypatch.syspath_prepend(str(SCRIPTS))
    # Scripts that rebind stdout/stderr or chdir must not leak into the session.
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    monkeypatch.setattr(os, "fdopen", lambda *a, **k: io.StringIO())
    monkeypatch.setattr(os, "chdir", lambda *a, **k: None)
    original_runner = sys.modules.pop("cv_runner", None)
    try:
        yield _Spy
    finally:
        sys.modules.pop("cv_runner", None)
        if original_runner is not None:
            sys.modules["cv_runner"] = original_runner


def _mismatches(captured, expected):
    return {k: (captured[k], v) for k, v in expected.items() if captured[k] != v}


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=("#381: pseudo-Cl producers ignore cosmo_val settings"),
)
@pytest.mark.parametrize(
    "rule, script, output_key",
    [
        ("pseudo_cl", "generate_pseudo_cl.py", "pseudo_cl"),
        ("pseudo_cl_cov", "generate_pseudo_cl_cov.py", "pseudo_cl_cov"),
    ],
)
def test_pseudo_cl_producers_take_cosmo_val_settings(
    spy, tmp_path, rule, script, output_key
):
    """The SACC producers must receive the exact configured harmonic settings.

    Each override differs from the constructor default and hardcoded script
    value, so the configured value is the correct expectation by construction.
    The constructor spy stops before catalogue access, not before forwarding.
    """
    config = _config()
    wildcards = dict(version="SP_toy", binning="powspace", nbins="32", blind="A")
    params = _rule_params(rule, wildcards, config, tmp_path)
    smk = types.SimpleNamespace(
        params=params,
        output=types.SimpleNamespace(
            **{output_key: str(tmp_path / f"{rule}_SP_toy.out")}
        ),
    )
    ns = runpy.run_path(str(SCRIPTS / script), run_name="producer_under_test")
    with pytest.raises(_Captured):
        ns["_from_snakemake"](smk)
    (captured,) = spy.calls
    bad = _mismatches(captured, PSEUDO_CL_OVERRIDES)
    assert not bad, (
        f"rule {rule} builds CosmologyValidation with values that ignore "
        f"config['cosmo_val'] ({{key: (used, configured)}}): {bad}"
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=("#381: rho/tau producer ignores covariance settings"),
)
def test_rho_tau_producer_takes_cosmo_val_settings(spy, tmp_path, monkeypatch):
    """Rho/tau's covariance settings must equal config['cosmo_val'] overrides.

    The same configured settings already govern the diagnostics; the producer
    of the SACC statistic must use them too, not independent defaults.
    """
    config = _config()
    wildcards = dict(
        version="SP_toy", min_sep="1.0", max_sep="250.0", nbins="20", npatch="1"
    )
    params = _rule_params("rho_tau_stats", wildcards, config, tmp_path)
    smk = types.SimpleNamespace(params=params, output={})
    fake = types.ModuleType("snakemake.script")
    fake.snakemake = smk
    if "snakemake" not in sys.modules:
        monkeypatch.setitem(sys.modules, "snakemake", types.ModuleType("snakemake"))
    monkeypatch.setitem(sys.modules, "snakemake.script", fake)
    with pytest.raises(_Captured):
        runpy.run_path(str(SCRIPTS / "run_rho_tau.py"), init_globals={"snakemake": smk})
    (captured,) = spy.calls
    bad = _mismatches(captured, RHO_TAU_OVERRIDES)
    assert not bad, (
        "rule rho_tau_stats builds CosmologyValidation with values that ignore "
        f"config['cosmo_val'] ({{key: (used, configured)}}): {bad}"
    )
