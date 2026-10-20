"""Guard: the workflow builds ``cv`` and names its products as the class does.

``workflow/common.py::cv_init_params`` builds the constructor kwargs for every
cosmo_val rule. A keyword it does not forward falls back to the constructor
default without any trace in the run config, so each keyword with a default is
either forwarded or exempted below with its reason. ``cv_basename`` names the
rule outputs, so it must spell ``CosmologyValidation.basename`` exactly.
"""

import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import yaml

from sp_validation.cosmo_val import CosmologyValidation

REPO = Path(__file__).resolve().parents[3]

EXEMPT = {
    # The workflow is the non-tomographic DAG: every rule script names the bin
    # pair it computes, and the flag only switches the lazy properties on to
    # tomographic products no rule declares.
    "compute_tomography": "rule scripts pass tomography per call",
    # Snakemake owns staleness: it removes a job's declared outputs before
    # running it, so the methods' skip-if-exists only ever reuses intermediates.
    "force_run": "Snakemake decides what reruns",
}


def _load_common():
    spec = importlib.util.spec_from_file_location(
        "workflow_common", REPO / "workflow" / "common.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _defaulted_keywords():
    params = inspect.signature(CosmologyValidation.__init__).parameters.values()
    return {p.name for p in params if p.default is not inspect.Parameter.empty}


def test_every_default_is_forwarded_or_exempt():
    config = yaml.safe_load(
        (REPO / "papers" / "cosmo_val" / "config" / "config.yaml").read_text()
    )
    forwarded = set(_load_common().cv_init_params(config))
    unaccounted = _defaulted_keywords() - forwarded - set(EXEMPT)
    assert not unaccounted, (
        f"CosmologyValidation defaults not forwarded by cv_init_params: "
        f"{sorted(unaccounted)}. Forward them from config['cosmo_val'] or add "
        f"them to EXEMPT with a reason."
    )


def test_exemptions_are_live_keywords():
    stale = set(EXEMPT) - _defaulted_keywords()
    assert not stale, f"EXEMPT names keywords the constructor lacks: {sorted(stale)}"


def test_cv_basename_is_the_class_basename():
    fiducial = {"min_sep": 1.0, "max_sep": 250.0, "nbins": 20, "npatch": 100}
    cv = SimpleNamespace(
        treecorr_config={k: fiducial[k] for k in ("min_sep", "max_sep", "nbins")},
        npatch=fiducial["npatch"],
    )
    assert _load_common().cv_basename(
        "SP_v1.4.6.3", fiducial
    ) == CosmologyValidation.basename(cv, "SP_v1.4.6.3")
