"""Guard: the workflow sets every CosmologyValidation constructor default.

``workflow/common.py::cv_init_params`` builds the constructor kwargs for every
cosmo_val rule. A keyword it does not forward falls back to the constructor
default without any trace in the run config, so each keyword with a default is
either forwarded or exempted below with its reason.
"""

import importlib.util
import inspect
from pathlib import Path

import yaml

from sp_validation.cosmo_val import CosmologyValidation

REPO = Path(__file__).resolve().parents[3]

EXEMPT = {
    "output_dir": "rules set the output tree via the run directory / COSMO_VAL",
    "blind": "None keeps the n(z) blind declared in the catalogue config",
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
