"""Protect the order of GetDist credible limits returned by ``compute_limits``.

Each named parameter returns its upper and lower 68% endpoints, then its upper and
lower 95% endpoints. The stub mirrors GetDist's default contours (0.68, 0.95, 0.99),
so reading the 99% limit in place of the 95% one is caught.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.fast,
    pytest.mark.decision("inference.posterior_summary"),
]


def _limit(lower, upper):
    return SimpleNamespace(lower=lower, upper=upper)


class _MargeStats:
    def __init__(self):
        self._parameters = {
            "nuisance": SimpleNamespace(
                limits=(_limit(-0.4, 0.2), _limit(-0.8, 0.7), _limit(-1.2, 1.1))
            ),
            "s8": SimpleNamespace(
                limits=(_limit(0.7, 0.9), _limit(0.6, 1.1), _limit(0.5, 1.3))
            ),
        }

    def parWithName(self, name):
        return self._parameters[name]


class _Chain:
    def getMargeStats(self):
        return _MargeStats()


def _load_chain_postprocessing(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    module_path = Path.cwd() / "cosmo_inference/scripts/chain_postprocessing.py"
    spec = importlib.util.spec_from_file_location(
        "chain_postprocessing_under_test", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_credible_limits_preserve_confidence_level_and_upper_lower_order(
    tmp_path, monkeypatch
):
    """Exact equality on fixed endpoints has statistical false-alarm probability 0."""
    chain_postprocessing = _load_chain_postprocessing(tmp_path, monkeypatch)

    assert chain_postprocessing.compute_limits(_Chain(), "s8") == (
        0.9,
        0.7,
        1.1,
        0.6,
    )
