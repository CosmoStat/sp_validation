"""Smoke tests for workflow CLI seams — cheap guards against signature rot.

A CLI script that calls a workflow function with a removed/renamed kwarg
TypeErrors only at invocation time (the compute is cluster-only, so it is never
exercised by the fast suite). These tests bind the exact call each seam makes
against the current signature via ``inspect.signature(...).bind(...)`` — no
compute, no data — so a drifted kwarg (e.g. run_xi_sweep's dropped save_fits)
fails here instead of on the cluster.
"""

import importlib.util
import inspect
from pathlib import Path

import pytest


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("could not locate repo root (no pyproject.toml above test)")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_xi_sweep_run_2pcf_call_binds():
    """The kwargs run_xi_sweep passes to run_2pcf must bind to its signature.

    Mirrors the call in papers/bmodes/scripts/run_xi_sweep.py — if run_2pcf drops
    or renames a parameter (save_fits was removed by the SACC migration), the
    bind raises TypeError here rather than on every cluster invocation.
    """
    root = _repo_root()
    run_2pcf_mod = _load(root / "workflow/scripts/run_2pcf.py", "run_2pcf_seam")
    sig = inspect.signature(run_2pcf_mod.run_2pcf)
    # Exactly the keyword set run_xi_sweep._from_cli passes (grid params spread
    # from GRIDS: min_sep/max_sep/nbins/npatch).
    sig.bind(
        ver="V",
        cat_config="/cfg.yaml",
        output_dir="/out",
        sacc_out="/out/V_xi_reporting_reporting.sacc",
        min_sep=1.0,
        max_sep=250.0,
        nbins=20,
        npatch=1,
    )
    # And the removed kwarg must NOT bind (guards against a silent re-add).
    with pytest.raises(TypeError):
        sig.bind(
            ver="V",
            cat_config="/cfg.yaml",
            output_dir="/out",
            save_fits=True,
            min_sep=1.0,
            max_sep=250.0,
            nbins=20,
            npatch=1,
        )
