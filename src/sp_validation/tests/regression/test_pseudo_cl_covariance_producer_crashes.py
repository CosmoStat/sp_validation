"""Pseudo-Cl covariance producers must run against their branch's public API."""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyccl as ccl
import pytest

import sp_validation
from sp_validation.cosmo_val import CosmologyValidation


def _scripts():
    root = Path(__file__).resolve().parents[4]
    code_root = Path(sp_validation.__file__).resolve().parents[2]
    if code_root != root:
        root = code_root
    return root / "workflow/scripts"


def _cv_method_calls(script):
    """Return name, positional count and keyword names of each cv method call."""
    calls = []
    for node in ast.walk(ast.parse(script.read_text())):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "cv"
        ):
            calls.append(
                (node.func.attr, len(node.args), [k.arg for k in node.keywords])
            )
    return calls


class _ReachedCatalogue(Exception):
    """The real fiducial-spectrum step succeeded and reached catalogue loading."""


@pytest.mark.xfail(
    strict=True,
    reason="#374: covariance producer multiplies a theory dictionary",
)
def test_pseudo_cl_cov_does_not_multiply_fiducial_dictionary(tmp_path, monkeypatch):
    """The covariance must get past its real fiducial-C_ell/pixel-window step.

    cs_util returns a dictionary keyed by tomographic pair ('W1xW1'), so the
    method must extract the spectrum before multiplying by the HEALPix window.
    The catalogue loader raises a sentinel: reaching it is correct by
    construction, because theory must finish before data loading starts. Tiny
    synthetic n(z) and nside=16 keep this fast; no theory result is patched.
    The tomography branch removed this method, so the API tests cover it there.
    """
    mod = pytest.importorskip(
        "sp_validation.cosmo_val.pseudo_cl",
        reason="tomography branch reorganized the pseudo-Cl covariance module",
    )
    if not hasattr(CosmologyValidation, "calculate_pseudo_cl_eb_cov"):
        pytest.skip("tomography branch removed calculate_pseudo_cl_eb_cov")

    z = np.linspace(0.01, 2.0, 100)
    nz = np.exp(-0.5 * ((z - 0.7) / 0.2) ** 2)
    zpath = tmp_path / "nz.txt"
    np.savetxt(zpath, np.column_stack([z, nz]))

    def open_entry(*args, **kwargs):
        raise _ReachedCatalogue

    def noop(*args, **kwargs):
        pass

    monkeypatch.setattr(mod, "get_params_rho_tau", lambda *args, **kwargs: {})
    monkeypatch.setattr(mod, "open_entry", open_entry)
    stub = SimpleNamespace(
        nside=16,
        versions=["v"],
        cc={"v": {"shear": {"redshift_path": str(zpath)}}},
        cosmo=ccl.CosmologyVanillaLCDM(),
        noise_bias_method="analytic",
        _output_path=lambda name: str(tmp_path / name),
        print_start=noop,
        print_magenta=noop,
        print_cyan=noop,
        print_done=noop,
        get_namaster_bin=noop,
    )
    with pytest.raises(_ReachedCatalogue):
        CosmologyValidation.calculate_pseudo_cl_eb_cov(stub)


def test_generate_pseudo_cl_cov_does_not_call_removed_methods():
    """Every cv method called by the covariance rule's script must exist.

    Existence is required by Python before any covariance can be produced.
    This passes on develop and guards against the tomography merge removing
    calculate_pseudo_cl_eb_cov while the producer still calls it.
    """
    script = _scripts() / "generate_pseudo_cl_cov.py"
    missing = [
        name
        for name, _, _ in _cv_method_calls(script)
        if not hasattr(CosmologyValidation, name)
    ]
    assert not missing, (
        f"{script.name} calls missing CosmologyValidation methods: {missing}"
    )


def test_cv_pseudo_cl_does_not_omit_required_plot_arguments():
    """The plot_pseudo_cl call must bind to the active branch's signature.

    Signature binding is exactly what Python requires at call time, before
    the cv_pseudo_cl rule can write its FITS product. This guards tomography's
    required pol_list argument; develop lacks that branch's cv_pseudo_cl script.
    """
    script = _scripts() / "cv_pseudo_cl.py"
    if not script.exists():
        pytest.skip("cv_pseudo_cl.py is only on the tomography branch")
    if not hasattr(CosmologyValidation, "plot_pseudo_cl"):
        pytest.skip("plot_pseudo_cl API is only on the tomography branch")
    sig = inspect.signature(CosmologyValidation.plot_pseudo_cl)
    calls = [call for call in _cv_method_calls(script) if call[0] == "plot_pseudo_cl"]
    assert calls, "cv_pseudo_cl.py must call plot_pseudo_cl"
    for _, npos, kws in calls:
        sig.bind(object(), *([None] * npos), **{key: None for key in kws})
