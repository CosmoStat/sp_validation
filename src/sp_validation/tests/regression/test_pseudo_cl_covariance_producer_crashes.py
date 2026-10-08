"""Pseudo-Cl covariance producers must run against their branch's public API."""

import ast
import inspect
from pathlib import Path

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


def test_pseudo_cl_cov_does_not_multiply_fiducial_dictionary(tmp_path, monkeypatch):
    """The covariance must get past its real fiducial-C_ell/pixel-window step.

    Fiducial theory returns a dictionary keyed by tomographic pair, so the
    producer must apply the HEALPix window to each spectrum, not the dictionary.
    The catalogue loader raises a sentinel: reaching it is correct by
    construction, because theory must finish before data loading starts.
    Synthetic n(z) and nside=16 keep this small; no theory result is patched.
    """
    mod = pytest.importorskip(
        "sp_validation.cosmo_val.pseudo_cl",
        reason="tomography branch reorganized the pseudo-Cl covariance module",
    )
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

    class Tiny(mod.PseudoClMixin):
        pass

    stub = Tiny()
    stub.__dict__.update(
        nside=16,
        cell_method="map",
        force_run=True,
        versions=["v"],
        cc={"v": {"shear": {"redshift_path": str(zpath)}}},
        cosmo=ccl.CosmologyVanillaLCDM(),
        noise_bias_method="analytic",
        _output_path=lambda name: str(tmp_path / name),
        _output_path_pseudo_cl_cov=lambda *args, **kwargs: str(tmp_path / "cov.fits"),
        print_start=noop,
        print_magenta=noop,
        print_cyan=noop,
        print_done=noop,
        get_namaster_bin=noop,
    )
    with pytest.raises(_ReachedCatalogue):
        stub.calculate_pseudo_cl_inka_cov(compute_tomography=False)


def test_generate_pseudo_cl_cov_does_not_call_removed_methods():
    """Every cv method called by the covariance producer must exist.

    Existence is required before Python can produce any covariance.
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
    """The workflow's plot call must bind to its actual plotting API.

    The plot-only producer ingests saved spectra and covariances, then calls
    plot_pseudo_cl_spectrum once per polarization rather than a cv method.
    """
    from sp_validation.cosmo_val.pseudo_cl import plot_pseudo_cl_spectrum

    script = _scripts() / "cv_plot_pseudo_cl.py"
    sig = inspect.signature(plot_pseudo_cl_spectrum)
    calls = [
        node
        for node in ast.walk(ast.parse(script.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "plot_pseudo_cl_spectrum"
    ]
    assert calls, "cv_plot_pseudo_cl.py must call plot_pseudo_cl_spectrum"
    for call in calls:
        sig.bind(
            *([None] * len(call.args)),
            **{key.arg: None for key in call.keywords},
        )
