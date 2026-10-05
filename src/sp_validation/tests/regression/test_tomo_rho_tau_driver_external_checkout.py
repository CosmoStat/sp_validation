"""The rho_tau_stats driver must use the checkout that launched the workflow."""

import io
import os
import runpy
import sys
import types
from pathlib import Path

import pytest

import sp_validation
import sp_validation.cosmo_val as cosmo_val_mod


class _Stop(Exception):
    """Stop before any catalogue reads or output writes."""


def test_rho_tau_driver_reads_cat_config_from_launched_checkout(monkeypatch, tmp_path):
    """Resolve the catalogue config and output inside the launched checkout.

    The constructor stub records cwd and arguments, then stops before reading
    survey data. Defaults './cat_config.yaml' and './output' resolve from cwd.
    The expected checkout contains the imported sp_validation code; ordinarily
    that is the repository containing this test, but a branch comparison can
    import a different checkout while retaining this test file.

    Develop passes explicit paths from rule params and does not chdir, so this
    test is unmarked: it guards against the tomography branch's regression of
    changing cwd to an unrelated, hard-coded checkout and using defaults.
    """
    checkout = Path(__file__).resolve().parents[4]
    code_checkout = Path(sp_validation.__file__).resolve().parents[2]
    if code_checkout != checkout:
        checkout = code_checkout
    driver = checkout / "workflow" / "scripts" / "run_rho_tau.py"
    if not driver.is_file():
        pytest.skip("rho/tau workflow driver from the tomography branch is absent")

    seen = {}

    def fake_cv(*args, **kwargs):
        seen["cwd"] = Path.cwd()
        seen["kwargs"] = kwargs
        raise _Stop

    monkeypatch.setattr(cosmo_val_mod, "CosmologyValidation", fake_cv)
    fake_snk = types.SimpleNamespace(
        params={
            "ver": "SP_v1.4.6.3",
            "min_sep": "1.0",
            "max_sep": "250.0",
            "nbins": "20",
            "npatch": "20",
            "cat_config": str(checkout / "cosmo_val" / "cat_config.yaml"),
            "output_dir": str(checkout / "cosmo_val" / "output"),
        },
        output={},
        input={},
    )
    snk_script = types.ModuleType("snakemake.script")
    snk_script.snakemake = fake_snk
    monkeypatch.setitem(sys.modules, "snakemake.script", snk_script)
    try:
        import IPython
    except ImportError:
        pytest.skip("tomography branch driver requires IPython")
    monkeypatch.setattr(IPython, "get_ipython", lambda: None)
    # Stream unbuffering is unrelated to path resolution. Avoid closing pytest's
    # captured descriptors when either driver generation calls os.fdopen.
    monkeypatch.setattr(os, "fdopen", lambda *args, **kwargs: io.StringIO())
    # Register the original streams for restoration after the driver's assignment.
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    monkeypatch.syspath_prepend(str(driver.parent))
    monkeypatch.chdir(tmp_path)

    try:
        with pytest.raises(_Stop):
            runpy.run_path(str(driver), init_globals={"snakemake": fake_snk})
    except FileNotFoundError as exc:
        # The tomography driver may chdir to a survey checkout absent on CI.
        missing = Path(exc.filename) if exc.filename else None
        if (
            missing
            and missing.is_absolute()
            and missing.parts[1] in {"n17data", "n09data", "n23data1", "home"}
        ):
            pytest.skip(
                f"tomography branch external checkout is unavailable: {missing}"
            )
        raise

    kw = seen["kwargs"]
    cat_cfg = (seen["cwd"] / kw.get("catalog_config", "./cat_config.yaml")).resolve()
    out_dir = (seen["cwd"] / (kw.get("output_dir") or "./output")).resolve()
    assert cat_cfg.is_relative_to(checkout) and out_dir.is_relative_to(checkout), (
        f"driver used catalog_config={cat_cfg}, output_dir={out_dir}, "
        f"cwd={seen['cwd']}; expected paths inside launched checkout {checkout}"
    )
