"""A failed PSF-leakage fit must not become a numerical shear correction."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from sp_validation import calibration, catalog, catalog_builders


def _leakage_statements():
    """Execute the real script's leakage block, not a copy of its arithmetic."""
    root = Path(__file__).resolve().parents[4]
    # When testing another checkout via PYTHONPATH, use its script as well.
    code_root = Path(catalog_builders.__file__).resolve().parents[2]
    if code_root != root:
        root = code_root
    path = root / "scripts/calibration/calibrate_comprehensive_cat.py"
    assert path.is_file(), "The catalogue calibration script must exist"
    tree = ast.parse(path.read_text(), filename=str(path))
    wanted = {
        "alpha_1",
        "alpha_2",
        "e1_leak_corrected",
        "e2_leak_corrected",
        "add_cols_data['e1_leak_corrected']",
        "add_cols_data['e2_leak_corrected']",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            ast.unparse(target) in wanted
            for assignment in node.targets
            for target in (
                assignment.elts if isinstance(assignment, ast.Tuple) else [assignment]
            )
        )
    ]
    assert len(nodes) == 5, "Calibration script changed; inspect the leakage block"
    return compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec")


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#382: failed fit publishes alpha=-99 shears",
)
def test_psf_leakage_fit_failure_aborts_before_writing_corrected_shears(tmp_path):
    """A failed PSF-leakage fit must abort rather than publish corrected shears.

    Thirty objects have distinct sizes but identical SNR, so the requested SNR
    quantiles have duplicate edges and no leakage coefficient can be estimated.
    An explicit ValueError/RuntimeError before writing is the correct outcome,
    not a numerical coefficient of -99. With g=(-0.025,-0.03) and
    e_PSF=(0.015,-0.02), that sentinel produces (1.46,-2.01) by subtraction;
    these numbers are corruption, not a leakage correction.
    The real script's assignments and package writer exercise publication,
    while unrelated metacal/configuration work is omitted.
    """
    n = 30
    g = np.array([np.full(n, -0.025), np.full(n, -0.03)])
    dat = {"e1_PSF": np.full(n, 0.015), "e2_PSF": np.full(n, -0.02)}
    cat_gal = {
        "snr": np.full(n, 30.0),
        "w_des": np.ones(n),
        "NGMIX_T_NOSHEAR": np.linspace(1.0, 3.0, n),
        "NGMIX_T_PSF_RECONV_NOSHEAR": np.ones(n),
    }
    # Establish the genuine upstream failure without injecting an exception.
    fit_input = dict(cat_gal, e1=g[0], e2=g[1], **dat)
    with pytest.raises(ValueError, match="Bin edges must be unique"):
        calibration.get_alpha_leakage_per_object(fit_input, 20, "des")

    namespace = {
        "sp_joint": catalog_builders,
        "cat_gal": cat_gal,
        "g_corr_mc": g,
        "dat": dat,
        "mask_combined": SimpleNamespace(_mask=np.ones(n, dtype=bool)),
        "mask_metacal": np.ones(n, dtype=bool),
        "add_cols_data": {},
    }
    output = tmp_path / "cut.fits"
    try:
        exec(_leakage_statements(), namespace)
    except (ValueError, RuntimeError) as error:
        assert str(error), "Fit failure must have an explicit diagnostic"
        assert not output.exists(), "Failed fit must not publish a catalogue"
        return

    catalog.write_shape_catalog(
        str(output),
        np.linspace(0, 1, n),
        np.zeros(n),
        cat_gal["w_des"],
        g=g,
        w_type="des",
        add_cols=namespace["add_cols_data"],
    )
    with fits.open(output) as hdus:
        e1 = float(hdus[1].data["e1_leak_corrected"][0])
        e2 = float(hdus[1].data["e2_leak_corrected"][0])
    assert not output.exists(), (
        "PSF leakage fit failed (Bin edges must be unique), but "
        f"alpha=({namespace['alpha_1']}, {namespace['alpha_2']}) wrote "
        f"e1_leak_corrected={e1:.6g}, e2_leak_corrected={e2:.6g}; "
        "expected an explicit error and no corrected catalogue"
    )
