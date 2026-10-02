"""Harmonic CosmoSIS export must carry the bandpower window to 2pt_like."""

import importlib
import os
from pathlib import Path

import numpy as np
import pytest

from sp_validation import sacc_io as sio

TWOPT_DIR = (
    Path(os.environ.get("CSL_DIR", "/opt/cosmosis-standard-library"))
    / "likelihood"
    / "2pt"
)


@pytest.mark.xfail(
    strict=True,
    reason="#385: harmonic export drops the bandpower window",
)
def test_cell_ee_prediction_is_window_averaged_not_point_sampled_at_ell_eff(
    tmp_path, monkeypatch
):
    """The harmonic likelihood must predict W_b @ C, not C(ell_eff).

    The SACC stores each pseudo-Cl bandpower as C_b = sum_l W_b(l) C(l),
    with its NaMaster window. The CosmoSIS 2pt FITS built from it by
    ``sacc_to_twopoint_fits`` is what the harmonic ``2pt_like`` fits, so
    the theory prediction that likelihood forms for a band must equal W_b @ C.
    Fixture: one band with W(ell=1)=W(ell=3)=1/2, ell_eff=2, theory C=ell^2.
    By hand W@C = (1+9)/2 = 5 (and the stored data value is 5), whereas
    sampling theory at ell_eff gives 4. A point-sampled export therefore
    yields chi2=1 for a perfect model; a window-aware export yields 5 and chi2=0.
    """
    if not (TWOPT_DIR / "2pt_like.py").is_file():
        pytest.skip("cosmosis-standard-library 2pt likelihood not installed")
    # Only make the installed consumer importable; do not alter its prediction.
    monkeypatch.syspath_prepend(str(TWOPT_DIR))
    twopoint = pytest.importorskip("twopoint")
    datablock = pytest.importorskip("cosmosis.datablock")
    # importorskip parses the module name as Python; this name starts with '2'.
    like_module = importlib.import_module("2pt_like")

    nz = {0: (np.array([0.1, 0.3, 0.5]), np.array([1.0, 2.0, 1.0]))}
    s = sio.new_sacc(nz)
    sio.add_xi(s, (0, 0), [1.0, 2.0], [0.0, 0.0], [0.0, 0.0], grid="reporting")
    sio.add_pseudo_cl(
        s,
        (0, 0),
        [2.0],
        [5.0],
        window_ells=[1.0, 3.0],
        window_weights=[[0.5], [0.5]],
    )
    s.add_covariance(np.ones(len(s.mean)))
    hdus = sio.sacc_to_twopoint_fits(s, str(tmp_path / "harmonic.fits"))
    spectrum = twopoint.SpectrumMeasurement.from_fits(hdus["CELL_EE"])

    ell = np.array([1.0, 2.0, 3.0])
    theory = ell**2
    expected = np.array([0.5 * theory[0] + 0.5 * theory[2]])  # W@C = 5

    block = datablock.DataBlock()
    block["shear_cl", "save_name"] = ""
    block["shear_cl", "is_auto"] = True
    block["shear_cl", "nbin"] = 1
    block["shear_cl", "sep_name"] = "ell"
    block["shear_cl", "ell"] = ell
    block["shear_cl", "bin_1_1"] = theory
    like = object.__new__(like_module.TwoPointLikelihood)
    like.theory_splines = {}
    predicted = np.asarray(like.extract_spectrum_prediction(block, spectrum, "")[0])

    np.testing.assert_allclose(
        predicted,
        expected,
        rtol=1e-12,
        atol=0,
        err_msg=(
            f"2pt_like prediction {predicted.tolist()} != W@C {expected.tolist()} "
            f"(WINDOWS={hdus['CELL_EE'].header['WINDOWS']}, "
            f"data={spectrum.value.tolist()}, "
            f"chi2 unit var={float(np.sum((spectrum.value - predicted) ** 2))})"
        ),
    )
