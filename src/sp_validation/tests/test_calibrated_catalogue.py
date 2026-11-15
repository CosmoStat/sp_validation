"""A calibrated catalogue reaches every reader once, with its own weights."""

import numpy as np
import pytest
from astropy.io import fits

from sp_validation import io, response
from sp_validation.tests.test_response import CUTS, synthetic

METACAL = dict(
    weight_scheme="des",
    gal_snr_min=CUTS.snr_min,
    gal_snr_max=CUTS.snr_max,
    gal_rel_size_min=CUTS.size_min,
    gal_rel_size_max=CUTS.size_max,
    gal_size_corr_ell=CUTS.size_corr_ell,
    des_num_bins=5,
    additive_correction=False,
)


def calibrated_catalogue(tmp_path):
    """A cut catalogue as ``calibrate_comprehensive_cat.py`` writes it."""
    cal = response.calibrate_catalogue(synthetic(2000), METACAL)
    header = fits.Header(dict(WGHTSCH="des", RAPPLIED=True, MEANSUB=False))
    cols = [
        fits.Column(name="e1", array=cal["e"][0], format="D"),
        fits.Column(name="e2", array=cal["e"][1], format="D"),
        fits.Column(name="w_des", array=cal["w"], format="D"),
    ]
    path = tmp_path / "cut.fits"
    fits.HDUList(
        [fits.PrimaryHDU(header=header), fits.BinTableHDU.from_columns(cols)]
    ).writeto(path)
    return path, cal["e"]


def test_scalar_response_entry_is_calibrated_as_one_sample():
    from sp_validation.cosmo_val.core import CosmologyValidation

    obj = object.__new__(CosmologyValidation)
    obj.cc = {"external": {"shear": dict(e1_col="e1", e2_col="e2", w_col="w", R=0.8)}}
    e = np.array([[0.12, -0.04, 0.03], [0.08, 0.16, -0.02]])
    w = np.array([1.0, 2.0, 0.5])
    g = obj._calibrated_g("external", dict(e1=e[0], e2=e[1], w=w))
    expected = e / 0.8 - ((e / 0.8) @ w / w.sum())[:, None]
    np.testing.assert_allclose(g, expected)


@pytest.mark.parametrize("adapter", ["xi", "pseudo_cl", "rho_tau", "leakage"])
def test_calibrated_matrix_shear_passes_reader_adapters_once(
    tmp_path, monkeypatch, adapter
):
    from types import SimpleNamespace

    from sp_validation.cosmo_val import core
    from sp_validation.rho_tau import _CatalogueLoader

    path, g = calibrated_catalogue(tmp_path)
    entry = dict(path=path, w_col="w_des", e1_col="e1", e2_col="e2", R=1.0)

    def read(block):
        if adapter == "rho_tau":
            loader = _CatalogueLoader({"shear": block}, {})
            filename = loader("shear")  # ordinary FITS is passed through
            with fits.open(filename) as hdus:
                return np.array([hdus[1].data["e1"], hdus[1].data["e2"]])
        if adapter == "leakage":
            obj = core._LeakageObject.__new__(core._LeakageObject)
            obj.entries = {"shear": block}
            obj.read_data()
            return np.array([obj._dat["e1"], obj._dat["e2"]])
        if adapter == "xi":
            monkeypatch.setattr(core.leakage, "cut_data", lambda data, *args: data)
            obj = core._LeakageScale.__new__(core._LeakageScale)
            obj.entries = {"shear": block}
            obj._params = {"cut": None, "verbose": False}
            obj.read_data(shear=True, psf=False)
            return np.array([obj.dat_shear["e1"], obj.dat_shear["e2"]])
        obj = SimpleNamespace(
            cc={"SP": {"shear": block}},
            results={"SP": SimpleNamespace(dat_shear=io.open_entry(block))},
        )
        return np.array(
            core.CosmologyValidation._read_shear_cols(obj, "SP", "e1_col", "e2_col")
        )

    np.testing.assert_allclose(read(entry), g)
    for wrong in (
        dict(entry, w_col="w_iv"),
        dict(entry, R=0.8),
        dict(entry, R11="R_g11"),
    ):
        with pytest.raises(ValueError):
            read(wrong)


def test_reader_refuses_other_weights_or_a_second_response(tmp_path):
    path, g = calibrated_catalogue(tmp_path)
    for entry in (
        {"path": path, "w_col": "w_iv", "R": 1},
        {"path": path, "w_col": "w_des", "R": 0.8},
    ):
        with pytest.raises(ValueError):
            io.open_entry(entry)
    assert len(io.open_entry({"path": path, "w_col": "w_des", "R": 1})) == len(g[0])
