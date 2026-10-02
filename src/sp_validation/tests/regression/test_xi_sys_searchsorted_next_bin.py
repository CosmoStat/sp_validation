"""PSF-leakage xi_sys must be added to the xi theory at the same angular bin.

Code under test: the CosmoSIS standard library fork installed in the
sp_validation image (``shear/xi_sys/xi_sys_psf.py`` and
``likelihood/2pt/2pt_like_xi_sys.py``), fed through the launched checkout's
``cosmo_inference/scripts/cosmosis_fitting.py`` FITS converters exactly as the
committed PSF pipeline (``cosmosis_pipeline_A_psf.ini``) wires them.
"""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

import sp_validation

REPO = Path(__file__).resolve().parents[4]
# Cross-checkout runs must exercise the package selected by PYTHONPATH.
ACTIVE_REPO = Path(sp_validation.__file__).resolve().parents[2]
if ACTIVE_REPO != REPO:
    REPO = ACTIVE_REPO

CSL = Path(os.environ.get("CSL_DIR", "/opt/cosmosis-standard-library"))
LIKE_DIR = CSL / "likelihood" / "2pt"
PSF_MODULE = CSL / "shear" / "xi_sys" / "xi_sys_psf.py"
FITTING = REPO / "cosmo_inference" / "scripts" / "cosmosis_fitting.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules(monkeypatch):
    for path in (LIKE_DIR / "2pt_like_xi_sys.py", PSF_MODULE, FITTING):
        if not path.exists():
            pytest.skip(f"missing {path}")
    monkeypatch.syspath_prepend(str(LIKE_DIR))
    twopoint = pytest.importorskip("twopoint")
    pytest.importorskip("cosmosis.datablock")

    return SimpleNamespace(
        fitting=_load("cosmosis_fitting_under_test", FITTING),
        psf=_load("xi_sys_psf_under_test", PSF_MODULE),
        like=_load("two_pt_like_xi_sys_under_test", LIKE_DIR / "2pt_like_xi_sys.py"),
        twopoint=twopoint,
    )


@pytest.mark.xfail(
    strict=True,
    reason="#391: xi_sys lookup selects the next angular bin",
)
def test_xi_sys_added_at_matching_angular_bin_not_next_bin(modules, tmp_path):
    """The xi_sys PSF-leakage term for angular bin i must be added to bin i.

    The fiducial v1.4.6.3 xi grid (20 log bins, 1-250 arcmin) is written to a
    TwoPoint HDU and rho statistics on the *same* theta are forced onto it via
    ``rho_to_fits(theta=xi_theta)``, as ``cosmosis_fitting.py`` does.  rho_0_p
    is set to ``bin_index + 1`` with alpha=1, beta=0, so xi_sys_plus in bin i is
    exactly ``i + 1`` by construction, and with zero cosmological theory the
    likelihood's theory vector over the 12-83 arcmin cut must equal
    ``selected_bin_indices + 1``.  The likelihood instead matches angles with
    ``np.searchsorted`` (left insertion) between ``np.radians(theta / 60)`` and
    astropy's arcmin->rad conversion of the same numbers; these differ by one
    ulp for some bins, and when the data angle is the larger one the lookup
    returns bin i+1, so that bin receives its neighbour's xi_sys.
    """
    m = modules
    nbins = 20
    theta = np.exp((np.arange(nbins) + 0.5) * np.log(250.0) / nbins)  # arcmin
    selection = (theta >= 12.0) & (theta <= 83.0)

    # rho statistics file as cosmo_val writes it, then forced onto xi theta.
    names = ["theta"] + [f"rho_{k}_{s}" for k in (0, 1, 2) for s in ("p", "m")]
    cols = {n: np.zeros(nbins) for n in names}
    cols["theta"] = theta * 1.001  # rho's own meanr, overridden below
    cols["rho_0_p"] = np.arange(nbins) + 1.0
    cols["rho_0_m"] = -(np.arange(nbins) + 1.0)
    rho_file = tmp_path / "rho.fits"
    fits.BinTableHDU.from_columns(
        [fits.Column(name=n, format="D", array=cols[n]) for n in names]
    ).writeto(rho_file)
    rho_hdu = m.fitting.rho_to_fits(str(rho_file), theta=theta)

    # PSF module: alpha=1, beta=0 -> xi_sys_p[i] == i + 1 exactly.
    from cosmosis.datablock import DataBlock

    block = DataBlock()
    block["psf_leakage_parameters", "alpha"] = 1.0
    block["psf_leakage_parameters", "beta"] = 0.0
    m.psf.execute(block, rho_hdu.data)

    obj = m.like.TwoPointLikelihood.__new__(m.like.TwoPointLikelihood)
    obj.suffixes = [""]
    obj.add_xi_sys = True
    obj.moped = ""
    spectra = []
    for name, quant in (("XI_PLUS", "G+R"), ("XI_MINUS", "G-R")):
        hdu = m.fitting._create_2pt_hdu(np.zeros(nbins), theta, name, quant, quant)
        spec = m.twopoint.SpectrumMeasurement.from_fits(hdu)
        spec.convert_angular_units("rad")
        spec.apply_mask(selection)
        spectra.append(spec)
    obj.two_point_data = SimpleNamespace(spectra=spectra)
    # Zero cosmological theory isolates the xi_sys addition.
    obj.extract_spectrum_prediction = lambda b, s, suffix: (
        np.zeros(len(s.angle)),
        s.angle,
        s.bin1,
        s.bin2,
    )

    actual = obj.extract_theory_points(block)
    idx = np.nonzero(selection)[0]
    expected = np.concatenate([idx + 1.0, -(idx + 1.0)])

    got_bins = np.abs(actual[: idx.size]) - 1
    wrong = idx[got_bins != idx].tolist()
    assert np.array_equal(actual, expected), (
        f"xi_sys taken from the wrong angular bin: expected bins {idx.tolist()}, "
        f"got bins {got_bins.astype(int).tolist()} (wrong at data bins {wrong})"
    )
