"""The ell axis prepended to unbinned mock C_ell arrays must label index i as ell=i."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

hp = pytest.importorskip("healpy")
pytest.importorskip("pymaster")

from sp_validation import glass_mock as gm  # noqa: E402

NSIDE = 8
ELL_IN = 4


def _single_mode_catalogue():
    """One galaxy per pixel carrying a pure-E spin-2 field with only (ell=4, m=0)."""
    lmax = 2 * NSIDE
    ra, dec = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)), lonlat=True)
    alm = np.zeros(hp.Alm.getsize(lmax), complex)
    alm[hp.Alm.getidx(lmax, ELL_IN, 0)] = 1.0
    e1, e2 = hp.alm2map_spin([alm, 0 * alm], NSIDE, 2, lmax=lmax)
    # The estimators flip e2 (factor = -1) to undo the catalogue convention.
    return {"ra": ra.copy(), "dec": dec.copy(), "e1": e1, "e2": -e2}


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#389: coupled C_ell multipole labels are shifted by +1",
)
@pytest.mark.parametrize(
    "estimator", ["compute_two_point_cl", "compute_two_point_cl_map"]
)
def test_coupled_cl_ell_axis_labels_input_mode_at_its_true_ell(estimator):
    """Protects the ell axis written as row 0 of the saved coupled pseudo-C_ell.

    ``nmt.compute_coupled_cell`` returns one value per multipole starting at
    ell = 0, so the prepended axis must be ``0..len-1``. A full-sky catalogue
    whose shear field contains only the pure-E (ell=4, m=0) harmonic has its
    coupled EE power peaked at array index 4 by construction, so the saved ell
    of the peak must read 4. Labelling the axis ``1..lmax`` shifts every value
    to ell+1 in ``cl_coupled_*glass_mock_*.npy``.
    """
    fun = getattr(gm, estimator, None)
    if fun is None:
        pytest.skip(f"{estimator} requires the tomography branch")
    coupled, _decoupled = fun(_single_mode_catalogue(), nside=NSIDE, lmin=2, n_bins=2)
    peak = int(np.argmax(coupled[1]))  # row 1 = EE
    assert peak == ELL_IN  # the field itself is right; only the label is in question
    assert coupled[0, peak] == ELL_IN, (
        f"{estimator}: injected ell={ELL_IN} mode saved at ell={coupled[0, peak]:g}; "
        f"saved ell axis runs {coupled[0, 0]:g}..{coupled[0, -1]:g} "
        f"for {coupled.shape[1]} multipoles (expected 0..{coupled.shape[1] - 1})"
    )


def _sky_script():
    root = Path(__file__).resolve().parents[4]
    # Keep this test file while cross-checking the imported checkout's code.
    code_root = Path(gm.__file__).resolve().parents[2]
    if code_root != root:
        root = code_root
    root = root / "scripts" / "glass_mock"
    for name in ("make_glass_sim.py", "make_unions_glass_sim.py"):
        if (root / name).exists():
            return root / name
    pytest.skip("GLASS driver missing (including tomography branch driver)")


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#389: CAMB source C_ell labels are shifted by +1",
)
def test_camb_source_cl_ell_column_starts_at_zero(tmp_path):
    """Protects the ``ell`` column of the CAMB source C_ell FITS (``--camb``).

    ``CAMBdata.get_source_cls_dict(lmax=L, raw_cl=True)`` returns arrays of
    length L+1 indexed ell = 0..L, with ell = 0, 1 identically zero for lensing
    windows. The saved ``ell`` column must therefore be ``0..L`` and the first
    non-zero entry must sit at ell = 2; labelling it ``1..L+1`` attributes every
    theory value to ell+1.
    """
    camb = pytest.importorskip("camb")
    pytest.importorskip("glass")
    pytest.importorskip("fitsio")
    path = _sky_script()
    spec = importlib.util.spec_from_file_location("glass_sky", path)
    mod = importlib.util.module_from_spec(spec)
    # Do not create bytecode beside the repository script.
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
    sky = mod.Sky.__new__(mod.Sky)
    cfg = replace(gm.GlassMockConfig(nside=NSIDE), nbins=1)
    sky.config = cfg
    sky.pars = gm.build_camb_params(cfg)
    sky.z = np.linspace(0.0, 0.3, 20)
    sky.bin_nz = [np.maximum(0.0, 1.0 - abs(sky.z - 0.15) / 0.15)]
    sky.prefix, sky.n_sim, sky.root = "regression", "00000", str(tmp_path)
    assert isinstance(sky.pars, camb.CAMBparams)
    data = sky.get_camb_cls(sav=False)
    first_nonzero = int(np.flatnonzero(data["0-0"])[0])
    assert first_nonzero == 2  # CAMB array index == ell
    assert data["ell"][first_nonzero] == 2, (
        f"first non-zero CAMB lensing C_ell (array index {first_nonzero}, ell=2) "
        f"saved at ell={data['ell'][first_nonzero]}; ell column runs "
        f"{data['ell'][0]}..{data['ell'][-1]} for lmax={cfg.lmax}"
    )
