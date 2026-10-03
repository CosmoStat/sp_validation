"""The committed pol_factor turns a catalogue-convention pure-E field into pure E.

The catalogue's e2 has the opposite sign to HEALPix's U, so a pure-E sky in the
catalogue convention is (e1, e2) = (Q, -U). The committed ``pol_factor`` must make
the pseudo-Cl producer negate e2 before NaMaster sees it; without the flip, E leaks
into B at order one. The tomography branch multiplies e2 by ``pol_factor`` instead
of reading it as a boolean, so after the merge (#374) the committed ``true`` reads
as +1, a silent no-flip; port this test to the merged producer and it fails until
the config says -1.
"""

from pathlib import Path

import healpy as hp
import numpy as np
import pytest
import yaml

from sp_validation.pseudo_cl import get_pseudo_cls_map

pytestmark = [pytest.mark.fast, pytest.mark.decision("spin2_sign_convention")]

NSIDE = 32
CONFIG = (
    Path(__file__).resolve().parents[4]
    / "papers"
    / "cosmo_val"
    / "config"
    / "config.yaml"
)


def _catalogue_convention_pure_e(seed=11):
    """A band-limited, full-sky pure-E field as the catalogue stores it."""
    lmax = 3 * NSIDE - 1
    ell = np.arange(lmax + 1)
    cl_e = np.where(ell >= 2, 1.0 / (ell + 10.0) ** 2, 0.0)
    zero = np.zeros_like(cl_e)
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        alms = hp.synalm([zero, cl_e, zero, zero], lmax=lmax, new=True)
    finally:
        np.random.set_state(rng_state)
    alms[2][:] = 0.0  # B exactly zero
    _, q, u = hp.alm2map(alms, NSIDE, lmax=lmax, pol=True)
    return q - 1j * u  # catalogue e2 = -U


def _bb_over_ee(pol_factor):
    shear_map = _catalogue_convention_pure_e()
    mask = np.ones(hp.nside2npix(NSIDE))
    _, cl, _ = get_pseudo_cls_map(
        shear_map, mask, NSIDE, "linear", pol_factor=pol_factor, ell_step=8
    )
    ee, bb = cl[0], cl[3]
    return np.abs(bb).sum() / ee.sum()


def test_committed_pol_factor_keeps_pure_e_out_of_b():
    """Full sky and band-limited, so BB/EE is numerical noise (~1e-4) with the flip
    and order one without it; the 0.05 threshold sits between them.
    """
    committed = yaml.safe_load(CONFIG.read_text())["cosmo_val"]["pol_factor"]
    assert _bb_over_ee(committed) < 0.05
    assert _bb_over_ee(not committed if isinstance(committed, bool) else 1) > 0.3
