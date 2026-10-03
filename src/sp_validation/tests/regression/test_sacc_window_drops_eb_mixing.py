"""SACC windows must preserve the masked-sky E-to-B bandpower response."""

import healpy as hp
import numpy as np
import pymaster as nmt
import pytest

from sp_validation import sacc_io
from sp_validation.cosmo_val.sacc_writers import pseudo_cl_to_sacc
from sp_validation.pseudo_cl import make_namaster_bin

NSIDE, LMAX = 16, 32


def _toy_workspace():
    theta, phi = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)))
    mask = ((theta < 1.3) & (theta > 0.4) & (phi > 0.2) & (phi < 2)).astype(float)
    bins = make_namaster_bin(8, LMAX, LMAX - 1, "powspace", n_ell_bins=4)
    field = nmt.NmtField(mask, [0 * mask, 0 * mask], lmax=LMAX - 1)
    return bins, nmt.NmtWorkspace.from_fields(field, field, bins)


@pytest.mark.xfail(
    strict=True,
    reason="#381: SACC window drops BB-from-EE mixing",
)
def test_sacc_pseudo_cl_window_does_not_drop_e_to_b_leakage():
    """The SACC pseudo-Cl part must forward-model decoupled BB of a pure-E sky.

    On a masked sky the binned mode-coupling inversion is not exact: decoupled
    BB is W[BB<-EE] @ C_EE, not zero. NaMaster's couple/decouple gives that
    expectation independently of sp_validation. A SACC consumer predicts each
    datum as window.weight[:, window_ind] @ C_X(ell) for its own spectrum X.
    If the stored window keeps only EE<-EE, predicted BB is identically zero
    and a BB null test uses the wrong expectation. The tolerance is one part
    per million of the largest EE bandpower, well below this toy sky's leakage.
    """
    bins, wsp = _toy_workspace()
    cl = np.zeros((4, LMAX))
    cl[0, 2:] = 1.0 / (np.arange(2, LMAX) + 1.0) ** 2
    exact = wsp.decouple_cell(wsp.couple_cell(cl))  # EE, EB, BE, BB

    sacc = pseudo_cl_to_sacc(
        {0: (np.array([0.0, 1.0]), np.ones(2))},
        {},
        bins.get_effective_ells(),
        exact,
        wsp,
    )
    idx = sacc.indices(sacc_io.CL_BB, ("source_0", "source_0"))
    win = sacc.get_bandpower_windows(idx)
    ells = win.values.astype(int)
    predicted_bb = win.weight.T @ cl[3, ells]

    matches = np.allclose(
        predicted_bb, exact[3], rtol=0, atol=1e-6 * abs(exact[0]).max()
    )
    assert matches, (
        f"SACC-window BB prediction for a pure-E sky {predicted_bb.tolist()} "
        f"!= NaMaster decoupled BB {exact[3].tolist()} "
        f"(max|W_BB<-EE|={abs(wsp.get_bandpower_windows()[3, :, 0]).max():.3g})"
    )
