"""What a blind shifts, and what the catalogue config may declare.

The shifts run the default theory, :func:`sp_validation.blinding.shear`, under
a blind whose hidden point raises S8 above the fiducial.
"""

import json

import numpy as np
import pytest

from sp_validation import b_modes
from sp_validation import blinding as bd
from sp_validation import sacc_io as sio


@pytest.fixture(scope="module")
def up(tmp_path_factory):
    path = tmp_path_factory.mktemp("blinds") / "up.blind.json"
    path.write_text(
        json.dumps(
            {"seed": "s", "envelope": {"S8": [0.05, 0.06]}, "fiducial": bd.fiducial()}
        )
    )
    return bd.Blind("up", path)


def _sacc():
    z = np.linspace(0.0, 3.0, 200)
    return sio.new_sacc({0: (z, np.exp(-0.5 * ((z - 0.7) / 0.2) ** 2))})


@pytest.mark.slow
def test_a_blind_shifts_xi_and_cl_ee_and_leaves_bb_eb(up):
    s = _sacc()
    theta = np.geomspace(2.0, 200.0, 6)
    sio.add_xi(s, (0, 0), theta, 0 * theta, 0 * theta, grid="reporting")
    ell = np.array([30.0, 80.0, 150.0, 280.0, 450.0])
    w_ell = np.arange(2, 501).astype(float)
    window = np.exp(-0.5 * ((w_ell[:, None] - ell) / 40.0) ** 2)
    zeros = 0 * ell
    sio.add_pseudo_cl(
        s,
        (0, 0),
        ell,
        zeros,
        zeros,
        zeros,
        window_ells=w_ell,
        window_weights=window / window.sum(axis=0),
    )

    sealed = sio.seal(s, up)
    delta = sealed.mean - s.mean
    for data_type in (sio.XI_PLUS, sio.XI_MINUS, sio.CL_EE):
        assert np.all(delta[s.indices(data_type)] > 0), data_type
    for data_type in (sio.CL_BB, sio.CL_EB):
        assert np.all(delta[s.indices(data_type)] == 0), data_type
    assert sio.stamp(sealed) == "up"


@pytest.mark.slow
def test_pure_eb_under_a_blind_shifts_e_and_not_b(up):
    """The pure-E/B operator on the integration-grid ξ± is linear, so a blind
    moves the modes by its response to the shift: E by several σ, B by under
    a tenth of σ (σ: UNIONS shape noise through the operator)."""
    edges_int = np.geomspace(0.08, 300.0, 1001)
    theta_int = np.sqrt(edges_int[:-1] * edges_int[1:])
    s = _sacc()
    sio.add_xi(s, (0, 0), theta_int, 0 * theta_int, 0 * theta_int, grid="integration")
    shifted = sio.xi_correlation(sio.seal(s, up))
    delta = np.r_[shifted.xip, shifted.xim]

    annuli = np.diff(edges_int**2)  # ∝ pair counts on a uniform field
    operator, _, _ = b_modes.pure_eb_operator(
        annuli, edges_int, np.geomspace(1.0, 250.0, 21)
    )
    # Var ξ± = σ_e⁴ / (2 N_pairs), SP_v1.4.6.3's area, n_eff and σ_e.
    n_eff, area, sigma_e = 4.96, 2894.0 * 3600, 0.378
    variance = np.tile(sigma_e**4 / (n_eff**2 * area * np.pi * annuli), 2)
    sigma = np.sqrt(np.einsum("ij,j,ij->i", operator, variance, operator))
    response = operator @ delta / sigma

    n = len(response) // 6  # blocks in _EB_KEYS order: E+, E−, B+, B−, amb
    assert np.max(np.abs(response[: 2 * n])) > 1
    assert np.max(np.abs(response[2 * n : 4 * n])) < 0.1


def test_seal_refuses_a_sacc_already_stamped():
    sealed = sio.seal(_sacc(), bd.NONE)
    with pytest.raises(ValueError, match="already carries a blind stamp"):
        sio.seal(sealed, bd.NONE)


def test_entries_reading_one_file_declare_one_blind():
    catalogues = {
        "paths": {"blinds": "/blinds"},
        "Y3": {"blind": "y3", "subdir": "/d", "shear": {"path": "y3.fits"}},
        "TWIN": {"blind": "none", "subdir": "/d", "shear": {"path": "twin.fits"}},
    }
    assert bd.blind_of(catalogues, "TWIN") == "none"
    catalogues["TWIN"]["shear"]["path"] = "/d/sub/../y3.fits"
    with pytest.raises(ValueError, match="different blinds"):
        bd.blind_of(catalogues, "TWIN")
