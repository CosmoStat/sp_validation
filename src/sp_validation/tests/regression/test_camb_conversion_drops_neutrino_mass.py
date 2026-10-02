"""CAMB fiducial C_ell must respond to the input cosmology's neutrino mass."""

import numpy as np
import pyccl as ccl
import pytest

sp_pcl = pytest.importorskip(
    "sp_validation.pseudo_cl", reason="CAMB fiducial API is on the tomography branch"
)


def _cosmo(m_nu):
    return ccl.Cosmology(
        Omega_c=0.26, Omega_b=0.049, h=0.68, n_s=0.965, A_s=2.1e-9, m_nu=m_nu
    )


@pytest.mark.slow
def test_camb_fiducial_cl_does_not_drop_neutrino_mass():
    """Protect tomography's CAMB covariance theory from dropping neutrino mass.

    At fixed A_s, Omega_c, Omega_b and h, raising sum m_nu from 0.06 to 0.2 eV
    adds massive-neutrino density and suppresses small-scale growth. The
    independent reference is CCL's response to these same cosmologies. CAMB's
    mean C(0.2)/C(0.06) over ell 100-1000 must track CCL's within 3%: different
    nonlinear models and neutrino splits differ by ~1.5% in this ratio, whereas
    dropping m_nu leaves it at exactly 1. Develop lacks this tomography API.
    """
    if not hasattr(sp_pcl, "get_fiducial_cl"):
        pytest.skip("get_fiducial_cl is only on the tomography branch")

    z = np.linspace(0.01, 2.5, 200)
    nz = z**2 * np.exp(-((z / 0.5) ** 1.5))
    nz = (nz / np.trapezoid(nz, z))[:, None]
    lmax = 1000
    ell = np.arange(1, lmax + 1)
    sel = (ell >= 100) & (ell <= 1000)

    lo, hi = _cosmo(0.06), _cosmo(0.2)
    camb_lo = np.asarray(sp_pcl.get_fiducial_cl(z, nz, lmax, lo)["W1xW1"])
    camb_hi = np.asarray(sp_pcl.get_fiducial_cl(z, nz, lmax, hi)["W1xW1"])
    tr_lo = ccl.WeakLensingTracer(lo, dndz=(z, nz[:, 0]))
    tr_hi = ccl.WeakLensingTracer(hi, dndz=(z, nz[:, 0]))
    ccl_lo = ccl.angular_cl(lo, tr_lo, tr_lo, ell)
    ccl_hi = ccl.angular_cl(hi, tr_hi, tr_hi, ell)

    r_camb = (camb_hi[sel] / camb_lo[sel]).mean()
    r_ccl = (ccl_hi[sel] / ccl_lo[sel]).mean()
    assert abs(r_camb - r_ccl) < 0.03, (
        f"CAMB C(mnu=0.2)/C(mnu=0.06) = {r_camb:.6f} but CCL gives {r_ccl:.6f} "
        "(ell 100-1000 mean)"
    )
