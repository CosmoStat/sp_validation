"""The CLOE configuration the theory runs on: cloelib's grids, CAMB background and engines.

The theory is CLOE (cloelib) with one massive neutrino species carrying the summed mass
m_nu, NLA intrinsic alignments and constant per-bin linear galaxy bias. Two P(k) engines are
supported, each a cloelib engine:

- ``cosmopower_mead2020_feedback``: the CosmoPower-JAX HMcode-2020 + baryonic-feedback
  emulator (cloelib's ``CosmoPowerJAXLCDMPerturbations``), with ``HMCode_logT_AGN``; it is
  trained to z = 5, above which P(k) continues by linear growth;
- ``camb_mead2020``: CAMB's dark-matter-only HMcode-2020 (cloelib's
  ``CAMBNonLinearPerturbations``), for mocks generated with it.

:class:`Grid` holds what cloelib's Limber projection is evaluated on: a uniform redshift grid
from ``Z_STEP`` to ``Z_MARGIN`` past the top of the delivered n(z), the P(k) redshift nodes
(``N_PK_REDSHIFTS`` uniform from 0, plus z = 5 when the emulator stops there) and wavenumbers
(``K_RANGE``, ``N_K`` log-spaced, Mpc^-1).
"""
import math

import numpy as np

ENGINES = {"cosmopower_mead2020_feedback": ("cosmopower_jax", "mead2020_feedback"),
           "camb_mead2020": ("camb", "mead2020")}
#: The highest redshift each engine is trained on (cloelib's CosmoPower-JAX bounds).
Z_TRAINED = {"cosmopower_jax": 5.0, "camb": np.inf}
Z_STEP = 0.002
Z_MARGIN = 0.2
N_PK_REDSHIFTS = 100
K_RANGE = (1e-4, 50.0)
N_K = 400

#: CCL's physical constants (ccl_constants.h), for the NLA normalisation below.
GNEWT = 6.6743e-11
MPC_TO_METER = 3.085677581491367e22
SOLAR_MASS = 1.988409871e30
#: Critical density today in h^2 Msun Mpc^-3, in CCL's operation order (bit-identical to it).
RHO_CRITICAL = (3 * 100 * 100) / (8 * math.pi * GNEWT) * (1000 * 1000 * MPC_TO_METER / SOLAR_MASS)
#: The NLA normalisation C1 rho_crit, with C1 = 5e-14 h^-2 Msun^-1 Mpc^3 (CCL's and CLOE's).
C1_RHO_CRIT = 5e-14 * RHO_CRITICAL


def camb_background(cosmology):
    """cloelib's CAMB background of a cosmology (``Omega_c, Omega_b, h, n_s, A_s, m_nu``):
    flat LCDM with one massive neutrino species of mass m_nu."""
    from cloelib.cosmology.camb_cosmology import CAMBBackground
    m_nu = float(cosmology["m_nu"])
    return CAMBBackground(H0=100 * cosmology["h"], Omega_b0=cosmology["Omega_b"],
                          Omega_cdm0=cosmology["Omega_c"], Omega_k0=0.0, As=cosmology["A_s"],
                          ns=cosmology["n_s"], mnu=m_nu, w0=-1.0, wa=0.0, gamma_MG=0.545,
                          N_mnu=int(m_nu > 0))


class Grid:
    """cloelib's evaluation grids for n(z) reaching ``z_top`` under ``engine``."""

    def __init__(self, z_top, engine):
        self.engine = engine
        self.pk, self.nonlinear_model = ENGINES[engine]
        self.z = Z_STEP * np.arange(1, int(np.ceil((z_top + Z_MARGIN) / Z_STEP)) + 1)
        z_pk = np.linspace(0.0, self.z[-1], N_PK_REDSHIFTS)
        trained = Z_TRAINED[self.pk]
        if trained < z_pk[-1]:
            near = np.flatnonzero(np.abs(z_pk - trained) <= 1e-3 * np.diff(z_pk).min())
            if near.size:
                z_pk[near[0]] = trained
            else:
                z_pk = np.unique(np.r_[z_pk, trained])
        self.z_pk = z_pk
        self.ks = np.geomspace(*K_RANGE, N_K)
        self.c1_rho_crit = C1_RHO_CRIT

    def camb_power(self, cosmology):
        """CAMB's HMcode-2020 (dark matter only) P(k) state at ``cosmology``, through cloelib."""
        from cloelib.cosmology.camb_cosmology import CAMBNonLinearPerturbations
        return CAMBNonLinearPerturbations(camb_background(cosmology), None, self.z_pk,
                                          nonlinear_model="mead2020")
