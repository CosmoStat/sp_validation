"""Firecrown cosmic-shear likelihood — the blinding box's theory engine.

:Name: blinding_likelihood.py

:Description: A minimal `firecrown <https://firecrown.readthedocs.io>`_ Gaussian
    likelihood over the tomographic real-space shear 2PCF (ξ±) written by
    :mod:`sp_validation.sacc_io`. It exists for one job: give Smokescreen
    (PR 6) a ``compute_theory_vector`` aligned element-for-element with the
    measured ξ± data vector, so the Muir et al. blinding shift
    ``d → d + t(hidden) − t(fiducial)`` can be evaluated. The sampling
    likelihood stays with CosmoSIS; only the theory-vector predict step of this
    module is ever called.

    **Scope.** Only the *coarse analysis* ξ± subset (``grid='coarse'``). A
    full analysis SACC file also carries pseudo-Cℓ, COSEBIs, pure-E/B and ρ/τ;
    this likelihood extracts the coarse ξ± block (with its aligned covariance
    sub-block) via :func:`sp_validation.sacc_io.extract` before handing it to
    firecrown, because firecrown's ``read`` models every point it is given and
    Smokescreen's ``_verify_sacc_consistency`` requires the SACC ``mean`` to
    equal the likelihood's data vector element-for-element. The fine-grid ξ±
    shift is computed in PR 6 as a direct CCL theory difference, not through
    this likelihood.

    **N tomographic bins.** Generic over any number of ``source_i`` NZ tracers:
    it builds one firecrown ``WeakLensing`` source per bin and one ``TwoPoint``
    (ξ+ and ξ−) statistic per unordered bin pair ``i ≤ j`` that the file
    actually contains. Today's single-bin data is the ``n = 1`` case.

    **Configuration surface.** :class:`TheoryConfig` is the single, explicit
    home for the fiducial cosmology and model choices (nonlinear P(k) recipe,
    intrinsic-alignment model, neutrino treatment). Its defaults mirror the
    ``cosmo_inference`` CosmoSIS fiducial (v1.4.6.3 ``_A_cell`` pipeline) so the
    two theory stacks agree by construction; PRD #241's open question about the
    fiducial choice changes these *values*, not this code. Pass a
    :class:`TheoryConfig` (or a mapping of overrides) through the firecrown
    ``build_parameters`` under the ``theory_config`` key to change it.
"""

import dataclasses
import pathlib

import numpy as np
import sacc

from . import sacc_io


# --------------------------------------------------------------------------- #
# Configuration surface — the ONE place fiducial cosmology + model choices live
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class TheoryConfig:
    """Fiducial cosmology and model configuration for the theory engine.

    Every field is a deliberate, configurable choice. The defaults reproduce
    the ``cosmo_inference`` CosmoSIS fiducial (the ``SP_v1.4.6.3_A_cell``
    pipeline + ``values_ia.ini`` central values): so the CCL theory computed
    here and the CAMB theory CosmoSIS computes agree to the level the
    CAMB↔CCL cross-check test asserts. PRD #241 §5 leaves the *named* fiducial
    choice open for the group; adopting a different one is a change to these
    values, not to any code.

    Cosmology is parametrised by ``S8`` and ``Omega_m`` — the axes the blind
    shifts (PRD §3) — and converted to CCL's native ``sigma8``/``Omega_c`` at
    build time. ``sigma8`` (not ``A_s``) is the CCL amplitude parameter; the
    ~0.5% σ8-vs-A_s convention offset against a CAMB ``A_s`` input is settled
    inside the cross-check test, not here.
    """

    # Cosmological parameters (blind axes S8, Omega_m + the rest).
    S8: float = 0.80  # values_ia.ini S_8_input central
    Omega_m: float = 0.30  # (omch2+ombh2)/h^2 with the fiducial baryons ≈ 0.29–0.31
    Omega_b: float = 0.0469  # ombh2=0.023 at h=0.7  ->  0.023/0.7^2
    h: float = 0.70  # h0 central
    n_s: float = 0.96  # n_s central
    m_nu: float = 0.06  # single massive neutrino, eV (mnu in values_ia.ini)
    w0: float = -1.0
    wa: float = 0.0

    # Nonlinear matter power spectrum: CAMB's HMCode2020 with baryonic
    # feedback, matching the CosmoSIS `[camb] halofit_version=mead2020_feedback`
    # line. CCL routes P(k) through CAMB when `matter_power_spectrum` resolves
    # to CAMB (via firecrown's CCLFactory camb_extra_params), so both stacks
    # share the identical nonlinear recipe.
    halofit_version: str = "mead2020_feedback"
    hmcode_logT_AGN: float = 7.5  # values_ia.ini logT_AGN central

    # Neutrino mass split: normal hierarchy (CosmoSIS `neutrino_hierarchy=normal`).
    mass_split: str = "normal"

    # Intrinsic alignments: NLA (linear alignment). The theory engine's sole
    # purpose is the (Omega_m, S8) blinding shift t(hidden) − t(fiducial); IA is
    # a nuisance that enters identically at both cosmologies and nearly cancels,
    # so the fiducial engine defaults IA OFF (ia_bias=0) to isolate the lensing
    # prediction — this is also what makes the CAMB↔CCL cross-check a clean test
    # of the shear calculation. Set `ia_bias` nonzero (CosmoSIS central A=1.0)
    # to include NLA; `ia_z_piv`/`ia_alphaz` set the pivot and z-power.
    ia_bias: float = 0.0
    ia_z_piv: float = 0.62
    ia_alphaz: float = 0.0

    # Derived / fixed CCL background parameters.
    Neff: float = 3.046
    T_CMB: float = 2.7255

    def sigma8(self):
        """CCL ``sigma8`` implied by ``S8`` and ``Omega_m``.

        ``S8 ≡ σ8 √(Ωm / 0.3)`` — the standard weak-lensing definition — so
        ``σ8 = S8 / √(Ωm / 0.3)``.
        """
        return self.S8 / np.sqrt(self.Omega_m / 0.3)

    def omega_c(self):
        """CCL cold-dark-matter density ``Omega_c = Omega_m − Omega_b − Omega_nu``.

        The neutrino density ``Ω_ν h² = Σm_ν / 93.14 eV`` is subtracted so that
        the *total* matter density is exactly ``Omega_m`` (CCL treats massive
        neutrinos as a separate species, not part of ``Omega_c``).
        """
        omega_nu = self.m_nu / (93.14 * self.h**2)
        return self.Omega_m - self.Omega_b - omega_nu

    def ccl_params(self):
        """Cosmological sampling parameters as firecrown expects them.

        Returns the ``{name: value}`` overrides to feed a firecrown
        ``ParamsMap`` so the modeling tools build the CCL cosmology at this
        fiducial. Names match firecrown's CCLFactory parameter set
        (``Omega_c``, ``Omega_b``, ``h``, ``n_s``, ``sigma8``, ``m_nu``, …).
        """
        return {
            "Omega_c": self.omega_c(),
            "Omega_b": self.Omega_b,
            "h": self.h,
            "n_s": self.n_s,
            "sigma8": self.sigma8(),
            "m_nu": self.m_nu,
            "w0": self.w0,
            "wa": self.wa,
            "Neff": self.Neff,
            "T_CMB": self.T_CMB,
            "Omega_k": 0.0,
        }

    def ia_params(self):
        """NLA intrinsic-alignment sampling parameters (firecrown names)."""
        return {
            "ia_bias": self.ia_bias,
            "z_piv": self.ia_z_piv,
            "alphaz": self.ia_alphaz,
        }

    @classmethod
    def from_overrides(cls, overrides):
        """Build from a mapping of field overrides (fail loud on unknown keys)."""
        fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(overrides) - fields
        if unknown:
            raise ValueError(
                f"unknown TheoryConfig fields {sorted(unknown)}; "
                f"valid fields are {sorted(fields)}"
            )
        return cls(**overrides)


# --------------------------------------------------------------------------- #
# SACC preparation: extract the coarse ξ± analysis subset
# --------------------------------------------------------------------------- #
def source_bins(s):
    """Sorted source-bin indices present in ``s`` (from ``source_i`` tracers)."""
    bins = []
    for name in s.tracers:
        if name.startswith("source_"):
            bins.append(int(name.split("_", 1)[1]))
    return sorted(bins)


def xi_pairs(s, grid="coarse"):
    """Unordered source-bin pairs ``(i, j)`` with ``i ≤ j`` carrying coarse ξ+.

    Discovered from the file rather than assumed, so the likelihood models
    exactly the pairs the measurement contains (single-bin ⇒ ``[(0, 0)]``).
    """
    bins = source_bins(s)
    pairs = []
    for a, i in enumerate(bins):
        for j in bins[a:]:
            tracers = (sacc_io.source_name(i), sacc_io.source_name(j))
            if len(s.indices(sacc_io.XI_PLUS, tracers, grid=grid)):
                pairs.append((i, j))
    return pairs


def extract_coarse_xi(s):
    """Return a SACC holding only the coarse ξ± subset (+ aligned covariance).

    firecrown models every point handed to ``read`` and Smokescreen requires
    ``sacc.mean == likelihood.get_data_vector()``, so the likelihood must
    receive a file containing *exactly* the ξ± rows it predicts — no COSEBIs,
    pseudo-Cℓ, ρ/τ. This keeps ξ+ and ξ− on the coarse grid for all tracers
    (in insertion order, so the retained covariance sub-block stays aligned)
    and drops everything else. ``keep_indices`` slices the covariance to match,
    exactly as :func:`sp_validation.sacc_io.extract` relies on.
    """
    keep = np.sort(
        np.concatenate(
            [
                s.indices(sacc_io.XI_PLUS, grid="coarse"),
                s.indices(sacc_io.XI_MINUS, grid="coarse"),
            ]
        )
    )
    sub = s.copy()
    sub.keep_indices(keep)
    return sub


# --------------------------------------------------------------------------- #
# firecrown build_likelihood entry point
# --------------------------------------------------------------------------- #
def _resolve_sacc(sacc_data):
    """Load a SACC from a path, or copy an in-memory Sacc."""
    if isinstance(sacc_data, (str, pathlib.Path)):
        return sacc.Sacc.load_fits(str(sacc_data))
    return sacc_data.copy()


def _resolve_config(build_parameters):
    """Extract a TheoryConfig from firecrown build parameters (default if absent)."""
    try:
        cfg = build_parameters["theory_config"]
    except (KeyError, TypeError):
        return TheoryConfig()
    if isinstance(cfg, TheoryConfig):
        return cfg
    return TheoryConfig.from_overrides(dict(cfg))


def build_likelihood(build_parameters):
    """Firecrown factory: a Gaussian ξ± likelihood over the coarse analysis grid.

    Parameters
    ----------
    build_parameters : firecrown NamedParameters
        Must carry ``sacc_data`` (a SACC path or in-memory ``sacc.Sacc`` — the
        *analysis* file; the coarse ξ± subset is extracted here). May carry
        ``theory_config`` (a :class:`TheoryConfig` or a mapping of overrides);
        absent ⇒ the fiducial defaults.

    Returns
    -------
    (firecrown.likelihood.ConstGaussian, firecrown.modeling_tools.ModelingTools)
        The Gaussian likelihood and the modeling tools whose CCL cosmology is
        configured from the :class:`TheoryConfig`.
    """
    from firecrown.likelihood import ConstGaussian, TwoPoint
    from firecrown.likelihood import weak_lensing as wl
    from firecrown.modeling_tools import (
        CAMBExtraParams,
        CCLCreationMode,
        CCLFactory,
        CCLPureModeTransferFunction,
        ModelingTools,
        PoweSpecAmplitudeParameter,
    )

    try:
        raw = build_parameters["sacc_data"]
    except TypeError:
        raw = build_parameters.data["sacc_data"]
    s = _resolve_sacc(raw)
    cfg = _resolve_config(build_parameters)

    coarse = extract_coarse_xi(s)
    pairs = xi_pairs(coarse, grid="coarse")
    if not pairs:
        raise ValueError(
            "no coarse ξ± data found in the SACC file — the blinding "
            "likelihood needs galaxy_shear_xi_plus/minus points tagged grid='coarse'"
        )

    # One WeakLensing source per bin (NLA systematic wired, off by default).
    bins = sorted({b for pair in pairs for b in pair})
    sources = {
        i: wl.WeakLensing(
            sacc_tracer=sacc_io.source_name(i),
            systematics=[wl.LinearAlignmentSystematic(sacc_tracer=None)],
        )
        for i in bins
    }

    # One (ξ+, ξ−) TwoPoint per pair. Statistic order does not constrain
    # correctness: each statistic exposes ``sacc_indices`` mapping its theory
    # sub-vector back to SACC rows, so callers realign by those indices.
    stats = []
    for i, j in pairs:
        stats.append(TwoPoint(sacc_io.XI_PLUS, sources[i], sources[j]))
        stats.append(TwoPoint(sacc_io.XI_MINUS, sources[i], sources[j]))

    likelihood = ConstGaussian(statistics=stats)
    likelihood.read(coarse)

    # Route the nonlinear P(k) through CAMB's HMCode2020 (matching CosmoSIS's
    # `halofit_version=mead2020_feedback`). firecrown gates camb_extra_params
    # behind PURE_CCL_MODE + the BOLTZMANN_CAMB transfer function — the pure-CCL
    # creation path that builds the ccl.Cosmology directly with CAMB as its
    # power-spectrum backend.
    tools = ModelingTools(
        ccl_factory=CCLFactory(
            require_nonlinear_pk=True,
            amplitude_parameter=PoweSpecAmplitudeParameter.SIGMA8,
            mass_split=cfg.mass_split,
            creation_mode=CCLCreationMode.PURE_CCL_MODE,
            pure_ccl_transfer_function=CCLPureModeTransferFunction.BOLTZMANN_CAMB,
            camb_extra_params=CAMBExtraParams(
                halofit_version=cfg.halofit_version,
                HMCode_logT_AGN=cfg.hmcode_logT_AGN,
            ),
        )
    )
    return likelihood, tools


def compute_theory_vector(likelihood, tools, config=None):
    """Predict the ξ± theory vector at ``config`` (default fiducial).

    A thin, deterministic driver over firecrown's update/prepare/predict cycle,
    for the cross-check test and callers that want a theory vector without
    Smokescreen. Smokescreen itself drives ``compute_theory_vector`` internally;
    this mirrors that path so tests exercise the same code.

    Returns the theory vector in firecrown's statistic order; use
    :func:`theory_vector_sacc_order` to realign it to SACC insertion order.
    """
    from firecrown.updatable import ParamsMap, get_default_params_map

    cfg = config or TheoryConfig()
    params = get_default_params_map(tools, likelihood)
    params = ParamsMap({**dict(params), **cfg.ccl_params(), **cfg.ia_params()})
    tools.update(params)
    tools.prepare()
    likelihood.update(params)
    tv = np.asarray(likelihood.compute_theory_vector(tools))
    likelihood.reset()
    tools.reset()
    return tv


def theory_vector_sacc_order(likelihood, theory_vector, n_sacc):
    """Scatter a firecrown theory vector into SACC insertion order.

    firecrown returns the theory vector in the order its statistics were listed;
    each statistic's ``sacc_indices`` say which SACC rows its block maps to.
    Scattering by those indices yields a vector aligned to the SACC data vector
    (and hence to the covariance) regardless of statistic order.
    """
    out = np.full(n_sacc, np.nan)
    cursor = 0
    for stat in likelihood.statistics:
        idx = np.asarray(stat.statistic.sacc_indices)
        out[idx] = theory_vector[cursor : cursor + len(idx)]
        cursor += len(idx)
    return out
