"""Born-as-SACC writers for the cosmo_val data products.

A thin, pure layer between the ``cosmo_val`` mixins (which compute statistics as
TreeCorr / NaMaster / b_modes arrays) and :mod:`sp_validation.sacc_io` (which
knows the file layout). Each ``*_to_sacc`` function turns one already-computed
statistic into a single-statistic SACC — a *part* — carrying that statistic's
own covariance as its one block. :func:`assemble_analysis_sacc` rebuilds the
single ``{version}.sacc`` analysis file from these parts.

The integration-grid ξ± part is an intermediate consumed by COSEBIs and
pure-E/B; it is blinded at birth but does not join ``{version}.sacc``.

Everything here is single-bin today (``bins=(0, 0)``); the interface is
tomography-native so a future round supplies real bin pairs unchanged.
"""

import numpy as np
import sacc

from .. import sacc_io as sio
from ..pseudo_cl import bandpower_window_from_workspace

# Statistics carried in the analysis file, and their custom-type k indices.
RHO_K = range(6)  # ρ_0 … ρ_5
TAU_K = (0, 2, 5)  # τ_0, τ_2, τ_5

# NaMaster spin-2 × spin-2 decoupled-spectrum row order (EE, EB, BE, BB).
_NMT_EE, _NMT_EB, _NMT_BB = 0, 1, 3

BIN = (0, 0)


def xi_to_sacc(
    nz,
    metadata,
    theta,
    xip,
    xim,
    *,
    grid,
    theta_nom=None,
    npairs=None,
    weight=None,
    variances=None,
):
    """One ξ± part (``bins=(0, 0)``) on the reporting or integration grid.

    ``variances`` is the concatenated ``[varxip; varxim]``; when given, attaches
    a ``DiagonalCovariance``.
    """
    s = sio.new_sacc(nz, metadata)
    sio.add_xi(
        s,
        BIN,
        theta,
        xip,
        xim,
        grid=grid,
        theta_nom=theta_nom,
        npairs=npairs,
        weight=weight,
    )
    if variances is not None:
        sio.add_diagonal_covariance(s, np.asarray(variances))
    return s


def pseudo_cl_to_sacc(nz, metadata, ell_eff, cl_all, wsp, covariance=None):
    """One pseudo-Cℓ part: EE/BB/EB with the shared bandpower window.

    ``cl_all`` is NaMaster's decoupled ``(4, nbp)`` array (EE, EB, BE, BB); the
    window comes from :func:`bandpower_window_from_workspace`. ``covariance``,
    when given, is the dense ``[EE; BB; EB]``-ordered block matching insertion.
    """
    window_ells, window_weights = bandpower_window_from_workspace(wsp)
    s = sio.new_sacc(nz, metadata)
    sio.add_pseudo_cl(
        s,
        BIN,
        ell_eff,
        cl_all[_NMT_EE],
        cl_all[_NMT_BB],
        cl_all[_NMT_EB],
        window_ells=window_ells,
        window_weights=window_weights,
    )
    if covariance is not None:
        s.add_covariance(np.asarray(covariance))
    return s


def cosebis_to_sacc(nz, metadata, result, scale_cut):
    """One COSEBIs part at the fiducial scale cut.

    ``result`` is a single scale-cut result dict from
    ``b_modes.calculate_cosebis`` — ``{"En", "Bn", "cov", ...}`` — where ``cov``
    is the ``[En; Bn]``-ordered COSEBIs covariance.
    """
    s = sio.new_sacc(nz, metadata)
    sio.add_cosebis(s, BIN, result["En"], scale_cut, Bn=result["Bn"])
    s.add_covariance(np.asarray(result["cov"]))
    return s


def pure_eb_to_sacc(nz, metadata, theta, eb, covariance=None):
    """One pure-E/B part: the six ``sacc_io.PURE_KEYS`` blocks.

    ``eb`` is a mapping with the six keys (``xip_E`` … ``xim_amb``); each array
    is sampled at ``theta``. ``covariance``, when given, is the dense block in
    ``PURE_KEYS`` order (matching ``b_modes._EB_KEYS`` and the insertion order).
    """
    s = sio.new_sacc(nz, metadata)
    sio.add_pure_eb(s, BIN, theta, **{key: eb[key] for key in sio.PURE_KEYS})
    if covariance is not None:
        s.add_covariance(np.asarray(covariance))
    return s


def rho_tau_to_sacc(nz, metadata, rho_stats, tau_stats, tau_cov_th=None):
    """One ρ/τ part: ρ_0…ρ_5 autos and τ_0/τ_2/τ_5 leakage.

    ``rho_stats`` / ``tau_stats`` are the ``shear_psf_leakage`` handler tables
    (columns ``theta``, ``rho_{k}_p``, ``varrho_{k}_p``, … and the τ analogue).
    ρ carries a ``varrho`` diagonal; τ carries a ``vartau`` diagonal, with
    ``tau_cov_th`` — a ``(3·nbin, 3·nbin)`` k-major matrix over the τ-plus points
    only — scattered into the τ-plus rows/columns when given. ``tau_cov_th=None``
    leaves the τ block fully diagonal.
    """
    s = sio.new_sacc(nz, metadata)
    theta_rho = np.asarray(rho_stats["theta"])
    for k in RHO_K:
        sio.add_rho(
            s,
            k,
            theta_rho,
            np.asarray(rho_stats[f"rho_{k}_p"]),
            np.asarray(rho_stats[f"rho_{k}_m"]),
        )
    theta_tau = np.asarray(tau_stats["theta"])
    for k in TAU_K:
        sio.add_tau(
            s,
            BIN,
            k,
            theta_tau,
            np.asarray(tau_stats[f"tau_{k}_p"]),
            np.asarray(tau_stats[f"tau_{k}_m"]),
        )
    nbin = len(theta_tau)
    rho_var = np.concatenate(
        [
            np.concatenate([rho_stats[f"varrho_{k}_p"], rho_stats[f"varrho_{k}_m"]])
            for k in RHO_K
        ]
    )
    tau_var = np.concatenate(
        [
            np.concatenate([tau_stats[f"vartau_{k}_p"], tau_stats[f"vartau_{k}_m"]])
            for k in TAU_K
        ]
    )
    if tau_cov_th is None:
        s.add_covariance(np.concatenate([rho_var, tau_var]))
        return s
    tau_cov_th = np.asarray(tau_cov_th)
    n_plus = len(TAU_K) * nbin
    if tau_cov_th.shape != (n_plus, n_plus):
        raise ValueError(
            f"tau_cov_th shape {tau_cov_th.shape} does not match the "
            f"{n_plus} τ-plus points ({len(TAU_K)} indices × {nbin} bins) — "
            "CovTauTh.build_cov returns one (plus-folded) component per τ index"
        )
    n_rho, n_tau = len(rho_var), len(tau_var)
    tau_block = np.diag(tau_var)
    # τ-plus local positions in the τ block, k-major (per-k layout is [+; −]).
    plus = np.concatenate(
        [np.arange(2 * i * nbin, 2 * i * nbin + nbin) for i in range(len(TAU_K))]
    )
    tau_block[np.ix_(plus, plus)] = tau_cov_th
    full = np.zeros((n_rho + n_tau, n_rho + n_tau))
    full[:n_rho, :n_rho] = np.diag(rho_var)
    full[n_rho:, n_rho:] = tau_block
    s.add_covariance(full)
    return s


# --------------------------------------------------------------------------- #
# Analysis-file assembly
# --------------------------------------------------------------------------- #
def _copy_data_points(dst, src):
    """Append every data point of ``src`` into ``dst`` (tags preserved)."""
    for dp in src.data:
        dst.add_data_point(dp.data_type, dp.tracers, dp.value, **dp.tags)


def assemble_analysis_sacc(parts):
    """Rebuild the single ``{version}.sacc`` analysis file from parts.

    Each part is a single-statistic Sacc (from a ``*_to_sacc`` writer, loaded
    from disk) carrying its own covariance = its block. Tracers and metadata are
    seeded from ``parts[0]`` (every part describes the same catalogue version).
    Data points are re-added in the order the parts are given, which must be the
    canonical order (ξ± reporting, pseudo-Cℓ, COSEBIs, pure-E/B, ρ/τ), and the
    per-part blocks become one ``BlockDiagonalCovariance``. Insertion order and
    block order therefore agree by construction — validated by
    :func:`sp_validation.sacc_io.assemble_covariance`.

    Parameters
    ----------
    parts : sequence of sacc.Sacc
        Single-statistic parts, each with a covariance, in canonical order.

    Returns
    -------
    sacc.Sacc
        The analysis Sacc with a ``BlockDiagonalCovariance`` covering every point.
    """
    s = sacc.Sacc()
    s.tracers.update(parts[0].tracers)
    s.metadata.update(parts[0].metadata)
    blocks = []
    cursor = 0
    for part in parts:
        if part.covariance is None:
            raise ValueError(
                "every analysis part must carry its own covariance block; "
                f"a part with data types {sorted(set(dp.data_type for dp in part.data))} "
                "has none"
            )
        n = len(part.mean)
        _copy_data_points(s, part)
        blocks.append((np.arange(cursor, cursor + n), part.covariance.dense))
        cursor += n
    return sio.assemble_covariance(s, blocks)
