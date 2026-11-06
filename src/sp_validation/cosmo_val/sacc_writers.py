"""Pair-native SACC writers and covariance helpers for cosmo_val products.

Each writer turns a computed statistic into a single-statistic SACC part.
Pairs are inserted in the supplied order, typically (0, 0), (0, 1), …, with
points and covariances pair-major. Covariance helpers map OneCovariance ξ± and
NaMaster iNKA matrices to the part's point order. :func:`assemble_analysis_sacc`
combines the parts and their covariance blocks into an analysis file.
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


def xi_pairs_to_sacc(nz, metadata, pairs, *, grid, variances=None, covariance=None):
    """One ξ± part over one or more bin pairs, on a named angular grid.

    ``pairs`` is a sequence of ``(bins, xi)``: ``bins`` a source bin pair
    ``(i, j)`` and ``xi`` a mapping with ``theta``, ``xip``, ``xim`` and,
    optionally, ``theta_nom``, ``npairs``, ``weight``. Each pair goes in as
    ``[ξ+; ξ−]`` in the order given, so the data vector is pair-major, and the
    covariance is tied to that order by position: ``covariance`` is the dense
    block in it, ``variances`` the matching diagonal. At most one may be given.
    ``nz`` must hold an n(z) for every bin the pairs name.
    """
    s = sio.new_sacc(nz, metadata)
    for bins, xi in pairs:
        sio.add_xi(
            s,
            bins,
            xi["theta"],
            xi["xip"],
            xi["xim"],
            grid=grid,
            theta_nom=xi.get("theta_nom"),
            npairs=xi.get("npairs"),
            weight=xi.get("weight"),
        )
    if covariance is not None and variances is not None:
        raise ValueError("give the ξ± writer a dense covariance or variances, not both")
    if covariance is not None:
        s.add_covariance(np.asarray(covariance))
    elif variances is not None:
        sio.add_diagonal_covariance(s, np.asarray(variances))
    return s


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
    covariance=None,
):
    """One ξ± part (``bins=(0, 0)``) on a named angular grid.

    The single-pair form of :func:`xi_pairs_to_sacc`. The grid's covariance
    comes in one of two shapes: ``covariance``, the dense ``[ξ+; ξ−]``-ordered
    block (a jackknife estimate), or ``variances``, the concatenated
    ``[varxip; varxim]`` diagonal. At most one may be given.
    """
    xi = {
        "theta": theta,
        "xip": xip,
        "xim": xim,
        "theta_nom": theta_nom,
        "npairs": npairs,
        "weight": weight,
    }
    return xi_pairs_to_sacc(
        nz,
        metadata,
        [((0, 0), xi)],
        grid=grid,
        variances=variances,
        covariance=covariance,
    )


def onecov_xi_covariance(s, cov):
    """OneCovariance's real-space ξ± matrix ``cov`` in the insertion order of ``s``.

    OneCovariance writes ξ± type-major: every ξ+ point, then every ξ− point;
    within each, bin pairs ``(i, j)`` with ``i <= j`` and ``i`` outermost, then
    θ ascending. A ξ± part is pair-major (:func:`xi_pairs_to_sacc`), so this
    finds each point of ``s`` in OneCovariance's order and permutes ``cov``
    to match. The two orders agree for a single bin pair. OneCovariance's bin
    ``k`` is column ``k`` of the n(z) file, which must be the part's
    ``source_k``.

    ``s`` must hold ξ± on one grid only, with the same θ points for every bin
    pair, as a part does.
    """
    cov = np.asarray(cov, float)
    types = {sio.XI_PLUS: 0, sio.XI_MINUS: 1}
    if any(dp.data_type not in types for dp in s.data):
        raise ValueError("onecov_xi_covariance takes a part holding ξ± only")
    grids = {dp.tags.get("grid") for dp in s.data}
    if len(grids) != 1:
        raise ValueError(f"the part's ξ± lies on several grids {sorted(grids)}")
    if cov.shape != (len(s.data), len(s.data)):
        raise ValueError(
            f"OneCovariance matrix is {cov.shape}, but the part has "
            f"{len(s.data)} ξ± points; check the binning and tomography match"
        )

    def bin_index(tracer):
        return int(tracer.removeprefix("source_"))

    pairs = sorted({tuple(sorted(map(bin_index, dp.tracers))) for dp in s.data})
    thetas = {}
    for dp in s.data:
        key = (dp.data_type, tuple(sorted(map(bin_index, dp.tracers))))
        thetas.setdefault(key, []).append(dp.tags["theta"])
    ntheta = {len(t) for t in thetas.values()}
    if len(ntheta) != 1 or len(thetas) != 2 * len(pairs):
        raise ValueError("every bin pair must carry ξ+ and ξ− on the same θ points")
    (ntheta,) = ntheta

    order = np.empty(len(s.data), dtype=int)
    for k, dp in enumerate(s.data):
        pair = tuple(sorted(map(bin_index, dp.tracers)))
        theta_rank = sorted(thetas[dp.data_type, pair]).index(dp.tags["theta"])
        order[k] = (
            types[dp.data_type] * len(pairs) * ntheta
            + pairs.index(pair) * ntheta
            + theta_rank
        )
    return cov[np.ix_(order, order)]


def namaster_cl_covariance(s, cov_fits):
    """Map iNKA pair-major covariance HDUs to a pseudo-Cℓ part's order.

    Each same-polarisation HDU is ordered by source pair, then increasing ℓ.
    Cross-polarisation blocks are zero, including across source pairs.
    """
    from astropy.io import fits

    hdus = {
        sio.CL_EE: "COVAR_EE_EE",
        sio.CL_BB: "COVAR_BB_BB",
        sio.CL_EB: "COVAR_EB_EB",
    }
    if not s.data or any(dp.data_type not in hdus for dp in s.data):
        raise ValueError("namaster_cl_covariance takes a pseudo-Cℓ part only")

    def pair(dp):
        return tuple(sorted(int(t.removeprefix("source_")) for t in dp.tracers))

    pairs = sorted({pair(dp) for dp in s.data})
    ells = {}
    for dp in s.data:
        ells.setdefault((dp.data_type, pair(dp)), []).append(dp.tags["ell"])
    if set(ells) != {(dtype, p) for dtype in hdus for p in pairs}:
        raise ValueError("every pair must carry EE, BB and EB")
    grids = [sorted(values) for values in ells.values()]
    if any(grid != grids[0] for grid in grids) or len(set(grids[0])) != len(grids[0]):
        raise ValueError("every spectrum must have the same distinct ℓ points")
    nell = len(grids[0])
    size = len(pairs) * nell
    with fits.open(cov_fits) as hdul:
        missing = [name for name in hdus.values() if name not in hdul]
        if missing:
            raise ValueError(f"{cov_fits} lacks the pseudo-Cℓ cov HDUs {missing}")
        blocks = {
            dtype: np.asarray(hdul[name].data, float) for dtype, name in hdus.items()
        }
    if any(block.shape != (size, size) for block in blocks.values()):
        raise ValueError(f"iNKA covariance blocks must have shape {(size, size)}")
    result = np.zeros((len(s.data), len(s.data)))
    for dtype, block in blocks.items():
        positions = [i for i, dp in enumerate(s.data) if dp.data_type == dtype]
        order = [
            pairs.index(pair(s.data[i])) * nell + grids[0].index(s.data[i].tags["ell"])
            for i in positions
        ]
        result[np.ix_(positions, positions)] = block[np.ix_(order, order)]
    return result


def pseudo_cl_to_sacc(nz, metadata, pairs, covariance=None):
    """Write pair-major EE/BB/EB spectra and their bandpower windows.

    Each payload contains ``ell_eff``, NaMaster's ``cl_all`` in row order
    [EE, EB, BE, BB], and ``wsp``. Supply ``nside`` only for HEALPix-map
    spectra, to include the squared pixel window pw²(ℓ); catalogue spectra
    have no pixel window. Covariance follows the pair-major insertion order.
    """
    s = sio.new_sacc(nz, metadata)
    for bins, payload in pairs:
        cl_all = payload["cl_all"]
        window_ells, window_weights = bandpower_window_from_workspace(
            payload["wsp"], payload.get("nside")
        )
        sio.add_pseudo_cl(
            s,
            bins,
            payload["ell_eff"],
            cl_all[_NMT_EE],
            cl_all[_NMT_BB],
            cl_all[_NMT_EB],
            window_ells=window_ells,
            window_weights=window_weights,
        )
    if covariance is not None:
        s.add_covariance(np.asarray(covariance))
    return s


def cosebis_to_sacc(nz, metadata, pairs, scale_cut, covariance=None):
    """Write each pair's COSEBIs in [En; Bn] order at the scale cut.

    Each payload contains ``En`` and ``Bn``. The supplied covariance follows
    that pair-major order and has its E×B and B×E blocks zeroed, including
    across pairs.
    """
    s = sio.new_sacc(nz, metadata)
    for bins, result in pairs:
        sio.add_cosebis(s, bins, result["En"], scale_cut, Bn=result["Bn"])
    if covariance is not None:
        s.add_covariance(np.asarray(covariance))
    return s


def pure_eb_to_sacc(nz, metadata, pairs, covariance=None):
    """Write each pair's ``theta`` and six arrays in ``PURE_KEYS`` order.

    The order is ``xip_E``, ``xim_E``, ``xip_B``, ``xim_B``, ``xip_amb``,
    ``xim_amb``; the covariance follows this order within each pair.
    """
    s = sio.new_sacc(nz, metadata)
    for bins, eb in pairs:
        sio.add_pure_eb(s, bins, eb["theta"], **{key: eb[key] for key in sio.PURE_KEYS})
    if covariance is not None:
        s.add_covariance(np.asarray(covariance))
    return s


def rho_tau_to_sacc(nz, metadata, rho_stats, tau_by_bin, tau_cov_th_by_bin=None):
    """One ρ/τ part: ρ_0…ρ_5 autos and τ_0/τ_2/τ_5 leakage.

    ``rho_stats`` contains ``theta``, ``rho_{k}_p``, ``rho_{k}_m``,
    ``varrho_{k}_p`` and ``varrho_{k}_m`` for k=0,…,5. Each table in
    ``tau_by_bin`` contains ``theta``, ``tau_{k}_p``, ``tau_{k}_m``,
    ``vartau_{k}_p`` and ``vartau_{k}_m`` for k=0,2,5.
    ρ carries a variance diagonal. Each τ block has a variance diagonal,
    with its optional CovTauTh matrix scattered into its plus rows. CovTauTh
    has shape (3·nθ, 3·nθ), ordered k-major (0, 2, 5), plus only.
    Cross-bin covariances are approximated as zero.
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
    rho_var = np.concatenate(
        [
            np.concatenate([rho_stats[f"varrho_{k}_p"], rho_stats[f"varrho_{k}_m"]])
            for k in RHO_K
        ]
    )
    theory = dict(tau_cov_th_by_bin or [])
    blocks = [np.diag(rho_var)]
    diagonals = [rho_var]
    for bin_index, tau_stats in tau_by_bin:
        theta = np.asarray(tau_stats["theta"])
        for k in TAU_K:
            sio.add_tau(
                s,
                (bin_index, bin_index),
                k,
                theta,
                np.asarray(tau_stats[f"tau_{k}_p"]),
                np.asarray(tau_stats[f"tau_{k}_m"]),
            )
        tau_var = np.concatenate(
            [
                np.concatenate([tau_stats[f"vartau_{k}_p"], tau_stats[f"vartau_{k}_m"]])
                for k in TAU_K
            ]
        )
        diagonals.append(tau_var)
        block = np.diag(tau_var)
        cov = theory.get(bin_index)
        if cov is not None:
            cov = np.asarray(cov)
            nbin = len(theta)
            n_plus = len(TAU_K) * nbin
            if cov.shape != (n_plus, n_plus):
                raise ValueError(
                    f"tau_cov_th shape {cov.shape} does not match {n_plus} τ-plus points"
                )
            plus = np.concatenate(
                [
                    np.arange(2 * i * nbin, 2 * i * nbin + nbin)
                    for i in range(len(TAU_K))
                ]
            )
            block[np.ix_(plus, plus)] = cov
        blocks.append(block)
    if not theory:
        s.add_covariance(np.concatenate(diagonals))
    else:
        from scipy.linalg import block_diag

        s.add_covariance(block_diag(*blocks))
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
