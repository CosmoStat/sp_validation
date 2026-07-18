"""TWOPOINT_CONVERT.

:Name: twopoint_convert.py

:Description: Convert an analysis SACC file into the "2pt FITS" that CosmoSIS's
              ``2pt_like`` (and Sacha Guerrini's ρ/τ ``2pt_like_xi_sys`` fork)
              reads. The output reproduces today's hand-assembled product from
              ``cosmo_inference/scripts/cosmosis_fitting.py`` HDU-for-HDU and
              byte-for-byte: an NZDATA table, XI_PLUS / XI_MINUS 2pt tables,
              optional CELL_EE / CELL_BB pseudo-Cℓ tables, the blocked COVMAT
              (with ``STRT_i`` block-offset headers) and separate COVMAT_CELL,
              and — when the SACC carries them — the TAU_{0,2}_PLUS 2pt tables
              and the RHO_STATS table.

              The converter is the *inverse* of the SACC writers in
              :mod:`sp_validation.sacc_io`: it reads statistics back through
              those readers and lays them into the DES ``twopoint`` FITS
              convention. SACC's canonical order is pair-major (per pair
              ``[ξ+; ξ−]``); the 2pt-FITS layout is type-major (all ξ+, then all
              ξ−), so the data-vector and its covariance are permuted here via
              ``s.indices`` rather than assuming any global order.

              Scope note (single-bin today, tomography-ready): the assembly this
              mirrors is single-tomographic-bin — BIN1/BIN2 are all 1, one NZ
              ``BIN1`` column. The converter reads bin ``(0, 0)`` accordingly.
              A tomographic 2pt-FITS layout (multiple bin pairs, per-pair
              BIN1/BIN2, one NZ column per bin) is a later extension; it is not
              what today's CosmoSIS pipeline consumes, so it is out of scope for
              the byte-compatible converter.

              Rho/tau caveat: the SACC layout stores ρ±/τ± *values* only, while
              the 2pt-FITS RHO_STATS table also carries the per-mode *variances*
              (``varrho_*``) that Sacha's fork's covariance path reads. Those
              variances are not recoverable from the analysis SACC. The
              converter therefore writes RHO_STATS / TAU HDUs only when a
              ``rho_stats``/``tau_stats`` sidecar FITS is supplied (the same
              file today's assembly copies verbatim); it never fabricates
              variances. ξ±, Cℓ, n(z) and the covariance — the data vector
              CosmoSIS fits — are fully reconstructed from SACC alone.
"""

import numpy as np
from astropy.io import fits

from . import sacc_io

# The QUANT1/QUANT2 header pair CosmoSIS stamps on each 2pt table, keyed by the
# extension name — copied from cosmosis_fitting.py so the headers match card
# for card.
_QUANT = {
    "XI_PLUS": ("G+R", "G+R"),
    "XI_MINUS": ("G-R", "G-R"),
    "CELL_EE": ("GEF", "GEF"),
    "CELL_BB": ("GBF", "GBF"),
    "TAU_0_PLUS": ("G+R", "P+R"),
    "TAU_2_PLUS": ("G+R", "SR+R"),
}


def _twopoint_hdu(name, values, ang, *, ang_unit=None):
    """Build one 2pt BinTableHDU (BIN1/BIN2/ANGBIN/VALUE/ANG).

    Reproduces ``cosmosis_fitting.py._create_2pt_hdu`` /``cl_to_fits`` exactly:
    same column order and formats, the ``2PTDATA`` marker, the QUANT pair for
    ``name``, and NZ_SOURCE kernels. ``ang_unit`` stamps ``TUNIT`` on the ANG
    column ("arcmin" for real-space ξ/τ; unset for Cℓ, whose ANG is ℓ).
    """
    nbins = len(values)
    angbin = np.arange(1, nbins + 1)
    columns = [
        fits.Column(name="BIN1", format="K", array=np.ones(nbins)),
        fits.Column(name="BIN2", format="K", array=np.ones(nbins)),
        fits.Column(name="ANGBIN", format="K", array=angbin),
        fits.Column(name="VALUE", format="D", array=values),
        fits.Column(name="ANG", format="D", unit=ang_unit, array=ang),
    ]
    hdu = fits.BinTableHDU.from_columns(fits.ColDefs(columns), name=name)
    quant1, quant2 = _QUANT[name]
    for key, value in {
        "2PTDATA": "T",
        "QUANT1": quant1,
        "QUANT2": quant2,
        "KERNEL_1": "NZ_SOURCE",
        "KERNEL_2": "NZ_SOURCE",
        "WINDOWS": "SAMPLE",
    }.items():
        hdu.header[key] = value
    return hdu


def _nz_hdu(s, n_bins):
    """Build the NZDATA HDU from the SACC ``source_i`` NZ tracers.

    Reproduces ``cosmosis_fitting.py.nz_to_fits``: Z_MID from the tracer ``z``
    grid (assumed uniform), Z_LOW/Z_HIGH as ± half a step, one ``BIN{i+1}``
    column per source bin, and the NZDATA/NBIN/NZ header cards. All source bins
    are required to share the ``z`` grid — the single ``Z_MID`` axis of the
    DES NZDATA table.
    """
    z_mid, nz0 = sacc_io.get_nz(s, 0)
    z_mid = np.asarray(z_mid, dtype=float)
    step = z_mid[1] - z_mid[0]
    z_low = z_mid - step / 2
    z_high = z_mid + step / 2

    columns = [
        fits.Column(name="Z_LOW", format="D", array=z_low),
        fits.Column(name="Z_MID", format="D", array=z_mid),
        fits.Column(name="Z_HIGH", format="D", array=z_high),
    ]
    for i in range(n_bins):
        z_i, nz_i = sacc_io.get_nz(s, i)
        if not np.array_equal(np.asarray(z_i, dtype=float), z_mid):
            raise ValueError(
                f"source bin {i} n(z) grid differs from source bin 0; the DES "
                "NZDATA table requires one shared Z_MID axis"
            )
        columns.append(fits.Column(name=f"BIN{i + 1}", format="D", array=nz_i))

    hdu = fits.BinTableHDU.from_columns(fits.ColDefs(columns), name="NZDATA")
    for key, value in {
        "NZDATA": "T  ",
        "EXTNAME": "NZ_SOURCE",
        "NBIN": n_bins,
        "NZ": len(z_low),
    }.items():
        hdu.header[key] = value
    return hdu


def _cov_hdu(matrix, block_names, block_starts, extname="COVMAT", name_in_ctor=False):
    """Build a covariance ImageHDU with ``NAME_i``/``STRT_i`` block headers.

    Reproduces the two covariance builders in ``cosmosis_fitting.py`` card for
    card. The blocked ξ/τ ``covdat_to_fits`` builds ``ImageHDU(cov)`` unnamed
    and stamps ``COVDATA`` then ``EXTNAME`` from a dict; the ``cov_cl_to_fits``
    CELL covariance builds ``ImageHDU(cov, name="COVMAT_CELL")`` (so the EXTNAME
    card is created early, with astropy's standard comment) before re-stamping.
    ``name_in_ctor`` selects the second form so the card order matches exactly.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"covariance must be square; got shape {matrix.shape}")
    hdu = fits.ImageHDU(matrix, name=extname) if name_in_ctor else fits.ImageHDU(matrix)
    hdu.header["COVDATA"] = "True"
    hdu.header["EXTNAME"] = extname
    for i, (name, start) in enumerate(zip(block_names, block_starts)):
        hdu.header[f"NAME_{i}"] = name
        hdu.header[f"STRT_{i}"] = int(start)
    return hdu


def _type_major_xi(s, bins):
    """Return ``(theta, xip, xim)`` for one bin pair from the SACC reporting grid.

    ``sacc_io.get_xi`` already returns each statistic in insertion (= ascending
    θ) order; the type-major split (all ξ+, then all ξ−) is exactly the two
    arrays it hands back, so no further permutation is needed for a single pair.
    """
    return sacc_io.get_xi(s, bins, grid="reporting")


def _require_single_bin(s, n_bins):
    """Fail fast unless the SACC is a valid single-bin ξ product.

    The converter emits the single-bin 2pt-FITS today's CosmoSIS pipeline reads
    (BIN1/BIN2 all 1, one NZ column). A tomographic SACC would otherwise slip
    through silently — ``n_bins`` alone drives the NZDATA column count while the
    ξ/covariance are read from bin ``(0, 0)`` only, so a 2-bin file would emit a
    ``NBIN=2`` n(z) beside a data vector holding just the ``(0, 0)`` pair.
    Guards both the empty-ξ case and the single-bin contract; tomographic
    emission lands with the tomographic round.
    """
    pairs = s.get_tracer_combinations(sacc_io.XI_PLUS)
    if not pairs:
        raise ValueError(
            f"SACC has no {sacc_io.XI_PLUS} points — nothing to convert; the "
            "2pt-FITS data vector is built from the ξ± statistics"
        )
    expected = (sacc_io.source_name(0), sacc_io.source_name(0))
    if n_bins != 1 or set(pairs) != {expected}:
        raise ValueError(
            f"converter is single-bin only (n_bins=1, ξ pairs == {{{expected}}}); "
            f"got n_bins={n_bins} and ξ pairs {sorted(pairs)}. Tomographic "
            "emission (multiple bin pairs, per-pair BIN1/BIN2, one NZ column per "
            "bin) lands with the tomographic round."
        )


def sacc_to_twopoint_fits(
    s,
    path,
    *,
    rho_stats_hdu=None,
    tau_stats_hdu=None,
    n_bins=1,
):
    """Convert an analysis SACC to a CosmoSIS 2pt-FITS file.

    The assembled ``HDUList`` matches today's ``cosmosis_fitting.py`` product
    for the configuration the SACC describes: PRIMARY, NZ_SOURCE, COVMAT, then
    (if present) COVMAT_CELL, XI_PLUS, XI_MINUS, (if present) CELL_EE / CELL_BB,
    and (if the rho/tau sidecars are supplied) TAU_0_PLUS, TAU_2_PLUS,
    RHO_STATS. The data vector and its covariance are laid out type-major
    (all ξ+, then all ξ−, then the τ blocks), which is the DES ``twopoint``
    convention CosmoSIS reads.

    Parameters
    ----------
    s : sacc.Sacc
        Analysis SACC (reporting ξ±, optional pseudo-Cℓ, covariance, and — for the
        ρ/τ product — the τ data points; see ``rho_stats_hdu``).
    path : str
        Output FITS path (overwritten).
    rho_stats_hdu, tau_stats_hdu : astropy.io.fits.BinTableHDU, optional
        The rho-stats / tau-stats sidecar HDUs, copied verbatim as today's
        assembly does. Required together to write the ρ/τ product; the SACC
        alone cannot rebuild the ``varrho_*`` columns Sacha's fork reads. When
        omitted, a pure ξ (± Cℓ) product is written.
    n_bins : int, optional
        Number of source tomographic bins. Must be ``1``: this converter emits
        the single-bin 2pt-FITS today's CosmoSIS pipeline consumes. Tomographic
        emission (multiple bin pairs, per-pair BIN1/BIN2, one NZ column per bin)
        lands with the tomographic round; the converter fails fast on anything
        else rather than silently truncating to bin ``(0, 0)``.

    Returns
    -------
    astropy.io.fits.HDUList
        The assembled list, also written to ``path``.

    Raises
    ------
    ValueError
        If the SACC has no ξ points; if ``n_bins != 1`` or the SACC's ξ tracer
        pairs are anything other than exactly ``{(source_0, source_0)}`` (the
        single-bin contract); or if exactly one of the ρ/τ sidecars is supplied.
    """
    if (rho_stats_hdu is None) != (tau_stats_hdu is None):
        raise ValueError(
            "rho_stats_hdu and tau_stats_hdu must be supplied together "
            "(the ρ/τ product needs both, or neither for a pure-ξ product)"
        )
    _require_single_bin(s, n_bins)
    use_rho_tau = rho_stats_hdu is not None
    bins = (0, 0)

    nz_hdu = _nz_hdu(s, n_bins)
    theta, xip, xim = _type_major_xi(s, bins)
    xip_hdu = _twopoint_hdu("XI_PLUS", xip, theta, ang_unit="arcmin")
    xim_hdu = _twopoint_hdu("XI_MINUS", xim, theta, ang_unit="arcmin")

    cell_hdu, cov_cell_hdu = _build_cell(s, bins)

    cov_hdu = _build_covmat(s, bins, use_rho_tau=use_rho_tau)

    tau_hdus, rho_hdu = _build_rho_tau(rho_stats_hdu, tau_stats_hdu, theta, use_rho_tau)

    # HDU order mirrors cosmosis_fitting.py's __main__: PRIMARY, NZ, COVMAT,
    # COVMAT_CELL, XI±, CELL_EE, then the τ/ρ tables.
    hdu_list = [fits.PrimaryHDU(), nz_hdu, cov_hdu]
    if cov_cell_hdu is not None:
        hdu_list.append(cov_cell_hdu)
    hdu_list.extend([xip_hdu, xim_hdu])
    if cell_hdu is not None:
        hdu_list.append(cell_hdu)
    if use_rho_tau:
        hdu_list.extend([*tau_hdus, rho_hdu])

    hdul = fits.HDUList(hdu_list)
    hdul.writeto(path, overwrite=True)
    return hdul


def _build_cell(s, bins):
    """Build the CELL_EE 2pt HDU plus the COVMAT_CELL HDU from the SACC pseudo-Cℓ.

    Returns ``(None, None)`` when the SACC has no pseudo-Cℓ. Only CELL_EE is
    emitted — the harmonic ``2pt_like`` fits ``data_sets=CELL_EE``, and today's
    assembly appends CELL_EE alone (it builds a CELL_BB HDU but discards it).
    The SACC still carries EE/BB/EB with bandpower windows for the B-mode
    null-test path; this converter surfaces only the block CosmoSIS reads. The
    CELL covariance (the EE bandpower covariance) lives in its own COVMAT_CELL
    ImageHDU, matching today's product.
    """
    if sacc_io.CL_EE not in s.get_data_types():
        return None, None

    ell, cl_ee, _cl_bb, _cl_eb, _window = sacc_io.get_pseudo_cl(s, bins)
    cell_hdu = _twopoint_hdu("CELL_EE", cl_ee, ell)
    cell_idx = sacc_io._indices(s, sacc_io.CL_EE, sacc_io._pair(bins), grid="reporting")
    cov_cell = s.covariance.dense[np.ix_(cell_idx, cell_idx)]
    cov_cell_hdu = _cov_hdu(
        cov_cell, ["CELL_EE"], [0], extname="COVMAT_CELL", name_in_ctor=True
    )
    return cell_hdu, cov_cell_hdu


def _build_covmat(s, bins, *, use_rho_tau):
    """Assemble the blocked COVMAT (ξ± type-major, then the τ blocks).

    The ξ covariance is pulled from the SACC as the contiguous ξ+/ξ− block for
    the pair and permuted from pair-major (SACC) to type-major (2pt-FITS). Under
    ``use_rho_tau`` the τ_0/τ_2 covariance blocks are appended block-diagonally
    with zero ξ↔τ cross-blocks, exactly as ``covdat_to_fits`` builds them.
    """
    pair = sacc_io._pair(bins)
    idx_p = sacc_io._indices(s, sacc_io.XI_PLUS, pair, grid="reporting")
    idx_m = sacc_io._indices(s, sacc_io.XI_MINUS, pair, grid="reporting")
    n_theta = len(idx_p)
    xi_idx = np.concatenate([idx_p, idx_m])  # type-major permutation
    xi_cov = s.covariance.dense[np.ix_(xi_idx, xi_idx)]

    names = ["XI_PLUS", "XI_MINUS"]
    starts = [0, n_theta]
    matrix = xi_cov

    if use_rho_tau:
        # The τ covariance couples τ_0+ and τ_2+ (today's assembly truncates the
        # 3-statistic CosmoCov τ covariance to its first 2 blocks and lays it in
        # as ONE contiguous [τ_0+; τ_2+] block — cross-correlation kept). In the
        # SACC those two selections are not adjacent (τ_0− sits between them), so
        # gather both index sets and extract the joint sub-block, ξ↔τ zero.
        tau_pair = (sacc_io.source_name(0), sacc_io.PSF_TRACER)
        idx_tau0 = sacc_io._indices(
            s, sacc_io.TAU_PLUS.format(k=0), tau_pair, grid="reporting"
        )
        idx_tau2 = sacc_io._indices(
            s, sacc_io.TAU_PLUS.format(k=2), tau_pair, grid="reporting"
        )
        tau_idx = np.concatenate([idx_tau0, idx_tau2])
        tau_cov = s.covariance.dense[np.ix_(tau_idx, tau_idx)]
        matrix = _block_diag(matrix, tau_cov)
        names += ["TAU_0_PLUS", "TAU_2_PLUS"]
        starts += [2 * n_theta, 2 * n_theta + len(idx_tau0)]

    return _cov_hdu(matrix, names, starts)


def _block_diag(*blocks):
    """Stack square blocks block-diagonally with zero cross-blocks."""
    sizes = [b.shape[0] for b in blocks]
    n = sum(sizes)
    out = np.zeros((n, n))
    start = 0
    for b in blocks:
        out[start : start + b.shape[0], start : start + b.shape[0]] = b
        start += b.shape[0]
    return out


def _build_rho_tau(rho_stats_hdu, tau_stats_hdu, theta, use_rho_tau):
    """Build the TAU_{0,2}_PLUS 2pt HDUs and the verbatim RHO_STATS HDU.

    Mirrors ``tau_to_fits`` / ``rho_to_fits``: τ_0/τ_2 read their ``tau_k_p``
    columns onto the shared ξ θ grid (consistency step); RHO_STATS is copied
    verbatim from the sidecar with its θ column forced onto the ξ grid. The
    ``varrho_*`` columns ride along in the copy — they are why the sidecar is
    required (the SACC cannot supply them).
    """
    if not use_rho_tau:
        return (), None

    tau = tau_stats_hdu.data
    tau0_hdu = _twopoint_hdu("TAU_0_PLUS", tau["tau_0_p"], theta, ang_unit="arcmin")
    tau2_hdu = _twopoint_hdu("TAU_2_PLUS", tau["tau_2_p"], theta, ang_unit="arcmin")

    rho_hdu = rho_stats_hdu.copy()
    rho_hdu.name = "RHO_STATS"
    rho_hdu.data = rho_hdu.data.copy()
    rho_hdu.data["theta"] = theta
    return (tau0_hdu, tau2_hdu), rho_hdu
