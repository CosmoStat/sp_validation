"""SACC_IO.

:Name: sacc_io.py

:Description: Read/write the standard SACC data-product layout for the
              weak-lensing validation package. Two files describe each
              catalogue version:

              - ``{version}.sacc`` — the analysis vector: NZ tracers, coarse
                ξ±, pseudo-Cℓ (EE/BB/EB) with bandpower windows, COSEBIs,
                pure E/B, and ρ/τ PSF diagnostics, all sharing a single
                ``FullCovariance`` assembled block-diagonally from the
                per-statistic covariances (zero cross-blocks).
              - ``{version}_xi_fine.sacc`` — the COSEBIs / pure-EB integration
                input: the same NZ tracers, a fine-grid ξ±, and a
                ``DiagonalCovariance`` from TreeCorr ``varxip``/``varxim``.

              The covariance order is the point-insertion order (SACC preserves
              it bitwise through FITS save/load). Writers below insert in the
              canonical order — ξ+ then ξ−, Cℓ (ee, bb, eb), COSEBIs (all Eₙ
              then all Bₙ), pure E/B (xip_E, xim_E, xip_B, xim_B, xip_amb,
              xim_amb — matching ``b_modes._EB_KEYS``), ρ, then τ — but readers
              never assume global order: they resolve indices through
              ``Sacc.indices(dtype, tracers, **tags)``.

              Tag filters are plain keyword arguments to ``indices`` /
              ``get_data_points`` / ``get_tag``; the ``tags={...}`` form
              silently selects nothing and must never be used.

              Tomographic ξ ordering: each ``add_xi`` call inserts one tracer
              pair as ``[xip; xim]``, so a multi-pair vector is *pair-major*
              — ``[pair_0 xip; pair_0 xim; pair_1 xip; …]`` — not type-major
              (``[all xip; all xim]``). A tomographic ξ covariance with
              cross-pair correlations is therefore supplied to
              ``assemble_covariance`` as ONE contiguous block spanning the
              consecutive ``add_xi`` calls, ordered pair-by-pair to match
              insertion. Writers must call ``add_xi`` in the same pair order
              the covariance was built in. Converters that need a type-major
              layout (e.g. the DES 2pt-FITS convention) permute explicitly via
              ``s.indices`` rather than assuming global order.
"""

import numpy as np
import sacc

PSF_TRACER = "psf_stars"

# Standard SACC data-type strings.
XI_PLUS = "galaxy_shear_xi_plus"
XI_MINUS = "galaxy_shear_xi_minus"
CL_EE = "galaxy_shear_cl_ee"
CL_BB = "galaxy_shear_cl_bb"
CL_EB = "galaxy_shear_cl_eb"
COSEBI_EE = "galaxy_shear_cosebi_ee"
COSEBI_BB = "galaxy_shear_cosebi_bb"

# Custom data-type strings (all parse under sacc.parse_data_type_name).
PURE_TYPES = {
    "xip_E": "galaxy_shear_xiPureE_plus",
    "xim_E": "galaxy_shear_xiPureE_minus",
    "xip_B": "galaxy_shear_xiPureB_plus",
    "xim_B": "galaxy_shear_xiPureB_minus",
    "xip_amb": "galaxy_shear_xiPureAmb_plus",
    "xim_amb": "galaxy_shear_xiPureAmb_minus",
}
# Insertion order of the six pure-EB blocks — matches b_modes._EB_KEYS, whose
# order is the [xip_E; xim_E; xip_B; xim_B; xip_amb; xim_amb] layout of the
# treecorr/MC pure-EB covariance (b_modes.calculate_eb_statistics, ~L392).
PURE_KEYS = ("xip_E", "xim_E", "xip_B", "xim_B", "xip_amb", "xim_amb")

RHO_PLUS = "psf_rho{k}_xi_plus"
RHO_MINUS = "psf_rho{k}_xi_minus"
TAU_PLUS = "galaxyPsf_tau{k}_xi_plus"
TAU_MINUS = "galaxyPsf_tau{k}_xi_minus"


def source_name(i):
    """SACC tracer name for source redshift bin ``i`` (0-based)."""
    return f"source_{i}"


def new_sacc(nz, metadata=None):
    """Create a Sacc with the survey's NZ (and PSF) tracers.

    Parameters
    ----------
    nz : dict or sequence
        Redshift distributions, one per source bin. Either a mapping
        ``{i: (z, nz)}`` keyed by 0-based bin index, or a sequence of
        ``(z, nz)`` array pairs (bin index = position). Tracers are named
        ``source_{i}``.
    metadata : dict, optional
        Key/value pairs stored on ``s.metadata``.

    Returns
    -------
    sacc.Sacc
        Sacc holding the ``source_{i}`` NZ tracers and the ``psf_stars``
        Misc tracer (needed by ρ/τ diagnostics).
    """
    items = nz.items() if isinstance(nz, dict) else enumerate(nz)
    s = sacc.Sacc()
    for i, (z, nz_i) in items:
        s.add_tracer("NZ", source_name(i), np.asarray(z), np.asarray(nz_i))
    s.add_tracer("Misc", PSF_TRACER)
    for key, value in (metadata or {}).items():
        s.metadata[key] = value
    return s


def _pair(bins):
    """Resolve a ``(i, j)`` bin pair to the ``(source_i, source_j)`` names.

    The pair is normalised to ``i <= j``: shear-shear statistics are symmetric
    in the tracer pair, and SACC stores each pair under one ordering, so
    ``(1, 0)`` must address the same points as ``(0, 1)``.
    """
    i, j = sorted(bins)
    return (source_name(i), source_name(j))


def _check_ascending(name, values):
    """Require ``values`` to be strictly ascending; else raise ValueError.

    Insertion order is the covariance (and bandpower-window) order, and readers
    return points in insertion order, so an out-of-order grid would silently
    desynchronise a data vector from its covariance. Enforce monotonicity at
    write time instead.
    """
    values = np.asarray(values)
    if not np.all(np.diff(values) > 0):
        raise ValueError(
            f"{name} must be strictly ascending (insertion order is the "
            f"covariance order); got {values.tolist()}"
        )


def add_xi(
    s,
    bins,
    theta,
    xip,
    xim,
    *,
    grid,
    theta_nom=None,
    npairs=None,
    weight=None,
):
    """Add a real-space shear 2PCF (ξ+ then ξ−) for one tracer pair.

    Parameters
    ----------
    s : sacc.Sacc
        Target, mutated in place.
    bins : tuple of int
        Source bin pair ``(i, j)``.
    theta : array_like
        Angular separations (arcmin) — TreeCorr ``meanr``.
    xip, xim : array_like
        ξ+ and ξ− at ``theta``.
    grid : {'coarse', 'fine'}
        Distinguishes the analysis grid from the fine integration grid;
        stored as the ``grid`` tag on every point.
    theta_nom : array_like, optional
        Nominal bin centres — TreeCorr ``rnom`` — stored as ``theta_nom``.
    npairs, weight : array_like, optional
        TreeCorr pair counts and weights, stored per point.
    """
    _check_ascending("theta", theta)
    tracers = _pair(bins)
    for dtype, xi in ((XI_PLUS, xip), (XI_MINUS, xim)):
        for n, th in enumerate(theta):
            tags = {"theta": float(th), "grid": grid}
            if theta_nom is not None:
                tags["theta_nom"] = float(theta_nom[n])
            if npairs is not None:
                tags["npairs"] = float(npairs[n])
            if weight is not None:
                tags["weight"] = float(weight[n])
            s.add_data_point(dtype, tracers, float(xi[n]), **tags)


def add_pseudo_cl(
    s,
    bins,
    ell_eff,
    cl_ee,
    cl_bb,
    cl_eb,
    *,
    window_ells,
    window_weights,
):
    """Add pseudo-Cℓ (EE, BB, EB) with a shared bandpower window.

    Parameters
    ----------
    s : sacc.Sacc
        Target, mutated in place.
    bins : tuple of int
        Source bin pair ``(i, j)``.
    ell_eff : array_like
        Effective multipole of each bandpower.
    cl_ee, cl_bb, cl_eb : array_like
        EE, BB and EB bandpowers at ``ell_eff``.
    window_ells : array_like
        Multipoles spanned by the bandpower window matrix (shape ``(nell,)``).
    window_weights : array_like
        Window matrix ``W`` of shape ``(nell, nbp)`` — one column per
        bandpower — from NaMaster ``get_bandpower_windows``. One
        ``sacc.BandpowerWindow`` is built and shared across EE/BB/EB.
    """
    _check_ascending("ell_eff", ell_eff)
    tracers = _pair(bins)
    window = sacc.BandpowerWindow(np.asarray(window_ells), np.asarray(window_weights))
    for dtype, cl in ((CL_EE, cl_ee), (CL_BB, cl_bb), (CL_EB, cl_eb)):
        s.add_ell_cl(
            dtype, *tracers, np.asarray(ell_eff), np.asarray(cl), window=window
        )


def add_cosebis(s, bins, En, Bn, scale_cut):
    """Add COSEBIs (all Eₙ then all Bₙ) for one scale cut.

    Parameters
    ----------
    s : sacc.Sacc
        Target, mutated in place.
    bins : tuple of int
        Source bin pair ``(i, j)``.
    En, Bn : array_like
        E- and B-mode COSEBI amplitudes, one per logarithmic mode ``n``
        (1-based). The ``[En; Bn]`` layout matches the COSEBI covariance.
    scale_cut : tuple of float
        ``(theta_min, theta_max)`` in arcmin, stored on every point as the
        ``theta_min``/``theta_max`` tags; multiple cuts coexist in one file,
        told apart by these tags.
    """
    tracers = _pair(bins)
    theta_min, theta_max = scale_cut
    for dtype, modes in ((COSEBI_EE, En), (COSEBI_BB, Bn)):
        for n, value in enumerate(modes, start=1):
            s.add_data_point(
                dtype,
                tracers,
                float(value),
                n=n,
                theta_min=float(theta_min),
                theta_max=float(theta_max),
            )


def add_pure_eb(s, bins, theta, xip_E, xim_E, xip_B, xim_B, xip_amb, xim_amb):
    """Add pure E/B-mode correlation functions for one tracer pair.

    Six blocks are inserted in ``PURE_KEYS`` order (xip_E, xim_E, xip_B,
    xim_B, xip_amb, xim_amb), matching ``b_modes._EB_KEYS`` and the pure-EB
    covariance layout.

    Parameters
    ----------
    s : sacc.Sacc
        Target, mutated in place.
    bins : tuple of int
        Source bin pair ``(i, j)``.
    theta : array_like
        Angular separations (arcmin), shared by all six blocks.
    xip_E, xim_E, xip_B, xim_B, xip_amb, xim_amb : array_like
        The six pure E/B / ambiguous mode arrays at ``theta``.
    """
    _check_ascending("theta", theta)
    tracers = _pair(bins)
    values = {
        "xip_E": xip_E,
        "xim_E": xim_E,
        "xip_B": xip_B,
        "xim_B": xim_B,
        "xip_amb": xip_amb,
        "xim_amb": xim_amb,
    }
    for key in PURE_KEYS:
        dtype, arr = PURE_TYPES[key], values[key]
        for n, th in enumerate(theta):
            s.add_data_point(dtype, tracers, float(arr[n]), theta=float(th))


def add_rho(s, k, theta, rho_p, rho_m):
    """Add a ρ_k PSF statistic (ρ+ then ρ−) on the ``psf_stars`` tracer.

    Parameters
    ----------
    s : sacc.Sacc
        Target, mutated in place.
    k : int
        ρ index (0…5).
    theta : array_like
        Angular separations (arcmin).
    rho_p, rho_m : array_like
        ρ_k+ and ρ_k− at ``theta``.
    """
    _check_ascending("theta", theta)
    tracers = (PSF_TRACER, PSF_TRACER)
    for dtype, arr in ((RHO_PLUS.format(k=k), rho_p), (RHO_MINUS.format(k=k), rho_m)):
        for n, th in enumerate(theta):
            s.add_data_point(dtype, tracers, float(arr[n]), theta=float(th))


def add_tau(s, bins, k, theta, tau_p, tau_m):
    """Add a τ_k PSF-leakage statistic (τ+ then τ−).

    Parameters
    ----------
    s : sacc.Sacc
        Target, mutated in place.
    bins : tuple of int
        Source bin ``i`` and PSF; the τ tracers are ``(source_i, psf_stars)``.
        Only ``bins[0]`` is used.
    k : int
        τ index (0, 2 or 5).
    theta : array_like
        Angular separations (arcmin).
    tau_p, tau_m : array_like
        τ_k+ and τ_k− at ``theta``.
    """
    _check_ascending("theta", theta)
    tracers = (source_name(bins[0]), PSF_TRACER)
    for dtype, arr in ((TAU_PLUS.format(k=k), tau_p), (TAU_MINUS.format(k=k), tau_m)):
        for n, th in enumerate(theta):
            s.add_data_point(dtype, tracers, float(arr[n]), theta=float(th))


def assemble_covariance(s, blocks):
    """Assemble a block-diagonal ``FullCovariance`` from per-statistic blocks.

    Each block is validated against the current insertion order: its indices
    must be contiguous and ascending, the blocks must tile ``0…len(s.mean)``
    exactly (no gap, no overlap), and each block must be square with a size
    matching its index span. Any violation raises ``ValueError`` naming the
    mismatch. Cross-blocks are left zero.

    Parameters
    ----------
    s : sacc.Sacc
        Target, mutated in place via ``add_covariance``.
    blocks : sequence
        Ordered ``(selector, cov)`` pairs (or a mapping of the same). Each
        ``selector`` is either an index array, or a ``(data_type, tracers)``
        / ``(data_type, tracers, tags)`` tuple resolved through
        ``s.indices``; ``cov`` is the block's dense covariance.

    Returns
    -------
    sacc.Sacc
        ``s``, with the assembled ``FullCovariance`` attached.
    """
    items = blocks.items() if isinstance(blocks, dict) else blocks
    ntot = len(s.mean)
    full = np.zeros((ntot, ntot))
    cursor = 0
    for selector, cov in items:
        idx = _resolve_indices(s, selector)
        cov = np.asarray(cov)
        if not np.array_equal(idx, np.arange(idx[0], idx[0] + len(idx))):
            raise ValueError(
                f"covariance block {selector!r} resolves to non-contiguous "
                f"or non-ascending indices {idx.tolist()}"
            )
        if idx[0] != cursor:
            raise ValueError(
                f"covariance block {selector!r} starts at index {idx[0]} but "
                f"the previous blocks cover through {cursor} — blocks must tile "
                "the data vector with no gap or overlap"
            )
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError(
                f"covariance block {selector!r} must be square; got shape {cov.shape}"
            )
        if cov.shape[0] != len(idx):
            raise ValueError(
                f"covariance block {selector!r} has size {cov.shape[0]} but "
                f"spans {len(idx)} data points"
            )
        full[np.ix_(idx, idx)] = cov
        cursor = idx[-1] + 1
    if cursor != ntot:
        raise ValueError(
            f"covariance blocks cover {cursor} of {ntot} data points — the "
            "blocks must tile the whole data vector"
        )
    s.add_covariance(full)
    return s


def _resolve_indices(s, selector):
    """Resolve a covariance-block selector to a sorted index array."""
    if isinstance(selector, (np.ndarray, list, tuple, range)) and not (
        len(selector) in (2, 3) and isinstance(selector[0], str)
    ):
        return np.asarray(selector, dtype=int)
    data_type, tracers = selector[0], selector[1]
    tags = selector[2] if len(selector) == 3 else {}
    return np.asarray(s.indices(data_type, tuple(tracers), **tags), dtype=int)


def add_diagonal_covariance(s, variances):
    """Attach a ``DiagonalCovariance`` from a 1-D variance array.

    The 1-D array is passed straight to ``add_covariance`` (never
    ``np.diag``), which is what makes SACC store a ``DiagonalCovariance``.

    Parameters
    ----------
    s : sacc.Sacc
        Target, mutated in place.
    variances : array_like
        Per-point variances, ``len == len(s.mean)``.

    Returns
    -------
    sacc.Sacc
        ``s``, with the ``DiagonalCovariance`` attached.
    """
    s.add_covariance(np.asarray(variances))
    return s


def get_nz(s, i):
    """Return ``(z, nz)`` for source bin ``i``."""
    tracer = s.tracers[source_name(i)]
    return tracer.z, tracer.nz


def get_xi(s, bins, *, grid):
    """Return ``(theta, xip, xim)`` for one tracer pair and grid."""
    tracers = _pair(bins)
    return (
        _tag(s, XI_PLUS, tracers, "theta", grid=grid),
        _mean(s, XI_PLUS, tracers, grid=grid),
        _mean(s, XI_MINUS, tracers, grid=grid),
    )


def get_pseudo_cl(s, bins):
    """Return ``(ell_eff, cl_ee, cl_bb, cl_eb, window)`` for one tracer pair.

    ``window`` is the shared ``sacc.BandpowerWindow`` recovered via
    ``get_bandpower_windows``; its columns are in the same insertion order as
    the returned ``ell_eff``/``cl`` arrays, so window column ``j`` corresponds
    to ``ell_eff[j]``.
    """
    tracers = _pair(bins)
    window = s.get_bandpower_windows(s.indices(CL_EE, tracers))
    return (
        _tag(s, CL_EE, tracers, "ell"),
        _mean(s, CL_EE, tracers),
        _mean(s, CL_BB, tracers),
        _mean(s, CL_EB, tracers),
        window,
    )


def get_cosebis(s, bins, scale_cut=None):
    """Return ``(n, En, Bn)`` for one tracer pair.

    Parameters
    ----------
    scale_cut : tuple of float, optional
        ``(theta_min, theta_max)`` to select when several cuts share the file.
    """
    tracers = _pair(bins)
    tags = (
        {"theta_min": float(scale_cut[0]), "theta_max": float(scale_cut[1])}
        if scale_cut is not None
        else {}
    )
    modes = _tag(s, COSEBI_EE, tracers, "n", **tags)
    return (
        modes.astype(int),
        _mean(s, COSEBI_EE, tracers, **tags),
        _mean(s, COSEBI_BB, tracers, **tags),
    )


def get_pure_eb(s, bins):
    """Return ``(theta, {key: array})`` for the six pure-EB blocks.

    The dict is keyed by ``PURE_KEYS`` (xip_E, xim_E, …).
    """
    tracers = _pair(bins)
    theta = _tag(s, PURE_TYPES["xip_E"], tracers, "theta")
    arrays = {key: _mean(s, PURE_TYPES[key], tracers) for key in PURE_KEYS}
    return theta, arrays


def get_rho(s, k):
    """Return ``(theta, rho_p, rho_m)`` for ρ index ``k``."""
    tracers = (PSF_TRACER, PSF_TRACER)
    dt_p, dt_m = RHO_PLUS.format(k=k), RHO_MINUS.format(k=k)
    return (
        _tag(s, dt_p, tracers, "theta"),
        _mean(s, dt_p, tracers),
        _mean(s, dt_m, tracers),
    )


def get_tau(s, bins, k):
    """Return ``(theta, tau_p, tau_m)`` for τ index ``k`` and source bin."""
    tracers = (source_name(bins[0]), PSF_TRACER)
    dt_p, dt_m = TAU_PLUS.format(k=k), TAU_MINUS.format(k=k)
    return (
        _tag(s, dt_p, tracers, "theta"),
        _mean(s, dt_p, tracers),
        _mean(s, dt_m, tracers),
    )


def _mean(s, data_type, tracers, **tag_filters):
    """Mean values for a selection, in ``s.indices`` (insertion) order.

    Never re-sort: insertion order is the covariance and bandpower-window
    order, so returning in ``s.indices`` order keeps every reader aligned with
    the covariance for any file (and ascending for canonically-written files,
    which the writers enforce).
    """
    return s.mean[s.indices(data_type, tracers, **tag_filters)]


def _tag(s, data_type, tracers, tag, **tag_filters):
    """Values of ``tag`` for a selection, in insertion order."""
    idx = s.indices(data_type, tracers, **tag_filters)
    return np.array([s.data[i].tags[tag] for i in idx])


def extract(s, data_type=None, tracers=None, **tag_filters):
    """Extract a sub-Sacc (points + aligned covariance sub-block).

    A copy is made and everything *not* matching the selection is removed, so
    the covariance sub-block comes out correctly aligned and the original is
    untouched.

    Parameters
    ----------
    s : sacc.Sacc
        Source, left unmodified.
    data_type : str, optional
        Data type to keep.
    tracers : tuple, optional
        Tracer pair to keep, as SACC tracer **names** (e.g.
        ``("source_0", "source_0")`` or ``("source_0", "psf_stars")``) — *not*
        integer bin indices. This differs deliberately from the ``add_*`` /
        ``get_*`` interface, whose ``bins`` argument takes integer pairs:
        ``extract`` is the generic selection escape hatch, mirroring
        ``Sacc.keep_selection`` and addressing non-source tracers uniformly.
    **tag_filters
        Tag filters (plain kwargs, e.g. ``grid='fine'``).

    Returns
    -------
    sacc.Sacc
        New Sacc holding only the selected points.
    """
    sub = s.copy()
    selection = {}
    if tracers is not None:
        selection["tracers"] = tuple(tracers)
    selection.update(tag_filters)
    sub.keep_selection(data_type, **selection)
    return sub


def save(s, path):
    """Write ``s`` to ``path`` (FITS), overwriting any existing file."""
    s.save_fits(path, overwrite=True)


def load(path):
    """Load a Sacc from ``path`` (FITS)."""
    return sacc.Sacc.load_fits(path)
