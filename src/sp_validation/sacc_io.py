"""SACC_IO.

:Name: sacc_io.py

:Description: Read/write the standard SACC data-product layout for the
              weak-lensing validation package. One file describes each
              catalogue version:

              - ``{version}.sacc`` — NZ tracers, reporting-grid ξ±, pseudo-Cℓ
                (EE/BB/EB) with bandpower windows, COSEBIs, pure E/B, ρ/τ PSF
                diagnostics, and the fine ξ± integration input for
                COSEBIs / pure-EB (``grid='integration'`` tagged points). The
                covariance is assembled block-diagonally from the
                per-statistic covariances (zero cross-blocks, never
                materialized): the analysis blocks first, then a dense
                per-pair integration-ξ block (the analytic integration-binning
                covariance when it exists — it feeds derived-statistic error
                propagation — or the TreeCorr ``varxip``/``varxim`` diagonal as
                degraded fallback). ``assemble_covariance`` hands sacc a list
                of blocks, which stores a ``BlockDiagonalCovariance`` (one
                FITS table per block, Σ block² on disk rather than a dense
                N²) — cost scales with the integration grid's size, a
                parameter set by the caller, not baked into the layout.

              Insertion order is load-bearing. A Sacc is a flat list of data
              points in the order ``add_data_point`` was called, and row/column
              ``i`` of the covariance refers to the ``i``-th inserted point —
              there is no other linkage between a point and its covariance
              entry. SACC preserves that order bitwise through FITS save/load,
              so writers define the covariance layout by their insertion
              sequence. Writers below insert in the canonical order — ξ+ then
              ξ−, Cℓ (ee, bb, eb), COSEBIs (all Eₙ then all Bₙ), pure E/B
              (xip_E, xim_E, xip_B, xim_B, xip_amb, xim_amb — matching
              ``b_modes._EB_KEYS``), ρ, then τ — but readers never assume
              global order: they resolve indices through
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

Optionality: a file's contents are flexible about which components of
              a statistic it actually has. ``add_pseudo_cl`` requires EE (the
              bandpower window's reference series) but BB and EB are optional
              — an analysis that never computed EB simply omits it.
              ``add_cosebis`` requires Eₙ but Bₙ is optional. ``add_pure_eb``
              requires xip_E/xim_E but the B and ambiguous-mode blocks are
              each optional, independently (B and amb are unrelated
              computations). Everything else a writer takes — θ/ℓ grids,
              tracer/bin identifiers, the value array for a component you ARE
              adding — is structurally necessary and has no default;
              supplying it partially would desynchronise the covariance
              layout, so it is refused rather than degraded. Readers mirror
              this split: a composite reader (``get_pseudo_cl``,
              ``get_cosebis``, ``get_pure_eb``) returns ``None`` (or omits the
              key) for a component the file doesn't carry, but a selection
              naming that component explicitly (``s.indices``, ``_mean``,
              ``extract``) still fails loud on no match — silence is reserved
              for "this file doesn't have that optional piece", never for
              "you asked for something specific and it isn't there".
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
# PURE_TYPES key order is the insertion order of the six pure-EB blocks —
# matches b_modes._EB_KEYS, whose order is the [xip_E; xim_E; xip_B; xim_B;
# xip_amb; xim_amb] layout of the treecorr/MC pure-EB covariance
# (b_modes.calculate_eb_statistics, ~L392).
PURE_KEYS = tuple(PURE_TYPES)

RHO_PLUS = "psf_rho{k}_xi_plus"
RHO_MINUS = "psf_rho{k}_xi_minus"
TAU_PLUS = "galaxyPsf_tau{k}_xi_plus"
TAU_MINUS = "galaxyPsf_tau{k}_xi_minus"


def source_name(i):
    """SACC tracer name for source redshift bin ``i`` (0-based).

    Kept as the single definition of the ``source_{i}`` naming contract —
    external consumers address tracers through it rather than the f-string.
    """
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
    # The PSF star sample sits alongside the source bins because a Sacc has a
    # single tracer namespace — every data point references tracers from one
    # flat list. This is bookkeeping, not physics: psf_stars is a Misc tracer
    # (no n(z)) that exists only so ρ/τ points have something to reference.
    s.add_tracer("Misc", PSF_TRACER)
    for key, value in (metadata or {}).items():
        s.metadata[key] = value
    return s


def _pair(bins):
    """Resolve a ``(i, j)`` bin pair to the ``(source_i, source_j)`` names.

    The pair is normalised to ``i <= j``: shear-shear statistics are symmetric
    in the tracer pair, and SACC stores each pair under one ordering, so
    ``(1, 0)`` must address the same points as ``(0, 1)``. Kept as a helper:
    every ``add_*``/``get_*`` function relies on this normalisation.
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


def _add_theta_series(s, dtype, tracers, theta, values, **tags):
    """Insert one theta-tagged series, one point per (theta, value) pair."""
    for th, value in zip(theta, values):
        s.add_data_point(dtype, tracers, float(value), theta=float(th), **tags)


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
    grid : {'reporting', 'integration'}
        Stored as the ``grid`` tag on every point. The reporting (analysis)
        ξ± and the fine COSEBIs/pure-EB integration input share the same
        data type and tracer pair, so this tag is the only thing that
        disambiguates them in the file.
    theta_nom : array_like, optional
        Nominal bin centres — TreeCorr ``rnom`` — stored as ``theta_nom``.
    npairs, weight : array_like, optional
        TreeCorr pair counts and weights, stored per point.
    """
    _check_ascending("theta", theta)
    tracers = _pair(bins)
    optional = {"theta_nom": theta_nom, "npairs": npairs, "weight": weight}
    extras = {key: arr for key, arr in optional.items() if arr is not None}
    for dtype, xi in ((XI_PLUS, xip), (XI_MINUS, xim)):
        for n, th in enumerate(theta):
            tags = {
                "theta": float(th),
                "grid": grid,
                **{key: float(arr[n]) for key, arr in extras.items()},
            }
            s.add_data_point(dtype, tracers, float(xi[n]), **tags)


def add_pseudo_cl(
    s,
    bins,
    ell_eff,
    cl_ee,
    cl_bb=None,
    cl_eb=None,
    *,
    window_ells,
    window_weights,
    grid="reporting",
):
    """Add pseudo-Cℓ EE (required) plus whichever of BB/EB were computed.

    Parameters
    ----------
    s : sacc.Sacc
        Target, mutated in place.
    bins : tuple of int
        Source bin pair ``(i, j)``.
    ell_eff : array_like
        Effective multipole of each bandpower.
    cl_ee : array_like
        EE bandpowers at ``ell_eff``.
    cl_bb, cl_eb : array_like, optional
        BB and/or EB bandpowers at ``ell_eff``. Each defaults to ``None`` and
        is then simply not written — EB in particular is often not computed
        at all.
    window_ells : array_like
        Multipoles spanned by the bandpower window matrix (shape ``(nell,)``).
    window_weights : array_like
        Window matrix ``W`` of shape ``(nell, nbp)`` — one column per
        bandpower — from NaMaster ``get_bandpower_windows``. One
        ``sacc.BandpowerWindow`` is built and shared across every component
        written.
    grid : str, optional
        Stored as the ``grid`` tag on every point (default ``'reporting'``),
        joining ``merge``'s ℓ-consistency group; variant ℓ binnings belong
        under different tag values.
    """
    _check_ascending("ell_eff", ell_eff)
    tracers = _pair(bins)
    window = sacc.BandpowerWindow(np.asarray(window_ells), np.asarray(window_weights))
    # add_ell_cl accepts no extra tags, so inline its per-point insertion
    # (ell + shared window + window_ind column index) plus the grid tag.
    components = [(CL_EE, cl_ee)]
    components += [
        (dtype, cl) for dtype, cl in ((CL_BB, cl_bb), (CL_EB, cl_eb)) if cl is not None
    ]
    for dtype, cl in components:
        for n, (ell, value) in enumerate(zip(ell_eff, cl)):
            s.add_data_point(
                dtype,
                tracers,
                float(value),
                ell=float(ell),
                window=window,
                window_ind=n,
                grid=grid,
            )


def add_cosebis(s, bins, En, scale_cut, Bn=None):
    """Add COSEBIs Eₙ (required) and Bₙ (optional) for one scale cut.

    Parameters
    ----------
    s : sacc.Sacc
        Target, mutated in place.
    bins : tuple of int
        Source bin pair ``(i, j)``.
    En : array_like
        E-mode COSEBI amplitudes, one per logarithmic mode ``n`` (1-based).
    scale_cut : tuple of float
        ``(theta_min, theta_max)`` in arcmin, stored on every point as the
        ``theta_min``/``theta_max`` tags; multiple cuts coexist in one file,
        told apart by these tags.
    Bn : array_like, optional
        B-mode COSEBI amplitudes at the same ``n``. Defaults to ``None`` and
        is then simply not written. The ``[En; Bn]`` layout, when both are
        present, matches the COSEBI covariance.
    """
    tracers = _pair(bins)
    theta_min, theta_max = scale_cut
    components = [(COSEBI_EE, En)]
    if Bn is not None:
        components.append((COSEBI_BB, Bn))
    for dtype, modes in components:
        for n, value in enumerate(modes, start=1):
            s.add_data_point(
                dtype,
                tracers,
                float(value),
                n=n,
                theta_min=float(theta_min),
                theta_max=float(theta_max),
            )


def add_pure_eb(
    s,
    bins,
    theta,
    xip_E,
    xim_E,
    xip_B=None,
    xim_B=None,
    xip_amb=None,
    xim_amb=None,
    *,
    grid="reporting",
):
    """Add pure E-mode (required) plus whichever of B/ambiguous were computed.

    Blocks are inserted in ``PURE_KEYS`` order (xip_E, xim_E, xip_B, xim_B,
    xip_amb, xim_amb), matching ``b_modes._EB_KEYS`` and the pure-EB
    covariance layout — whichever subset is present.

    Parameters
    ----------
    s : sacc.Sacc
        Target, mutated in place.
    bins : tuple of int
        Source bin pair ``(i, j)``.
    theta : array_like
        Angular separations (arcmin), shared by every block written.
    xip_E, xim_E : array_like
        The pure E-mode correlation functions at ``theta``.
    xip_B, xim_B : array_like, optional
        The pure B-mode correlation functions. Both default to ``None``; the
        pair is written together or not at all — supply both or neither.
    xip_amb, xim_amb : array_like, optional
        The ambiguous-mode correlation functions. Both default to ``None``;
        same both-or-neither rule as B.
    grid : str, optional
        Stored as the ``grid`` tag on every point (default ``'reporting'``),
        joining ξ's theta-consistency group in ``merge``'s guard.
    """
    _check_ascending("theta", theta)
    tracers = _pair(bins)
    pairs = {
        "B": (xip_B, xim_B),
        "amb": (xip_amb, xim_amb),
    }
    for label, (p, m) in pairs.items():
        if (p is None) != (m is None):
            raise ValueError(
                f"add_pure_eb: xip_{label} and xim_{label} must both be "
                "given or both omitted"
            )
    values = {
        "xip_E": xip_E,
        "xim_E": xim_E,
        "xip_B": xip_B,
        "xim_B": xim_B,
        "xip_amb": xip_amb,
        "xim_amb": xim_amb,
    }
    for key in PURE_KEYS:
        arr = values[key]
        if arr is not None:
            _add_theta_series(s, PURE_TYPES[key], tracers, theta, arr, grid=grid)


def add_rho(s, k, theta, rho_p, rho_m, *, grid="reporting"):
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
    grid : str, optional
        Stored as the ``grid`` tag on every point (default ``'reporting'``),
        joining ξ's theta-consistency group in ``merge``'s guard.
    """
    _check_ascending("theta", theta)
    tracers = (PSF_TRACER, PSF_TRACER)
    _add_theta_series(s, RHO_PLUS.format(k=k), tracers, theta, rho_p, grid=grid)
    _add_theta_series(s, RHO_MINUS.format(k=k), tracers, theta, rho_m, grid=grid)


def add_tau(s, bins, k, theta, tau_p, tau_m, *, grid="reporting"):
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
    grid : str, optional
        Stored as the ``grid`` tag on every point (default ``'reporting'``),
        joining ξ's theta-consistency group in ``merge``'s guard.
    """
    _check_ascending("theta", theta)
    tracers = (source_name(bins[0]), PSF_TRACER)
    _add_theta_series(s, TAU_PLUS.format(k=k), tracers, theta, tau_p, grid=grid)
    _add_theta_series(s, TAU_MINUS.format(k=k), tracers, theta, tau_m, grid=grid)


def assemble_covariance(s, blocks):
    """Assemble a ``BlockDiagonalCovariance`` from per-statistic blocks.

    Each block is validated against the current insertion order: its indices
    must be contiguous and ascending, the blocks must tile ``0…len(s.mean)``
    exactly (no gap, no overlap), and each block must be square with a size
    matching its index span. Any violation raises ``ValueError`` naming the
    mismatch. Cross-blocks are zero and implicit — never materialized —
    because the blocks are passed to ``add_covariance`` as a list, which
    ``sacc.BaseCovariance.make`` turns into a ``BlockDiagonalCovariance``
    (one FITS table per block, Σ block² on disk rather than a dense N² file).

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
        ``s``, with the assembled ``BlockDiagonalCovariance`` attached.
    """
    items = blocks.items() if isinstance(blocks, dict) else blocks
    ntot = len(s.mean)
    ordered_blocks = []
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
        ordered_blocks.append(cov)
        cursor = idx[-1] + 1
    if cursor != ntot:
        raise ValueError(
            f"covariance blocks cover {cursor} of {ntot} data points — the "
            "blocks must tile the whole data vector"
        )
    s.add_covariance(ordered_blocks)
    return s


def _resolve_indices(s, selector):
    """Resolve a covariance-block selector to a sorted index array."""
    if isinstance(selector, (np.ndarray, list, tuple, range)) and not (
        len(selector) in (2, 3) and isinstance(selector[0], str)
    ):
        return np.asarray(selector, dtype=int)
    data_type, tracers = selector[0], selector[1]
    tags = selector[2] if len(selector) == 3 else {}
    return _indices(s, data_type, tuple(tracers), **tags)


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


def _get_pm(s, dtype_p, dtype_m, tracers, **tags):
    """Return ``(theta, plus, minus)`` for a +/− data-type pair."""
    return (
        _tag(s, dtype_p, tracers, "theta", **tags),
        _mean(s, dtype_p, tracers, **tags),
        _mean(s, dtype_m, tracers, **tags),
    )


def get_xi(s, bins, *, grid):
    """Return ``(theta, xip, xim)`` for one tracer pair and grid."""
    return _get_pm(s, XI_PLUS, XI_MINUS, _pair(bins), grid=grid)


def get_pseudo_cl(s, bins):
    """Return ``(ell_eff, cl_ee, cl_bb, cl_eb, window)`` for one tracer pair.

    ``cl_bb``/``cl_eb`` come back ``None`` if the file doesn't carry that
    component (``add_pseudo_cl`` makes both optional). ``window`` is the
    shared ``sacc.BandpowerWindow`` recovered via ``get_bandpower_windows``;
    its columns are in the same insertion order as the returned
    ``ell_eff``/``cl`` arrays, so window column ``j`` corresponds to
    ``ell_eff[j]``.
    """
    tracers = _pair(bins)
    window = s.get_bandpower_windows(s.indices(CL_EE, tracers))
    return (
        _tag(s, CL_EE, tracers, "ell"),
        _mean(s, CL_EE, tracers),
        _mean_optional(s, CL_BB, tracers),
        _mean_optional(s, CL_EB, tracers),
        window,
    )


def get_cosebis(s, bins, scale_cut=None):
    """Return ``(n, En, Bn)`` for one tracer pair.

    ``Bn`` comes back ``None`` if the file doesn't carry it (``add_cosebis``
    makes it optional).

    Parameters
    ----------
    scale_cut : tuple of float, optional
        ``(theta_min, theta_max)`` to select when several cuts share the file.
    """
    tracers = _pair(bins)
    if scale_cut is None:
        cuts = {
            (s.data[i].tags["theta_min"], s.data[i].tags["theta_max"])
            for i in _indices(s, COSEBI_EE, tracers)
        }
        if len(cuts) > 1:
            raise ValueError(
                f"several COSEBIs scale cuts share the file ({sorted(cuts)}) "
                "— pass scale_cut=(theta_min, theta_max) to pick one"
            )
    tags = dict(zip(("theta_min", "theta_max"), map(float, scale_cut or ())))
    modes = _tag(s, COSEBI_EE, tracers, "n", **tags)
    return (
        modes.astype(int),
        _mean(s, COSEBI_EE, tracers, **tags),
        _mean_optional(s, COSEBI_BB, tracers, **tags),
    )


def get_pure_eb(s, bins):
    """Return ``(theta, {key: array})`` for whichever pure-EB blocks exist.

    xip_E/xim_E are always present (``add_pure_eb`` requires them); the dict
    holds whichever of the B and ambiguous-mode keys (out of ``PURE_KEYS``:
    xip_E, xim_E, xip_B, xim_B, xip_amb, xim_amb) the file actually carries —
    a key absent from the file is simply absent from the dict, not mapped to
    ``None``.
    """
    tracers = _pair(bins)
    theta = _tag(s, PURE_TYPES["xip_E"], tracers, "theta")
    arrays = {}
    for key in PURE_KEYS:
        values = _mean_optional(s, PURE_TYPES[key], tracers)
        if values is not None:
            arrays[key] = values
    return theta, arrays


def get_rho(s, k):
    """Return ``(theta, rho_p, rho_m)`` for ρ index ``k``."""
    tracers = (PSF_TRACER, PSF_TRACER)
    return _get_pm(s, RHO_PLUS.format(k=k), RHO_MINUS.format(k=k), tracers)


def get_tau(s, bins, k):
    """Return ``(theta, tau_p, tau_m)`` for τ index ``k`` and source bin."""
    tracers = (source_name(bins[0]), PSF_TRACER)
    return _get_pm(s, TAU_PLUS.format(k=k), TAU_MINUS.format(k=k), tracers)


def _indices(s, data_type, tracers, **tag_filters):
    """``Sacc.indices`` that fails loud instead of selecting nothing.

    ``Sacc.indices`` returns an *empty array* (warning only) when a selection
    matches no point — e.g. a typo'd tag value, or a float tag filter that is
    not bitwise-identical to the stored one. Every reader here funnels through
    this guard so an unmatched selection raises instead of propagating empty
    arrays downstream.
    """
    idx = np.asarray(s.indices(data_type, tracers, **tag_filters), dtype=int)
    if len(idx) == 0:
        raise ValueError(
            f"selection matched no points: ({data_type}, {tracers}, "
            f"{tag_filters}) — note float tags match by exact equality"
        )
    return idx


def _mean(s, data_type, tracers, **tag_filters):
    """Mean values for a selection, in ``s.indices`` (insertion) order.

    Never re-sort: insertion order is the covariance and bandpower-window
    order, so returning in ``s.indices`` order keeps every reader aligned with
    the covariance for any file (and ascending for canonically-written files,
    which the writers enforce).
    """
    return s.mean[_indices(s, data_type, tracers, **tag_filters)]


def _tag(s, data_type, tracers, tag, **tag_filters):
    """Values of ``tag`` for a selection, in insertion order."""
    idx = _indices(s, data_type, tracers, **tag_filters)
    return np.array([s.data[i].tags[tag] for i in idx])


def _mean_optional(s, data_type, tracers, **tag_filters):
    """Mean values for a selection, or ``None`` if the file has none.

    Used by composite readers (``get_pseudo_cl``, ``get_cosebis``,
    ``get_pure_eb``) for the components a writer made optional (BB/EB,
    COSEBI Bₙ, pure B/ambiguous): absence is a legitimate "this file doesn't
    have that piece", not a typo to fail loud on — unlike ``_mean``/
    ``_indices``, used for selections that name a component explicitly.
    """
    idx = np.asarray(s.indices(data_type, tracers, **tag_filters), dtype=int)
    return s.mean[idx] if len(idx) else None


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
        Tag filters (plain kwargs, e.g. ``grid='integration'``).

    Returns
    -------
    sacc.Sacc
        New Sacc holding only the selected points.
    """
    sub = s.copy()
    tracer_filter = {"tracers": tuple(tracers)} if tracers is not None else {}
    sub.keep_selection(data_type, **tracer_filter, **tag_filters)
    if len(sub.mean) == 0:
        raise ValueError(
            f"extract selected no points: ({data_type}, {tracers}, {tag_filters})"
        )
    return sub


def merge(saccs):
    """Merge several per-statistic Sacc objects into one file's worth.

    A thin wrapper around ``sacc.concatenate_data_sets``: data points
    concatenate in input order, tracers shared by several inputs (the
    ``source_i`` NZ tracers, ``psf_stars``) are stored once, and the
    covariance combines block-diagonally in the same order — the library
    requires either **all** inputs to carry a covariance or **none**, and
    raises otherwise (cross-statistic covariance assembly beyond
    block-diagonal is out of scope here; see ``assemble_covariance``).

    Metadata must be consistent: keys present in several inputs must carry
    equal values (a ``type: data`` file cannot merge with a ``type: mock``
    file), and the union lands on the result. This deliberately replaces the
    library's clash behaviour, which mangles clashing keys by appending
    labels.

    Grid consistency follows tagging semantics: the ``grid`` tag declares
    which binning a set of points lives on, so all same-length theta (or
    ell) arrays under one tag value must be bitwise identical — sacc itself
    never validates angles across data types/tracers, and a grid that
    differs only at floating-point level chokes CosmoSIS downstream instead
    of failing loud here. Different lengths within a tag group pass
    (scale-cut subsets are legitimate); grids under different tag values
    are unconstrained (``reporting`` vs ``integration`` differ by design);
    θ and ℓ are separate domains, each checked against itself only. ℓ
    series sharing a bitwise-equal grid must also share the bandpower
    window (series without windows skip that check).

    Parameters
    ----------
    saccs : sequence of sacc.Sacc
        The per-statistic data sets, in the insertion order the merged file
        should have. Inputs are left unmodified.

    Returns
    -------
    sacc.Sacc
        The merged data set.

    Raises
    ------
    ValueError
        If metadata conflicts, a shared tracer differs across inputs, two
        same-length theta or ell arrays under the same ``grid`` tag value
        are not bitwise identical, or two ℓ series sharing a grid carry
        different bandpower windows.
    """
    saccs = list(saccs)
    metadata = {}
    for s in saccs:
        for key, value in s.metadata.items():
            if key in metadata and metadata[key] != value:
                raise ValueError(
                    f"conflicting metadata across merge inputs: {key!r} is "
                    f"{metadata[key]!r} in one input and {value!r} in another"
                )
            metadata[key] = value
    # Strip metadata before concatenating (the library "resolves" clashing
    # keys by renaming them), then restore the validated union.
    stripped = [s.copy() for s in saccs]
    for s in stripped:
        s.metadata.clear()
    seen, shared = set(), set()  # tracers appearing in more than one input
    for s in saccs:
        shared |= seen & set(s.tracers)
        seen |= set(s.tracers)
    same_tracers = sorted(shared)
    # The library keeps the FIRST input's tracer on a name clash with no
    # equality check — verify shared tracers really are the same object.
    for name in same_tracers:
        first, *rest = [s.tracers[name] for s in saccs if name in s.tracers]
        for other in rest:
            if type(other) is not type(first) or not all(
                np.array_equal(getattr(first, a, None), getattr(other, a, None))
                for a in ("z", "nz")
            ):
                raise ValueError(
                    f"shared tracer {name!r} differs across merge inputs — "
                    "the merged file would silently keep the first"
                )
    merged = sacc.concatenate_data_sets(*stripped, same_tracers=same_tracers)
    for key, value in metadata.items():
        merged.metadata[key] = value
    _check_grid_consistency(merged, "theta")
    _check_grid_consistency(merged, "ell")
    return merged


def _grid_groups(s, angle):
    """Nested map ``grid-tag-value -> (data_type, tracers) -> point indices``.

    One entry per ``(data_type, tracers)`` series carrying an ``angle``
    (``'theta'`` or ``'ell'``) tag, in each series' own insertion order
    (never re-sorted), nested under the ``grid`` tag value it lives on
    (``None`` for untagged series) — the shape ``merge``'s consistency
    guard checks within each tag value. Indices (not angle values) are
    kept so the ℓ check can also recover each series' bandpower window.
    """
    groups = {}
    for i, point in enumerate(s.data):
        if angle in point.tags:
            groups.setdefault(point.tags.get("grid"), {}).setdefault(
                (point.data_type, point.tracers), []
            ).append(i)
    return groups


def _check_grid_consistency(s, angle):
    """Raise unless same-length ``angle`` arrays under one ``grid`` tag match.

    Consistency follows tagging semantics: the ``grid`` tag declares which
    binning a series lives on, so all same-length theta (or ell) arrays
    sharing a tag value must be bitwise identical — sacc never validates
    angles across data types/tracers, and a grid diverging at
    floating-point level chokes CosmoSIS downstream instead of failing
    loud here. Different lengths within a tag value pass (scale-cut
    subsets are legitimate); series under different tag values are
    unconstrained (``reporting`` vs ``integration`` differ by design).
    θ and ℓ are separate domains, each checked against itself only. For
    ℓ, two series on a bitwise-equal grid must also share the bandpower
    window (equal window ells and weight matrix); series without windows
    (foreign files) skip the window check.
    """
    for tag, by_series in _grid_groups(s, angle).items():
        series = [
            (key, np.array([s.data[i].tags[angle] for i in idx]), idx)
            for key, idx in by_series.items()
        ]
        for i, (key_a, arr_a, idx_a) in enumerate(series):
            for key_b, arr_b, idx_b in series[i + 1 :]:
                if len(arr_a) != len(arr_b):
                    continue
                if not np.array_equal(arr_a, arr_b):
                    max_diff = np.max(np.abs(arr_a - arr_b))
                    raise ValueError(
                        f"{angle} grids under the same grid tag ({tag!r}) "
                        f"differ; harmonize upstream — groups {key_a!r} and "
                        f"{key_b!r} (max abs diff {max_diff:.3e})"
                    )
                if angle != "ell" or any(
                    "window" not in s.data[idx[0]].tags for idx in (idx_a, idx_b)
                ):
                    continue
                win_a, win_b = map(s.get_bandpower_windows, (idx_a, idx_b))
                if not (
                    np.array_equal(win_a.values, win_b.values)
                    and np.array_equal(win_a.weight, win_b.weight)
                ):
                    raise ValueError(
                        f"bandpower windows differ between series sharing an "
                        f"ell grid under grid tag {tag!r}; harmonize upstream "
                        f"— groups {key_a!r} and {key_b!r}"
                    )


def update_statistic(s, sub):
    """Overwrite the values of ``s``'s points that match ``sub``'s, in place.

    The merge-back half of the extract → conceal → merge blinding flow
    (PRD #241 §4): each point of ``sub`` is matched to exactly one point of
    ``s`` by ``(data_type, tracers, tags)``, and that point's *value* is
    replaced. Nothing else changes — insertion order, tags, windows and the
    covariance are untouched (blinding shifts the mean only), so ``sub``'s
    own covariance (e.g. the sub-block ``extract`` attaches) is deliberately
    not consulted. A ``sub`` point with no match, or with several, raises
    ``ValueError``.

    Parameters
    ----------
    s : sacc.Sacc
        Target, mutated in place.
    sub : sacc.Sacc
        The replacement block, e.g. ``extract(s, ...)`` after concealment.
    """
    claimed = set()
    for point in sub.data:
        idx = s.indices(point.data_type, point.tracers, **point.tags)
        if len(idx) != 1:
            raise ValueError(
                f"update_statistic: {len(idx)} points in the target match "
                f"({point.data_type}, {point.tracers}, {point.tags}) — need "
                "exactly one"
            )
        if idx[0] in claimed:
            raise ValueError(
                f"update_statistic: two sub points match the same target "
                f"point ({point.data_type}, {point.tracers}, {point.tags})"
            )
        claimed.add(idx[0])
        s.data[idx[0]].value = point.value


def save(s, path, *, type):
    """Write ``s`` to ``path`` (FITS), overwriting any existing file.

    Parameters
    ----------
    s : sacc.Sacc
        Data set to write; its metadata is stamped in place.
    type : {'data', 'mock'}
        Provenance of the underlying catalogue, stored as the required
        ``type`` metadata tag (PRD #241 §4, "Mocks vs data"). The caller —
        the pipeline computing the data vector — knows whether its input
        catalogue is a mock; there is deliberately no default. ``load``
        refuses ``type='data'`` files that are not blinded.
    """
    if type not in ("data", "mock"):
        raise ValueError(f"type must be 'data' or 'mock'; got {type!r}")
    if s.metadata.get("type", type) != type:
        raise ValueError(
            f"Sacc metadata already carries type={s.metadata['type']!r}; "
            f"refusing to re-stamp as {type!r}"
        )
    s.metadata["type"] = type
    s.save_fits(path, overwrite=True)


def load(path, *, allow_unblinded=False):
    """Load a Sacc from ``path`` (FITS), failing closed on unblinded data.

    Every sacc_io file carries a ``type: data|mock`` metadata tag (stamped by
    ``save``); blinded files are additionally stamped ``concealed=True`` by
    Smokescreen. A ``type='data'`` file without that stamp is real, unblinded
    data, and loading it raises — skipping the blind can never silently
    expose the measured vector (PRD #241 §4). Mocks load freely, blinded or
    not.

    Parameters
    ----------
    path : str
        File to load.
    allow_unblinded : bool, optional
        Escape hatch for the two legitimate consumers of unblinded data:
        the blinding step itself (which must read the true vector to conceal
        it) and the unblinding/verification tooling. Nothing else — no
        analysis, plotting or inference code — may pass ``True``.

    Returns
    -------
    sacc.Sacc
        The loaded data set.
    """
    s = sacc.Sacc.load_fits(path)
    if (
        s.metadata["type"] == "data"
        and not s.metadata.get("concealed", False)
        and not allow_unblinded
    ):
        raise ValueError(
            f"{path} holds real data (type='data') without the "
            "concealed=True blinding stamp — refusing to load an unblinded "
            "data vector. Only the blinding/unblinding tooling may pass "
            "allow_unblinded=True."
        )
    return s
