"""Pseudo-Cl / harmonic-space estimator primitives for cosmology validation.

Stateless shared primitives for the pseudo-Cl (harmonic-space) estimator,
mirroring ``b_modes.py`` and ``rho_tau.py``: the orchestrator mixin in
``sp_validation.cosmo_val.pseudo_cl`` (and analysis scripts directly) call these
free functions. Everything here is pure computation -- NaMaster binning,
weighted galaxy number-density maps, random-rotation noise debiasing, and the
map-/catalog-based pseudo-Cl estimators. Depends on pymaster (NaMaster) and
healpy.

The harmonic geometry the estimators use is fixed by ``nside``: ``lmin = 8``,
``lmax = 2 * nside``, ``b_lmax = lmax - 1``. ``pseudo_cl_geometry`` returns that
triple so the binning and the fields stay in lockstep.
"""

import healpy as hp
import numpy as np
import pymaster as nmt

# Lowest multipole retained by the pseudo-Cl estimators.
LMIN = 8


def pseudo_cl_geometry(nside):
    """Return ``(lmin, lmax, b_lmax)`` for the pseudo-Cl estimator at ``nside``.

    ``lmax = 2 * nside`` is the NaMaster band-power ceiling; ``b_lmax = lmax - 1``
    is the field/binning ``lmax``. ``lmin`` is the fixed low-ell floor.
    """
    lmax = 2 * nside
    return LMIN, lmax, lmax - 1


def make_namaster_bin(
    lmin, lmax, b_lmax, binning, *, ell_step=10, n_ell_bins=32, power=0.5
):
    """Build a NaMaster binning object for one of the supported schemes.

    Parameters
    ----------
    lmin, lmax : int
        Multipole range.
    b_lmax : int
        Maximum multipole for the ``NmtBin`` object.
    binning : {'linear', 'logspace', 'powspace'}
        Binning scheme.
    ell_step : int, optional
        Bin width in ell for ``'linear'`` binning.
    n_ell_bins : int, optional
        Number of ell bins for ``'logspace'`` / ``'powspace'`` binning.
    power : float, optional
        Exponent for ``'powspace'`` binning.

    Returns
    -------
    nmt.NmtBin
    """
    ells = np.arange(lmin, lmax + 1)

    if binning == "linear":
        bpws = (ells - lmin) // ell_step
        bpws = np.minimum(bpws, bpws[-1])
        b = nmt.NmtBin(ells=ells, bpws=bpws, lmax=b_lmax)
    elif binning == "logspace":
        # Start geomspace at ell_min_log (>= lmin) to avoid
        # sub-multipole bins at low ell that destabilize the MCM.
        # All ell below ell_min_log go into bin 0 as padding.
        ell_min_log = max(lmin, 50)
        bins_ell = np.geomspace(ell_min_log, lmax, n_ell_bins + 1)
        bpws = np.digitize(ells.astype(float), bins_ell) - 1
        bpws = np.clip(bpws, 0, n_ell_bins - 1)
        b = nmt.NmtBin(ells=ells, bpws=bpws, lmax=b_lmax)
    elif binning == "powspace":
        start = np.power(lmin, power)
        end = np.power(lmax, power)
        bins_ell = np.power(np.linspace(start, end, n_ell_bins + 1), 1 / power)
        bpws = np.digitize(ells.astype(float), bins_ell) - 1
        bpws[0] = 0
        bpws[-1] = n_ell_bins - 1
        b = nmt.NmtBin(ells=ells, bpws=bpws, lmax=b_lmax)
    else:
        raise ValueError(
            f"Unknown binning '{binning}'. "
            "Choose from 'linear', 'logspace', 'powspace'."
        )

    return b


def get_n_gal_map(nside, ra, dec, weights=None):
    """Weighted galaxy number-density HEALPix map plus pixel bookkeeping.

    Bins ``(ra, dec)`` (degrees) onto an ``nside`` HEALPix grid. With
    ``weights=None`` each object counts as 1 (galaxy counts); pass per-object
    ``weights`` for a weight-summed occupancy map.

    Returns
    -------
    n_gal : np.ndarray
        Map of summed weights (or counts) per pixel, shape ``(npix,)``.
    unique_pix : np.ndarray
        Sorted unique occupied pixel indices.
    idx : np.ndarray
        First-occurrence indices into the input from ``np.unique``.
    idx_rep : np.ndarray
        Inverse map: pixel-group index for each input object.
    """
    theta = (90.0 - dec) * np.pi / 180.0
    phi = ra * np.pi / 180.0
    pix = hp.ang2pix(nside, theta, phi)

    unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)
    n_gal = np.zeros(hp.nside2npix(nside))
    n_gal[unique_pix] = np.bincount(idx_rep, weights=weights)
    return n_gal, unique_pix, idx, idx_rep


def apply_random_rotation(e1, e2, rng=None):
    """Apply a uniform random rotation to ellipticity components.

    Parameters
    ----------
    e1, e2 : np.ndarray
        Ellipticity components.
    rng : np.random.Generator, optional
        Random generator for the rotation angles. Pass a seeded generator
        (e.g. ``np.random.default_rng(seed)``) for reproducible draws; when
        ``None`` a fresh entropy-seeded generator is used (non-reproducible).

    Returns
    -------
    e1_out, e2_out : np.ndarray
        Rotated ellipticity components.
    """
    if rng is None:
        rng = np.random.default_rng()
    rot_angle = rng.random(len(e1)) * 2 * np.pi
    e1_out = e1 * np.cos(rot_angle) + e2 * np.sin(rot_angle)
    e2_out = -e1 * np.sin(rot_angle) + e2 * np.cos(rot_angle)
    return e1_out, e2_out


def get_field_and_workspace_from_map(
    b,
    mask_a,
    e1_map_a=None,
    e2_map_a=None,
    mask_b=None,
    e1_map_b=None,
    e2_map_b=None,
    pol_factor=-1,
):
    """Compute a NaMaster field and workspace object from the input maps.

    If the shear maps are None, returns field objects but only the workspace objects is relevant and contains the mixing matrix.
    If the second mask and shear maps (indexed b) are provided, the mixing matrix is computed between the two fields.

    Parameters
    ----------
    b : nmt.NmtBin
        NaMaster binning object.
    mask_a : np.ndarray
        Field mask for the first map.
    e1_map_a : np.ndarray, optional
        E1 map for the first field.
    e2_map_a : np.ndarray, optional
        E2 map for the first field.
    mask_b : np.ndarray, optional
        Field mask for the second map.
    e1_map_b : np.ndarray, optional
        E1 map for the second field.
    e2_map_b : np.ndarray, optional
        E2 map for the second field.
    pol_factor : float, optional
        Polarization factor to apply to the E2 map.

    Returns
    -------
    field_a : nmt.NmtField
        NaMaster field object for the first map.
    field_b : nmt.NmtField
        NaMaster field object for the second map (if provided, same than the first map otherwise).
    wsp : nmt.NmtWorkspace
        NaMaster workspace object containing the mixing matrix.

    """
    nside = hp.npix2nside(len(mask_a))
    lmax = b.get_ell_max()
    if e1_map_a is None or e2_map_a is None:
        e1_map_a = np.zeros(hp.nside2npix(nside))
        e2_map_a = np.zeros(hp.nside2npix(nside))

    # Create NaMaster field
    field_a = nmt.NmtField(
        mask=mask_a, maps=[e1_map_a, pol_factor * e2_map_a], lmax=lmax
    )

    if mask_b is not None:
        if e1_map_b is None or e2_map_b is None:
            e1_map_b = np.zeros(hp.nside2npix(nside))
            e2_map_b = np.zeros(hp.nside2npix(nside))

        field_b = nmt.NmtField(
            mask=mask_b, maps=[e1_map_b, pol_factor * e2_map_b], lmax=lmax
        )
    else:
        field_b = field_a
    # Create NaMaster workspace
    wsp = nmt.NmtWorkspace.from_fields(field_a, field_b, b)

    return field_a, field_b, wsp


def get_field_and_workspace_from_catalog(
    b,
    ra_a,
    dec_a,
    e1_a,
    e2_a,
    w_a,
    ra_b=None,
    dec_b=None,
    e1_b=None,
    e2_b=None,
    w_b=None,
    pol_factor=-1,
):
    """Create a NaMaster field and workspace from the input catalog.

    If the second catalog is provided, the mixing matrix is computed between the two fields.

    Parameters
    ----------
    b : nmt.NmtBin
        NaMaster binning object.
    ra_a : np.ndarray
        Right ascension of sources in the first catalog.
    dec_a : np.ndarray
        Declination of sources in the first catalog.
    e1_a : np.ndarray
        E1 shear component of sources in the first catalog.
    e2_a : np.ndarray
        E2 shear component of sources in the first catalog.
    w_a : np.ndarray
        Weights of sources in the first catalog.
    ra_b : np.ndarray, optional
        Right ascension of sources in the second catalog.
    dec_b : np.ndarray, optional
        Declination of sources in the second catalog.
    e1_b : np.ndarray, optional
        E1 shear component of sources in the second catalog.
    e2_b : np.ndarray, optional
        E2 shear component of sources in the second catalog.
    w_b : np.ndarray, optional
        Weights of sources in the second catalog.
    pol_factor : float, optional
        Polarization factor to apply to the E2 component.

    Returns
    -------
    field_a : nmt.NmtFieldCatalog
        NaMaster field object for the first catalog.
    field_b : nmt.NmtFieldCatalog
        NaMaster field object for the second catalog (if provided, same as the first catalog otherwise).
    wsp : nmt.NmtWorkspace
        NaMaster workspace object containing the mixing matrix.

    """
    lmax = b.get_ell_max()
    # Get field for input catalog a
    field_a = nmt.NmtFieldCatalog(
        positions=[ra_a, dec_a],
        weights=w_a,
        field=[e1_a, pol_factor * e2_a],
        lmax=lmax,
        lmax_mask=lmax,
        spin=2,
        lonlat=True,
    )

    if (
        ra_b is not None
        and dec_b is not None
        and e1_b is not None
        and e2_b is not None
        and w_b is not None
    ):
        field_b = nmt.NmtFieldCatalog(
            positions=[ra_b, dec_b],
            weights=w_b,
            field=[e1_b, pol_factor * e2_b],
            lmax=lmax,
            lmax_mask=lmax,
            spin=2,
            lonlat=True,
        )
    else:
        field_b = field_a

    wsp = nmt.NmtWorkspace.from_fields(field_a, field_b, b)
    return field_a, field_b, wsp


def compute_cl_from_field_and_workspace(field_a, field_b, wsp, b):
    """Compute the angular power spectrum from the input NaMaster field and workspace

    Parameters
    ----------
    field_a : nmt.NmtField
        NaMaster field object for the first catalog.
    field_b : nmt.NmtField
        NaMaster field object for the second catalog.
    wsp : nmt.NmtWorkspace
        NaMaster workspace object containing the mixing matrix.
    b : nmt.NmtBin
        NaMaster binning object.

    Returns
    -------
    cl_coupled : np.ndarray
        Coupled angular power spectrum.
    cl_decoupled : np.ndarray
        Decoupled angular power spectrum.
    """
    cl_coupled = nmt.compute_coupled_cell(field_a, field_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled)

    return cl_coupled, cl_decoupled


def get_pseudo_cls_map(
    shear_map,
    mask,
    nside,
    binning,
    *,
    pol_factor=True,
    wsp=None,
    ell_step=10,
    n_ell_bins=32,
    power=0.5,
):
    """Map-based pseudo-Cl for a complex shear map.

    Parameters
    ----------
    shear_map : np.ndarray
        Complex shear map (``e1 + 1j * e2``).
    mask : np.ndarray
        Field mask (the galaxy number-density map).
    nside : int
        HEALPix resolution; fixes the harmonic geometry.
    binning : str
        Binning scheme passed to :func:`make_namaster_bin`.
    pol_factor : bool, optional
        If ``True`` flip the sign of the imaginary (e2) component.
    wsp : nmt.NmtWorkspace, optional
        Reuse a coupling workspace; built from the field if ``None``.
    ell_step, n_ell_bins, power : optional
        Binning-scheme parameters forwarded to :func:`make_namaster_bin`.

    Returns
    -------
    ell_eff : np.ndarray
        Effective multipoles of the bandpowers.
    cl_all : np.ndarray
        Decoupled EE/EB/BE/BB spectra, shape ``(4, n_bands)``.
    wsp : nmt.NmtWorkspace
        The coupling workspace (newly built or the one passed in).
    """
    lmin, lmax, b_lmax = pseudo_cl_geometry(nside)

    b = make_namaster_bin(
        lmin,
        lmax,
        b_lmax,
        binning,
        ell_step=ell_step,
        n_ell_bins=n_ell_bins,
        power=power,
    )
    ell_eff = b.get_effective_ells()

    factor = -1 if pol_factor else 1

    f_all = nmt.NmtField(
        mask=mask, maps=[shear_map.real, factor * shear_map.imag], lmax=b_lmax
    )
    if wsp is None:
        wsp = nmt.NmtWorkspace.from_fields(f_all, f_all, b)

    cl_coupled = nmt.compute_coupled_cell(f_all, f_all)
    cl_all = wsp.decouple_cell(cl_coupled)

    return ell_eff, cl_all, wsp


def get_pseudo_cls_catalog(
    catalog,
    params,
    nside,
    binning,
    *,
    pol_factor=True,
    wsp=None,
    ell_step=10,
    n_ell_bins=32,
    power=0.5,
):
    """Catalog-based pseudo-Cl via NaMaster's ``NmtFieldCatalog``.

    Parameters
    ----------
    catalog : np.ndarray
        Structured catalog array with the columns named in ``params``.
    params : dict
        Column-name mapping (``ra_col``, ``dec_col``, ``w_col``, ``e1_col``,
        ``e2_col``).
    nside : int
        HEALPix resolution; fixes the harmonic geometry.
    binning : str
        Binning scheme passed to :func:`make_namaster_bin`.
    pol_factor : bool, optional
        If ``True`` flip the sign of the e2 component.
    wsp : nmt.NmtWorkspace, optional
        Reuse a coupling workspace; built from the field if ``None``.
    ell_step, n_ell_bins, power : optional
        Binning-scheme parameters forwarded to :func:`make_namaster_bin`.

    Returns
    -------
    ell_eff : np.ndarray
        Effective multipoles of the bandpowers.
    cl_all : np.ndarray
        Decoupled EE/EB/BE/BB spectra, shape ``(4, n_bands)``.
    wsp : nmt.NmtWorkspace
        The coupling workspace (newly built or the one passed in).
    """
    lmin, lmax, b_lmax = pseudo_cl_geometry(nside)

    b = make_namaster_bin(
        lmin,
        lmax,
        b_lmax,
        binning,
        ell_step=ell_step,
        n_ell_bins=n_ell_bins,
        power=power,
    )
    ell_eff = b.get_effective_ells()

    factor = -1 if pol_factor else 1

    f_all = nmt.NmtFieldCatalog(
        positions=[catalog[params["ra_col"]], catalog[params["dec_col"]]],
        weights=catalog[params["w_col"]],
        field=[catalog[params["e1_col"]], factor * catalog[params["e2_col"]]],
        lmax=b_lmax,
        lmax_mask=b_lmax,
        spin=2,
        lonlat=True,
    )

    if wsp is None:
        wsp = nmt.NmtWorkspace.from_fields(f_all, f_all, b)

    cl_coupled = nmt.compute_coupled_cell(f_all, f_all)
    cl_all = wsp.decouple_cell(cl_coupled)

    return ell_eff, cl_all, wsp
