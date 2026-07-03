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


# ---------------------- Binning utility functions ----------------------
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


# ---------------------- Map computation utility functions ----------------------
def get_pixels(ra, dec, nside):
    """
    Get the HEALPix pixel indices for given RA and Dec.

    Parameters
    ----------
    ra : np.ndarray
        Right ascension in degrees.
    dec : np.ndarray
        Declination in degrees.
    nside : int
        HEALPix nside parameter.

    Returns
    -------
    unique_pix : np.ndarray
        Sorted unique pixel indices.
    idx : np.ndarray
        First-occurrence indices into the input from ``np.unique``.
    idx_rep : np.ndarray
        Inverse map: pixel-group index for each input object.
    """
    pixels = hp.ang2pix(nside, theta=np.radians(90 - dec), phi=np.radians(ra))

    unique_pix, idx, idx_rep = np.unique(pixels, return_index=True, return_inverse=True)
    return unique_pix, idx, idx_rep


def get_n_gal_map(
    nside, ra, dec, weights=None, unique_pix=None, idx=None, idx_rep=None
):
    """Weighted galaxy number-density HEALPix map plus pixel bookkeeping.

    Bins ``(ra, dec)`` (degrees) onto an ``nside`` HEALPix grid. With
    ``weights=None`` each object counts as 1 (galaxy counts); pass per-object
    ``weights`` for a weight-summed occupancy map.

    Returns
    -------
    n_gal : np.ndarray
        Map of summed weights (or counts) per pixel, shape ``(npix,)``.
    """
    if unique_pix is None or idx is None or idx_rep is None:
        unique_pix, idx, idx_rep = get_pixels(ra, dec, nside)

    n_gal = np.zeros(hp.nside2npix(nside))
    n_gal[unique_pix] = np.bincount(idx_rep, weights=weights)
    return n_gal


def get_shear_map(
    ra, dec, e1, e2, w, nside, unique_pix=None, idx=None, idx_rep=None, n_gal_map=None
):
    """Weighted shear HEALPix maps plus pixel bookkeeping.

    Bins ``(ra, dec)`` (degrees) onto an ``nside`` HEALPix grid. The shear
    components ``(e1, e2)`` are weighted by ``w`` and summed per pixel. If
    ``unique_pix``, ``idx``, and ``idx_rep`` are provided, they are used to
    avoid recomputing the pixel indices.
    If ``n_gal_map`` is provided, it is used to normalize the shear maps by the galaxy density.

    Returns
    -------
    e1_map : np.ndarray
        Weighted sum of E1 per pixel, shape ``(npix,)``.
    e2_map : np.ndarray
        Weighted sum of E2 per pixel, shape ``(npix,)``.
    """
    if unique_pix is None or idx is None or idx_rep is None:
        unique_pix, idx, idx_rep = get_pixels(ra, dec, nside)

    if n_gal_map is None:
        n_gal_map = get_n_gal_map(
            nside, ra, dec, weights=w, unique_pix=unique_pix, idx=idx, idx_rep=idx_rep
        )

    npix = hp.nside2npix(nside)
    e1_map = np.zeros(npix)
    e2_map = np.zeros(npix)

    e1_map[unique_pix] = np.bincount(idx_rep, weights=e1 * w)
    e2_map[unique_pix] = np.bincount(idx_rep, weights=e2 * w)

    non_zero = n_gal_map > 0
    e1_map[non_zero] /= n_gal_map[non_zero]
    e2_map[non_zero] /= n_gal_map[non_zero]

    return e1_map, e2_map


def get_variance_map(
    nside, ra, dec, e1, e2, w, unique_pix=None, idx=None, idx_rep=None
):
    """Compute the variance map of the shear components.

    The variance is computed as the weighted variance of the shear components in each pixel.

    Returns
    -------
    variance_map : np.ndarray
        Variance map of the shear components, shape ``(npix,)``.
    """
    if unique_pix is None or idx is None or idx_rep is None:
        unique_pix, idx, idx_rep = get_pixels(ra, dec, nside)

    npix = hp.nside2npix(nside)
    variance_map = np.zeros(npix)

    variance_map[unique_pix] = np.bincount(
        idx_rep, weights=0.5 * (e1**2 + e2**2) * w**2
    )

    return variance_map


def get_noise_bias_analytical(
    ra, dec, e1, e2, w, lmax, nside=1024, unique_pix=None, idx=None, idx_rep=None
):
    """
    Compute the analytical noise bias for shear power spectrum.
    """
    variance_map = get_variance_map(
        nside=nside,
        ra=ra,
        dec=dec,
        e1=e1,
        e2=e2,
        w=w,
        unique_pix=unique_pix,
        idx=idx,
        idx_rep=idx_rep,
    )

    noise_bias = hp.nside2pixarea(nside) * np.mean(variance_map)

    noise_bias_cl = np.zeros((4, lmax))

    noise_bias_cl[0, :] = noise_bias  # EE
    noise_bias_cl[3, :] = noise_bias  # BB

    return noise_bias_cl


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


def get_noise_realisation(
    ra,
    dec,
    e1,
    e2,
    w,
    nside,
    unique_pix=None,
    idx=None,
    idx_rep=None,
    n_gal_map=None,
    rng=None,
):
    """
    Generate a random noise realisation of the shear maps by applying a random rotation to the ellipticity components.

    Parameters
    ----------
    ra, dec : np.ndarray
        Right ascension and declination of the sources.
    e1, e2 : np.ndarray
        Ellipticity components.
    w : np.ndarray
        Weights of the sources.
    nside : int
        HEALPix resolution.
    unique_pix, idx, idx_rep : np.ndarray, optional
        Pixel indices and bookkeeping arrays. If not provided, they will be computed.
    n_gal_map : np.ndarray, optional
        Galaxy number density map. If not provided, it will be computed.

    Returns
    -------
    noise_map_e1, noise_map_e2 : np.ndarray
        Noise map for ellipticity components.
    """
    # Apply random rotation to the ellipticity components
    e1_rot, e2_rot = apply_random_rotation(e1, e2, rng=rng)

    # Compute the noise maps using the rotated ellipticity components
    noise_map_e1, noise_map_e2 = get_shear_map(
        ra=ra,
        dec=dec,
        e1=e1_rot,
        e2=e2_rot,
        w=w,
        nside=nside,
        unique_pix=unique_pix,
        idx=idx,
        idx_rep=idx_rep,
        n_gal_map=n_gal_map,
    )

    return noise_map_e1, noise_map_e2


# ---------------------- Cl computation functions ----------------------
def get_field_and_workspace_from_map(
    b,
    mask_a,
    e1_map_a=None,
    e2_map_a=None,
    mask_b=None,
    e1_map_b=None,
    e2_map_b=None,
    pol_factor=-1,
    return_wsp=True,
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
    return_wsp : bool, optional
        If True, return the NaMaster workspace object containing the mixing matrix.

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
    lmax = b.lmax
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

    if return_wsp:
        # Create NaMaster workspace
        wsp = nmt.NmtWorkspace.from_fields(field_a, field_b, b)

        return field_a, field_b, wsp
    else:
        return field_a, field_b, None


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
    return_wsp=True,
    same_bin=False,
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
    return_wsp : bool, optional
        If True, return the NaMaster workspace object containing the mixing matrix.

    Returns
    -------
    field_a : nmt.NmtFieldCatalog
        NaMaster field object for the first catalog.
    field_b : nmt.NmtFieldCatalog
        NaMaster field object for the second catalog (if provided, same as the first catalog otherwise).
    wsp : nmt.NmtWorkspace
        NaMaster workspace object containing the mixing matrix.

    """
    lmax = b.lmax
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
        and not same_bin
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

    if return_wsp:
        wsp = nmt.NmtWorkspace.from_fields(field_a, field_b, b)
        return field_a, field_b, wsp
    else:
        return field_a, field_b, None


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
    shear_map_a,
    mask_a,
    nside,
    binning,
    *,
    shear_map_b=None,
    mask_b=None,
    pol_factor=-1,
    wsp=None,
    ell_step=10,
    n_ell_bins=32,
    power=0.5,
):
    """Map-based pseudo-Cl for a complex shear map.

    Parameters
    ----------
    shear_map_a : np.ndarray
        Complex shear map (``e1 + 1j * e2``).
    mask_a : np.ndarray
        Field mask (the galaxy number-density map).
    nside : int
        HEALPix resolution; fixes the harmonic geometry.
    binning : str
        Binning scheme passed to :func:`make_namaster_bin`.
    shear_map_b : np.ndarray, optional
        Complex shear map for the second field (``e1 + 1j * e2``).
    mask_b : np.ndarray, optional
        Field mask for the second field (the galaxy number-density map).
    pol_factor : float, optional
        Polarization factor to apply to the E2 component.
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
    # First do some assertion checks
    if shear_map_b is not None:
        assert mask_b is not None, "mask_b must be provided if shear_map_b is provided"
        assert shear_map_a.shape == shear_map_b.shape, (
            "shear_map_a and shear_map_b must have the same shape"
        )
        assert mask_a.shape == mask_b.shape, (
            "mask_a and mask_b must have the same shape"
        )

    if mask_b is not None:
        assert shear_map_b is not None, (
            "shear_map_b must be provided if mask_b is provided"
        )
        assert shear_map_a.shape == shear_map_b.shape, (
            "shear_map_a and shear_map_b must have the same shape"
        )
        assert mask_a.shape == mask_b.shape, (
            "mask_a and mask_b must have the same shape"
        )

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

    if wsp is None:
        field_a, field_b, wsp = get_field_and_workspace_from_map(
            b,
            mask_a,
            e1_map_a=shear_map_a.real,
            e2_map_a=shear_map_a.imag,
            mask_b=mask_b,
            e1_map_b=shear_map_b.real if shear_map_b is not None else None,
            e2_map_b=shear_map_b.imag if shear_map_b is not None else None,
            pol_factor=pol_factor,
            return_wsp=True,
        )
    else:
        field_a, field_b, _ = get_field_and_workspace_from_map(
            b,
            mask_a,
            e1_map_a=shear_map_a.real,
            e2_map_a=shear_map_a.imag,
            mask_b=mask_b,
            e1_map_b=shear_map_b.real if shear_map_b is not None else None,
            e2_map_b=shear_map_b.imag if shear_map_b is not None else None,
            pol_factor=pol_factor,
            return_wsp=False,
        )

    cl_coupled, cl_decoupled = compute_cl_from_field_and_workspace(
        field_a, field_b, wsp, b
    )

    return ell_eff, cl_decoupled, wsp


def get_pseudo_cls_catalog(
    catalog,
    params,
    nside,
    binning,
    *,
    tomo_bin_a=None,
    tomo_bin_b=None,
    pol_factor=-1,
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
    pol_factor : int, optional
        Polarization factor to apply to the E2 component.
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
    # First make some assertion checks reagarding the run mode
    assert (tomo_bin_a is None and tomo_bin_b is None) or (
        tomo_bin_a is not None and tomo_bin_b is not None
    ), "Both tomo_bin_a and tomo_bin_b must be provided or both must be None"

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

    is_tomography = tomo_bin_a is not None and tomo_bin_b is not None
    if is_tomography:
        mask_tomo_a = catalog[params["tomo_bin_col"]] == tomo_bin_a
        mask_tomo_b = catalog[params["tomo_bin_col"]] == tomo_bin_b
        catalog_a = catalog[mask_tomo_a]
        catalog_b = catalog[mask_tomo_b]
        same_bin = tomo_bin_a == tomo_bin_b
    else:
        catalog_a = catalog
        catalog_b = catalog
        same_bin = True

    if wsp is None:
        field_a, field_b, wsp = get_field_and_workspace_from_catalog(
            b,
            ra_a=catalog_a[params["ra_col"]],
            dec_a=catalog_a[params["dec_col"]],
            e1_a=catalog_a[params["e1_col"]],
            e2_a=catalog_a[params["e2_col"]],
            w_a=catalog_a[params["w_col"]],
            ra_b=catalog_b[params["ra_col"]],
            dec_b=catalog_b[params["dec_col"]],
            e1_b=catalog_b[params["e1_col"]],
            e2_b=catalog_b[params["e2_col"]],
            w_b=catalog_b[params["w_col"]],
            pol_factor=pol_factor,
            return_wsp=True,
            same_bin=same_bin,
        )
    else:
        field_a, field_b, _ = get_field_and_workspace_from_catalog(
            b,
            ra_a=catalog_a[params["ra_col"]],
            dec_a=catalog_a[params["dec_col"]],
            e1_a=catalog_a[params["e1_col"]],
            e2_a=catalog_a[params["e2_col"]],
            w_a=catalog_a[params["w_col"]],
            ra_b=catalog_b[params["ra_col"]],
            dec_b=catalog_b[params["dec_col"]],
            e1_b=catalog_b[params["e1_col"]],
            e2_b=catalog_b[params["e2_col"]],
            w_b=catalog_b[params["w_col"]],
            pol_factor=pol_factor,
            return_wsp=False,
            same_bin=same_bin,
        )

    cl_coupled, cl_decoupled = compute_cl_from_field_and_workspace(
        field_a, field_b, wsp, b
    )

    return ell_eff, cl_decoupled, wsp
