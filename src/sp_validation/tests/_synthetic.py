"""A small deterministic synthetic catalogue, on disk, with its catalogue config.

The glue tests run the real compute seams on it: a shear catalogue
(RA/Dec/e1/e2/w), a PSF star catalogue with the columns the leakage and ρ/τ
seams read, a cs_util-readable dndz, and a ``cat_config.yaml`` whose blinds
live in ``blinds/`` beside it. Every catalogue entry in the config reads its
own copy of the same galaxies and declares its own blind.

``toy_shear`` stands in for :func:`sp_validation.theory.shear`, so a blind
shifts without running CAMB.
"""

import copy

import numpy as np
import yaml

from sp_validation import sacc_io

SIGNAL = (sacc_io.XI_PLUS, sacc_io.XI_MINUS, sacc_io.CL_EE)


def toy_shear(params, s):
    """A power law in θ or ℓ for ξ± and Cℓ_EE, its amplitude growing with S8
    and Ωm; zeros for every other row."""
    out = np.zeros(len(s.mean))
    for i, dp in enumerate(s.data):
        if dp.data_type in SIGNAL:
            x = dp.tags.get("theta", dp.tags.get("ell"))
            out[i] = params["S8"] ** 2 * params["Omega_m"] ** 0.3 * (x / 10.0) ** -0.8
    return 1e-4 * out


def write_synthetic_catalogs(
    tmp_path,
    n_gal=2000,
    n_star=800,
    ra_range=(10.0, 14.0),
    dec_range=(10.0, 14.0),
    seed=1234,
    coherent_shear=False,
    with_psf=False,
    catalogues=None,
):
    """Write the catalogue files and their config; return ``(params, version)``.

    Parameters
    ----------
    coherent_shear : bool
        Inject a smooth position-dependent shear on top of shape noise, so ξ±
        is smooth (the pure-E/B integral needs it to be well-posed).
    with_psf : bool
        Add a ``psf`` block (ρ/τ and pseudo-Cℓ read it via
        ``get_params_rho_tau``).
    catalogues : dict, optional
        ``{version: blind}``, one catalogue entry each, reading its own copy of
        the shear catalogue. ``blind`` is the entry's ``blind:`` value, or
        ``None`` for an entry that declares none. Defaults to one public
        catalogue, ``TestCatalog``.

    Returns
    -------
    (dict, str)
        ``CosmologyValidation`` keyword arguments (``catalog_config``,
        ``output_dir``) and the first catalogue's version.
    """
    from astropy.table import Table

    catalogues = catalogues or {"TestCatalog": "none"}
    rng = np.random.default_rng(seed)

    cat_dir = tmp_path / "catalog"
    nz_dir = tmp_path / "nz"
    output_dir = tmp_path / "output"
    for directory in (cat_dir, nz_dir, output_dir):
        directory.mkdir()

    ra = rng.uniform(*ra_range, n_gal)
    dec = rng.uniform(*dec_range, n_gal)
    if coherent_shear:
        e1 = 0.02 * np.cos(np.radians(ra) * 40) + rng.normal(0, 0.05, n_gal)
        e2 = 0.02 * np.sin(np.radians(dec) * 40) + rng.normal(0, 0.05, n_gal)
    else:
        e1 = rng.normal(0, 0.25, n_gal)
        e2 = rng.normal(0, 0.25, n_gal)
    w = rng.uniform(0.5, 1.0, n_gal)
    shear = Table({"RA": ra, "Dec": dec, "e1": e1, "e2": e2, "w": w})

    Table(
        {
            "RA": rng.uniform(*ra_range, n_star),
            "Dec": rng.uniform(*dec_range, n_star),
            "HSM_G1_PSF": rng.normal(0, 0.03, n_star),
            "HSM_G2_PSF": rng.normal(0, 0.03, n_star),
            "HSM_G1_STAR": rng.normal(0, 0.03, n_star),
            "HSM_G2_STAR": rng.normal(0, 0.03, n_star),
            "HSM_T_PSF": rng.uniform(0.4, 0.6, n_star),
            "HSM_T_STAR": rng.uniform(0.4, 0.6, n_star),
            "HSM_FLAG_PSF": np.zeros(n_star, dtype=int),
            "HSM_FLAG_STAR": np.zeros(n_star, dtype=int),
        }
    ).write(cat_dir / "star.fits", overwrite=True)

    # cs_util.read_dndz's commented-header format: "z" holds the n+1 bin
    # edges, "dn_dz" the densities.
    z_edges = np.linspace(0.05, 3.0, 31)
    dndz = np.exp(-(((z_edges - 0.7) / 0.3) ** 2))
    lines = ["# z dn_dz"] + [f"{zz} {nn}" for zz, nn in zip(z_edges, dndz)]
    (nz_dir / "dndz_SP_A.txt").write_text("\n".join(lines) + "\n")

    entry = {
        "subdir": str(cat_dir),
        "pipeline": "SP",
        "colour": "orange",
        "ls": "-",
        "marker": "o",
        "shear": {
            "redshift_path": str(nz_dir / "dndz_SP_A.txt"),
            "w_col": "w",
            "e1_col": "e1",
            "e2_col": "e2",
            "R": 1.0,
            "e1_col_corrected": "e1",
            "e2_col_corrected": "e2",
        },
        "star": {
            "path": "star.fits",
            "ra_col": "RA",
            "dec_col": "Dec",
            "e1_col": "HSM_G1_PSF",
            "e2_col": "HSM_G2_PSF",
        },
    }
    if with_psf:
        entry["psf"] = {
            "path": "star.fits",
            "hdu": 1,
            "ra_col": "RA",
            "dec_col": "Dec",
            "e1_PSF_col": "HSM_G1_PSF",
            "e2_PSF_col": "HSM_G2_PSF",
            "e1_star_col": "HSM_G1_STAR",
            "e2_star_col": "HSM_G2_STAR",
            "PSF_size": "HSM_T_PSF",
            "star_size": "HSM_T_STAR",
            "PSF_flag": "HSM_FLAG_PSF",
            "star_flag": "HSM_FLAG_STAR",
        }

    config = {
        "nz": {"subdir": str(nz_dir), "dndz": {"path": "dndz_{pipeline}_A.txt"}},
        "paths": {"output": str(output_dir), "blinds": str(tmp_path / "blinds")},
    }
    for version, blind in catalogues.items():
        config[version] = copy.deepcopy(entry)
        config[version]["shear"]["path"] = f"shear_{version}.fits"
        shear.write(cat_dir / f"shear_{version}.fits", overwrite=True)
        if blind is not None:
            config[version]["blind"] = blind
    config_path = tmp_path / "cat_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    params = {"catalog_config": str(config_path), "output_dir": str(output_dir)}
    return params, next(iter(catalogues))
