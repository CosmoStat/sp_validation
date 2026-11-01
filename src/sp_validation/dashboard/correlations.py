"""Position–shear and scale-dependent leakage products for the dashboard.

One catalogue pass builds the TreeCorr catalogues once and runs the
catalogue-paper correlation tests, all on the same k-means jackknife patches
(computed once from the galaxy positions and saved):

``stars``
    Galaxy γ_t, γ_× around PSF validation stars (valid HSM, one entry per star
    after merging per-exposure detections within ``--dedup-arcsec``), for all
    stars and in star-magnitude tertiles. No random subtraction.
``lambda``
    PSF-model ellipticity (λ₁) and star-minus-model residual (λ₃) of the PSF
    validation stars, tangential and cross, around galaxy positions (a random
    subsample of ``--lens-fraction``), minus the same around footprint randoms.
``tiles``
    Galaxy γ_t, γ_× around the centres of the tiles in the catalogue.
``gaia``
    Galaxy γ_t, γ_× around Gaia DR3 stars (G < 20, RUWE < 1.4) inside the
    footprint, in G bins, minus the same around footprint randoms.
``alpha``
    Scale-dependent leakage α(θ) = τ₀(θ)/ρ₀(θ) for all galaxies and in
    galaxy-magnitude tertiles; ρ₀ from the PSF model at the stars, τ₀ the
    galaxy–PSF-model correlation, both as in :mod:`sp_validation.rho_tau`.

Galaxy shapes are the calibrated ``e1``, ``e2`` with ``w_des``; in the α
magnitude bins they are rescaled to the bin's own shear response. Each
measurement writes ``<name>_<version>.npz`` with θ, the correlations, their
jackknife errors and joint covariance, pair counts and a JSON provenance.

Usage (inside the container)::

    python -m sp_validation.dashboard.correlations \\
        --catalogue-dir run-r425/catalogue --version SP_v2.0 \\
        --out-dir catpaper/products --gaia gaia_dr3_G20_ruwe1p4_unions20k.fits
"""

import argparse
import json
import os
import resource
import time
from pathlib import Path

import fitsio
import healpy as hp
import numpy as np
import treecorr
from cs_util import cfis
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from sp_validation.io import read_catalogue
from sp_validation.statistics import jackknife_patch_centers

from .position_shear import measure

GALAXY_COLUMNS = ["RA", "Dec", "e1", "e2", "w_des", "mag", "R_g11", "R_g22"]
STAR_COLUMNS = [
    "RA",
    "DEC",
    "MAG",
    "HSM_G1_PSF",
    "HSM_G2_PSF",
    "HSM_G1_STAR",
    "HSM_G2_STAR",
    "HSM_FLAG_PSF",
    "HSM_FLAG_STAR",
]
MEASUREMENTS = ("stars", "lambda", "tiles", "gaia", "alpha")
GAIA_G_EDGES = (3.0, 17.9, 19.2, 20.0)
TC = dict(min_sep=1.0, max_sep=250.0, nbins=20)


def log(message):
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[{time.strftime('%H:%M:%S')} peak {peak:.1f} GB] {message}", flush=True)


def in_region(ra, dec, region):
    if region is None:
        return np.ones(len(ra), dtype=bool)
    ra_min, ra_max, dec_min, dec_max = region
    return (ra >= ra_min) & (ra < ra_max) & (dec >= dec_min) & (dec < dec_max)


def tile_name(tile_id):
    """``XXX.YYY`` from a ``TILE_ID`` written as a float string (``51.34``)."""
    x, _, y = str(tile_id).partition(".")
    return f"{int(x):03d}.{y.ljust(3, '0')}"


def tile_centres(tile_ids):
    """RA, Dec (deg) of the CFIS tile centres for unique ``TILE_ID`` values."""
    names = sorted({tile_name(t) for t in tile_ids})
    nix, niy = zip(*(cfis.get_tile_number(n) for n in names))
    ra, dec = cfis.get_tile_coord_from_nixy(list(nix), list(niy))
    return names, np.asarray(ra.deg), np.asarray(dec.deg)


def merge_detections(ra, dec, radius_arcsec):
    """Group detections closer than ``radius_arcsec`` (friends of friends).

    Returns the group label of each detection and the number of groups.
    """
    xyz = hp.ang2vec(ra, dec, lonlat=True)
    chord = 2 * np.sin(np.deg2rad(radius_arcsec / 3600) / 2)
    pairs = cKDTree(xyz).query_pairs(chord, output_type="ndarray")
    n = len(ra)
    graph = coo_matrix(
        (np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
    )
    return connected_components(graph, directed=False)[::-1]


def group_mean(labels, n_groups, values):
    counts = np.bincount(labels, minlength=n_groups)
    return np.bincount(labels, weights=values, minlength=n_groups) / counts


def unique_stars(stars, radius_arcsec):
    """One position and magnitude per star from its per-exposure detections."""
    labels, n = merge_detections(stars["RA"], stars["DEC"], radius_arcsec)
    xyz = hp.ang2vec(stars["RA"], stars["DEC"], lonlat=True)
    mean = np.column_stack([group_mean(labels, n, xyz[:, i]) for i in range(3)])
    ra, dec = hp.vec2ang(mean, lonlat=True)
    return ra, dec, group_mean(labels, n, stars["MAG"].astype(float))


def read_galaxies(path, region, chunk=5_000_000):
    """Galaxy columns (float64) and the set of tile IDs, streamed in chunks."""
    parts, tiles = {k: [] for k in GALAXY_COLUMNS}, set()
    with fitsio.FITS(path) as fits:
        table = fits[1]
        for start in range(0, table.get_nrows(), chunk):
            rows = table[GALAXY_COLUMNS + ["TILE_ID"]][start : start + chunk]
            keep = in_region(rows["RA"], rows["Dec"], region)
            for key in GALAXY_COLUMNS:
                parts[key].append(rows[key][keep].astype(np.float64))
            tiles.update(np.unique(rows["TILE_ID"][keep]).tolist())
    return {k: np.concatenate(v) for k, v in parts.items()}, tiles


def read_stars(path, region):
    stars = read_catalogue(path, columns=STAR_COLUMNS)
    keep = (stars["HSM_FLAG_PSF"] == 0) & (stars["HSM_FLAG_STAR"] == 0)
    keep &= in_region(stars["RA"], stars["DEC"], region)
    return stars[keep]


def footprint_pixels(path):
    """Occupied pixels (RING) and nside of an ``occupied-mask-nside*.npz``."""
    mask = np.load(path)
    assert str(mask["ordering"]) == "RING"
    return int(mask["nside"]), np.asarray(mask["pixel"])


def in_footprint(ra, dec, nside, pixels):
    return np.isin(hp.ang2pix(nside, ra, dec, lonlat=True), pixels)


def footprint_randoms(n, nside, pixels, region, rng, depth=6):
    """Points uniform over the footprint pixels, to 1/2**depth of a pixel."""
    nest = hp.ring2nest(nside, pixels)
    fine = nest[rng.integers(len(nest), size=n)] * 4**depth
    fine += rng.integers(4**depth, size=n)
    ra, dec = hp.pix2ang(nside * 2**depth, fine, nest=True, lonlat=True)
    keep = in_region(ra, dec, region)
    return ra[keep], dec[keep]


def catalog(ra, dec, centres, **fields):
    return treecorr.Catalog(
        ra=ra, dec=dec, ra_units="deg", dec_units="deg", patch_centers=centres, **fields
    )


def save(out, name, version, results, provenance):
    """Write one measurement: ``results`` maps a label to a ``measure`` dict."""
    arrays = {}
    for label, result in results.items():
        for key, value in result.items():
            arrays[f"{label}__{key}"] = np.asarray(value)
    path = Path(out) / f"{name}_{version}.npz"
    np.savez_compressed(
        path,
        labels=np.array(list(results)),
        provenance=json.dumps(provenance),
        **arrays,
    )
    log(f"wrote {path}")


def quantile_edges(values, n):
    """Edges of ``n`` equal-count bins; the last edge just above the maximum."""
    edges = np.quantile(values, np.linspace(0, 1, n + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def alpha_measurement(gal, stars, centres, n_bins, threads, r_selection):
    """τ₀ per magnitude bin, ρ₀, and α = τ₀/ρ₀ with a joint jackknife covariance."""
    config = dict(
        TC, sep_units="arcmin", var_method="jackknife", cross_patch_weight="match"
    )
    psf = catalog(
        stars["RA"],
        stars["DEC"],
        centres,
        g1=stars["HSM_G1_PSF"],
        g2=stars["HSM_G2_PSF"],
    )
    rho0 = treecorr.GGCorrelation(config)
    rho0.process(psf, psf, num_threads=threads)
    log("rho0 done")
    edges = quantile_edges(gal["mag"], n_bins)
    selections = [("all", np.ones(len(gal["mag"]), dtype=bool))] + [
        (f"mag{i}", (gal["mag"] >= lo) & (gal["mag"] < hi))
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))
    ]
    r_global = None
    taus, responses = [], {}
    for label, sel in selections:
        w = gal["w_des"][sel]
        r_shear = np.array(
            [
                np.average(gal["R_g11"][sel], weights=w),
                np.average(gal["R_g22"][sel], weights=w),
            ]
        )
        r_bin = r_shear + r_selection
        if r_global is None:
            r_global = r_bin
        # Calibrated e is (e_uncal - c)/R_global; rescale to the bin's response.
        scale = r_global / r_bin
        g = []
        for i, key in enumerate(("e1", "e2")):
            e = gal[key][sel] * scale[i]
            g.append(e - np.average(e, weights=w))
        cat = catalog(gal["RA"][sel], gal["Dec"][sel], centres, g1=g[0], g2=g[1], w=w)
        tau0 = treecorr.GGCorrelation(config)
        tau0.process(cat, psf, num_threads=threads)
        taus.append(tau0)
        responses[label] = {
            "n": int(sel.sum()),
            "R_shear": r_shear.tolist(),
            "R": r_bin.tolist(),
            "rescale": scale.tolist(),
        }
        del cat
        log(f"tau0 {label} done")
    n = TC["nbins"]

    def vector(corrs):
        rho = corrs[0].xip
        return np.concatenate([c.xip / rho for c in corrs[1:]])

    def raw(corrs):
        return np.concatenate([c.xip for c in corrs])

    corrs = [rho0, *taus]
    alpha = vector(corrs)
    cov = treecorr.estimate_multi_cov(corrs, "jackknife", func=vector)
    raw_cov = treecorr.estimate_multi_cov(corrs, "jackknife", func=raw)
    results = {}
    for i, (label, _) in enumerate(selections):
        block = slice(i * n, (i + 1) * n)
        results[label] = dict(
            theta=np.exp(rho0.meanlogr),
            alpha=alpha[block],
            alpha_error=np.sqrt(np.diag(cov)[block]),
            tau0=taus[i].xip,
            tau0_error=np.sqrt(np.diag(raw_cov)[(i + 1) * n : (i + 2) * n]),
            npairs=taus[i].npairs,
        )
    results["rho0"] = dict(
        theta=np.exp(rho0.meanlogr),
        rho0=rho0.xip,
        rho0_error=np.sqrt(np.diag(raw_cov)[:n]),
        npairs=rho0.npairs,
    )
    results["joint"] = dict(alpha_covariance=cov, raw_covariance=raw_cov)
    provenance = {
        "galaxy_mag_edges": edges.tolist(),
        "responses": responses,
        "R_selection_global": r_selection.tolist(),
        "shapes": "calibrated e1, e2 rescaled by R_global/R_bin, weighted mean "
        "subtracted per bin (as rho_tau TauStat); w_des",
        "stars": "PSF validation stars, HSM_FLAG_STAR == 0 & HSM_FLAG_PSF == 0, "
        "every exposure detection (as the dashboard rho/tau)",
        "rho0": "xi+ of HSM_G*_PSF x HSM_G*_PSF",
        "tau0": "xi+ of galaxy e x HSM_G*_PSF",
        "raw_covariance_order": ["rho0"] + [s for s, _ in selections],
    }
    return results, provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--catalogue-dir", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gaia", help="staged Gaia DR3 FITS (stage_gaia.py)")
    parser.add_argument("--only", nargs="+", choices=MEASUREMENTS, default=None)
    parser.add_argument(
        "--region",
        type=float,
        nargs=4,
        default=None,
        metavar=("RA_MIN", "RA_MAX", "DEC_MIN", "DEC_MAX"),
        help="restrict every sample to a box (pilot runs)",
    )
    parser.add_argument("--npatch", type=int, default=100)
    parser.add_argument(
        "--lens-fraction",
        type=float,
        default=0.1,
        help="galaxy subsample used as lenses for lambda",
    )
    parser.add_argument(
        "--random-factor", type=float, default=1.0, help="randoms per lambda lens"
    )
    parser.add_argument("--dedup-arcsec", type=float, default=0.5)
    parser.add_argument("--mag-bins", type=int, default=3)
    parser.add_argument("--threads", type=int, default=len(os.sched_getaffinity(0)))
    parser.add_argument("--seed", type=int, default=425)
    args = parser.parse_args(argv)
    todo = args.only or [m for m in MEASUREMENTS if m != "gaia" or args.gaia]
    cdir = Path(args.catalogue_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    base = {
        "version": args.version,
        "catalogue_dir": str(cdir.resolve()),
        "region": args.region,
        "npatch": args.npatch,
        "treecorr": dict(
            TC, sep_units="arcmin", var_method="jackknife", version=treecorr.__version__
        ),
        "units": "theta in arcmin; shears dimensionless",
    }
    timings = {}

    t0 = time.time()
    gal, tile_ids = read_galaxies(cdir / "shape_catalog_cut_ngmix.fits", args.region)
    log(f"{len(gal['RA'])} galaxies, {len(tile_ids)} tiles")
    stars = read_stars(cdir / "stars.hdf5", args.region)
    log(f"{len(stars)} valid-HSM star detections")
    timings["read"] = time.time() - t0

    patch_path = out / f"patches_{args.version}.npz"
    n_gal = len(gal["RA"])
    saved = dict(np.load(patch_path)) if patch_path.exists() else {}
    if (
        saved
        and len(saved["centres"]) == args.npatch
        and int(saved.get("n_galaxies", -1)) == n_gal
    ):
        centres = saved["centres"]
    else:
        every = max(1, len(gal["RA"]) // 2_000_000)
        sub = treecorr.Catalog(
            ra=gal["RA"][::every],
            dec=gal["Dec"][::every],
            ra_units="deg",
            dec_units="deg",
        )
        centres = jackknife_patch_centers(sub, args.npatch, seed=args.seed)
        np.savez(
            patch_path,
            centres=centres,
            npatch=args.npatch,
            n_galaxies=n_gal,
            every_nth=every,
            seed=args.seed,
            region=str(args.region),
        )
        log(f"patches -> {patch_path}")

    sources = None
    if {"stars", "tiles", "gaia"} & set(todo):
        sources = catalog(
            gal["RA"], gal["Dec"], centres, g1=gal["e1"], g2=gal["e2"], w=gal["w_des"]
        )
    nside, pixels = footprint_pixels(cdir / "occupied-mask-nside4096.npz")

    if "stars" in todo:
        t = time.time()
        ra, dec, mag = unique_stars(stars, args.dedup_arcsec)
        log(f"{len(ra)} unique stars from {len(stars)} detections")
        edges = quantile_edges(mag, args.mag_bins)
        results = {
            "all": measure(
                catalog(ra, dec, centres), sources, num_threads=args.threads, **TC
            )
        }
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            sel = (mag >= lo) & (mag < hi)
            results[f"mag{i}"] = measure(
                catalog(ra[sel], dec[sel], centres),
                sources,
                num_threads=args.threads,
                **TC,
            )
        timings["stars"] = time.time() - t
        save(
            out,
            "gamma_stars",
            args.version,
            results,
            dict(
                base,
                lenses="PSF validation stars (valid HSM), detections merged within "
                f"{args.dedup_arcsec} arcsec",
                n_lenses=len(ra),
                n_detections=len(stars),
                star_mag_edges=edges.tolist(),
                sources="galaxies, calibrated e1/e2, w_des",
                random_subtraction=False,
                seconds=timings["stars"],
            ),
        )

    if "tiles" in todo:
        t = time.time()
        names, ra, dec = tile_centres(tile_ids)
        results = {
            "all": measure(
                catalog(ra, dec, centres), sources, num_threads=args.threads, **TC
            )
        }
        timings["tiles"] = time.time() - t
        save(
            out,
            "gamma_tiles",
            args.version,
            results,
            dict(
                base,
                lenses="CFIS tile centres of the tiles in the catalogue",
                n_lenses=len(names),
                sources="galaxies, calibrated e1/e2, w_des",
                random_subtraction=False,
                seconds=timings["tiles"],
            ),
        )

    randoms = None
    if {"lambda", "gaia"} & set(todo):
        n_lens = int(args.lens_fraction * len(gal["RA"]))
        rra, rdec = footprint_randoms(
            int(args.random_factor * n_lens * 1.05), nside, pixels, args.region, rng
        )
        randoms = catalog(
            rra[: int(args.random_factor * n_lens)],
            rdec[: int(args.random_factor * n_lens)],
            centres,
        )
        log(f"{randoms.ntot} randoms")

    if "gaia" in todo:
        t = time.time()
        gaia = fitsio.read(args.gaia, columns=["ra", "dec", "phot_g_mean_mag", "ruwe"])
        keep = (gaia["phot_g_mean_mag"] < 20) & (gaia["ruwe"] < 1.4)
        keep &= in_region(gaia["ra"], gaia["dec"], args.region)
        keep &= in_footprint(gaia["ra"], gaia["dec"], nside, pixels)
        gaia = gaia[keep]
        g = gaia["phot_g_mean_mag"]
        results, counts = {}, {}
        for i, (lo, hi) in enumerate(zip(GAIA_G_EDGES[:-1], GAIA_G_EDGES[1:])):
            sel = (g >= lo) & (g < hi)
            counts[f"G{i}"] = int(sel.sum())
            results[f"G{i}"] = measure(
                catalog(
                    gaia["ra"][sel].astype(float),
                    gaia["dec"][sel].astype(float),
                    centres,
                ),
                sources,
                randoms=randoms,
                num_threads=args.threads,
                **TC,
            )
            log(f"gaia G{i} done")
        timings["gaia"] = time.time() - t
        save(
            out,
            "gamma_gaia",
            args.version,
            results,
            dict(
                base,
                lenses="Gaia DR3, G < 20, RUWE < 1.4, inside the occupied "
                "nside-4096 footprint",
                gaia_file=str(Path(args.gaia).resolve()),
                g_edges=list(GAIA_G_EDGES),
                n_lenses=counts,
                n_randoms=randoms.ntot,
                sources="galaxies, calibrated e1/e2, w_des",
                random_subtraction=True,
                seconds=timings["gaia"],
            ),
        )

    del sources
    if "lambda" in todo:
        t = time.time()
        pick = rng.random(len(gal["RA"])) < args.lens_fraction
        lenses = catalog(gal["RA"][pick], gal["Dec"][pick], centres)
        results = {}
        for label, g1, g2 in (
            ("psf", stars["HSM_G1_PSF"], stars["HSM_G2_PSF"]),
            (
                "residual",
                stars["HSM_G1_STAR"] - stars["HSM_G1_PSF"],
                stars["HSM_G2_STAR"] - stars["HSM_G2_PSF"],
            ),
        ):
            # The survey-mean ellipticity is an additive offset, not structure
            # correlated with galaxy positions; subtract it as TauStat does.
            g1 = g1 - np.mean(g1)
            g2 = g2 - np.mean(g2)
            src = catalog(stars["RA"], stars["DEC"], centres, g1=g1, g2=g2)
            results[label] = measure(
                lenses, src, randoms=randoms, num_threads=args.threads, **TC
            )
            log(f"lambda {label} done")
        timings["lambda"] = time.time() - t
        save(
            out,
            "lambda_stars",
            args.version,
            results,
            dict(
                base,
                lenses=f"random {args.lens_fraction:g} of the galaxies",
                n_lenses=lenses.ntot,
                n_randoms=randoms.ntot,
                sources="PSF validation star detections (valid HSM): psf = HSM_G*_PSF "
                "(lambda_1), residual = HSM_G*_STAR - HSM_G*_PSF (lambda_3), "
                "each with its mean subtracted",
                n_sources=len(stars),
                random_subtraction=True,
                seconds=timings["lambda"],
            ),
        )
        del lenses

    if "alpha" in todo:
        t = time.time()
        calib = json.loads((cdir / "calibration.json").read_text())
        r_sel = np.diag(np.asarray(calib["R_selection"]))
        results, prov = alpha_measurement(
            gal, stars, centres, args.mag_bins, args.threads, r_sel
        )
        timings["alpha"] = time.time() - t
        save(
            out,
            "alpha_mag",
            args.version,
            results,
            dict(base, **prov, seconds=timings["alpha"]),
        )

    log(f"timings (s): {json.dumps({k: round(v) for k, v in timings.items()})}")


if __name__ == "__main__":
    main()
