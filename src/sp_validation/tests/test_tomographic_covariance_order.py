"""Independent checks of tomographic covariance ordering."""

import runpy
from pathlib import Path

import numpy as np

from sp_validation import sacc_io as sio
from sp_validation.blinding import Blind
from sp_validation.cosmo_val.sacc_writers import xi_pairs_to_sacc


def test_onecov_assembly_matches_written_point_labels(tmp_path):
    pairs = [(i, j) for i in range(3) for j in range(i, 3)]
    theta = [2.0, 7.0, 20.0]
    nz = {i: (np.array([0.1, 0.5, 1.0]), np.array([0.1, 1.0, 0.1])) for i in range(3)}
    measurements = [
        (pair, dict(theta=theta, xip=np.zeros(3), xim=np.zeros(3)))
        for pair in reversed(pairs)
    ]
    part = xi_pairs_to_sacc(nz, {}, measurements, grid="reporting")
    part_path = tmp_path / "xi.sacc"
    sio.save(part, part_path, blind=Blind("none"))
    labels = [
        (dtype, pair, t)
        for dtype in (sio.XI_PLUS, sio.XI_MINUS)
        for pair in pairs
        for t in theta
    ]
    rng = np.random.default_rng(376)
    a = rng.normal(size=(len(labels), len(labels)))
    source = a + a.T
    cov_path = tmp_path / "onecov.dat"
    np.savetxt(cov_path, source, header="OneCovariance covariance matrix")
    script = Path(__file__).resolve().parents[3] / "workflow/scripts/assemble_sacc.py"
    assemble = runpy.run_path(str(script))["assemble_sacc"]
    output = tmp_path / "assembled.sacc"
    assemble("v", {"xi_reporting": part_path}, output, blind="none", xi_cov=cov_path)
    written = sio.load(output, blind="none")
    perm = [
        labels.index(
            (
                dp.data_type,
                tuple(int(t.removeprefix("source_")) for t in dp.tracers),
                dp.tags["theta"],
            )
        )
        for dp in written.data
    ]
    np.testing.assert_array_equal(written.covariance.dense, source[np.ix_(perm, perm)])


def test_tomographic_jackknife_matches_independent_pair_order(tmp_path):
    import treecorr
    from astropy.table import Table

    from sp_validation.cosmo_val import CosmologyValidation
    from sp_validation.statistics import jackknife_patch_centers
    from sp_validation.tests.test_cosmo_val import TestCosmologyValidation

    params, version = TestCosmologyValidation._write_synthetic_catalogs(
        tmp_path, n_gal=3000, with_tomography=True
    )
    path = tmp_path / "catalog/shear.fits"
    table = Table.read(path)
    table["tomo_bin_id"] = np.arange(len(table)) % 3 + 1
    table.write(path, overwrite=True)
    z = np.linspace(0.05, 3, 31)
    np.savetxt(
        tmp_path / "nz/dndz_SP_A.txt",
        np.column_stack([z] + [np.exp(-(((z - m) / 0.3) ** 2)) for m in (0.5, 1, 1.5)]),
        header="z bin1 bin2 bin3",
    )
    cv = CosmologyValidation(
        versions=[version], npatch=4, theta_min=5, theta_max=100, nbins=5, **params
    )
    cv.calculate_2pcf_version(version, compute_tomography=True, num_threads=1)
    part = cv.xi_tomo_parts[version, "reporting"]
    cols = cv._shear_columns(version, True)
    full = treecorr.Catalog(
        ra=cols["ra"],
        dec=cols["dec"],
        w=cols["w"],
        ra_units=cv.treecorr_config["ra_units"],
        dec_units=cv.treecorr_config["dec_units"],
    )
    centers = jackknife_patch_centers(full, 4)
    catalogs = {}
    for b in range(3):
        mask = cols["tomo_bin"] == b + 1
        catalogs[b] = treecorr.Catalog(
            ra=cols["ra"][mask],
            dec=cols["dec"][mask],
            g1=cols["g1"][mask],
            g2=cols["g2"][mask],
            w=cols["w"][mask],
            ra_units=cv.treecorr_config["ra_units"],
            dec_units=cv.treecorr_config["dec_units"],
            patch_centers=centers,
        )
    pairs = [(i, j) for i in range(3) for j in range(i, 3)][::-1]
    correlations = []
    patched_config = {**cv._binning(), "var_method": "jackknife"}
    # The covariance pass uses TreeCorr's default tolerance, independently of
    # the explicit bin_slop used for the full-sample means.
    patched_config.pop("bin_slop", None)
    for i, j in pairs:
        gg = treecorr.GGCorrelation(patched_config)
        gg.process(catalogs[i], cat2=catalogs[j] if i != j else None, num_threads=1)
        correlations.append(gg)
    independent = treecorr.estimate_multi_cov(correlations, "jackknife")
    perm = []
    for dp in part.data:
        pair = tuple(int(t.removeprefix("source_")) for t in dp.tracers)
        same = [
            p.tags["theta"]
            for p in part.data
            if p.data_type == dp.data_type and p.tracers == dp.tracers
        ]
        rank = sorted(same).index(dp.tags["theta"])
        component = (sio.XI_PLUS, sio.XI_MINUS).index(dp.data_type)
        perm.append(pairs.index(pair) * 10 + component * 5 + rank)
    np.testing.assert_allclose(
        part.covariance.dense, independent[np.ix_(perm, perm)], rtol=1e-12, atol=0
    )


def test_full_tomographic_assembly_covariance_blocks(tmp_path):
    from astropy.io import fits

    pairs = [(i, j) for i in range(3) for j in range(i, 3)]
    theta = [2.0, 7.0, 20.0]
    ell = [20.0, 50.0, 90.0]
    nz = {i: (np.array([0.1, 0.5, 1.0]), np.array([0.1, 1.0, 0.1])) for i in range(3)}
    xi = xi_pairs_to_sacc(
        nz,
        {},
        [
            (p, dict(theta=theta, xip=np.zeros(3), xim=np.zeros(3)))
            for p in reversed(pairs)
        ],
        grid="reporting",
    )
    cl, co, pure, rt = [sio.new_sacc(nz, {}) for _ in range(4)]
    for p in reversed(pairs):
        sio.add_pseudo_cl(
            cl,
            p,
            ell,
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            window_ells=ell,
            window_weights=np.eye(3),
        )
        sio.add_cosebis(co, p, np.zeros(3), (1.0, 100.0), Bn=np.zeros(3))
        sio.add_pure_eb(pure, p, theta, **{k: np.zeros(3) for k in sio.PURE_KEYS})
    for k in range(6):
        sio.add_rho(rt, k, theta, np.zeros(3), np.zeros(3))
    for b in reversed(range(3)):
        for k in (0, 2, 5):
            sio.add_tau(rt, (b, b), k, theta, np.zeros(3), np.zeros(3))
    parts = dict(xi_reporting=xi, pseudo_cl=cl, cosebis=co, pure_eb=pure, rho_tau=rt)
    rng = np.random.default_rng(376)

    def matrix(n, ident):
        a = rng.normal(size=(n, n))
        return ident * 1000 + a + a.T

    xi_source = matrix(len(xi.data), 1)
    xi_path = tmp_path / "onecov.dat"
    np.savetxt(xi_path, xi_source, header="OneCovariance covariance matrix")
    xi_labels = [
        (d, p, t) for d in (sio.XI_PLUS, sio.XI_MINUS) for p in pairs for t in theta
    ]
    cl_types = {sio.CL_EE: "EE", sio.CL_BB: "BB", sio.CL_EB: "EB"}
    cl_blocks = {
        d: matrix(len(pairs) * len(ell), 2 + i) for i, d in enumerate(cl_types)
    }
    cl_path = tmp_path / "inka.fits"
    fits.HDUList(
        [fits.PrimaryHDU()]
        + [
            fits.ImageHDU(cl_blocks[d], name=f"COVAR_{pol}_{pol}")
            for d, pol in cl_types.items()
        ]
    ).writeto(cl_path)

    expected = {}

    def label(dp, coordinate):
        return (
            dp.data_type,
            tuple(int(t.removeprefix("source_")) for t in dp.tracers),
            dp.tags[coordinate],
        )

    perm = [xi_labels.index(label(dp, "theta")) for dp in xi.data]
    expected["xi_reporting"] = xi_source[np.ix_(perm, perm)]
    cl_expected = np.zeros((len(cl.data), len(cl.data)))
    for dtype, block in cl_blocks.items():
        positions = [i for i, dp in enumerate(cl.data) if dp.data_type == dtype]
        order = [
            pairs.index(label(cl.data[i], "ell")[1]) * len(ell)
            + ell.index(cl.data[i].tags["ell"])
            for i in positions
        ]
        cl_expected[np.ix_(positions, positions)] = block[np.ix_(order, order)]
    expected["pseudo_cl"] = cl_expected
    paths = {}
    for ident, (name, part) in enumerate(parts.items(), 5):
        if name not in expected:
            expected[name] = matrix(len(part.data), ident)
            part.add_covariance(expected[name])
        paths[name] = tmp_path / f"{name}.sacc"
        sio.save(part, paths[name], blind=Blind("none"))
    script = Path(__file__).resolve().parents[3] / "workflow/scripts/assemble_sacc.py"
    assemble = runpy.run_path(str(script))["assemble_sacc"]
    out = tmp_path / "assembled.sacc"
    assemble("v", paths, out, blind="none", xi_cov=xi_path, pseudo_cl_cov=cl_path)
    s = sio.load(out, blind="none")
    indices = {}
    for name, part in parts.items():
        # Query each (type, tracers) separately, in the part's insertion order.
        keys = list(dict.fromkeys((dp.data_type, dp.tracers) for dp in part.data))
        indices[name] = np.concatenate(
            [s.indices(dtype, tracers) for dtype, tracers in keys]
        )
        part_indices = np.concatenate(
            [part.indices(dtype, tracers) for dtype, tracers in keys]
        )
        np.testing.assert_array_equal(
            s.covariance.dense[np.ix_(indices[name], indices[name])],
            expected[name][np.ix_(part_indices, part_indices)],
        )
    for i, name in enumerate(parts):
        for other in list(parts)[i + 1 :]:
            np.testing.assert_array_equal(
                s.covariance.dense[np.ix_(indices[name], indices[other])], 0
            )
