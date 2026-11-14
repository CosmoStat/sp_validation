"""Known-answer checks for the clustering-redshift estimator."""

import numpy as np
import pytest

from sp_validation import nz_clustering as nc


def gaussian(z, mean, sigma=0.12):
    return np.exp(-0.5 * ((z - mean) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def test_estimator_recovers_bias_and_density():
    """w_rr = b_r² M / Δz and w_ur = n b_u b_r M give back b_r and n b_u."""
    cosmo = nc.fiducial_cosmology()
    edges = nc.slice_edges(0.4, 0.8, 0.05)
    z = (edges[:-1] + edges[1:]) / 2
    annulus = (1.5, 5.0)
    sel = (nc.R_EDGES[:-1] >= 1.49) & (nc.R_EDGES[1:] <= 5.05)
    M = np.array([nc.kernel(cosmo, x, nc.R_EDGES[:-1][sel][0], nc.R_EDGES[1:][sel][-1]) for x in z])
    b_r, nb = 2.0, 1.4 * gaussian(z, 0.6)
    xi = np.zeros((len(z), 2, len(nc.R_EDGES) - 1))
    xi[:, 0] = (b_r**2 * M / 0.05)[:, None]
    xi[:, 1] = (nb * b_r * M)[:, None]
    m = nc.Measurement(
        dict(z_edges=edges, samples=["auto", "bin1"], xi=xi, jackknife=xi[None]),
        annulus,
        cosmo,
    )
    assert np.allclose(m.bias(), b_r)
    assert np.allclose(m.nb("bin1"), nb)
    assert np.allclose(m.nb("bin1", jk=True)[0], nb)


def test_shift_fit_recovers_a_shift():
    grid = np.linspace(0, 5, 101)
    lo = nc.slice_edges(0.2, 1.0, 0.05)[:-1]
    hi = lo + 0.05
    y = 3.0 * nc.slice_average(grid, gaussian(grid, 0.6), lo, hi, 0.03)
    fit = nc.shift_fit(
        y,
        np.eye(len(y)) * 1e4,
        lambda s: nc.slice_average(grid, gaussian(grid, 0.6), lo, hi, s),
    )
    assert fit["best"] == pytest.approx(0.03, abs=0.0026)
    assert fit["lo"] <= 0.03 <= fit["hi"]


def test_profiled_shift_with_correlated_noise():
    rng = np.random.default_rng(12)
    grid = np.linspace(0, 2, 401)
    lo = nc.slice_edges(0.15, 1.15, 0.05)[:-1]
    hi, z = lo + 0.05, lo + 0.025
    nz = gaussian(grid, 0.5, 0.09) + 0.6 * gaussian(grid, 0.85, 0.07)

    def model(s):
        return nc.slice_average(grid, nz, lo, hi, s)

    cov = 0.03**2 * (0.8 ** np.abs(np.subtract.outer(np.arange(len(z)), np.arange(len(z)))))
    # Generate leave-one-out rows with exactly this sample covariance.
    n = 200
    rows = rng.multivariate_normal(np.zeros(len(z)), cov / (n - 1), n)
    precision = nc.jackknife_precision(rows)
    np.testing.assert_allclose(
        precision @ nc.jackknife_covariance(rows),
        np.eye(len(z)) * (n - len(z) - 2) / (n - 1), atol=1e-12,
    )
    y = 2 * model(0.03) * (1 + z) ** 2 + rng.multivariate_normal(np.zeros(len(z)), cov)
    fit = nc.profiled_shift_fit(y, precision, model, z)
    fixed = nc.shift_fit(y, precision, lambda s: model(s) * (1 + z) ** 2)
    assert fit["best"] == pytest.approx(0.03, abs=0.005)
    assert fit["lo"] <= 0.03 <= fit["hi"]
    assert fit["hi"] - fit["lo"] >= fixed["hi"] - fixed["lo"]
    assert not fit["alpha_boundary"]
    assert fit["alpha"] == pytest.approx(2, abs=0.4)
    m = model(0)
    A = (m @ precision @ y) / (m @ precision @ m)
    residual = y - A * m
    assert nc.amplitude_fit(y, precision, m) == pytest.approx(residual @ precision @ residual)
    diagonal = np.diag(np.diag(precision))
    assert not np.isclose(nc.amplitude_fit(y, precision, m), nc.amplitude_fit(y, diagonal, m))
    boundary = nc.profiled_shift_fit(2 * model(0) * (1 + z) ** 6, precision, model, z)
    assert boundary["alpha_boundary"]
    assert boundary["alpha"] == 4


def test_alpha_boundary_includes_entire_shift_interval(monkeypatch):
    monkeypatch.setattr(nc, "ALPHAS", np.array([-1., 0., 1.]))
    monkeypatch.setattr(nc, "SHIFTS", np.array([-0.1, 0., 0.1]))
    z = np.linspace(0.2, 1, 8)

    def model(s):
        return (1 + z) ** (-10 * s) * (1 + 2 * s**2 * z**2)

    fit = nc.profiled_shift_fit(np.ones(8), np.eye(8) * 100, model, z)
    assert fit["best"] == 0
    assert fit["alpha"] == 0
    assert fit["alpha_boundary"]
    # The edge solutions leave the interval when the data are more precise.
    tight = nc.profiled_shift_fit(np.ones(8), np.eye(8) * 1e8, model, z)
    assert tight["lo"] == tight["hi"] == 0
    assert not tight["alpha_boundary"]


def test_analysis_reports_profile_and_renders_figures(tmp_path, monkeypatch):
    """Exercise diagnostic serialization and figures without modifying SOM products."""
    monkeypatch.setattr(nc, "fiducial_cosmology", lambda: None)
    monkeypatch.setattr(nc, "kernel", lambda *args: 0.01)
    rng = np.random.default_rng(5)
    data = tmp_path / "clustering" / "data"
    data.mkdir(parents=True)
    grid = np.linspace(0, 2, 101)
    hists = np.array([gaussian(grid, z) for z in (0.45, 0.5, 0.55, 0.6)])
    product = tmp_path / "calibration_arrays.npz"
    np.savez(product, grid=grid, bin_hists=hists)
    before = product.read_bytes()
    edges = nc.slice_edges(0.3, 0.9, 0.05)
    n = len(edges) - 1
    xi = np.ones((n, 5, len(nc.R_EDGES) - 1)) * 0.2
    xi[:, 1:] *= 0.3
    jk = xi + rng.normal(0, 0.002, (80, *xi.shape))
    out = dict(z_edges=edges, samples=["auto", "bin1", "bin2", "bin3", "bin4"],
               xi=xi, jackknife=jk)
    np.savez(data / "xcorr_LRG.npz", **out)
    np.savez(data / "xcorr_gate.npz", z_edges=edges, samples=["auto", "half"],
             xi=xi[:, :2], jackknife=jk[:, :, :2])
    np.savez(data / "LRG_dat.npz", z=rng.uniform(0.3, 0.9, 2000), w=np.ones(2000))
    config = dict(output=str(tmp_path), clustering=dict(
        output=str(data.parent), reference=dict(slices={"LRG": [0.3, 0.9, 0.05]}),
        annulus=[1.5, 5], core=[[0.3, 0.9]] * 4, gate=["LRG", 0.3, 0.9, 0.05],
    ))
    result = nc.analyse(config, tmp_path / "figures")
    assert "profiled_shift" in result["bins"][0]["fits"]["calibration"]
    assert np.asarray(result["bins"][0]["covariance"]).shape == (n, n)
    assert (data.parent / "clustering.json").exists()
    for name in ("clustering_gate.png", "clustering_nz.png"):
        assert (tmp_path / "figures" / name).stat().st_size > 1000
    assert product.read_bytes() == before


@pytest.mark.parametrize("window", [(0.31, 0.5), (0.3, 0.51)])
def test_overlap_edges_must_preserve_nominal_grid(window):
    cl = dict(reference=dict(slices={"A": [0.2, 0.4, 0.05], "B": [0.4, 0.7, 0.05]},
                             overlaps=[["A", "B", *window]]))
    with pytest.raises(ValueError, match="dz=.*grid"):
        nc.reference_edges(cl, "A")


@pytest.mark.parametrize("missing", ["dat", "ran"])
def test_prepare_rejects_unsupported_overlap_before_sources(tmp_path, monkeypatch, missing):
    import healpy as hp

    monkeypatch.setattr(hp, "read_map", lambda _: np.ones(12))

    def reference(directory, tracer, kind, n_random):
        z = np.arange(0.825, 1.1, 0.05)
        if kind == missing:
            z = z[z >= 0.9]
        cat = np.zeros(len(z), dtype=[(name, "f8") for name in ("RA", "DEC", "Z", "WEIGHT")])
        cat["Z"], cat["WEIGHT"] = z, 1
        return cat

    monkeypatch.setattr(nc, "_read_reference", reference)
    cl = dict(output=str(tmp_path), footprint="unused", reference=dict(
        directory="unused", randoms=1,
        slices={"LRG": [0.4, 0.8, 0.05], "ELG": [0.8, 1.6, 0.05]},
        overlaps=[["LRG", "ELG", 0.8, 1.1]],
    ))
    # No source configuration: preparation must stop at reference support.
    with pytest.raises(ValueError, match=f"LRG {missing} lacks redshift support"):
        nc.prepare(dict(clustering=cl))


def test_overlapping_tracers_recover_same_nz_with_different_biases(tmp_path, monkeypatch):
    monkeypatch.setattr(nc, "kernel", lambda *args: 0.001)
    data = tmp_path / "data"
    data.mkdir()
    cl = dict(output=str(tmp_path), annulus=[1.5, 5], reference=dict(
        slices={"A": [0.2, 0.4, 0.05], "B": [0.4, 0.7, 0.05]},
        overlaps=[["A", "B", 0.3, 0.5]],
    ))
    np.testing.assert_allclose(nc.reference_edges(cl, "A"), nc.slice_edges(0.2, 0.5, 0.05))
    np.testing.assert_allclose(nc.reference_edges(cl, "B"), nc.slice_edges(0.3, 0.7, 0.05))
    rng = np.random.default_rng(7)
    shared = rng.normal(0, 0.01, (80, 10))
    for tracer, norm, slope in (("A", 1.2, 0.5), ("B", 2.0, 2.5)):
        edges = nc.reference_edges(cl, tracer)
        z = (edges[:-1] + edges[1:]) / 2
        index = np.rint((z - 0.225) / 0.05).astype(int)
        br = norm * (1 + z) ** slope
        nb = 2 * gaussian(z, 0.45) * (1 + z) ** 1.3
        xi = np.zeros((len(z), 2, len(nc.R_EDGES) - 1))
        xi[:, 0] = (br**2 * 0.001 / 0.05)[:, None]
        xi[:, 1] = (nb * br * 0.001)[:, None]
        jk = np.broadcast_to(xi, (80, *xi.shape)).copy()
        fluctuation = shared[:, index] + rng.normal(0, 0.002, (80, len(z)))
        jk[:, :, 1] += (fluctuation * br * 0.001)[..., None]
        np.savez(data / f"xcorr_{tracer}.npz", z_edges=edges,
                 samples=["auto", "bin1"], xi=xi, jackknife=jk)
    comparison = dict(bins=[dict(fits={"calibration": dict(amplitude=2, alpha=1.3)})])
    out = nc.tracer_overlaps(dict(clustering=cl), None, comparison)[0]
    result = out["bins"]["bin1"]
    z = (np.array(out["z_lo"]) + out["z_hi"]) / 2
    for tracer in ("A", "B"):
        np.testing.assert_allclose(result["n"][tracer], gaussian(z, 0.45), rtol=1e-12)
    assert result["consistent_95"]
    assert result["dof"] == 4
    assert result["pte"] == pytest.approx(1)
    # Paired patches remove the common source fluctuation from the difference.
    assert np.diag(result["difference_covariance"]).max() < 0.002
    original = dict(np.load(data / "xcorr_B.npz"))
    original["xi"][:, 1] *= 1.2
    np.savez(data / "xcorr_B.npz", **original)
    bad = nc.tracer_overlaps(dict(clustering=cl), None, comparison)[0]["bins"]["bin1"]
    assert not bad["consistent_95"]
    assert bad["pte"] < 0.05


def test_precision_requires_enough_patches():
    with pytest.raises(ValueError, match="patches"):
        nc.jackknife_precision(np.ones((6, 4)))


def test_correlate_finds_no_signal_between_random_points():
    rng = np.random.default_rng(0)

    def points(n):
        return dict(
            ra=rng.uniform(0, 6, n),
            dec=np.degrees(np.arcsin(rng.uniform(0, 0.1, n))),
            z=rng.uniform(0.4, 0.5, n),
            w=np.ones(n),
        )

    dat, ran, unk = points(4000), points(20000), points(4000)
    import treecorr

    centers = treecorr.Catalog(
        ra=ran["ra"], dec=ran["dec"], ra_units="deg", dec_units="deg", npatch=10
    ).patch_centers
    out = nc.correlate(
        dat,
        ran,
        {"u": (unk["ra"], unk["dec"], unk["w"])},
        nc.slice_edges(0.4, 0.5, 0.05),
        centers,
        nc.fiducial_cosmology(),
    )
    assert out["xi"].shape == (2, 2, len(nc.R_EDGES) - 1)
    assert out["jackknife"].shape == (10, 2, 2, len(nc.R_EDGES) - 1)
    sel = nc.R_EDGES[:-1] >= 2
    err = out["jackknife"][..., sel].mean(-1).std(0) * 3
    assert np.all(np.abs(out["xi"][..., sel].mean(-1)) < 4 * err + 0.05)
