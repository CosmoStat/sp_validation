"""Aperture-mass error bars must use each mode's full jackknife variance."""

import numpy as np
import pytest
import treecorr
import yaml
from astropy.table import Table

from sp_validation.cosmo_val import CosmologyValidation

R_TOL = 0.1


def _schneider_kernels(gg, radii):
    """Independently construct the linear maps from (xi+, xi-) to E/B modes."""
    s = np.outer(1.0 / radii, gg.meanr)
    a = s[s < 2]
    a2 = a * a
    tp = np.zeros_like(s)
    tm = np.zeros_like(s)
    tp[s < 2] = 12 / (5 * np.pi) * (2 - 15 * a2) * np.arccos(a / 2) + a * np.sqrt(
        4 - a2
    ) / (100 * np.pi) * (120 + a2 * (2320 + a2 * (-754 + a2 * (132 - 9 * a2))))
    tm[s < 2] = 3 / (70 * np.pi) * a * a2 * (4 - a2) ** 3.5
    tp *= s * s
    tm *= s * s
    half = 0.5 * gg.bin_size
    kernel_e = np.concatenate([tp, tm], axis=1) * half
    kernel_b = np.concatenate([tp, -tm], axis=1) * half
    return kernel_e, kernel_b


@pytest.fixture
def cv(tmp_path):
    """A small synthetic catalogue: no survey data or theory calculation needed."""
    rng = np.random.default_rng(901)
    n = 600
    ra = rng.uniform(10, 14, n)
    dec = rng.uniform(10, 14, n)
    w = rng.uniform(0.5, 2, n)
    e1 = rng.normal(0.02, 0.08, n)
    e2 = rng.normal(0.01, 0.08, n)
    Table(dict(RA=ra, Dec=dec, e1=e1, e2=e2, w=w, tomo=np.where(ra < 12, 1, 2))).write(
        tmp_path / "shear.fits"
    )
    Table(dict(RA=ra[:20], Dec=dec[:20], e1=e1[:20], e2=e2[:20])).write(
        tmp_path / "star.fits"
    )
    nz = tmp_path / "dndz_SP_A.txt"
    np.savetxt(nz, np.array([[0.1, 1.0], [0.5, 1.0], [1.0, 0.1]]), header="z dn_dz")
    cfg = {
        "nz": {"subdir": str(tmp_path), "dndz": {"path": str(nz), "blind": "A"}},
        "paths": {"output": str(tmp_path / "out")},
        "Test": {
            "subdir": str(tmp_path),
            "pipeline": "SP",
            "shear": {
                "path": "shear.fits",
                "redshift_path": str(nz),
                "e1_col": "e1",
                "e2_col": "e2",
                "w_col": "w",
                "R": 1.0,
                "tomo_bin_col": "tomo",
            },
            "star": {
                "path": "star.fits",
                "ra_col": "RA",
                "dec_col": "Dec",
                "e1_col": "e1",
                "e2_col": "e2",
            },
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    validator = CosmologyValidation(
        ["Test"],
        catalog_config=str(config_path),
        output_dir=str(tmp_path / "run"),
        npatch=1,
        theta_min=5,
        theta_max=100,
        nbins=8,
    )
    validator.treecorr_config.update(num_threads=2, bin_slop=0, angle_slop=0)
    return validator


@pytest.mark.xfail(
    strict=True,
    reason=("#378: aperture-mass errors omit full jackknife covariance"),
)
def test_aperture_mass_variance_is_jackknife_not_diagonal_xi_propagation(
    cv, monkeypatch
):
    """Protect the plotted errors on <M_ap^2> (E) and <M_x^2> (B).

    With patches, the expected variance is TreeCorr's jackknife estimate_cov
    evaluated on calculateMapSq, rather than its diagonal-only varmapsq.
    Both statistics are linear in the binned (xi+, xi-) vector: M = K xi.
    The independent Schneider kernels below propagate the full jackknife
    covariance C as diag(K C K^T), which agrees with the direct jackknife within
    10%; the small residual comes from meanr varying per jackknife sample.
    E and B kernels differ in the sign of the xi- term, so their variances
    differ in general.
    TreeCorr's varmapsq instead propagates only diag(C), drops the xi+/xi-
    cross terms, and returns one array for both modes, matching neither.
    """
    captured = []
    original = treecorr.GGCorrelation

    def record(*args, **kwargs):
        gg = original(*args, **kwargs)
        captured.append(gg)
        return gg

    # Observe the actual measurement without modifying its covariance or output.
    with monkeypatch.context() as patch:
        patch.setattr(treecorr, "GGCorrelation", record)
        cv.calculate_aperture_mass_dispersion(
            theta_min=4, theta_max=80, nbins=20, nbins_map=2, npatch=4
        )

    gg = captured[-1]
    radii = cv.map2["theta_map"]
    entry = cv.map2["Test"]
    entry = entry.get("tomo_bin_all_tomo_bin_all", entry)

    def jk_var(idx):
        return np.diag(
            gg.estimate_cov(
                "jackknife",
                func=lambda c: c.calculateMapSq(R=radii, m2_uform="Schneider")[idx],
            )
        )

    var_e, var_b = jk_var(0), jk_var(2)
    kernel_e, kernel_b = _schneider_kernels(gg, radii)
    np.testing.assert_allclose(np.diag(kernel_e @ gg.cov @ kernel_e.T), var_e, rtol=0.1)
    np.testing.assert_allclose(np.diag(kernel_b @ gg.cov @ kernel_b.T), var_b, rtol=0.1)

    # Accept separate E/B variances without requiring the legacy shared key.
    reported_e = entry["varmapsq_E"] if "varmapsq_E" in entry else entry["varmapsq"]
    reported_b = entry.get("varmxsq", entry.get("varmapsq_B"))
    if reported_b is None:
        reported_b = entry["varmapsq"]
    assert np.allclose(reported_e, var_e, rtol=R_TOL, atol=0) and np.allclose(
        reported_b, var_b, rtol=R_TOL, atol=0
    ), (
        "aperture-mass variance is not the jackknife variance: "
        f"reported E={np.asarray(reported_e).tolist()} vs jackknife E={var_e.tolist()} "
        f"(ratio {(np.asarray(reported_e) / var_e).round(3).tolist()}); "
        f"reported B={np.asarray(reported_b).tolist()} vs jackknife B={var_b.tolist()} "
        f"(ratio {(np.asarray(reported_b) / var_b).round(3).tolist()})"
    )
