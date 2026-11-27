"""Known-answer checks of sp_validation.cloe: the operators, the SACC data vector,
scale cuts, the coupled-bandpower covariance and the CLOE JAX theory."""

import json

import numpy as np
import numpy.testing as npt
import pytest
from scipy.special import eval_legendre

from sp_validation import sacc_io
from sp_validation.cloe import vector as v
from sp_validation.cloe.survey import Survey


# --------------------------------------------------------------------------- #
# operators
# --------------------------------------------------------------------------- #
def test_wigner_d_closed_forms():
    theta = np.linspace(0.01, 3.1, 7)
    x = np.cos(theta)
    d00 = v.wigner_d(10, 0, 0, theta)
    for ell in range(11):
        npt.assert_allclose(d00[ell], eval_legendre(ell, x), atol=1e-12)
    npt.assert_allclose(v.wigner_d(2, 2, 2, theta)[2], ((1 + x) / 2) ** 2, atol=1e-14)
    npt.assert_allclose(v.wigner_d(2, 2, -2, theta)[2], ((1 - x) / 2) ** 2, atol=1e-14)
    npt.assert_allclose(
        v.wigner_d(2, 2, 0, theta)[2], np.sqrt(3 / 8) * np.sin(theta) ** 2, atol=1e-14
    )


@pytest.mark.parametrize("stat", ["xip", "xim", "gammat", "w"])
def test_points_invert_by_orthogonality(stat):
    """C_ell = 2 pi int xi(theta) d^ell_{mn} sin(theta) dtheta, back from points."""
    lmax = 40
    rng = np.random.default_rng(1)
    cl = rng.uniform(0.5, 1.5, lmax + 1)
    cl[:2] = 0
    x, w = np.polynomial.legendre.leggauss(200)
    theta = np.arccos(x)
    xi = v.points(stat, theta, lmax) @ cl
    m, n = v.WIGNER[stat]
    d = v.wigner_d(lmax, m, n, theta)
    back = 2 * np.pi * d @ (w * xi)
    npt.assert_allclose(back[2:], cl[2:], rtol=1e-10)


def test_angular_is_the_area_average_of_points():
    edges = np.radians([1.0, 1.5])  # one annulus, 1 to 1.5 degrees
    lmax = 300
    t = np.linspace(*edges, 4001)
    weight = np.sin(t) / np.trapezoid(np.sin(t), t)
    direct = np.trapezoid(weight[:, None] * v.points("xip", t, lmax), t, axis=0)
    npt.assert_allclose(v.angular("xip", edges, lmax)[0], direct, atol=1e-6)


# --------------------------------------------------------------------------- #
# the SACC data vector and its cuts
# --------------------------------------------------------------------------- #
def three_by_two(order):
    """A small 3x2pt SACC: one lens, two sources, three bands per spectrum,
    random windows and covariance, spectra inserted in ``order``."""
    rng = np.random.default_rng(0)
    z = np.linspace(0.01, 2.0, 200)
    nz = [np.exp(-((z - m) ** 2) / 0.02) for m in (0.4, 0.8, 0.3)]
    s = sacc_io.new_sacc([(z, nz[0]), (z, nz[1])], lens_nz=[(z, nz[2])])
    spectra = {
        "gg": (sacc_io.CL_GG, ("lens_0", "lens_0")),
        "gl0": (sacc_io.CL_GE, ("source_0", "lens_0")),
        "ll01": (sacc_io.CL_EE, ("source_0", "source_1")),
    }
    ell = np.arange(51.0)
    windows = {}
    for key in order:
        dtype, tracers = spectra[key]
        windows[key] = rng.uniform(size=(51, 3))
        sacc_io.add_cl(
            s,
            dtype,
            tracers,
            [10.0, 25.0, 40.0],
            rng.normal(size=3),
            window_ells=ell,
            window_weights=windows[key],
        )
    a = rng.normal(size=(9, 9))
    s.add_covariance(a @ a.T + 9 * np.eye(9))
    return s, windows


def test_vector_from_sacc_reads_windows_data_and_covariance():
    s, windows = three_by_two(["ll01", "gg", "gl0"])
    vec = v.Vector.from_sacc(s)
    assert sorted(vec.pairs) == [("G0", "G0"), ("L0", "G0"), ("L0", "L1")]
    assert vec.gc_bins == [0] and vec.wl_bins == [0, 1]
    for i, (key, pair) in enumerate(zip(vec.keys, vec.pairs)):
        name = {("G0", "G0"): "gg", ("L0", "G0"): "gl0", ("L0", "L1"): "ll01"}[pair]
        # the operator on the nodes reproduces the window on any spectrum that the
        # node spline holds exactly: ell (ell + 1) C_ell constant
        cl = 1 / np.maximum(np.arange(51.0) * np.arange(1.0, 52.0), 1)
        cl[:2] = 0
        nodes = cl[vec.ells.astype(int)]
        npt.assert_allclose(
            vec.operators[i] @ nodes, windows[name].T @ cl, rtol=1e-10, atol=1e-15
        )
        rows = s.indices(
            *{
                "gg": (sacc_io.CL_GG, ("lens_0", "lens_0")),
                "gl0": (sacc_io.CL_GE, ("source_0", "lens_0")),
                "ll01": (sacc_io.CL_EE, ("source_0", "source_1")),
            }[name]
        )
        npt.assert_array_equal(vec.data[3 * i : 3 * i + 3], s.mean[rows])
    index = np.concatenate(
        [
            s.indices(d, t)
            for d, t in [
                (dt, tuple(p))
                for dt in s.get_data_types()
                for p in s.get_tracer_combinations(dt)
            ]
        ]
    )
    npt.assert_allclose(
        vec.icov @ s.covariance.dense[np.ix_(index, index)], np.eye(9), atol=1e-10
    )


def test_cut_keeps_shear_range_lens_autos_and_pairs():
    s, _ = three_by_two(["gg", "gl0", "ll01"])
    shear = v.cut(s, families=("LL",), shear=(20, 50))
    assert [d.get_tag("ell") for d in shear.data] == [25.0, 40.0]
    assert shear.covariance.dense.shape == (2, 2)
    # k_max h chi(<z> = 0.3) = 0.1 * 0.7 * 1230 Mpc = 86 > 40: every lens band
    # above ell 15 stays; the unlisted galaxy-shear pair goes; at k_max 0.04 the
    # cut is ell 34
    reference = dict(Omega_c=0.26, Omega_b=0.05, h=0.7, n_s=0.96, A_s=2.1e-9, m_nu=0.06)
    lens = v.cut(
        s,
        shear=(0, 100),
        lens_ell_min=15,
        lens_k_max={"GG": 0.1, "GL": 0.1},
        lens_source_pairs=[],
        reference=reference,
    )
    kinds = [(d.data_type, d.get_tag("ell")) for d in lens.data]
    assert kinds == [(sacc_io.CL_GG, 25.0), (sacc_io.CL_GG, 40.0)] + [
        (sacc_io.CL_EE, e) for e in (10.0, 25.0, 40.0)
    ]
    tight = v.cut(
        s,
        families=("GG",),
        lens_ell_min=15,
        lens_k_max={"GG": 0.04},
        reference=reference,
    )
    assert [d.get_tag("ell") for d in tight.data] == [25.0]


# --------------------------------------------------------------------------- #
# coupled bandpowers on the full sky
# --------------------------------------------------------------------------- #
@pytest.fixture
def full_sky_survey(tmp_path):
    z = np.linspace(0, 2, 41)
    np.savetxt(tmp_path / "nz.txt", np.c_[z, np.exp(-((z - 0.7) ** 2) / 0.05)])
    samples = {"bin1": {"n_eff": 2.0, "sigma_e_per_component": 0.3}}
    (tmp_path / "survey.json").write_text(json.dumps({"samples": samples}))
    return Survey(
        dict(
            nside=16,
            lmax=47,
            bands=dict(ell_min=4, ell_max=40, n=4),
            z_grid=[0.01, 1.99, 0.01],
            sources=dict(
                nz=str(tmp_path / "nz.txt"), survey=str(tmp_path / "survey.json")
            ),
            lenses=dict(
                count=1e6,
                area_deg2=41253.0,
                edges=[0.1, 0.3],
                bias=[1.0],
                fractions=[1.0],
                mag_slope=0.4,
                dndz_z0=0.2,
                sigma_z=0.01,
            ),
        )
    )


def test_full_sky_white_noise_covariance(full_sky_survey, tmp_path):
    """On the full sky the window is the binning, and white noise N alone gives
    Var(band) = sum_l w_l^2 2 N^2 / (2l + 1) for an auto (N^2 for counts x shear)."""
    from sp_validation.cloe import covariance as cv

    sv = full_sky_survey
    npix = 12 * 16**2
    masks = {"g": np.ones(npix), "l": np.ones(npix)}
    couplings = cv.Couplings(masks, sv["lmax"], tmp_path / "couplings")
    op = cv.operators(sv, masks, couplings)
    for o in op:
        npt.assert_allclose(o, sv.binning, rtol=1e-4, atol=1e-7)  # float32 couplings
    cl = np.zeros((len(sv.pairs), sv["lmax"] + 1))
    cov = cv.gaussian(sv, masks, cl, couplings, processes=1)
    noise = cv.noise(sv, masks)
    ell = np.arange(sv["lmax"] + 1)
    nb = len(sv.binning)
    for p, (a, b) in enumerate(sv.pairs):
        block = cov[p * nb : (p + 1) * nb, p * nb : (p + 1) * nb]
        factor = 2 if a == b else 1
        expect = sv.binning**2 @ (factor * noise[a] * noise[b] / (2 * ell + 1))
        npt.assert_allclose(np.diag(block), expect, rtol=1e-5)


# --------------------------------------------------------------------------- #
# the theory
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_theory_noiseless_vector_and_s8_basis():
    """A filled vector has chi^2 = 0 at its point and a finite gradient; in the S8
    basis the derived S8 is the sampled one."""
    import jax
    import jax.numpy as jnp

    z = np.arange(0.005, 3.0, 0.01)
    nz = {
        "L0": (z, np.exp(-((z - 0.6) ** 2) / 0.05)),
        "L1": (z, np.exp(-((z - 1.0) ** 2) / 0.08)),
    }
    ell = np.arange(1001)
    edges = np.geomspace(30, 1000, 9).astype(int)
    window = np.array([(ell >= a) & (ell < b) for a, b in zip(edges[:-1], edges[1:])])
    window = window / window.sum(1, keepdims=True)
    pairs = [("L0", "L0"), ("L0", "L1"), ("L1", "L1")]
    vec = v.Vector([v.Block("cl", pairs, window)], nz, cov=np.eye(24) * 1e-20)
    fixed = dict(m_nu=0.06, ombh2=0.0224, n_s=0.965, HMCode_logT_AGN=7.8)
    names = ["S8", "omch2", "h", "A_IA"]
    model = v.Theory(vec, names, fixed=fixed)
    theta = jnp.array([0.8, 0.12, 0.68, 0.5])
    model.fill(theta)
    assert float(model.chi2(theta)) == pytest.approx(0, abs=1e-6)
    grad = jax.grad(model.loglike)(theta + jnp.array([0.01, 0, 0, 0]))
    assert np.all(np.isfinite(np.asarray(grad))) and float(grad[0]) < 0
    npt.assert_allclose(float(model.derived(theta)[1]), 0.8, rtol=1e-4)
