"""What a blind shifts, and what the catalogue config may declare.

Small SACCs exercise the default theory and Smokescreen's parameter draw.
The ``up`` fixture raises S8 above the fiducial for the E/B response tests.
"""

import json

import numpy as np
import pytest

from sp_validation import b_modes
from sp_validation import blinding as bd
from sp_validation import sacc_io as sio


@pytest.fixture(scope="module")
def up(tmp_path_factory):
    path = tmp_path_factory.mktemp("blinds") / "up.blind.json"
    path.write_text(
        json.dumps(
            {"seed": "s", "envelope": {"S8": [0.05, 0.06]}, "fiducial": bd.fiducial()}
        )
    )
    return bd.Blind("up", path)


def _sacc():
    z = np.linspace(0.0, 3.0, 200)
    return sio.new_sacc({0: (z, np.exp(-0.5 * ((z - 0.7) / 0.2) ** 2))})


@pytest.mark.slow
def test_a_blind_shifts_xi_and_cl_ee_and_leaves_bb_eb(up):
    s = _sacc()
    theta = np.geomspace(2.0, 200.0, 6)
    sio.add_xi(s, (0, 0), theta, 0 * theta, 0 * theta, grid="reporting")
    ell = np.array([30.0, 80.0, 150.0, 280.0, 450.0])
    w_ell = np.arange(2, 501).astype(float)
    window = np.exp(-0.5 * ((w_ell[:, None] - ell) / 40.0) ** 2)
    zeros = 0 * ell
    sio.add_pseudo_cl(
        s,
        (0, 0),
        ell,
        zeros,
        zeros,
        zeros,
        window_ells=w_ell,
        window_weights=window / window.sum(axis=0),
    )

    sealed = sio.seal(s, up)
    delta = sealed.mean - s.mean
    for data_type in (sio.XI_PLUS, sio.XI_MINUS, sio.CL_EE):
        assert np.all(delta[s.indices(data_type)] > 0), data_type
    for data_type in (sio.CL_BB, sio.CL_EB):
        assert np.all(delta[s.indices(data_type)] == 0), data_type
    assert sio.stamp(sealed) == "up"


@pytest.mark.slow
def test_pure_eb_under_a_blind_shifts_e_and_not_b(up):
    """The pure-E/B operator on the integration-grid ξ± is linear, so a blind
    moves the modes by its response to the shift: E by several σ, B by under
    a tenth of σ (σ: UNIONS shape noise through the operator)."""
    edges_int = np.geomspace(0.08, 300.0, 1001)
    theta_int = np.sqrt(edges_int[:-1] * edges_int[1:])
    s = _sacc()
    sio.add_xi(s, (0, 0), theta_int, 0 * theta_int, 0 * theta_int, grid="integration")
    shifted = sio.xi_correlation(sio.seal(s, up))
    delta = np.r_[shifted.xip, shifted.xim]

    annuli = np.diff(edges_int**2)  # ∝ pair counts on a uniform field
    operator, _, _ = b_modes.pure_eb_operator(
        annuli, edges_int, np.geomspace(1.0, 250.0, 21)
    )
    # Var ξ± = σ_e⁴ / (2 N_pairs), SP_v1.4.6.3's area, n_eff and σ_e.
    n_eff, area, sigma_e = 4.96, 2894.0 * 3600, 0.378
    variance = np.tile(sigma_e**4 / (n_eff**2 * area * np.pi * annuli), 2)
    sigma = np.sqrt(np.einsum("ij,j,ij->i", operator, variance, operator))
    response = operator @ delta / sigma

    n = len(response) // 6  # blocks in _EB_KEYS order: E+, E−, B+, B−, amb
    assert np.max(np.abs(response[: 2 * n])) > 1
    assert np.max(np.abs(response[2 * n : 4 * n])) < 0.1


def _small_xi():
    s = _sacc()
    theta = np.array([10.0, 50.0])
    sio.add_xi(s, (0, 0), theta, 0 * theta, 0 * theta, grid="reporting")
    return s


def test_distinct_blinds_and_determinism(tmp_path):
    catalogues = {"paths": {"blinds": str(tmp_path / "blinds")}}
    a, b = (bd.init(name, catalogues) for name in ("a", "b"))
    s = _small_xi()
    first = bd.conceal(s, a).mean
    np.testing.assert_array_equal(first, bd.conceal(s, a).mean)
    assert not np.array_equal(first, bd.conceal(s, b).mean)
    np.testing.assert_array_equal(s.mean, np.zeros(len(s.mean)))


def test_smokescreen_draw_stays_in_envelope(tmp_path):
    from smokescreen.datavector import concealing_factor

    blind = bd.init("envelope", {"paths": {"blinds": str(tmp_path)}})
    record = blind.record()
    keys = ("S8", "Omega_m")
    fid = np.array([record["fiducial"][k] for k in keys])
    envelope = np.array([record["envelope"][k] for k in keys])
    for seed in [record["seed"], *range(1000)]:
        # An identity theory observes the actual draw, without cosmology work.
        delta = concealing_factor(
            record["fiducial"],
            record["envelope"],
            seed=seed,
            theory_fn=lambda p: np.array([p[k] for k in keys]),
        )
        hidden = fid + delta
        assert np.all(hidden >= fid - envelope)
        assert np.all(hidden <= fid + envelope)


def test_multitracer_shift_matches_single_pair_theories(up, monkeypatch):
    import pyccl as ccl
    from smokescreen.datavector import draw_param_shifts

    record = up.record()
    shifts = draw_param_shifts(record["envelope"], record["seed"])
    fid = record["fiducial"]
    hidden = {k: value + shifts.get(k, 0.0) for k, value in fid.items()}
    z = np.linspace(0.0, 3.0, 100)
    s = sio.new_sacc(
        {
            i: (z, np.exp(-0.5 * ((z - centre) / 0.2) ** 2))
            for i, centre in enumerate((0.5, 1.1))
        }
    )
    pairs = ((0, 0), (0, 1), (1, 1))
    for pair in pairs:
        theta = np.array([10.0, 50.0])
        sio.add_xi(s, pair, theta, 0 * theta, 0 * theta, grid="reporting")
        ell = np.array([50.0, 100.0, 200.0])
        sio.add_pseudo_cl(
            s,
            pair,
            ell,
            0 * ell,
            0 * ell,
            0 * ell,
            window_ells=ell,
            window_weights=np.eye(3),
        )
    # Observe CCL's actual inputs independently of blinding's pair selection.
    make_tracer, angular_cl = ccl.WeakLensingTracer, ccl.angular_cl
    nz_by_tracer, seen = {}, []

    def tracer(*args, **kwargs):
        result = make_tracer(*args, **kwargs)
        nz_by_tracer[id(result)] = np.asarray(kwargs["dndz"][1]).copy()
        return result

    def spectrum(cosmo, a, b, ell, **kwargs):
        seen.append((nz_by_tracer[id(a)], nz_by_tracer[id(b)]))
        return angular_cl(cosmo, a, b, ell, **kwargs)

    monkeypatch.setattr(ccl, "WeakLensingTracer", tracer)
    monkeypatch.setattr(ccl, "angular_cl", spectrum)
    shifted = bd.conceal(s, up)
    for a, b in pairs:
        nz_a, nz_b = s.tracers[f"source_{a}"].nz, s.tracers[f"source_{b}"].nz
        # Each pair reaches angular_cl at fiducial and hidden points, for ξ and Cℓ.
        assert (
            sum(np.array_equal(x, nz_a) and np.array_equal(y, nz_b) for x, y in seen)
            == 4
        )
    for a, b in pairs:
        rows = s.indices(tracers=(f"source_{a}", f"source_{b}"))
        single = s.copy()
        single.keep_indices(rows)
        expected = bd.shear(hidden, single) - bd.shear(fid, single)
        np.testing.assert_allclose(
            shifted.mean[rows] - s.mean[rows], expected, rtol=1e-12, atol=0
        )
        for kind in (sio.XI_PLUS, sio.XI_MINUS, sio.CL_EE):
            assert np.any(expected[single.indices(kind)] != 0)


def test_saved_blind_is_secret_free(tmp_path, capfd):
    s = _small_xi()
    path = tmp_path / "sealed.sacc"
    blind = bd.init("secret-free", {"paths": {"blinds": str(tmp_path / "blinds")}})
    params_seen = []

    def theory(params, vector):
        params_seen.append(dict(params))
        return bd.shear(params, vector)

    written = sio.save(s, path, blind=blind, theory=theory)
    record = blind.record()
    hidden = next(p for p in params_seen if p != record["fiducial"])
    captured = capfd.readouterr()
    secrets = [record["seed"]]
    for key in ("S8", "Omega_m"):
        value = hidden[key]
        secrets.extend((repr(value), f"{value:.4f}", f"{value:.3f}"))
    for secret in secrets:
        assert secret.encode() not in path.read_bytes()
        assert secret not in captured.out + captured.err
    assert written.metadata == {**s.metadata, "blind": blind.name}
    loaded = sio.load(path)
    assert loaded.metadata == written.metadata
    np.testing.assert_array_equal(loaded.mean, written.mean)


def test_seal_refuses_a_sacc_already_stamped():
    sealed = sio.seal(_sacc(), bd.NONE)
    with pytest.raises(ValueError, match="already carries a blind stamp"):
        sio.seal(sealed, bd.NONE)


def test_entries_reading_one_file_declare_one_blind(tmp_path):
    catalogues = {
        "paths": {"blinds": "/blinds"},
        "Y3": {"blind": "y3", "subdir": "/d", "shear": {"path": "y3.fits"}},
        "TWIN": {"blind": "none", "subdir": "/d", "shear": {"path": "twin.fits"}},
    }
    assert bd.blind_of(catalogues, "TWIN") == "none"
    catalogues["TWIN"]["shear"]["path"] = "/d/sub/../y3.fits"
    with pytest.raises(ValueError, match="different blinds"):
        bd.blind_of(catalogues, "TWIN")

    physical = tmp_path / "y3.fits"
    physical.touch()
    alias = tmp_path / "alias.fits"
    alias.symlink_to(physical)
    catalogues["Y3"]["shear"]["path"] = str(physical)
    catalogues["TWIN"]["shear"]["path"] = str(alias)
    with pytest.raises(ValueError, match="different blinds"):
        bd.blind_of(catalogues, "TWIN")
    catalogues["TWIN"]["blind"] = "y3"
    assert bd.blind_of(catalogues, "TWIN") == "y3"
