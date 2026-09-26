"""The blind and the file door: invariants I1–I11.

A blind is drawn once (``blinding init``) into a registry beside a catalogue
config; ``sacc_io.save`` is the only writer, and conceals a blinded
catalogue's ξ± and Cℓ_EE rows in memory before the file exists. Blinds here
use the fast Eisenstein–Hu theory, so the whole module runs in the fast suite;
``test_camb_ccl_crosscheck.py`` covers the production CAMB recipe.
"""

import dataclasses
import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest
import yaml

from sp_validation import blinding as bd
from sp_validation import custody as cu
from sp_validation import sacc_io as sio
from sp_validation.blinding_theory import TheoryConfig

FAST = {"theory": {"transfer_function": "eisenstein_hu"}}
DATA = Path(__file__).parent / "data"
# The hidden (S8, Ωm) of the blind committed under DATA/blinds/committed.
COMMITTED_POINT = (0.8374952413635888, 0.32730573347638175)


# --------------------------------------------------------------------------- #
# A registry with real blinds, and custody under them
# --------------------------------------------------------------------------- #
def _cat_config(root):
    """A catalogue config declaring one catalogue per custody state."""
    entries = {
        "TOY": {},
        "OTHER": {},
        "TOY_OPEN": {"blinding": "unblinded"},
        "TOY_MOCK": {"blinding": "mock"},
        "paths": {"output": str(root / "output")},
    }
    path = root / "cat_config.yaml"
    path.write_text(yaml.safe_dump(entries))
    (root / "fast.json").write_text(json.dumps(FAST))
    return path


def _init(root, blind, *bases):
    bd.main(
        [
            "init",
            blind,
            *bases,
            "--cat-config",
            str(root / "cat_config.yaml"),
            "--config",
            str(root / "fast.json"),
        ]
    )


def _custody(root, version):
    path = root / "cat_config.yaml"
    return cu.custody_of(
        yaml.safe_load(path.read_text()), version, registry=cu.registry_of(path)
    )


@pytest.fixture(scope="module")
def blinds(tmp_path_factory):
    """Blinds `toy` (TOY) and `other` (OTHER), and the four custodies.

    Their seeds are fixed, so every run tests the same hidden points.
    """
    root = tmp_path_factory.mktemp("registry")
    _cat_config(root)
    with pytest.MonkeyPatch.context() as m:
        for blind, base in (("toy", "TOY"), ("other", "OTHER")):
            m.setattr(bd.secrets, "token_hex", lambda n, b=blind: f"{b}-seed")
            _init(root, blind, base)
    return {v: _custody(root, v) for v in ("TOY", "OTHER", "TOY_OPEN", "TOY_MOCK")}


@pytest.fixture
def fresh(tmp_path):
    """A registry of its own, for tests that edit or reveal a blind."""
    _cat_config(tmp_path)
    _init(tmp_path, "toy", "TOY")
    return tmp_path


# --------------------------------------------------------------------------- #
# Synthetic parts
# --------------------------------------------------------------------------- #
def _gauss_nz(z0, sigma, n=200):
    z = np.linspace(0.0, 3.0, n)
    nz = np.exp(-0.5 * ((z - z0) / sigma) ** 2)
    return z, nz / np.trapezoid(nz, z)


NZ2 = {0: _gauss_nz(0.5, 0.15), 1: _gauss_nz(0.9, 0.2)}
PAIRS = ((0, 0), (0, 1), (1, 1))
# The ξ± tags a part may carry; blocks never depend on them (None: no tag).
XI_TAGS = ("reporting", "integration", "cosebis", "mystery", None)


def _xi_template(theta, k=0):
    xip = 1e-4 * (1 + 0.1 * k) * (theta / 10.0) ** -0.6
    xim = 0.5e-4 * (1 + 0.1 * k) * (theta / 10.0) ** -0.9
    return xip, xim


def _add_xi_rows(s, pair, theta, tag, k=0):
    xip, xim = _xi_template(theta, k)
    if tag is not None:
        sio.add_xi(s, pair, theta, xip, xim, grid=tag)
        return
    tracers = sio._pair(pair)
    for dtype, values in ((sio.XI_PLUS, xip), (sio.XI_MINUS, xim)):
        for th, v in zip(theta, values):
            s.add_data_point(dtype, tracers, float(v), theta=float(th))


def _add_cl_rows(s, pair, k=0):
    ell_eff = np.array([30.0, 80.0, 150.0, 280.0, 450.0])
    w_ell = np.arange(2, 501).astype(float)
    w_mat = np.exp(-0.5 * ((w_ell[:, None] - ell_eff[None, :]) / 40.0) ** 2)
    w_mat /= w_mat.sum(axis=0)
    ee = 1e-8 * (1 + 0.1 * k) * (ell_eff / 100.0) ** -1.2
    sio.add_pseudo_cl(
        s,
        pair,
        ell_eff,
        ee,
        0.01 * ee,
        0.02 * ee,
        window_ells=w_ell,
        window_weights=w_mat,
    )


def _add_rho_rows(s):
    theta = np.geomspace(5.0, 250.0, 6)
    sio.add_rho(s, 0, theta, np.arange(6) * 1e-7, np.arange(6) * 2e-7)
    sio.add_tau(s, (0,), 0, theta, np.arange(6) * 3e-7, np.arange(6) * 4e-7)


def two_bin_sacc(*, xi_tags=XI_TAGS, cl=True, rho=False, covariance=True):
    """ξ± on every tag in ``xi_tags`` and pseudo-Cℓ, over pairs (0,0), (0,1), (1,1)."""
    s = sio.new_sacc(NZ2, metadata={"catalogue_version": "TOY"})
    for k, pair in enumerate(PAIRS):
        for n, tag in enumerate(xi_tags):
            _add_xi_rows(s, pair, np.geomspace(2.0, 200.0, 6) * (1 + 0.013 * n), tag, k)
        if cl:
            _add_cl_rows(s, pair, k)
    if rho:
        _add_rho_rows(s)
    if covariance:
        rng = np.random.default_rng(3)
        s.add_covariance(
            np.abs(np.asarray(s.mean)) ** 2 * rng.uniform(1, 2, len(s.mean))
        )
    return s


def cosebis_sacc():
    s = sio.new_sacc({0: NZ2[0]}, metadata={"catalogue_version": "TOY"})
    sio.add_cosebis(s, (0, 0), np.arange(1, 6) * 1e-10, (12.0, 83.0), Bn=np.ones(5))
    return s


def pure_eb_sacc():
    s = sio.new_sacc({0: NZ2[0]}, metadata={"catalogue_version": "TOY"})
    theta = np.geomspace(2.0, 200.0, 4)
    sio.add_pure_eb(s, (0, 0), theta, **{k: np.ones(4) for k in sio.PURE_KEYS})
    return s


def rho_sacc():
    s = sio.new_sacc({0: NZ2[0]}, metadata={"catalogue_version": "TOY"})
    _add_rho_rows(s)
    return s


def _rows(s, *types):
    return np.array([i for i, dp in enumerate(s.data) if dp.data_type in types])


# --------------------------------------------------------------------------- #
# An independent CCL reference for the shift
# --------------------------------------------------------------------------- #
def _reference_params(seed):
    """The hidden and fiducial CCL points, drawn with the fork directly."""
    from smokescreen.param_shifts import draw_param_shifts

    shift = draw_param_shifts({"S8": 0.075, "Omega_m": 0.1}, seed)

    def ccl_point(s8, om):
        h, ob, mnu = 0.70, 0.0469, 0.06
        return dict(
            Omega_c=om - ob - mnu / (93.14 * h**2),
            Omega_b=ob,
            h=h,
            n_s=0.96,
            sigma8=s8 / np.sqrt(om / 0.3),
            m_nu=mnu,
            mass_split="normal",
            w0=-1.0,
            wa=0.0,
            Neff=3.046,
            T_CMB=2.7255,
        )

    return (
        ccl_point(0.80 + shift["S8"], 0.30 + shift["Omega_m"]),
        ccl_point(0.80, 0.30),
    )


def reference_shift(s, seed):
    """t(hidden) − t(fiducial) on every ξ± and Cℓ_EE row of ``s``, from scratch."""
    import pyccl as ccl

    hidden, fiducial = _reference_params(seed)
    ell = np.unique(
        np.concatenate([np.arange(2, 50), np.geomspace(50, 6e4, 200)]).astype(float)
    )

    def cosmo(p):
        return ccl.Cosmology(
            **p, transfer_function="eisenstein_hu", matter_power_spectrum="halofit"
        )

    def spectra(p, tracers, ells):
        c = cosmo(p)
        a, b = (s.tracers[t] for t in tracers)
        return c, ccl.angular_cl(
            c,
            ccl.WeakLensingTracer(c, dndz=(a.z, a.nz)),
            ccl.WeakLensingTracer(c, dndz=(b.z, b.nz)),
            ells,
        )

    out = np.full(len(s.mean), np.nan)
    for tracers in {dp.tracers for dp in s.data if dp.data_type in sio.SHIFTABLE}:
        for dtype, kind in ((sio.XI_PLUS, "GG+"), (sio.XI_MINUS, "GG-")):
            rows = [
                i
                for i, dp in enumerate(s.data)
                if (dp.data_type, dp.tracers) == (dtype, tracers)
            ]
            theta = np.array([s.data[i].tags["theta"] for i in rows]) / 60.0
            xi = []
            for p in (hidden, fiducial):
                c, cl = spectra(p, tracers, ell)
                xi.append(ccl.correlation(c, ell=ell, C_ell=cl, theta=theta, type=kind))
            out[rows] = xi[0] - xi[1]
        rows = [
            i
            for i, dp in enumerate(s.data)
            if (dp.data_type, dp.tracers) == (sio.CL_EE, tracers)
        ]
        for i in rows:
            window = s.data[i].tags["window"]
            ells = np.asarray(window.values, float)
            w = np.asarray(window.weight, float)[:, s.data[i].tags["window_ind"]]
            out[i] = w @ (
                spectra(hidden, tracers, ells)[1] - spectra(fiducial, tracers, ells)[1]
            )
    return out


# --------------------------------------------------------------------------- #
# I1: the shift lands on every ξ± and Cℓ_EE row, from the content
# --------------------------------------------------------------------------- #
def test_shift_is_the_theory_difference_on_every_shiftable_row(blinds):
    """seal(s) − s equals an independent CCL reference on every ξ± and Cℓ_EE row.

    Two bins, all three pairs; ξ± rows under every tag and none, so blocks are
    discovered from the data types and tracers, never from grid names.
    """
    s = two_bin_sacc()
    sealed = sio.seal(s, blinds["TOY"])
    shift = np.asarray(sealed.mean) - np.asarray(s.mean)
    expected = reference_shift(s, bd.open_blind(blinds["TOY"]).seed)

    rows = _rows(s, sio.XI_PLUS, sio.XI_MINUS, sio.CL_EE)
    assert len(rows) and np.all(np.isfinite(expected[rows]))
    for pair in PAIRS:
        tracers = sio._pair(pair)
        for types in ((sio.XI_PLUS, sio.XI_MINUS), (sio.CL_EE,)):
            block = [
                i
                for i in rows
                if s.data[i].tracers == tracers and s.data[i].data_type in types
            ]
            scale = np.max(np.abs(expected[block]))
            gap = np.max(np.abs(shift[block] - expected[block]))
            assert scale > 0 and gap <= 1e-8 * scale, (pair, types, gap / scale)


# --------------------------------------------------------------------------- #
# I2: nothing else moves
# --------------------------------------------------------------------------- #
def test_only_shiftable_values_move(blinds):
    """BB, EB and ρ/τ values, every tag, tracer, row and the covariance are bitwise kept."""
    s = two_bin_sacc(rho=True)
    sealed = sio.seal(s, blinds["TOY"])
    moved = _rows(s, sio.XI_PLUS, sio.XI_MINUS, sio.CL_EE)
    kept = np.setdiff1d(np.arange(len(s.mean)), moved)
    assert len(kept) and np.array_equal(
        np.asarray(sealed.mean)[kept], np.asarray(s.mean)[kept]
    )
    assert np.all(np.asarray(sealed.mean)[moved] != np.asarray(s.mean)[moved])
    assert np.array_equal(sealed.covariance.dense, s.covariance.dense)
    for a, b in zip(s.data, sealed.data):
        assert (a.data_type, a.tracers) == (b.data_type, b.tracers)
        assert {k: v for k, v in a.tags.items() if k != "window"} == {
            k: v for k, v in b.tags.items() if k != "window"
        }
    for name, tracer in s.tracers.items():
        if hasattr(tracer, "nz"):
            assert np.array_equal(sealed.tracers[name].nz, tracer.nz)


def test_derived_statistics_keep_their_values(blinds, tmp_path):
    """COSEBIs and pure-E/B written as derivations carry their values unchanged."""
    xi = sio.seal(two_bin_sacc(cl=False), blinds["TOY"])
    for s in (cosebis_sacc(), pure_eb_sacc()):
        path = tmp_path / "derived.sacc"
        sio.save(s, path, derived_from=[xi])
        assert np.array_equal(np.asarray(sio.load(path).mean), np.asarray(s.mean))


# --------------------------------------------------------------------------- #
# I3: the hidden point is uniform in the physical (S8, Ωm) box
# --------------------------------------------------------------------------- #
def test_hidden_point_is_uniform_in_the_s8_om_box():
    from scipy import stats

    config = bd.BlindingConfig()
    fid = config.theory
    hidden = [bd.hidden_theory(f"seed-{i}", config) for i in range(5000)]
    s8 = np.array([h.S8 for h in hidden])
    om = np.array([h.Omega_m for h in hidden])

    assert np.all(np.abs(s8 - fid.S8) <= 0.075) and np.all(
        np.abs(om - fid.Omega_m) <= 0.1
    )
    assert stats.kstest((s8 - fid.S8 + 0.075) / 0.15, "uniform").pvalue > 1e-3
    assert stats.kstest((om - fid.Omega_m + 0.1) / 0.2, "uniform").pvalue > 1e-3
    for h in hidden[:50]:
        p = h.ccl_params()
        assert p["sigma8"] == pytest.approx(h.S8 / np.sqrt(h.Omega_m / 0.3), rel=1e-12)
        omega_nu = h.m_nu / (93.14 * h.h**2)
        assert p["Omega_c"] == pytest.approx(
            h.Omega_m - h.Omega_b - omega_nu, rel=1e-12
        )
    assert bd.hidden_theory("seed-1", config) == hidden[1]


# --------------------------------------------------------------------------- #
# I4: the shift leaks no B-modes
# --------------------------------------------------------------------------- #
def _shape_noise_variance(left, right):
    """Var ξ± per bin from shape noise at UNIONS n_eff, σ_e and area."""
    n_eff, sigma_e, area = 4.96, 0.378, 2894.0 * 3600.0  # arcmin⁻², —, arcmin²
    pairs = np.pi * area * n_eff**2 * (right**2 - left**2) / 2.0
    return 2.0 * sigma_e**4 / pairs


@pytest.mark.parametrize("ds8", [0.075, -0.075])
@pytest.mark.parametrize("dom", [0.1, -0.1])
def test_shift_leaks_no_b_modes(ds8, dom):
    """B-modes a shift at the envelope's edge induces stay within the audit's bound.

    At each corner of the (S8, Ωm) envelope, on the production grids (the
    0.08–300′ 1000-bin integration grid, the 1–250′ 20-bin reporting grid),
    single-bin n(z), σ from shape noise at UNIONS depth: the 20 COSEBIs B-modes
    on [12, 83]′ and pure-mode ξ_B move by at most ``blinding.B_SIGMA``.
    """
    from sp_validation import b_modes

    config = bd.BlindingConfig(theory=TheoryConfig(transfer_function="eisenstein_hu"))
    fid = config.theory
    hidden = dataclasses.replace(
        fid,
        S8=fid.S8 + np.sign(ds8) * config.envelope["S8"],
        Omega_m=fid.Omega_m + np.sign(dom) * config.envelope["Omega_m"],
    )
    grids = {
        "integration": b_modes.log_bin_edges(0.08, 300.0, 1000),
        "reporting": b_modes.log_bin_edges(1.0, 250.0, 20),
    }
    s = sio.new_sacc({0: NZ2[0]}, metadata={"catalogue_version": "TOY"})
    theta = {}
    for grid, (left, right) in grids.items():
        theta[grid] = np.sqrt(left * right)
        sio.add_xi(s, (0, 0), theta[grid], *_xi_template(theta[grid]), grid=grid)
    shift = np.zeros(len(s.mean))
    for block, factor in bd.factors(s, fid, hidden):
        shift[block.rows] = factor

    def delta(grid):
        return tuple(
            shift[s.indices(t, grid=grid)] for t in (sio.XI_PLUS, sio.XI_MINUS)
        )

    left, right = grids["integration"]
    var = _shape_noise_variance(left, right)
    (result,) = b_modes.cosebis_scan_from_xi(
        theta["integration"],
        *delta("integration"),
        np.diag(np.concatenate([var, var])),
        left,
        right,
        nmodes=20,
        scale_cuts=[(12.0, 83.0)],
    ).values()
    sigma_b = np.sqrt(np.diag(result["cov"])[20:])
    assert np.max(np.abs(result["En"])) > 1e2 * np.max(np.abs(result["Bn"]))
    assert np.max(np.abs(result["Bn"]) / sigma_b) <= bd.B_SIGMA[sio.COSEBI_BB]

    rep_left, rep_right = grids["reporting"]
    modes = b_modes.pure_eb_from_xi(
        theta["reporting"],
        *delta("reporting"),
        theta["integration"],
        *delta("integration"),
        rep_left[0],
        rep_right[-1],
    )
    sigma = np.sqrt(_shape_noise_variance(rep_left, rep_right))
    for key in ("xip_B", "xim_B"):
        finite = np.isfinite(modes[key])
        assert finite.sum() > 10
        bound = bd.B_SIGMA[sio.PURE_TYPES[key]]
        assert np.max(np.abs(modes[key][finite]) / sigma[finite]) <= bound, key


# --------------------------------------------------------------------------- #
# I5: the seal table
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("version", ["TOY_OPEN", "TOY_MOCK"])
def test_unblinded_and_mock_births_are_stamped_untouched(blinds, version):
    s = two_bin_sacc(rho=True)
    sealed = sio.seal(s, blinds[version])
    assert np.array_equal(np.asarray(sealed.mean), np.asarray(s.mean))
    assert cu.read_stamp(sealed.metadata).stamp == blinds[version].stamp
    assert "blinding" not in s.metadata  # the input is not stamped in place


def test_a_blinded_birth_is_concealed_and_stamped(blinds):
    s = two_bin_sacc(cl=False)
    sealed = sio.seal(s, blinds["TOY"])
    assert not np.array_equal(np.asarray(sealed.mean), np.asarray(s.mean))
    assert cu.read_stamp(sealed.metadata).stamp == blinds["TOY"].stamp


@pytest.mark.parametrize(
    "make", [cosebis_sacc, pure_eb_sacc], ids=["cosebis", "pure_eb"]
)
def test_a_blinded_birth_of_a_derived_statistic_is_refused(blinds, make):
    with pytest.raises(ValueError, match="derived_from"):
        sio.seal(make(), blinds["TOY"])


def test_a_blinded_birth_without_signal_never_opens_the_blind(blinds, monkeypatch):
    def refuse(custody):
        raise AssertionError("ρ/τ opened the blind")

    monkeypatch.setattr(bd, "open_blind", refuse)
    s = rho_sacc()
    sealed = sio.seal(s, blinds["TOY"])
    assert np.array_equal(np.asarray(sealed.mean), np.asarray(s.mean))
    assert cu.read_stamp(sealed.metadata).stamp == blinds["TOY"].stamp


def test_a_stamped_sacc_is_not_born_again(blinds, tmp_path):
    sealed = sio.seal(two_bin_sacc(cl=False), blinds["TOY_OPEN"])
    with pytest.raises(ValueError, match="already stamped"):
        sio.seal(sealed, blinds["TOY"])
    path = tmp_path / "part.sacc"
    sio.save(two_bin_sacc(cl=False), path, custody=blinds["TOY_OPEN"])
    with pytest.raises(ValueError, match="already stamped"):
        sio.save(sio.load(path), path, custody=blinds["TOY_OPEN"])


# --------------------------------------------------------------------------- #
# I6: the gates
# --------------------------------------------------------------------------- #
def test_save_needs_a_custody_or_inputs(tmp_path):
    with pytest.raises(ValueError, match="custody"):
        sio.save(two_bin_sacc(), tmp_path / "x.sacc")
    assert not (tmp_path / "x.sacc").exists()


def _raw(tmp_path, name, **metadata):
    s = two_bin_sacc(cl=False)
    s.metadata.update(metadata)
    path = tmp_path / f"{name}.sacc"
    s.save_fits(str(path), overwrite=True)
    return path


def test_load_refuses_unstamped_and_malformed_files(blinds, tmp_path):
    stamp = blinds["TOY"].stamp
    for name, metadata in {
        "unstamped": {},
        "unknown": {**stamp, "blinding": "open"},
        "no_catalogue": {"blinding": "unblinded"},
        "no_commitment": {k: v for k, v in stamp.items() if k != "blinding_commitment"},
        "mixed": {**blinds["TOY_OPEN"].stamp, "blinding_blind": "toy"},
    }.items():
        with pytest.raises(ValueError, match="stamp"):
            sio.load(_raw(tmp_path, name, **metadata))


@pytest.mark.parametrize("version", ["TOY", "TOY_OPEN", "TOY_MOCK"])
def test_every_custody_loads(blinds, tmp_path, version):
    path = tmp_path / "part.sacc"
    written = sio.save(two_bin_sacc(), path, custody=blinds[version])
    loaded = sio.load(path)
    assert cu.read_stamp(loaded.metadata).stamp == blinds[version].stamp
    assert np.array_equal(np.asarray(loaded.mean), np.asarray(written.mean))


# --------------------------------------------------------------------------- #
# I7: derivations inherit one stamp; assembly equals the declaration
# --------------------------------------------------------------------------- #
def test_a_derivation_inherits_its_inputs_stamp(blinds, tmp_path):
    xi = sio.seal(two_bin_sacc(cl=False), blinds["TOY"])
    written = sio.save(cosebis_sacc(), tmp_path / "c.sacc", derived_from=[xi, xi])
    assert cu.read_stamp(written.metadata).stamp == blinds["TOY"].stamp


def test_inputs_under_two_stamps_are_refused(blinds, tmp_path):
    a = sio.seal(two_bin_sacc(cl=False), blinds["TOY_OPEN"])
    b = sio.seal(two_bin_sacc(cl=False), blinds["TOY_MOCK"])
    with pytest.raises(ValueError, match="stamps"):
        sio.save(cosebis_sacc(), tmp_path / "c.sacc", derived_from=[a, b])


def test_plaintext_cannot_be_laundered_under_a_concealed_stamp(blinds, tmp_path):
    """A derivation's ξ± and Cℓ_EE rows must be copies of its inputs' rows."""
    plain = two_bin_sacc()
    concealed = sio.seal(plain, blinds["TOY"])
    with pytest.raises(ValueError, match="not copies"):
        sio.save(plain, tmp_path / "x.sacc", derived_from=[concealed])
    assert not (tmp_path / "x.sacc").exists()
    sio.save(concealed.copy(), tmp_path / "y.sacc", derived_from=[concealed])


@pytest.mark.parametrize(
    "parts_under, declared",
    [
        ("TOY", "TOY_OPEN"),  # a stale concealed part after a reveal
        ("OTHER", "TOY"),  # another blind and catalogue
        ("TOY_MOCK", "TOY_OPEN"),  # a mock inside data
        ("TOY_OPEN", "TOY"),  # unblinded inside blinded
    ],
)
def test_assembly_refuses_parts_under_another_custody(
    blinds, tmp_path, parts_under, declared
):
    part = sio.seal(two_bin_sacc(cl=False), blinds[parts_under])
    with pytest.raises(ValueError, match="declared"):
        sio.save(
            part.copy(),
            tmp_path / "a.sacc",
            derived_from=[part],
            custody=blinds[declared],
        )
    assert not (tmp_path / "a.sacc").exists()


def test_assembly_under_the_declaration(blinds, tmp_path):
    part = sio.seal(two_bin_sacc(cl=False), blinds["TOY"])
    written = sio.save(
        part.copy(), tmp_path / "a.sacc", derived_from=[part], custody=blinds["TOY"]
    )
    assert cu.read_stamp(written.metadata).stamp == blinds["TOY"].stamp


# --------------------------------------------------------------------------- #
# I8: the digest binds the whole config; the commitment is the fork's
# --------------------------------------------------------------------------- #
def _variants(config):
    """Configs differing from ``config`` in exactly one field, at any depth."""
    for key, value in config.envelope.items():
        yield dataclasses.replace(
            config, envelope={**config.envelope, key: value + 0.01}
        )
        rest = {k: v for k, v in config.envelope.items() if k != key}
        yield dataclasses.replace(config, envelope={**rest, key + "_": value})
    yield dataclasses.replace(config, envelope={**config.envelope, "h": 0.01})
    for field in dataclasses.fields(TheoryConfig):
        value = getattr(config.theory, field.name)
        changed = value + "x" if isinstance(value, str) else value + 0.01
        yield dataclasses.replace(
            config, theory=dataclasses.replace(config.theory, **{field.name: changed})
        )


def test_the_digest_changes_with_every_field():
    config = bd.BlindingConfig()
    digests = [v.digest() for v in _variants(config)]
    assert len(digests) == len(set(digests)) and config.digest() not in digests
    assert len(digests) > len(dataclasses.fields(TheoryConfig))


def test_the_digest_ignores_int_versus_float():
    a = bd.BlindingConfig(envelope={"S8": 0.075, "Omega_m": 0.1})
    b = bd.BlindingConfig(
        envelope={"S8": 0.075, "Omega_m": 0.1},
        theory=TheoryConfig(w0=-1, wa=0, ia_bias=0),
    )
    assert a.digest() == b.digest()
    assert bd.BlindingConfig.from_record(json.loads(json.dumps(a.record()))) == a


def test_the_commitment_is_the_forks():
    import smokescreen

    assert cu.COMMITMENT_DOMAIN == smokescreen.COMMITMENT_DOMAIN
    for seed in ("the-secret", "2112"):
        assert cu.seed_commitment(seed) == smokescreen.seed_commitment(seed)


def test_the_commitment_does_not_embed_the_rng_seed():
    """The fork seeds its RNG from the undomained sha256 of the seed.

    A commitment over the bare seed would publish that RNG seed in its first
    16 hex characters, and with the public config anyone could redraw the
    hidden cosmology; the domain prefix breaks that identity.
    """
    import secrets

    from smokescreen.param_shifts import _normalize_seed

    for seed in ("my_secret_seed", "the-secret", secrets.token_hex(16)):
        assert int(cu.seed_commitment(seed)[:16], 16) != _normalize_seed(seed)


# --------------------------------------------------------------------------- #
# I9: the seed never touches disk
# --------------------------------------------------------------------------- #
SEED = "5eed" * 8


def _files_containing(root, needle):
    return [
        p for p in root.rglob("*") if p.is_file() and needle.encode() in p.read_bytes()
    ]


@pytest.mark.parametrize("fault", [None, "after_key", "encrypt"])
def test_init_never_writes_the_seed(tmp_path, monkeypatch, fault):
    """A normal init, or one interrupted anywhere, leaves no file holding the seed."""
    from cryptography import fernet

    _cat_config(tmp_path)
    monkeypatch.setattr(bd.secrets, "token_hex", lambda n: SEED)
    if fault == "after_key":

        def replace(src, dst):
            raise OSError("interrupted after the key was written")

        monkeypatch.setattr(bd.os, "replace", replace)
    elif fault == "encrypt":

        def encrypt(self, data):
            raise RuntimeError("interrupted while encrypting")

        monkeypatch.setattr(fernet.Fernet, "encrypt", encrypt)

    if fault is None:
        _init(tmp_path, "toy", "TOY")
        assert (tmp_path / "blinds" / "toy" / "key").exists()
    else:
        with pytest.raises((OSError, RuntimeError)):
            _init(tmp_path, "toy", "TOY")
    assert _files_containing(tmp_path, SEED) == []


def test_init_refuses_existing_state(fresh):
    with pytest.raises(cu.CustodyError, match="exists"):
        _init(fresh, "toy", "OTHER")
    with pytest.raises(cu.CustodyError, match="already covered"):
        _init(fresh, "again", "TOY")
    with pytest.raises(cu.CustodyError, match="blinded first"):
        _init(fresh, "open", "TOY_OPEN")


def test_the_record_is_read_only(fresh):
    record = fresh / "blinds" / "toy"
    for name in ("commitment.json", "seed.fernet", "key"):
        assert not os.stat(record / name).st_mode & (
            stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        )


# --------------------------------------------------------------------------- #
# I10: an opened blind is the one committed
# --------------------------------------------------------------------------- #
def _edit(path, edit):
    os.chmod(path, 0o644)
    record = json.loads(path.read_text())
    edit(record)
    path.write_text(json.dumps(record))


def test_open_blind_verifies_the_record(fresh):
    blind = bd.open_blind(_custody(fresh, "TOY"))
    assert blind.name == "toy" and blind.seed not in repr(blind)


@pytest.mark.parametrize(
    "tamper",
    ["wrong_key", "config", "config_and_digest", "scheme", "name"],
)
def test_open_blind_refuses_a_tampered_record(fresh, tamper):
    from cryptography.fernet import Fernet

    record = fresh / "blinds" / "toy"
    commitment = record / "commitment.json"
    if tamper == "wrong_key":
        os.chmod(record / "key", 0o644)
        (record / "key").write_bytes(Fernet.generate_key())
    elif tamper == "config":
        _edit(commitment, lambda r: r["config"]["envelope"].update(S8=0.3))
    elif tamper == "config_and_digest":

        def edit(r):
            r["config"]["envelope"]["S8"] = 0.3
            r["config_digest"] = bd.record_digest(r["config"])

        _edit(commitment, edit)
    elif tamper == "scheme":
        _edit(commitment, lambda r: r.update(draw_scheme=r["draw_scheme"] + 1))
    elif tamper == "name":
        os.rename(record, fresh / "blinds" / "toy2")
        _edit(
            fresh / "blinds" / "toy2" / "commitment.json",
            lambda r: r.update(blind="toy2"),
        )
        (fresh / "blinds" / "toy2" / "bases").write_text("TOY\n")
    with pytest.raises(cu.CustodyError):
        bd.open_blind(_custody(fresh, "TOY"))


def _init_storing(root, edit):
    """Draw blind toy for TOY (seed ``toy-seed``), its config stored as ``edit``
    makes it: the record a code with another TheoryConfig schema writes."""
    record = bd.BlindingConfig.record
    with pytest.MonkeyPatch.context() as m:
        m.setattr(bd.BlindingConfig, "record", lambda self: edit(record(self)))
        m.setattr(bd.secrets, "token_hex", lambda n: "toy-seed")
        _init(root, "toy", "TOY")


def test_a_record_lacking_a_field_opens_to_the_shift_it_was_drawn_with(
    tmp_path, blinds, monkeypatch
):
    """A record that predates a TheoryConfig field opens once the field's
    neutral value is declared, and conceals exactly as the blind that names it;
    until then it is refused by the field's name, never as an edited record."""
    _cat_config(tmp_path)

    def without_alphaz(record):
        del record["theory"]["ia_alphaz"]
        return record

    _init_storing(tmp_path, without_alphaz)
    custody = _custody(tmp_path, "TOY")
    with pytest.raises(cu.CustodyError, match="ia_alphaz") as refused:
        bd.open_blind(custody)
    assert "edited" not in str(refused.value)

    monkeypatch.setattr(bd, "NEUTRAL", {"ia_alphaz": 0.0})
    s = two_bin_sacc(cl=False)
    assert np.array_equal(
        np.asarray(sio.seal(s, custody).mean),
        np.asarray(sio.seal(s, blinds["TOY"]).mean),
    )


def test_a_record_naming_a_field_this_code_lacks_is_refused_by_name(tmp_path):
    _cat_config(tmp_path)
    _init_storing(
        tmp_path, lambda r: {**r, "theory": {**r["theory"], "baryon_boost": 0.0}}
    )
    with pytest.raises(cu.CustodyError, match="baryon_boost") as refused:
        bd.open_blind(_custody(tmp_path, "TOY"))
    assert "edited" not in str(refused.value)


def test_the_committed_blind_opens_under_this_code():
    """``tests/data/blinds/committed`` opens to the point it was drawn at.

    A TheoryConfig field added without its value in ``blinding.NEUTRAL`` would
    strand every live blind; this record turns that red.
    """
    blind = bd._open(DATA / "blinds", "committed")
    assert blind.config == bd.BlindingConfig()
    assert (blind.hidden.S8, blind.hidden.Omega_m) == pytest.approx(
        COMMITTED_POINT, rel=1e-12
    )


# --------------------------------------------------------------------------- #
# I11: the audit proves blinded − true = shift(seed)
# --------------------------------------------------------------------------- #
def _concealed(root, s):
    """``s`` sealed under blind toy, before its seed is published."""
    return sio.seal(s, _custody(root, "TOY"))


def _write(path, s, stamp):
    """``s`` on disk under ``stamp`` as given, as the door would never write it."""
    s = s.copy()
    s.metadata.update(stamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    s.save_fits(str(path), overwrite=True)


def _audit(root, archived, live, *, centres=("a", "a"), seed=None):
    """Publish toy's seed (or ``seed``), archive ``archived`` under toy's stamp
    with ``live`` as its re-measured twin, and audit the pair."""
    blinded = _custody(root, "TOY")
    seed = seed or bd.open_blind(blinded).seed
    (root / "blinds" / "toy" / "revealed.json").write_text(json.dumps({"seed": seed}))
    for tree, s, stamp, centre in (
        ("archive", archived, blinded.stamp, centres[0]),
        ("live", live, cu.Custody("unblinded", "TOY").stamp, centres[1]),
    ):
        s = s.copy()
        s.metadata["patch_centers_sha256"] = centre
        _write(root / tree / "sub" / "part.sacc", s, stamp)
    return bd.audit(
        "toy",
        archive=root / "archive",
        true_root=root / "live",
        cat_config=root / "cat_config.yaml",
    )


def _problems(report):
    return report["problems"] + [
        problem for part in report["parts"].values() for problem in part["problems"]
    ]


def test_the_audit_passes_on_a_true_reveal(fresh):
    s = two_bin_sacc(rho=True)
    report = _audit(fresh, _concealed(fresh, s), s)
    assert report["ok"], report
    ((path, part),) = report["parts"].items()
    assert path == "sub/part.sacc" and part["residual"] <= 1e-6
    assert (
        abs(report["shift"]["S8"]) <= 0.075 and abs(report["shift"]["Omega_m"]) <= 0.1
    )


def test_the_audit_fails_on_a_wrong_seed(fresh):
    s = two_bin_sacc()
    report = _audit(fresh, _concealed(fresh, s), s, seed="not-the-seed")
    assert not report["ok"] and "published seed" in _problems(report)[0]


def test_the_audit_fails_on_other_patch_centres(fresh):
    s = two_bin_sacc()
    report = _audit(fresh, _concealed(fresh, s), s, centres=("a", "b"))
    assert not report["ok"] and "patch centres" in _problems(report)[0]


@pytest.mark.parametrize("times", [0, 2], ids=["left_true", "shifted_twice"])
def test_the_audit_fails_on_a_mixed_file(fresh, times):
    """One ξ± pair under the blinded stamp carries 0× or 2× the shift."""
    s = two_bin_sacc(rho=True)
    archived = _concealed(fresh, s)
    for i in _rows(s, sio.XI_PLUS, sio.XI_MINUS):
        if s.data[i].tracers == sio._pair((1, 1)):
            shift = archived.data[i].value - s.data[i].value
            archived.data[i].value = s.data[i].value + times * shift
    report = _audit(fresh, archived, s)
    assert not report["ok"]
    assert any("≠ shift(seed)" in problem for problem in _problems(report))


@pytest.mark.parametrize("kind", [sio.CL_BB, sio.RHO_PLUS.format(k=0)])
def test_the_audit_fails_when_an_unshifted_row_moved(fresh, kind):
    s = two_bin_sacc(rho=True)
    archived = _concealed(fresh, s)
    i = _rows(s, kind)[-1]
    archived.data[i].value += 1e-6 * abs(s.data[i].value)
    report = _audit(fresh, archived, s)
    assert not report["ok"]
    assert _problems(report) == [f"{kind} moved, but the blind leaves it unshifted"]


@pytest.mark.parametrize(
    "e_kind, b_kind",
    [
        (sio.COSEBI_EE, sio.COSEBI_BB),
        (sio.PURE_TYPES["xim_E"], sio.PURE_TYPES["xim_B"]),
    ],
    ids=["cosebis", "pure_eb"],
)
@pytest.mark.parametrize("b_sigma", [0.0, 1.0])
def test_the_audit_bounds_derived_b_modes(fresh, e_kind, b_kind, b_sigma):
    """A derived part's E rows move with the blind; its B rows by ≤ ``B_SIGMA``."""
    true = cosebis_sacc() if e_kind == sio.COSEBI_EE else pure_eb_sacc()
    true.add_covariance(np.full(len(true.mean), 0.01))
    archived = true.copy()
    for i in _rows(true, e_kind):
        archived.data[i].value += 0.3
    for i in _rows(true, b_kind):
        archived.data[i].value += b_sigma * 0.1
    report = _audit(fresh, archived, true)
    assert report["ok"] == (b_sigma == 0.0), report
    (part,) = report["parts"].values()
    assert part["shift_over_sigma"][e_kind] == pytest.approx(3.0)
    if b_sigma:
        assert _problems(report) == [f"{b_kind} moved by 1.00e+00σ under the blind"]
