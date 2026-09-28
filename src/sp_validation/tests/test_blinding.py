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
import string
from pathlib import Path

import numpy as np
import pytest
import yaml
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from sp_validation import blinding as bd
from sp_validation import custody as cu
from sp_validation import sacc_io as sio
from sp_validation.blinding_theory import TheoryConfig

FAST = {"theory": {"transfer_function": "eisenstein_hu"}}
DATA = Path(__file__).parent / "data"
# The hidden (S8, Ωm) of the blind committed under DATA/blinds/committed.
COMMITTED_POINT = (0.8374952413635888, 0.32730573347638175)
VERSIONS = ("TOY", "OTHER", "TOY_OPEN", "OTHER_OPEN", "TOY_MOCK")


# --------------------------------------------------------------------------- #
# Registries with real blinds, and custody under them
# --------------------------------------------------------------------------- #
def _registry(root, *blinds):
    """A catalogue config declaring one catalogue per custody state, and
    ``blinds`` ((name, base) pairs) drawn for it, each from a fixed seed."""
    entries = {
        "TOY": {},
        "OTHER": {},
        "TOY_OPEN": {"blinding": "unblinded"},
        "OTHER_OPEN": {"blinding": "unblinded"},
        "TOY_MOCK": {"blinding": "mock"},
        "paths": {"output": str(root / "output")},
    }
    (root / "cat_config.yaml").write_text(yaml.safe_dump(entries))
    (root / "fast.json").write_text(json.dumps(FAST))
    with pytest.MonkeyPatch.context() as m:
        for blind, base in blinds:
            m.setattr(bd.secrets, "token_hex", lambda n, b=blind: f"{b}-seed")
            _init(root, blind, base)
    return root


def _init(root, blind, *bases):
    cat_config, config = root / "cat_config.yaml", root / "fast.json"
    bd.main(
        ["init", blind, *bases, "--cat-config", str(cat_config)]
        + ["--config", str(config)]
    )


def _custody(root, version):
    return bd.declared_custody(root / "cat_config.yaml", version)


def _declare(root, **entries):
    """Set catalogue entries of ``root``'s catalogue config."""
    path = root / "cat_config.yaml"
    path.write_text(yaml.safe_dump({**yaml.safe_load(path.read_text()), **entries}))


@pytest.fixture(scope="module")
def blinds(tmp_path_factory):
    """The five custodies: TOY under blind `toy`, OTHER under `other`."""
    root = _registry(
        tmp_path_factory.mktemp("registry"), ("toy", "TOY"), ("other", "OTHER")
    )
    return {v: _custody(root, v) for v in VERSIONS}


@pytest.fixture
def fresh(tmp_path):
    """A registry of its own, for tests that edit or reveal a blind."""
    return _registry(tmp_path, ("toy", "TOY"))


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
# Data types no blind has a rule for, galaxy–galaxy lensing and CMB κ × shear,
# with the tag SACC requires of each.
UNRULED = {
    "galaxy_shearDensity_xi_t": "theta",
    "cmbGalaxy_convergenceShear_cl_e": "ell",
}


def _xi_template(theta, k=0):
    return 1e-4 * (1 + 0.1 * k) * (theta / 10.0) ** -0.6, 0.5e-4 * (1 + 0.1 * k) * (
        theta / 10.0
    ) ** -0.9


def _add_xi_rows(s, pair, theta, tag, k=0):
    xip, xim = _xi_template(theta, k)
    if tag is not None:
        sio.add_xi(s, pair, theta, xip, xim, grid=tag)
        return
    for dtype, values in ((sio.XI_PLUS, xip), (sio.XI_MINUS, xim)):
        for th, v in zip(theta, values):
            s.add_data_point(dtype, sio._pair(pair), float(v), theta=float(th))


def _add_cl_rows(s, pair, k=0):
    ell_eff = np.array([30.0, 80.0, 150.0, 280.0, 450.0])
    w_ell = np.arange(2, 501).astype(float)
    w_mat = np.exp(-0.5 * ((w_ell[:, None] - ell_eff[None, :]) / 40.0) ** 2)
    ee = 1e-8 * (1 + 0.1 * k) * (ell_eff / 100.0) ** -1.2
    sio.add_pseudo_cl(
        s,
        pair,
        ell_eff,
        ee,
        0.01 * ee,
        0.02 * ee,
        window_ells=w_ell,
        window_weights=w_mat / w_mat.sum(axis=0),
    )


def _add_rho_rows(s):
    theta = np.geomspace(5.0, 250.0, 6)
    sio.add_rho(s, 0, theta, np.arange(6) * 1e-7, np.arange(6) * 2e-7)
    sio.add_tau(s, (0,), 0, theta, np.arange(6) * 3e-7, np.arange(6) * 4e-7)


def two_bin_sacc(
    *,
    pairs=PAIRS,
    xi_tags=XI_TAGS,
    cl=True,
    rho=False,
    derived=(),
    unruled=None,
    covariance=True,
):
    """A catalogue's part: ξ± under each of ``xi_tags`` and pseudo-Cℓ (EE, BB,
    EB) over ``pairs`` of two bins, and optionally ρ/τ, COSEBIs and pure-E/B
    rows and rows of a data type without a blinding rule."""
    s = sio.new_sacc(NZ2, metadata={"catalogue_version": "TOY"})
    for pair in pairs:
        k = PAIRS.index(pair)
        for n, tag in enumerate(xi_tags):
            _add_xi_rows(s, pair, np.geomspace(2.0, 200.0, 6) * (1 + 0.013 * n), tag, k)
        if cl:
            _add_cl_rows(s, pair, k)
    if rho:
        _add_rho_rows(s)
    if "cosebis" in derived:
        sio.add_cosebis(s, (0, 0), np.arange(1, 6) * 1e-10, (12.0, 83.0), Bn=np.ones(5))
    if "pure_eb" in derived:
        theta = np.geomspace(2.0, 200.0, 4)
        sio.add_pure_eb(s, (0, 0), theta, **{k: np.ones(4) for k in sio.PURE_KEYS})
    if unruled is not None:
        for x in (5.0, 20.0, 80.0):
            s.add_data_point(
                unruled, ("source_0", "source_0"), 1e-5, **{UNRULED[unruled]: x}
            )
    if covariance:
        rng = np.random.default_rng(3)
        s.add_covariance(
            np.abs(np.asarray(s.mean)) ** 2 * rng.uniform(1, 2, len(s.mean))
        )
    return s


def derived_sacc(kind):
    """A COSEBIs or pure-E/B part alone, as its rule writes it."""
    return two_bin_sacc(pairs=((0, 0),), xi_tags=(), cl=False, derived=(kind,))


def rho_sacc():
    return two_bin_sacc(pairs=(), rho=True)


def _rows(s, *types):
    return np.array([i for i, dp in enumerate(s.data) if dp.data_type in types], int)


def _values(s):
    return np.asarray(s.mean)


# --------------------------------------------------------------------------- #
# An independent CCL reference for the shift
# --------------------------------------------------------------------------- #
def _reference_cosmologies(seed):
    """The hidden and fiducial CCL cosmologies, the draw made by the fork directly."""
    import pyccl as ccl
    from smokescreen.param_shifts import draw_param_shifts

    shift = draw_param_shifts({"S8": 0.075, "Omega_m": 0.1}, seed)

    def cosmology(s8, om):
        h, ob, mnu = 0.70, 0.0469, 0.06
        return ccl.Cosmology(
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
            transfer_function="eisenstein_hu",
            matter_power_spectrum="halofit",
        )

    return cosmology(0.80 + shift["S8"], 0.30 + shift["Omega_m"]), cosmology(0.80, 0.30)


def reference_shift(s, seed):
    """t(hidden) − t(fiducial) on every ξ± and Cℓ_EE row of ``s``, from scratch."""
    import pyccl as ccl

    cosmologies = _reference_cosmologies(seed)
    ell = np.unique(np.concatenate([np.arange(2, 50), np.geomspace(50, 6e4, 200)]))
    out = np.full(len(s.mean), np.nan)
    for tracers in {dp.tracers for dp in s.data if dp.data_type in sio.SHIFTABLE}:
        a, b = (s.tracers[t] for t in tracers)

        def cl(c, ells):
            lens = [ccl.WeakLensingTracer(c, dndz=(t.z, t.nz)) for t in (a, b)]
            return ccl.angular_cl(c, *lens, ells)

        for dtype, kind in ((sio.XI_PLUS, "GG+"), (sio.XI_MINUS, "GG-")):
            rows = [
                i
                for i, dp in enumerate(s.data)
                if (dp.data_type, dp.tracers) == (dtype, tracers)
            ]
            theta = np.array([s.data[i].tags["theta"] for i in rows]) / 60.0
            hidden, fiducial = (
                ccl.correlation(c, ell=ell, C_ell=cl(c, ell), theta=theta, type=kind)
                for c in cosmologies
            )
            out[rows] = hidden - fiducial
        for i, dp in enumerate(s.data):
            if (dp.data_type, dp.tracers) == (sio.CL_EE, tracers):
                window = dp.tags["window"]
                ells = np.asarray(window.values, float)
                w = np.asarray(window.weight, float)[:, dp.tags["window_ind"]]
                out[i] = w @ (cl(cosmologies[0], ells) - cl(cosmologies[1], ells))
    return out


# --------------------------------------------------------------------------- #
# I1, I2, I5: seal is the seal table; a blinded seal shifts exactly the
# shiftable rows, by the theory difference, and nothing else
# --------------------------------------------------------------------------- #
PARTS = st.fixed_dictionaries(
    {
        "pairs": st.lists(st.sampled_from(PAIRS), min_size=1, max_size=3, unique=True),
        "xi_tags": st.lists(st.sampled_from(XI_TAGS), max_size=3, unique=True),
        "cl": st.booleans(),
        "rho": st.booleans(),
        "derived": st.sampled_from([(), ("cosebis",), ("pure_eb",)]),
        "unruled": st.sampled_from([None, *UNRULED]),
    }
).filter(lambda p: p["xi_tags"] or p["cl"] or p["rho"] or p["derived"] or p["unruled"])
EVERY_SHIFTABLE_ROW = dict(
    pairs=PAIRS, xi_tags=XI_TAGS, cl=True, rho=True, derived=(), unruled=None
)


@settings(max_examples=6, deadline=None)
@given(content=PARTS, version=st.sampled_from(["TOY", "TOY_OPEN", "TOY_MOCK"]))
@example(content=EVERY_SHIFTABLE_ROW, version="TOY")
@example(content={**EVERY_SHIFTABLE_ROW, "xi_tags": [], "cl": False}, version="TOY")
def test_seal_is_the_seal_table(blinds, content, version):
    """Under a blind, a birth with derived or unruled rows is refused; any
    other has ``seal(s) − s`` equal to an independent CCL reference at (fid,
    hidden) on every ξ± and Cℓ_EE row, within 1e-8 of each block's largest
    shift, and every other value, tag, tracer and the covariance bitwise kept.
    Unblinded and mock births keep every value. The blind opens only to shift
    rows, and the input is never stamped in place.

    Blocks come from the data types and tracers: ξ± under every tag and none,
    over the two bins' three pairs, whatever subset a part holds.
    """
    s, custody = two_bin_sacc(**content), blinds[version]
    blinded = custody.status == "blinded"
    opened, open_blind = [], bd.open_blind
    with pytest.MonkeyPatch.context() as m:
        m.setattr(bd, "open_blind", lambda c: opened.append(c) or open_blind(c))
        if blinded and (content["derived"] or content["unruled"]):
            with pytest.raises(ValueError, match="derived_from|no blinding rule"):
                sio.seal(s, custody)
            return
        sealed = sio.seal(s, custody)

    assert "blinding" not in s.metadata
    assert cu.read_stamp(sealed.metadata).stamp == custody.stamp
    moved = _rows(s, *sio.SHIFTABLE) if blinded else np.array([], int)
    assert bool(opened) == bool(len(moved))
    kept = np.setdiff1d(np.arange(len(s.mean)), moved)
    assert np.array_equal(_values(sealed)[kept], _values(s)[kept])
    assert np.array_equal(sealed.covariance.dense, s.covariance.dense)
    for a, b in zip(s.data, sealed.data):
        assert (a.data_type, a.tracers) == (b.data_type, b.tracers)
        assert {k: v for k, v in a.tags.items() if k != "window"} == {
            k: v for k, v in b.tags.items() if k != "window"
        }
    for name, tracer in s.tracers.items():
        if hasattr(tracer, "nz"):
            assert np.array_equal(sealed.tracers[name].nz, tracer.nz)
    if not len(moved):
        return

    shift = _values(sealed) - _values(s)
    expected = reference_shift(s, open_blind(custody).seed)
    for pair in content["pairs"]:
        for types in ((sio.XI_PLUS, sio.XI_MINUS), (sio.CL_EE,)):
            block = [
                i
                for i in moved
                if s.data[i].tracers == sio._pair(pair) and s.data[i].data_type in types
            ]
            if block:
                scale = np.max(np.abs(expected[block]))
                gap = np.max(np.abs(shift[block] - expected[block]))
                assert scale > 0 and gap <= 1e-8 * scale, (pair, types, gap / scale)


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

    assert np.all(np.abs(s8 - fid.S8) <= 0.075)
    assert np.all(np.abs(om - fid.Omega_m) <= 0.1)
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
# I6: the gates
# --------------------------------------------------------------------------- #
def test_nothing_is_written_unstamped_or_stamped_twice(blinds, tmp_path):
    with pytest.raises(ValueError, match="custody"):
        sio.save(two_bin_sacc(), tmp_path / "x.sacc")
    assert not (tmp_path / "x.sacc").exists()
    sealed = sio.seal(two_bin_sacc(cl=False), blinds["TOY_OPEN"])
    with pytest.raises(ValueError, match="already stamped"):
        sio.seal(sealed, blinds["TOY"])
    path = tmp_path / "part.sacc"
    sio.save(two_bin_sacc(cl=False), path, custody=blinds["TOY_OPEN"])
    with pytest.raises(ValueError, match="already stamped"):
        sio.save(sio.load(path), path, custody=blinds["TOY_OPEN"])


@settings(max_examples=20, deadline=None)
@given(
    version=st.sampled_from(["TOY", "TOY_OPEN", "TOY_MOCK"]),
    spoil=st.sampled_from([None, "drop", "add", "status"]),
    data=st.data(),
)
def test_load_opens_exactly_the_stamps_the_door_mints(
    blinds, tmp_path_factory, version, spoil, data
):
    """A file stamped as ``save`` stamps it loads with that stamp; one missing
    a stamp key, carrying a blind's keys under another status, or naming an
    unknown status is refused."""
    stamp = dict(blinds[version].stamp)
    assume(not (spoil == "add" and version == "TOY"))
    if spoil == "drop":
        del stamp[data.draw(st.sampled_from(sorted(stamp)))]
    elif spoil == "add":
        key = data.draw(st.sampled_from(cu.STAMP_KEYS[2:]))
        stamp[key] = blinds["TOY"].stamp[key]
    elif spoil == "status":
        status = st.text(string.ascii_letters, min_size=1, max_size=8)
        stamp["blinding"] = data.draw(status.filter(lambda t: t not in cu.STATUSES))
    path = tmp_path_factory.mktemp("raw") / "part.sacc"
    _write(path, two_bin_sacc(cl=False), stamp)
    if spoil is None:
        assert cu.read_stamp(sio.load(path).metadata).stamp == stamp
    else:
        with pytest.raises(ValueError, match="stamp"):
            sio.load(path)


# --------------------------------------------------------------------------- #
# I7: a derivation carries its inputs' one stamp; an assembly its declaration
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def parts(blinds):
    """A ξ± part born under each custody."""
    return {
        v: sio.seal(two_bin_sacc(xi_tags=("reporting",), cl=False), blinds[v])
        for v in VERSIONS
    }


DERIVATIONS = ("copy", "cosebis", "pure_eb", "plaintext", "unruled")


@settings(max_examples=25, deadline=None)
@given(
    inputs=st.lists(st.sampled_from(VERSIONS), min_size=1, max_size=3),
    declared=st.sampled_from([None, *VERSIONS]),
    content=st.sampled_from(DERIVATIONS),
)
@example(inputs=["TOY"], declared=None, content="plaintext")  # laundering
@example(inputs=["TOY"], declared="TOY_OPEN", content="copy")  # stale after a reveal
@example(inputs=["OTHER"], declared="TOY", content="copy")  # another blind
@example(inputs=["TOY_MOCK"], declared="TOY_OPEN", content="copy")  # mock into data
@example(inputs=["TOY_OPEN"], declared="TOY", content="copy")  # unblinded into blinded
@example(inputs=["TOY_OPEN", "TOY_MOCK"], declared=None, content="cosebis")
@example(inputs=["TOY", "OTHER"], declared=None, content="cosebis")
@example(inputs=["TOY_OPEN", "OTHER_OPEN"], declared=None, content="cosebis")
@example(inputs=["TOY"], declared=None, content="unruled")
@example(inputs=["TOY", "TOY"], declared="TOY", content="copy")  # an assembly
@example(inputs=["TOY_OPEN"], declared="TOY_OPEN", content="plaintext")
@example(inputs=["TOY_MOCK"], declared=None, content="pure_eb")
def test_a_derivation_carries_its_inputs_one_stamp(
    blinds, parts, tmp_path_factory, inputs, declared, content
):
    """``save(s, derived_from=inputs, custody=declared)`` writes ``s``'s values
    under the inputs' stamp exactly when the inputs share one stamp, it is the
    declared custody's (if any), and, under a blind, every ξ± row of ``s`` is a
    copy of an input row and every data type has a blinding rule. Otherwise
    nothing is written."""
    first = parts[inputs[0]]
    s = {
        "copy": lambda: first.copy(),
        "cosebis": lambda: derived_sacc("cosebis"),
        "pure_eb": lambda: derived_sacc("pure_eb"),
        "plaintext": lambda: two_bin_sacc(xi_tags=("reporting",), cl=False),
        "unruled": lambda: two_bin_sacc(
            xi_tags=(), cl=False, unruled=next(iter(UNRULED))
        ),
    }[content]()
    stamp = blinds[inputs[0]].stamp
    blinded = stamp["blinding"] == "blinded"
    allowed = (
        all(blinds[v].stamp == stamp for v in inputs)
        and (declared is None or blinds[declared].stamp == stamp)
        and not (blinded and content in ("plaintext", "unruled"))
    )
    path = tmp_path_factory.mktemp("derived") / "d.sacc"
    custody = blinds[declared] if declared else None
    derive = [parts[v] for v in inputs]
    if not allowed:
        with pytest.raises(ValueError):
            sio.save(s, path, derived_from=derive, custody=custody)
        assert not path.exists()
        return
    sio.save(s, path, derived_from=derive, custody=custody)
    loaded = sio.load(path)
    assert cu.read_stamp(loaded.metadata).stamp == stamp
    assert np.array_equal(_values(loaded), _values(s))


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


def test_the_digest_changes_with_every_field_and_no_literal():
    config = bd.BlindingConfig()
    digests = [v.digest() for v in _variants(config)]
    assert len(digests) == len(set(digests)) and config.digest() not in digests
    assert len(digests) > len(dataclasses.fields(TheoryConfig))

    ints = bd.BlindingConfig(theory=TheoryConfig(w0=-1, wa=0, ia_bias=0))
    assert ints.digest() == config.digest()
    assert bd._recorded_config("toy", json.loads(json.dumps(config.record()))) == config


@settings(max_examples=50)
@given(seed=st.text(min_size=1))
@example(seed="my_secret_seed")
def test_the_commitment_is_the_forks_and_hides_its_rng_seed(seed):
    """The fork seeds its RNG from the undomained sha256 of the seed; the
    domain prefix keeps the commitment from publishing that RNG seed in its
    first 16 hex characters, from which anyone could redraw the hidden point."""
    import smokescreen
    from smokescreen.param_shifts import _normalize_seed

    assert cu.COMMITMENT_DOMAIN == smokescreen.COMMITMENT_DOMAIN
    assert cu.seed_commitment(seed) == smokescreen.seed_commitment(seed)
    assert int(cu.seed_commitment(seed)[:16], 16) != _normalize_seed(seed)


# --------------------------------------------------------------------------- #
# I9 and blind-drawn-once: init draws once, and the seed never touches disk
# --------------------------------------------------------------------------- #
SEED = "5eed" * 8


@pytest.mark.parametrize("fault", [None, "after_key", "encrypt"])
def test_init_never_writes_the_seed(tmp_path, monkeypatch, fault):
    """A normal init, or one interrupted anywhere, leaves no file holding the
    seed; a completed record is read-only."""
    from cryptography import fernet

    _registry(tmp_path)
    monkeypatch.setattr(bd.secrets, "token_hex", lambda n: SEED)
    if fault == "after_key":
        monkeypatch.setattr(bd.os, "replace", lambda src, dst: 1 / 0)
    elif fault == "encrypt":
        monkeypatch.setattr(fernet.Fernet, "encrypt", lambda self, data: 1 / 0)

    if fault is None:
        _init(tmp_path, "toy", "TOY")
        for name in ("commitment.json", "seed.fernet", "key"):
            mode = os.stat(tmp_path / "blinds" / "toy" / name).st_mode
            assert not mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    else:
        with pytest.raises(ZeroDivisionError):
            _init(tmp_path, "toy", "TOY")
    hits = [
        p
        for p in tmp_path.rglob("*")
        if p.is_file() and SEED.encode() in p.read_bytes()
    ]
    assert hits == []


def test_init_refuses_existing_state(fresh):
    with pytest.raises(cu.CustodyError, match="exists"):
        _init(fresh, "toy", "OTHER")
    with pytest.raises(cu.CustodyError, match="already covered"):
        _init(fresh, "again", "TOY")
    with pytest.raises(cu.CustodyError, match="blinded first"):
        _init(fresh, "open", "TOY_OPEN")
    (fresh / "blinds" / ".later.tmp").mkdir()
    with pytest.raises(cu.CustodyError, match=r"\.later\.tmp exists"):
        _init(fresh, "later", "OTHER")


@pytest.mark.parametrize(
    "config, named",
    [
        ({"envelope": {"S8": 0.075, "Omega_M": 0.1}}, "Omega_M"),
        ({"theory": {"transfer_function": "eisenstein_hu", "Omega_M": 0.3}}, "Omega_M"),
        ({"theory": {"transfer_function": "eisenstein-hu"}}, "eisenstein-hu"),
    ],
    ids=["envelope_key", "theory_key", "unbuildable"],
)
def test_init_refuses_a_config_no_blind_conceals_under(tmp_path, config, named):
    """Refused before any record exists, so the base stays free for a blind
    drawn under a good config."""
    _registry(tmp_path)
    (tmp_path / "bad.json").write_text(json.dumps(config))
    args = ["init", "toy", "TOY", "--cat-config", str(tmp_path / "cat_config.yaml")]
    with pytest.raises(cu.CustodyError, match=named):
        bd.main([*args, "--config", str(tmp_path / "bad.json")])
    assert not any((tmp_path / "blinds").glob("*"))
    _init(tmp_path, "toy", "TOY")


def test_share_adds_a_base_to_a_concealed_blind(fresh):
    _declare(fresh, TOY_V={"base": "TOY"}, LATER={})
    cat_config = str(fresh / "cat_config.yaml")
    for base, refusal in {
        "TOY": "already covered",
        "TOY_V": "variant",
        "TOY_leak_corr": "variant",
        "TOY_OPEN": "blinded first",
        "TOY_MOCK": "blinded first",
    }.items():
        with pytest.raises(cu.CustodyError, match=refusal):
            bd.share("toy", base, cat_config=cat_config)

    assert bd.main(["share", "toy", "OTHER", "--cat-config", cat_config]) == 0
    toy, other = _custody(fresh, "TOY"), _custody(fresh, "OTHER")
    assert (other.blind, other.commitment) == ("toy", toy.commitment)

    _publish(fresh, bd.open_blind(toy).seed)
    with pytest.raises(cu.CustodyError, match="revealed"):
        bd.share("toy", "LATER", cat_config=cat_config)


def test_a_blind_is_drawn_for_every_catalogue_reading_its_file(tmp_path):
    """init and share refuse a catalogue whose shear file another catalogue
    reads under another custody, before anything is written; drawn for both,
    the blind covers both."""
    _registry(tmp_path)
    toy, later = ({"shear": {"path": str(tmp_path / f)}} for f in ("a.fits", "b.fits"))
    _declare(
        tmp_path,
        TOY=toy,
        TOY_TWIN={**toy, "blinding": "unblinded"},
        LATER=later,
        LATER_TWIN={**later, "blinding": "unblinded"},
    )
    with pytest.raises(cu.CustodyError, match="TOY_TWIN"):
        _init(tmp_path, "toy", "TOY")
    assert not (tmp_path / "blinds").exists()

    _declare(tmp_path, TOY_TWIN=toy)
    _init(tmp_path, "toy", "TOY", "TOY_TWIN")
    with pytest.raises(cu.CustodyError, match="LATER_TWIN"):
        bd.share("toy", "LATER", cat_config=str(tmp_path / "cat_config.yaml"))
    bases = (tmp_path / "blinds" / "toy" / "bases").read_text().split()
    assert bases == ["TOY", "TOY_TWIN"]


# --------------------------------------------------------------------------- #
# I10: an opened blind is the one committed
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tamperable(tmp_path_factory):
    """Blind `toy`'s record, restored after each edit, and TOY's custody."""
    root = _registry(tmp_path_factory.mktemp("tamperable"), ("toy", "TOY"))
    return root / "blinds" / "toy" / "commitment.json", _custody(root, "TOY")


def _leaves(record, path=()):
    for key, value in record.items():
        if isinstance(value, dict):
            yield from _leaves(value, (*path, key))
        else:
            yield (*path, key)


# The fields of a blind's record the sealed seed binds: its name, commitment,
# draw scheme, config digest and every field of its config.
BOUND = sorted(
    [("blind",), ("seed_commitment",), ("config_digest",), ("draw_scheme",)]
    + [("config", *leaf) for leaf in _leaves(bd.BlindingConfig().record())]
)


def _edited(value):
    if isinstance(value, bool):
        return not value
    return value + ("x" if isinstance(value, str) else 1)


@settings(max_examples=15, deadline=None)
@given(leaf=st.sampled_from(BOUND), redigest=st.booleans())
@example(leaf=("config", "envelope", "S8"), redigest=True)
@example(leaf=("config", "theory", "transfer_function"), redigest=False)
def test_open_blind_refuses_any_edit_of_the_record(tamperable, leaf, redigest):
    """Edit any field the sealed seed binds, a config field with its digest
    recomputed or not: opening the blind is refused."""
    path, custody = tamperable
    original = path.read_bytes()
    record = json.loads(original)
    *parents, key = leaf
    node = record
    for parent in parents:
        node = node[parent]
    node[key] = _edited(node[key])
    if redigest and leaf != ("config_digest",):
        record["config_digest"] = bd.record_digest(record["config"])
    os.chmod(path, 0o644)
    try:
        path.write_text(json.dumps(record))
        bd._open.cache_clear()
        with pytest.raises(cu.CustodyError):
            bd.open_blind(custody)
    finally:
        path.write_bytes(original)
        bd._open.cache_clear()
    assert bd.open_blind(custody).seed not in repr(bd.open_blind(custody))


@pytest.mark.parametrize("tamper", ["wrong_key", "renamed", "custody"])
def test_open_blind_refuses_another_key_name_or_commitment(fresh, tamper):
    from cryptography.fernet import Fernet

    record = fresh / "blinds" / "toy"
    if tamper == "wrong_key":
        os.chmod(record / "key", 0o644)
        (record / "key").write_bytes(Fernet.generate_key())
    elif tamper == "renamed":
        os.rename(record, record.with_name("toy2"))
        commitment = record.with_name("toy2") / "commitment.json"
        os.chmod(commitment, 0o644)
        commitment.write_text(
            json.dumps({**json.loads(commitment.read_text()), "blind": "toy2"})
        )
    custody = _custody(fresh, "TOY")
    if tamper == "custody":
        custody = dataclasses.replace(custody, commitment="0" * 64)
    with pytest.raises(cu.CustodyError):
        bd.open_blind(custody)


def _init_storing(root, edit):
    """Draw blind toy for TOY (seed ``toy-seed``), its config stored as ``edit``
    makes it: the record a code with another TheoryConfig schema writes."""
    record = bd.BlindingConfig.record
    config = bd.BlindingConfig(theory=TheoryConfig(**FAST["theory"]))
    with pytest.MonkeyPatch.context() as m:
        m.setattr(bd.BlindingConfig, "record", lambda self: edit(record(self)))
        m.setattr(bd.secrets, "token_hex", lambda n: "toy-seed")
        # That code's init opened the record under its own schema.
        m.setattr(bd, "_conceal_one_row", lambda name, seed, record: None)
        bd.init("toy", ["TOY"], cat_config=root / "cat_config.yaml", config=config)


@pytest.mark.parametrize(
    "edit, field",
    [
        (
            lambda r: {
                **r,
                "theory": {k: v for k, v in r["theory"].items() if k != "ia_alphaz"},
            },
            "ia_alphaz",
        ),
        (
            lambda r: {**r, "theory": {**r["theory"], "baryon_boost": 0.0}},
            "baryon_boost",
        ),
    ],
    ids=["lacks_a_field", "names_a_foreign_field"],
)
def test_a_record_of_another_schema_is_refused_by_the_field(
    tmp_path, blinds, monkeypatch, edit, field
):
    """A record lacking a TheoryConfig field, or naming one this code lacks,
    is refused by the field's name, never as an edited record. Once a lacking
    field's neutral value is declared, it conceals as the blind that names it."""
    _registry(tmp_path)
    _init_storing(tmp_path, edit)
    custody = _custody(tmp_path, "TOY")
    with pytest.raises(cu.CustodyError, match=field) as refused:
        bd.open_blind(custody)
    assert "edited" not in str(refused.value)
    if field == "ia_alphaz":
        monkeypatch.setattr(bd, "NEUTRAL", {"ia_alphaz": 0.0})
        s = two_bin_sacc(cl=False)
        assert np.array_equal(
            _values(sio.seal(s, custody)), _values(sio.seal(s, blinds["TOY"]))
        )


def test_the_committed_blind_opens_under_this_code():
    """``tests/data/blinds/committed`` opens to the point it was drawn at.

    A TheoryConfig field added without its value in ``blinding.NEUTRAL`` would
    strand every live blind; this record turns that red.
    """
    blind = bd._open(DATA / "blinds", "committed")
    assert (blind.hidden.S8, blind.hidden.Omega_m) == pytest.approx(
        COMMITTED_POINT, rel=1e-12
    )


def test_a_blind_drawn_under_another_draw_scheme_is_refused(tmp_path, monkeypatch):
    """Its seed and record agree, but this install's fork draws differently:
    opening it is refused, and a file stamped under it fails ``verify``."""
    _registry(tmp_path)
    installed = bd.draw_scheme()
    with monkeypatch.context() as m:
        m.setattr(bd, "draw_scheme", lambda: installed + 1)
        _init(tmp_path, "toy", "TOY")
    custody = _custody(tmp_path, "TOY")
    with pytest.raises(cu.CustodyError, match="draw scheme"):
        bd.open_blind(custody)
    part = tmp_path / "rho_tau.sacc"
    sio.save(rho_sacc(), part, custody=custody)  # no signal: the blind stays shut
    assert bd.verify(part, cat_config=tmp_path / "cat_config.yaml") == [
        "this install draws under another scheme"
    ]


# --------------------------------------------------------------------------- #
# verify: a file's stamp against its catalogue's custody, seedless
# --------------------------------------------------------------------------- #
def test_verify_names_what_disagrees_with_the_declaration(fresh, capsys):
    def verify(path):
        code = bd.main(
            ["verify", str(path), "--cat-config", str(fresh / "cat_config.yaml")]
        )
        return code, capsys.readouterr().out

    part, forged = fresh / "part.sacc", fresh / "forged.sacc"
    custody = _custody(fresh, "TOY")
    sio.save(two_bin_sacc(cl=False), part, custody=custody)
    assert verify(part)[0] == 0

    _write(
        forged,
        two_bin_sacc(cl=False),
        {**custody.stamp, "blinding_commitment": "0" * 64},
    )
    code, out = verify(forged)
    assert code == 1 and "['blinding_commitment']" in out

    _publish(fresh, bd.open_blind(custody).seed)
    _declare(fresh, TOY={"blinding": "unblinded"})
    code, out = verify(part)
    assert code == 1 and "'blinding'" in out and "unblinded:TOY" in out


# --------------------------------------------------------------------------- #
# I11: reveal publishes and archives; the audit proves blinded − true = shift(seed)
# --------------------------------------------------------------------------- #
def _write(path, s, stamp):
    """``s`` on disk under ``stamp`` as given, as the door would never write it."""
    s = s.copy()
    s.metadata.update(stamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    s.save_fits(str(path), overwrite=True)


def _publish(root, seed):
    (root / "blinds" / "toy" / "revealed.json").write_text(json.dumps({"seed": seed}))


def test_reveal_refuses_a_root_holding_nothing_concealed(fresh):
    """A mistyped or unbound ``--root`` is refused before the seed is published;
    a reveal interrupted after its first move runs again."""
    root = fresh / "output"
    part = root / "sub" / "part.sacc"
    part.parent.mkdir(parents=True)
    sio.save(two_bin_sacc(cl=False), part, custody=_custody(fresh, "TOY"))
    revealed = fresh / "blinds" / "toy" / "revealed.json"
    cat_config = fresh / "cat_config.yaml"
    (fresh / "empty").mkdir()
    for wrong in (fresh / "outptu", fresh / "empty"):
        with pytest.raises(cu.CustodyError, match="nothing concealed"):
            bd.reveal("toy", root=wrong, cat_config=cat_config)
        assert not revealed.exists()

    archive = bd.reveal("toy", root=root, cat_config=cat_config)
    assert revealed.exists() and (archive / "sub" / "part.sacc").exists()
    assert bd.reveal("toy", root=root, cat_config=cat_config) == archive
    with pytest.raises(cu.CustodyError, match="nothing concealed"):
        bd.reveal("toy", root=fresh / "empty", cat_config=cat_config)


def test_reveal_refuses_a_record_publishing_another_seed(fresh):
    part = fresh / "output" / "part.sacc"
    part.parent.mkdir()
    sio.save(two_bin_sacc(cl=False), part, custody=_custody(fresh, "TOY"))
    _publish(fresh, "0" * 32)
    with pytest.raises(cu.CustodyError, match="another seed"):
        bd.reveal("toy", root=part.parent, cat_config=fresh / "cat_config.yaml")
    assert part.exists()


@pytest.fixture(scope="module")
def revealed(tmp_path_factory):
    """A registry whose blind `toy` has published its seed, TOY's custody
    before the reveal, ``AUDITED`` sealed under it, and ``audit(archived,
    live)``, which archives ``archived`` beside ``live`` as its re-measured
    twin and audits the pair."""
    root = _registry(tmp_path_factory.mktemp("revealed"), ("toy", "TOY"))
    blinded = _custody(root, "TOY")
    _publish(root, bd.open_blind(blinded).seed)
    true = cu.Custody("unblinded", "TOY")

    def audit(
        archived, live, *, stamps=(blinded.stamp, true.stamp), centres=("a", "a")
    ):
        for tree, s, stamp, centre in zip(
            ("archive", "live"), (archived, live), stamps, centres
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

    fields = ["root", "blinded", "sealed", "audit"]
    return dataclasses.make_dataclass("Revealed", fields)(
        root, blinded, sio.seal(AUDITED, blinded), audit
    )


def _problems(report):
    return report["problems"] + [
        p for part in report["parts"].values() for p in part["problems"]
    ]


AUDITED = two_bin_sacc(xi_tags=("reporting",), rho=True)
BLOCKS = [
    (types, sio._pair(pair))
    for pair in PAIRS
    for types in ((sio.XI_PLUS, sio.XI_MINUS), (sio.CL_EE,))
]


@settings(max_examples=4, deadline=None)
@given(
    times=st.lists(
        st.sampled_from([1.0, 0.0, 2.0, 1 + 1e-4, -1.0]), min_size=6, max_size=6
    )
)
@example(times=[1.0] * 6)
@example(times=[1.0] * 5 + [0.0])  # one block left true
@example(times=[1.0] * 3 + [2.0] + [1.0] * 2)  # one block shifted twice
@example(times=[1 + 1e-4] + [1.0] * 5)
def test_the_audit_accepts_exactly_the_shift(revealed, times):
    """An archived part whose ξ± and Cℓ_EE blocks carry ``times`` × the blind's
    shift passes the audit exactly when every block carries it once."""
    true = AUDITED
    shift = _values(revealed.sealed) - _values(true)
    archived = true.copy()
    for k, (types, tracers) in zip(times, BLOCKS):
        for i, dp in enumerate(true.data):
            if dp.data_type in types and dp.tracers == tracers:
                archived.data[i].value = dp.value + k * shift[i]
    report = revealed.audit(archived, true)
    assert report["ok"] == all(k == 1.0 for k in times), report
    if report["ok"]:
        (part,) = report["parts"].values()
        assert part["residual"] <= 1e-6
        assert (
            abs(report["shift"]["S8"]) <= 0.075
            and abs(report["shift"]["Omega_m"]) <= 0.1
        )
    else:
        assert any("≠ shift(seed)" in problem for problem in _problems(report))


SPOILT = {
    "covariance": "covariances differ",
    "rows": "rows, tags or tracers differ",
    "live_mock": "the live file is stamped mock:TOY",
    "live_other_catalogue": "the live file is stamped unblinded:OTHER",
    "archive_other_blind": "not concealed under this blind",
    "patch_centres": "patch centres differ: ['a', 'b']",
    "wrong_seed": "the published seed is not the committed one",
}


@pytest.mark.parametrize("spoilt", SPOILT)
def test_the_audit_names_a_pair_that_is_not_concealed_and_true(revealed, spoilt):
    """An archived part is audited only under the published committed seed,
    against the same catalogue re-measured unblinded with the same rows,
    covariance and patch centres."""
    true = AUDITED
    live = (
        two_bin_sacc(xi_tags=("reporting",), cl=False)
        if spoilt == "rows"
        else true.copy()
    )
    if spoilt == "covariance":
        live.add_covariance(1.01 * np.asarray(true.covariance.dense), overwrite=True)
    archived_stamp = revealed.blinded.stamp
    if spoilt == "archive_other_blind":
        archived_stamp = {**archived_stamp, "blinding_commitment": "0" * 64}
    live_custody = {
        "live_mock": cu.Custody("mock", "TOY"),
        "live_other_catalogue": cu.Custody("unblinded", "OTHER"),
    }.get(spoilt, cu.Custody("unblinded", "TOY"))
    published = revealed.root / "blinds" / "toy" / "revealed.json"
    seed = published.read_text()
    try:
        if spoilt == "wrong_seed":
            _publish(revealed.root, "not-the-seed")
        report = revealed.audit(
            revealed.sealed,
            live,
            stamps=(archived_stamp, live_custody.stamp),
            centres=("a", "b" if spoilt == "patch_centres" else "a"),
        )
    finally:
        published.write_text(seed)
    assert not report["ok"] and _problems(report) == [SPOILT[spoilt]]


def test_the_audit_of_an_empty_archive_says_so(revealed, tmp_path):
    report = bd.audit(
        "toy",
        archive=tmp_path,
        true_root=tmp_path,
        cat_config=revealed.root / "cat_config.yaml",
    )
    assert not report["ok"] and report["problems"] == ["the archive holds no parts"]


@pytest.mark.parametrize("beside_signal", [False, True], ids=["alone", "beside_signal"])
def test_the_audit_passes_rho_tau_remeasured_at_run_noise(revealed, beside_signal):
    """ρ/τ carries no signal, and a re-run does not reproduce it (TreeCorr's
    k-means patches move θ by ~1e-3 and the values by O(1)): its part is
    judged by its stamp, and beside signal rows only the signal is compared."""
    true = AUDITED if beside_signal else rho_sacc()
    rerun = true.copy()
    cov = np.array(true.covariance.dense)
    for i, dp in enumerate(true.data):
        if not sio.is_signal(dp.data_type):
            rerun.data[i].value = 1.5 * dp.value + 1e-7
            rerun.data[i].tags["theta"] *= 1 + 1.7e-3
            cov[i, i] *= 1.036
    rerun.add_covariance(cov, overwrite=True)
    archived = revealed.sealed if beside_signal else sio.seal(true, revealed.blinded)
    report = revealed.audit(archived, rerun)
    assert report["ok"], report


@pytest.mark.parametrize("kind", [sio.CL_BB, sio.CL_EB])
def test_the_audit_fails_when_an_unshifted_row_moved(revealed, kind):
    true = AUDITED
    archived = revealed.sealed.copy()
    i = _rows(true, kind)[-1]
    archived.data[i].value += 1e-6 * abs(true.data[i].value)
    report = revealed.audit(archived, true)
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
def test_the_audit_bounds_derived_b_modes(revealed, e_kind, b_kind, b_sigma):
    """A derived part's E rows move with the blind; its B rows by ≤ ``B_SIGMA``."""
    true = derived_sacc("cosebis" if e_kind == sio.COSEBI_EE else "pure_eb")
    true.add_covariance(np.full(len(true.mean), 0.01), overwrite=True)
    archived = true.copy()
    for i in _rows(true, e_kind):
        archived.data[i].value += 0.3
    for i in _rows(true, b_kind):
        archived.data[i].value += b_sigma * 0.1
    report = revealed.audit(archived, true)
    assert report["ok"] == (b_sigma == 0.0), report
    (part,) = report["parts"].values()
    assert part["shift_over_sigma"][e_kind] == pytest.approx(3.0)
    if b_sigma:
        assert _problems(report) == [f"{b_kind} moved by 1.00e+00σ under the blind"]
