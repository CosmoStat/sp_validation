"""The blind and the file door.

A blind is drawn once (``blinding init``) into a registry beside a catalogue
config; ``sacc_io.save`` is the only writer, and conceals a blinded
catalogue's ξ± and Cℓ_EE rows in memory before the file exists. Blinds here
use the fast Eisenstein–Hu theory.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from hypothesis import example, given, settings
from hypothesis import strategies as st

from sp_validation import blinding as bd
from sp_validation import custody as cu
from sp_validation import sacc_io as sio

DATA = Path(__file__).parent / "data"
VERSIONS = ("TOY", "OTHER", "TOY_OPEN", "OTHER_OPEN", "TOY_MOCK")


def _registry(root, *blinds):
    """A catalogue config declaring one catalogue per custody state, and
    ``blinds`` ((name, base) pairs) drawn for it, each from a fixed seed."""
    entries = {
        "TOY": {},
        "OTHER": {},
        "TOY_OPEN": {"blinding": "unblinded"},
        "OTHER_OPEN": {"blinding": "unblinded"},
        "TOY_MOCK": {"blinding": "mock"},
    }
    (root / "cat_config.yaml").write_text(yaml.safe_dump(entries))
    (root / "fast.json").write_text(
        json.dumps({"theory": {"transfer_function": "eisenstein_hu"}})
    )
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


@pytest.fixture(scope="module")
def blinds(tmp_path_factory):
    """The five custodies: TOY under blind `toy`, OTHER under `other`."""
    root = _registry(
        tmp_path_factory.mktemp("registry"), ("toy", "TOY"), ("other", "OTHER")
    )
    return {v: _custody(root, v) for v in VERSIONS}


def _nz(z0):
    z = np.linspace(0.0, 3.0, 200)
    return z, np.exp(-0.5 * ((z - z0) / 0.2) ** 2)


def part(*, xi_tags=("reporting",), cl=True, rho=False, derived=None, unruled=False):
    """A two-bin part: ξ± under each of ``xi_tags`` (None: no tag) and
    pseudo-Cℓ (EE, BB, EB) on pairs (0,0), (0,1), (1,1), and optionally ρ/τ,
    a derived statistic's rows and rows of a type with no blinding rule."""
    s = sio.new_sacc({0: _nz(0.5), 1: _nz(0.9)})
    theta = np.geomspace(2.0, 200.0, 6)
    ell = np.array([30.0, 80.0, 150.0, 280.0, 450.0])
    w_ell = np.arange(2, 501).astype(float)
    window = np.exp(-0.5 * ((w_ell[:, None] - ell[None, :]) / 40.0) ** 2)
    for pair in ((0, 0), (0, 1), (1, 1)):
        for tag in xi_tags:
            xip, xim = 1e-4 * (theta / 10) ** -0.6, 0.5e-4 * (theta / 10) ** -0.9
            if tag is not None:
                sio.add_xi(s, pair, theta, xip, xim, grid=tag)
            else:
                for dtype, values in ((sio.XI_PLUS, xip), (sio.XI_MINUS, xim)):
                    for th, v in zip(theta, values):
                        s.add_data_point(dtype, sio._pair(pair), v, theta=th)
        if cl:
            ee = 1e-8 * (ell / 100.0) ** -1.2
            sio.add_pseudo_cl(
                s,
                pair,
                ell,
                ee,
                0.01 * ee,
                0.02 * ee,
                window_ells=w_ell,
                window_weights=window / window.sum(axis=0),
            )
    if rho:
        sio.add_rho(s, 0, theta, np.arange(1, 7) * 1e-7, np.arange(1, 7) * 2e-7)
    if derived == "cosebis":
        sio.add_cosebis(s, (0, 0), np.arange(1, 6) * 1e-10, (12.0, 83.0), Bn=np.ones(5))
    elif derived == "pure_eb":
        sio.add_pure_eb(s, (0, 0), theta[:4], **{k: np.ones(4) for k in sio.PURE_KEYS})
    if unruled:
        for x in (5.0, 20.0, 80.0):
            s.add_data_point(
                "galaxy_shearDensity_xi_t", ("source_0", "source_0"), 1e-5, theta=x
            )
    s.add_covariance(np.abs(np.asarray(s.mean)) ** 2 + 1e-20)
    return s


def _values(s):
    return np.asarray(s.mean)


# --------------------------------------------------------------------------- #
# Born sealed: a blind shifts exactly the ξ± and Cℓ_EE rows, and nothing else
# --------------------------------------------------------------------------- #
PARTS = st.fixed_dictionaries(
    {
        "xi_tags": st.lists(
            st.sampled_from(["reporting", "integration", "cosebis", None]),
            max_size=2,
            unique=True,
        ),
        "cl": st.booleans(),
        "rho": st.booleans(),
        "derived": st.sampled_from([None, "cosebis", "pure_eb"]),
        "unruled": st.booleans(),
    }
).filter(lambda p: p["xi_tags"] or p["cl"] or p["rho"])


@settings(max_examples=10, deadline=None)
@given(content=PARTS, version=st.sampled_from(["TOY", "TOY_OPEN", "TOY_MOCK"]))
@example(
    content=dict(
        xi_tags=["mystery", None], cl=True, rho=True, derived=None, unruled=False
    ),
    version="TOY",
)
def test_seal_shifts_only_the_signal_it_has_a_rule_for(blinds, content, version):
    """Under a blind every ξ± and Cℓ_EE row moves, whatever its grid tag, and
    every other value and the covariance stay bitwise; a birth carrying
    derived rows or a type with no blinding rule is refused. Unblinded and
    mock births keep their values. Each is stamped with its custody."""
    s, custody = part(**content), blinds[version]
    blinded = custody.status == "blinded"
    if blinded and (content["derived"] or content["unruled"]):
        with pytest.raises(ValueError, match="derived_from|no blinding rule"):
            sio.seal(s, custody)
        return
    sealed = sio.seal(s, custody)
    assert cu.read_stamp(sealed.metadata).stamp == custody.stamp
    shiftable = np.array([dp.data_type in sio.SHIFTABLE for dp in s.data])
    moved = _values(sealed) != _values(s)
    assert np.array_equal(moved, shiftable & blinded)
    assert np.array_equal(sealed.covariance.dense, s.covariance.dense)


def test_nothing_is_written_unstamped(blinds, tmp_path):
    """``save`` needs a custody or its inputs; ``load`` refuses a file born
    outside the door."""
    with pytest.raises(ValueError, match="custody"):
        sio.save(part(), tmp_path / "x.sacc")
    assert not (tmp_path / "x.sacc").exists()
    part().save_fits(str(tmp_path / "raw.sacc"))
    with pytest.raises(ValueError, match="stamp"):
        sio.load(tmp_path / "raw.sacc")


# --------------------------------------------------------------------------- #
# Derived: a derivation carries its inputs' one stamp, never plaintext
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def parts(blinds):
    """A ξ± part born under each custody."""
    return {v: sio.seal(part(cl=False), blinds[v]) for v in VERSIONS}


@settings(max_examples=25, deadline=None)
@given(
    inputs=st.lists(st.sampled_from(VERSIONS), min_size=1, max_size=3),
    declared=st.sampled_from([None, *VERSIONS]),
    content=st.sampled_from(["copy", "cosebis", "plaintext"]),
)
@example(inputs=["TOY"], declared=None, content="plaintext")  # laundering
@example(inputs=["TOY", "TOY"], declared="TOY", content="copy")  # an assembly
@example(inputs=["OTHER"], declared="TOY", content="copy")  # another blind
@example(inputs=["TOY_OPEN"], declared="TOY", content="copy")  # unblinded into blinded
@example(inputs=["TOY", "OTHER"], declared=None, content="cosebis")
def test_a_derivation_carries_its_inputs_one_stamp(
    blinds, parts, tmp_path_factory, inputs, declared, content
):
    """``save(s, derived_from=inputs, custody=declared)`` writes ``s`` under
    the inputs' stamp exactly when they share one, it is the declared
    custody's (if any), and, under a blind, ``s``'s ξ± rows are copies of
    theirs. Otherwise nothing is written."""
    s = {
        "copy": lambda: parts[inputs[0]].copy(),
        "cosebis": lambda: part(xi_tags=(), cl=False, derived="cosebis"),
        "plaintext": lambda: part(cl=False),
    }[content]()
    stamp = blinds[inputs[0]].stamp
    allowed = (
        all(blinds[v].stamp == stamp for v in inputs)
        and (declared is None or blinds[declared].stamp == stamp)
        and not (stamp["blinding"] == "blinded" and content == "plaintext")
    )
    path = tmp_path_factory.mktemp("derived") / "d.sacc"
    kwargs = dict(
        derived_from=[parts[v] for v in inputs],
        custody=blinds[declared] if declared else None,
    )
    if not allowed:
        with pytest.raises(ValueError):
            sio.save(s, path, **kwargs)
        assert not path.exists()
        return
    sio.save(s, path, **kwargs)
    loaded = sio.load(path)
    assert cu.read_stamp(loaded.metadata).stamp == stamp
    assert np.array_equal(_values(loaded), _values(s))


# --------------------------------------------------------------------------- #
# The blind: drawn once, its seed never on disk
# --------------------------------------------------------------------------- #
SEED = "5eed" * 8


@pytest.mark.parametrize("fault", [None, "after_key", "encrypt"])
def test_init_never_writes_the_seed(tmp_path, monkeypatch, fault):
    """A normal init, or one interrupted anywhere, leaves no file holding the
    seed."""
    from cryptography import fernet

    _registry(tmp_path)
    monkeypatch.setattr(bd.secrets, "token_hex", lambda n: SEED)
    if fault == "after_key":
        monkeypatch.setattr(bd.os, "replace", lambda src, dst: 1 / 0)
    elif fault == "encrypt":
        monkeypatch.setattr(fernet.Fernet, "encrypt", lambda self, data: 1 / 0)
    if fault is None:
        _init(tmp_path, "toy", "TOY")
    else:
        with pytest.raises(ZeroDivisionError):
            _init(tmp_path, "toy", "TOY")
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert not [p for p in files if SEED.encode() in p.read_bytes()]


def test_a_blind_is_drawn_once(tmp_path):
    _registry(tmp_path, ("toy", "TOY"))
    with pytest.raises(cu.CustodyError, match="exists"):
        _init(tmp_path, "toy", "OTHER")
    with pytest.raises(cu.CustodyError, match="already covered"):
        _init(tmp_path, "again", "TOY")


def test_the_commitment_hides_the_rng_seed():
    """The fork seeds its RNG from the undomained sha256 of the seed; the
    domain prefix keeps the public commitment from publishing it."""
    from smokescreen.param_shifts import _normalize_seed

    seed = "the-secret"
    assert int(cu.seed_commitment(seed)[:16], 16) != _normalize_seed(seed)


def test_the_committed_blind_opens_under_this_code():
    """A TheoryConfig field added without its value in ``blinding.NEUTRAL``
    would strand every live blind; ``tests/data/blinds/committed`` turns that
    red."""
    blind = bd._open(DATA / "blinds", "committed")
    assert (blind.hidden.S8, blind.hidden.Omega_m) == pytest.approx(
        (0.8374952413635888, 0.32730573347638175), rel=1e-12
    )


# --------------------------------------------------------------------------- #
# The reveal: publish, then prove blinded − true = shift(seed)
# --------------------------------------------------------------------------- #
def test_reveal_refuses_a_root_holding_nothing_concealed(tmp_path):
    """Publishing cannot be undone: a mistyped ``--root`` is refused before
    the seed is published."""
    root = _registry(tmp_path, ("toy", "TOY"))
    (root / "out").mkdir()
    sio.save(part(cl=False), root / "out" / "part.sacc", custody=_custody(root, "TOY"))
    cat_config = root / "cat_config.yaml"
    with pytest.raises(cu.CustodyError, match="nothing concealed"):
        bd.reveal("toy", root=root / "outptu", cat_config=cat_config)
    assert not (root / "blinds" / "toy" / "revealed.json").exists()
    archive = bd.reveal("toy", root=root / "out", cat_config=cat_config)
    assert (archive / "part.sacc").exists()


@pytest.mark.parametrize("left_true", [False, True], ids=["revealed", "mixed"])
def test_the_audit_proves_the_shift(tmp_path, left_true):
    """An archived part passes the audit against its re-measured twin only if
    every shiftable row carries the shift: a pair left true fails."""
    root = _registry(tmp_path, ("toy", "TOY"))
    blinded = _custody(root, "TOY")
    true = part()
    archived = sio.seal(true, blinded)
    if left_true:
        for i, dp in enumerate(true.data):
            if dp.tracers == sio._pair((1, 1)) and dp.data_type in sio.SHIFTABLE:
                archived.data[i].value = dp.value
    (root / "blinds" / "toy" / "revealed.json").write_text(
        json.dumps({"seed": bd.open_blind(blinded).seed})
    )
    for tree, s, stamp in (
        ("archive", archived, blinded.stamp),
        ("live", true, cu.Custody("unblinded", "TOY").stamp),
    ):
        s = s.copy()
        s.metadata.update(stamp)
        (root / tree).mkdir()
        s.save_fits(str(root / tree / "part.sacc"))
    report = bd.audit(
        "toy",
        archive=root / "archive",
        true_root=root / "live",
        cat_config=root / "cat_config.yaml",
    )
    assert report["ok"] != left_true, report
