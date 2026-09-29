"""The blind and the file door.

A blind is drawn once (``blinding init``) into a registry outside git;
``sacc_io.save`` is the only writer, and conceals a blinded catalogue's ξ± and
Cℓ_EE rows in memory before the file exists. Blinds here use the fast
Eisenstein–Hu theory.
"""

import dataclasses
import json
import stat
import traceback

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from sp_validation import blinding as bd
from sp_validation import custody as cu
from sp_validation import sacc_io as sio
from sp_validation.blinding_theory import TheoryConfig

VERSIONS = ("TOY", "OTHER", "TOY_OPEN", "OTHER_OPEN", "TOY_MOCK")
FAST = dataclasses.asdict(TheoryConfig(transfer_function="eisenstein_hu"))


def _catalogues(root):
    """A catalogue config declaring one catalogue per custody."""
    blinds = {
        "TOY": "toy",
        "OTHER": "other",
        "TOY_OPEN": "none",
        "OTHER_OPEN": "none",
        "TOY_MOCK": "mock",
    }
    config = {"paths": {"blinds": str(root / "blinds")}}
    for name, blind in blinds.items():
        config[name] = {"shear": {"path": str(root / f"{name}.fits")}, "blind": blind}
    return config


@pytest.fixture(scope="module")
def blinds(tmp_path_factory):
    """The five custodies: TOY under blind `toy`, OTHER under `other`."""
    cats = _catalogues(tmp_path_factory.mktemp("registry"))
    for name in ("toy", "other"):
        bd.init(name, cats, fiducial=FAST)
    return {v: cu.custody_of(cats, v) for v in VERSIONS}


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
    blinded = custody.blinded
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
        and not (blinds[inputs[0]].blinded and content == "plaintext")
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
# The blind: drawn once, opened only under its commitment, never shown
# --------------------------------------------------------------------------- #
def test_a_blind_is_drawn_once_and_kept_private(tmp_path):
    cats = _catalogues(tmp_path)
    blind = bd.init("toy", cats, fiducial=FAST)
    assert stat.S_IMODE(blind.path.stat().st_mode) == 0o440
    assert stat.S_IMODE(blind.path.parent.stat().st_mode) == 0o700
    with pytest.raises(cu.CustodyError, match="drawn once"):
        bd.init("toy", cats, fiducial=FAST)


def test_a_blind_opens_only_under_its_commitment(blinds, monkeypatch):
    custody = blinds["TOY"]
    assert bd.open_blind(custody) == bd.Blind(
        "toy", custody.registry / "toy.blind.json"
    )
    with pytest.raises(cu.CustodyError, match="launch again"):
        bd.open_blind(dataclasses.replace(custody, commitment="0" * 64))
    monkeypatch.setattr(bd, "draw_scheme", lambda: 99)
    with pytest.raises(cu.CustodyError, match="draw scheme 2; .* under 99"):
        bd.open_blind(custody)


def _hidden_failure(blind, fiducial):
    """The traceback, locals included, of a theory failing at the hidden point
    with its parameters in its message."""

    def failing(params, *args):
        raise RuntimeError(f"cannot evaluate {params}")

    with pytest.MonkeyPatch.context() as m:
        m.setattr(bd, "xi_ccl", failing)
        with pytest.raises(bd.BlindingError) as failure:
            bd._at_hidden(bd._blocks(part(cl=False)), fiducial, blind)
    trace = traceback.TracebackException.from_exception(
        failure.value, capture_locals=True
    )
    return "".join(trace.format())


def test_the_hidden_cosmology_never_shows(tmp_path, capfd):
    """Neither the seed nor the hidden S8, Ωm or σ8 reaches a repr, the
    terminal or a traceback's locals."""
    cats = _catalogues(tmp_path)
    blind = bd.init("toy", cats, fiducial=FAST)
    record = json.loads(blind.path.read_text())
    hidden = bd._hidden(blind)
    sigma8 = hidden["S8"] / np.sqrt(hidden["Omega_m"] / 0.3)
    needles = [record["seed"], record["seed"][:16]] + [
        form(x)
        for x in (hidden["S8"], hidden["Omega_m"], sigma8)
        for form in (repr, "{:.4g}".format, "{:.6g}".format)
    ]
    bd.show("toy", cats)
    sio.seal(part(cl=False), cu.custody_of(cats, "TOY"))
    haystacks = [
        repr(blind),
        repr(hidden),
        str(hidden),
        f"{hidden}",
        "".join(capfd.readouterr()),
        _hidden_failure(blind, TheoryConfig(**record["fiducial"])),
    ]
    for i, haystack in enumerate(haystacks):
        # The message names the haystack only: a failure must not print a needle.
        assert not any(n in haystack for n in needles), f"haystack {i} shows it"
