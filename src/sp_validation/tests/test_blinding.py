"""The blind and the file door.

A blind is drawn once (``blinding init``) into a registry outside git;
``sacc_io.save`` is the only writer, and conceals a blinded catalogue's
shiftable rows in memory before the file exists. Blinds here shift by the
analytic ``toy_theory``; one slow test runs the CCL defaults.
"""

import dataclasses
import stat

import numpy as np
import pytest
from _synthetic import TOY_STANDARD, toy_theory

from sp_validation import blinding as bd
from sp_validation import custody as cu
from sp_validation import sacc_io as sio
from sp_validation import theory

VERSIONS = ("TOY", "OTHER", "TOY_OPEN", "TOY_MOCK")
DEFAULTS = {t: f for t, f in bd.STANDARD.items() if callable(f)}
CUSTOM = "galaxy_density_xi"


@pytest.fixture(scope="module", autouse=True)
def _toy_theory():
    with pytest.MonkeyPatch.context() as m:
        m.setattr(bd, "STANDARD", TOY_STANDARD)
        yield


def _catalogues(root):
    """A catalogue config declaring one catalogue per custody."""
    blinds = {
        "TOY": "toy",
        "OTHER": "other",
        "TOY_OPEN": "none",
        "TOY_MOCK": "mock",
    }
    config = {"paths": {"blinds": str(root / "blinds")}}
    for name, blind in blinds.items():
        config[name] = {"shear": {"path": str(root / f"{name}.fits")}, "blind": blind}
    return config


@pytest.fixture(scope="module")
def blinds(tmp_path_factory):
    """The four custodies: TOY under blind `toy`, OTHER under `other`."""
    cats = _catalogues(tmp_path_factory.mktemp("registry"))
    for name in ("toy", "other"):
        bd.init(name, cats)
    return {v: cu.custody_of(cats, v) for v in VERSIONS}


def _nz(z0):
    z = np.linspace(0.0, 3.0, 200)
    return z, np.exp(-0.5 * ((z - z0) / 0.2) ** 2)


def part(
    *,
    xi_tags=("reporting",),
    cl=True,
    rho=False,
    derived=None,
    unruled=False,
):
    """A two-bin part: ξ± under each of ``xi_tags`` (None: no tag) and
    pseudo-Cℓ (EE, BB, EB) on pairs (0,0), (0,1), (1,1), and optionally ρ,
    a derived statistic's rows and rows of a custom type."""
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
            s.add_data_point(CUSTOM, ("source_0", "source_0"), 1e-4, theta=x)
    s.add_covariance(np.abs(np.asarray(s.mean)) ** 2 + 1e-20)
    return s


def _values(s):
    return np.asarray(s.mean)


# --------------------------------------------------------------------------- #
# Born sealed: a blind shifts exactly the ξ± and Cℓ_EE rows, nothing else
# --------------------------------------------------------------------------- #
EVERYTHING = dict(xi_tags=("reporting", "mystery", None), rho=True)


@pytest.mark.parametrize(
    "content, version, refused",
    [
        (EVERYTHING, "TOY", False),
        (EVERYTHING, "TOY_OPEN", False),
        (EVERYTHING, "TOY_MOCK", False),
        (dict(derived="cosebis"), "TOY", True),
        (dict(derived="pure_eb"), "TOY", True),
        (dict(unruled=True), "TOY", True),
        (dict(derived="pure_eb", unruled=True), "TOY_OPEN", False),
    ],
)
def test_seal_shifts_only_the_signal_it_has_a_rule_for(
    blinds, content, version, refused
):
    """Under a blind every ξ± and Cℓ_EE row moves, whatever its grid tag, and
    every other value (Cℓ_BB/EB, ρ) and the covariance stay bitwise; a blinded
    birth carrying derived rows or a custom type without a theory is refused.
    Unblinded and mock births keep their values. Each is stamped with its
    custody."""
    s, custody = part(**content), blinds[version]
    if refused:
        with pytest.raises(ValueError, match="derived_from|no blinding rule"):
            sio.seal(s, custody)
        return
    sealed = sio.seal(s, custody)
    assert cu.read_stamp(sealed.metadata) == custody
    shiftable = np.array([callable(bd.STANDARD.get(dp.data_type)) for dp in s.data])
    moved = _values(sealed) != _values(s)
    assert np.array_equal(moved, shiftable & custody.blinded)
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


@pytest.mark.parametrize(
    "inputs, declared, content, allowed",
    [
        (["TOY"], None, "copy", True),
        (["TOY"], None, "cosebis", True),
        (["TOY_OPEN"], None, "plaintext", True),
        (["TOY", "TOY"], "TOY", "copy", True),  # an assembly
        (["TOY"], None, "plaintext", False),  # laundering
        (["OTHER"], "TOY", "copy", False),  # another blind
        (["TOY_OPEN"], "TOY", "copy", False),  # unblinded into blinded
        (["TOY", "OTHER"], None, "cosebis", False),  # mixed stamps
        (["TOY_MOCK", "TOY_OPEN"], None, "copy", False),  # mixed stamps
    ],
)
def test_a_derivation_carries_its_inputs_one_stamp(
    blinds, parts, tmp_path, inputs, declared, content, allowed
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
    path = tmp_path / "d.sacc"
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
    assert cu.read_stamp(loaded.metadata) == blinds[inputs[0]]
    assert np.array_equal(_values(loaded), _values(s))


# --------------------------------------------------------------------------- #
# The blind: drawn once, opened only under its commitment
# --------------------------------------------------------------------------- #
def test_a_blind_is_drawn_once_and_kept_private(tmp_path):
    cats = _catalogues(tmp_path)
    blind = bd.init("toy", cats)
    assert stat.S_IMODE(blind.path.stat().st_mode) == 0o440
    assert stat.S_IMODE(blind.path.parent.stat().st_mode) == 0o700
    with pytest.raises(cu.CustodyError, match="drawn once"):
        bd.init("toy", cats)


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


# --------------------------------------------------------------------------- #
# The shift: from each data type's theory, refused where it cannot be made
# --------------------------------------------------------------------------- #
def test_a_custom_type_is_shifted_by_its_own_theory(blinds, tmp_path):
    """Under a blind a custom data type is shifted when its birth passes a
    theory function and refused without one; its sealed rows then pass an
    assembly. A standard type the blind leaves alone takes no theory."""
    custody, s = blinds["TOY"], part(cl=False, unruled=True)
    with pytest.raises(ValueError, match="no blinding rule"):
        sio.save(s, tmp_path / "refused.sacc", custody=custody)
    sealed = sio.save(
        s, tmp_path / "custom.sacc", custody=custody, theory={CUSTOM: toy_theory}
    )
    rows = s.indices(CUSTOM)
    assert np.all(_values(sealed)[rows] != _values(s)[rows])
    sio.save(sealed.copy(), tmp_path / "assembled.sacc", derived_from=[sealed])
    with pytest.raises(ValueError, match="no theory"):
        sio.seal(part(), custody, theory={sio.CL_BB: toy_theory})


def test_conceal_refuses_a_theory_blind_to_cosmology(blinds):
    """A theory that ignores the cosmology is refused rather than stamped as
    blinded."""

    def flat(params, s, rows):
        return np.ones(len(rows))

    blind = bd.open_blind(blinds["TOY"])
    with pytest.raises(bd.BlindingError, match="unmoved"):
        bd.conceal(part(cl=False), blind, theory={sio.XI_PLUS: flat})


@pytest.mark.slow
def test_the_ccl_defaults_shift_with_s8(blinds, monkeypatch):
    """The default theories, at a hidden point with S8 above the fiducial,
    raise ξ+ and Cℓ_EE."""
    fiducial = theory.fiducial()
    monkeypatch.setattr(bd, "_hidden", lambda blind: {**fiducial, "S8": 0.9})
    blind = bd.open_blind(blinds["TOY"])
    s = part()
    delta = bd.conceal(s, blind, theory=DEFAULTS).mean - s.mean
    for data_type in (sio.XI_PLUS, sio.CL_EE):
        assert np.all(delta[s.indices(data_type)] > 0), data_type
