"""Blinds, concealment and the blind stamp.

Blinds here shift by the analytic ``toy_shear``; one slow test runs the CCL
default.
"""

import json
import stat

import numpy as np
import pytest
from _synthetic import toy_shear
from smokescreen.param_shifts import draw_param_shifts

from sp_validation import blinding as bd
from sp_validation import sacc_io as sio
from sp_validation import theory

DELTA_SIGMA = "galaxy_shearDensity_xi_t"


@pytest.fixture(scope="module")
def catalogues(tmp_path_factory):
    return {"paths": {"blinds": str(tmp_path_factory.mktemp("blinds"))}}


@pytest.fixture(scope="module")
def toy(catalogues):
    return bd.init("toy", catalogues)


@pytest.fixture(scope="module")
def other(catalogues):
    return bd.init("other", catalogues)


def _nz(z0):
    z = np.linspace(0.0, 3.0, 200)
    return z, np.exp(-0.5 * ((z - z0) / 0.2) ** 2)


def part(*, cl=True, rho=False, delta_sigma=False):
    """ξ± and pseudo-Cℓ (EE, BB, EB) on pairs (0,0), (0,1), (1,1), and
    optionally ρ and a ΔΣ-like custom type."""
    s = sio.new_sacc({0: _nz(0.5), 1: _nz(0.9)})
    theta = np.geomspace(2.0, 200.0, 6)
    ell = np.array([30.0, 80.0, 150.0, 280.0, 450.0])
    w_ell = np.arange(2, 501).astype(float)
    window = np.exp(-0.5 * ((w_ell[:, None] - ell[None, :]) / 40.0) ** 2)
    for pair in ((0, 0), (0, 1), (1, 1)):
        xip, xim = 1e-4 * (theta / 10) ** -0.6, 0.5e-4 * (theta / 10) ** -0.9
        sio.add_xi(s, pair, theta, xip, xim, grid="reporting")
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
    if delta_sigma:
        for r in (0.5, 2.0, 8.0):  # tagged theta, as sacc requires
            s.add_data_point(DELTA_SIGMA, ("source_0", "source_1"), 10.0, theta=r)
    s.add_covariance(np.abs(np.asarray(s.mean)) ** 2 + 1e-20)
    return s


def hidden(blind):
    """The blind's hidden point, as Smokescreen draws it."""
    record = blind.record()
    shift = draw_param_shifts(record["envelope"], record["seed"])
    return {k: v + shift.get(k, 0.0) for k, v in record["fiducial"].items()}


# --------------------------------------------------------------------------- #
# Concealing: every row moves by its theory's t(hidden) − t(fiducial)
# --------------------------------------------------------------------------- #
def test_conceal_moves_signal_rows_by_the_theory_shift(toy, other):
    """ξ± and Cℓ_EE move by t(hidden) − t(fiducial); Cℓ_BB/EB and the
    covariance stay identical; the same blind gives the same shift, another
    blind another."""
    s = part()
    shift = bd.conceal(s, toy, toy_shear).mean - s.mean
    fiducial = toy.record()["fiducial"]
    expected = toy_shear(hidden(toy), s) - toy_shear(fiducial, s)
    np.testing.assert_allclose(shift, expected, rtol=1e-10, atol=0)
    signal = np.isin([dp.data_type for dp in s.data], [sio.XI_PLUS, sio.XI_MINUS])
    signal |= np.array([dp.data_type == sio.CL_EE for dp in s.data])
    assert np.all(shift[signal] != 0) and np.all(shift[~signal] == 0)
    concealed = bd.conceal(s, toy, toy_shear)
    np.testing.assert_array_equal(concealed.covariance.dense, s.covariance.dense)
    np.testing.assert_array_equal(concealed.mean - s.mean, shift)
    assert not np.allclose(bd.conceal(s, other, toy_shear).mean - s.mean, shift)


def test_a_theory_failure_names_no_value(toy):
    """Any failure, including a vector of the wrong length, is a
    BlindingError carrying neither the hidden point nor the original message."""

    def broken(params, s):
        raise RuntimeError(f"S8 = {params['S8']}")

    with pytest.raises(bd.BlindingError) as err:
        bd.conceal(part(), toy, broken)
    assert "S8" not in str(err.value) and "RuntimeError" in str(err.value)
    assert err.value.__cause__ is None and err.value.__context__ is None
    with pytest.raises(bd.BlindingError, match="values for"):
        bd.conceal(part(), toy, lambda params, s: np.zeros(3))


def test_a_custom_theory_blinds_a_custom_type(toy):
    """A ΔΣ-like type is shifted by a theory that knows it; the default
    theory refuses it; theory.none leaves every value but stamps."""

    def delta_sigma(params, s):
        out = toy_shear(params, s)
        for i, dp in enumerate(s.data):
            if dp.data_type == DELTA_SIGMA:
                out[i] = 10.0 * params["S8"] / dp.tags["theta"]
        return out

    s = part(cl=False, delta_sigma=True)
    rows = s.indices(DELTA_SIGMA)
    sealed = sio.seal(s, toy, delta_sigma)
    assert np.all(sealed.mean[rows] != s.mean[rows])
    assert sio.stamp(sealed) == "toy"
    with pytest.raises(bd.BlindingError, match="ValueError"):
        sio.seal(s, toy)  # theory.shear: no prediction for this type
    rho = part(cl=False, rho=True)
    kept = sio.seal(rho, toy, theory.none)
    np.testing.assert_array_equal(kept.mean, rho.mean)
    assert sio.stamp(kept) == "toy"


def test_none_stamps_without_concealing(tmp_path):
    s = part()
    saved = sio.save(s, tmp_path / "x.sacc", blind=bd.NONE)
    np.testing.assert_array_equal(saved.mean, s.mean)
    assert sio.stamp(sio.load(tmp_path / "x.sacc")) == "none"
    assert sio.stamp(part()) == "none"  # absent reads as none
    with pytest.raises(ValueError, match="blind= .* or derived_from="):
        sio.save(s, tmp_path / "y.sacc")


# --------------------------------------------------------------------------- #
# Derivations inherit their inputs' one stamp
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "inputs, declared, allowed",
    [
        (["toy"], None, True),
        (["toy", "toy"], "toy", True),  # an assembly
        (["none"], None, True),
        (["toy", "other"], None, False),  # mixed stamps
        (["toy", "none"], None, False),
        (["none"], "toy", False),  # a stale public part under a blind
        (["other"], "toy", False),
    ],
)
def test_a_derivation_carries_its_inputs_one_stamp(
    toy, other, tmp_path, inputs, declared, allowed
):
    blinds = {"toy": toy, "other": other, "none": bd.NONE}
    parts = [sio.seal(part(cl=False), blinds[b], toy_shear) for b in inputs]
    path = tmp_path / "d.sacc"
    kwargs = dict(derived_from=parts, blind=bd.Blind(declared) if declared else None)
    if not allowed:
        with pytest.raises(ValueError):
            sio.save(parts[0], path, **kwargs)
        assert not path.exists()
        return
    derived = sio.save(parts[0], path, **kwargs)
    assert sio.stamp(sio.load(path)) == inputs[0]
    np.testing.assert_array_equal(derived.mean, parts[0].mean)


# --------------------------------------------------------------------------- #
# The blind: drawn once, opened by name
# --------------------------------------------------------------------------- #
def test_a_blind_is_drawn_once_and_opened_by_name(tmp_path):
    catalogues = {"paths": {"blinds": str(tmp_path / "blinds")}}
    blind = bd.init("toy", catalogues)
    assert stat.S_IMODE(blind.path.stat().st_mode) == 0o440
    assert set(json.loads(blind.path.read_text())) == {"seed", "envelope", "fiducial"}
    with pytest.raises(bd.BlindingError, match="exists"):
        bd.init("toy", catalogues)
    assert bd.open_blind("toy", catalogues) == blind
    assert bd.open_blind("none", {}) is bd.NONE
    with pytest.raises(bd.BlindingError, match="init gone"):
        bd.open_blind("gone", catalogues)


@pytest.mark.slow
def test_the_ccl_default_shifts_with_s8(tmp_path):
    """theory.shear, at a hidden point with S8 above the fiducial, raises ξ+
    and Cℓ_EE and leaves Cℓ_BB/EB."""
    record = {
        "seed": "s",
        "envelope": {"S8": [0.05, 0.06]},
        "fiducial": theory.fiducial(),
    }
    path = tmp_path / "up.blind.json"
    path.write_text(json.dumps(record))
    s = part()
    delta = bd.conceal(s, bd.Blind("up", path)).mean - s.mean
    for data_type in (sio.XI_PLUS, sio.CL_EE):
        assert np.all(delta[s.indices(data_type)] > 0), data_type
    for data_type in (sio.CL_BB, sio.CL_EB):
        assert np.all(delta[s.indices(data_type)] == 0), data_type
