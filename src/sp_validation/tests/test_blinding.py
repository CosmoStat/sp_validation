"""Tests for :mod:`sp_validation.blinding` — per-part Smokescreen blinding.

Acceptance criteria AC1–AC9 of the blinding PRD, plus fast unit coverage of
the config/custody surface. Fast tests (no CCL import — the envelope
calibration, digest, commitment, fork-draw determinism, blind-init custody,
the merge alignment against a monkeypatched concealing factor, and the
assembly hash assertion) run in the default suite; the theory tests (fork +
CCL) are marked ``slow``; the derived-statistics tests additionally
``importorskip`` ``cosmo_numba``.

All fixtures are synthetic and deterministic. Each blindable intermediate is
its own standalone part SACC (reporting ξ±, integration ξ±, pseudo-Cℓ), as in the
per-part-at-birth architecture; derived statistics (COSEBIs, pure-E/B) are
never stored in parts — they are computed downstream from the (blinded) integration
ξ± through the pipeline seams ``b_modes.cosebis_from_xi`` /
``b_modes.pure_eb_from_xi``, exactly as the pipeline does. The blinding path
hands the fork no data vector at all (``smokescreen.concealing_factor`` is a
pure theory difference), so fixture ξ± values are smooth synthetic templates —
no theory fill is needed to blind.
"""

import json
import pathlib

import numpy as np
import pytest

from sp_validation import blinding as bd
from sp_validation import sacc_io as sio
from sp_validation.blinding_theory import TheoryConfig

_NOLOG = lambda *a, **k: None  # noqa: E731


# --------------------------------------------------------------------------- #
# Synthetic part fixtures
# --------------------------------------------------------------------------- #
def _gauss_nz(z0, sigma, n=200):
    z = np.linspace(0.0, 3.0, n)
    nz = np.exp(-0.5 * ((z - z0) / sigma) ** 2)
    return z, nz / np.trapezoid(nz, z)


def _reporting_theta(n=8):
    return np.geomspace(5.0, 250.0, n)


def _integration_theta(n=80):
    # The integration grid is the pure-E/B INTEGRATION grid, so it spans wider than
    # the reporting range on both ends (production: ~0.08–300 arcmin).
    return np.geomspace(0.1, 300.0, n)


def _xi_template(theta, k=0):
    """Smooth synthetic ξ± for pair index ``k`` (no CCL needed)."""
    theta = np.asarray(theta)
    xip = 1e-4 * (1 + 0.1 * k) * (theta / 10.0) ** -0.6
    xim = 0.5e-4 * (1 + 0.1 * k) * (theta / 10.0) ** -0.9
    return xip, xim


def _b_mode_template(theta, amplitude):
    """A smooth ξ_B(θ) template. B contributes +ξ_B to ξ+, −ξ_B to ξ−."""
    return amplitude * np.exp(-((np.log(np.asarray(theta) / 30.0)) ** 2) / 2.0)


def _nz_dict(nbins):
    return {i: _gauss_nz(0.5 + 0.3 * i, 0.15 + 0.02 * i) for i in range(nbins)}


def _pairs(nbins):
    return [(i, j) for i in range(nbins) for j in range(i, nbins)]


def make_xi_part(grid, nbins=1, b_amplitude=0.0):
    """A standalone ξ± part SACC (one grid), synthetic values, eye covariance.

    ``b_amplitude`` injects a pure B-mode (+ξ_B to ξ+, −ξ_B to ξ−;
    b_modes.py sign convention) — used on the integration part for AC4/AC9.
    """
    theta = _reporting_theta() if grid == "reporting" else _integration_theta()
    s = sio.new_sacc(
        _nz_dict(nbins), metadata={"catalogue_version": "vTEST", "type": "mock"}
    )
    blocks = []
    for k, (i, j) in enumerate(_pairs(nbins)):
        xip, xim = _xi_template(theta, k)
        xi_b = _b_mode_template(theta, b_amplitude)
        sio.add_xi(s, (i, j), theta, xip + xi_b, xim - xi_b, grid=grid)
        tr = sio._pair((i, j))
        idx = np.concatenate(
            [
                s.indices(sio.XI_PLUS, tr, grid=grid),
                s.indices(sio.XI_MINUS, tr, grid=grid),
            ]
        )
        blocks.append((idx, np.eye(len(idx)) * 1e-12))
    sio.assemble_covariance(s, blocks)
    return s


def make_cl_part(nbins=1):
    """A standalone pseudo-Cℓ part SACC (EE/BB/EB + bandpower windows)."""
    s = sio.new_sacc(
        _nz_dict(nbins), metadata={"catalogue_version": "vTEST", "type": "mock"}
    )
    ell_eff = np.array([30.0, 80.0, 150.0, 280.0, 450.0])
    w_ell = np.arange(2, 501).astype(float)
    w_mat = np.zeros((len(w_ell), len(ell_eff)))
    for b, le in enumerate(ell_eff):
        w_mat[:, b] = np.exp(-0.5 * ((w_ell - le) / 40.0) ** 2)
        w_mat[:, b] /= w_mat[:, b].sum()
    blocks = []
    for k, (i, j) in enumerate(_pairs(nbins)):
        cl_ee = 1e-8 * (1 + 0.1 * k) * (ell_eff / 100.0) ** -1.2
        sio.add_pseudo_cl(
            s,
            (i, j),
            ell_eff,
            cl_ee,
            np.zeros(5),
            np.zeros(5),
            window_ells=w_ell,
            window_weights=w_mat,
        )
        tr = sio._pair((i, j))
        idx = np.concatenate(
            [s.indices(dt, tr) for dt in (sio.CL_EE, sio.CL_BB, sio.CL_EB)]
        )
        blocks.append((idx, np.eye(len(idx)) * 1e-16))
    sio.assemble_covariance(s, blocks)
    return s


def make_rho_part():
    """A standalone ρ/τ PSF-diagnostics part SACC — never blindable."""
    ctheta = _reporting_theta()
    s = sio.new_sacc(
        _nz_dict(1), metadata={"catalogue_version": "vTEST", "type": "mock"}
    )
    blocks = []
    for k in range(2):
        sio.add_rho(
            s, k, ctheta, np.arange(len(ctheta)) * 1e-7, np.arange(len(ctheta)) * 2e-7
        )
        idx = np.concatenate(
            [s.indices(sio.RHO_PLUS.format(k=k)), s.indices(sio.RHO_MINUS.format(k=k))]
        )
        blocks.append((idx, np.eye(len(idx)) * 1e-18))
    sio.add_tau(
        s, (0,), 0, ctheta, np.arange(len(ctheta)) * 3e-7, np.arange(len(ctheta)) * 4e-7
    )
    idx = np.concatenate(
        [s.indices(sio.TAU_PLUS.format(k=0)), s.indices(sio.TAU_MINUS.format(k=0))]
    )
    blocks.append((idx, np.eye(len(idx)) * 1e-18))
    sio.assemble_covariance(s, blocks)
    return s


def make_parts(nbins=1, b_amplitude=0.0, with_rho=True):
    """All intermediate parts of one catalogue version, keyed by name."""
    parts = {
        "xi_reporting": make_xi_part("reporting", nbins),
        "xi_integration": make_xi_part("integration", nbins, b_amplitude=b_amplitude),
        "cl": make_cl_part(nbins),
    }
    if with_rho:
        parts["rho_tau"] = make_rho_part()
    return parts


def _derive_downstream(reporting_part, integration_part, nmodes=6):
    """COSEBIs + pure-E/B the way the pipeline derives them downstream.

    COSEBIs from the integration ξ± (full-range scale cut); pure-E/B from the
    measured reporting reporting ξ± + the integration integration ξ±, with the
    edge-based bounds set to the reporting grid's span — the outermost
    reporting point sits at tmax with no interior support and comes back
    NaN (the AC9 boundary case).
    """
    from sp_validation import b_modes

    theta_f, xip_f, xim_f = sio.get_xi(integration_part, (0, 0), grid="integration")
    theta_c, xip_c, xim_c = sio.get_xi(reporting_part, (0, 0), grid="reporting")
    En, Bn = b_modes.cosebis_from_xi(
        theta_f, xip_f, xim_f, nmodes, scale_cut=(theta_f.min(), theta_f.max())
    )
    modes = b_modes.pure_eb_from_xi(
        theta_c,
        xip_c,
        xim_c,
        theta_f,
        xip_f,
        xim_f,
        float(theta_c[0]),
        float(theta_c[-1]),
    )
    return En, Bn, modes


# --------------------------------------------------------------------------- #
# BlindingConfig: envelope calibration + digest (fast)
# --------------------------------------------------------------------------- #
def test_blinding_config_defaults():
    c = bd.BlindingConfig()
    assert c.s8_half_width == 0.075
    assert c.omega_m_half_width == 0.1
    assert c.theory.S8 == 0.80  # fiducial TheoryConfig defaults


def test_blinding_config_overrides_fail_loud():
    c = bd.BlindingConfig.from_overrides({"s8_half_width": 0.05})
    assert c.s8_half_width == 0.05
    with pytest.raises(ValueError, match="unknown BlindingConfig fields"):
        bd.BlindingConfig.from_overrides({"s8_half_width": 0.05, "bogus": 1})
    with pytest.raises(ValueError, match="unknown TheoryConfig fields"):
        bd.BlindingConfig.from_overrides({"theory": {"nope": 1}})


def test_blinding_config_is_frozen():
    with pytest.raises(Exception):
        bd.BlindingConfig().s8_half_width = 0.2


def test_envelope_calibration_maps_s8_box_to_ccl_halfwidths():
    """(S8, Ωm) half-widths → {sigma8, Omega_c} at the fiducial (exact forms)."""
    c = bd.BlindingConfig()
    shifts = c.shifts_dict()
    assert set(shifts) == {"sigma8", "Omega_c"}
    assert shifts["sigma8"] == pytest.approx(
        c.s8_half_width / np.sqrt(c.theory.Omega_m / 0.3)
    )
    assert shifts["Omega_c"] == c.omega_m_half_width
    # every shift key must exist in the fiducial point (fork contract)
    assert set(shifts) <= set(c.theory.ccl_params())


def test_config_digest_stable_and_sensitive():
    """Canonical digest: byte-stable across runs, moves with every bound field."""
    c = bd.BlindingConfig()
    assert c.config_digest() == bd.BlindingConfig().config_digest()
    assert len(c.config_digest()) == 64
    assert bd.BlindingConfig(s8_half_width=0.05).config_digest() != c.config_digest()
    assert (
        bd.BlindingConfig.from_overrides({"theory": {"S8": 0.79}}).config_digest()
        != c.config_digest()
    )
    # the P(k) recipe tokens are bound: a different halofit token = new digest
    assert (
        bd.BlindingConfig.from_overrides(
            {"theory": {"ccl_halofit_version": "takahashi"}}
        ).config_digest()
        != c.config_digest()
    )
    # the Boltzmann backend (#280) is bound too — a different transfer
    # function is a different P(k) path, so a different blind
    assert (
        bd.BlindingConfig.from_overrides(
            {"theory": {"transfer_function": "eisenstein_hu"}}
        ).config_digest()
        != c.config_digest()
    )


def test_theory_config_transfer_function_default_and_override():
    """#280: the Boltzmann backend is one config knob, CAMB by default (the
    inference pipeline's Boltzmann code), overridable like any other field."""
    assert TheoryConfig().transfer_function == "boltzmann_camb"
    cfg = TheoryConfig.from_overrides({"transfer_function": "eisenstein_hu"})
    assert cfg.transfer_function == "eisenstein_hu"


def test_config_digest_int_float_canonical():
    """One physical cosmology has one digest, regardless of int-vs-float literals.

    Configs come from YAML/CLI/humans, so a field can arrive as ``-1`` (int) or
    ``-1.0`` (float). The digest must depend on the numeric *value*, not the
    literal's Python type — otherwise an int-vs-float mismatch between the blind
    and a later unblind would raise "config digest mismatch" and deny a
    legitimate unblind. Every declared-float field must be canonical this way.
    """
    for field, int_val, float_val in [
        ("w0", -1, -1.0),
        ("wa", 0, 0.0),
        ("Omega_m", 1, 1.0),
        ("m_nu", 0, 0.0),
        ("S8", 1, 1.0),
    ]:
        assert (
            bd.BlindingConfig.from_overrides(
                {"theory": {field: int_val}}
            ).config_digest()
            == bd.BlindingConfig.from_overrides(
                {"theory": {field: float_val}}
            ).config_digest()
        ), f"int-vs-float digest split on theory.{field}"
    # the envelope half-widths (BlindingConfig's own float fields) too
    assert (
        bd.BlindingConfig.from_overrides({"s8_half_width": 1}).config_digest()
        == bd.BlindingConfig.from_overrides({"s8_half_width": 1.0}).config_digest()
    )
    # and a full round-trip: an all-int override matches the float default digest
    assert (
        bd.BlindingConfig.from_overrides(
            {"theory": {"w0": -1, "Omega_m": 0, "wa": 0}}
        ).config_digest()
        == bd.BlindingConfig.from_overrides(
            {"theory": {"w0": -1.0, "Omega_m": 0.0, "wa": 0.0}}
        ).config_digest()
    )


def test_theory_config_ccl_params_exact_keyset():
    """ccl_params() carries exactly the contracted keys — nothing rides along."""
    params = TheoryConfig().ccl_params()
    assert set(params) == {
        "Omega_c",
        "Omega_b",
        "h",
        "n_s",
        "sigma8",
        "m_nu",
        "mass_split",
        "w0",
        "wa",
        "Neff",
        "T_CMB",
    }
    assert params["sigma8"] == pytest.approx(0.80)  # S8=0.80 at Ωm=0.30
    assert params["Neff"] == 3.046 and params["T_CMB"] == 2.7255


# --------------------------------------------------------------------------- #
# Commitment + the fork's draw (fast; smokescreen import is light)
# --------------------------------------------------------------------------- #
def test_commitment_is_the_forks_domain_separated_digest():
    """One definition of the commitment, and it is the fork's."""
    import smokescreen

    seed = "the-secret"
    assert bd.seed_commitment(seed) == smokescreen.seed_commitment(seed)
    assert bd.seed_commitment("right") != bd.seed_commitment("wrong")


def test_commitment_does_not_embed_the_rng_seed():
    """The published commitment must not carry the effective RNG seed.

    The fork derives the base seed for its per-key RNG from the *undomained*
    sha256 of the seed string, taking the digest's first 8 bytes. A commitment
    hashed over the bare seed would therefore publish that base seed verbatim in
    its first 16 hex characters, and anyone holding the (public) commitment and
    the (public) fiducial config could redraw the hidden cosmology and subtract
    the blind. The domain prefix is what breaks that identity — this is the
    regression guard for it.
    """
    import secrets

    from smokescreen.param_shifts import _normalize_seed

    # The third seed is drawn exactly as blind_init draws a production one.
    for seed in ("my_secret_seed", "the-secret", secrets.token_hex(16)):
        commitment = bd.seed_commitment(seed)
        assert int(commitment[:16], 16) != _normalize_seed(seed)


def test_hidden_params_deterministic_and_in_envelope():
    """Same (seed, config) ⇒ same hidden point; draws respect the envelope."""
    import secrets

    c = bd.BlindingConfig()
    fid = c.theory.ccl_params()
    h1, h2 = bd.hidden_params("a-seed", c), bd.hidden_params("a-seed", c)
    assert h1 == h2
    assert bd.hidden_params("другой", c) != h1
    shifts = c.shifts_dict()
    for _ in range(50):
        h = bd.hidden_params(secrets.token_hex(8), c)
        for key, half in shifts.items():
            assert abs(h[key] - fid[key]) <= half
        # only the enveloped keys move
        assert all(h[k] == fid[k] for k in fid if k not in shifts)


def test_hidden_params_no_global_rng_state():
    """The fork's draw uses a local RNG — global numpy state is untouched."""
    np.random.seed(0)
    before = np.random.get_state()[1].copy()
    bd.hidden_params("whatever", bd.BlindingConfig())
    assert np.array_equal(before, np.random.get_state()[1])


# --------------------------------------------------------------------------- #
# Per-part merge + provenance with a monkeypatched factor (fast — no CCL)
# --------------------------------------------------------------------------- #
def _patch_constant_factor(monkeypatch, value=1e-6):
    def fake(part, indices, factory, config, seed):
        return np.arange(len(indices), dtype=float) * value + value

    monkeypatch.setattr(bd, "_concealing_factor", fake)
    return fake


def test_merge_places_shift_at_recorded_indices_only(monkeypatch):
    """AC5 (merge half), per part: the shift lands exactly on the blindable
    rows, in stored order; covariance, n(z), and every tag are untouched."""
    _patch_constant_factor(monkeypatch)
    for name, part in make_parts(nbins=2, with_rho=False).items():
        orig = np.array(part.mean)
        orig_cov = part.covariance.dense.copy()
        orig_nz = part.tracers["source_0"].nz.copy()

        blinded = bd.blind_sacc(part, "seed", log=_NOLOG)

        blocks = bd._blindable_blocks(part)
        assert len(blocks) == 1, f"{name}: a part carries exactly one block"
        shifted = np.zeros(len(orig), dtype=bool)
        for _, indices, _ in blocks:
            expected = orig[indices] + (np.arange(len(indices)) * 1e-6 + 1e-6)
            assert np.array_equal(np.array(blinded.mean)[indices], expected)
            shifted[indices] = True
        assert np.array_equal(np.array(blinded.mean)[~shifted], orig[~shifted])
        assert np.array_equal(blinded.covariance.dense, orig_cov)
        assert np.array_equal(blinded.tracers["source_0"].nz, orig_nz)
        # row-order preservation: type/tracers/tags sequence is bitwise unchanged
        for a, b in zip(part.data, blinded.data):
            assert a.data_type == b.data_type
            assert a.tracers == b.tracers
            assert a.tags == b.tags


def test_provenance_metadata_contract(monkeypatch):
    """Blinded parts carry concealed/blind/commitment/digest; no seed key."""
    _patch_constant_factor(monkeypatch)
    s = make_xi_part("reporting")
    s.metadata["seed_smokescreen"] = "leaked!"  # must be stripped
    c = bd.BlindingConfig()
    blinded = bd.blind_sacc(s, "seed", config=c, label="B", log=_NOLOG)
    assert blinded.metadata["concealed"] is True
    assert blinded.metadata["blind"] == "B"
    assert blinded.metadata["blind_commitment"] == bd.seed_commitment("seed")
    assert blinded.metadata["blind_config_digest"] == c.config_digest()
    assert blinded.metadata["blind_draw_scheme"] == bd.draw_scheme()
    assert "seed_smokescreen" not in blinded.metadata
    assert blinded.metadata["catalogue_version"] == "vTEST"


def test_blind_refuses_double_blind(monkeypatch):
    _patch_constant_factor(monkeypatch)
    s = make_xi_part("reporting")
    blinded = bd.blind_sacc(s, "seed", log=_NOLOG)
    with pytest.raises(ValueError, match="already concealed"):
        bd.blind_sacc(blinded, "seed2", log=_NOLOG)


def test_blind_refuses_non_blindable_part():
    """A ρ/τ diagnostic part must never see a blind call — loud refusal."""
    with pytest.raises(ValueError, match="no blindable block"):
        bd.blind_sacc(make_rho_part(), "seed", log=_NOLOG)


def test_unblind_fails_closed_on_wrong_seed_or_config(monkeypatch):
    """AC6 (in-memory half): wrong seed and wrong config both refuse loudly."""
    _patch_constant_factor(monkeypatch)
    s = make_xi_part("reporting")
    blinded = bd.blind_sacc(s, "right-seed", log=_NOLOG)
    with pytest.raises(ValueError, match="blind_commitment"):
        bd.unblind_sacc(blinded, "wrong-seed", log=_NOLOG)
    with pytest.raises(ValueError, match="blind_config_digest"):
        bd.unblind_sacc(
            blinded,
            "right-seed",
            config=bd.BlindingConfig(s8_half_width=0.01),
            log=_NOLOG,
        )
    with pytest.raises(ValueError, match="not concealed"):
        bd.unblind_sacc(s, "right-seed", log=_NOLOG)


# --------------------------------------------------------------------------- #
# The vector core: full-length theory in, block slice out (fast — fake backend)
# --------------------------------------------------------------------------- #
def _fake_factory(indices, hole=None):
    """A backend that fills ``indices`` with ``sigma8 * arange`` and nothing else.

    Mimics the real backends' contract — a block's ``theory_fn`` reads its
    layout off the SACC it is given and returns a full-length vector, NaN on
    every row outside its own block — without importing CCL. ``hole`` leaves
    one row of the block itself unfilled.
    """

    def factory(s, theory):
        def theory_fn(params):
            out = np.full(len(s.mean), np.nan)
            out[indices] = params["sigma8"] * np.arange(len(indices))
            if hole is not None:
                out[hole] = np.nan
            return out

        return theory_fn

    return factory


def test_concealing_factor_slices_its_own_block_from_a_full_length_vector():
    """The factor is the theory difference at the block's rows, and the rows
    the backend does not fill are never read.

    This is the shape of the whole path after the sub-SACC carving came out:
    the backend is driven off the assembled SACC directly, returns a
    full-length vector that is NaN everywhere but its own block, and
    ``_concealing_factor`` returns exactly that block's rows. Checked against
    :func:`hidden_params`, which reaches the same hidden point by an
    independent route.
    """
    ordered = ["xi_reporting", "cl", "rho_tau", "xi_integration"]
    s = sio.gather([make_parts(nbins=1)[k] for k in ordered])
    blocks = bd._blindable_blocks(s)
    assert len(blocks) == 3, "assembled file carries all three blindable blocks"

    cfg = bd.BlindingConfig()
    delta = bd.hidden_params("seed", cfg)["sigma8"] - cfg.theory.ccl_params()["sigma8"]
    assert delta != 0.0
    for _, indices, _ in blocks:
        factor = bd._concealing_factor(s, indices, _fake_factory(indices), cfg, "seed")
        assert factor.shape == (len(indices),)
        assert np.allclose(factor, delta * np.arange(len(indices)))


def test_concealing_factor_refuses_a_row_its_backend_cannot_fill():
    """A NaN on a row the block *claims* is a layout the backend cannot cover.

    Slicing to the block drops the NaNs outside it by construction; this is
    the converse guard, and it has to be explicit — without it a row the
    backend silently skipped would be shifted by NaN, destroying that point
    with no error anywhere.
    """
    part = make_xi_part("reporting")
    ((_, indices, _),) = bd._blindable_blocks(part)
    with pytest.raises(ValueError, match="unfilled"):
        bd._concealing_factor(
            part,
            indices,
            _fake_factory(indices, hole=indices[0]),
            bd.BlindingConfig(),
            "seed",
        )


# --------------------------------------------------------------------------- #
# Draw-scheme binding: the blind is (seed, config, draw semantics)
# --------------------------------------------------------------------------- #
def test_draw_scheme_is_the_installed_fork_constant():
    """The recorded scheme is read from Smokescreen, not hardcoded here."""
    import smokescreen

    assert bd.draw_scheme() == int(smokescreen.DRAW_SCHEME)
    assert isinstance(bd.draw_scheme(), int)


def test_assert_draw_scheme_message_names_both_versions(monkeypatch):
    """A scheme mismatch says which scheme made the blind and which is installed."""
    monkeypatch.setattr(bd, "draw_scheme", lambda: 2)
    bd._assert_draw_scheme(2, "the blind")  # matching scheme is silent
    with pytest.raises(ValueError, match=r"DRAW_SCHEME=1.*DRAW_SCHEME=2"):
        bd._assert_draw_scheme(1, "the blind")
    with pytest.raises(ValueError, match="no draw-scheme record"):
        bd._assert_draw_scheme(None, "the blind")


def test_unblind_refuses_a_blind_drawn_under_another_scheme(monkeypatch):
    """The finding this closes: a blind added under one draw scheme and
    subtracted under another silently produces a wrong data vector, because
    the seed hash, the config digest and the escrow check all still pass. The
    scheme must be part of the custody state, checked before any subtraction.
    """
    _patch_constant_factor(monkeypatch)
    blinded = bd.blind_sacc(make_xi_part("reporting"), "seed", log=_NOLOG)
    assert blinded.metadata["blind_draw_scheme"] == bd.draw_scheme()

    # everything else about this file is valid — only the draw semantics moved
    monkeypatch.setattr(
        bd, "draw_scheme", lambda: blinded.metadata["blind_draw_scheme"] + 1
    )
    with pytest.raises(ValueError, match="DRAW_SCHEME"):
        bd.unblind_sacc(blinded, "seed", log=_NOLOG)


def test_unblind_refuses_a_file_predating_scheme_binding(monkeypatch):
    """A blinded file with no scheme record fails closed, not open."""
    _patch_constant_factor(monkeypatch)
    blinded = bd.blind_sacc(make_xi_part("reporting"), "seed", log=_NOLOG)
    del blinded.metadata["blind_draw_scheme"]
    with pytest.raises(ValueError, match="no draw-scheme record"):
        bd.unblind_sacc(blinded, "seed", log=_NOLOG)


def test_unblind_strips_the_scheme_stamp_with_the_rest(monkeypatch):
    _patch_constant_factor(monkeypatch)
    blinded = bd.blind_sacc(make_xi_part("reporting"), "seed", log=_NOLOG)
    part = bd.unblind_sacc(blinded, "seed", log=_NOLOG)
    assert "blind_draw_scheme" not in part.metadata


def test_read_seed_fails_closed_on_scheme_drift(tmp_path, monkeypatch):
    """blind_part reads the seed through _read_seed, so a scheme change between
    blinding part 1 and part 2 is caught before the second part is shifted."""
    bd.blind_init(str(tmp_path), log=_NOLOG)
    monkeypatch.setattr(bd, "draw_scheme", lambda: 99)
    with pytest.raises(ValueError, match="DRAW_SCHEME"):
        bd._read_seed(str(tmp_path), bd.BlindingConfig())


def test_stamp_passthrough_carries_the_committed_scheme(tmp_path, monkeypatch):
    """A pass-through part inherits the blind's scheme, and cannot be stamped
    from an install that draws differently."""
    paths = bd.blind_init(str(tmp_path), log=_NOLOG)
    s = bd.stamp_concealed_passthrough(make_rho_part(), paths["commitment"])
    assert s.metadata["blind_draw_scheme"] == bd.draw_scheme()

    monkeypatch.setattr(bd, "draw_scheme", lambda: 99)
    with pytest.raises(ValueError, match="DRAW_SCHEME"):
        bd.stamp_concealed_passthrough(make_rho_part(), paths["commitment"])


def test_assert_consistent_blind_refuses_divergent_or_foreign_schemes(monkeypatch):
    """Parts blinded under different schemes never assemble; nor does a set
    that agrees with itself but not with the installed fork."""
    parts = make_parts(nbins=1, with_rho=False)
    for p in parts.values():
        _stamp(p)
    parts["cl"].metadata["blind_draw_scheme"] = bd.draw_scheme() + 1
    with pytest.raises(ValueError, match="different blind commitments"):
        bd.assert_consistent_blind(list(parts.values()))

    parts = make_parts(nbins=1, with_rho=False)
    for p in parts.values():
        _stamp(p)
        p.metadata["blind_draw_scheme"] = bd.draw_scheme() + 1
    with pytest.raises(ValueError, match="DRAW_SCHEME"):
        bd.assert_consistent_blind(list(parts.values()))


# --------------------------------------------------------------------------- #
# blind-init custody + assembly hash assertion (fast — encryption only)
# --------------------------------------------------------------------------- #
def test_blind_init_writes_commitment_and_encrypted_bundle_only(tmp_path):
    """AC6 (init): commitment.json + encrypted bundle; never a plaintext seed."""
    paths = bd.blind_init(str(tmp_path), log=_NOLOG)
    with open(paths["commitment"], encoding="utf-8") as f:
        commitment = json.load(f)
    assert set(commitment) == {
        "label",
        "seed_commitment",
        "config_digest",
        "draw_scheme",
    }
    assert len(commitment["seed_commitment"]) == 64
    assert commitment["config_digest"] == bd.BlindingConfig().config_digest()
    # exactly the three custody outputs, no plaintext bundle
    assert {p.name for p in tmp_path.iterdir()} == {
        "commitment.json",
        "blind_seed.encrpt",
        "blind_seed.key",
    }
    # the decrypted seed matches the public commitment
    bundle = bd._read_encrypted_json(paths["bundle"], paths["key"])
    assert bd.seed_commitment(bundle["seed"]) == commitment["seed_commitment"]
    # one-shot custody: a second init in the same dir refuses
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        bd.blind_init(str(tmp_path), log=_NOLOG)


def test_read_seed_fails_closed_on_drifted_config(tmp_path):
    bd.blind_init(str(tmp_path), log=_NOLOG)
    with pytest.raises(ValueError, match="config digest"):
        bd._read_seed(str(tmp_path), bd.BlindingConfig(s8_half_width=0.01))


def _stamp(s, seed="s", label="A", config=None):
    bd._stamp_provenance(
        s,
        bd.seed_commitment(seed),
        label,
        (config or bd.BlindingConfig()).config_digest(),
    )
    return s


def test_assert_consistent_blind_shared_stamp():
    """One commitment across all blindable parts ⇒ the shared stamp returns;
    ρ/τ parts are exempt from the assertion."""
    parts = make_parts(nbins=1)
    for name in ("xi_reporting", "xi_integration", "cl"):
        _stamp(parts[name])
    stamp = bd.assert_consistent_blind(list(parts.values()))
    assert stamp == {
        "concealed": True,
        "blind": "A",
        "blind_commitment": bd.seed_commitment("s"),
        "blind_config_digest": bd.BlindingConfig().config_digest(),
        "blind_draw_scheme": bd.draw_scheme(),
    }


def test_assert_consistent_blind_fails_closed():
    """AC6 (assembly): mismatched commitments and mixed states both refuse."""
    parts = make_parts(nbins=1, with_rho=False)
    _stamp(parts["xi_reporting"], seed="one")
    _stamp(parts["xi_integration"], seed="one")
    _stamp(parts["cl"], seed="two")  # different seed ⇒ different commitment
    with pytest.raises(ValueError, match="different blind commitments"):
        bd.assert_consistent_blind(list(parts.values()))

    parts = make_parts(nbins=1, with_rho=False)
    _stamp(parts["xi_reporting"])  # blinded beside plaintext blindable parts
    with pytest.raises(ValueError, match="mixed"):
        bd.assert_consistent_blind(list(parts.values()))


def test_assert_consistent_blind_all_plaintext_is_none():
    """A declared-mock plaintext assembly (nothing blinded) asserts nothing."""
    assert bd.assert_consistent_blind(list(make_parts().values())) is None


def test_assert_consistent_blind_unconcealed_data_fails_closed():
    """PRD §4 "Mocks vs data": an unconcealed blindable part may assemble only
    if its metadata declares ``type == "mock"`` — an unconcealed ``data`` part,
    or one missing the tag, fails closed (skipping the blind can never
    silently expose real data). Mixed concealed/plaintext still fails on the
    mixed guard regardless of type."""
    parts = make_parts(nbins=1, with_rho=False)
    parts["xi_integration"].metadata["type"] = "data"
    with pytest.raises(ValueError, match="type"):
        bd.assert_consistent_blind(list(parts.values()))

    parts = make_parts(nbins=1, with_rho=False)
    del parts["cl"].metadata["type"]  # missing tag counts as not-a-mock
    with pytest.raises(ValueError, match="<missing>"):
        bd.assert_consistent_blind(list(parts.values()))

    # concealed data parts assemble integration (that is the whole point of the blind)
    parts = make_parts(nbins=1, with_rho=False)
    for p in parts.values():
        p.metadata["type"] = "data"
        _stamp(p)
    assert bd.assert_consistent_blind(list(parts.values()))["concealed"] is True

    # mixed concealed/plaintext fails on the mixed guard even for mocks
    parts = make_parts(nbins=1, with_rho=False)
    _stamp(parts["xi_reporting"])
    with pytest.raises(ValueError, match="mixed"):
        bd.assert_consistent_blind(list(parts.values()))


def test_gather_fails_closed_on_unconcealed_data_part():
    """The type guard reaches the terminal gather surface too."""
    parts = make_parts(nbins=1, with_rho=False)
    parts["xi_reporting"].metadata["type"] = "data"
    with pytest.raises(ValueError, match="type"):
        sio.gather(list(parts.values()))


def test_assert_consistent_blind_differing_labels_warn_not_fail():
    """Same seed+config, different --label ⇒ assemble cleanly with a warning.

    The label is provenance, not custody state: parts blinded under one blind
    but tagged with different labels must not be misread as different blinds.
    The assembly succeeds (keyed on commitment+digest); a distinct warning
    surfaces the label divergence rather than a false "different commitments".
    """
    parts = make_parts(nbins=1, with_rho=False)
    _stamp(parts["xi_reporting"], seed="one", label="A")
    _stamp(parts["xi_integration"], seed="one", label="A")
    _stamp(parts["cl"], seed="one", label="B")  # same blind, different label
    with pytest.warns(UserWarning, match="different labels"):
        stamp = bd.assert_consistent_blind(list(parts.values()))
    assert stamp["blind_commitment"] == bd.seed_commitment("one")
    assert stamp["blind"] == "A"  # deterministic: sorted-first label


def test_gather_assembles_parts_and_stamps_blind():
    """The sacc_io gather combines parts (points, tags, covariance blocks),
    calls the assembly assertion, and stamps the shared blind."""
    parts = make_parts(nbins=1)
    for name in ("xi_reporting", "xi_integration", "cl"):
        _stamp(parts[name])
    ordered = [parts[k] for k in ("xi_reporting", "cl", "rho_tau", "xi_integration")]
    s = sio.gather(ordered, metadata={"catalogue_version": "vTEST", "type": "mock"})

    assert len(s.mean) == sum(len(p.mean) for p in ordered)
    assert np.array_equal(np.array(s.mean), np.concatenate([p.mean for p in ordered]))
    # covariance: block-diagonal of the parts, in order
    cursor = 0
    for p in ordered:
        n = len(p.mean)
        assert np.array_equal(
            s.covariance.dense[cursor : cursor + n, cursor : cursor + n],
            p.covariance.dense,
        )
        cursor += n
    # the integration rows are addressable by the grid tag in the assembled file
    assert len(s.indices(sio.XI_PLUS, grid="integration")) == len(
        parts["xi_integration"].indices(sio.XI_PLUS, grid="integration")
    )
    # bandpower windows survive assembly
    assert s.get_bandpower_windows(s.indices(sio.CL_EE)) is not None
    # blind stamp on the assembled file
    assert s.metadata["concealed"] is True
    assert s.metadata["blind_commitment"] == bd.seed_commitment("s")
    assert s.metadata["catalogue_version"] == "vTEST"


def test_gather_fails_closed_on_mismatched_blinds():
    parts = make_parts(nbins=1, with_rho=False)
    _stamp(parts["xi_reporting"], seed="one")
    _stamp(parts["xi_integration"], seed="two")
    _stamp(parts["cl"], seed="one")
    with pytest.raises(ValueError, match="different blind commitments"):
        sio.gather(list(parts.values()))


def test_cli_blind_init_refuses_existing_state(tmp_path):
    """The blind-init CLI refuses to overwrite a previous blind's state."""
    import importlib.util

    script = (
        pathlib.Path(__file__).resolve().parents[3] / "scripts" / "blind_data_vector.py"
    )
    spec = importlib.util.spec_from_file_location("_blind_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    (tmp_path / "commitment.json").write_text("{}")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        cli.main(["blind-init", str(tmp_path)])


# --------------------------------------------------------------------------- #
# AC2 + AC3 + AC7: the shift itself (slow — fork + CCL)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.parametrize(
    "transfer_function",
    ["boltzmann_camb", "eisenstein_hu"],
    ids=["default-camb", "non-default-eh"],
)
def test_ac2_on_file_shift_equals_theory_difference_per_part(transfer_function):
    """AC2: per-row shift on each blinded part == theory_fn(hidden) −
    theory_fn(fiducial), hidden recovered by re-running the fork's draw —
    for all three parts, on a two-bin fixture; and the recovered hidden
    cosmology is identical across the three parts (one seed → one hidden).

    Scope: this verifies fork-draw recovery + placement (the shift on the
    file is exactly what re-running the same backend at the recovered
    hidden/fiducial points predicts, at the recorded rows). It is NOT a
    backend-correctness test: both sides run the identical ``theory_fn``, so
    any wrong-cosmology dependence cancels and a wrong backend would still
    pass here. Backend correctness is carried by AC3.

    Parametrized over the Boltzmann backend (#280): the default CAMB route
    and one non-default (Eisenstein–Hu — cheap, no CAMB run) both thread the
    same ``transfer_function`` knob through all three theory backends."""
    cfg = bd.BlindingConfig.from_overrides(
        {"theory": {"transfer_function": transfer_function}}
    )
    seed = "ac2-seed"
    parts = make_parts(nbins=2, with_rho=False)

    hiddens, worst = [], 0.0
    for name, part in parts.items():
        blinded = bd.blind_sacc(part, seed, config=cfg, log=_NOLOG)
        hidden = bd.hidden_params(seed, cfg)  # recovered per part
        hiddens.append(hidden)
        fiducial = cfg.theory.ccl_params()
        ((block_name, indices, factory),) = bd._blindable_blocks(part)
        # Independent of the blinding path: the factory is driven directly off
        # the part, at the hidden point recovered by hidden_params, and the two
        # theory vectors are differenced here rather than by the fork. The
        # factory fills only its own block, so slice to it.
        theory = factory(part, cfg.theory)
        expected = (theory(hidden) - theory(fiducial))[indices]
        actual = np.array(blinded.mean)[indices] - np.array(part.mean)[indices]
        gap = np.max(np.abs(actual - expected))
        scale = np.max(np.abs(expected))
        worst = max(worst, gap / scale)
        assert gap <= 1e-10 * max(scale, 1e-30), f"{name}/{block_name}: |Δ|={gap:.3e}"
    # one seed → one hidden cosmology across all parts
    assert hiddens[0] == hiddens[1] == hiddens[2]
    print(f"\nAC2 max relative shift mismatch across parts: {worst:.3e}")


@pytest.mark.slow
def test_ac3_cross_backend_against_independent_ccl_reference():
    """AC3: the realized shift matches an independently written direct-CCL
    reference (self-contained here; does not touch the blinding backends)."""
    import pyccl as ccl

    cfg = bd.BlindingConfig()
    seed = "ac3-seed"
    reporting = make_xi_part("reporting", nbins=2)
    cl_part = make_cl_part(nbins=2)
    blinded_xi = bd.blind_sacc(reporting, seed, config=cfg, log=_NOLOG)
    blinded_cl = bd.blind_sacc(cl_part, seed, config=cfg, log=_NOLOG)
    hidden = bd.hidden_params(seed, cfg)
    fiducial = cfg.theory.ccl_params()

    # ----- independent reference (from scratch; same fixture n(z), θ, ℓ) ----
    def ref_cosmo(params):
        return ccl.Cosmology(
            **params,
            matter_power_spectrum="camb",
            extra_parameters={
                "camb": {"halofit_version": "mead2020_feedback", "HMCode_logT_AGN": 7.5}
            },
        )

    ell = np.unique(
        np.concatenate([np.arange(2, 50), np.geomspace(50, 6e4, 200)]).astype(float)
    )

    def ref_xi(params, nz_i, nz_j, theta):
        cosmo = ref_cosmo(params)
        ti = ccl.WeakLensingTracer(cosmo, dndz=nz_i)
        tj = ccl.WeakLensingTracer(cosmo, dndz=nz_j)
        cl = ccl.angular_cl(cosmo, ti, tj, ell)
        xip = ccl.correlation(cosmo, ell=ell, C_ell=cl, theta=theta / 60.0, type="GG+")
        xim = ccl.correlation(cosmo, ell=ell, C_ell=cl, theta=theta / 60.0, type="GG-")
        return xip, xim

    worst = 0.0
    for i, j in bd.xi_pairs(reporting, "reporting"):
        tr = sio._pair((i, j))
        theta = sio._tag(reporting, sio.XI_PLUS, tr, "theta", grid="reporting")
        nz_i, nz_j = sio.get_nz(reporting, i), sio.get_nz(reporting, j)
        xip_h, xim_h = ref_xi(hidden, nz_i, nz_j, theta)
        xip_f, xim_f = ref_xi(fiducial, nz_i, nz_j, theta)
        for dt, ref_shift in (
            (sio.XI_PLUS, xip_h - xip_f),
            (sio.XI_MINUS, xim_h - xim_f),
        ):
            idx = reporting.indices(dt, tr, grid="reporting")
            realized = np.array(blinded_xi.mean)[idx] - np.array(reporting.mean)[idx]
            worst = max(worst, np.max(np.abs(realized - ref_shift)))
    print(f"\nAC3 max |realized − independent reference| (reporting ξ±): {worst:.3e}")
    assert worst < 1e-8  # observed ~1e-10; factor magnitudes ~1e-6

    # pseudo-Cℓ: W @ ΔCℓ_EE against the same independent reference
    for i, j in bd.cl_pairs(cl_part):
        tr = sio._pair((i, j))
        idx = cl_part.indices(sio.CL_EE, tr)
        window = cl_part.get_bandpower_windows(idx)
        w_ell = np.asarray(window.values, dtype=float)
        w_mat = np.asarray(window.weight, dtype=float)
        nz_i, nz_j = sio.get_nz(cl_part, i), sio.get_nz(cl_part, j)

        def ref_cl(params):
            cosmo = ref_cosmo(params)
            ti = ccl.WeakLensingTracer(cosmo, dndz=nz_i)
            tj = ccl.WeakLensingTracer(cosmo, dndz=nz_j)
            return ccl.angular_cl(cosmo, ti, tj, w_ell)

        ref_shift = w_mat.T @ (ref_cl(hidden) - ref_cl(fiducial))
        realized = np.array(blinded_cl.mean)[idx] - np.array(cl_part.mean)[idx]
        assert np.max(np.abs(realized - ref_shift)) < 1e-12  # bandpowers ~1e-9


@pytest.mark.slow
def test_ac7_reproducibility_same_seed_same_shift():
    """AC7: two blind runs of the same part with the same (seed, config)
    produce identical shifts; a different seed produces a different one."""
    part = make_xi_part("reporting")
    b1 = bd.blind_sacc(part, "repro-seed", log=_NOLOG)
    b2 = bd.blind_sacc(part, "repro-seed", log=_NOLOG)
    assert np.array_equal(np.array(b1.mean), np.array(b2.mean))
    b3 = bd.blind_sacc(part, "other-seed", log=_NOLOG)
    assert not np.array_equal(np.array(b1.mean), np.array(b3.mean))


# --------------------------------------------------------------------------- #
# AC1, AC4, AC5, AC9: born-blinded derived statistics (slow + cosmo_numba)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_ac1_zero_shift_is_identity():
    """AC1: a zero envelope reproduces every part exactly, and the integration part
    run downstream through the b_modes seams yields COSEBIs and pure-E/B
    identical to the unblinded run — the per-part plumbing and the
    born-blinded derivation path are the identity at zero shift."""
    pytest.importorskip("cosmo_numba")
    zero = bd.BlindingConfig(s8_half_width=0.0, omega_m_half_width=0.0)
    parts = make_parts(nbins=1, with_rho=False)
    blinded = {
        name: bd.blind_sacc(p, "any-seed", config=zero, log=_NOLOG)
        for name, p in parts.items()
    }
    for name, part in parts.items():
        assert np.array_equal(np.array(part.mean), np.array(blinded[name].mean)), (
            f"zero-shift blind changed {name}"
        )
    # derived statistics downstream: identical inputs ⇒ identical numbers
    En_t, Bn_t, modes_t = _derive_downstream(
        parts["xi_reporting"], parts["xi_integration"]
    )
    En_b, Bn_b, modes_b = _derive_downstream(
        blinded["xi_reporting"], blinded["xi_integration"]
    )
    assert np.array_equal(En_t, En_b) and np.array_equal(Bn_t, Bn_b)
    for key in modes_t:
        t, b = modes_t[key], modes_b[key]
        both_nan = np.isnan(t) & np.isnan(b)
        assert np.array_equal(t[~both_nan], b[~both_nan]), key
        assert np.array_equal(np.isnan(t), np.isnan(b)), key


@pytest.mark.slow
def test_ac4_b_mode_invariance_and_leakage_floor():
    """AC4: the ΔBₙ induced by deriving B-modes from the blinded integration ξ± is
    independent of the injected B amplitude (a fixed absolute E→B leakage
    offset, not fractional). The magnitude is measured and reported, never
    asserted against a constant."""
    pytest.importorskip("cosmo_numba")
    seed = "ac4-seed"
    deltas, reports = [], []
    for amp in (2e-6, 2e-5):
        reporting = make_xi_part("reporting")
        integration = make_xi_part("integration", b_amplitude=amp)
        blinded_reporting = bd.blind_sacc(reporting, seed, log=_NOLOG)
        blinded_integration = bd.blind_sacc(integration, seed, log=_NOLOG)
        _, Bn_t, modes_t = _derive_downstream(reporting, integration)
        _, Bn_b, modes_b = _derive_downstream(blinded_reporting, blinded_integration)
        d_bn = Bn_b - Bn_t
        d_xib = modes_b["xip_B"] - modes_t["xip_B"]
        finite = np.isfinite(d_xib)
        deltas.append((d_bn, d_xib[finite]))
        reports.append(
            f"B={amp:.0e}: max|ΔBₙ|={np.max(np.abs(d_bn)):.3e} "
            f"(ΔBₙ/Bₙ={np.max(np.abs(d_bn)) / np.max(np.abs(Bn_t)):.2e}), "
            f"max|Δξ+_B|={np.max(np.abs(d_xib[finite])):.3e} "
            f"({np.max(np.abs(d_xib[finite])) / amp:.2%} of injected B)"
        )
    print("\nAC4 " + "\nAC4 ".join(reports))

    (d_bn_1, d_xib_1), (d_bn_2, d_xib_2) = deltas
    scale = max(np.max(np.abs(d_bn_1)), 1e-30)
    gap = np.max(np.abs(d_bn_1 - d_bn_2))
    print(
        f"AC4 ΔBₙ amplitude-independence: max|ΔBₙ(2e-6) − ΔBₙ(2e-5)| = "
        f"{gap:.3e} ({gap / scale:.2e} of |ΔBₙ|)"
    )
    assert gap <= 1e-9 * scale + 1e-24, (
        "ΔBₙ depends on the injected B amplitude — the shift is not pure E"
    )
    # The pure-ξ_B leakage is also amplitude-independent, but only to the
    # adaptive-quadrature floor: cosmo_numba's Schneider integrals subdivide
    # adaptively, so the estimator is not bit-linear in its inputs and the
    # two runs differ at a small fraction of the (tiny) leakage itself. The
    # COSEBIs assertion above carries the exact-identity criterion; this one
    # bounds the quadrature wobble.
    scale_x = max(np.max(np.abs(d_xib_1)), 1e-30)
    gap_x = np.max(np.abs(d_xib_1 - d_xib_2))
    print(
        f"AC4 Δξ+_B amplitude-independence: {gap_x:.3e} "
        f"({gap_x / scale_x:.2e} of the leakage)"
    )
    assert gap_x <= 0.05 * scale_x


@pytest.mark.slow
def test_ac5_untouched_blocks_and_row_order():
    """AC5: each part's covariance byte-identical; the Cℓ BB/EB rows and the
    ρ/τ part never blinded; every part's shifted rows land at their original
    within-part indices (order-preservation)."""
    parts = make_parts(nbins=1)
    seed = "ac5-seed"
    blinded = {
        name: bd.blind_sacc(parts[name], seed, log=_NOLOG)
        for name in ("xi_reporting", "xi_integration", "cl")
    }
    for name, b in blinded.items():
        part = parts[name]
        assert np.array_equal(b.covariance.dense, part.covariance.dense), name
        # row order: the identity of every row (type/tracers/tags) unchanged
        for a, c in zip(part.data, b.data):
            assert (a.data_type, a.tracers, a.tags) == (c.data_type, c.tracers, c.tags)
    # BB/EB rows of the Cℓ part untouched (pure E-mode shift)
    for dt in (sio.CL_BB, sio.CL_EB):
        idx = parts["cl"].indices(dt)
        assert np.array_equal(
            np.array(blinded["cl"].mean)[idx], np.array(parts["cl"].mean)[idx]
        ), f"{dt} was touched by the blind"
    # the blindable rows did move (the blind actually blinded)
    for name, grid in (
        ("xi_reporting", "reporting"),
        ("xi_integration", "integration"),
    ):
        idx = bd._xi_indices(parts[name], grid)
        assert not np.allclose(
            np.array(blinded[name].mean)[idx], np.array(parts[name].mean)[idx], atol=0
        )
    # ρ/τ: refused by blind_sacc (test_blind_refuses_non_blindable_part) and
    # exempt in assembly — pass through gather untouched
    s = sio.gather(
        [
            blinded["xi_reporting"],
            parts["rho_tau"],
            blinded["cl"],
            blinded["xi_integration"],
        ]
    )
    idx = s.indices(sio.RHO_PLUS.format(k=0))
    assert np.array_equal(
        np.array(s.mean)[idx],
        np.array(parts["rho_tau"].mean)[
            parts["rho_tau"].indices(sio.RHO_PLUS.format(k=0))
        ],
    )


@pytest.mark.slow
def test_ac9_pure_eb_nan_parity_under_blind():
    """AC9: the pure-E/B NaN pattern born from the blinded parts is identical
    to the true parts' — blinding never moves a NaN.

    The Schneider estimator returns NaN wherever a reporting point lacks
    interior support against the edge-based integration bounds. Whatever
    that pattern is on the true parts (the fixture puts the outermost
    reporting point at the boundary, so it is non-empty here; on production
    files it is empty), the blinded derivation must reproduce it bit-for-bit:
    the blind is a pure shift of the estimator's inputs, not a change of
    estimator support. The finite values move (the ξ± shifted); the NaN mask
    does not."""
    pytest.importorskip("cosmo_numba")
    seed = "ac9-seed"
    reporting, integration = make_xi_part("reporting"), make_xi_part("integration")
    _, _, modes_t = _derive_downstream(reporting, integration)
    _, _, modes_b = _derive_downstream(
        bd.blind_sacc(reporting, seed, log=_NOLOG),
        bd.blind_sacc(integration, seed, log=_NOLOG),
    )
    for key in modes_t:
        t, b = modes_t[key], modes_b[key]
        assert np.array_equal(np.isnan(t), np.isnan(b)), (
            f"blinding moved the pure-E/B NaN pattern for {key}"
        )
    # the finite values did move (the blind actually shifted the ξ±)
    t, b = modes_t["xip_E"], modes_b["xip_E"]
    finite = np.isfinite(t)
    assert finite.any() and not np.allclose(t[finite], b[finite], atol=0)


# --------------------------------------------------------------------------- #
# AC6 + AC8: end-to-end custody through the file surface (slow + cosmo_numba)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_ac6_ac8_end_to_end_init_parts_gather_unblind(tmp_path):
    """AC8: blind-init → blind-part on each intermediate part → terminal
    gather (hash assertion passes, stamp lands) → unblind restores each part
    bit-for-bit, and the derived statistics re-derived from the unblinded
    integration part reproduce the truth. AC6: no plaintext part or seed survives
    on disk, no ``seed_smokescreen`` key; unblind fails closed on a tampered
    commitment."""
    pytest.importorskip("cosmo_numba")
    parts = make_parts(nbins=1)
    En_true, Bn_true, modes_true = _derive_downstream(
        parts["xi_reporting"], parts["xi_integration"]
    )

    blind_dir = tmp_path / "blind"
    blind_dir.mkdir()
    init = bd.blind_init(str(blind_dir), log=_NOLOG)

    part_files, out_paths = {}, {}
    for name in ("xi_reporting", "xi_integration", "cl"):
        path = tmp_path / f"{name}.fits"
        sio.save(parts[name], str(path), type="mock")
        out_paths[name] = bd.blind_part(str(path), str(blind_dir), log=_NOLOG)
        part_files[name] = path

    # -- custody hygiene (AC6) --------------------------------------------- --
    for name, path in part_files.items():
        assert not path.exists(), f"plaintext part {name} was not deleted"
        blinded = sio.load(out_paths[name]["blinded"])
        assert blinded.metadata["concealed"] is True
        assert "seed_smokescreen" not in blinded.metadata
        assert pathlib.Path(out_paths[name]["escrow"]).exists()
        assert not np.array_equal(np.array(blinded.mean), np.array(parts[name].mean))
    with open(init["commitment"], encoding="utf-8") as f:
        commitment = json.load(f)
    assert set(commitment) == {
        "label",
        "seed_commitment",
        "config_digest",
        "draw_scheme",
    }
    blinded_parts = {n: sio.load(p["blinded"]) for n, p in out_paths.items()}
    for b in blinded_parts.values():
        assert b.metadata["blind_commitment"] == commitment["seed_commitment"]
    # no plaintext json anywhere beside the blind outputs
    assert not list(tmp_path.rglob("*escrow.json"))
    assert not (blind_dir / "blind_seed.json").exists()

    # -- terminal assembly: hash assertion + stamp (AC8) -------------------- --
    assembled = sio.gather(
        [
            blinded_parts["xi_reporting"],
            blinded_parts["cl"],
            parts["rho_tau"],
            blinded_parts["xi_integration"],
        ],
        metadata={"catalogue_version": "vTEST", "type": "mock"},
    )
    assert assembled.metadata["blind_commitment"] == commitment["seed_commitment"]
    # born-blinded derived statistics from the blinded parts differ from truth
    En_b, Bn_b, _ = _derive_downstream(
        blinded_parts["xi_reporting"], blinded_parts["xi_integration"]
    )
    assert not np.allclose(En_b, En_true, atol=0)

    # -- fail-closed on tampered commitment (AC6) --------------------------- --
    tampered = dict(commitment, seed_commitment="0" * 64)
    with open(init["commitment"], "w", encoding="utf-8") as f:
        json.dump(tampered, f)
    with pytest.raises(ValueError, match="committed seed commitment"):
        bd.unblind_part(
            out_paths["xi_integration"]["blinded"],
            str(blind_dir),
            str(tmp_path / "never.fits"),
            log=_NOLOG,
        )
    with open(init["commitment"], "w", encoding="utf-8") as f:
        json.dump(commitment, f)
    with pytest.raises(ValueError, match="config digest"):
        bd.unblind_part(
            out_paths["xi_integration"]["blinded"],
            str(blind_dir),
            str(tmp_path / "never.fits"),
            config=bd.BlindingConfig(s8_half_width=0.01),
            log=_NOLOG,
        )

    # -- bit-for-bit restoration per part (AC8) ----------------------------- --
    restored = {}
    for name in ("xi_reporting", "xi_integration", "cl"):
        out = tmp_path / f"{name}_restored.fits"
        bd.unblind_part(
            out_paths[name]["blinded"], str(blind_dir), str(out), log=_NOLOG
        )
        restored[name] = sio.load(str(out))
        assert np.array_equal(
            np.array(restored[name].mean), np.array(parts[name].mean)
        ), f"{name} not restored bit-for-bit"
        assert not restored[name].metadata.get("concealed", False)
        assert "blind_commitment" not in restored[name].metadata

    # unblinding then re-deriving reproduces the true derived statistics
    En_r, Bn_r, modes_r = _derive_downstream(
        restored["xi_reporting"], restored["xi_integration"]
    )
    assert np.array_equal(En_r, En_true) and np.array_equal(Bn_r, Bn_true)
    for key in modes_true:
        t, r = modes_true[key], modes_r[key]
        both_nan = np.isnan(t) & np.isnan(r)
        assert np.array_equal(t[~both_nan], r[~both_nan]), key
        assert np.array_equal(np.isnan(t), np.isnan(r)), key


def test_ac8_dotted_versioned_part_names_escrow_and_restore(tmp_path):
    """AC8 under the canonical catalogue-version naming (dotted stems).

    Production part files carry the versioned name ``v1.4.6.3_xi_reporting.fits``
    etc. ``smokescreen.encryption.encrypt_file`` names its outputs from
    ``basename.split('.')[0]``, so both these parts would misfile onto
    ``v1.encrpt``/``v1.key`` and the second would silently overwrite the
    first's escrowed truth. Guard: the escrow lands at the exact
    :func:`part_paths` name, two dot-prefix-sharing parts do not collide, and
    each restores bit-for-bit."""
    parts = make_parts(nbins=1)
    blind_dir = tmp_path / "blind"
    blind_dir.mkdir()
    bd.blind_init(str(blind_dir), log=_NOLOG)

    version = "v1.4.6.3"
    out_paths, part_files = {}, {}
    for name in ("xi_reporting", "xi_integration"):
        path = tmp_path / f"{version}_{name}.fits"
        sio.save(parts[name], str(path), type="mock")
        out_paths[name] = bd.blind_part(str(path), str(blind_dir), log=_NOLOG)
        part_files[name] = path

    # escrow bundles landed at the declared names (no split('.') truncation),
    # and the two dot-prefix-sharing parts did not collide onto one bundle.
    escrow_files = {n: p["escrow"] for n, p in out_paths.items()}
    assert escrow_files["xi_reporting"] != escrow_files["xi_integration"]
    for name, path in part_files.items():
        assert not path.exists(), f"plaintext part {name} was not deleted"
        assert pathlib.Path(out_paths[name]["escrow"]).exists(), name
        assert pathlib.Path(out_paths[name]["escrow_key"]).exists(), name
    # the truncated-name collision target must not exist
    assert not (tmp_path / "v1.encrpt").exists()
    assert not (tmp_path / "v1.key").exists()
    assert not list(tmp_path.rglob("*escrow.json"))

    # each part restores bit-for-bit via its own escrow (not subtraction-only)
    for name in ("xi_reporting", "xi_integration"):
        out = tmp_path / f"{version}_{name}_restored.fits"
        bd.unblind_part(
            out_paths[name]["blinded"], str(blind_dir), str(out), log=_NOLOG
        )
        restored = sio.load(str(out))
        assert np.array_equal(np.array(restored.mean), np.array(parts[name].mean)), (
            f"{name} not restored bit-for-bit"
        )
