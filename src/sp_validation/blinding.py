"""Blinding — conceal each intermediate data product behind a hidden cosmology.

:Name: blinding.py

:Description: Smokescreen blinding, per part, at birth. Each blindable
    intermediate SACC product — reporting ξ±, integration ξ±, pseudo-Cℓ — is
    shifted the moment the pipeline computes it by a difference of theory
    vectors between the fiducial and a *hidden* cosmology drawn inside a fixed
    amplitude envelope (Muir et al. 2019: ``d → d + t(hidden) − t(fiducial)``),
    so S8 cannot be read off the data before the collaboration unblinds. Only
    blinded parts persist on disk.

    The ``UNIONS-WL/Smokescreen`` fork draws the hidden cosmology and computes
    the shift; blinding is a vector operation, so this module calls its vector
    core (``smokescreen.concealing_factor``), which never sees a SACC. What is
    sp_validation-specific is supplied here: the ``theory_fn`` backends
    matching each part's row layout, and the SACC handling around the factors.

    Derived statistics are born blinded — COSEBIs and pure-E/B run downstream
    on the already-blinded integration ξ±. Covariance and ρ/τ are never blinded:
    blinding hides the vector, not the uncertainty, and the shift is pure
    E-mode so B-mode null tests stay honest under the blind.

    **Custody: hash commitment, no keyholder.** A blind is reproduced by a
    *triple* — seed, config digest, and Smokescreen ``DRAW_SCHEME`` (the seed
    alone is not enough; see :func:`draw_scheme`). :func:`blind_init` fixes all
    three per catalogue version, publishing the commitment triple as a
    repo-committable ``commitment.json`` and encrypting the seed into a Fernet
    bundle — the plaintext seed is never written. Every gate that draws or
    subtracts a shift fails closed unless the triple matches
    (:func:`_assert_draw_scheme`, :func:`assert_consistent_blind`,
    :func:`unblind_sacc`), and there is no override.
"""

import dataclasses
import functools
import hashlib
import json
import os
import secrets
import warnings

import numpy as np

from . import sacc_io
from .blinding_paths import init_paths, part_paths  # noqa: F401  (re-exported)
from .blinding_theory import TheoryConfig, cl_ee, coerce_fields, xi_ccl, xi_ell_grid


# --------------------------------------------------------------------------- #
# Configuration surface — the blinding envelope
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class BlindingConfig:
    """Blinding envelope and fiducial.

    The hidden cosmology is drawn (by the fork) uniformly and independently
    per key inside ``shifts_dict()``'s half-widths about the fiducial. The
    half-widths are the deliberate, configurable size of the blind — config,
    not code; the group may resize the envelope. ``theory`` carries the
    fiducial :class:`TheoryConfig` whose defaults *are* the blinding
    fiducial.
    """

    s8_half_width: float = 0.075
    omega_m_half_width: float = 0.1
    theory: TheoryConfig = dataclasses.field(default_factory=TheoryConfig)

    def shifts_dict(self):
        """The (S8, Ωm) envelope as CCL-native ``{sigma8, Omega_c}`` half-widths.

        Evaluated at the fiducial: a ΔS8 half-width maps to
        ``ΔS8/√(Ωm_fid/0.3)`` in σ8 (at fixed Ωm), and a ΔΩm half-width maps
        one-to-one to Ω_c (Ω_b and Ω_ν fixed). Exact enough for a smear whose
        target is a characteristic amplitude, not a precise posterior.
        """
        return {
            "sigma8": self.s8_half_width / np.sqrt(self.theory.Omega_m / 0.3),
            "Omega_c": self.omega_m_half_width,
        }

    def config_digest(self):
        """sha256 of a canonical serialization of the full blinding config.

        Binds the envelope half-widths and the complete fiducial
        :class:`TheoryConfig` into one digest, as JSON with sorted keys.
        Fields go through :func:`~sp_validation.blinding_theory.coerce_fields`,
        so the digest depends on the numeric value rather than on an
        int-vs-float literal, and ``json`` emits floats by shortest round-trip
        ``repr`` — two runs of one config give byte-identical digests. Checked
        with the seed commitment at unblind.
        """
        payload = coerce_fields(
            type(self),
            {
                "s8_half_width": self.s8_half_width,
                "omega_m_half_width": self.omega_m_half_width,
            },
        )
        payload["theory"] = coerce_fields(
            TheoryConfig,
            {
                f.name: getattr(self.theory, f.name)
                for f in dataclasses.fields(self.theory)
            },
        )
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    @classmethod
    def from_overrides(cls, overrides):
        """Build from a mapping of field overrides (fail loud on unknown keys).

        ``theory`` may be given as a :class:`TheoryConfig` or a mapping of
        TheoryConfig overrides, mirroring :meth:`TheoryConfig.from_overrides`.
        """
        overrides = coerce_fields(cls, overrides)
        theory = overrides.get("theory")
        if theory is not None and not isinstance(theory, TheoryConfig):
            overrides["theory"] = TheoryConfig.from_overrides(dict(theory))
        return cls(**overrides)


# --------------------------------------------------------------------------- #
# Custody primitives
# --------------------------------------------------------------------------- #
def seed_commitment(seed):
    """Public commitment for a seed: the fork's domain-separated sha256 digest.

    Domain-separated because the fork derives its RNG base seed from the bare
    sha256 of the same string — an undomained commitment would publish it.
    """
    from smokescreen import seed_commitment as _fork_seed_commitment

    return _fork_seed_commitment(seed)


def draw_scheme():
    """The installed Smokescreen fork's shift-draw semantics version.

    ``smokescreen.DRAW_SCHEME`` versions *how* a seed becomes parameter deltas
    (scheme 1: upstream DESC's single global RNG over the sorted keys; scheme
    2: this fork's per-key RNG from ``(seed, key)``). Two installs agreeing on
    ``(seed, config)`` but not on this number draw different hidden cosmologies.

    This is the one blinding failure with no loud symptom — a scheme mismatch
    leaves a smooth residual cosmological shift in the "unblinded" vector, and
    every hash, digest and escrow check still passes. Hence the scheme is
    custody state, checked wherever a shift is drawn or subtracted.
    """
    from smokescreen import DRAW_SCHEME

    return int(DRAW_SCHEME)


def _assert_draw_scheme(recorded, what):
    """Fail closed unless ``recorded`` is the installed fork's draw scheme.

    ``what`` names the surface the scheme was read from, for the message.
    A missing record (``None``) is a failure, not a pass: a blind whose scheme
    is unknown cannot be shown to be reproducible by this install (see
    :func:`draw_scheme`).
    """
    installed = draw_scheme()
    if recorded is None:
        raise ValueError(
            f"{what} carries no draw-scheme record — refusing to proceed. "
            f"It predates draw-scheme binding, so there is no way to tell "
            f"whether the installed Smokescreen (DRAW_SCHEME={installed}) "
            f"reproduces the shift it was blinded with."
        )
    if int(recorded) != installed:
        raise ValueError(
            f"{what} was drawn under Smokescreen DRAW_SCHEME={int(recorded)} "
            f"but the installed fork implements DRAW_SCHEME={installed} — "
            f"refusing to proceed. The same seed draws a different hidden "
            f"cosmology under a different scheme, so this install would "
            f"subtract the wrong shift and pass every other check. Install "
            f"the Smokescreen the blind was made with."
        )


def hidden_params(seed, config):
    """The hidden CCL parameter point the fork realizes for ``(seed, config)``.

    Re-runs the fork's draw and overlays the deltas on the fiducial, as
    ``concealing_factor`` does internally. Introspection only (what *was* the
    hidden cosmology, once revealed) — the blinding path never calls it, so it
    does not gate on the draw scheme; every gate that acts on a shift does.
    """
    from smokescreen.param_shifts import draw_param_shifts

    deltas = draw_param_shifts(config.shifts_dict(), seed)
    params = dict(config.theory.ccl_params())
    for key, delta in deltas.items():
        params[key] += delta
    return params


# --------------------------------------------------------------------------- #
# Block discovery on a SACC (a standalone part, or the assembled file)
# --------------------------------------------------------------------------- #
def source_bins(s):
    """Sorted source-bin indices present in ``s`` (from ``source_i`` tracers)."""
    return sorted(
        int(name.split("_", 1)[1]) for name in s.tracers if name.startswith("source_")
    )


def _pairs(s, data_type, **tags):
    """Unordered source-bin pairs ``(i ≤ j)`` carrying ``data_type`` in ``s``."""
    bins = source_bins(s)
    return [
        (i, j)
        for a, i in enumerate(bins)
        for j in bins[a:]
        if len(s.indices(data_type, sacc_io._pair((i, j)), **tags))
    ]


def xi_pairs(s, grid):
    """Unordered source-bin pairs ``(i ≤ j)`` carrying ξ+ on ``grid``."""
    return _pairs(s, sacc_io.XI_PLUS, grid=grid)


def cl_pairs(s):
    """Source-bin pairs ``(i ≤ j)`` carrying pseudo-Cℓ_EE."""
    return _pairs(s, sacc_io.CL_EE)


def _xi_indices(s, grid):
    """Row indices of the ξ± block on ``grid`` (ascending)."""
    return np.sort(
        np.concatenate(
            [
                s.indices(sacc_io.XI_PLUS, grid=grid),
                s.indices(sacc_io.XI_MINUS, grid=grid),
            ]
        )
    ).astype(int)


def _cl_ee_indices(s):
    """Row indices of the pseudo-Cℓ_EE block (ascending).

    Only EE: a pure E-mode cosmology shift leaves BB and EB identically
    zero, so those blocks are never extracted, never concealed.
    """
    return np.sort(s.indices(sacc_io.CL_EE)).astype(int)


def _pair_nz(s, i, j):
    """The two per-bin n(z) for pair ``(i, j)`` as ``((z_i, n_i), (z_j, n_j))``."""
    z_i, n_i = sacc_io.get_nz(s, i)
    z_j, n_j = sacc_io.get_nz(s, j)
    return (np.asarray(z_i), np.asarray(n_i)), (np.asarray(z_j), np.asarray(n_j))


# --------------------------------------------------------------------------- #
# The three theory backends — callables aligned to a sub-SACC block's rows
# --------------------------------------------------------------------------- #
def xi_theory_fn(block, theory, grid):
    """``theory_fn`` for a ξ± sub-SACC block (reporting or integration grid).

    Reads each bin's n(z) from the block's own tracers and lays the output out
    to match the block's SACC rows element-for-element: per pair, ξ± at that
    pair's stored θ (the ``theta`` tag, arcmin), scattered to the rows
    ``block.indices`` reports — never an assumed pairing.
    """
    pairs = xi_pairs(block, grid)
    layout = []
    for i, j in pairs:
        tr = sacc_io._pair((i, j))
        idx_p = block.indices(sacc_io.XI_PLUS, tr, grid=grid)
        idx_m = block.indices(sacc_io.XI_MINUS, tr, grid=grid)
        theta = sacc_io._tag(block, sacc_io.XI_PLUS, tr, "theta", grid=grid)
        layout.append(((i, j), idx_p, idx_m, np.asarray(theta, dtype=float)))
    ell = xi_ell_grid()

    def theory_fn(params):
        out = np.full(len(block.mean), np.nan)
        for (i, j), idx_p, idx_m, theta in layout:
            nz_i, nz_j = _pair_nz(block, i, j)
            xip, xim = xi_ccl(params, theory, nz_i, nz_j, theta, ell)
            out[idx_p] = xip
            out[idx_m] = xim
        return out

    return theory_fn


def cl_theory_fn(block, theory):
    """``theory_fn`` for the pseudo-Cℓ_EE sub-SACC block.

    Per pair: theory Cℓ_EE on the stored ``BandpowerWindows`` support, binned
    by the same window matrix the measurement used (``W @ Cℓ_EE``), scattered
    to the block's own rows — so the shift lands in the measured bandpowers.
    ΔBB = ΔEB ≡ 0 for a pure E-mode shift, hence EE-only rows.
    """
    layout = []
    for i, j in cl_pairs(block):
        tr = sacc_io._pair((i, j))
        idx = block.indices(sacc_io.CL_EE, tr)
        window = block.get_bandpower_windows(idx)
        layout.append(
            (
                (i, j),
                np.asarray(idx),
                np.asarray(window.values, dtype=float),  # (n_ell,)
                np.asarray(window.weight, dtype=float),  # (n_ell, n_bp)
            )
        )

    def theory_fn(params):
        out = np.full(len(block.mean), np.nan)
        for (i, j), idx, w_ell, w_mat in layout:
            nz_i, nz_j = _pair_nz(block, i, j)
            out[idx] = w_mat.T @ cl_ee(params, theory, nz_i, nz_j, w_ell)
        return out

    return theory_fn


# --------------------------------------------------------------------------- #
# The concealing factor, per block
# --------------------------------------------------------------------------- #
def _blindable_blocks(s):
    """The blindable blocks of a SACC as ``(name, indices, factory)``.

    Works identically on a standalone part (exactly one block) and on the
    assembled file (integration rows selected by the ``grid`` tag). ``indices``
    are each block's recorded row indices (ascending); ``factory`` builds the
    matching ``theory_fn``. Blocks absent from the file are not listed.
    """
    blocks = []
    for grid in ("reporting", "integration"):
        idx = _xi_indices(s, grid)
        if len(idx):
            blocks.append(
                (f"{grid} ξ±", idx, functools.partial(xi_theory_fn, grid=grid))
            )
    idx = _cl_ee_indices(s)
    if len(idx):
        blocks.append(("pseudo-Cℓ_EE", idx, cl_theory_fn))
    return blocks


def _concealing_factor(s, indices, factory, config, seed):
    """The fork-computed additive concealing factor for one block of ``s``.

    ``smokescreen.concealing_factor`` draws the hidden deltas from ``seed``,
    evaluates the block's ``theory_fn`` at both cosmologies and differences
    them. No data vector and no SACC reach the fork, and ``s`` is not modified.
    Both :func:`blind_sacc` and :func:`unblind_sacc` come through here, so the
    added and subtracted shifts cannot drift apart.

    A ``theory_fn`` fills only its own rows and leaves the rest NaN, so
    slicing to ``indices`` drops the NaNs by construction and the finite check
    proves the converse: that *all* of this block's rows were filled. A row
    the block claims but the factory cannot cover (a pair with ξ− but no ξ+,
    say) would otherwise be shifted by NaN, silently.

    Returns
    -------
    np.ndarray
        ``t(hidden) − t(fiducial)``, aligned to ``indices``.
    """
    from smokescreen import concealing_factor

    full = np.asarray(
        concealing_factor(
            config.theory.ccl_params(),
            config.shifts_dict(),
            seed=seed,
            theory_fn=factory(s, config.theory),
            factor_type="add",
        ),
        dtype=float,
    )
    factor = full[indices]
    if not np.all(np.isfinite(factor)):
        raise ValueError(
            f"the theory backend left {int(np.sum(~np.isfinite(factor)))} of "
            f"{len(indices)} blindable rows unfilled — refusing to apply the "
            "concealing factor (these rows would be shifted by NaN). The "
            "block's row layout is not fully covered by its theory_fn."
        )
    return factor


def _set_values(s, indices, values):
    """Overwrite ``s.data[i].value`` for ``indices`` with ``values`` (aligned)."""
    for i, v in zip(indices, values):
        s.data[int(i)].value = float(v)


def _concealed(s):
    """Whether ``s`` is already a blinded file (its ``concealed`` mark is set)."""
    return bool(s.metadata.get("concealed"))


def _apply_blocks(src, seed, config, sign, verb, log):
    """Return a copy of ``src`` with each block's concealing factor applied.

    ``sign`` is ``+1`` to conceal and ``-1`` to reveal; the factor itself comes
    from the one :func:`_concealing_factor` call both directions share.
    """
    dst = src.copy()
    for name, indices, factory in _blindable_blocks(src):
        factor = _concealing_factor(src, indices, factory, config, seed)
        _set_values(dst, indices, np.asarray(dst.mean)[indices] + sign * factor)
        log(f"[{verb}] {name}: {len(indices)} points")
    return dst


def blind_sacc(part, seed, config=None, label="smokescreen", log=print):
    """Return a blinded copy of a part SACC (covariance and tags untouched).

    Adds each blindable block's concealing factor at the block's recorded
    indices; only ``value`` changes, and only on blindable rows. Provenance is
    stamped and any leaked seed key stripped. A file with no blindable block
    (a ρ/τ diagnostic part, say) is refused loudly.
    """
    config = config or BlindingConfig()
    if _concealed(part):
        raise ValueError("already concealed — unblind first")
    if not _blindable_blocks(part):
        raise ValueError(
            "no blindable block (reporting/integration ξ± or pseudo-Cℓ_EE) in this SACC "
            "— ρ/τ diagnostic parts are never blinded"
        )

    blinded = _apply_blocks(part, seed, config, +1, "blind", log)
    _stamp_provenance(blinded, seed_commitment(seed), label, config.config_digest())
    return blinded


def unblind_sacc(blinded, seed, config=None, log=print):
    """Recover the true part SACC from a blinded one + the revealed ``seed``.

    Verifies the custody triple — draw scheme, seed commitment, config digest —
    against the file's stamps before subtracting anything, then recomputes each
    block's shift the same way :func:`blind_sacc` added it. Works on a part or
    on the assembled file. Derived statistics, if present, are *not* recomputed
    here — the pipeline re-derives them from the unblinded integration ξ±.
    """
    config = config or BlindingConfig()
    if not _concealed(blinded):
        raise ValueError("file is not concealed — nothing to unblind")
    _assert_draw_scheme(blinded.metadata.get("blind_draw_scheme"), "this blinded file")
    if seed_commitment(seed) != blinded.metadata["blind_commitment"]:
        raise ValueError(
            "seed does not match blind_commitment — refusing to unblind "
            "(a wrong seed would silently produce a wrong data vector)"
        )
    if config.config_digest() != blinded.metadata["blind_config_digest"]:
        raise ValueError(
            "blinding config does not match blind_config_digest — refusing to "
            "unblind (this config would subtract a different shift than was "
            "added)"
        )

    part = _apply_blocks(blinded, seed, config, -1, "unblind", log)
    for key in (
        "concealed",
        "blind",
        "blind_commitment",
        "blind_config_digest",
        "blind_draw_scheme",
    ):
        part.metadata.pop(key, None)
    return part


def _stamp_provenance(s, commitment, label, config_digest):
    """Stamp the custody triple and the ``concealed``/``blind`` marks.

    Pops ``seed_smokescreen`` (which upstream Smokescreen's writer would
    stamp): the seed never rides a kept file. The scheme stamped is the
    installed fork's, which every caller has already checked the blind against.
    """
    s.metadata.pop("seed_smokescreen", None)
    s.metadata["concealed"] = True
    s.metadata["blind"] = label
    s.metadata["blind_commitment"] = commitment
    s.metadata["blind_config_digest"] = config_digest
    s.metadata["blind_draw_scheme"] = draw_scheme()


def stamp_concealed_passthrough(s, commitment_path):
    """Stamp a part concealed under an existing blind, values untouched.

    The seam for parts already blind (COSEBIs / pure-E/B, re-derived from the
    blinded integration ξ±) or blind-irrelevant (ρ/τ, no cosmological vector):
    it shifts nothing and needs no blindable block, only the custody stamp that
    lets the load gate and :func:`assert_consistent_blind` admit the part. The
    stamp is read from the version's ``commitment.json``, so a pass-through
    part carries the exact custody state of the blinded parts. The committed
    scheme is checked first — stamping from an install that draws differently
    would mint a custody claim it cannot honour.
    """
    with open(commitment_path, encoding="utf-8") as f:
        commitment = json.load(f)
    _assert_draw_scheme(
        commitment.get("draw_scheme"), f"the blind at {commitment_path}"
    )
    _stamp_provenance(
        s,
        commitment["seed_commitment"],
        commitment["label"],
        commitment["config_digest"],
    )
    return s


# --------------------------------------------------------------------------- #
# Assembly-time custody: one blind across all parts
# --------------------------------------------------------------------------- #
def assert_consistent_blind(parts):
    """Assert every blindable part shares one blind; return the shared stamp.

    The assembly gate of :func:`sp_validation.sacc_io.gather`. A part is
    *blindable* if it carries a blindable block (ξ± or pseudo-Cℓ_EE); ρ/τ and
    covariance-only parts are exempt. Fails closed on a blinded/plaintext mix,
    on parts whose custody triples disagree, or on a shared scheme this install
    does not implement (it could not unblind what it is assembling). The
    consistency key is the triple; the ``blind`` label is provenance, not
    custody, so differing labels warn rather than fail.

    Unconcealed blindable parts must all be declared ``type == "mock"``: an
    unconcealed ``type == "data"`` part, or one missing the tag, fails closed,
    so skipping the blind can never silently expose real data.

    Returns
    -------
    dict or None
        The shared blind metadata (``concealed``, ``blind``,
        ``blind_commitment``, ``blind_config_digest``, ``blind_draw_scheme``)
        for the gather to stamp on the assembled file, or ``None`` when
        nothing is blinded.
    """
    blindable = [p for p in parts if _blindable_blocks(p)]
    concealed = [p for p in blindable if _concealed(p)]
    if not concealed:
        # `.get` is deliberate here: a missing `type` tag must count as
        # not-a-mock and fail closed, not KeyError with less context.
        exposed = sorted(
            {str(p.metadata.get("type", "<missing>")) for p in blindable} - {"mock"}
        )
        if exposed:
            raise ValueError(
                f"unconcealed blindable parts with type {exposed} in assembly "
                "— only parts declared `type: mock` may assemble without a "
                "blind (an unconcealed data part exposes the real vector)"
            )
        return None
    if len(concealed) != len(blindable):
        raise ValueError(
            f"blinded and plaintext blindable parts mixed in one assembly "
            f"({len(concealed)} of {len(blindable)} blinded) — refusing to "
            "combine (a plaintext part beside blinded ones leaks the shift)"
        )
    # `.get` on the scheme so a part predating scheme binding reads as None and
    # fails at _assert_draw_scheme with its explanation, not with a KeyError.
    stamps = {
        (
            p.metadata["blind_commitment"],
            p.metadata["blind_config_digest"],
            p.metadata.get("blind_draw_scheme"),
        )
        for p in concealed
    }
    if len(stamps) != 1:
        raise ValueError(
            "parts carry different blind commitments — they were blinded "
            "under different seeds, configs or draw schemes and must never be "
            "combined: "
            + "; ".join(f"({c[:12]}…, {d[:12]}…, scheme {v})" for c, d, v in stamps)
        )
    ((commitment, digest, scheme),) = stamps
    _assert_draw_scheme(scheme, "the blind these parts share")
    labels = sorted({p.metadata["blind"] for p in concealed})
    if len(labels) != 1:
        warnings.warn(
            f"blindable parts share one blind (commitment {commitment[:12]}…, "
            f"config {digest[:12]}…) but carry different labels {labels} — "
            "assembling anyway; the label is provenance, not custody state. "
            f"Stamping the assembled file with label {labels[0]!r}."
        )
    return {
        "concealed": True,
        "blind": labels[0],
        "blind_commitment": commitment,
        "blind_config_digest": digest,
        "blind_draw_scheme": int(scheme),
    }


# --------------------------------------------------------------------------- #
# File-level custody: blind-init / blind-part / unblind
# --------------------------------------------------------------------------- #
def verify(s, commitment):
    """Problems found comparing a blinded SACC to a commitment, seedlessly.

    Returns a (possibly empty) list of human-readable strings. No seed is read,
    so this cannot confirm the blind is *subtractable* — only that the file's
    custody triple matches ``commitment`` (a parsed ``commitment.json``) and
    that the recorded draw scheme is the one this install implements. That last
    check is environment-dependent by design: a machine carrying a different
    Smokescreen could not unblind the file, so it reports a problem.
    """
    problems = []
    if not _concealed(s):
        problems.append("file is not marked concealed")
    if s.metadata.get("blind_commitment") != commitment["seed_commitment"]:
        problems.append("blind_commitment does not match the committed seed commitment")
    if s.metadata.get("blind_config_digest") != commitment["config_digest"]:
        problems.append("blind_config_digest does not match the committed digest")
    scheme = s.metadata.get("blind_draw_scheme")
    if scheme != commitment.get("draw_scheme"):
        problems.append(
            f"blind_draw_scheme {scheme!r} does not match the committed "
            f"draw_scheme {commitment.get('draw_scheme')!r}"
        )
    try:
        _assert_draw_scheme(scheme, "the blinded file")
    except ValueError as exc:
        problems.append(str(exc))
    if "seed_smokescreen" in s.metadata:
        problems.append("PLAINTEXT SEED LEAKED into file metadata (seed_smokescreen)")
    return problems


def blind_init(blind_dir, config=None, label="smokescreen", log=print):
    """Fix the blind for one catalogue version: seed, commitment, seed bundle.

    Draws an OS-entropy seed (never written in plaintext, never returned),
    writes the repo-committable ``commitment.json`` (the custody triple plus
    the label), and encrypts the seed into a Fernet bundle. Every
    :func:`blind_part` and :func:`unblind_part` call reads this fixed state.

    Custody caveat: the bundle and its Fernet key land in the *same*
    ``blind_dir``, and anyone with both can decrypt the seed. Keep the key
    out-of-band; colocation is convenience, not at-rest protection.

    Returns
    -------
    dict
        Paths written: ``commitment``, ``bundle``, ``key``.
    """
    config = config or BlindingConfig()
    paths = init_paths(blind_dir)
    for path in paths.values():
        if os.path.exists(path):
            raise FileExistsError(
                f"refusing to overwrite existing blind state {path} — a blind "
                "is a one-shot custody event; choose another directory"
            )

    seed = secrets.token_hex(16)
    commitment = {
        "label": label,
        "seed_commitment": seed_commitment(seed),
        "config_digest": config.config_digest(),
        "draw_scheme": draw_scheme(),
    }
    with open(paths["commitment"], "w", encoding="utf-8") as f:
        json.dump(commitment, f, indent=2, sort_keys=True)
    _write_encrypted_json(paths["bundle"], {"label": label, "seed": seed})

    log(f"[blind-init] commitment (repo-committable): {paths['commitment']}")
    log(f"[blind-init] encrypted seed bundle + key: {paths['bundle']}, {paths['key']}")
    log(
        "[blind-init] custody: keep the bundle key out-of-band from the bundle "
        "(colocation in the blind dir is not at-rest protection)"
    )
    return paths


def _read_seed(blind_dir, config):
    """Decrypt the seed bundle and verify it against the commitment.

    The whole custody triple is checked before the seed is handed to any
    caller, whether it is about to blind or to unblind.

    Returns
    -------
    tuple
        ``(seed, commitment_dict)``.
    """
    paths = init_paths(blind_dir)
    bundle = _read_encrypted_json(paths["bundle"], paths["key"])
    with open(paths["commitment"], encoding="utf-8") as f:
        commitment = json.load(f)
    _assert_draw_scheme(commitment.get("draw_scheme"), f"the blind in {blind_dir}")
    if seed_commitment(bundle["seed"]) != commitment["seed_commitment"]:
        raise ValueError(
            "bundle seed does not match the committed seed commitment — refusing "
            "to proceed"
        )
    if config.config_digest() != commitment["config_digest"]:
        raise ValueError(
            "blinding config does not match the committed config digest — "
            "refusing to proceed (a wrong envelope or P(k) recipe would "
            "silently produce a wrong shift)"
        )
    return bundle["seed"], commitment


def blind_part(part_path, blind_dir, config=None, keep_input=False, log=print):
    """Blind one intermediate part SACC at birth, under the fixed blind state.

    Reads the fixed state :func:`blind_init` wrote, conceals the part
    (:func:`blind_sacc`), writes the blinded part beside the input with the
    true vector escrowed into a per-part Fernet bundle, and deletes the
    plaintext — only the blinded part persists. Each escrow is self-contained,
    so corruption of one bundle loses one part, not all. ``keep_input=True``
    retains the plaintext part (and its custody implication).

    Returns
    -------
    dict
        Paths written: ``blinded``, ``escrow``, ``escrow_key``.
    """
    config = config or BlindingConfig()
    seed, commitment = _read_seed(blind_dir, config)
    paths = part_paths(part_path)
    for path in paths.values():
        if os.path.exists(path):
            raise FileExistsError(
                f"refusing to overwrite existing blind output {path} — a blind "
                "is a one-shot custody event"
            )

    # The plaintext part is real, unblinded data — the blinding step is the one
    # legitimate reader of the true vector, so it passes the load escape hatch.
    part = sacc_io.load(part_path, allow_unblinded=True)
    blinded = blind_sacc(part, seed, config=config, label=commitment["label"], log=log)

    _write_encrypted_json(
        paths["escrow"],
        {
            "label": commitment["label"],
            "seed_commitment": commitment["seed_commitment"],
            "true_mean": np.asarray(part.mean, dtype=float).tolist(),
        },
    )
    # The blinded file inherits the part's provenance (data vs mock); it also
    # carries concealed=True (stamped by blind_sacc), so it loads without the
    # escape hatch.
    sacc_io.save(blinded, paths["blinded"], type=part.metadata["type"])
    if not keep_input:
        os.remove(part_path)
        log(f"[blind-part] deleted plaintext part {part_path}")
    else:
        log(f"[blind-part] plaintext part RETAINED at {part_path} (keep_input=True)")
    log(f"[blind-part] wrote {paths['blinded']} (escrow beside it)")
    return paths


def unblind_part(blinded_path, blind_dir, out_path, config=None, log=print):
    """Unblind one blinded part (or the assembled file), verifying first.

    Verifies the custody triple against ``commitment.json`` and against the
    file's own stamps, then subtracts the seed-recomputed shift
    (:func:`unblind_sacc`). The seed-subtracted vector is the authority.

    A part's escrow bundle, when present beside the blinded file and only once
    its stored ``seed_commitment`` is confirmed to be this blind's, plays two
    subordinate roles: a tighter equality check (disagreement beyond ``1e-6``
    relative fails closed) and removal of the ~ulp residue float
    add-then-subtract leaves, making the restore bit-for-bit. It is never the
    source of correctness. The assembled file has no escrow and the
    subtraction stands alone.
    """
    config = config or BlindingConfig()
    seed, commitment = _read_seed(blind_dir, config)
    blinded = sacc_io.load(blinded_path)
    part = unblind_sacc(blinded, seed, config=config, log=log)

    stem, ext = os.path.splitext(blinded_path)
    unblinded_stem = os.path.join(
        os.path.dirname(stem), os.path.basename(stem).replace("_blinded", "")
    )
    escrow = part_paths(unblinded_stem + ext)
    if os.path.exists(escrow["escrow"]):
        bundle = _read_encrypted_json(escrow["escrow"], escrow["escrow_key"])
        if bundle.get("seed_commitment") != commitment["seed_commitment"]:
            raise ValueError(
                "escrow bundle beside the blinded file was written under a "
                "different seed than the commitment — refusing to trust it "
                "(the seed subtraction is authoritative; this escrow is not "
                "bound to this blind)"
            )
        true_mean = np.asarray(bundle["true_mean"], dtype=float)
        recovered = np.asarray(part.mean, dtype=float)
        residual = np.nanmax(
            np.abs(recovered - true_mean) / (np.abs(true_mean) + 1e-30)
        )
        if residual > 1e-6:
            raise ValueError(
                f"unblinded vector disagrees with the escrowed true vector "
                f"(max rel {residual:.2e}) — wrong escrow for this part?"
            )
        # Seed-bound escrow: clear the add-then-subtract ulp residue.
        _set_values(part, range(len(true_mean)), true_mean)
        log(f"[unblind] escrow verified (subtraction residual {residual:.2e})")
    # unblind_sacc stripped the concealed/blind stamps, so this is the true
    # revealed vector; it inherits the blinded file's provenance (data vs mock).
    sacc_io.save(part, out_path, type=blinded.metadata["type"])
    log(f"[unblind] wrote {out_path}")
    return out_path


def _write_encrypted_json(encrpt_path, payload):
    """Encrypt ``payload`` (JSON) to ``encrpt_path`` + sibling ``.key``.

    Smokescreen's ``save_file`` mode names outputs from
    ``basename.split('.')[0]``, which truncates our dotted catalogue-version
    stems (``v1.4.6.3_…`` → ``v1.encrpt``) and collides across parts, so we
    take the returned ``(ciphertext, key)`` and write them ourselves.
    """
    from smokescreen.encryption import encrypt_file

    key_path = encrpt_path.replace(".encrpt", ".key")
    plaintext = encrpt_path.replace(".encrpt", ".json")
    with open(plaintext, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    ciphertext, key = encrypt_file(plaintext, save_file=False, keep_original=False)
    with open(encrpt_path, "wb") as f:
        f.write(ciphertext)
    with open(key_path, "wb") as f:
        f.write(key)


def _read_encrypted_json(encrpt_path, key_path):
    """Decrypt and parse a Fernet-encrypted JSON bundle."""
    from smokescreen.encryption import decrypt_file

    return json.loads(decrypt_file(encrpt_path, key_path).decode("utf-8"))
