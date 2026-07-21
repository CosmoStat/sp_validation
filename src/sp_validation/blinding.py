"""Blinding — conceal each intermediate data product behind a hidden cosmology.

:Name: blinding.py

:Description: Smokescreen blinding wiring, per part, at birth. Each blindable
    intermediate SACC product — reporting ξ±, integration ξ±, pseudo-Cℓ — is shifted,
    the moment the pipeline computes it, by a difference of theory vectors
    between the fiducial cosmology and a *hidden* cosmology drawn inside a
    fixed amplitude envelope, so no one can read S8 off the data until the
    collaboration agrees to unblind (Muir et al. 2019:
    ``d → d + t(hidden) − t(fiducial)``). Only blinded parts persist on disk.

    **The fork is the concealment engine.** The hidden cosmology is drawn by
    the ``UNIONS-WL/Smokescreen`` fork's own CCL-native, per-key-independent
    draw; each part goes through its ``ConcealDataVector(fiducial_params,
    shifts_dict, sacc_data, *, seed, theory_fn)`` entry point. This module
    supplies the two sp_validation-specific pieces: the three ``theory_fn``
    backends matching each part's row layout (reporting ξ±, integration ξ±, pseudo-Cℓ)
    and the SACC handling around the concealed vectors.

    **Envelope calibration.** The blinding intent is an amplitude smear of a
    chosen (S8, Ωm) box. The fork draws in CCL-native primitives, so
    :meth:`BlindingConfig.shifts_dict` maps the intended box to
    ``{sigma8, Omega_c}`` half-widths evaluated at the fiducial —
    ``σ8 = S8/√(Ωm/0.3)``, ``Ω_c = Ωm − Ω_b − Ω_ν``. The same seed yields
    the same hidden cosmology across all part passes (the fork's draw depends
    only on ``(key, seed)``), so the blinded parts are mutually consistent by
    construction.

    **Derived statistics are born blinded.** COSEBIs and pure-E/B are never
    touched by this module: the pipeline's own estimators
    (:mod:`sp_validation.b_modes`) run in their normal downstream place on
    the already-blinded integration ξ± part, so their outputs are born blinded.
    Covariance and ρ/τ PSF diagnostics are never blinded — blinding hides
    the vector, not the uncertainty, and the shift is pure E-mode so the
    B-mode null tests stay honest under the blind.

    **Custody: hash commitment, no keyholder.** :func:`blind_init` runs once
    per catalogue version: it draws an OS-entropy seed, publishes
    ``sha256(seed)`` plus a canonical config digest as a repo-committable
    ``commitment.json``, and encrypts the seed into a Fernet bundle
    (``smokescreen.encryption``) — the plaintext seed is never written.
    Each :func:`blind_part` call reads that fixed state, conceals one part,
    escrows the part's true vector into its own encrypted bundle beside the
    blinded output, and deletes the plaintext part. Terminal assembly
    (:func:`sp_validation.sacc_io.gather`) calls
    :func:`assert_consistent_blind` to fail closed unless every blindable
    part carries the identical ``blind_commitment``. :func:`unblind_part`
    verifies both hashes against the commitment *before* subtracting
    anything, then restores the true part.
"""

import dataclasses
import hashlib
import json
import os
import secrets
import warnings

import numpy as np

from . import sacc_io
from .blinding_theory import TheoryConfig, cl_ee, xi_ccl, xi_ell_grid


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
        one-to-one to Ω_c (Ω_b and Ω_ν are fixed). Exact enough for a
        blinding smear — the target is a characteristic amplitude, not a
        precise (S8, Ωm) posterior. The fork draws each key independently as
        ``U(fid − h, fid + h)``.
        """
        return {
            "sigma8": self.s8_half_width / np.sqrt(self.theory.Omega_m / 0.3),
            "Omega_c": self.omega_m_half_width,
        }

    def config_digest(self):
        """sha256 of a canonical serialization of the full blinding config.

        Binds the envelope half-widths, the complete fiducial
        :class:`TheoryConfig` (cosmology + the two ``halofit_version``
        tokens + IA fields) into one digest: JSON with sorted keys over the
        full ordered field set. Every ``float``-declared field is coerced
        through :func:`float` before serialization, so the digest depends on
        the numeric *value*, not on whether a config wrote ``-1`` or ``-1.0``
        — an int-vs-float literal difference (routine when configs come from
        YAML/CLI/humans) can no longer split one physical cosmology into two
        digests and deny a legitimate unblind. Python's ``json`` then emits
        each float via its shortest round-trip ``repr``, so two runs of the
        same config produce byte-identical digests. Checked (with
        ``sha256(seed)``) at unblind, so a wrong envelope or a mismatched
        P(k) recipe cannot silently subtract a wrong shift.
        """
        payload = {
            "s8_half_width": float(self.s8_half_width),
            "omega_m_half_width": float(self.omega_m_half_width),
            "theory": {
                f.name: (
                    float(getattr(self.theory, f.name))
                    if f.type in (float, "float")
                    else getattr(self.theory, f.name)
                )
                for f in dataclasses.fields(self.theory)
            },
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    @classmethod
    def from_overrides(cls, overrides):
        """Build from a mapping of field overrides (fail loud on unknown keys).

        ``theory`` may be given as a :class:`TheoryConfig` or a mapping of
        TheoryConfig overrides, mirroring :meth:`TheoryConfig.from_overrides`.
        """
        by_name = {f.name: f for f in dataclasses.fields(cls)}
        unknown = set(overrides) - set(by_name)
        if unknown:
            raise ValueError(
                f"unknown BlindingConfig fields {sorted(unknown)}; "
                f"valid fields are {sorted(by_name)}"
            )
        overrides = {
            name: (float(v) if by_name[name].type in (float, "float") else v)
            for name, v in overrides.items()
        }
        theory = overrides.get("theory")
        if theory is not None and not isinstance(theory, TheoryConfig):
            overrides["theory"] = TheoryConfig.from_overrides(dict(theory))
        return cls(**overrides)


# --------------------------------------------------------------------------- #
# Custody primitives
# --------------------------------------------------------------------------- #
def seed_commitment(seed):
    """Public commitment for a seed: its sha256 hex digest.

    Safe to publish and commit to the repo — it ties a blinded file to its
    blind without revealing the seed, and lets unblind refuse a wrong seed.
    """
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def hidden_params(seed, config):
    """The hidden CCL parameter point the fork realizes for ``(seed, config)``.

    Re-runs the fork's own draw (``smokescreen.param_shifts.draw_param_shifts``
    — per-key-independent, local RNG) on the calibrated envelope and overlays
    the deltas on the fiducial, exactly as ``ConcealDataVector`` does
    internally. Deterministic: same ``(seed, config)`` ⇒ same hidden point,
    forever — the reproducibility contract unblinding relies on, and what
    makes every part share one hidden cosmology under one seed.
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


def xi_pairs(s, grid):
    """Unordered source-bin pairs ``(i ≤ j)`` carrying ξ+ on ``grid``."""
    bins = source_bins(s)
    return [
        (i, j)
        for a, i in enumerate(bins)
        for j in bins[a:]
        if len(s.indices(sacc_io.XI_PLUS, sacc_io._pair((i, j)), grid=grid))
    ]


def cl_pairs(s):
    """Source-bin pairs ``(i ≤ j)`` carrying pseudo-Cℓ_EE."""
    bins = source_bins(s)
    return [
        (i, j)
        for a, i in enumerate(bins)
        for j in bins[a:]
        if len(s.indices(sacc_io.CL_EE, sacc_io._pair((i, j))))
    ]


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

    Reads each bin's n(z) directly from the block's own tracers and lays the
    output out to match the block's SACC rows element-for-element: for every
    pair, ξ± is computed at that pair's stored θ (the ``theta`` tag,
    arcmin) and scattered to the rows ``block.indices`` reports — the
    block's own row order, never an assumed pairing. IA enters from
    ``theory``'s NLA fields, identically at every parameter point.
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

    Per pair: theory Cℓ_EE on the stored ``BandpowerWindows`` support,
    binned by the same window matrix the measurement used (``W @ Cℓ_EE``),
    scattered to the block's own rows. The concealing factor the fork forms
    from this is ``W @ ΔCℓ_EE`` — the shift lands in the measured
    bandpowers; ΔBB = ΔEB ≡ 0 by construction (pure E-mode shift), so those
    blocks are simply not part of this backend's rows.
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
# Extract → conceal → merge, per block
# --------------------------------------------------------------------------- #
def _blindable_blocks(s):
    """The blindable blocks of a SACC as ``(name, indices, factory)``.

    Works identically on a standalone part (which carries exactly one block)
    and on the assembled one-file product (whose integration rows are selected by
    the ``grid`` tag — the layout contract's per-block tag selection).
    ``indices`` are each block's recorded row indices (ascending, so the
    extracted sub-SACC preserves row order); ``factory`` builds the matching
    ``theory_fn`` from the extracted sub-SACC. Blocks absent from the file
    are simply not listed.
    """
    blocks = []
    for grid in ("reporting", "integration"):
        idx = _xi_indices(s, grid)
        if len(idx):
            blocks.append(
                (
                    f"{grid} ξ±",
                    idx,
                    lambda sub, theory, grid=grid: xi_theory_fn(sub, theory, grid),
                )
            )
    idx = _cl_ee_indices(s)
    if len(idx):
        blocks.append(("pseudo-Cℓ_EE", idx, cl_theory_fn))
    return blocks


def _extract_block(s, indices):
    """Extract rows ``indices`` (ascending) into a sub-SACC, order preserved.

    Deliberately index-based rather than ``sacc_io.extract`` /
    ``update_statistic`` (which select and merge by ``(data_type, tracers,
    tags)``): Smokescreen's ``ConcealDataVector`` aligns its ``theory_fn``
    output to the sub-SACC's ``mean`` element-for-element by row position, so
    the block must be carved and written back by contiguous integer index, not
    by tag-matching. This is the one place the row-index path is load-bearing.
    """
    sub = s.copy()
    sub.keep_indices(np.asarray(indices, dtype=int))
    return sub


def _concealing_factor(s, indices, factory, config, seed):
    """The fork-computed additive concealing factor for one block.

    Extracts the block into a sub-SACC containing exactly the rows the
    ``theory_fn`` spans (the fork's length guard enforces the agreement),
    drives ``ConcealDataVector`` with the fiducial point, the calibrated
    envelope, and the block's ``theory_fn``, and returns the factor
    ``t(hidden) − t(fiducial)`` aligned to ``indices``.
    """
    from smokescreen import ConcealDataVector

    sub = _extract_block(s, indices)
    smoke = ConcealDataVector(
        config.theory.ccl_params(),
        config.shifts_dict(),
        sub,
        seed=seed,
        theory_fn=factory(sub, config.theory),
    )
    smoke.calculate_concealing_factor(factor_type="add")
    concealed = smoke.apply_concealing_to_likelihood_datavec()
    return np.asarray(concealed, dtype=float) - np.asarray(sub.mean, dtype=float)


def _set_values(s, indices, values):
    """Overwrite ``s.data[i].value`` for ``indices`` with ``values`` (aligned)."""
    for i, v in zip(indices, values):
        s.data[int(i)].value = float(v)


def _concealed(s):
    """Whether ``s`` is already a blinded file (its ``concealed`` mark is set)."""
    return bool(s.metadata.get("concealed"))


def blind_sacc(part, seed, config=None, label="A", log=print):
    """Return a blinded copy of a part SACC (covariance and tags untouched).

    Per blindable block present (a standalone part carries exactly one —
    reporting ξ±, integration ξ±, or pseudo-Cℓ_EE): extract the block into a sub-SACC,
    conceal through the fork, write the shifted values back at their
    recorded indices (row order preserved). Provenance is stamped and any
    leaked seed key stripped. A file with no blindable block (e.g. a ρ/τ
    diagnostic part) is refused loudly — it should never see a blind call.
    """
    config = config or BlindingConfig()
    if _concealed(part):
        raise ValueError("already concealed — unblind first")
    blocks = _blindable_blocks(part)
    if not blocks:
        raise ValueError(
            "no blindable block (reporting/integration ξ± or pseudo-Cℓ_EE) in this SACC "
            "— ρ/τ diagnostic parts are never blinded"
        )

    blinded = part.copy()
    for name, indices, factory in blocks:
        factor = _concealing_factor(part, indices, factory, config, seed)
        _set_values(blinded, indices, np.asarray(blinded.mean)[indices] + factor)
        log(f"[blind] {name}: shifted {len(indices)} points via Smokescreen fork")

    _stamp_provenance(blinded, seed_commitment(seed), label, config.config_digest())
    return blinded


def unblind_sacc(blinded, seed, config=None, log=print):
    """Recover the true part SACC from a blinded one + the revealed ``seed``.

    Verifies ``sha256(seed)`` and the config digest against the stamped
    metadata (loud failure on either mismatch — verification precedes
    subtraction), recomputes each block's shift from the seed through the
    same backends, and subtracts it. Works on a standalone part or on the
    assembled file (integration rows selected by the ``grid`` tag). Derived
    statistics, if present (assembled file), are *not* recomputed here — the
    pipeline's own estimators re-derive them from the unblinded integration ξ±.
    """
    config = config or BlindingConfig()
    if not _concealed(blinded):
        raise ValueError("file is not concealed — nothing to unblind")
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

    hidden = hidden_params(seed, config)
    fiducial = config.theory.ccl_params()
    part = blinded.copy()
    for name, indices, factory in _blindable_blocks(blinded):
        theory = factory(_extract_block(blinded, indices), config.theory)
        factor = theory(hidden) - theory(fiducial)
        _set_values(part, indices, np.asarray(part.mean)[indices] - factor)
        log(f"[unblind] {name}: subtracted {len(indices)} shifts")

    for key in ("concealed", "blind", "blind_commitment", "blind_config_digest"):
        part.metadata.pop(key, None)
    return part


def _stamp_provenance(s, commitment, label, config_digest):
    """Stamp the blind's public provenance; strip any leaked seed.

    ``blind_commitment`` (sha256 of the seed) ties the file to its blind
    without revealing it; ``blind_config_digest`` pins the envelope +
    fiducial the shift was drawn against; ``concealed``/``blind`` mark the
    file. ``seed_smokescreen`` — the raw seed Smokescreen's own writer would
    stamp — is popped defensively: the seed must never ride a kept file.
    """
    s.metadata.pop("seed_smokescreen", None)
    s.metadata["concealed"] = True
    s.metadata["blind"] = label
    s.metadata["blind_commitment"] = commitment
    s.metadata["blind_config_digest"] = config_digest


def stamp_concealed_passthrough(s, commitment_path):
    """Stamp a part concealed under an existing blind, values untouched.

    A born-blinded derived statistic (COSEBIs / pure-E/B) has its E-mode vector
    re-derived from the *already-blinded* integration ξ± before this call, so its
    values are blind by construction and only the provenance stamp is missing.
    ρ/τ carries no cosmological vector at all — the stamp merely clears it for
    assembly under the blind (values unchanged either way). Both cases need the
    four custody keys the fail-closed load gate and :func:`assert_consistent_blind`
    check: ``concealed``, ``blind`` (label), ``blind_commitment``
    (= ``seed_sha256``), ``blind_config_digest``. This reads those from the
    version's ``commitment.json`` (written by :func:`blind_init`) and stamps them
    via :func:`_stamp_provenance`, so a pass-through part shares the exact same
    ``(commitment, digest)`` custody state as the blinded ξ±/pseudo-Cℓ parts.

    Unlike :func:`blind_sacc`, this shifts nothing and does not require a
    blindable block — it is the seam for parts blinded (or made blind-irrelevant)
    upstream of the SACC writer.
    """
    with open(commitment_path, encoding="utf-8") as f:
        commitment = json.load(f)
    _stamp_provenance(
        s, commitment["seed_sha256"], commitment["label"], commitment["config_digest"]
    )
    return s


# --------------------------------------------------------------------------- #
# Assembly-time custody: one blind across all parts
# --------------------------------------------------------------------------- #
def assert_consistent_blind(parts):
    """Assert every blindable part shares one blind; return the shared stamp.

    Custody logic for the terminal :func:`sp_validation.sacc_io.gather`: a
    part is *blindable* if it carries a blindable block (ξ± or pseudo-Cℓ_EE);
    ρ/τ diagnostic and covariance-only parts are exempt. Fails closed —
    ``ValueError`` — if blinded and plaintext blindable parts are mixed, or
    if two parts carry different ``blind_commitment``/``blind_config_digest``
    (they were blinded under different seeds or configs and must never be
    combined). The consistency key is ``(commitment, digest)`` only — the
    ``blind`` *label* is informational provenance, not custody state, so
    parts blinded under one seed+config but tagged with different ``--label``
    values assemble cleanly (a distinct warning is logged, not a failure).

    **Unconcealed parts must be declared mocks** (PRD §4, "Mocks vs data"):
    when no blindable part is concealed, every blindable part's metadata must
    carry ``type == "mock"`` — the tag :mod:`sp_validation.sacc_io` stamps
    when the data vector is computed. An unconcealed ``type == "data"`` part,
    or one missing the tag, fails assembly closed: skipping the blind can
    never silently expose real data. A concealed+plaintext mix fails closed
    regardless of ``type`` (see above). A fully-mock plaintext assembly
    returns ``None``.

    Returns
    -------
    dict or None
        The shared blind metadata (``concealed``, ``blind``,
        ``blind_commitment``, ``blind_config_digest``) for the gather to
        stamp on the assembled file, or ``None`` when nothing is blinded.
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
    # Custody state is (commitment, digest) only — the label is provenance.
    stamps = {
        (p.metadata["blind_commitment"], p.metadata["blind_config_digest"])
        for p in concealed
    }
    if len(stamps) != 1:
        raise ValueError(
            "parts carry different blind commitments — they were blinded "
            "under different seeds or configs and must never be combined: "
            + "; ".join(f"({c[:12]}…, {d[:12]}…)" for c, d in stamps)
        )
    ((commitment, digest),) = stamps
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
    }


# --------------------------------------------------------------------------- #
# File-level custody: blind-init / blind-part / unblind
# --------------------------------------------------------------------------- #
def init_paths(blind_dir):
    """The fixed custody state written by :func:`blind_init` in ``blind_dir``."""
    return {
        "commitment": os.path.join(blind_dir, "commitment.json"),
        "bundle": os.path.join(blind_dir, "blind_seed.encrpt"),
        "key": os.path.join(blind_dir, "blind_seed.key"),
    }


def part_paths(part_path):
    """Blinded-output and escrow-bundle paths beside a part file."""
    stem, ext = os.path.splitext(part_path)
    return {
        "blinded": f"{stem}_blinded{ext or '.fits'}",
        "escrow": f"{stem}_escrow.encrpt",
        "escrow_key": f"{stem}_escrow.key",
    }


def blind_init(blind_dir, config=None, label="A", log=print):
    """Fix the blind for one catalogue version: seed, commitment, seed bundle.

    Runs once per catalogue version:

    1. Draw an OS-entropy seed (never written in plaintext, never returned).
    2. Write ``commitment.json`` (repo-committable): ``sha256(seed)`` + the
       canonical config digest + the blind label.
    3. Encrypt the seed into a Fernet bundle (``smokescreen.encryption``);
       the temporary plaintext is deleted by the encryptor.

    These outputs are the blind's fixed state; every :func:`blind_part` and
    :func:`unblind_part` call reads them.

    Custody caveat: the bundle and its Fernet key land in the *same*
    ``blind_dir``. Fernet is only as protective as the key's separation —
    anyone with both files can decrypt the seed. Keep the key out-of-band;
    colocation is convenience, not at-rest protection.

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
        "seed_sha256": seed_commitment(seed),
        "config_digest": config.config_digest(),
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

    Both ``sha256(seed)`` and the config digest are checked before the seed
    is handed to any caller — a tampered bundle or a drifted config fails
    loud here, whether the caller is about to blind or to unblind.

    Returns
    -------
    tuple
        ``(seed, commitment_dict)``.
    """
    paths = init_paths(blind_dir)
    bundle = _read_encrypted_json(paths["bundle"], paths["key"])
    with open(paths["commitment"], encoding="utf-8") as f:
        commitment = json.load(f)
    if seed_commitment(bundle["seed"]) != commitment["seed_sha256"]:
        raise ValueError(
            "bundle seed does not match the committed sha256(seed) — refusing "
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

    Reads the encrypted seed and ``commitment.json`` written by
    :func:`blind_init` (verifying both hashes), conceals the part through its
    matching backend (:func:`blind_sacc`), writes the blinded part beside the
    input with the part's true vector escrowed into a per-part Fernet bundle,
    and deletes the plaintext part — only the blinded part persists on disk.
    Each part's escrow is self-contained: restoring it needs only that
    part's bundle plus the seed bundle; corruption of one bundle loses one
    part, not all. Pass ``keep_input=True`` to retain the plaintext part
    (and own the custody implication).

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
            "seed_sha256": commitment["seed_sha256"],
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

    Decrypts the seed bundle and verifies both ``sha256(seed)`` and the
    config digest against ``commitment.json`` (fail closed on either —
    verification precedes subtraction), recomputes the part's shift from the
    seed and subtracts it (:func:`unblind_sacc`). **The seed-subtracted
    vector is the authority** — the seed plus its commitment is the custody
    root of trust, and the returned vector is what the seed math produced.

    When the part's escrow bundle exists beside the blinded file it serves
    two subordinate roles, and only after its stored ``seed_sha256`` is
    verified to match the same commitment (so an escrow from a different
    blind can never be trusted): (1) a *tighter equality check* — the seed
    subtraction and the escrowed truth must agree to ``1e-6`` relative or the
    unblind fails closed; (2) *ulp-residue removal* — float add-then-subtract
    leaves ~ulp residue against the pre-blind truth, and once the escrow is
    bound to this commitment its exact value clears that residue so the
    restore is bit-for-bit. The escrow is never the source of correctness:
    if it disagrees materially the guard raises, and a mismatched or
    unbound escrow never determines the output. On the assembled file — no
    escrow — the subtraction stands alone, with integration rows selected by the
    ``grid`` tag.
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
        if bundle.get("seed_sha256") != commitment["seed_sha256"]:
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
        # Seed-bound escrow: clear the add-then-subtract ulp residue so the
        # restore is bit-for-bit. Correctness already came from the seed
        # subtraction above; the guard proved the escrow agrees with it.
        _set_values(part, np.arange(len(true_mean)), true_mean)
        log(f"[unblind] escrow verified (subtraction residual {residual:.2e})")
    # unblind_sacc stripped the concealed/blind stamps, so this is the true
    # revealed vector; it inherits the blinded file's provenance (data vs mock).
    sacc_io.save(part, out_path, type=blinded.metadata["type"])
    log(f"[unblind] wrote {out_path}")
    return out_path


def _write_encrypted_json(encrpt_path, payload):
    """Encrypt ``payload`` (JSON) to ``encrpt_path`` + sibling ``.key``; no
    plaintext survives.

    We drive ``smokescreen.encryption.encrypt_file`` (Fernet) for the crypto
    but write the ciphertext and key at the exact names we control. Its
    ``save_file`` mode names outputs from ``basename.split('.')[0]`` — it
    truncates at the first dot — so for a dotted stem (the canonical
    catalogue-version case, e.g. ``v1.4.6.3_xi_integration_escrow``) it would land at
    ``v1.encrpt``/``v1.key``, diverging from what :func:`part_paths` declares
    and colliding across parts that share a first-dot prefix. Instead we take
    the returned ``(ciphertext, key)`` and write them ourselves.
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
