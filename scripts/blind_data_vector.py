#!/usr/bin/env python3

"""Script blind_data_vector.py

Per-part data-vector blinding with :mod:`sp_validation.blinding`
(Smokescreen-fork concealment, hash-commitment custody).

``blind-init`` runs once per catalogue version: it draws an OS-entropy seed
(never printed, never written in plaintext), publishes a repo-committable
``commitment.json`` (``sha256(seed)`` + config digest), and encrypts the seed
into a Fernet bundle. ``blind-part`` blinds one intermediate part SACC
(reporting ξ±, integration ξ±, or pseudo-Cℓ) under that fixed state, escrows
the true vector into a per-part encrypted bundle beside the blinded output,
and deletes the plaintext part. ``unblind`` verifies both commitment hashes
and restores a true part (bit-for-bit when the part's escrow bundle is beside
it); it also works on the assembled ``{version}.sacc`` (integration rows
selected by the ``grid`` tag). ``verify`` is a cheap, seedless check that a
blinded file matches a commitment.

:Authors: Cail Daley

Examples
--------
Once per catalogue version::

    blind_data_vector.py blind-init blinded/

Per intermediate part, at birth::

    blind_data_vector.py blind-part parts/xi_integration.fits --blind-dir blinded/

Unblind one part::

    blind_data_vector.py unblind parts/xi_integration_blinded.fits \\
        --blind-dir blinded/ -o parts/xi_integration.fits

Verify::

    blind_data_vector.py verify parts/xi_integration_blinded.fits \\
        blinded/commitment.json
"""

import argparse
import json
import pathlib
import sys

from sp_validation import blinding, sacc_io


def _config_from_args(args):
    """A :class:`blinding.BlindingConfig` from optional CLI overrides."""
    overrides = {}
    if args.s8_half_width is not None:
        overrides["s8_half_width"] = args.s8_half_width
    if args.omega_m_half_width is not None:
        overrides["omega_m_half_width"] = args.omega_m_half_width
    return blinding.BlindingConfig.from_overrides(overrides)


def _blind_init(args):
    config = _config_from_args(args)
    blind_dir = pathlib.Path(args.blind_dir)
    blind_dir.mkdir(parents=True, exist_ok=True)
    # Refuse before drawing anything: a blind is a one-shot custody event and
    # silently overwriting a previous blind's state would destroy the record
    # tying that blind to its seed.
    clashes = [
        p
        for p in blinding.init_paths(str(blind_dir)).values()
        if pathlib.Path(p).exists()
    ]
    if clashes:
        raise SystemExit(
            "refusing to overwrite existing blind state:\n  "
            + "\n  ".join(clashes)
            + "\nPick a fresh --blind-dir (never overwrite a blind)."
        )
    blinding.blind_init(str(blind_dir), config=config, label=args.label)
    print(
        "Commit the commitment JSON to the repo; keep the bundle + key safe "
        "and separated (colocation in the blind dir is not at-rest protection)."
    )


def _blind_part(args):
    blinding.blind_part(
        args.part,
        args.blind_dir,
        config=_config_from_args(args),
        keep_input=args.keep_input,
    )


def _unblind(args):
    blinding.unblind_part(
        args.blinded,
        args.blind_dir,
        args.output,
        config=_config_from_args(args),
    )


def _verify(args):
    """Seedless check: blinded-file metadata ↔ commitment JSON."""
    s = sacc_io.load(args.blinded)
    with open(args.commitment, encoding="utf-8") as f:
        commitment = json.load(f)
    problems = []
    if not s.metadata.get("concealed"):
        problems.append("file is not marked concealed")
    if s.metadata.get("blind_commitment") != commitment["seed_sha256"]:
        problems.append("blind_commitment does not match the committed sha256(seed)")
    if s.metadata.get("blind_config_digest") != commitment["config_digest"]:
        problems.append("blind_config_digest does not match the committed digest")
    # The draw scheme is checked three ways: file ↔ commitment, and both against
    # the installed fork. A scheme mismatch is the one blinding failure with no
    # numerical symptom — the unblind would subtract a different shift than was
    # added and every other check here would still pass.
    installed = blinding.draw_scheme()
    file_scheme = s.metadata.get("blind_draw_scheme")
    committed = commitment.get("draw_scheme")
    if file_scheme is None or committed is None:
        problems.append(
            "no draw-scheme record on the file and/or in the commitment — "
            "this blind predates draw-scheme binding and cannot be verified "
            "reproducible"
        )
    elif int(file_scheme) != int(committed):
        problems.append(
            f"blind_draw_scheme {int(file_scheme)} does not match the committed "
            f"draw_scheme {int(committed)}"
        )
    elif int(file_scheme) != installed:
        problems.append(
            f"blind was drawn under Smokescreen DRAW_SCHEME={int(file_scheme)} "
            f"but the installed fork implements DRAW_SCHEME={installed} — this "
            "install cannot reproduce the shift"
        )
    if "seed_smokescreen" in s.metadata:
        problems.append("PLAINTEXT SEED LEAKED into file metadata (seed_smokescreen)")
    if problems:
        raise SystemExit("verification FAILED:\n  " + "\n  ".join(problems))
    print(
        f"OK: {args.blinded} matches {args.commitment} "
        f"(blind {s.metadata.get('blind')!r})"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    sub = parser.add_subparsers(dest="mode", required=True)

    for name in ("blind-init", "blind-part", "unblind"):
        p = sub.add_parser(name)
        p.add_argument("--s8-half-width", type=float, default=None)
        p.add_argument("--omega-m-half-width", type=float, default=None)
        if name == "blind-init":
            p.add_argument(
                "blind_dir",
                help="directory for the blind's fixed state (commitment + "
                "encrypted seed bundle)",
            )
            p.add_argument("--label", default="A", help="blind label (default A)")
            p.set_defaults(func=_blind_init)
        elif name == "blind-part":
            p.add_argument("part", help="intermediate part SACC file to blind")
            p.add_argument(
                "--blind-dir", required=True, help="blind-init state directory"
            )
            p.add_argument(
                "--keep-input",
                action="store_true",
                help="retain the plaintext input part (default: delete it "
                "after blinding — the true vector is escrowed beside the "
                "blinded output)",
            )
            p.set_defaults(func=_blind_part)
        else:
            p.add_argument("blinded", help="blinded part (or assembled) SACC file")
            p.add_argument(
                "--blind-dir", required=True, help="blind-init state directory"
            )
            p.add_argument("-o", "--output", required=True, help="output SACC path")
            p.set_defaults(func=_unblind)

    p = sub.add_parser("verify")
    p.add_argument("blinded", help="blinded SACC file")
    p.add_argument("commitment", help="commitment JSON")
    p.set_defaults(func=_verify)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
