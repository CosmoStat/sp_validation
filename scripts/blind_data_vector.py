#!/usr/bin/env python3

"""Script blind_data_vector.py

CLI for :mod:`sp_validation.blinding`. ``blind-init`` fixes the blind for one
catalogue version; ``blind-part`` conceals one intermediate part SACC under it,
escrowing the true vector and deleting the plaintext; ``unblind`` verifies the
custody triple and restores a true part (or the assembled file); ``verify`` is
a cheap seedless check of a blinded file against a commitment.

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
    try:
        blinding.blind_init(str(blind_dir), config=config, label=args.label)
    except FileExistsError as exc:
        raise SystemExit(f"{exc}\nPick a fresh blind dir (never overwrite a blind).")
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
    # allow_unblinded=True: reporting that a file is *not* concealed is one of
    # the outcomes here, so the fail-closed loader must not pre-empt it.
    s = sacc_io.load(args.blinded, allow_unblinded=True)
    with open(args.commitment, encoding="utf-8") as f:
        commitment = json.load(f)
    problems = blinding.verify(s, commitment)
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
