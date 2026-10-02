#!/usr/bin/env python
"""Build the image-simulation campaign manifest from the sims' own records.

This is the head of the image-simulation DAG: it reads the injected shear that
the sim campaign recorded for each requested branch, cross-checks it against the
branch's *name*, and writes a single ``manifest.yaml`` that every downstream
stage reads.  The point is one home for the injected-shear facts -- amplitude,
per-branch ``(g1, g2)``, the reference branch, and the ``+/-`` pairing -- so no
literal amplitude or hard-coded branch list survives anywhere else.

The source of truth is each branch's ``basic_info.txt``, written by the sim
campaign.  The one line this parser needs looks like::

    g_cosmic           =   0.025 0.0

i.e. the literal key ``g_cosmic``, run-together whitespace, an ``=``, then the
two injected shear components ``g1 g2`` as space-separated floats (sign as a
leading ``-``; ``0.0`` for an un-sheared component).  We parse *only* that line,
by stdlib string ops -- no YAML/regex dependency -- so the script runs inside
the container with nothing but the standard library.

Branch names follow the ``1{X}2{Y}`` convention: the character after ``1`` is
the g1 sign, the character after ``2`` is the g2 sign, each one of ``p`` (+),
``m`` (-), ``z`` (0).  So ``1p2z`` injects ``(+|g|, 0)``, ``1z2m`` injects
``(0, -|g|)``, ``1z2z`` is the un-sheared reference.  The suffix
(``_grid_1`` etc.) is appended by the workflow and is not part of the sign code.

Validation (all fail-loud, each message naming the offending file and field):

* every branch's parsed ``(g1, g2)`` sign/axis matches its name's sign code;
* the reference branch parses to exactly ``(0, 0)``;
* the derived ``|g|`` (the single nonzero magnitude of a sheared branch) is
  equal across all four sheared branches -- one injected amplitude for the
  whole campaign.

Only the branches this run requests are read and validated.
"""

import argparse
import os
import sys

import yaml

# Branch-name sign code: the character after "1" (g1) and after "2" (g2).
_SIGN = {"p": +1, "m": -1, "z": 0}


def die(msg):
    """Abort with a clear, prefixed message on stderr."""
    sys.exit(f"im_build_manifest: {msg}")


def basic_info_path(input_sims_base, branch):
    """Path to a branch's basic_info.txt (the sim campaign's own record)."""
    return os.path.join(input_sims_base, branch, "basic_info.txt")


def parse_g_cosmic(path):
    """Parse the ``g_cosmic = g1 g2`` line from a basic_info.txt file.

    Returns ``(g1, g2)`` as floats.  Fails loud, naming the file, if the line
    is absent, malformed, or does not carry exactly two float components.
    """
    if not os.path.isfile(path):
        die(f"basic_info.txt not found: {path}")

    with open(path) as fh:
        lines = fh.readlines()

    matches = [ln for ln in lines if ln.split("=", 1)[0].strip() == "g_cosmic"]
    if not matches:
        die(f"no 'g_cosmic' line in {path}")
    if len(matches) > 1:
        die(f"multiple 'g_cosmic' lines in {path}")

    rhs = matches[0].split("=", 1)[1].split()
    if len(rhs) != 2:
        die(
            f"'g_cosmic' in {path}: expected two components 'g1 g2', "
            f"got {len(rhs)}: {matches[0].strip()!r}"
        )
    try:
        return float(rhs[0]), float(rhs[1])
    except ValueError:
        die(f"'g_cosmic' in {path}: components not floats: {matches[0].strip()!r}")


def sign_code(branch):
    """Extract the ``(g1_sign, g2_sign)`` code from a ``1{X}2{Y}`` branch name.

    ``branch`` is the bare sign code (e.g. ``1p2z``), suffix already stripped.
    Fails loud if the name does not match the convention.
    """
    if (
        len(branch) != 4
        or branch[0] != "1"
        or branch[2] != "2"
        or branch[1] not in _SIGN
        or branch[3] not in _SIGN
    ):
        die(
            f"branch name {branch!r} does not match the 1{{X}}2{{Y}} convention "
            f"(X, Y each one of p/m/z)"
        )
    return _SIGN[branch[1]], _SIGN[branch[3]]


def build_manifest(input_sims_base, sims_type, num, branches):
    """Parse + validate every requested branch, return the manifest dict.

    ``branches`` are bare sign codes (``1z2z``, ``1p2z``, ...); the on-disk
    directory name is ``{branch}{suffix}`` with ``suffix`` derived from
    ``sims_type``/``num`` exactly as the workflow builds it.
    """
    suffix = f"_{sims_type}_{num}" if sims_type == "grid" else f"_{num}"

    parsed = {}  # branch -> (g1, g2) from basic_info.txt
    for branch in branches:
        dirname = f"{branch}{suffix}"
        g1, g2 = parse_g_cosmic(basic_info_path(input_sims_base, dirname))
        s1, s2 = sign_code(branch)

        # Sign/axis must agree with the name: a component is nonzero iff its
        # sign code is nonzero, and its sign matches.
        for comp, (g, s) in enumerate(((g1, s1), (g2, s2)), start=1):
            path = basic_info_path(input_sims_base, dirname)
            if s == 0 and g != 0.0:
                die(
                    f"{path}: branch {branch!r} names g{comp} un-sheared (z) but "
                    f"g_cosmic gives g{comp} = {g}"
                )
            if s != 0 and (g == 0.0 or (g > 0) != (s > 0)):
                die(
                    f"{path}: branch {branch!r} names g{comp} sign {'+' if s > 0 else '-'} "
                    f"but g_cosmic gives g{comp} = {g}"
                )
        parsed[branch] = (g1, g2)

    # Reference branch: the one whose name codes (0, 0). Must exist and be (0,0).
    refs = [b for b in branches if sign_code(b) == (0, 0)]
    if len(refs) != 1:
        die(
            f"expected exactly one reference branch (name code 1z2z) among "
            f"{branches}, found {refs}"
        )
    reference = refs[0]
    if parsed[reference] != (0.0, 0.0):
        die(
            f"{basic_info_path(input_sims_base, f'{reference}{suffix}')}: reference "
            f"branch {reference!r} must inject (0, 0), got {parsed[reference]}"
        )

    # Derived amplitude: the single nonzero magnitude of each sheared branch,
    # cross-checked equal across all four.
    amplitudes = {}  # branch -> |g|
    for branch in branches:
        if branch == reference:
            continue
        g1, g2 = parsed[branch]
        amplitudes[branch] = abs(g1) if g1 != 0.0 else abs(g2)
    distinct = sorted(set(amplitudes.values()))
    if len(distinct) != 1:
        die(
            "injected |g| differs across sheared branches (must be one campaign "
            f"amplitude): {amplitudes} "
            f"[files under {input_sims_base}/<branch>{suffix}/basic_info.txt]"
        )
    shear_amplitude = distinct[0]

    # Pairs: (+component, -component) for each sheared axis, in branch order so
    # the estimator's per-pair processing order is stable.
    plus = {}  # component (0/1) -> branch with +|g| on that component
    minus = {}
    for branch in branches:
        if branch == reference:
            continue
        s1, s2 = sign_code(branch)
        comp = 0 if s1 != 0 else 1
        (plus if (s1 or s2) > 0 else minus)[comp] = branch
    pairs = [
        {"plus": plus[comp], "minus": minus[comp], "component": comp}
        for comp in sorted(set(plus) & set(minus))
    ]

    return {
        "input_sims_base": input_sims_base,
        "sims_type": sims_type,
        "num": num,
        "shear_amplitude": shear_amplitude,
        "reference": reference,
        # Branch order preserved (dict insertion order round-trips through
        # yaml.safe_dump with sort_keys=False) so downstream load order is fixed.
        "branches": {
            branch: {"g1": parsed[branch][0], "g2": parsed[branch][1]}
            for branch in branches
        },
        "pairs": pairs,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-sims-base",
        required=True,
        help="root under which each branch dir holds basic_info.txt",
    )
    p.add_argument(
        "--sims-type", required=True, help="'grid' -> _grid_{num} suffix, else _{num}"
    )
    p.add_argument("--num", required=True, type=int, help="run number")
    p.add_argument(
        "--branch",
        required=True,
        action="append",
        dest="branches",
        help="bare branch sign code (1z2z, 1p2z, ...); repeatable",
    )
    p.add_argument("-o", "--output", required=True, help="manifest.yaml output path")
    return p.parse_args()


def main():
    args = parse_args()
    manifest = build_manifest(
        args.input_sims_base, args.sims_type, args.num, args.branches
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fh:
        yaml.safe_dump(manifest, fh, sort_keys=False)
    print(f"im_build_manifest: wrote {args.output}")
    print(f"  shear_amplitude = {manifest['shear_amplitude']}")
    print(f"  reference       = {manifest['reference']}")
    print(f"  branches        = {list(manifest['branches'])}")
    for pair in manifest["pairs"]:
        print(
            f"  pair g{pair['component'] + 1}: {pair['plus']} (+) / {pair['minus']} (-)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
