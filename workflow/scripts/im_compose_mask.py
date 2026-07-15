#!/usr/bin/env python
"""Compose the image-sim mask/calibration config from base + declared overlay.

The image-sim calibration differs from the data calibration in only a handful
of places (input path, dropped coverage cuts, unweighted global response,
additive-bias off).  Instead of maintaining a second full copy of the config --
which can silently drift from the base it was branched from -- we keep the
*data* config (``mask_v1.X.9.yaml``) as the single home for the shared cuts and
declare the sim-specific delta in an overlay
(``mask_v1.X.9_im_sim.overlay.yaml``).  This script applies the overlay to the
base and emits the resolved config, which is byte-for-byte the committed runtime
file ``mask_v1.X.9_im_sim.yaml``.  A test locks that equality, so the declaration
and the runtime file cannot diverge.

The overlay is a list of block operations on the base *text* (not on parsed
YAML), so the resolved file preserves the base's exact formatting and comments
-- the property that makes byte-identity with a hand-maintained runtime file
achievable, and the delta legible as a plain diff.  Each op:

* ``drop:``    remove a verbatim block of base text;
* ``replace:`` / ``with:``  swap a verbatim block for new text.

Every anchor (the ``drop`` block, or a ``replace`` block) must occur **exactly
once** in the base -- zero or multiple matches is a hard error, so an overlay
that has fallen out of sync with the base fails loudly instead of composing
something wrong.  ``why`` is prose for the human reader and is ignored here.

Stdlib + PyYAML only, so it runs inside the sp_validation container with nothing
extra.
"""

import argparse
import os
import sys

import yaml


def die(msg):
    """Abort with a clear, prefixed message on stderr."""
    sys.exit(f"im_compose_mask: {msg}")


def _apply_once(text, anchor, replacement, *, kind, i):
    """Replace the single occurrence of ``anchor`` in ``text`` with ``replacement``.

    ``anchor`` must occur exactly once; anything else (missing, or ambiguous)
    means the overlay no longer matches the base and is a hard error naming the
    offending op.
    """
    n = text.count(anchor)
    if n != 1:
        die(
            f"op {i} ({kind}): anchor block occurs {n} time(s) in the base, "
            "expected exactly 1 -- the overlay is out of sync with the base.\n"
            f"--- anchor ---\n{anchor}\n--------------"
        )
    return text.replace(anchor, replacement)


def compose(base_text, overlay):
    """Apply ``overlay['ops']`` to ``base_text`` and return the resolved text."""
    text = base_text
    for i, op in enumerate(overlay["ops"]):
        if "drop" in op:
            text = _apply_once(text, op["drop"], "", kind="drop", i=i)
        elif "replace" in op:
            if "with" not in op:
                die(f"op {i} (replace): missing 'with:' block")
            text = _apply_once(text, op["replace"], op["with"], kind="replace", i=i)
        else:
            die(f"op {i}: needs a 'drop:' or 'replace:'/'with:' block")
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "overlay",
        help="overlay yaml declaring base + block ops "
        "(e.g. mask_v1.X.9_im_sim.overlay.yaml)",
    )
    ap.add_argument(
        "-o",
        "--output",
        help="write resolved config here; default: stdout",
    )
    args = ap.parse_args(argv)

    with open(args.overlay) as fh:
        overlay = yaml.safe_load(fh)

    # The base path is stated in the overlay, relative to the overlay's own dir,
    # so the pair travels together (both live in config/calibration/).
    base_path = os.path.join(os.path.dirname(args.overlay), overlay["base"])
    with open(base_path) as fh:
        base_text = fh.read()

    resolved = compose(base_text, overlay)

    if args.output:
        with open(args.output, "w") as fh:
            fh.write(resolved)
    else:
        sys.stdout.write(resolved)


if __name__ == "__main__":
    main()
