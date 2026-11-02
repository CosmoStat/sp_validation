"""The image-sim mask config is a declared overlay on the data mask config.

The image-sim calibration does not keep an independent copy of the mask /
calibration config: it keeps the *data* config (``mask_v1.X.9.yaml``) as the one
home for the shared cuts, and declares the sim-specific delta in an overlay
(``mask_v1.X.9_im_sim.overlay.yaml``).  ``im_compose_mask.py`` applies the
overlay to the base and must reproduce the committed runtime file
(``mask_v1.X.9_im_sim.yaml``) **byte-for-byte**.

This guard locks that equality, so the two artefacts cannot drift:

* if someone edits the runtime file without updating the overlay (or vice
  versa), :func:`test_compose_reproduces_runtime_byte_identical` goes red;
* if the base config changes such that an overlay anchor no longer matches,
  the compose fails loudly rather than emitting a wrong file --
  :func:`test_compose_fails_loud_on_stale_anchor` locks that fail-fast.

The runtime file is a tracked input to ``im_init``; keeping it byte-stable is
what keeps the reproduction gate bit-exact, so this test's unit is bytes, not
parsed YAML.
"""

import importlib.util
from pathlib import Path

import pytest


def _repo_root() -> Path:
    """Locate the repo root by walking up to the ``pyproject.toml`` marker."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("could not locate repo root (no pyproject.toml above test)")


_CALIB_DIR = _repo_root() / "config" / "calibration"
_BASE = _CALIB_DIR / "mask_v1.X.9.yaml"
_OVERLAY = _CALIB_DIR / "mask_v1.X.9_im_sim.overlay.yaml"
_RUNTIME = _CALIB_DIR / "mask_v1.X.9_im_sim.yaml"


def _compose_module():
    """Import ``workflow/scripts/im_compose_mask.py`` (lives outside the package)."""
    path = _repo_root() / "workflow" / "scripts" / "im_compose_mask.py"
    spec = importlib.util.spec_from_file_location("im_compose_mask", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compose_reproduces_runtime_byte_identical():
    """compose(base, overlay) == the committed runtime file, byte-for-byte."""
    import yaml

    compose = _compose_module().compose
    overlay = yaml.safe_load(_OVERLAY.read_text())
    base_text = _BASE.read_text()

    composed = compose(base_text, overlay)

    assert composed == _RUNTIME.read_text(), (
        "compose(mask_v1.X.9.yaml, overlay) no longer reproduces "
        "mask_v1.X.9_im_sim.yaml byte-for-byte -- the runtime file and its "
        "declared overlay have drifted; reconcile one against the other."
    )


def test_compose_fails_loud_on_stale_anchor():
    """A base whose text no longer carries an overlay anchor aborts, not composes.

    This is the drift-proofing: the overlay anchors to verbatim base text, so if
    the base config is edited such that an anchor vanishes, the compose must die
    with a clear message rather than silently emit a file missing that delta.
    """
    import yaml

    module = _compose_module()
    overlay = yaml.safe_load(_OVERLAY.read_text())
    # Drop the IMAFLAGS_ISO cut from the base text so its overlay anchor no
    # longer matches; compose must abort (SystemExit from die()).
    mangled = _BASE.read_text().replace("IMAFLAGS_ISO", "SOMETHING_ELSE")

    with pytest.raises(SystemExit, match="out of sync with the base"):
        module.compose(mangled, overlay)
