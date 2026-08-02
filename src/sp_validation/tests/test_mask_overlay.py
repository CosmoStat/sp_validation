"""The image-sim mask configs are declared overlays on the data mask configs.

The image-sim calibration does not keep independent copies of the mask /
calibration configs: it keeps the *data* config (e.g. ``mask_v1.X.9.yaml``) as
the one home for the shared cuts, and declares the sim-specific delta in an
overlay (e.g. ``mask_v1.X.9_im_sim.overlay.yaml``).  ``im_compose_mask.py``
applies the overlay to the base and must reproduce the committed runtime file
(e.g. ``mask_v1.X.9_im_sim.yaml``) **byte-for-byte**.

This guard locks that equality for every ``*_im_sim.overlay.yaml`` in
``config/calibration``, so the two artefacts cannot drift:

* if someone edits a runtime file without updating its overlay (or vice
  versa), :func:`test_compose_reproduces_runtime_byte_identical` goes red;
* if a base config changes such that an overlay anchor no longer matches,
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
_OVERLAYS = sorted(_CALIB_DIR.glob("*_im_sim.overlay.yaml"))


def _compose_module():
    """Import ``workflow/scripts/im_compose_mask.py`` (lives outside the package)."""
    path = _repo_root() / "workflow" / "scripts" / "im_compose_mask.py"
    spec = importlib.util.spec_from_file_location("im_compose_mask", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_pair(overlay_path):
    """Return (overlay dict, base text, runtime path) for one overlay file."""
    import yaml

    overlay = yaml.safe_load(overlay_path.read_text())
    base_text = (overlay_path.parent / overlay["base"]).read_text()
    runtime = overlay_path.with_name(
        overlay_path.name.replace(".overlay.yaml", ".yaml")
    )
    return overlay, base_text, runtime


def test_overlays_present():
    """The glob finds the declared overlays (guards against a silent no-op run)."""
    assert _OVERLAYS, f"no *_im_sim.overlay.yaml found in {_CALIB_DIR}"


@pytest.mark.parametrize("overlay_path", _OVERLAYS, ids=lambda p: p.name)
def test_compose_reproduces_runtime_byte_identical(overlay_path):
    """compose(base, overlay) == the committed runtime file, byte-for-byte."""
    compose = _compose_module().compose
    overlay, base_text, runtime = _load_pair(overlay_path)

    composed = compose(base_text, overlay)

    assert composed == runtime.read_text(), (
        f"compose({overlay['base']}, {overlay_path.name}) no longer reproduces "
        f"{runtime.name} byte-for-byte -- the runtime file and its "
        "declared overlay have drifted; reconcile one against the other."
    )


@pytest.mark.parametrize("overlay_path", _OVERLAYS, ids=lambda p: p.name)
def test_compose_fails_loud_on_stale_anchor(overlay_path):
    """A base whose text no longer carries an overlay anchor aborts, not composes.

    This is the drift-proofing: the overlay anchors to verbatim base text, so if
    the base config is edited such that an anchor vanishes, the compose must die
    with a clear message rather than silently emit a file missing that delta.
    """
    module = _compose_module()
    overlay, base_text, _ = _load_pair(overlay_path)
    # Drop the IMAFLAGS_ISO cut from the base text so its overlay anchor no
    # longer matches; compose must abort (SystemExit from die()).
    mangled = base_text.replace("IMAFLAGS_ISO", "SOMETHING_ELSE")

    with pytest.raises(SystemExit, match="out of sync with the base"):
        module.compose(mangled, overlay)
