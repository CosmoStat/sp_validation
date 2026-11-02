"""Every CSL module the pipeline templates reference must actually import.

The image build only *compiles* CSL (``make -C shear``); pure-Python modules
are never loaded until CosmoSIS assembles a pipeline at runtime.  That gap let
a scipy>=1.15 incompatibility (``scipy.special.lpn`` removed, imported by
``legendre.py`` via ``spec_tools`` — i.e. by every real-space likelihood) ship
in a "validated" image (#316).  This test walks the ``file =`` entries of the
committed ini templates and imports each referenced ``.py`` module, so any
CSL-vs-environment drift on a module we actually use fails the in-image suite.

Skips outside the image (no ``CSL_DIR`` / no cosmosis).
"""

import importlib.util
import os
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("cosmosis")

CSL_DIR = os.environ.get("CSL_DIR")
if not CSL_DIR or not Path(CSL_DIR).is_dir():
    pytest.skip(
        "CSL_DIR not set or missing (not in the image)", allow_module_level=True
    )

TEMPLATES = (
    Path(__file__).parents[3] / "cosmo_inference" / "cosmosis_config" / "templates"
)

_FILE_RE = re.compile(r"^file\s*=\s*%\((?:COSMOSIS_DIR|CSL_DIR)\)s/(.+)$")


def _template_modules():
    seen = set()
    for ini in sorted(TEMPLATES.glob("*.ini")):
        for line in ini.read_text().splitlines():
            m = _FILE_RE.match(line.strip())
            if m and m.group(1) not in seen:
                seen.add(m.group(1))
                yield m.group(1)


MODULES = list(_template_modules())


def test_templates_reference_csl_modules():
    """The parser found the template module list (guards against ini drift)."""
    assert len(MODULES) >= 10


@pytest.mark.parametrize("relpath", MODULES)
def test_csl_module_loads(relpath):
    path = Path(CSL_DIR) / relpath
    if path.suffix == ".so":
        # Compiled in the image (and hard-checked by the Dockerfile's `test -f`);
        # absent in a bare source checkout.
        if not path.exists():
            pytest.skip(f"{relpath} not built in this CSL checkout")
        return
    assert path.exists(), f"template references missing CSL file: {relpath}"
    # CosmoSIS modules import bare-named siblings (e.g. `import spec_tools`),
    # so load with the module's own directory on sys.path, like CosmoSIS does.
    sys.path.insert(0, str(path.parent))
    try:
        name = "csl_probe_" + relpath.replace("/", "_").replace(".py", "")
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
