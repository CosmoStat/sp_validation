"""Make pip-installed firecrown importable without NumCosmo.

Run *inside* the target environment, after installing the ``[blinding]`` extra:

    python scripts/patch_firecrown.py

Why this exists (PRD #241, PR 1): firecrown is the theory engine for
Smokescreen blinding — only ``compute_theory_vector`` on the SACC-read
cosmic-shear path is used. Upstream distributes firecrown via conda-forge,
where NumCosmo (a GObject-introspection C library, absent from PyPI) is always
present; in a pip/uv environment, firecrown 1.15.1 hits NumCosmo at *import
time* through two paths that have nothing to do with cosmic shear:

1. ``firecrown/generators/__init__.py`` eagerly re-exports the LSST Y1/Y10
   predefined n(z) bin constants, defeating the lazy ``__getattr__`` that
   ``_inferred_galaxy_zdist`` already provides — and computing those constants
   imports NumCosmo.
2. ``firecrown/likelihood/__init__.py`` eagerly imports the cluster
   likelihoods, which import ``crow`` (lsstdesc-crow), which subclasses a
   NumCosmo C class at module load (``class CountsIntegralND(Ncm.IntegralND)``).

This script (a) restores laziness in ``generators``, (b) makes the cluster
imports optional, and (c) installs a *loud* ``numcosmo_py`` shim so that any
genuine NumCosmo use raises immediately instead of being silently faked.
Everything is exact-string surgery against the pinned firecrown v1.15.1: if a
target string is missing (e.g. after a version bump), the script fails loudly
so the pin and the patch get reviewed together. Idempotent — safe to re-run.

The right long-term fix is upstream (guarded/lazy imports in firecrown); until
then this file is the entire cost of staying pip-installable.
"""

import importlib.metadata
import importlib.util
import subprocess
import sys
from pathlib import Path

EXPECTED_FIRECROWN = "1.15.1"

GENERATORS_OLD = """\
    # Lazy-loaded bins (via __getattr__)
    Y1_LENS_BINS,
    Y1_SOURCE_BINS,
    Y10_LENS_BINS,
    Y10_SOURCE_BINS,
    LSST_Y1_LENS_HARMONIC_BIN_COLLECTION,
    LSST_Y1_SOURCE_HARMONIC_BIN_COLLECTION,
    LSST_Y10_LENS_HARMONIC_BIN_COLLECTION,
    LSST_Y10_SOURCE_HARMONIC_BIN_COLLECTION,
)
"""

GENERATORS_NEW = """\
)

# NOTE (sp_validation patch, scripts/patch_firecrown.py): the LSST Y1/Y10
# predefined bin constants are computed lazily in _inferred_galaxy_zdist via a
# module-level __getattr__ that imports NumCosmo. Importing them EAGERLY here
# forced NumCosmo at `import firecrown.generators` (hence at
# `import firecrown.likelihood`), which pip cannot satisfy. Re-expose them
# lazily instead; the SACC-read cosmic-shear path never touches them.
_LAZY_BIN_NAMES = frozenset(
    {
        "Y1_LENS_BINS",
        "Y1_SOURCE_BINS",
        "Y10_LENS_BINS",
        "Y10_SOURCE_BINS",
        "LSST_Y1_LENS_HARMONIC_BIN_COLLECTION",
        "LSST_Y1_SOURCE_HARMONIC_BIN_COLLECTION",
        "LSST_Y10_LENS_HARMONIC_BIN_COLLECTION",
        "LSST_Y10_SOURCE_HARMONIC_BIN_COLLECTION",
    }
)


def __getattr__(name):
    if name in _LAZY_BIN_NAMES:
        from . import _inferred_galaxy_zdist as _z

        return getattr(_z, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""

LIKELIHOOD_OLD = """\
# Cluster statistics
from firecrown.likelihood._binned_cluster import BinnedCluster
from firecrown.likelihood._binned_cluster_number_counts import (
    BinnedClusterNumberCounts,
)
from firecrown.likelihood._binned_cluster_number_counts_shear import (
    BinnedClusterShearProfile,
)
"""

LIKELIHOOD_NEW = """\
# Cluster statistics.
# NOTE (sp_validation patch, scripts/patch_firecrown.py): the cluster
# likelihoods import `crow` (lsstdesc-crow), which subclasses NumCosmo C
# classes at module load. NumCosmo is conda-forge-only, so in a pip/uv env
# these imports fail. They are NOT on the cosmic-shear (TwoPoint/WeakLensing)
# path, so they become optional: without NumCosmo the cluster classes are
# unavailable but everything else loads.
try:
    from firecrown.likelihood._binned_cluster import BinnedCluster
    from firecrown.likelihood._binned_cluster_number_counts import (
        BinnedClusterNumberCounts,
    )
    from firecrown.likelihood._binned_cluster_number_counts_shear import (
        BinnedClusterShearProfile,
    )
except (ImportError, RuntimeError, TypeError):  # pragma: no cover
    BinnedCluster = None  # type: ignore[assignment,misc]
    BinnedClusterNumberCounts = None  # type: ignore[assignment,misc]
    BinnedClusterShearProfile = None  # type: ignore[assignment,misc]
"""

SHIM = '''\
"""Minimal loud shim for numcosmo_py (installed by sp_validation).

NumCosmo is a GObject-introspection C library available only via conda-forge.
With the companion patches to firecrown (scripts/patch_firecrown.py), the
SACC-read cosmic-shear likelihood path never imports it; this shim provides
the import-time names so the patched package loads, and any genuine numerical
use of NumCosmo raises loudly rather than being silently faked.
"""


class _Missing:
    def __init__(self, path="numcosmo_py"):
        self._p = path

    def __getattr__(self, name):
        return _Missing(f"{self._p}.{name}")

    def __call__(self, *a, **k):
        raise RuntimeError(
            f"{self._p} was called, but NumCosmo is not installed (conda-forge "
            "only, not on PyPI). It is not needed for the SACC-read "
            "cosmic-shear likelihood path."
        )

    def __getitem__(self, item):
        return _Missing(f"{self._p}[...]")


Ncm = _Missing("numcosmo_py.Ncm")
Nc = _Missing("numcosmo_py.Nc")
GObject = _Missing("numcosmo_py.GObject")


def dict_to_var_dict(*a, **k):
    raise RuntimeError("numcosmo_py.dict_to_var_dict unavailable (no NumCosmo)")


def var_dict_to_dict(*a, **k):
    raise RuntimeError("numcosmo_py.var_dict_to_dict unavailable (no NumCosmo)")
'''


def patch_file(path: Path, old: str, new: str) -> str:
    text = path.read_text()
    if new in text:
        return "already patched"
    if old not in text:
        sys.exit(
            f"FATAL: expected text not found in {path}.\n"
            "firecrown has probably been bumped past the pinned version this "
            "patch targets — review scripts/patch_firecrown.py together with "
            "the [blinding] pin in pyproject.toml."
        )
    path.write_text(text.replace(old, new, 1))
    return "patched"


def main() -> None:
    spec = importlib.util.find_spec("firecrown")
    if spec is None or spec.origin is None:
        sys.exit("FATAL: firecrown is not installed in this environment.")
    pkg = Path(spec.origin).parent

    # Metadata, not `import firecrown` — pre-patch, importing is what's broken.
    version = importlib.metadata.version("firecrown")
    if version != EXPECTED_FIRECROWN:
        sys.exit(
            f"FATAL: firecrown {version} != expected {EXPECTED_FIRECROWN}; "
            "review this patch against the new version before bumping "
            "EXPECTED_FIRECROWN."
        )

    print(
        "generators/__init__.py:",
        patch_file(pkg / "generators" / "__init__.py", GENERATORS_OLD, GENERATORS_NEW),
    )
    print(
        "likelihood/__init__.py:",
        patch_file(pkg / "likelihood" / "__init__.py", LIKELIHOOD_OLD, LIKELIHOOD_NEW),
    )

    # Loud numcosmo_py shim — only when no real NumCosmo is present.
    if importlib.util.find_spec("numcosmo_py") is None:
        shim_dir = pkg.parent / "numcosmo_py"
        shim_dir.mkdir(exist_ok=True)
        (shim_dir / "__init__.py").write_text(SHIM)
        print("numcosmo_py shim: installed")
    else:
        print("numcosmo_py shim: skipped (numcosmo_py importable)")

    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import firecrown.likelihood; import smokescreen",
        ],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        sys.exit(f"FATAL: post-patch import check failed:\n{check.stderr}")
    print("post-patch import check: firecrown.likelihood + smokescreen OK")


if __name__ == "__main__":
    main()
