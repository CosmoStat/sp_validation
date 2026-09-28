"""Back-pressure guard #3: configured filesystem paths exist on Candide.

This is deliberately Candide-local. It checks that path-shaped values in the
science configs still point at real files/directories before restructuring
moves repo files around.
"""

import configparser
import socket
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

PATH_KEY_PARTS = (
    "path",
    "file",
    "dir",
    "folder",
    "root",
    "catalog",
    "catalogue",
)
# Keys whose values are never filesystem paths to check. ``extra_output`` is a
# flag, not a path. ``why``/``replace``/``with``/``drop`` are the declaration
# keys of a mask *overlay* (config/calibration/*.overlay.yaml): ``why`` is
# rationale prose and ``replace``/``with``/``drop`` are verbatim blocks of base
# config text -- content, not paths -- so the path walker must not treat them as
# files to stat. (The overlay's one real path, ``base:``, is deliberately not
# listed, so it is still validated.)
NON_PATH_KEYS = ("extra_output", "why", "replace", "with", "drop")
PATH_PREFIX_KEYS = ("nz.dndz.path",)
TEXT_SUFFIXES = (
    ".fits",
    ".fits.gz",
    ".hdf5",
    ".h5",
    ".yaml",
    ".yml",
    ".ini",
    ".json",
    ".txt",
    ".dat",
    ".npy",
    ".npz",
    ".pkl",
    ".sacc",
    ".tgz",
    ".tar.gz",
)
SKIP_VALUE_PARTS = ("$(", "${", "%(", "{", "}")


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("could not locate repo root (no pyproject.toml above test)")


def _on_candide() -> bool:
    return (
        socket.gethostname().split(".")[0] == "candide"
        or Path("/automnt/n17data/cdaley").exists()
    )


def _pathish_key(key: str) -> bool:
    lowered = key.lower()
    if lowered.endswith(NON_PATH_KEYS):
        return False
    return any(part in lowered for part in PATH_KEY_PARTS)


def _non_path_key(key: str) -> bool:
    return key.lower().endswith(NON_PATH_KEYS)


def _path_prefix_key(key: str) -> bool:
    return key.lower() in PATH_PREFIX_KEYS


def _pathish_value(value: str) -> bool:
    if not value or any(part in value for part in SKIP_VALUE_PARTS):
        return False
    return (
        value.startswith(("/", "./", "../", "~"))
        or "/" in value
        or value.lower().endswith(TEXT_SUFFIXES)
    )


def _walk_yaml(
    value, trail: tuple[str, ...] = (), base_dir: Path | None = None
) -> Iterator[tuple[str, str, Path | None]]:
    if isinstance(value, dict):
        local_base = base_dir
        subdir = value.get("subdir")
        if isinstance(subdir, str) and _pathish_value(subdir):
            local_base = Path(subdir).expanduser()
        for key, child in value.items():
            child_base = base_dir if key == "subdir" else local_base
            yield from _walk_yaml(child, trail + (str(key),), child_base)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_yaml(child, trail + (str(index),), base_dir)
    elif isinstance(value, str):
        key = ".".join(trail)
        if (
            not _non_path_key(key)
            and not _path_prefix_key(key)
            and (_pathish_key(key) or _pathish_value(value))
        ):
            yield key, value, base_dir


def _iter_yaml_paths(config_path: Path) -> Iterator[tuple[Path, str, str, Path | None]]:
    with config_path.open() as handle:
        data = yaml.safe_load(handle) or {}
    for key, value, base_dir in _walk_yaml(data):
        if _pathish_value(value):
            yield config_path, key, value, base_dir


def _iter_ini_paths(config_path: Path) -> Iterator[tuple[Path, str, str, Path | None]]:
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    try:
        parser.read(config_path)
    except configparser.MissingSectionHeaderError:
        yield from _iter_colon_config_paths(config_path)
        return
    for key, value in parser.defaults().items():
        stripped = value.strip()
        if _non_path_key(key):
            continue
        if (_pathish_key(key) or _pathish_value(stripped)) and _pathish_value(stripped):
            yield config_path, f"DEFAULT.{key}", stripped, None
    for section in parser.sections():
        for key, value in parser._sections[section].items():
            if key == "__name__":
                continue
            stripped = value.strip()
            if _non_path_key(key):
                continue
            if _pathish_key(key) or _pathish_value(stripped):
                if _pathish_value(stripped):
                    yield config_path, f"{section}.{key}", stripped, None


def _iter_colon_config_paths(
    config_path: Path,
) -> Iterator[tuple[Path, str, str, Path | None]]:
    """Extract paths from sectionless ``key : value`` scientific configs."""
    for line_number, line in enumerate(config_path.read_text().splitlines(), start=1):
        stripped = line.split("#", maxsplit=1)[0].strip()
        if not stripped or ":" not in stripped:
            continue
        key, value = (part.strip() for part in stripped.split(":", maxsplit=1))
        if _non_path_key(key):
            continue
        if (_pathish_key(key) or _pathish_value(value)) and _pathish_value(value):
            yield config_path, f"line {line_number}.{key}", value, None


def _config_files() -> list[Path]:
    root = _repo_root()
    ini_files = subprocess.run(
        ["git", "ls-files", "cosmo_inference/**/*.ini"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.splitlines()
    return [
        root / "papers/bmodes/config/config.yaml",
        root / "cosmo_val/cat_config.yaml",
        *sorted((root / "config/calibration").glob("*.yaml")),
        *(root / name for name in sorted(ini_files)),
    ]


def _candidate_paths() -> list[tuple[Path, str, Path]]:
    root = _repo_root()
    candidates = []
    for config_path in _config_files():
        iterator = _iter_ini_paths if config_path.suffix == ".ini" else _iter_yaml_paths
        calibration = config_path.parent == root / "config/calibration"
        for source, key, value, base_dir in iterator(config_path):
            expanded = Path(value).expanduser()
            if (
                calibration
                and key == "params.input_path"
                and not expanded.is_absolute()
            ):
                # A calibration config runs as config_mask.yaml in its run
                # directory; a relative input_path names a file there.
                continue
            if expanded.is_absolute():
                resolved = expanded
            elif base_dir is not None:
                resolved = base_dir / expanded
            elif (
                config_path.suffix == ".ini"
                and root / "cosmo_inference" in config_path.parents
            ):
                resolved = root / "cosmo_inference" / expanded
            else:
                resolved = source.parent / expanded
            candidates.append((source.relative_to(root), key, resolved))
    return candidates


def test_configured_paths_exist_on_candide():
    """Every extracted path-shaped config value points at something real."""
    if not _on_candide():
        pytest.skip("Candide-local path guard skipped: /automnt/n17data/cdaley absent")

    candidates = _candidate_paths()
    missing = [
        f"{source}:{key} -> {path}"
        for source, key, path in candidates
        if not path.exists()
    ]
    assert candidates, "no path-shaped config values extracted"
    assert not missing, (
        f"{len(missing)} missing configured paths out of {len(candidates)} checked:\n"
        + "\n".join(missing[:25])
    )


# Columns each cat_config block declares for the rho/tau path. The psf block's
# are read from the PSF file (``rho_tau.get_params_rho_tau``); the shear
# block's are every ``*_col`` key, with the RA/Dec defaults rho/tau assumes.
PSF_COLUMN_KEYS = (
    "ra_col",
    "dec_col",
    "e1_PSF_col",
    "e2_PSF_col",
    "e1_star_col",
    "e2_star_col",
    "PSF_size",
    "star_size",
    "PSF_flag",
    "star_flag",
)
SHEAR_COLUMN_DEFAULTS = {"ra_col": "RA", "dec_col": "Dec"}

# Entries whose declared columns are known not to exist in their files, with
# why. The test fails if one of these starts passing, so the entry is dropped
# here once its config or data is fixed.
KNOWN_COLUMN_GAPS = {
    ("SP_axel_v0.0", "shear"): "file carries no PSF-shape columns",
    ("SP_v1.3", "psf"): "2022 star file stores T_{PSF,STAR}_HSM, in neither"
    " ShapePipe grammar and of unconfirmed convention",
    ("SP_v1.4.6_glass_mock", "shear"): "GLASS mock carries no PSF-shape columns",
    ("SP_v1.4.6.3_uncal_w_1", "shear"): "w_col 'one' names no column",
}


def _declared_columns(block, kind):
    if kind == "psf":
        return {key: block[key] for key in PSF_COLUMN_KEYS if key in block}
    declared = {key: val for key, val in block.items() if key.endswith("_col")}
    return SHEAR_COLUMN_DEFAULTS | declared


def test_cat_config_columns_exist_on_candide():
    """Every cat_config psf/shear column is among those its file presents.

    Reads only FITS headers, through ``grammar.read_column_names``, so a v1
    file is checked against the v2 names it presents to the rho/tau path.
    Entries whose file is absent are left to the path guard above.
    """
    if not _on_candide():
        pytest.skip("Candide-local column guard skipped: no /automnt/n17data/cdaley")

    from sp_validation.grammar import read_column_names

    with (_repo_root() / "cosmo_val/cat_config.yaml").open() as handle:
        config = yaml.safe_load(handle)

    checked, missing = 0, {}
    for version, entry in config.items():
        if not isinstance(entry, dict) or "subdir" not in entry:
            continue
        for kind in ("psf", "shear"):
            block = entry.get(kind) or {}
            if "path" not in block:
                continue
            path = Path(block["path"])
            if not path.is_absolute():
                path = Path(entry["subdir"]) / path
            if not path.exists():
                continue
            hdu = block.get("hdu") or 1
            present = set(read_column_names(path, hdu=hdu))
            absent = {
                key: col
                for key, col in _declared_columns(block, kind).items()
                if col not in present
            }
            checked += 1
            if absent:
                missing[(version, kind)] = f"{path}: {absent}"

    assert checked, "no cat_config catalogue found to check"
    unexpected = {k: v for k, v in missing.items() if k not in KNOWN_COLUMN_GAPS}
    healed = sorted(set(KNOWN_COLUMN_GAPS) - set(missing))
    assert not unexpected, "declared columns absent from their files:\n" + "\n".join(
        f"{version}.{kind} -> {detail}"
        for (version, kind), detail in unexpected.items()
    )
    assert not healed, f"known column gaps now pass; drop them: {healed}"
