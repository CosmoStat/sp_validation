"""Guard: ``cosmo_val/cat_config.yaml`` has no repeated mapping keys.

PyYAML accepts duplicate keys silently, keeping the last occurrence. A merge
that resolves a conflict by keeping both sides therefore produces a config
where the *shadowed* block is invisible to every consumer and to every test
that loads it with ``yaml.safe_load`` -- which is how two ``SP_v1.4.6`` and
two ``SP_v1.3.6`` entries lived on ``develop`` with the stale ones winning.

This loads the catalogue config with a loader that raises instead.

:Author: cdaley

"""

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.fast


class DuplicateKeyError(ValueError):
    """Raised when a YAML mapping repeats a key."""


class UniqueKeySafeLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` that rejects repeated keys instead of overwriting."""

    def construct_mapping(self, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise DuplicateKeyError(
                    f"duplicate key {key!r} at line {key_node.start_mark.line + 1} "
                    f"(first seen at line {mapping[key] + 1})"
                )
            mapping[key] = key_node.start_mark.line
        return super().construct_mapping(node, deep=deep)


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("could not locate repo root (no pyproject.toml above test)")


def load_unique(path: Path):
    """Parse ``path`` as YAML, raising ``DuplicateKeyError`` on repeated keys."""
    with Path(path).open() as handle:
        return yaml.load(handle, Loader=UniqueKeySafeLoader)


def test_loader_rejects_duplicate_keys(tmp_path):
    """The loader itself catches a repeated key (and ``safe_load`` does not)."""
    config = tmp_path / "dup.yaml"
    config.write_text("a:\n  x: 1\nb:\n  y: 2\na:\n  x: 3\n")

    assert yaml.safe_load(config.read_text()) == {"a": {"x": 3}, "b": {"y": 2}}
    with pytest.raises(DuplicateKeyError, match="duplicate key 'a'"):
        load_unique(config)


def test_cat_config_has_no_duplicate_keys():
    """``cosmo_val/cat_config.yaml`` parses with no key shadowing another."""
    config_path = _repo_root() / "cosmo_val" / "cat_config.yaml"
    assert config_path.exists(), f"missing config: {config_path}"

    config = load_unique(config_path)
    assert config, "cat_config.yaml parsed empty"
