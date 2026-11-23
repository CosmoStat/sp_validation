"""The fetch manifest records every file's hash and checks a checksum list."""

import hashlib
import json

import pytest

from sp_validation import fetch


@pytest.fixture
def release(tmp_path):
    product = tmp_path / "v9" / "nz"
    product.mkdir(parents=True)
    (product / "a.txt").write_text("alpha")
    (product / "sub").mkdir()
    (product / "sub" / "b.txt").write_text("beta")
    (tmp_path / "v9" / "other.txt").write_text("gamma")
    lines = [
        f"{hashlib.md5(text.encode()).hexdigest()}  ./{name}"
        for name, text in [("nz/a.txt", "alpha"), ("nz/sub/b.txt", "beta"), ("other.txt", "gamma")]
    ]
    (tmp_path / "v9" / "MD5SUMS").write_text("\n".join(lines) + "\n")
    return tmp_path / "v9"


def test_manifest_hashes_every_file(release, tmp_path):
    out = tmp_path / "m.json"
    fetch.main([str(release / "nz"), "--version", "v9", "--md5sums", str(release / "MD5SUMS"), "--out", str(out)])
    record = json.loads(out.read_text())
    assert record["version"] == "v9"
    assert record["files"]["sub/b.txt"] == {"size": 4, "sha256": hashlib.sha256(b"beta").hexdigest()}
    assert set(record["files"]) == {"a.txt", "sub/b.txt"}


def test_single_file(release):
    record = fetch.manifest(str(release / "other.txt"), "v9", str(release / "MD5SUMS"))
    assert set(record["files"]) == {"other.txt"}


def test_changed_bytes_fail(release):
    (release / "nz" / "a.txt").write_text("ALPHA")
    with pytest.raises(SystemExit, match="1 differ"):
        fetch.manifest(str(release / "nz"), "v9", str(release / "MD5SUMS"))


def test_missing_file_fails(release):
    (release / "nz" / "sub" / "b.txt").unlink()
    with pytest.raises(SystemExit, match="1 missing"):
        fetch.manifest(str(release / "nz"), "v9", str(release / "MD5SUMS"))


def test_product_beside_its_checksum_list(release):
    record = fetch.manifest(str(release), "v9", str(release / "MD5SUMS"))
    assert set(record["files"]) >= {"nz/a.txt", "other.txt"}
    (release / "other.txt").write_text("GAMMA")
    with pytest.raises(SystemExit, match="1 differ"):
        fetch.manifest(str(release), "v9", str(release / "MD5SUMS"))
