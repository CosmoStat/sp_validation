"""Bring in a product that is given to the analysis, and record what arrived.

A delivered product (a ShapePipe snapshot, PhotoPipe's photometry, a mask
ladder) is too large, or has too many files, to commit with the analysis. Its
ASTRA output is therefore this manifest: the product's source and version, and
the size and SHA-256 of every file. It names no machine-local path, so the same
bytes give the same manifest everywhere; a recipe that reads the product
resolves ``source`` with :mod:`sp_validation.data_roots`.

``source`` is written as in ``config/data_roots.yaml`` (``${unions_data}/...``)
and resolved by :mod:`sp_validation.data_roots`, which fetches a ``vos:`` path
into the local cache. That cache is not refreshed, so for a ``vos:`` product
pass ``--md5sums``: the checksum list is fetched afresh every time, and a file
the list names under the product that is missing or differs (a stale cached
copy included) is an error.

    python -m sp_validation.fetch '${unions_v2}/v2.0.0/nz' --version v2.0.0 \
        --md5sums '${unions_v2}/v2.0.0/MD5SUMS' --out nz.json
"""

import argparse
import hashlib
import json
from pathlib import Path

from sp_validation import data_roots

CHUNK = 1 << 24


def digest(path):
    """SHA-256 and MD5 of a file, read once."""
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), md5.hexdigest()


def read_md5sums(path):
    """``relative path -> md5`` from an ``md5sum``-format list."""
    table = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            md5, name = line.split(maxsplit=1)
            table[name.lstrip("*").removeprefix("./")] = md5
    return table


def manifest(source, version, md5sums=None):
    """The manifest of ``source``: its local path and every file's size and hash."""
    root = Path(data_roots.local(data_roots.expand(source)))
    files = (
        sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else [root]
    )
    entries, md5_of = {}, {}
    for path in files:
        name = path.relative_to(root).as_posix() if root.is_dir() else path.name
        sha, md5 = digest(path)
        entries[name] = {"size": path.stat().st_size, "sha256": sha}
        md5_of[name] = md5
    if md5sums:
        md5sums = data_roots.expand(md5sums)
        if md5sums.startswith(data_roots.VOS):
            cached = data_roots.CACHE / md5sums.removeprefix(data_roots.VOS).strip("/")
            cached.unlink(missing_ok=True)
        listed = Path(data_roots.local(md5sums))
        prefix = root.relative_to(listed.parent).as_posix()
        under = "" if prefix == "." else prefix + "/"
        expected = {
            Path(name).name if not root.is_dir() else name.removeprefix(under): md5
            for name, md5 in read_md5sums(listed).items()
            if name == prefix or name.startswith(under)
        }
        missing = sorted(set(expected) - set(md5_of))
        wrong = sorted(n for n, m in expected.items() if n in md5_of and md5_of[n] != m)
        if missing or wrong:
            raise SystemExit(
                f"{source} does not match {md5sums}: "
                f"{len(missing)} missing {missing[:5]}, {len(wrong)} differ {wrong[:5]}"
            )
    return {"source": source, "version": version, "files": entries}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", help="data-root path of the product")
    parser.add_argument("--version", required=True, help="the product's version")
    parser.add_argument("--md5sums", help="data-root path of an md5sum list to check")
    parser.add_argument("--out", required=True, help="where to write the manifest")
    args = parser.parse_args(argv)
    record = manifest(args.source, args.version, args.md5sums)
    Path(args.out).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
