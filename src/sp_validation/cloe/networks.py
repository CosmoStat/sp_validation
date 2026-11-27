"""The CosmoPower-JAX networks the theory evaluates, fetched by checksum.

cloelib reads its networks from ``cloelib/cosmology/emulator-data-jax/`` and, when one
is missing, downloads it from the emulator repository's ``main``, which has moved:
the sigma8 network there no longer matches the bytes the CLOE chains were run and
validated with. :func:`fetch` puts the pinned bytes in cloelib's directory first,
from the repository's commit 61144e21 (4 Oct 2026), and refuses any file whose
sha256 differs. The container build runs it once::

    python -m sp_validation.cloe.networks
"""

import hashlib
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/cosmopower-organization/"
    "Euclid-DR1-matter-emulators/61144e21f807a59025b2e4f7aebf2e17f3c86e65/"
    "emulators/lcdm/hmcode"
)
#: file -> sha256: HMcode-2020 + feedback P(k), sigma8, and the linear total and cold
#: P(k) of the ``camb_mead2020`` engine's feedback response.
SHA256 = {
    "lcdm-1mass-nonlinear.npz": "25dfe45bc34ddd8b6d837b5728324f7f8cb9691438c9264c3aef1380c38ee5ac",
    "lcdm-1mass-s8-fs8.npz": "007956e51f7c1ad5e0ccc417ed6244bb8fdd0f92325c7fd6aa23a5bc098be53d",
    "lcdm-1mass-linear.npz": "07f69ba38526adc81717b77c6a5cf0bcb42e843abf5047bb79ee537f3a0c8686",
    "lcdm-1mass-cb-linear.npz": "7bef62326b01058ca2df8c24b519e480161ca5ce50c5bf00bdcf6dea7bf83911",
}


def directory():
    """cloelib's network directory."""
    import cloelib.cosmology.cosmopower_jax_cosmology as cp

    return Path(cp.__file__).parent / "emulator-data-jax"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch():
    """Every network of :data:`SHA256` in :func:`directory`, checked; returns it."""
    out = directory()
    out.mkdir(exist_ok=True)
    for name, digest in SHA256.items():
        path = out / name
        if not path.exists():
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(f"{URL}/{name}", tmp)
            tmp.replace(path)
        if sha256(path) != digest:
            raise RuntimeError(f"{path} does not match its pinned sha256 {digest}")
    return out


if __name__ == "__main__":
    print(fetch())
