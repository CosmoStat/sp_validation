"""End to end: one blinded catalogue through its rule scripts, then its reveal.

A synthetic catalogue ``TOY`` is declared blinded and its blind drawn with
``blinding init``. The ξ± and assembly rule scripts run as Snakemake runs them
(``runpy`` with a ``snakemake`` object); the assembled file verifies against
the declaration, and no file of the blinded run holds a true ξ± value. The
blind is then revealed, the declaration flipped, the chain re-run, and the
audit proves blinded − true = shift(seed).
"""

import json
import re
import runpy
import types
from pathlib import Path

import numpy as np
import yaml
from _synthetic import write_synthetic_catalogs

from sp_validation import blinding as bd
from sp_validation import custody as cu
from sp_validation import sacc_io as sio

SCRIPTS = Path(__file__).resolve().parents[3] / "workflow" / "scripts"
GRID = {"min_sep": 5.0, "max_sep": 60.0, "nbins": 6, "npatch": 1}


class Named(list):
    """The shape of Snakemake's ``input``/``output``/``params``: a list with names."""

    def __init__(self, **items):
        super().__init__(items.values())
        self._items = items

    def __getitem__(self, key):
        return self._items[key] if isinstance(key, str) else super().__getitem__(key)

    def __getattr__(self, name):
        try:
            return self._items[name]
        except KeyError:
            raise AttributeError(name) from None

    def get(self, key, default=None):
        return self._items.get(key, default)


def run_rule(script, **fields):
    smk = types.SimpleNamespace(**{k: Named(**v) for k, v in fields.items()})
    runpy.run_path(
        str(SCRIPTS / script), init_globals={"snakemake": smk}, run_name="__main__"
    )


def run_chain(cat_config, out, cov):
    """Rules xi and assemble_sacc for TOY into ``out``."""
    out.mkdir(exist_ok=True)
    token = bd.declared_custody(cat_config, "TOY").token
    common = {"cat_config": str(cat_config), "custody": token}
    xi = out / "TOY_xi_reporting.sacc"
    run_rule(
        "run_2pcf.py",
        input={},
        output={"sacc": str(xi)},
        params={"ver": "TOY", **GRID, "output_dir": str(out), "grid": "reporting"}
        | common,
    )
    run_rule(
        "assemble_sacc.py",
        input={"xi_reporting": str(xi), "xi_cov": str(cov)},
        output={"sacc": str(out / "TOY.sacc")},
        params={"version": "TOY", "expected": ["xi_reporting"]} | common,
    )


def plaintext(blobs, values):
    """Names of ``blobs`` ({name: bytes}) holding any of ``values``, as float64
    of either byte order at any offset, or as text in exponent notation."""
    values = np.sort(np.asarray(values, float))

    def near(x, rtol):
        keep = np.isfinite(x) & (x != 0)
        x, rtol = x[keep], np.broadcast_to(rtol, keep.shape)[keep]
        i = np.clip(np.searchsorted(values, x), 1, len(values) - 1)
        gap = np.minimum(np.abs(x - values[i - 1]), np.abs(x - values[i]))
        return bool(np.any(gap <= rtol * np.abs(x)))

    found = []
    for name, blob in blobs.items():
        for order in "<>":
            for offset in range(8):
                count = (len(blob) - offset) // 8
                if count > 0 and near(
                    np.frombuffer(blob, f"{order}f8", count, offset), 1e-9
                ):
                    found.append(name)
        tokens = re.findall(rb"(-?\d\.(\d+)e[-+]\d+)", blob)
        digits = np.array([0.51 * 10.0 ** -len(d) for _, d in tokens])
        if tokens and near(np.array([float(t) for t, _ in tokens]), digits):
            found.append(name)
    return found


def test_a_blinded_catalogue_from_birth_to_audit(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import cv_runner

    # Rule jobs line-buffer their own streams; under pytest they are captured.
    monkeypatch.setattr(cv_runner, "_unbuffer_streams", lambda: None)
    params, _ = write_synthetic_catalogs(
        tmp_path, n_gal=20000, coherent_shear=True, catalogues={"TOY": None}
    )
    cat_config = Path(params["catalog_config"])
    fast = tmp_path / "fast.json"
    fast.write_text(json.dumps({"theory": {"transfer_function": "eisenstein_hu"}}))
    monkeypatch.setattr(bd.secrets, "token_hex", lambda n: "e2e-seed")
    bd.main(
        ["init", "toy", "TOY", "--cat-config", str(cat_config), "--config", str(fast)]
    )
    cov = tmp_path / "cov.txt"
    np.savetxt(cov, np.diag(np.full(2 * GRID["nbins"], 1e-10)))
    out = tmp_path / "cosmo_val"

    # --- the blinded run -----------------------------------------------------
    run_chain(cat_config, out, cov)
    blinded = bd.declared_custody(cat_config, "TOY")
    for part in out.glob("*.sacc"):
        assert cu.read_stamp(sio.load(part).metadata).stamp == blinded.stamp
    verify = ["verify", str(out / "TOY.sacc"), "--cat-config", str(cat_config)]
    assert bd.main(verify) == 0
    blinded_run = {str(p): p.read_bytes() for p in out.rglob("*") if p.is_file()}

    # --- the reveal: archive, flip the declaration, re-measure, audit --------
    reveal = ["reveal", "toy", "--root", str(out), "--cat-config", str(cat_config)]
    assert bd.main(reveal) == 0
    config = yaml.safe_load(cat_config.read_text())
    config["TOY"]["blinding"] = "unblinded"
    cat_config.write_text(yaml.safe_dump(config, sort_keys=False))
    run_chain(cat_config, out, cov)
    archive = out / "revealed" / "toy"
    report = bd.audit("toy", archive=archive, true_root=out, cat_config=cat_config)
    assert report["ok"], json.dumps(report, indent=1, default=str)
    assert len(report["parts"]) == 2

    # --- no true ξ± value was written by the blinded run ---------------------
    gg = sio.xi_correlation(sio.load(out / "TOY_xi_reporting.sacc"))
    assert not plaintext(blinded_run, np.concatenate([gg.xip, gg.xim]))
    assert not list(tmp_path.rglob("*_xi_*.txt"))
