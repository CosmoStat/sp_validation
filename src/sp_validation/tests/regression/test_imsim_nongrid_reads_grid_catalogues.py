"""Non-grid image-sims runs must measure m from their own catalogues."""

import ast
import os
import runpy
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy.io import fits

import sp_validation
from sp_validation.image_sims import ImageSimMBias

REPO = Path(__file__).resolve().parents[4]
# Use the selected package's checkout during cross-branch verification.
PACKAGE_REPO = Path(sp_validation.__file__).resolve().parents[2]
if PACKAGE_REPO != REPO:
    REPO = PACKAGE_REPO
SMK = REPO / "workflow" / "rules" / "image_sims.smk"
CONFIG_SCRIPT = REPO / "workflow" / "scripts" / "im_mbias_config.py"

BRANCHES = ["1z2z", "1p2z", "1m2z", "1z2p", "1z2m"]
SHEAR = {
    "1z2z": (0, 0),
    "1p2z": (1, 0),
    "1m2z": (-1, 0),
    "1z2p": (0, 1),
    "1z2m": (0, -1),
}
PAIRS = [
    {"plus": "1p2z", "minus": "1m2z", "component": 0},
    {"plus": "1z2p", "minus": "1z2m", "component": 1},
]
G_IN = 0.02
CAT = "shape_catalog_cut_ngmix.fits"


def _workflow_sim_dirs(sims_type, num):
    """The per-branch directory names the workflow builds, from the rule's own
    ``_SUFFIX`` expression (evaluated, not re-implemented)."""
    if not SMK.is_file():
        pytest.skip("image_sims.smk missing; requires the tomography branch workflow")
    line = next(ln for ln in SMK.read_text().splitlines() if ln.startswith("_SUFFIX ="))
    suffix = eval(line.split("=", 1)[1], {"SIMS_TYPE": sims_type, "NUM": num})
    return {b: f"{b}{suffix}" for b in BRANCHES}


def _write(path, branch, gain):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 50
    g1, g2 = SHEAR[branch]
    cols = [
        fits.Column("RA", "D", array=10 + 0.01 * np.arange(n)),
        fits.Column("Dec", "D", array=np.full(n, 5.0)),
        fits.Column("e1", "D", array=np.full(n, gain * g1 * G_IN)),
        fits.Column("e2", "D", array=np.full(n, gain * g2 * G_IN)),
    ]
    fits.BinTableHDU.from_columns(cols).writeto(path)


def _mbias_config(tmp_path, grids, num, sims_type):
    """Build config with the production script or the rule's inline dictionary."""
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump({"shear_amplitude": G_IN, "branches": BRANCHES, "pairs": PAIRS})
    )
    params = dict(
        grids_base=str(grids),
        num=num,
        sims_type=sims_type,
        cat_name=CAT,
        sif="none",
        shapepipe_repo=str(tmp_path),
        sp_validation_repo=str(tmp_path),
        results_dir=str(tmp_path / "results"),
        results=str(tmp_path / "results" / "m_bias_results.yaml"),
        match_radius_deg=1e-4,
        w_cols=["none"],
        n_bootstrap=5,
        pair_match=True,
        bootstrap_seed=1,
    )
    if CONFIG_SCRIPT.is_file():
        out = tmp_path / "m_bias_config.yaml"
        smk = types.SimpleNamespace(
            params=types.SimpleNamespace(**params),
            input={"manifest": str(manifest)},
            output={"cfg": str(out)},
        )
        runpy.run_path(str(CONFIG_SCRIPT), init_globals={"snakemake": smk})
        return yaml.safe_load(out.read_text())
    # Tomography emits this config inline. Evaluate only that dictionary,
    # avoiding the rule's container launch and unrelated provenance collection.
    rule = SMK.read_text().split("rule im_mbias:", 1)[1]
    tree = ast.parse(textwrap.dedent(rule.split("    run:\n", 1)[1]))
    config_node = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "mbias_cfg" for t in node.targets)
    )
    return eval(
        compile(ast.Expression(config_node), str(SMK), "eval"),
        {
            "params": types.SimpleNamespace(**params),
            "manifest": yaml.safe_load(manifest.read_text()),
            "output": types.SimpleNamespace(results=params["results"]),
            "os": os,
            "provenance": {},
            "SIMS_TYPE": sims_type,
        },
    )


@pytest.mark.xfail(
    strict=True,
    reason="#386: non-grid runs silently load stale grid catalogues",
)
def test_nongrid_run_measures_m_from_its_own_catalogues_not_stale_grid(
    tmp_path, monkeypatch
):
    """With ``sims_type: random`` and ``num: 1`` the workflow writes and
    calibrates ``<branch>_1/`` (image_sims.smk ``_SUFFIX``).  The fixture puts a
    noise-free catalogue with response 1.1 there (so m = 0.1 exactly) and, as
    a stale leftover of an earlier grid run with the same num, a response-1.0
    catalogue in ``<branch>_grid_1/`` (m = 0).  Running the estimator on the
    config the workflow builds must report m1 = m2 = 0.1, the bias of the run
    it was asked about; m = 0 means it silently read the stale grid files."""
    # Container provenance is irrelevant here; don't scan a multi-GB SIF.
    monkeypatch.delenv("APPTAINER_CONTAINER", raising=False)
    grids = tmp_path / "grids"
    sims_type, num = "random", 1
    run_dirs = _workflow_sim_dirs(sims_type, num)
    for b in BRANCHES:
        _write(grids / run_dirs[b] / CAT, b, gain=1.1)  # this run
        _write(grids / f"{b}_grid_{num}" / CAT, b, gain=1.0)  # stale grid run
    assert run_dirs["1p2z"] == "1p2z_1"

    cfg = _mbias_config(tmp_path, grids, num, sims_type)
    est = ImageSimMBias(cfg)
    est.load_catalogs(verbose=False)
    res = est.run(verbose=False)

    assert res["m1"] == pytest.approx(0.1, abs=1e-9) and res["m2"] == pytest.approx(
        0.1, abs=1e-9
    ), (
        f"sims_type={sims_type!r}: workflow built {sorted(run_dirs.values())} "
        f"but the estimator reported m1={res['m1']:.4f}, m2={res['m2']:.4f} "
        f"(expected 0.1 from this run; 0.0 = stale *_grid_{num} catalogues)"
    )
