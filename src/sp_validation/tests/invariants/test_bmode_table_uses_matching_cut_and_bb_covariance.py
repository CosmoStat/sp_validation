"""The B-mode table must preserve each version's selected pure-mode, COSEBIs,
and BB PTEs. A two-version analytic fixture distinguishes the fiducial cut,
statistic labels, and BB covariance from nearby alternatives.
"""

import json
import runpy
import sys
import types
from pathlib import Path

import pytest

pytestmark = [pytest.mark.fast, pytest.mark.decision("bmodes.bmode_summary")]


def _write_inputs(tmp_path, np):
    from astropy.io import fits

    from sp_validation import sacc_io

    versions = ("A", "B")
    pure_eb_paths = []
    cosebis_paths = []
    pseudo_cl_paths = []
    covariance_paths = []
    edges = np.geomspace(1.0, 250.0, 21)
    row, col = np.indices((20, 20))

    for version_index, version in enumerate(versions):
        pure_path = tmp_path / f"{version}_pure_eb.npz"
        matrices = {
            f"pte_matrices_{stat}": (
                0.01 * (1 + row)
                + 0.001 * col
                + 0.1 * (stat_index + 1)
                + 0.01 * (version_index + 1)
            )
            for stat_index, stat in enumerate(("xip_B", "xim_B", "combined"))
        }
        np.savez(
            pure_path,
            left_edges=edges[:-1],
            right_edges=edges[1:],
            **matrices,
        )
        pure_eb_paths.append(str(pure_path))

        cosebis_path = tmp_path / f"{version}_cosebis.npz"
        np.savez(cosebis_path, pte_B=(0.123, 0.456)[version_index])
        cosebis_paths.append(str(cosebis_path))

        bb = (np.array([1.0, 2.0]), np.array([2.0, 1.0]))[version_index]
        sacc = sacc_io.new_sacc(
            {0: (np.array([0.0, 1.0]), np.ones(2))},
            metadata={"version": version},
        )
        ell = np.array([100.0, 200.0])
        sacc_io.add_pseudo_cl(
            sacc,
            (0, 0),
            ell,
            np.array([9.0, 9.0]),
            bb,
            np.zeros(2),
            window_ells=ell,
            window_weights=np.eye(2),
        )
        pseudo_path = tmp_path / f"{version}_pseudo_cl.sacc"
        sacc_io.save(sacc, str(pseudo_path), type="mock")
        pseudo_cl_paths.append(str(pseudo_path))

        covariance_path = tmp_path / f"{version}_pseudo_cl_cov.fits"
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(np.diag([1.0, 4.0]), name="COVAR_BB_BB"),
                fits.ImageHDU(np.diag([100.0, 100.0]), name="COVAR_EE_EE"),
                fits.ImageHDU(np.eye(2), name="COVAR_EB_EB"),
            ]
        ).writeto(covariance_path)
        covariance_paths.append(str(covariance_path))

    return {
        "versions": versions,
        "pure_eb": pure_eb_paths,
        "cosebis": cosebis_paths,
        "pseudo_cl": pseudo_cl_paths,
        "pseudo_cl_cov": covariance_paths,
        "pure_matrices": [
            {
                stat: float(
                    (
                        0.01 * (1 + row)
                        + 0.001 * col
                        + 0.1 * (stat_index + 1)
                        + 0.01 * (version_index + 1)
                    )[9, 15]
                )
                for stat_index, stat in enumerate(("xip_B", "xim_B", "combined"))
            }
            for version_index in range(2)
        ],
    }


def _run_summary(tmp_path, monkeypatch, inputs, *, include_pseudo_cl):
    runner = types.ModuleType("cv_runner")
    runner._unbuffer_streams = lambda: None
    runner.verify_outputs = lambda _snakemake: None
    monkeypatch.setitem(sys.modules, "cv_runner", runner)

    output_path = tmp_path / f"summary_{include_pseudo_cl}.json"
    snakemake = types.SimpleNamespace(
        params={
            "versions": list(inputs["versions"]),
            "fiducial_scale_cut": (12.0, 83.0),
            "include_pseudo_cl": include_pseudo_cl,
        },
        input={
            key: inputs[key]
            for key in ("pure_eb", "cosebis", "pseudo_cl", "pseudo_cl_cov")
        },
        output={"summary_json": str(output_path)},
    )
    script = Path.cwd() / "workflow" / "scripts" / "cv_summarize_bmodes.py"
    runpy.run_path(str(script), init_globals={"snakemake": snakemake})
    return json.loads(output_path.read_text())


def test_bmode_table_uses_matching_cut_and_bb_covariance(tmp_path, monkeypatch):
    """The deterministic identity has statistical false-alarm probability 0;
    exact matrix/COSEBIs cells and BB PTEs within rtol=1e-12, atol=1e-14 allow
    only JSON and floating-point round-off.
    """
    import numpy as np

    inputs = _write_inputs(tmp_path, np)
    table = _run_summary(tmp_path, monkeypatch, inputs, include_pseudo_cl=True)

    expected_keys = {"xip_B", "xim_B", "combined", "COSEBIS", "C_l_BB"}
    for version_index, version in enumerate(inputs["versions"]):
        row = table[version]
        assert set(row) == expected_keys
        for stat, value in inputs["pure_matrices"][version_index].items():
            assert row[stat] == value
        assert row["COSEBIS"] == (0.123, 0.456)[version_index]
        chi2 = (2.0, 4.25)[version_index]
        np.testing.assert_allclose(
            row["C_l_BB"],
            np.exp(-chi2 / 2),
            rtol=1e-12,
            atol=1e-14,
        )

    without_pseudo_cl = _run_summary(
        tmp_path, monkeypatch, inputs, include_pseudo_cl=False
    )
    for version_index, version in enumerate(inputs["versions"]):
        row = without_pseudo_cl[version]
        assert set(row) == expected_keys - {"C_l_BB"}
        for stat, value in inputs["pure_matrices"][version_index].items():
            assert row[stat] == value
        assert row["COSEBIS"] == (0.123, 0.456)[version_index]
