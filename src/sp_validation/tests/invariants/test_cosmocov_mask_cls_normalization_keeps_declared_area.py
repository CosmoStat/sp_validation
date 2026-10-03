"""The normalized footprint-mask spectrum integrates to its declared area in steradians.

The exact area identity must survive the production two-column text serialization.
"""

import ast
from pathlib import Path

import healpy as hp
import numpy as np
import pytest

import sp_validation

pytestmark = [
    pytest.mark.fast,
    pytest.mark.decision("covariance.footprint_mask_power"),
]

NSIDE = 8
LMAX = 3 * NSIDE - 1
CODE_ROOT = Path(sp_validation.__file__).resolve().parents[2]
MASKING_SCRIPT = CODE_ROOT / "scripts" / "masking.py"


def _normalization_code():
    """Compile the real CLI's area-normalization and text-output statements."""
    assert MASKING_SCRIPT.is_file(), f"Missing estimator script: {MASKING_SCRIPT}"
    tree = ast.parse(MASKING_SCRIPT.read_text(), filename=str(MASKING_SCRIPT))
    main_guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and len(node.test.comparators) == 1
        and ast.literal_eval(node.test.comparators[0]) == "__main__"
    ]
    assert len(main_guards) == 1, "Cannot isolate masking.py's CLI block"

    wanted = {"integral_w", "norm_factor", "norm_cls", "idx", "data_to_save"}
    selected = []
    found = set()
    savetxt_calls = 0
    for node in main_guards[0].body:
        if isinstance(node, ast.Assign):
            names = {
                target.id for target in node.targets if isinstance(target, ast.Name)
            }
            matches = names & wanted
            if matches:
                selected.append(node)
                found.update(matches)
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "np"
            and node.value.func.attr == "savetxt"
        ):
            selected.append(node)
            savetxt_calls += 1

    assert found == wanted, f"Normalization assignments changed: found {found}"
    assert savetxt_calls == 1, "Cannot isolate masking.py's text serialization"
    return compile(
        ast.Module(body=selected, type_ignores=[]), str(MASKING_SCRIPT), "exec"
    )


def _run_normalization(cl_mask, area_obs_deg2, tmp_path, label):
    """Run the estimator's original AST statements on a synthetic spectrum."""
    ells = np.arange(len(cl_mask))
    norm_path = tmp_path / f"{label}_norm.txt"
    namespace = {
        "np": np,
        "cl_mask": cl_mask,
        "ells": ells,
        "area_obs_deg2": area_obs_deg2,
        "norm_path": norm_path,
    }
    exec(_normalization_code(), namespace)
    return namespace


def test_cosmocov_mask_cls_normalization_keeps_declared_area(tmp_path):
    """The deterministic identity has false-alarm probability 0; tolerances
    allow floating-point error on this fixed fixture."""
    z = hp.pix2vec(NSIDE, np.arange(hp.nside2npix(NSIDE)), nest=False)[2]
    cap_mask = (z > 0).astype(float)
    cap_cls = hp.anafast(cap_mask, lmax=LMAX)
    cap_area_deg2 = cap_mask.sum() * hp.nside2pixarea(NSIDE, degrees=True)
    cap = _run_normalization(cap_cls, cap_area_deg2, tmp_path, "hemisphere")

    norm_cls = cap["norm_cls"]
    ells = cap["ells"]
    expected_area_sr = cap["area_obs_deg2"] * (np.pi / 180) ** 2
    integrated_area_sr = np.sum((2 * ells + 1) * norm_cls / (4 * np.pi))
    np.testing.assert_allclose(
        integrated_area_sr,
        expected_area_sr,
        rtol=1e-12,
        atol=0,
        err_msg="normalized cap spectrum does not integrate to its declared area",
    )

    nonzero = cap["cl_mask"] != 0
    ratios = norm_cls[nonzero] / cap["cl_mask"][nonzero]
    np.testing.assert_allclose(
        ratios,
        np.full(ratios.shape, cap["norm_factor"]),
        rtol=1e-12,
        atol=0,
        err_msg="normalized multipoles do not share the script's single scale factor",
    )

    serialized = np.loadtxt(cap["norm_path"])
    assert serialized.shape == (len(ells), 2)
    np.testing.assert_array_equal(serialized[:, 0], ells)
    serialized_area_sr = np.sum(
        (2 * serialized[:, 0] + 1) * serialized[:, 1] / (4 * np.pi)
    )
    np.testing.assert_allclose(
        serialized_area_sr,
        expected_area_sr,
        rtol=2e-10,
        atol=0,
        err_msg="%.10e mask spectrum does not preserve its declared area",
    )

    full_sky_cls = np.zeros(LMAX + 1)
    full_sky_cls[0] = 4 * np.pi
    full_sky_area_deg2 = hp.nside2pixarea(NSIDE, degrees=True) * hp.nside2npix(NSIDE)
    full = _run_normalization(full_sky_cls, full_sky_area_deg2, tmp_path, "full_sky")
    # Here 4pi is the raw monopole. Area normalization scales it by 4pi again,
    # giving 16pi^2 so that the spectrum's integral is the full-sky area 4pi.
    np.testing.assert_allclose(
        full["cl_mask"][0],
        4 * np.pi,
        rtol=1e-6,
        atol=0,
        err_msg="full-sky unit mask does not have the expected raw monopole",
    )
    np.testing.assert_allclose(
        full["norm_cls"][0],
        16 * np.pi**2,
        rtol=1e-6,
        atol=0,
        err_msg="area-normalized full-sky monopole is inconsistent with its area",
    )
    assert np.max(np.abs(full["norm_cls"][1:])) < 1e-6, (
        "normalized full-sky mask has non-negligible higher multipoles: "
        f"max={np.max(np.abs(full['norm_cls'][1:])):.3e}"
    )
