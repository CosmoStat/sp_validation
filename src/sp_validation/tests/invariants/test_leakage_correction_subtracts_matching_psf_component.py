"""Selected-row leakage correction must publish e - alpha times matching PSF e.

Drives the real script assignments and compute_PSF_leakage (PSF column
extraction under the two-stage row selection); only the binned per-object
alpha estimator is replaced by injected coefficients.

Zero leakage must preserve both shear components, with deterministic identities
and zero statistical false-alarm probability.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from sp_validation import catalog, catalog_builders

pytestmark = [
    pytest.mark.fast,
    pytest.mark.decision("calibration.objectwise_leakage_correction"),
]


def _leakage_statements():
    """Compile the real script's leakage assignments without running its setup."""
    root = Path(catalog_builders.__file__).resolve().parents[2]
    path = root / "scripts/calibration/calibrate_comprehensive_cat.py"
    assert path.is_file(), "The catalogue calibration script must exist"
    tree = ast.parse(path.read_text(), filename=str(path))
    wanted = {
        "alpha_1",
        "alpha_2",
        "e1_leak_corrected",
        "e2_leak_corrected",
        "add_cols_data['e1_leak_corrected']",
        "add_cols_data['e2_leak_corrected']",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            ast.unparse(target) in wanted
            for assignment in node.targets
            for target in (
                assignment.elts if isinstance(assignment, ast.Tuple) else [assignment]
            )
        )
    ]
    assert len(nodes) == 5, "Calibration script changed; inspect the leakage block"
    return compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec")


def _apply_and_publish(tmp_path, monkeypatch, alpha_1, alpha_2):
    """Run the real assignments and compute_PSF_leakage with injected binned
    coefficients, then write the FITS."""
    shapes = np.array([[0.1, 0.2, -0.3, 0.4], [-0.2, 0.3, 0.4, -0.1]], dtype=np.float64)
    # Seven raw rows; the two-stage selection keeps rows 0, 3, 4, 6. Rejected
    # rows carry PSF values far from the selected ones so misselection shows.
    dat = {
        "e1_PSF": np.array([0.02, 0.9, 0.9, -0.01, 0.03, 0.9, -0.04]),
        "e2_PSF": np.array([-0.03, -0.9, -0.9, 0.04, 0.01, -0.9, 0.02]),
    }
    mask_sel = np.array([True, True, False, True, True, True, True])
    mask_flg = np.array([True, False, True, True, False, True])
    rows = np.array([0, 3, 4, 6])
    psf1, psf2 = dat["e1_PSF"][rows], dat["e2_PSF"][rows]

    def injected_coefficients(cat_gal, num_bins, weight_type="des"):
        return alpha_1, alpha_2

    monkeypatch.setattr(
        catalog_builders.calibration,
        "get_alpha_leakage_per_object",
        injected_coefficients,
    )
    namespace = {
        "sp_joint": catalog_builders,
        "cat_gal": {"w_des": np.ones(shapes.shape[1])},
        "g_corr_mc": shapes,
        "dat": dat,
        "mask_combined": SimpleNamespace(_mask=mask_sel),
        "mask_metacal": mask_flg,
        "add_cols_data": {},
    }
    exec(_leakage_statements(), namespace)

    rng = np.random.default_rng(2026)
    ra = rng.uniform(0.0, 360.0, size=shapes.shape[1])
    dec = rng.uniform(-60.0, 60.0, size=shapes.shape[1])
    output_path = tmp_path / "corrected.fits"
    catalog.write_shape_catalog(
        str(output_path),
        ra,
        dec,
        np.ones(shapes.shape[1]),
        g=shapes,
        w_type="des",
        add_cols=namespace["add_cols_data"],
    )

    prewrite = (
        namespace["e1_leak_corrected"].copy(),
        namespace["e2_leak_corrected"].copy(),
    )
    with fits.open(output_path, memmap=False) as hdus:
        readback = (
            np.asarray(hdus[1].data["e1_leak_corrected"]).copy(),
            np.asarray(hdus[1].data["e2_leak_corrected"]).copy(),
        )
    return shapes, psf1, psf2, prewrite, readback


def _assert_published_columns(prewrite, readback, expected):
    for component, name, before_write, from_fits, wanted in zip(
        (1, 2),
        ("e1_leak_corrected", "e2_leak_corrected"),
        prewrite,
        readback,
        expected,
    ):
        np.testing.assert_allclose(
            before_write,
            wanted,
            rtol=0.0,
            atol=1e-12,
            err_msg=(
                f"pre-write {name} must subtract alpha_{component} times its "
                "matching PSF component"
            ),
        )
        float32_ulp = np.abs(np.spacing(np.asarray(before_write, dtype=np.float32)))
        error = np.abs(np.asarray(from_fits, dtype=np.float64) - before_write)
        assert np.all(error <= 2.0 * float32_ulp), (
            f"FITS {name} read-back exceeds two float32 ulps: "
            f"max error={np.max(error):.3g}, "
            f"max allowed={np.max(2.0 * float32_ulp):.3g}"
        )


def test_leakage_correction_subtracts_matching_psf_component(tmp_path, monkeypatch):
    """False-alarm probability is zero; tolerances allow 1e-12 arithmetic and
    two float32 ULPs on FITS read-back."""
    alpha1 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    alpha2 = np.array([-0.4, 0.3, -0.2, 0.1], dtype=np.float64)
    shapes, psf1, psf2, prewrite, readback = _apply_and_publish(
        tmp_path, monkeypatch, alpha1, alpha2
    )
    expected = (shapes[0] - alpha1 * psf1, shapes[1] - alpha2 * psf2)
    _assert_published_columns(prewrite, readback, expected)


def test_zero_leakage_coefficient_preserves_published_shears(tmp_path, monkeypatch):
    """False-alarm probability is zero; tolerances allow 1e-12 arithmetic and
    two float32 ULPs on FITS read-back."""
    zero = np.zeros(4, dtype=np.float64)
    shapes, _psf1, _psf2, prewrite, readback = _apply_and_publish(
        tmp_path, monkeypatch, zero, zero
    )
    _assert_published_columns(prewrite, readback, (shapes[0], shapes[1]))
