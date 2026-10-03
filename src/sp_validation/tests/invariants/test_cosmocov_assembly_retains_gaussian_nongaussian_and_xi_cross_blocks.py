"""CosmoCov upper-triangle records must reconstruct the xi covariance faithfully.
Gaussian and non-Gaussian blocks stay separate and sum into a full xi+/xi- matrix.
"""

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import sp_validation

pytestmark = [
    pytest.mark.fast,
    pytest.mark.decision("covariance.cosmocov_terms_per_grid"),
]


def test_cosmocov_assembly_retains_gaussian_nongaussian_and_xi_cross_blocks(
    tmp_path,
):
    """False-alarm probability is 0; exact component checks and
    rtol=1e-14, atol=0 on the total allow only floating-point roundoff.
    """
    ndata = 6
    gaussian = np.diag(np.arange(10.0, 16.0)) + 0.125 * np.ones((ndata, ndata))
    nongaussian = 0.25 * np.eye(ndata) + 0.0625 * np.ones((ndata, ndata))

    blocks = {"++": [], "--": [], "+-": []}
    for i in range(ndata):
        for j in range(i, ndata):
            block = "++" if j < 3 else "--" if i >= 3 else "+-"
            blocks[block].append(
                [i, j, i, j, i, j, i, j, gaussian[i, j], nongaussian[i, j]]
            )

    rng = np.random.default_rng(1729)
    block_files = {}
    for name, records in blocks.items():
        shuffled = np.asarray(records, dtype=float)[rng.permutation(len(records))]
        path = tmp_path / f"covariance_{name}.txt"
        np.savetxt(path, shuffled, fmt="%.16e")
        block_files[name] = path

    raw = tmp_path / "covariance_cat.txt"
    raw.write_text(
        "".join(block_files[name].read_text() for name in ("+-", "--", "++")),
        encoding="ascii",
    )
    raw_records = np.loadtxt(raw)
    assert np.any((raw_records[:, 0] == 5) & (raw_records[:, 1] == 5))
    assert np.any(raw_records[:, 0] == 5) and np.any(raw_records[:, 1] == 5)

    matrix_path = tmp_path / "covariance.txt"
    gaussian_path = tmp_path / "gaussian.txt"
    plot_path = tmp_path / "correlation.png"
    script = (
        Path(sp_validation.__file__).resolve().parents[2]
        / "workflow/scripts/cosmocov_process.py"
    )
    snakemake = SimpleNamespace(
        input=[raw],
        output=SimpleNamespace(
            matrix=matrix_path, gaussian=gaussian_path, plot=plot_path
        ),
    )
    namespace = runpy.run_path(str(script), init_globals={"snakemake": snakemake})

    assert namespace["ndata"] == ndata
    np.testing.assert_array_equal(namespace["cov_g"], gaussian)
    np.testing.assert_array_equal(namespace["cov_ng"], nongaussian)
    np.testing.assert_array_equal(np.loadtxt(gaussian_path), gaussian)
    np.testing.assert_allclose(
        np.loadtxt(matrix_path),
        gaussian + nongaussian,
        rtol=1e-14,
        atol=0,
        err_msg="saved covariance must include both terms and xi+/xi- cross blocks",
    )
