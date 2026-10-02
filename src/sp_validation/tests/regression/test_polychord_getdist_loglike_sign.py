"""PolyChord conversion must write -log(posterior) in GetDist's second column."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import sp_validation

REPO = Path(__file__).resolve().parents[4]
# Cross-checkout runs must exercise the package selected by PYTHONPATH.
ACTIVE_REPO = Path(sp_validation.__file__).resolve().parents[2]
if ACTIVE_REPO != REPO:
    REPO = ACTIVE_REPO


def _load_module():
    path = REPO / "cosmo_inference" / "scripts" / "chain_postprocessing.py"
    if not path.exists():
        pytest.skip(f"{path} not found")
    spec = importlib.util.spec_from_file_location("chain_postprocessing", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Columns as CosmoSIS writes a PolyChord chain: params..., prior, like, post, weight.
HEADER = (
    "#cosmological_parameters--omega_m\tcosmological_parameters--s_8"
    "\tprior\tlike\tpost\tweight\n"
)
# Sample 0 has the highest log-posterior (-1.0); sample 2 the lowest (-50.0).
ROWS = np.array(
    [
        [0.10, 0.80, 0.0, -1.0, -1.0, 0.3],
        [0.50, 0.75, 0.0, -10.0, -10.0, 0.4],
        [0.90, 0.70, 0.0, -50.0, -50.0, 0.3],
    ]
)


def _write_chain(tmp_path):
    path = tmp_path / "samples_test.txt"
    with open(path, "w") as f:
        f.write(HEADER)
        np.savetxt(f, ROWS)
    return path


@pytest.mark.xfail(
    strict=True,
    reason="#391: PolyChord writes log posterior with wrong sign",
)
def test_polychord_getdist_column_is_minus_log_posterior(tmp_path):
    """GetDist reads column 2 of a chain as -log(likelihood/posterior), so the
    PolyChord branch of write_samples_getdist_format must write -post there.
    The fixture's log-posteriors are -1, -10, -50 by construction, so the
    expected column is +1, +10, +50; the Nautilus branch already writes
    (prior - post), i.e. -log(likelihood), and both branches must agree in sign.
    """
    cp = _load_module()
    chain = _write_chain(tmp_path)
    gd = tmp_path / "gd.txt"
    cp.write_samples_getdist_format(str(chain), str(gd), chain_type="polychord")
    written = np.loadtxt(gd)
    np.testing.assert_allclose(written[:, 0], ROWS[:, -1])  # weights fine
    np.testing.assert_allclose(
        written[:, 1],
        -ROWS[:, 4],
        err_msg="GetDist -log(post) column has the wrong sign",
    )


@pytest.mark.xfail(
    strict=True,
    reason="#391: PolyChord sign inversion selects worst best fit",
)
def test_polychord_getdist_bestfit_is_max_posterior_sample(tmp_path):
    """End to end through GetDist: the best-fit sample GetDist reports
    (getLikeStats, the minimum of its -log-like column) must be the sample with
    the highest log-posterior, omega_m = 0.10 by construction, not the worst
    one (omega_m = 0.90).
    """
    cp = _load_module()
    chain = _write_chain(tmp_path)
    root = tmp_path / "gd"
    cp.load_samples_and_write_paramnames(str(chain), str(root) + ".paramnames")
    cp.write_samples_getdist_format(str(chain), str(root) + ".txt")
    samples = cp.load_chain(str(root))
    like = samples.getLikeStats()
    bestfit_om = like.parWithName("omega_m").bestfit_sample
    assert bestfit_om == pytest.approx(0.10), (
        f"GetDist best-fit omega_m = {bestfit_om} (worst-posterior sample); "
        f"expected 0.10 (max-posterior sample); logLike_sample={like.logLike_sample}"
    )
