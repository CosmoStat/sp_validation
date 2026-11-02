"""Gaussian-sim covariance must aggregate the spectra its seeded workers wrote.

Develop lacks this tomography-branch script, so these tests skip there.
"""

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest

import sp_validation

SCRIPT_REL = "scripts/cosmo_val/run_cl_gaussian_sims.py"
VERSION = "vtest"
TOMO = False
SEED = 7
N_SIMS = 3
N_ELL = 4


@pytest.fixture
def gaussian_sims():
    # Resolve from the imported package, so PYTHONPATH can select another branch
    # while this regression file stays in the original checkout.
    root = Path(sp_validation.__file__).resolve().parents[2]
    script = root / SCRIPT_REL
    if not script.is_file():
        pytest.skip(f"{SCRIPT_REL} is only available on the tomography branch")
    pytest.importorskip("mpi4py", reason="Gaussian-sim script requires mpi4py")
    spec = importlib.util.spec_from_file_location("run_cl_gaussian_sims", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_covariance_from_simulated_spectra"):
        pytest.skip("Covariance aggregator is only available on the tomography branch")
    return module


def _cl_dict(rng):
    # NaMaster-decoupled spin-2 x spin-2: (EE, EB, BE, BB) x n_ell.
    return {"W1xW1": rng.normal(size=(4, N_ELL))}


def _writer_name(out_dir, sim_id):
    # Naming used by run_one_simulation when it saves a realisation.
    return out_dir / (f"cl_sample_{sim_id}_{VERSION}_tomography_{TOMO}_seed_{SEED}.npz")


def _stale_name(out_dir, sim_id):
    # Pre-seed naming: what an older run left behind in the same directory.
    return out_dir / f"cl_sample_{sim_id}_{VERSION}_tomography_{TOMO}.npz"


def _call(module, out_dir):
    fn = module.get_covariance_from_simulated_spectra
    kwargs = {}
    if "seed" in inspect.signature(fn).parameters:
        kwargs["seed"] = SEED
    return fn(N_SIMS, VERSION, TOMO, [1], "EE", str(out_dir), **kwargs)


def test_covariance_reads_seed_suffixed_spectra_in_fresh_dir(tmp_path, gaussian_sims):
    """A seed-S run must aggregate exactly its cl_sample_*_seed_S.npz files.

    Nothing else exists in this fresh directory, so aggregation must succeed
    and equal np.cov of the seed-S EE realisations, computed independently here.
    No HEALPix simulation, catalogue or MPI work is needed.
    """
    rng = np.random.default_rng(0)
    seeded = []
    for i in range(N_SIMS):
        cl = _cl_dict(rng)
        np.savez(_writer_name(tmp_path, i), cl_decoupled=cl)
        seeded.append(cl["W1xW1"][0])
    cov = _call(gaussian_sims, tmp_path)
    np.testing.assert_allclose(cov, np.cov(np.array(seeded), rowvar=False))


def test_covariance_ignores_stale_unsuffixed_spectra(tmp_path, gaussian_sims):
    """A seed-S covariance must not silently use stale unsuffixed spectra.

    Both sets exist, but their distinct amplitudes make their sample covariances
    different. The right answer is np.cov of this run's seeded EE realisations,
    not the pre-seed files left by an older run.
    """
    rng = np.random.default_rng(1)
    seeded, stale = [], []
    for i in range(N_SIMS):
        cl = _cl_dict(rng)
        np.savez(_writer_name(tmp_path, i), cl_decoupled=cl)
        seeded.append(cl["W1xW1"][0])
        old = {"W1xW1": 100.0 * rng.normal(size=(4, N_ELL))}
        np.savez(_stale_name(tmp_path, i), cl_decoupled=old)
        stale.append(old["W1xW1"][0])
    want = np.cov(np.array(seeded), rowvar=False)
    old_cov = np.cov(np.array(stale), rowvar=False)
    assert not np.allclose(want, old_cov), "Fixture must distinguish the two runs"
    cov = _call(gaussian_sims, tmp_path)
    np.testing.assert_allclose(cov, want)
