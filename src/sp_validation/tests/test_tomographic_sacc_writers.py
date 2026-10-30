"""Pair-major SACC writers on synthetic tomographic data."""

import numpy as np

from sp_validation import sacc_io as sio
from sp_validation.cosmo_val import sacc_writers as sw


def test_pair_native_writers_and_tau_covariance():
    nz = {i: (np.array([0.1, 0.5]), np.array([0.5, 1.0])) for i in range(3)}
    pairs = [(i, j) for i in range(3) for j in range(i, 3)]
    theta = np.array([1.0, 2.0])

    class Workspace:
        def get_bandpower_windows(self):
            w = np.zeros((4, 2, 4, 2))
            for i in range(4):
                w[i, :, i, :] = np.eye(2)
            return w

    cl = [
        ((i, j), {"ell_eff": theta, "cl_all": np.ones((4, 2)) * k, "wsp": Workspace()})
        for k, (i, j) in enumerate(pairs)
    ]
    co = [(p, {"En": np.ones(2) * k, "Bn": np.zeros(2)}) for k, p in enumerate(pairs)]
    eb = [
        (p, {"theta": theta, **{key: np.ones(2) * k for key in sio.PURE_KEYS}})
        for k, p in enumerate(pairs)
    ]
    for part in (
        sw.pseudo_cl_to_sacc(nz, {}, cl, covariance=np.eye(36)),
        sw.cosebis_to_sacc(nz, {}, co, (1.0, 2.0), covariance=np.eye(24)),
        sw.pure_eb_to_sacc(nz, {}, eb, covariance=np.eye(72)),
    ):
        found = list(dict.fromkeys(dp.tracers for dp in part.data))
        assert found == [(f"source_{i}", f"source_{j}") for i, j in pairs]
        assert np.array_equal(part.covariance.dense, np.eye(len(part.mean)))

    rho = {"theta": theta}
    tau = {"theta": theta}
    for k in sw.RHO_K:
        for sign in ("p", "m"):
            rho[f"rho_{k}_{sign}"] = theta
            rho[f"varrho_{k}_{sign}"] = theta
    for k in sw.TAU_K:
        for sign in ("p", "m"):
            tau[f"tau_{k}_{sign}"] = theta
            tau[f"vartau_{k}_{sign}"] = theta
    cov = np.ones((6, 6)) * 3
    part = sw.rho_tau_to_sacc(
        nz, {}, rho, [(i, tau) for i in range(3)], tau_cov_th_by_bin=[(1, cov)]
    )
    indices = [
        np.concatenate(
            [
                part.indices(sio.TAU_PLUS.format(k=k), (f"source_{i}", sio.PSF_TRACER))
                for k in sw.TAU_K
            ]
        )
        for i in range(3)
    ]
    assert np.array_equal(part.covariance.dense[np.ix_(indices[1], indices[1])], cov)
    assert not np.any(part.covariance.dense[np.ix_(indices[0], indices[1])])
