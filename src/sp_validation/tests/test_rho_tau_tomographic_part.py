"""Tomographic ρ/τ parts from synthetic handler tables."""

from types import SimpleNamespace

import numpy as np

from sp_validation import blinding, sacc_io
from sp_validation.cosmo_val import psf_systematics as psf
from sp_validation.cosmo_val import sacc_writers as sw


def test_tomographic_rho_tau_part(tmp_path, monkeypatch):
    theta = np.array([1.0, 2.0])
    rho = {"theta": theta}
    for k in sw.RHO_K:
        for sign in ("p", "m"):
            rho[f"rho_{k}_{sign}"] = theta * (k + 1)
            rho[f"varrho_{k}_{sign}"] = theta
    calls = []

    def measure(*args, **kwargs):
        base = args[5]
        label = base.split("_")[-1]
        bin_id = 4 if label == "all" else int(label)
        calls.append(bin_id)
        tau = {"theta": theta}
        for k in sw.TAU_K:
            for sign in ("p", "m"):
                tau[f"tau_{k}_{sign}"] = theta * bin_id
                tau[f"vartau_{k}_{sign}"] = theta
        np.save(
            tmp_path / "rho_tau_stats" / f"cov_tau_{base}_th.npy",
            np.eye(6) * bin_id,
        )
        return SimpleNamespace(rho_stats=rho), SimpleNamespace(tau_stats=tau)

    monkeypatch.setattr(psf, "get_rho_tau_w_cov", measure)
    saved = {}
    monkeypatch.setattr(
        sacc_io, "save", lambda s, path, **kw: saved.update(part=s, path=path, **kw)
    )

    class CV(psf.PSFSystematicsMixin):
        cc = {"paths": {"output": str(tmp_path)}}
        versions = ["v"]
        treecorr_config = {}
        cov_estimate_method = "th"
        compute_cov_rho = False
        npatch = 1
        psf_leakage = {"sky_mask": None}

        def _get_tomo_bins(self, v):
            return [9, 2, 5], [(2, 2)]

        def _get_star_mask(self, v):
            return None

        def _get_galaxy_mask(self, v, i):
            return None

        def basename(self, v, tomo_bin_a="all"):
            return f"bin_{tomo_bin_a}"

        def sacc_nz(self, v, tomography=False):
            return {
                i: (np.array([0.1, 0.5]), np.array([0.5, 1.0]))
                for i in range(3 if tomography else 1)
            }

        def sacc_metadata(self, v):
            return {}

        def blind(self, v):
            return "none"

        def print_start(self, *args):
            pass

        print_done = print_cyan = print_magenta = print_start

    CV().calculate_rho_tau_stats(tomography=True)
    part = saved["part"]
    assert calls == [2, 5, 9]
    assert saved["path"].endswith("rho_tau_bin_all_tomo.sacc")
    assert saved["theory"] is blinding.no_signal
    assert len([d for d in part.data if d.tracers == (sacc_io.PSF_TRACER,) * 2]) == 24
    for k in sw.RHO_K:
        _, plus, minus = sacc_io.get_rho(part, k)
        np.testing.assert_array_equal(plus, rho[f"rho_{k}_p"])
        np.testing.assert_array_equal(minus, rho[f"rho_{k}_m"])
    rows = []
    for i, value in enumerate(calls):
        inds = np.concatenate(
            [
                part.indices(
                    sacc_io.TAU_PLUS.format(k=k), (f"source_{i}", sacc_io.PSF_TRACER)
                )
                for k in sw.TAU_K
            ]
        )
        rows.append(inds)
        for k in sw.TAU_K:
            _, plus, minus = sacc_io.get_tau(part, (i, 0), k)
            np.testing.assert_array_equal(plus, theta * value)
            np.testing.assert_array_equal(minus, theta * value)
        np.testing.assert_array_equal(
            part.covariance.dense[np.ix_(inds, inds)], np.eye(6) * value
        )
    for i in range(3):
        for j in range(i):
            assert not np.any(part.covariance.dense[np.ix_(rows[i], rows[j])])

    CV().calculate_rho_tau_stats(tomography=False)
    for k in sw.RHO_K:
        for tomo_values, all_values in zip(
            sacc_io.get_rho(part, k), sacc_io.get_rho(saved["part"], k)
        ):
            np.testing.assert_array_equal(tomo_values, all_values)
