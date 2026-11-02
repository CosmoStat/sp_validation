"""Tomographic iNKA merging must transpose the (Q,P) block, not (P,Q).

Develop lacks this tomography-branch merger, so the test skips there.
"""

import itertools

import numpy as np
import pytest

pseudo_cl = pytest.importorskip(
    "sp_validation.cosmo_val.pseudo_cl",
    reason="Tomographic iNKA merger module is only on the tomography branch",
)
PseudoClMixin = getattr(pseudo_cl, "PseudoClMixin", None)

POLS = ["EE", "EB", "BE", "BB"]


def test_tomo_inka_merge_lower_triangle_uses_transposed_qp_block(tmp_path):
    """The merged tomographic iNKA covariance must equal the joint covariance.

    Draw a symmetric positive-definite covariance C over the data vector indexed
    (polarisation P, spectrum pair s, ell). The correct COVAR_P_Q is therefore
    C[P, :, :, Q, :, :] by construction. Write only upper-triangle spectrum-pair
    blocks (a <= b), as the pipeline does, in its [ell, pol, ell, pol] layout
    using the real _save_iNKA_covariance method.

    The lower block (b, a) of COVAR_P_Q is Cov(C_b^P, C_a^Q), which equals
    block_{Q,P}(a, b).T. Using block_{P,Q}(a, b).T is only right when P == Q.
    This protects all 16 polarisation blocks, including EE_BB and EB_BE, and
    the symmetry COVAR_P_Q == COVAR_Q_P.T. It consolidates both audit reproductions
    of the same incorrect cross-polarisation transpose.
    """
    if PseudoClMixin is None or not hasattr(PseudoClMixin, "_merge_iNKA_covariance"):
        pytest.skip("Tomographic iNKA merger is only on the tomography branch")

    n_bins_src = [1, 2]
    pairs = list(itertools.combinations_with_replacement(n_bins_src, 2))
    n_s, n_ell, n_pol = len(pairs), 3, 4

    rng = np.random.default_rng(7)
    dim = n_pol * n_s * n_ell
    draws = rng.standard_normal((dim, dim))
    covariance = (draws @ draws.T / dim + np.eye(dim)).reshape(
        n_pol, n_s, n_ell, n_pol, n_s, n_ell
    )

    class Tiny(PseudoClMixin):
        def _get_tomo_bins(self, ver):
            return n_bins_src, pairs

        def _output_path_pseudo_cl_cov(self, ver, method, tomography):
            return str(tmp_path / "merged.fits")

        def _output_path_iNKA_block_cov(self, ver, tomo_bin_quad):
            return str(tmp_path / ("_".join(map(str, tomo_bin_quad)) + ".fits"))

    instance = Tiny()
    for ia, ib in itertools.combinations_with_replacement(range(n_s), 2):
        # [ell_a, pol_a, ell_b, pol_b]
        block = np.transpose(covariance[:, ia, :, :, ib, :], (1, 0, 3, 2))
        quad = (*pairs[ia], *pairs[ib])
        with instance._save_iNKA_covariance(
            block, instance._output_path_iNKA_block_cov("v", quad)
        ):
            pass

    with instance._merge_iNKA_covariance("v", True) as merged:
        errors = {}
        for i, pa in enumerate(POLS):
            for j, pb in enumerate(POLS):
                expected = covariance[i, :, :, j, :, :].reshape(
                    n_s * n_ell, n_s * n_ell
                )
                got = merged[f"COVAR_{pa}_{pb}"].data
                errors[f"{pa}_{pb}"] = np.max(np.abs(got - expected))
        bad = {k: f"{v:.3g}" for k, v in errors.items() if v > 1e-12}
        asym = np.max(np.abs(merged["COVAR_EE_BB"].data - merged["COVAR_BB_EE"].data.T))
        assert not bad and asym < 1e-12, (
            f"Merged blocks differ from the joint covariance: {bad}; "
            f"max|COVAR_EE_BB - COVAR_BB_EE.T| = {asym:.3g}"
        )
        joint = np.block(
            [[merged[f"COVAR_{pa}_{pb}"].data for pb in POLS] for pa in POLS]
        )
        np.testing.assert_allclose(joint, joint.T, rtol=0, atol=1e-12)
        eigenvalues = np.linalg.eigvalsh(joint)
        assert eigenvalues.min() > 0
        print(
            f"iNKA joint covariance: symmetry error={np.max(np.abs(joint - joint.T)):.3g}, "
            f"min eigenvalue={eigenvalues.min():.12g}"
        )


def test_fresh_cv_merges_tomographic_covariance(tmp_path, monkeypatch):
    from sp_validation.cosmo_val import CosmologyValidation

    cv = CosmologyValidation.__new__(CosmologyValidation)
    cv.versions = ["v"]
    cv.nside = 4
    cv.cell_method = "catalog"
    cv.fiducial_input_inka = "uncoupled"
    cv.noise_bias_method = "analytic"
    cv.force_run = False
    cv.cc = {"v": {"shear": "unused"}}
    pairs = [(1, 1), (1, 2), (2, 2)]
    monkeypatch.setattr(cv, "_get_tomo_bins", lambda ver: ([1, 2], pairs))
    monkeypatch.setattr(cv, "_output_path", lambda *a: str(tmp_path))
    monkeypatch.setattr(
        cv, "_output_path_pseudo_cl_cov", lambda *a, **k: str(tmp_path / "merged.fits")
    )
    monkeypatch.setattr(
        cv,
        "_output_path_iNKA_block_cov",
        lambda ver, tomo_bin_quad: str(
            tmp_path / ("_".join(map(str, tomo_bin_quad)) + ".fits")
        ),
    )
    monkeypatch.setattr(
        cv, "get_fiducial_cl", lambda *a: {f"W{x}xW{y}": np.zeros(8) for x, y in pairs}
    )
    monkeypatch.setattr(cv, "_get_tomographic_bin", lambda params, cat, b: cat)
    monkeypatch.setattr(cv, "get_namaster_bin", lambda *a: None)
    monkeypatch.setattr(
        pseudo_cl,
        "get_params_rho_tau",
        lambda entry: {
            k: k for k in ("ra_col", "dec_col", "e1_col", "e2_col", "w_col")
        },
    )
    monkeypatch.setattr(
        pseudo_cl,
        "open_entry",
        lambda *a: {
            k: np.zeros(2) for k in ("ra_col", "dec_col", "e1_col", "e2_col", "w_col")
        },
    )
    monkeypatch.setattr(
        pseudo_cl.spv_pseudo_cl,
        "get_field_and_workspace_from_catalog",
        lambda *a, **k: (None, None, None),
    )
    monkeypatch.setattr(
        pseudo_cl.spv_pseudo_cl,
        "get_pseudo_cl_iNKA_covariance",
        lambda *a, **k: np.zeros((3, 4, 3, 4)),
    )
    cv.pol_factor = 1
    cv.calculate_pseudo_cl_inka_cov(compute_tomography=True)
    with pseudo_cl.fits.open(tmp_path / "merged.fits") as merged:
        assert merged["COVAR_EE_EE"].data.shape == (9, 9)
    assert all(
        "pseudo_cl" not in cv._pseudo_cls["v"][f"tomo_bin_{a}_tomo_bin_{b}"]
        for a, b in pairs
    )
