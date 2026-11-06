"""Tomographic derived covariances and blind inheritance."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sp_validation import b_modes, blinding, sacc_io
from sp_validation.cosmo_val import sacc_writers
from sp_validation.cosmo_val.derived_covariance import propagate

SCRIPTS = Path(__file__).resolve().parents[3] / "workflow" / "scripts"


@pytest.fixture
def synthetic(tmp_path):
    n = 64
    edges = np.geomspace(0.1, 300, n + 1)
    theta = np.sqrt(edges[:-1] * edges[1:])
    pairs = [(i, j) for i in range(3) for j in range(i, 3)]
    z = np.linspace(0.01, 2, 10)
    nz = {i: (z, np.exp(-(((z - 0.5 - i * 0.1) / 0.3) ** 2))) for i in range(3)}
    rng = np.random.default_rng(42)
    a = rng.normal(size=(len(pairs) * 2 * n, 30))
    cov = a @ a.T + np.eye(len(a)) * 4
    measurements = [
        (
            pair,
            dict(
                theta=theta,
                theta_nom=theta,
                xip=theta * 1e-5,
                xim=theta * 0.3e-5,
                weight=(1 + k * theta / 300),
                npairs=np.ones(n),
            ),
        )
        for k, pair in enumerate(pairs)
    ]
    xi = sacc_writers.xi_pairs_to_sacc(
        nz, {"npatch": 100}, measurements, grid="integration", covariance=cov
    )
    blind = blinding.init("toy", {"paths": {"blinds": str(tmp_path / "blinds")}})
    path = tmp_path / "xi.sacc"
    sacc_io.save(
        xi,
        path,
        blind=blind,
        theory=lambda params, s: np.full(len(s.mean), params["S8"]),
    )
    return path, pairs, theta, edges, cov


def test_single_pair_operators_match_existing(synthetic):
    path, pairs, theta, edges, cov = synthetic
    part = sacc_io.load(path, blind="toy")
    gg = sacc_io.xi_correlation(part, pairs[0], grid="integration")
    c = cov[:128, :128]
    reporting = np.geomspace(2, 60, 4)
    result = b_modes.calculate_pure_eb_correlation(
        theta, gg.xip, gg.xim, gg.weight, edges, c, reporting
    )
    op = b_modes.pure_eb_operator(gg.weight, edges, reporting)[0]
    np.testing.assert_array_equal(result["cov"], op @ c @ op.T)
    cuts = [(edges[4], edges[-5])]
    result = b_modes.cosebis_scan_from_xi(
        theta, gg.xip, gg.xim, c, edges[:-1], edges[1:], nmodes=2, scale_cuts=cuts
    )[cuts[0]]
    op = b_modes.cosebis_operator(theta, edges[:-1], edges[1:], cuts[0], 2)
    np.testing.assert_allclose(
        propagate([op], c, cosebis=True), result["cov"], rtol=1e-12, atol=1e-25
    )


@pytest.mark.parametrize("kind", ["pure", "cosebis"])
@pytest.mark.parametrize("analytic", [False, True])
def test_workflow_full_pair_covariance_and_stamp(
    synthetic, tmp_path, monkeypatch, kind, analytic
):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import cv_runner

    monkeypatch.setattr(cv_runner, "_unbuffer_streams", lambda: None)
    monkeypatch.setattr(cv_runner, "verify_outputs", lambda _: None)
    path, pairs, theta, edges, cov = synthetic
    part = sacc_io.load(path, blind="toy")
    n = len(theta)
    # Construct OneCovariance's type-major layout independently.
    perm = [
        k * 2 * n + t * n + q
        for t in range(2)
        for k in range(len(pairs))
        for q in range(n)
    ]
    cov_path = tmp_path / "cov.txt"
    np.savetxt(cov_path, cov[np.ix_(perm, perm)])
    inputs = {
        "xi": path,
        "xi_integration": path,
        "cov": str(cov_path) if analytic else None,
        "cov_integration": str(cov_path) if analytic else None,
    }
    params = dict(
        version="v",
        blind="toy",
        fiducial_scale_cut=(edges[4], edges[-5]),
        min_sep=2.0,
        max_sep=60.0,
        nbins=3,
        nmodes=2,
        scale_cuts=[(edges[4], edges[-5])],
        integration=dict(min_sep=edges[0], max_sep=edges[-1], nbins=n),
    )
    if kind == "cosebis":
        params.update(min_sep=edges[0], max_sep=edges[-1], nbins=n)
    output = {
        "npz": str(tmp_path / "derived.npz"),
        "sacc": str(tmp_path / "derived.sacc"),
    }
    smk = SimpleNamespace(input=inputs, output=output, params=params)
    monkeypatch.syspath_prepend(str(SCRIPTS))
    script = "cv_pure_eb.py" if kind == "pure" else "cv_cosebis.py"
    runpy.run_path(str(SCRIPTS / script), init_globals={"snakemake": smk})
    out = sacc_io.load(output["sacc"], blind="toy")
    assert sacc_io.stamp(out) == sacc_io.stamp(part) == "toy"
    operators = []
    for pair in pairs:
        gg = sacc_io.xi_correlation(part, pair, grid="integration")
        if kind == "pure":
            op = b_modes.pure_eb_operator(gg.weight, edges, np.geomspace(2, 60, 4))[0]
        else:
            op = b_modes.cosebis_operator(
                theta, edges[:-1], edges[1:], params["scale_cuts"][0], 2
            )
        operators.append(op)
    expected = np.zeros_like(out.covariance.dense)
    for a, pa in enumerate(pairs):
        ra = [
            i
            for i, dp in enumerate(out.data)
            if tuple(int(t.removeprefix("source_")) for t in dp.tracers) == pa
        ]
        for b, pb in enumerate(pairs):
            rb = [
                i
                for i, dp in enumerate(out.data)
                if tuple(int(t.removeprefix("source_")) for t in dp.tracers) == pb
            ]
            source = cov[a * 2 * n : (a + 1) * 2 * n, b * 2 * n : (b + 1) * 2 * n]
            block = operators[a] @ source @ operators[b].T
            if kind == "cosebis":
                block[:2, 2:] = 0
                block[2:, :2] = 0
            expected[np.ix_(ra, rb)] = block
    # Block-wise and full-matrix products round differently across BLAS builds;
    # a misplaced block would differ at order unity.
    np.testing.assert_allclose(out.covariance.dense, expected, rtol=1e-10, atol=1e-23)


@pytest.mark.parametrize("kind", ["pure", "cosebis"])
def test_single_pair_workflow_preserves_existing_results(
    synthetic, tmp_path, monkeypatch, kind
):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import cv_runner

    monkeypatch.setattr(cv_runner, "_unbuffer_streams", lambda: None)
    monkeypatch.setattr(cv_runner, "verify_outputs", lambda _: None)
    path, _, theta, edges, cov = synthetic
    part = sacc_io.load(path, blind="toy")
    gg = sacc_io.xi_correlation(part, (0, 0), grid="integration")
    n = len(theta)
    single_cov = cov[: 2 * n, : 2 * n]
    single = sacc_writers.xi_pairs_to_sacc(
        {0: sacc_io.get_nz(part, 0)},
        part.metadata,
        [
            (
                (0, 0),
                dict(
                    theta=gg.meanr,
                    theta_nom=gg.rnom,
                    xip=gg.xip,
                    xim=gg.xim,
                    weight=gg.weight,
                    npairs=gg.npairs,
                ),
            )
        ],
        grid="integration",
        covariance=single_cov,
    )
    single_path = tmp_path / "single.sacc"
    sacc_io.save(single, single_path, derived_from=[part])
    cov_path = tmp_path / "single.txt"
    np.savetxt(cov_path, single_cov)
    cuts = [(edges[4], edges[-5])]
    params = dict(
        version="v",
        blind="toy",
        fiducial_scale_cut=cuts[0],
        min_sep=2.0,
        max_sep=60.0,
        nbins=3,
        nmodes=2,
        scale_cuts=cuts,
        integration=dict(min_sep=edges[0], max_sep=edges[-1], nbins=n),
    )
    if kind == "cosebis":
        params.update(min_sep=edges[0], max_sep=edges[-1], nbins=n)
        result = b_modes.cosebis_scan_from_xi(
            gg.meanr,
            gg.xip,
            gg.xim,
            single_cov,
            *b_modes.log_bin_edges(edges[0], edges[-1], n),
            nmodes=2,
            scale_cuts=cuts,
        )[cuts[0]]
        expected_mean = np.r_[result["En"], result["Bn"]]
        script = "cv_cosebis.py"
    else:
        result = b_modes.calculate_pure_eb_correlation(
            gg.meanr,
            gg.xip,
            gg.xim,
            gg.weight,
            edges,
            single_cov,
            np.geomspace(2, 60, 4),
        )
        expected_mean = np.concatenate([result[key] for key in sacc_io.PURE_KEYS])
        script = "cv_pure_eb.py"
    smk = SimpleNamespace(
        params=params,
        input=dict(
            xi=single_path,
            xi_integration=single_path,
            cov=cov_path,
            cov_integration=cov_path,
        ),
        output=dict(
            npz=str(tmp_path / "single.npz"), sacc=str(tmp_path / "single-derived.sacc")
        ),
    )
    runpy.run_path(str(SCRIPTS / script), init_globals={"snakemake": smk})
    out = sacc_io.load(smk.output["sacc"], blind="toy")
    np.testing.assert_array_equal(out.mean, expected_mean)
    np.testing.assert_array_equal(out.covariance.dense, result["cov"])


@pytest.mark.parametrize("kind", ["pure", "cosebis"])
@pytest.mark.parametrize("analytic", [False, True])
def test_tomographic_mixins_take_each_pair_covariance(
    synthetic, tmp_path, monkeypatch, kind, analytic
):
    from sp_validation.cosmo_val import cosebis, pure_eb

    path, pairs, theta, edges, cov = synthetic
    part = sacc_io.load(path, blind="toy")
    n = len(theta)
    order = [
        k * 2 * n + t * n + q
        for t in range(2)
        for k in range(len(pairs))
        for q in range(n)
    ]
    cov_path = tmp_path / "tomo_cov.txt"
    np.savetxt(cov_path, cov[np.ix_(order, order)])
    views = {
        str(pair): sacc_io.xi_correlation(part, pair, grid="integration")
        for pair in pairs
    }
    cv = SimpleNamespace(
        print_start=lambda *args: None,
        calculate_2pcf_version=lambda *args, **kwargs: views,
        xi_tomo_parts={("v", "integration"): part},
        _binning=lambda a, b, c: dict(min_sep=a, max_sep=b, nbins=c),
    )
    seen = []
    if kind == "cosebis":

        def fake(gg, *, cov_xi, npatch, **kwargs):
            seen.append((cov_xi, npatch))
            return {(1, 2): {}}

        monkeypatch.setattr(cosebis, "calculate_cosebis", fake)
        cosebis.CosebisMixin.calculate_cosebis(
            cv,
            "v",
            compute_tomography=True,
            cov_path=str(cov_path) if analytic else None,
        )
    else:

        def fake(*args, npatch):
            seen.append((args[5], npatch))
            return {}

        monkeypatch.setattr(pure_eb, "calculate_pure_eb_correlation", fake)
        pure_eb.PureEBMixin.calculate_pure_eb(
            cv,
            "v",
            min_sep=2,
            max_sep=60,
            nbins=3,
            min_sep_int=edges[0],
            max_sep_int=edges[-1],
            nbins_int=n,
            compute_tomography=True,
            cov_path_int=str(cov_path) if analytic else None,
        )
    assert len(seen) == len(pairs)
    for i, (block, npatch) in enumerate(seen):
        np.testing.assert_array_equal(
            block, cov[i * 2 * n : (i + 1) * 2 * n, i * 2 * n : (i + 1) * 2 * n]
        )
        assert npatch == (None if analytic else 100)
