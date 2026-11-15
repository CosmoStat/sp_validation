"""The metacal response core (``sp_validation.response``), against its definitions.

Small hand-computable cases, additivity over chunks and groups, the table's
pooling rule, the jackknife against brute-force deletion, the self-weighting
bias of a self-fitted table, the recovery of a known shear from synthetic
simulations, and the in-memory catalogue calibration.
"""

from itertools import product

import numpy as np
import pytest
from astropy.io import fits

from sp_validation import response
from sp_validation.response import (
    BRANCHES,
    ESTIMATES,
    SUFFIX,
    Cuts,
    Table,
    accumulate,
    estimate,
    fit_table,
    jackknife,
    refit,
    shear_bias,
    shear_bias_jackknife,
    unit_table,
)

STEP = 0.01
SHIFT = dict(ns=(0, 0), p1=(STEP, 0), m1=(-STEP, 0), p2=(0, STEP), m2=(0, -STEP))
CUTS = Cuts(10.0, 500.0, 0.5, 3.0, True)
SIGMA_E = 0.34


def rows_from(g, snr, size, flags=None, errors=None, extra=None):
    """Catalogue columns from per-branch shapes ``g[b]`` (2, n), ``snr[b]``,
    ``size[b]`` and optionally ``flags[b]`` and ``errors[b]`` (2, n)."""
    rows = {}
    for b in BRANCHES:
        s = SUFFIX[b]
        n = len(snr[b])
        rows[f"NGMIX_G1_{s}"], rows[f"NGMIX_G2_{s}"] = np.asarray(g[b], float)
        rows[f"NGMIX_FLUX_{s}"] = np.asarray(snr[b], float)
        rows[f"NGMIX_FLUX_ERR_{s}"] = np.ones(n)
        rows[f"NGMIX_T_{s}"] = np.asarray(size[b], float)
        rows[f"NGMIX_T_ERR_{s}"] = np.zeros(n)
        rows[f"NGMIX_T_PSF_RECONV_{s}"] = np.ones(n)
        rows[f"NGMIX_FLAGS_{s}"] = np.zeros(n) if flags is None else flags[b]
        if errors is not None:
            rows[f"NGMIX_G1_ERR_{s}"], rows[f"NGMIX_G2_ERR_{s}"] = errors[b]
    rows.update(extra or {})
    return rows


def synthetic(n, shear=(0.0, 0.0), seed=0, noise_seed=None, tiles=6):
    """A metacal catalogue of ``n`` galaxies sheared by ``shear``.

    Galaxies (from ``seed``) carry an intrinsic shape, a per-object response
    matrix, and an S/N and size that move with the applied shear ``s`` through
    ``e_int . s``, so both the cuts and the table weights respond to shear.
    Every branch applies ``shear`` plus its metacal step. Measurement noise
    (from ``noise_seed``) is shared by the branches. A few rows have nonfinite
    or degenerate coordinates, and a few are flagged.
    """
    rng = np.random.default_rng(seed)
    e_int = rng.normal(0, 0.25, (2, n))
    e_int /= np.maximum(1, np.hypot(*e_int) / 0.9)
    diag = rng.uniform(0.5, 1.0, n)
    matrix = np.array(
        [[diag, 0.05 * rng.normal(size=n)], [0.03 * rng.normal(size=n), 0.9 * diag]]
    )
    snr0 = np.exp(rng.normal(np.log(30), 0.9, n))
    size0 = np.exp(rng.normal(np.log(1.1), 0.45, n))
    tpsf = rng.uniform(0.8, 1.2, n)
    tile = rng.integers(0, tiles, n)
    flagged = rng.random(n) < 0.02
    noise_rng = np.random.default_rng(seed + 1 if noise_seed is None else noise_seed)
    noise = noise_rng.normal(0, 1, (2, n)) * 2 / snr0
    names = ("G1", "G2", "G1_ERR", "G2_ERR", "FLUX", "FLUX_ERR", "T", "T_ERR")
    fields = [
        (f"NGMIX_{k}_{SUFFIX[b]}", "f8")
        for b in BRANCHES
        for k in (*names, "T_PSF_RECONV", "FLAGS")
    ]
    data = np.zeros(n, dtype=fields + [("TILE_ID", "U8"), ("RA", "f8"), ("Dec", "f8")])
    for b in BRANCHES:
        s = np.add(shear, SHIFT[b])
        e = e_int + np.einsum("ijn,j->in", matrix, s) + noise
        projection = e_int[0] * s[0] + e_int[1] * s[1]
        snr = snr0 * (1 + 6 * projection)
        size = size0 * (1 + 4 * projection)
        x = SUFFIX[b]
        data[f"NGMIX_G1_{x}"], data[f"NGMIX_G2_{x}"] = e
        data[f"NGMIX_G1_ERR_{x}"] = data[f"NGMIX_G2_ERR_{x}"] = 1.5 / snr
        data[f"NGMIX_FLUX_{x}"] = 2 * snr
        data[f"NGMIX_FLUX_ERR_{x}"] = 2
        data[f"NGMIX_T_{x}"] = size * tpsf
        data[f"NGMIX_T_PSF_RECONV_{x}"] = tpsf
        data[f"NGMIX_FLAGS_{x}"] = flagged
    # Nonfinite and degenerate coordinates, on every branch and on one only.
    for b in BRANCHES:
        x = SUFFIX[b]
        data[f"NGMIX_FLUX_{x}"][:3] = np.nan
        data[f"NGMIX_T_PSF_RECONV_{x}"][3:5] = 0
        data[f"NGMIX_T_{x}"][5:7] = np.nan
    data["NGMIX_FLUX_1P"][7:9] = np.inf
    data["NGMIX_T_2M"][9:11] = np.nan
    data["TILE_ID"] = [f"t{t:02d}" for t in tile]
    data["RA"] = rng.uniform(0, 10, n)
    data["Dec"] = rng.uniform(0, 10, n)
    return data


def relative(new, old):
    """Largest absolute difference relative to the largest reference value."""
    new, old = np.asarray(new, float), np.asarray(old, float)
    return np.max(np.abs(new - old)) / np.max(np.abs(old))


# Behaviour of the core, from its definitions


def hand_case():
    """Four objects, a two-cell table, every branch in its own cells."""
    e = np.array([[0.2, -0.1, 0.4, -0.2], [0.1, 0.3, -0.2, 0.2]])
    matrix = np.array([[0.8, 0.1], [-0.04, 0.9]])
    g = {}
    for b in BRANCHES:
        g[b] = e + (matrix @ np.array(SHIFT[b]))[:, None]
    snr = dict(
        ns=[20, 20, 100, 100],
        p1=[100, 20, 100, 20],
        m1=[20, 100, 20, 100],
        p2=[100, 100, 20, 100],
        m2=[20, 20, 100, 20],
    )
    size = {b: np.ones(4) for b in BRANCHES}
    flags = {b: np.zeros(4) for b in BRANCHES}
    flags["m1"][3] = 1
    errors = {b: np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.1, 0.3]]) for b in g}
    errors["p2"] = errors["p2"][:, ::-1]
    rows = rows_from(g, snr, size, flags, errors)
    edges = (np.array([10.0, 50.0, 500.0]), np.array([0.5, 3.0]))
    table = Table(*edges, np.array([[1.0], [4.0]]), np.zeros((2, 1), int), 0)
    cuts = Cuts(10.0, 500.0, 0.5, 3.0, False)
    return rows, g, snr, flags, errors, matrix, edges, table, cuts


@pytest.mark.parametrize("sigma_e", [None, 0.3])
def test_estimate_by_hand(sigma_e):
    rows, g, snr, flags, errors, matrix, edges, table, cuts = hand_case()
    result = estimate(accumulate(rows, cuts, edges, sigma_e=sigma_e), table)

    def mean(x, b):
        keep = flags[b] == 0
        w = np.where(np.array(snr[b]) < 50, 1.0, 4.0)
        if sigma_e is not None:
            w = w / (2 * sigma_e**2 + errors[b][0] ** 2 + errors[b][1] ** 2)
        return np.average(x[:, keep], axis=1, weights=w[keep])

    expected_selection = np.empty((2, 2))
    expected_joint = np.empty((2, 2))
    for j, (p, m) in enumerate(response.PAIRS):
        expected_selection[:, j] = (mean(g["ns"], p) - mean(g["ns"], m)) / 0.02
        expected_joint[:, j] = (mean(g[p], p) - mean(g[m], m)) / 0.02
    np.testing.assert_allclose(result["shear"], matrix, rtol=1e-12)
    np.testing.assert_allclose(result["selection"], expected_selection, rtol=1e-12)
    np.testing.assert_allclose(result["joint"], expected_joint, rtol=1e-12)
    np.testing.assert_allclose(result["total"], matrix + expected_selection)
    np.testing.assert_allclose(result["c_e"], mean(g["ns"], "ns"), rtol=1e-12)
    np.testing.assert_allclose(
        result["c_g"], np.linalg.solve(result["total"], result["c_e"])
    )
    assert result["counts"] == dict(ns=4, p1=4, m1=3, p2=4, m2=4)
    # Pure cell migration moves every element of the selection response.
    assert np.all(np.abs(expected_selection) > 1)


def test_unit_table_and_weights():
    rows, g, snr, flags, errors, matrix, edges, table, cuts = hand_case()
    unit = estimate(accumulate(rows, cuts, edges), unit_table(edges))
    np.testing.assert_allclose(unit["c_e"], g["ns"].mean(axis=1))
    sums = accumulate(rows, cuts, edges, sigma_e=0.3)
    result = estimate(sums, table)
    columns = response.branch_columns(rows, "p2")
    w = response.weights(table, columns, sigma_e=0.3)
    assert w.sum() == pytest.approx(result["weight_sums"]["p2"], rel=1e-14)
    np.testing.assert_allclose(
        table.evaluate([5.0, 50.0, 50.1, 1e4, np.nan], [1, 1, 1, 9, 1]),
        [1, 1, 4, 4, 0],
    )
    # Calibrated no-shear shapes have zero weighted mean.
    ns = response.branch_columns(rows, "ns")
    shapes = response.calibrate(np.array([ns["g1"], ns["g2"]]), result)
    np.testing.assert_allclose(
        np.average(shapes, axis=1, weights=response.weights(table, ns, 0.3)),
        0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        response.calibrate(np.array([ns["g1"], ns["g2"]]), result, additive=False)
        - shapes,
        np.repeat(result["c_g"][:, None], 4, axis=1),
    )


def test_sums_add_over_chunks_and_groups():
    data = synthetic(3000)
    edges = response.table_edges(CUTS)
    pre = np.random.default_rng(3).random(len(data)) < 0.9
    labels = np.unique(data["TILE_ID"])
    whole = accumulate(data, CUTS, edges, preselection=pre, groups=data["TILE_ID"])
    cuts = [0, 7, 1000, 1001, 2500, len(data)]
    chunks = sum(
        accumulate(
            data[a:b],
            CUTS,
            edges,
            preselection=pre[a:b],
            groups=data["TILE_ID"][a:b],
            labels=labels,
        )
        for a, b in zip(cuts[:-1], cuts[1:], strict=True)
    )
    ungrouped = accumulate(data, CUTS, edges, preselection=pre)
    assert ungrouped.labels is None
    for key, value in whole.arrays.items():
        assert value.shape[-2] == len(labels)
        np.testing.assert_allclose(chunks.arrays[key], value, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(
            whole.total().arrays[key], ungrouped.arrays[key], rtol=1e-12, atol=1e-15
        )
    np.testing.assert_array_equal(chunks.arrays["count"], whole.arrays["count"])
    # Rows with nonfinite coordinates on a branch are left out of it alone.
    counts = estimate(ungrouped, unit_table(edges))["counts"]
    assert counts["p1"] < counts["ns"]
    with pytest.raises(ValueError, match="labels lacks"):
        accumulate(data, CUTS, edges, groups=data["TILE_ID"], labels=labels[1:])


def sparse_moments():
    count = np.array([[40, 0], [2, 0]])
    rsum = 0.8 * count.astype(float)
    esum = np.array([[40 * 0.04, 0], [2 * 1e-6, 0]])
    return count, rsum, esum


def test_pool_sparse_cell_takes_its_window_moments():
    count, rsum, esum = sparse_moments()
    raw = response.pool(count, rsum, esum, 0)
    pooled = response.pool(count, rsum, esum, 3)
    assert raw[1, 0] == pytest.approx(0.8**2 / 1e-6)
    expected = (0.8 * 42) ** 2 / (42 * (40 * 0.04 + 2e-6))
    np.testing.assert_allclose(pooled[[1, 0, 1], [0, 1, 1]], expected, rtol=1e-14)
    assert pooled[0, 0] == raw[0, 0]
    # Above the total, every cell takes the whole table's moments.
    np.testing.assert_allclose(response.pool(count, rsum, esum, 100), expected)


def test_pool_window_is_clipped_square():
    count = np.array([[1, 1, 50], [1, 1, 50], [50, 50, 50]])
    rsum = np.arange(9.0).reshape(3, 3) + count
    esum = 0.1 * count + 0.01 * np.arange(9).reshape(3, 3)
    pooled = response.pool(count, rsum, esum, 3)
    corner = np.s_[:2, :2]
    assert pooled[0, 0] == pytest.approx(
        rsum[corner].sum() ** 2 / (count[corner].sum() * esum[corner].sum())
    )
    middle = np.s_[0:3, 0:3]
    assert pooled[1, 1] == pytest.approx(
        rsum[middle].sum() ** 2 / (count[middle].sum() * esum[middle].sum())
    )
    assert pooled[2, 2] == rsum[2, 2] ** 2 / (count[2, 2] * esum[2, 2])


def test_table_fit_and_product_roundtrip(tmp_path):
    data = synthetic(2000)
    sums = accumulate(data, CUTS, response.table_edges(CUTS, 5))
    table = fit_table(sums, 3)
    assert table.count.sum() == estimate(sums, table)["counts"]["ns"]
    path = tmp_path / "table.fits"
    response.write_table(path, table)
    read = response.read_table(path)
    for a, b in zip(read, table, strict=True):
        np.testing.assert_array_equal(a, b)
    with fits.open(path, mode="update") as hdus:
        del hdus[0].header["WSIZDEF"]
    with pytest.raises(ValueError, match="coordinates"):
        response.read_table(path)


@pytest.mark.parametrize("rule", ["fixed", "refit", "iv"])
def test_jackknife_is_brute_force_deletion(rule):
    data = synthetic(3000)
    edges = response.table_edges(CUTS, 8)
    sigma_e = SIGMA_E if rule == "iv" else None
    sums = accumulate(data, CUTS, edges, groups=data["TILE_ID"], sigma_e=sigma_e)
    table = dict(fixed=fit_table(sums, 3), refit=refit(10), iv=unit_table(edges))[rule]
    labels, deleted, cov = jackknife(sums, table)
    vectors = []
    for k, label in enumerate(labels):
        kept = accumulate(
            data, CUTS, edges, preselection=data["TILE_ID"] != label, sigma_e=sigma_e
        )
        brute = estimate(kept, table)
        for key in ESTIMATES:
            np.testing.assert_allclose(deleted[key][k], brute[key], rtol=1e-10)
        vectors.append(np.concatenate([np.ravel(brute[key]) for key in ESTIMATES]))
    vectors = np.array(vectors)
    delta = vectors - vectors.mean(axis=0)
    expected = (len(labels) - 1) / len(labels) * delta.T @ delta
    np.testing.assert_allclose(cov, expected, rtol=1e-8, atol=1e-20)
    assert cov.shape == (20, 20)


@pytest.mark.parametrize("occupancy", [3, 4, 8])
def test_self_fitted_table_bias_is_inverse_cell_occupancy(occupancy):
    """A table fitted on the measured shapes suppresses their shear: m = -1/N.

    Every Rademacher sample of ``occupancy`` signs is one cell; each appears
    twice, so cells carry independent training signs. A table fitted on other
    shapes leaves the shear intact.
    """
    signs = np.array(list(product((-1.0, 1.0), repeat=occupancy)))
    e1 = (np.repeat(signs, 2, axis=0) * 0.25).ravel()
    independent = np.repeat(np.tile([-0.25, 0.25], len(signs)), occupancy)
    cells = len(e1) // occupancy
    snr = np.repeat(np.arange(cells) + 0.5, occupancy)
    edges = (np.arange(cells + 1.0), np.array([0.0, 2.0]))
    cuts = Cuts(0.0, cells + 1.0, 0.0, 2.0, False)
    shear = 1e-5

    def sums(x, sheared):
        e = np.array([x + sheared, np.full_like(x, 0.25)])
        g = {b: e + np.array(SHIFT[b])[:, None] for b in BRANCHES}
        rows = rows_from(g, {b: snr for b in g}, {b: np.ones(len(x)) for b in g})
        return accumulate(rows, cuts, edges)

    pairs = {0: (sums(e1, shear), sums(e1, -shear))}
    self_fitted = shear_bias(pairs, shear, refit(3))["m1"]
    independent_table = fit_table(sums(independent, 0), 3)
    other = shear_bias(pairs, shear, independent_table)["m1"]
    assert self_fitted == pytest.approx(-1 / occupancy, abs=3e-9)
    assert other == pytest.approx(0, abs=3e-9)


def simulations(sigma_e, n=200_000, g=0.02, tiles=16):
    """+-g simulations of shared galaxies and noise, grouped by shared tiles."""
    edges = response.table_edges(CUTS)
    labels = np.array([f"t{t:02d}" for t in range(tiles)])
    pairs = {}
    for j in range(2):
        sims = []
        for sign in (1, -1):
            shear = np.zeros(2)
            shear[j] = sign * g
            data = synthetic(n, shear, seed=7 + j, tiles=tiles)
            sims.append(
                accumulate(
                    data,
                    CUTS,
                    edges,
                    groups=data["TILE_ID"],
                    labels=labels,
                    sigma_e=sigma_e,
                )
            )
        pairs[j] = tuple(sims)
    return pairs, edges


@pytest.mark.slow
@pytest.mark.parametrize("sigma_e", [None, SIGMA_E])
def test_known_shear_is_recovered(sigma_e):
    """With a table independent of the simulations, m is consistent with zero.

    Shear moves S/N and size, so the cuts and the weights respond: calibrating
    with the shear response alone leaves a significant m, the total response
    removes it.
    """
    pairs, edges = simulations(sigma_e)
    if sigma_e is None:
        table = fit_table(accumulate(synthetic(200_000, seed=100), CUTS, edges), 3)
    else:
        table = unit_table(edges)
    for calibration, consistent in (("total", True), ("shear", False)):
        m = shear_bias(pairs, 0.02, table, calibration)
        _, _, cov = shear_bias_jackknife(pairs, 0.02, table, calibration)
        error = dict(zip(m, np.sqrt(np.diag(cov)), strict=True))
        for key in ("m1", "m2", "c1", "c2"):
            pull = abs(m[key] / error[key])
            assert (pull < 3) if consistent or key[0] == "c" else (pull > 5), (
                calibration,
                key,
                m[key],
                error[key],
            )


def test_sums_save_and_load(tmp_path):
    data = synthetic(500)
    edges = response.table_edges(CUTS, 4)
    for sums in (
        accumulate(data, CUTS, edges, groups=data["TILE_ID"]),
        accumulate(data, CUTS, edges, sigma_e=SIGMA_E),
    ):
        sums.save(tmp_path / "sums.npz")
        loaded = response.Sums.load(tmp_path / "sums.npz")
        assert loaded.step == sums.step and loaded.sigma_e == sums.sigma_e
        np.testing.assert_array_equal(
            loaded.labels if sums.labels is not None else [],
            sums.labels if sums.labels is not None else [],
        )
        for key, value in sums.arrays.items():
            np.testing.assert_array_equal(loaded.arrays[key], value)
        loaded + sums  # same groups, cells, step and weights


def metacal_block(**overrides):
    block = dict(
        weight_scheme="des",
        gal_snr_min=CUTS.snr_min,
        gal_snr_max=CUTS.snr_max,
        gal_rel_size_min=CUTS.size_min,
        gal_rel_size_max=CUTS.size_max,
        gal_size_corr_ell=CUTS.size_corr_ell,
        sigma_eps_prior=SIGMA_E,
        des_num_bins=8,
        des_min_count=3,
    )
    return {**block, **overrides}


@pytest.mark.parametrize("scheme", ["des", "iv", "external"])
def test_calibrate_catalogue(tmp_path, scheme):
    """Per-object weights and shapes are those the response was measured with."""
    data = synthetic(3000)
    pre = np.random.default_rng(4).random(len(data)) < 0.9
    edges = response.table_edges(CUTS, 8)
    if scheme == "external":
        table = fit_table(accumulate(synthetic(3000, seed=9), CUTS, edges), 3)
        response.write_table(tmp_path / "table.fits", table)
        metacal = metacal_block(weight_table=str(tmp_path / "table.fits"))
    else:
        metacal = metacal_block(weight_scheme=scheme)
    cal = response.calibrate_catalogue(
        data, metacal, preselection=pre, groups=data["TILE_ID"]
    )
    result = cal["result"]
    assert pre[cal["index"]].all()
    assert len(cal["index"]) == result["counts"]["ns"]
    assert cal["w"].sum() == pytest.approx(result["weight_sums"]["ns"], rel=1e-12)
    np.testing.assert_allclose(
        np.average(cal["g"], axis=1, weights=cal["w"]), result["c_e"], rtol=1e-12
    )
    np.testing.assert_allclose(
        np.average(cal["e"], axis=1, weights=cal["w"]), 0, atol=1e-15
    )
    np.testing.assert_allclose(
        np.average(cal["R_g"], axis=2, weights=cal["w"]), result["shear"], rtol=1e-12
    )
    labels, deleted, cov = cal["jackknife"]
    np.testing.assert_array_equal(labels, np.unique(data["TILE_ID"][pre]))
    if scheme == "external":
        np.testing.assert_array_equal(cal["table"].w, table.w)
    # A self-fitted table is refitted on every deletion.
    rule = refit(3) if scheme == "des" else cal["table"]
    np.testing.assert_allclose(cov, jackknife(cal["sums"], rule)[2])
    record = response.record(result, cov)
    assert record["R"] == result["total"].tolist()
    assert record["covariance_order"] == list(ESTIMATES)
    sim = response.calibrate_catalogue(
        data, dict(metacal, additive_correction=False), preselection=pre
    )
    np.testing.assert_allclose(
        np.average(sim["e"], axis=1, weights=sim["w"]), result["c_g"], rtol=1e-10
    )


def test_calibrate_catalogue_needs_a_weight_scheme():
    metacal = metacal_block()
    del metacal["weight_scheme"]
    with pytest.raises(ValueError, match="weight_scheme"):
        response.calibrate_catalogue(synthetic(100), metacal)
