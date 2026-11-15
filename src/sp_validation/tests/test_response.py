"""The metacal response core (``sp_validation.response``), against its definitions.

The first part tests the core from definitions: small hand-computable cases,
additivity over chunks and groups, the table's pooling rule, the jackknife
against brute-force deletion, the self-weighting bias of a self-fitted table
and the recovery of a known shear from synthetic simulations.

The second part (``Equivalence with the row-array implementations``) compares
the core with ``response.responses``/``row_jackknife``, ``calibration.metacal``,
``des_jackknife.DESJackknife``, ``image_sims.ImageSimMBias`` and the per-cell
sums of the calibrated-sample build, on synthetic catalogues.
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


# Equivalence with the row-array implementations


def old_metacal(data, pre, scheme, min_count=3, product=None):
    from sp_validation import calibration

    return calibration.metacal(
        data,
        pre,
        snr_min=CUTS.snr_min,
        snr_max=CUTS.snr_max,
        rel_size_min=CUTS.size_min,
        rel_size_max=CUTS.size_max,
        size_corr_ell=CUTS.size_corr_ell,
        sigma_eps=SIGMA_E,
        weight_scheme=scheme,
        des_num_bins=20,
        des_min_count=min_count,
        des_weight_product=product,
    )


def preselected(n=4000, seed=0):
    data = synthetic(n, seed=seed)
    pre = np.random.default_rng(seed + 5).random(n) < 0.95
    return data, pre


def assert_estimates(new, old, rtol=1e-12):
    for key in ESTIMATES:
        difference = relative(new[key], old[key])
        print(f"  {key}: {difference:.1e}")
        assert difference < rtol, key


@pytest.mark.parametrize("scheme", ["des", "iv"])
def test_metacal_responses_and_fixed_jackknife(scheme):
    """``metacal`` (``response.responses``) and ``row_jackknife``, fixed weights."""
    data, pre = preselected()
    gm = old_metacal(data, pre, scheme)
    groups = data["TILE_ID"][pre]
    sigma_e = SIGMA_E if scheme == "iv" else None
    sums = accumulate(
        data,
        CUTS,
        response.table_edges(CUTS),
        preselection=pre,
        groups=data["TILE_ID"],
        sigma_e=sigma_e,
    )
    if scheme == "des":
        table = fit_table(sums, 3)
        np.testing.assert_array_equal(table.count, gm.W[3])
        print(f"\ntable: {relative(table.w, gm.W[2]):.1e}")
        assert relative(table.w, gm.W[2]) < 1e-12
    else:
        table = unit_table(sums.edges)
    result = estimate(sums, table)
    old = dict(
        shear=gm.R_shear_global,
        selection=gm.R_selection,
        total=gm.R,
        joint=gm.R_joint,
        c_e=gm.c_e,
        c_g=gm.c_g,
    )
    print(f"\n{scheme} point")
    assert_estimates(result, old)
    assert result["counts"] == gm.branch_counts
    for b in BRANCHES:
        assert relative(result["weight_sums"][b], gm.branch_weight_sums[b]) < 1e-12
    old_labels, old_deleted, old_cov = gm.response_jackknife(groups)
    labels, deleted, cov = jackknife(sums, table)
    np.testing.assert_array_equal(labels, old_labels)
    print(f"{scheme} fixed jackknife")
    assert_estimates(deleted, old_deleted)
    print(f"  covariance: {relative(cov, old_cov):.1e}")
    assert relative(cov, old_cov) < 1e-10


@pytest.mark.parametrize("min_count", [0, 3, 30, 10_000])
def test_metacal_refit_jackknife(min_count):
    """``response_jackknife(refit_w=True)``: ``DESJackknife`` and the K rescans."""
    data, pre = preselected()
    gm = old_metacal(data, pre, "des", min_count)
    groups = data["TILE_ID"][pre]
    sums = accumulate(
        data,
        CUTS,
        response.table_edges(CUTS),
        preselection=pre,
        groups=data["TILE_ID"],
    )
    labels, deleted, cov = jackknife(sums, refit(min_count))
    for reference in (False, True):
        old_labels, old_deleted, old_cov = gm.response_jackknife(
            groups, refit_w=True, reference=reference
        )
        np.testing.assert_array_equal(labels, old_labels)
        print(f"\nrefit jackknife, min_count={min_count}, reference={reference}")
        assert_estimates(deleted, old_deleted)
        print(f"  covariance: {relative(cov, old_cov):.1e}")
        assert relative(cov, old_cov) < 1e-10


def test_pooling_matches_both_old_rules():
    """``pool`` equals ``response.fit_w``'s loop and ``des_jackknife``'s integral
    image on random sparse grids."""
    from sp_validation.des_jackknife import pooled_weight

    rng = np.random.default_rng(1)
    worst = [0, 0]
    for _ in range(200):
        shape = tuple(rng.integers(1, 21, 2))
        dense = rng.random(shape) < rng.uniform(0, 1)
        count = np.where(dense, rng.integers(0, 2000, shape), rng.integers(0, 4, shape))
        cell = np.repeat(np.arange(count.size), count.ravel())
        snr, size = np.unravel_index(cell, shape)
        e = rng.normal(0, 0.3, (2, len(cell)))
        r = rng.uniform(0.3, 1.0, len(cell))
        edges = np.arange(shape[0] + 1.0), np.arange(shape[1] + 1.0)
        min_count = int(rng.choice([0, 1, 3, 30, 10**4, 10**8]))
        loop = response.fit_w(snr + 0.5, size + 0.5, e, r, *edges, min_count)[2]
        moments = np.array(
            [
                np.bincount(cell, weights=x, minlength=count.size)
                for x in (np.ones(len(cell)), r, np.mean(e**2, axis=0))
            ]
        )
        new = response.pool(*moments.reshape(3, *shape), min_count)
        integral = pooled_weight(moments, shape, min_count).reshape(shape)
        if loop.any():
            worst[0] = max(worst[0], relative(new, loop))
            worst[1] = max(worst[1], relative(new, integral))
    print(f"\npool vs fit_w loop: {worst[0]:.1e}; vs integral image: {worst[1]:.1e}")
    assert worst[0] < 1e-12
    assert worst[1] < 1e-12


def calibrated_sample_sums(branches, masks, coordinates, rg, edges_s, edges_r):
    """``response.cell_sums`` of the calibrated-sample build (PR #490)."""
    n_cells = (len(edges_s) - 1) * (len(edges_r) - 1)
    sums = {"n": {}, "ns_e": {}, "own_e": {}}
    for b, idx in masks.items():
        snr, size = (x[idx] for x in coordinates[b])
        finite = np.isfinite(snr) & np.isfinite(size)
        idx = idx[finite]
        cell = response.cell_index(snr[finite], edges_s) * (len(edges_r) - 1)
        cell += response.cell_index(size[finite], edges_r)

        def total(values, cell=cell):
            return np.bincount(cell, weights=values, minlength=n_cells)

        sums["n"][b] = np.bincount(cell, minlength=n_cells)
        sums["ns_e"][b] = np.array([total(x[idx]) for x in branches["ns"]])
        sums["own_e"][b] = np.array([total(x[idx]) for x in branches[b]])
        if b == "ns":
            sums["rg"] = np.array(
                [[total(rg[i, j, finite]) for j in range(2)] for i in range(2)]
            )
    return sums


def calibrated_sample_responses(sums, table, step):
    """``response.responses_from_sums`` of the calibrated-sample build."""
    w = table[2].ravel()
    den = {b: n @ w for b, n in sums["n"].items()}
    shear = sums["rg"] @ w / den["ns"]
    selection = np.empty((2, 2))
    joint = np.empty((2, 2))
    for j, (p, m) in enumerate((("p1", "m1"), ("p2", "m2"))):
        for name, out in (("ns_e", selection), ("own_e", joint)):
            plus, minus = (sums[name][b] @ w / den[b] for b in (p, m))
            out[:, j] = (plus - minus) / (2 * step)
    total = shear + selection
    ce = sums["ns_e"]["ns"] @ w / den["ns"]
    return dict(
        shear=shear,
        selection=selection,
        total=total,
        joint=joint,
        c_e=ce,
        c_g=np.linalg.solve(total, ce),
        counts={b: int(n.sum()) for b, n in sums["n"].items()},
        weight_sums={b: float(d) for b, d in den.items()},
    )


def test_calibrated_sample_sums():
    """The per-cell sums that built the v2.0.0 calibrated sample (PR #490)."""
    data, pre = preselected()
    gm = old_metacal(data, pre, "des")
    coordinates = {
        b: (getattr(gm, b)["flux"] / getattr(gm, b)["flux_err"],) for b in BRANCHES
    }
    coordinates = {
        b: (c[0], getattr(gm, b)["T"] / getattr(gm, b)["Tpsf"])
        for b, c in coordinates.items()
    }
    old_sums = calibrated_sample_sums(
        gm.branches, gm.mask_dict, coordinates, gm.R_shear, *gm.W[:2]
    )
    old = calibrated_sample_responses(old_sums, gm.W, gm._step)
    result = estimate(
        accumulate(data, CUTS, gm.W[:2], preselection=pre), fit_table_like(gm)
    )
    print("\ncalibrated-sample sums")
    assert_estimates(result, old)
    assert result["counts"] == old["counts"]


def fit_table_like(gm):
    es, er, w, count = gm.W
    return Table(es, er, w, count, gm._des_min_count)


def write_simulation(directory, data, pre, scheme, min_count, product):
    """Calibrate one simulation as the image-simulation chain does."""
    from sp_validation import calibration

    gm = old_metacal(data, pre, scheme, min_count, product)
    shear, raw, w, selected, _, _ = calibration.get_calibrated_m_c(gm, False)
    tiles = data["TILE_ID"][pre]
    columns = dict(
        RA=data["RA"][pre][selected],
        Dec=data["Dec"][pre][selected],
        e1=shear[0],
        e2=shear[1],
        e1_uncal=raw[0],
        e2_uncal=raw[1],
        **{f"w_{scheme}": w},
    )
    directory.mkdir()
    path = directory / "cat.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU.from_columns(
                [fits.Column(name=k, array=v, format="D") for k, v in columns.items()]
                + [fits.Column(name="TILE_ID", array=tiles[selected], format="8A")]
            ),
        ]
    ).writeto(path)
    calibration.save_response_product(path, gm, tiles)


@pytest.mark.parametrize("weights", ["self", "external", "iv"])
def test_image_sim_bias(tmp_path, weights):
    """``ImageSimMBias`` on catalogues calibrated by ``metacal`` and its R_JK."""
    from sp_validation.image_sims import ImageSimMBias

    g_in, n, min_count = 0.02, 6000, 3
    scheme = "iv" if weights == "iv" else "des"
    product, table = None, None
    edges = response.table_edges(CUTS)
    if weights == "external":
        table = fit_table(accumulate(synthetic(n, seed=100), CUTS, edges), min_count)
        product = dict(name="independent", path=str(tmp_path / "table.fits"))
        response.write_table(product["path"], table)
    names = dict(zip(range(2), (("1p2z", "1m2z"), ("1z2p", "1z2m")), strict=True))
    pairs = {}
    for j, sims in names.items():
        pairs[j] = []
        for name, sign in zip(sims, (1, -1), strict=True):
            shear = np.zeros(2)
            shear[j] = sign * g_in
            data, pre = synthetic(n, shear, seed=7 + j), np.ones(n, bool)
            pre[::17] = False
            write_simulation(
                tmp_path / f"{name}_grid_1", data, pre, scheme, min_count, product
            )
            pairs[j].append(
                accumulate(
                    data,
                    CUTS,
                    edges,
                    preselection=pre,
                    groups=data["TILE_ID"],
                    labels=np.unique(data["TILE_ID"]),
                    sigma_e=SIGMA_E if scheme == "iv" else None,
                )
            )
    estimator = ImageSimMBias(
        dict(
            grids_dir=str(tmp_path),
            num=1,
            catalog_name="cat.fits",
            shear_amplitude=g_in,
            match_radius_deg=0.0002,
            pair_match=False,
            w_cols=[f"w_{scheme}"],
            n_bootstrap=2,
            bootstrap_seed=1,
            branches=[name for sims in names.values() for name in sims],
        )
    )
    estimator.load_catalogs(False)
    old = estimator.run(False)
    rule = dict(self=refit(min_count), external=table, iv=unit_table(edges))[weights]
    new = shear_bias(pairs, g_in, rule)
    labels, deleted, cov = shear_bias_jackknife(pairs, g_in, rule)
    assert list(new) == old["covariance_order"]
    np.testing.assert_array_equal(labels, old["jackknife_tiles"])
    print(f"\nImageSimMBias, {weights}")
    for key in new:
        print(f"  {key}: {relative(new[key], old[key]):.1e}")
        assert relative(new[key], old[key]) < 1e-12
    draws = np.array(list(deleted.values())).T
    print(f"  jackknife: {relative(draws, old['mc_jackknife']):.1e}")
    print(f"  covariance: {relative(cov, old['mc_covariance']):.1e}")
    assert relative(draws, old["mc_jackknife"]) < 1e-12
    assert relative(cov, old["mc_covariance"]) < 1e-10
