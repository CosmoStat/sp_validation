"""Each sample cosmo_val measures is calibrated with its own response.

A release catalogue carries the whole sample's response in its ``e1``, ``e2``;
its response record holds each tomographic bin's own. Two populations of
different response, each its own bin, are sheared by a known shear through
metacal branches; cosmo_val recovers that shear in each bin.
"""

import contextlib
import json

import numpy as np
import pytest

from sp_validation import response
from sp_validation.cosmo_val.core import CosmologyValidation
from sp_validation.response import BRANCHES, accumulate, estimate, record
from sp_validation.tests.test_response import CUTS, SHIFT, rows_from

G_IN = np.array([0.02, -0.01])
RESPONSES = {
    1: np.array([[0.90, 0.02], [-0.01, 0.85]]),
    2: np.array([[0.60, -0.03], [0.02, 0.65]]),
}
OFFSET = {1: np.array([3e-3, -1e-3]), 2: np.array([-2e-3, 4e-3])}


def population(R, offset, n=4000, seed=0):
    """Metacal branches of ``n`` galaxies of response ``R`` and additive bias
    ``offset``, half sheared by ``+G_IN`` and half by ``-G_IN``.

    Each intrinsic shape comes with its opposite under each shear sign, so
    every mean is exact. Returns the branch columns and the shear signs.
    """
    rng = np.random.default_rng(seed)
    e_int = rng.uniform(-0.2, 0.2, (2, n // 4))
    e_int = np.concatenate([e_int, -e_int, e_int, -e_int], axis=1)
    sign = np.repeat([1.0, 1.0, -1.0, -1.0], n // 4)
    g = {
        b: R @ (np.outer(G_IN, sign) + np.array(SHIFT[b])[:, None])
        + e_int
        + offset[:, None]
        for b in BRANCHES
    }
    snr = {b: np.full(n, 30.0) for b in BRANCHES}
    size = {b: np.full(n, 1.1) for b in BRANCHES}
    return rows_from(g, snr, size), sign


@pytest.fixture
def release(tmp_path):
    """The catalogue and response record as the release writes them."""
    parts = [population(RESPONSES[k], OFFSET[k], seed=k) for k in (1, 2)]
    rows = {key: np.concatenate([p[0][key] for p in parts]) for key in parts[0][0]}
    sign = np.concatenate([p[1] for p in parts])
    tomo_bin = np.repeat([1, 2], [len(p[1]) for p in parts])
    edges = response.table_edges(CUTS)
    table = response.unit_table(edges)

    def measure(selection):
        sums = accumulate({k: v[selection] for k, v in rows.items()}, CUTS, edges)
        result = estimate(sums, table)
        # Every galaxy is selected on every branch: no selection response.
        assert set(result["counts"].values()) == {selection.sum()}
        return result

    whole = measure(np.ones(len(sign), bool))
    rec = record(whole)
    rec["bins"] = {
        "column": "TOM_BIN_ID",
        "records": {str(k): record(measure(tomo_bin == k)) for k in (1, 2)},
    }
    path = tmp_path / "response.json"
    path.write_text(json.dumps(rec))

    g_ns = np.array([rows["NGMIX_G1_NOSHEAR"], rows["NGMIX_G2_NOSHEAR"]])
    e = response.calibrate(g_ns, whole)
    catalogue = dict(e1=e[0], e2=e[1], w=np.ones(len(sign)), TOM_BIN_ID=tomo_bin)
    cv = object.__new__(CosmologyValidation)
    cv.cc = {
        "SP": {
            "shear": dict(
                e1_col="e1",
                e2_col="e2",
                w_col="w",
                tomo_bin_col="TOM_BIN_ID",
                response=str(path),
            )
        }
    }
    return cv, catalogue, sign, tomo_bin


def recovered(g, sign):
    """Shear of the ``+G_IN`` half minus that of the ``-G_IN`` half, over 2."""
    return (g[:, sign > 0].mean(axis=1) - g[:, sign < 0].mean(axis=1)) / 2


def assert_each_bin_recovers_its_input_shear(g, sign, tomo_bin):
    for k in (1, 2):
        rows = tomo_bin == k
        np.testing.assert_allclose(recovered(g[:, rows], sign[rows]), G_IN, rtol=1e-9)


def test_each_bin_recovers_its_input_shear(release):
    cv, catalogue, sign, tomo_bin = release
    g = cv._calibrated_g("SP", catalogue, tomography=True)
    whole = cv._calibrated_g("SP", catalogue)
    assert_each_bin_recovers_its_input_shear(g, sign, tomo_bin)
    for k in (1, 2):
        rows = tomo_bin == k
        np.testing.assert_allclose(g[:, rows].mean(axis=1), 0, atol=1e-15)
        # The whole sample's response applied to a bin misses its shear by
        # that bin's response relative to the whole sample's.
        off = recovered(whole[:, rows], sign[rows]) / G_IN - 1
        assert np.all(np.abs(off) > 0.05), off


def test_whole_sample_is_the_released_shear_centred(release):
    cv, catalogue, _, _ = release
    e = np.array([catalogue["e1"], catalogue["e2"]])
    g = cv._calibrated_g("SP", catalogue)
    np.testing.assert_allclose(g, e - e.mean(axis=1)[:, None], atol=1e-15)


def test_xi_measures_each_bins_own_shear(release):
    cv, catalogue, sign, tomo_bin = release
    n = len(sign)
    dat = dict(catalogue, RA=np.zeros(n), Dec=np.zeros(n))

    class Results:
        dat_shear = dat

        @contextlib.contextmanager
        def temporarily_read_data(self):
            yield

    cv._results = {"SP": Results()}
    cols = cv._shear_columns("SP", compute_tomography=True)
    g = np.array([cols["g1"], cols["g2"]])
    assert_each_bin_recovers_its_input_shear(g, sign, cols["tomo_bin"])


def test_pseudo_cl_measures_each_bins_own_shear(release, monkeypatch):
    from sp_validation.cosmo_val import pseudo_cl

    cv, catalogue, sign, tomo_bin = release
    monkeypatch.setattr(pseudo_cl, "open_entry", lambda entry: catalogue)
    cat = cv._pseudo_cl_catalogue("SP", tomography=True)
    g = np.array([cat["e1"], cat["e2"]])
    assert_each_bin_recovers_its_input_shear(g, sign, tomo_bin)


def test_tau_measures_each_bins_own_shear(release, monkeypatch, tmp_path):
    from sp_validation.cosmo_val import psf_systematics

    cv, catalogue, sign, tomo_bin = release
    handed = []

    def measure(*args, shear, **kwargs):
        handed.append(shear)
        raise StopIteration

    monkeypatch.setattr(psf_systematics, "open_entry", lambda entry: catalogue)
    monkeypatch.setattr(psf_systematics, "get_rho_tau_w_cov", measure)
    cv.cc["paths"] = {"output": str(tmp_path)}
    cv.versions = ["SP"]
    cv.treecorr_config = cv.npatch = cv.cov_estimate_method = None
    cv.compute_cov_rho = False
    for name in ("print_start", "print_cyan"):
        setattr(cv, name, lambda msg: None)
    cv._get_tomo_bins = lambda ver: ([1, 2], [(1, 1), (1, 2), (2, 2)])
    cv._get_star_mask = cv._get_galaxy_mask = lambda *args: None
    cv.basename = lambda *args, **kwargs: "base"
    with pytest.raises(StopIteration):
        cv.calculate_rho_tau_stats(tomography=True)
    assert_each_bin_recovers_its_input_shear(handed[0], sign, tomo_bin)


def test_a_bin_missing_from_the_record_is_an_error(release, tmp_path):
    cv, catalogue, _, _ = release
    path = tmp_path / "whole_only.json"
    rec = json.loads(open(cv.cc["SP"]["shear"]["response"]).read())
    del rec["bins"]["records"]["2"]
    path.write_text(json.dumps(rec))
    cv.cc["SP"]["shear"]["response"] = str(path)
    with pytest.raises(KeyError, match="TOM_BIN_ID == 2"):
        cv._calibrated_g("SP", catalogue, tomography=True)


def test_rho_tau_reads_the_calibrated_shear(tmp_path):
    """The τ galaxies take the shear they are handed, with no second response."""
    from astropy.table import Table

    from sp_validation.rho_tau import _CatalogueLoader, get_params_rho_tau

    path = tmp_path / "shear.fits"
    n = 5
    Table(
        dict(
            RA=np.arange(n),
            Dec=np.arange(n),
            e1=np.ones(n),
            e2=np.ones(n),
            w=np.ones(n),
            R11=np.full(n, 0.5),
            R22=np.full(n, 0.5),
        )
    ).write(path)
    entry = dict(
        path=str(path), e1_col="e1", e2_col="e2", w_col="w", R11="R11", R22="R22"
    )
    cat = dict(
        shear=entry,
        psf=dict(
            ra_col="RA",
            dec_col="Dec",
            e1_PSF_col="a",
            e2_PSF_col="b",
            e1_star_col="c",
            e2_star_col="d",
            PSF_size="s",
            star_size="t",
        ),
        patch_number=1,
    )
    params = get_params_rho_tau(cat, calibrated=True)
    assert params["R11"] is None and params["R22"] is None
    shear = np.array([np.arange(n) * 0.1, -np.arange(n) * 0.1])
    with _CatalogueLoader(cat, params, shear) as load:
        written = Table.read(load("shear"))
    np.testing.assert_array_equal([written["e1"], written["e2"]], shear)
