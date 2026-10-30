"""Analytic S1/case-d checks; no astronomical input required."""

import numpy as np
import pytest
from astropy.io import fits

from sp_validation import calibration, io, response


def catalogue(n=80):
    fields = []
    for b in ("NOSHEAR", "1P", "1M", "2P", "2M"):
        fields += [
            (f"NGMIX_{k}_{b}", "f8")
            for k in (
                "FLAGS",
                "G1",
                "G2",
                "G1_ERR",
                "G2_ERR",
                "FLUX",
                "FLUX_ERR",
                "T",
                "T_ERR",
                "T_PSF_RECONV",
            )
        ]
    d = np.zeros(n, dtype=fields)
    matrix = np.array([[0.8, 0.12], [-0.06, 0.9]])
    intrinsic = np.tile([[0.2, -0.2], [0.1, -0.1]], (1, n // 2))
    ce = matrix @ np.array([0.02, -0.01]) + np.array([0.004, -0.002])
    for b, shift in [
        ("NOSHEAR", [0, 0]),
        ("1P", [0.01, 0]),
        ("1M", [-0.01, 0]),
        ("2P", [0, 0.01]),
        ("2M", [0, -0.01]),
    ]:
        e = intrinsic + ce[:, None] + (matrix @ shift)[:, None]
        for i in (1, 2):
            d[f"NGMIX_G{i}_{b}"] = e[i - 1]
        d[f"NGMIX_FLUX_{b}"] = np.repeat([30.0, 100.0], n // 2)
        d[f"NGMIX_FLUX_ERR_{b}"] = 1
        d[f"NGMIX_T_{b}"] = 1
        d[f"NGMIX_T_PSF_RECONV_{b}"] = 1
    return d, matrix, ce


@pytest.mark.parametrize("scheme", ["des", "iv"])
def test_full_matrix_c_units_and_sim_signal(scheme):
    d, r, ce = catalogue()
    gm = calibration.metacal(
        d,
        np.ones(len(d), bool),
        weight_scheme=scheme,
        size_corr_ell=False,
        des_num_bins=2,
    )
    np.testing.assert_allclose(gm.R, r, atol=1e-14)
    np.testing.assert_allclose(gm.R_joint, r, atol=1e-14)
    np.testing.assert_allclose(gm.c_e, ce, atol=1e-14)
    np.testing.assert_allclose(gm.c_g, np.linalg.solve(r, ce), atol=1e-14)
    sim, _, w, _, _, _ = calibration.get_calibrated_m_c(gm, False)
    real, *_ = calibration.get_calibrated_m_c(gm, True)
    np.testing.assert_allclose(np.average(real, weights=w, axis=1), 0, atol=1e-14)
    np.testing.assert_allclose(np.average(sim, weights=w, axis=1), gm.c_g)


def test_fixed_weight_c_uses_final_weights():
    d, _, _ = catalogue()
    d["NGMIX_G1_ERR_NOSHEAR"] = np.linspace(0.01, 0.8, len(d))
    gm = calibration.metacal(
        d, np.ones(len(d), bool), weight_scheme="iv", size_corr_ell=False
    )
    _, raw, w, _, ce, _ = calibration.get_calibrated_m_c(gm)
    np.testing.assert_allclose(ce, np.average(raw, weights=w, axis=1))
    assert abs(ce[0] - raw[0].mean()) > 1e-5


def fixture_response():
    e = np.array([[0.2, -0.1, 0.4, -0.2], [0.1, 0.3, -0.2, 0.2]])
    r = np.array([[0.8, 0.1], [-0.04, 0.9]])
    branches = {"ns": e}
    for j, (p, m) in enumerate((("p1", "m1"), ("p2", "m2"))):
        branches[p] = e + 0.01 * r[:, j, None]
        branches[m] = e - 0.01 * r[:, j, None]
    masks = {b: np.arange(4) for b in branches}
    rg = np.repeat(r[:, :, None], 4, axis=2)
    w = {b: np.array([1.0, 2.0, 3.0, 4.0]) for b in branches}
    return branches, masks, rg, w


def test_selection_separate_denominators():
    b, masks, rg, w = fixture_response()
    masks["p1"] = np.array([0, 1, 2])
    masks["m1"] = np.array([1, 3])
    res = response.responses(b, masks, w, rg, 0.01)
    expected = (
        np.average(b["ns"][:, masks["p1"]], weights=w["p1"][masks["p1"]], axis=1)
        - np.average(b["ns"][:, masks["m1"]], weights=w["m1"][masks["m1"]], axis=1)
    ) / 0.02
    np.testing.assert_allclose(res["selection"][:, 0], expected)


def test_pure_cell_migration_all_elements():
    b, masks, rg, _ = fixture_response()
    # Fixed two-cell W. No cut crossing; all four selection elements nonzero.
    table = (
        np.array([10.0, 50.0, 500.0]),
        np.array([0.5, 3.0]),
        np.array([[1.0], [4.0]]),
        None,
    )
    positions = {
        "ns": [20, 20, 100, 100],
        "p1": [100, 20, 100, 20],
        "m1": [20, 100, 20, 100],
        "p2": [100, 100, 20, 100],
        "m2": [20, 20, 100, 20],
    }
    w = {
        k: response.evaluate_w(table, np.array(s), np.ones(4))
        for k, s in positions.items()
    }
    res = response.responses(b, masks, w, rg, 0.01)
    for j, (p, m) in enumerate((("p1", "m1"), ("p2", "m2"))):
        expected = (
            np.average(b["ns"], weights=w[p], axis=1)
            - np.average(b["ns"], weights=w[m], axis=1)
        ) / 0.02
        np.testing.assert_allclose(res["selection"][:, j], expected)
    frozen = response.responses(b, masks, {k: w["ns"] for k in w}, rg, 0.01)
    np.testing.assert_allclose(frozen["selection"], 0, atol=1e-14)
    assert np.all(abs(res["selection"]) > 0.1)
    np.testing.assert_allclose(res["joint"], res["total"], atol=1e-14)
    scaled = response.responses(b, masks, {k: 17 * v for k, v in w.items()}, rg, 0.01)
    np.testing.assert_allclose(scaled["total"], res["total"])


def test_w_moments_edges_empty_invalid_and_new_rows():
    table = response.fit_w(
        np.array([10.0, 50.0, 100.0]),
        np.ones(3),
        np.array([[0.2, 0.2, 0.4], [0.2, 0.2, 0.4]]),
        np.array([0.8, 0.8, 0.8]),
        np.array([10.0, 50.0, 500.0]),
        np.array([0.5, 2.0, 3.0]),
    )
    np.testing.assert_allclose(table[2], [[16.0, 0], [4.0, 0]])
    s = np.array([1.0, 50.0, 51.0, 1000.0, np.nan, 20.0])
    size = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.5])
    np.testing.assert_allclose(
        response.evaluate_w(table, s, size), [16, 16, 4, 4, 0, 0]
    )


def test_spatial_jackknife_shared_selection_and_covariance():
    b, masks, rg, w = fixture_response()
    groups = np.array(["a", "b", "c", "d"])
    masks["p1"] = np.array([0, 1, 2])
    masks["m1"] = np.array([1, 2, 3])
    labels, estimates, cov = response.jackknife(b, masks, w, rg, 0.01, groups)
    brute = [response.responses(b, masks, w, rg, 0.01, groups != g) for g in labels]
    for key in estimates:
        np.testing.assert_allclose(estimates[key], [x[key] for x in brute])
    v = np.concatenate([estimates[k].reshape(4, -1) for k in estimates], axis=1)
    dv = v - v.mean(axis=0)
    np.testing.assert_allclose(cov, 0.75 * dv.T @ dv)
    assert cov[4, 4] > 0


def test_end_to_end_cell_migration_and_sheared_only_row():
    d, _, _ = catalogue()
    for branch in ("NOSHEAR", "1P", "1M", "2P", "2M"):
        d[f"NGMIX_G1_{branch}"][40:] += np.tile([0.15, -0.15], 20)
    d["NGMIX_FLUX_1P"][:40:2] = 100
    d["NGMIX_FLUX_1M"][1:40:2] = 100
    # A common-parent row absent from S_ns still gets a defined branch W.
    d["NGMIX_FLUX_NOSHEAR"][0] = 9
    gm = calibration.metacal(
        d,
        np.ones(len(d), bool),
        weight_scheme="des",
        size_corr_ell=False,
        des_num_bins=2,
    )
    assert 0 not in gm.mask_dict["ns"]
    assert 0 in gm.mask_dict["p1"]
    assert gm.branch_weights["p1"][0] > 0
    assert gm.branch_migration["p1"] > 0
    manual = []
    for key in ("p1", "m1"):
        idx = gm.mask_dict[key]
        manual.append(
            np.average(
                np.array([gm.ns["g1"][idx], gm.ns["g2"][idx]]),
                weights=gm.branch_weights[key][idx],
                axis=1,
            )
        )
    np.testing.assert_allclose(gm.R_selection[:, 0], (manual[0] - manual[1]) / 0.02)
    frozen = {b: gm.branch_weights["ns"] for b in response.BRANCHES}
    res = response.responses(gm.branches, gm.mask_dict, frozen, gm.R_shear, 0.01)
    assert abs(res["selection"][0, 0] - gm.R_selection[0, 0]) > 0.1


def test_coherent_spatial_errors_exceed_independent_object_errors():
    d, _, _ = catalogue()
    groups = np.repeat(np.arange(4), 20)
    for comp in (1, 2):
        for sign, step in (("P", 0.01), ("M", -0.01)):
            d[f"NGMIX_G{comp}_{comp}{sign}"] += step * groups * 0.1
    gm = calibration.metacal(
        d, np.ones(len(d), bool), weight_scheme="iv", size_corr_ell=False
    )
    _, _, spatial = gm.response_jackknife(groups)
    _, _, objects = gm.response_jackknife(np.arange(len(d)))
    assert spatial[0, 0] > 10 * objects[0, 0]


def test_mismatched_config_and_recalibration_refusal(tmp_path):
    d, _, _ = catalogue()
    with pytest.raises(ValueError, match="disagrees"):
        calibration.metacal(d, np.ones(len(d), bool), weight_scheme="iv", w_col="w_des")
    hdr = fits.Header({"R_S11": 0.1, "R_S12": 0.0, "R_S21": 0.0, "R_S22": 0.1})
    path = tmp_path / "insufficient.fits"
    fits.HDUList([fits.PrimaryHDU(header=hdr), fits.BinTableHDU()]).writeto(path)
    with pytest.raises(ValueError, match="parent five-branch"):
        calibration.get_calibrate_e_from_cat(path)


def test_recalibration_requires_unchanged_population_with_nonzero_selection(tmp_path):
    path = tmp_path / "compatible.fits"
    truth = np.repeat([0.02, -0.02], 4)
    raw = 0.9 * truth
    zeros = np.zeros(8)
    cols = [
        fits.Column(name="e1_uncal", array=raw, format="D"),
        fits.Column(name="e2_uncal", array=zeros, format="D"),
        fits.Column(name="w_des", array=np.ones(8), format="D"),
    ]
    for i in (1, 2):
        for j in (1, 2):
            cols.append(
                fits.Column(
                    name=f"R_g{i}{j}",
                    array=np.full(8, 0.8 if i == j else 0),
                    format="D",
                )
            )
    hdr = fits.Header(
        {
            "R_S11": 0.1,
            "R_S12": 0.0,
            "R_S21": 0.0,
            "R_S22": 0.1,
            "WGHTSCH": "des",
            "N_ns": 8,
            "SW_ns": 8.0,
        }
    )
    fits.HDUList(
        [fits.PrimaryHDU(header=hdr), fits.BinTableHDU.from_columns(cols)]
    ).writeto(path)
    g1, _ = calibration.get_calibrate_e_from_cat(path)
    np.testing.assert_allclose(g1, truth)
    with fits.open(path, mode="update") as hdus:
        hdus[0].header["N_ns"] = 9
    with pytest.raises(ValueError, match="population/weights"):
        calibration.get_calibrate_e_from_cat(path)


def test_downstream_c_is_calibrated_not_raw():
    from types import SimpleNamespace

    from sp_validation.cosmo_val.core import CosmologyValidation

    e = np.array([[0.12, -0.04], [0.08, 0.16]])
    cg = np.array([0.02, -0.03])
    obj = SimpleNamespace(
        _read_shear_cols=lambda *args: e,
        _component_responses=lambda ver: (0.8, 0.8),
        cc={"external": {"shear": {"R": 0.8}}},
        c1={"external": cg[0]},
        c2={"external": cg[1]},
    )
    result = CosmologyValidation._calibrated_g(obj, "external")
    np.testing.assert_allclose(result, e / 0.8 - cg[:, None])
    assert not np.allclose(result, (e - cg[:, None]) / 0.8)


@pytest.mark.parametrize("adapter", ["xi", "pseudo_cl", "rho_tau", "leakage"])
def test_calibrated_matrix_shear_passes_reader_adapters_once(
    tmp_path, monkeypatch, adapter
):
    from types import SimpleNamespace

    from sp_validation.cosmo_val import core
    from sp_validation.rho_tau import _CatalogueLoader

    d, _, _ = catalogue()
    gm = calibration.metacal(
        d,
        np.ones(len(d), bool),
        weight_scheme="des",
        size_corr_ell=False,
        des_num_bins=2,
    )
    g, _, w, _, _, _ = calibration.get_calibrated_m_c(gm, False)
    path = tmp_path / "matrix_calibrated.fits"
    cols = [
        fits.Column(name="e1", array=g[0], format="D"),
        fits.Column(name="e2", array=g[1], format="D"),
        fits.Column(name="w_des", array=w, format="D"),
    ]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(path)
    calibration.save_response_product(path, gm)
    entry = dict(path=path, w_col="w_des", e1_col="e1", e2_col="e2", R=1.0)

    def read(block):
        if adapter == "rho_tau":
            loader = _CatalogueLoader({"shear": block}, {})
            filename = loader("shear")  # ordinary FITS is passed through
            with fits.open(filename) as hdus:
                return np.array([hdus[1].data["e1"], hdus[1].data["e2"]])
        if adapter == "leakage":
            obj = core._LeakageObject.__new__(core._LeakageObject)
            obj.entries = {"shear": block}
            obj.read_data()
            return np.array([obj._dat["e1"], obj._dat["e2"]])
        if adapter == "xi":
            monkeypatch.setattr(core.leakage, "cut_data", lambda data, *args: data)
            obj = core._LeakageScale.__new__(core._LeakageScale)
            obj.entries = {"shear": block}
            obj._params = {"cut": None, "verbose": False}
            obj.read_data(shear=True, psf=False)
            return np.array([obj.dat_shear["e1"], obj.dat_shear["e2"]])
        obj = SimpleNamespace(
            cc={"SP": {"shear": block}},
            results={"SP": SimpleNamespace(dat_shear=io.open_entry(block))},
        )
        return np.array(
            core.CosmologyValidation._read_shear_cols(obj, "SP", "e1_col", "e2_col")
        )

    np.testing.assert_allclose(read(entry), g)
    for wrong in (
        dict(entry, w_col="w_iv"),
        dict(entry, R=0.8),
        dict(entry, R11="R_g11"),
    ):
        with pytest.raises(ValueError):
            read(wrong)


def test_persistence_and_downstream_no_double_R(tmp_path):
    d, _, _ = catalogue()
    gm = calibration.metacal(
        d,
        np.ones(len(d), bool),
        weight_scheme="des",
        size_corr_ell=False,
        des_num_bins=2,
    )
    path = tmp_path / "cut.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU.from_columns(
                [fits.Column(name="w_des", array=gm.ns["w"], format="D")]
            ),
        ]
    ).writeto(path)
    calibration.save_response_product(path, gm, np.arange(len(d)) % 4)
    with fits.open(path) as hdus:
        assert hdus["R_COV"].data.shape == (20, 20)
        assert hdus[0].header["RAPPLIED"]
        assert hdus[0].header["WGHTSCH"] == "des"
    for entry in (
        {"path": path, "w_col": "w_iv", "R": 1},
        {"path": path, "w_col": "w_des", "R": 0.8},
    ):
        with pytest.raises(ValueError):
            io.open_entry(entry)
    assert len(io.open_entry({"path": path, "w_col": "w_des", "R": 1})) == len(d)
