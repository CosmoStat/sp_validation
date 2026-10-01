"""A v1 comprehensive catalogue calibrates exactly like its v2 twin.

One synthetic catalogue is written twice: as a post-processed ShapePipe v1
comprehensive HDF5 (``data`` in the v1 grammar, ``data_ext`` holding the
healsparse mask bits under their ``{b}_{label}`` names) and as a ShapePipe v2
catalogue (one ``data`` dataset, v2 grammar, ``MASK_{b}_{label}``). Read
through the readers the pipeline uses, the two must give identical mask
selections, for ``galaxy.mask_cut`` and for every mask config, and identical
metacal inputs.
Also covers the campaign and star readers on v1-grammar files.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml
from astropy.io import fits
from cs_util.size import T_to_sigma

from sp_validation import catalog, galaxy, grammar
from sp_validation.calibration import metacal
from sp_validation.catalog_builders import CalibrateCat, JointCat
from sp_validation.masks import Mask, get_masks_from_config

N = 4000
CONFIG_DIR = Path(__file__).resolve().parents[3] / "config" / "calibration"
BITS = (1, 2, 4, 8, 64, 1024)  # the bits v1 post-processing wrote


def _twins():
    """Return (v1 data, v1 data_ext, v2 data) holding the same catalogue."""
    rng = np.random.default_rng(11)
    common = {
        "RA": rng.uniform(150, 160, N),
        "Dec": rng.uniform(30, 40, N),
        "FLAGS": (rng.random(N) < 0.1).astype(np.int16) * 2,
        "overlap": rng.random(N) < 0.95,
        "N_EPOCH": rng.integers(1, 8, N).astype(np.int16),
        "mag": rng.uniform(14.5, 30.5, N),
        "NGMIX_N_EPOCH": rng.integers(0, 6, N).astype(np.int16),
        "NGMIX_MCAL_FLAGS": (rng.random(N) < 0.05).astype(np.int32),
        "w_iv": rng.uniform(0.5, 2, N),
        # passes through unmapped in both
        "IMAFLAGS_ISO": (rng.random(N) < 0.1).astype(np.int16) * 2,
    }
    v1 = dict(common)
    v2 = dict(common)
    fail = (rng.random(N) < 0.05).astype(np.int16)
    v1["NGMIX_MOM_FAIL"] = v2["NGMIX_MCAL_TYPES_FAIL"] = fail
    for shear in grammar.SHEARS:
        g = rng.normal(0, 0.3, (2, N))
        err = rng.uniform(0.05, 0.2, (2, N))
        Tpsf = rng.uniform(0.3, 0.6, N)
        for i in (0, 1):
            v1[f"NGMIX_ELL_{shear}_{i}"] = v2[f"NGMIX_G{i + 1}_{shear}"] = g[i]
            v1[f"NGMIX_ELL_ERR_{shear}_{i}"] = v2[f"NGMIX_G{i + 1}_ERR_{shear}"] = err[
                i
            ]
        v1[f"NGMIX_Tpsf_{shear}"] = v2[f"NGMIX_T_PSF_RECONV_{shear}"] = Tpsf
        columns = {
            "T": Tpsf * rng.uniform(0.5, 3.5, N),
            "T_ERR": rng.uniform(0.01, 0.1, N),
            "FLUX": rng.uniform(5, 400, N),
            "FLUX_ERR": rng.uniform(0.5, 2, N),
            "FLAGS": (rng.random(N) < 0.03).astype(np.int32),
        }
        for key, values in columns.items():
            v1[f"NGMIX_{key}_{shear}"] = v2[f"NGMIX_{key}_{shear}"] = values
    # v1's no-shear reconvolved-PSF size is wrong; the reader uses the 1P kernel.
    v2["NGMIX_T_PSF_RECONV_NOSHEAR"] = v1["NGMIX_Tpsf_1P"]
    v1["NGMIX_Tpsf_NOSHEAR"] = 1.02 * v1["NGMIX_Tpsf_1P"]
    psf = rng.normal(0, 0.02, (2, N))
    psf[:, rng.random(N) < 0.02] = -10
    for i in (0, 1):
        v1[f"NGMIX_ELL_PSFo_NOSHEAR_{i}"] = v2[f"NGMIX_G{i + 1}_PSF_ORIG_NOSHEAR"] = (
            psf[i]
        )
    sigma = rng.uniform(0.3, 0.5, N)
    v1["SIGMA_PSF_HSM"] = sigma
    v2["HSM_T_PSF"] = 2 * sigma**2

    ext = {}
    for bit in BITS:
        flag = rng.random(N) < 0.08
        ext[f"{bit}_{grammar.MASK_LABELS[bit]}"] = flag
        v2[grammar.mask_column(bit)] = flag
    npoint = rng.integers(0, 6, N).astype(np.int16)
    ext["npoint3"] = v2["npoint3"] = npoint
    return _structured(v1), _structured(ext), _structured(v2)


def _structured(columns):
    out = np.empty(N, dtype=[(k, np.asarray(v).dtype) for k, v in columns.items()])
    for key, value in columns.items():
        out[key] = value
    return out


@pytest.fixture
def comprehensive(tmp_path):
    """Read the v1 and v2 comprehensive files through ``CalibrateCat``."""
    v1, ext, v2 = _twins()
    with h5py.File(tmp_path / "v1.hdf5", "w") as handle:
        handle.create_dataset("data", data=v1)
        handle.create_dataset("data_ext", data=ext)
    with h5py.File(tmp_path / "v2.hdf5", "w") as handle:
        handle.create_dataset("data", data=v2)

    tables, readers = {}, []
    for label in ("v1", "v2"):
        reader = CalibrateCat()
        reader._params["input_path"] = str(tmp_path / f"{label}.hdf5")
        tables[label] = reader.read_cat()
        readers.append(reader)
    yield tables
    for reader in readers:
        reader._hd5file.close()


def test_v1_comprehensive_presents_the_v2_columns(comprehensive):
    v1, v2 = comprehensive["v1"], comprehensive["v2"]
    assert isinstance(v1, grammar.V2View)
    assert set(v1.dtype.names) == set(v2.dtype.names)
    for name in v2.dtype.names:
        np.testing.assert_allclose(v1[name], v2[name], rtol=1e-14, err_msg=name)


def test_mask_cut_is_identical(comprehensive):
    v1, v2 = comprehensive["v1"], comprehensive["v2"]
    for columns in (None, [grammar.mask_column(bit) for bit in BITS], ["MASK_4_Stars"]):
        np.testing.assert_array_equal(
            galaxy.mask_cut(v1, columns), galaxy.mask_cut(v2, columns)
        )
    assert 0 < galaxy.mask_cut(v1).sum() < N


@pytest.mark.parametrize(
    "config_name",
    sorted(
        path.name
        for path in CONFIG_DIR.glob("mask_*.yaml")
        if not path.name.endswith("overlay.yaml")
    ),
)
def test_config_selection_is_identical(comprehensive, config_name):
    config = yaml.safe_load((CONFIG_DIR / config_name).read_text())
    available = set(comprehensive["v2"].dtype.names)
    cuts = [cut for cut in config["dat"] if cut["col_name"] in available]
    if not cuts:
        pytest.skip(f"{config_name} cuts on no column of the twin catalogue")
    config = {"dat": cuts}

    combined = {}
    for label, table in comprehensive.items():
        masks, labels = get_masks_from_config(config, table)
        for mask in masks:
            assert mask._mask.shape == (N,)
        combined[label] = (
            Mask.from_list(masks)._mask,
            [m._mask for m in masks],
            labels,
        )
    assert combined["v1"][2] == combined["v2"][2]
    for mask_v1, mask_v2 in zip(combined["v1"][1], combined["v2"][1], strict=True):
        np.testing.assert_array_equal(mask_v1, mask_v2)
    np.testing.assert_array_equal(combined["v1"][0], combined["v2"][0])


def test_v1_configs_cut_every_mask_column_they_name(comprehensive):
    """The v1 configs' mask cuts all resolve on a v1 comprehensive file."""
    names = set(comprehensive["v1"].dtype.names)
    for path in CONFIG_DIR.glob("mask_v1.X.*.yaml"):
        if path.name.endswith("overlay.yaml"):
            continue
        config = yaml.safe_load(path.read_text())
        for cut in config["dat"]:
            if cut["col_name"].startswith("MASK_"):
                assert cut["col_name"] in names, (path.name, cut["col_name"])


def test_metacal_inputs_are_identical(comprehensive):
    config = yaml.safe_load((CONFIG_DIR / "mask_v1.X.6.yaml").read_text())
    cm = config["metacal"]
    results = {}
    for label, table in comprehensive.items():
        masks, _ = get_masks_from_config(config, table)
        selection = Mask.from_list(masks)._mask
        results[label] = metacal(
            table,
            selection,
            snr_min=cm["gal_snr_min"],
            snr_max=cm["gal_snr_max"],
            rel_size_min=cm["gal_rel_size_min"],
            rel_size_max=cm["gal_rel_size_max"],
            size_corr_ell=cm["gal_size_corr_ell"],
            sigma_eps=cm["sigma_eps_prior"],
            global_R_weight=None,
        )
    mc_v1, mc_v2 = results["v1"], results["v2"]
    assert mc_v1._n_input == mc_v2._n_input > 0
    assert np.all(np.isfinite(mc_v1.R))
    for attr in ("m1", "p1", "m2", "p2", "ns"):
        d1, d2 = getattr(mc_v1, attr), getattr(mc_v2, attr)
        assert d1.keys() == d2.keys()
        for key in d1:
            np.testing.assert_array_equal(d1[key], d2[key], err_msg=f"{attr}.{key}")
    np.testing.assert_array_equal(mc_v1.R, mc_v2.R)


def test_load_into_memory_matches_the_view(comprehensive, tmp_path):
    reader = CalibrateCat()
    reader._params["input_path"] = str(tmp_path / "v1.hdf5")
    loaded = reader.read_cat(load_into_memory=True)
    reader._hd5file.close()
    assert isinstance(loaded, np.ndarray)
    for name in loaded.dtype.names:
        np.testing.assert_array_equal(loaded[name], comprehensive["v1"][name])


# -- campaign and star readers on v1-grammar files ----------------------------


def _write_campaign(path, tiles):
    with h5py.File(path, "w") as handle:
        group = handle.create_group("patches").create_group("P3")
        for tile_id, dat in tiles.items():
            group.create_dataset(tile_id, data=dat)
        handle.attrs["n_tiles"] = len(tiles)


def test_campaign_reader_presents_v1_tiles_in_v2(tmp_path):
    v1, _, v2 = _twins()
    _write_campaign(
        tmp_path / "final_cat_P3.hdf5", {"000.000": v1[:1500], "001.000": v1[1500:]}
    )
    wanted = ["RA", "NGMIX_G1_NOSHEAR", "NGMIX_MCAL_TYPES_FAIL", "HSM_T_PSF"]
    dat = catalog.read_campaign_catalogue(
        str(tmp_path / "final_cat_P3.hdf5"), param_list=wanted, verbose=False
    )
    assert dat.dtype.names == tuple(wanted)
    for name in wanted:
        np.testing.assert_allclose(dat[name], v2[name], rtol=1e-14)

    n_rows, dtype = catalog.campaign_shape(
        str(tmp_path / "final_cat_P3.hdf5"), param_list=wanted
    )
    assert n_rows == N and dtype.names == tuple(wanted)

    merger = JointCat()
    merger._params["verbose"] = False
    merger._params["param_path"] = None
    merged = merger.merge_catalogues([str(tmp_path / "final_cat_P3.hdf5")])
    np.testing.assert_array_equal(merged["NGMIX_G2_ERR_1P"], v2["NGMIX_G2_ERR_1P"])
    assert "NGMIX_ELL_1P_0" not in merged.dtype.names


def test_star_reader_presents_v1_fits_in_v2(tmp_path):
    rng = np.random.default_rng(5)
    T = rng.uniform(0.3, 0.6, 50)
    v1 = np.zeros(
        50,
        dtype=[
            ("RA", "f8"),
            ("DEC", "f8"),
            ("E1_STAR_HSM", "f8"),
            ("SIGMA_STAR_HSM", "f8"),
        ],
    )
    v1["E1_STAR_HSM"] = rng.normal(0, 0.05, 50)
    v1["SIGMA_STAR_HSM"] = T_to_sigma(T)
    fits.BinTableHDU(v1).writeto(tmp_path / "stars.fits")
    stars = catalog.read_star_catalogue(str(tmp_path / "stars.fits"), verbose=False)
    assert isinstance(stars, np.ndarray)
    np.testing.assert_allclose(stars["HSM_T_STAR"], T, rtol=1e-12)
    np.testing.assert_array_equal(stars["HSM_G1_STAR"], v1["E1_STAR_HSM"])
