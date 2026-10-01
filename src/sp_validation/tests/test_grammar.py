"""The ShapePipe v1 -> v2 grammar adapter presents v1 products as their v2 twins.

One synthetic catalogue is written in both grammars: v2 (T, split G1/G2,
boolean MASK_{b}_{label}) and v1 (sigma, 2-vector or flattened ELL,
{b}_{label} mask flags, plus the IMAFLAGS_ISO bitmask the adapter leaves alone).
Adapting the v1 form must reproduce the v2 form column for column, for every
table flavour the pipeline reads (numpy, FITS_rec, h5py), and row selection
must commute with adaptation.
"""

import h5py
import numpy as np
import pytest
from astropy.io import fits
from cs_util.size import T_to_sigma
from numpy.lib import recfunctions as rfn

from sp_validation import grammar, io
from sp_validation.grammar import (
    MASK_LABELS,
    SHEARS,
    V2View,
    adapt,
    detect_generation,
    v2_names,
)

N = 64

#: cat_config ``psf:`` block naming the v2 HSM columns.
PSF_BLOCK = {
    "ra_col": "RA",
    "dec_col": "DEC",
    "e1_PSF_col": "HSM_G1_PSF",
    "e2_PSF_col": "HSM_G2_PSF",
    "e1_star_col": "HSM_G1_STAR",
    "e2_star_col": "HSM_G2_STAR",
    "PSF_size": "HSM_T_PSF",
    "star_size": "HSM_T_STAR",
    "PSF_flag": "HSM_FLAG_PSF",
    "star_flag": "HSM_FLAG_STAR",
}
SHEAR_BLOCK = {"w_col": "w", "e1_col": "e1", "e2_col": "e2"}


def _twins(vector_ell=True):
    """Return (v1, v2) structured arrays holding the same content."""
    rng = np.random.default_rng(42)
    v2 = {
        "RA": rng.uniform(0, 360, N),
        "DEC": rng.uniform(-30, 30, N),
        "MAG": rng.uniform(18, 24, N),
    }
    v1 = dict(v2)
    for obj in ("PSF", "STAR"):
        g1, g2 = rng.normal(0, 0.05, (2, N))
        T = rng.uniform(0.4, 0.9, N)
        flag = rng.integers(0, 3, N).astype(np.int16)
        v2 |= {
            f"HSM_G1_{obj}": g1,
            f"HSM_G2_{obj}": g2,
            f"HSM_T_{obj}": T,
            f"HSM_FLAG_{obj}": flag,
        }
        v1 |= {
            f"E1_{obj}_HSM": g1,
            f"E2_{obj}_HSM": g2,
            f"SIGMA_{obj}_HSM": T_to_sigma(T),
            f"FLAG_{obj}_HSM": flag,
        }
    for shear in SHEARS:
        for v1_tag, v2_tag in (("", ""), ("_ERR", "_ERR"), ("_PSFo", "_PSF_ORIG")):
            g = rng.normal(0, 0.2, (N, 2))
            v2[f"NGMIX_G1{v2_tag}_{shear}"] = g[:, 0]
            v2[f"NGMIX_G2{v2_tag}_{shear}"] = g[:, 1]
            name = f"NGMIX_ELL{v1_tag}_{shear}"
            if vector_ell:
                v1[name] = g
            else:
                v1[f"{name}_0"], v1[f"{name}_1"] = g[:, 0], g[:, 1]
        T_orig, T_reconv = rng.uniform(0.3, 0.8, (2, N))
        v2[f"NGMIX_T_PSF_ORIG_{shear}"] = v1[f"NGMIX_T_PSFo_{shear}"] = T_orig
        v2[f"NGMIX_T_PSF_RECONV_{shear}"] = v1[f"NGMIX_Tpsf_{shear}"] = T_reconv
    # v1's no-shear reconvolved-PSF size is wrong; the view reads the 1P kernel.
    v1["NGMIX_Tpsf_NOSHEAR"] = 1.02 * v1["NGMIX_Tpsf_1P"]
    v2["NGMIX_T_PSF_RECONV_NOSHEAR"] = v1["NGMIX_Tpsf_1P"]
    fail = rng.integers(0, 2, N).astype(np.int16)
    v2["NGMIX_MCAL_TYPES_FAIL"] = v1["NGMIX_MOM_FAIL"] = fail
    v1["IMAFLAGS_ISO"] = rng.integers(0, 256, N).astype(np.int16)
    for b, label in MASK_LABELS.items():
        v2[grammar.mask_column(b)] = v1[f"{b}_{label}"] = rng.integers(0, 2, N).astype(
            bool
        )
    return _structured(v1), _structured(v2)


def _structured(columns):
    dtype = [(k, v.dtype, v.shape[1:]) for k, v in columns.items()]
    out = np.empty(N, dtype=dtype)
    for key, value in columns.items():
        out[key] = value
    return out


def _to_fits_rec(arr, tmp_path):
    path = tmp_path / "cat.fits"
    fits.BinTableHDU(arr).writeto(path)
    return fits.getdata(path, 1), path


@pytest.fixture(params=["numpy-vector", "numpy-flat", "fits-vector", "h5py-flat"])
def v1_and_v2(request, tmp_path):
    """A v1 table of each flavour, plus its v2 twin as a numpy array."""
    flavour, layout = request.param.split("-")
    v1, v2 = _twins(vector_ell=layout == "vector")
    if flavour == "fits":
        v1, _ = _to_fits_rec(v1, tmp_path)
    elif flavour == "h5py":
        handle = h5py.File(tmp_path / "cat.hdf5", "w")
        handle.create_dataset("data", data=v1)
        request.addfinalizer(handle.close)
        v1 = handle["data"]
    return v1, v2


def _assert_same_columns(view, v2):
    names = set(grammar.column_names(view))
    assert names - {"IMAFLAGS_ISO"} == set(v2.dtype.names)
    for name in v2.dtype.names:
        expected, got = v2[name], np.asarray(view[name])
        assert got.shape == expected.shape, name
        if name.startswith("HSM_T_"):
            np.testing.assert_allclose(got, expected, rtol=1e-12, err_msg=name)
        else:
            np.testing.assert_array_equal(got, expected, err_msg=name)


def test_v1_adapts_to_its_v2_twin(v1_and_v2):
    v1, v2 = v1_and_v2
    view = adapt(v1)
    assert isinstance(view, V2View)
    assert len(view) == len(v2)
    _assert_same_columns(view, v2)
    assert view.dtype.names == view.names
    _assert_same_columns(view.to_structured(), v2)


def test_v1_names_are_hidden_but_imaflags_passes_through(v1_and_v2):
    view = adapt(v1_and_v2[0])
    assert "SIGMA_PSF_HSM" not in view
    assert "NGMIX_ELL_NOSHEAR" not in view.names
    assert "NGMIX_ELL_NOSHEAR_0" not in view.names
    assert "4_Stars" not in view
    with pytest.raises(KeyError):
        view["E1_PSF_HSM"]
    np.testing.assert_array_equal(
        view["IMAFLAGS_ISO"], np.asarray(v1_and_v2[0]["IMAFLAGS_ISO"])
    )


def test_row_selection_commutes_with_adaptation(v1_and_v2):
    v1, v2 = v1_and_v2
    view = adapt(v1)
    mask = np.asarray(v2["HSM_T_PSF"]) > 0.6
    _assert_same_columns(view[mask], v2[mask])
    _assert_same_columns(view[5:40:3], v2[5:40:3])
    _assert_same_columns(view[mask][::2], v2[mask][::2])
    _assert_same_columns(view[4:50][::-3], v2[4:50][::-3])
    _assert_same_columns(view[10:5], v2[10:5])
    if isinstance(v1, np.ndarray):
        _assert_same_columns(adapt(v1[mask]), v2[mask])
    row = view[7]
    assert row["HSM_T_STAR"] == pytest.approx(v2["HSM_T_STAR"][7], rel=1e-12)


def test_header_names_match_view_names(v1_and_v2):
    v1, _ = v1_and_v2
    names = grammar.column_names(v1)
    assert v2_names(names) == adapt(v1).names


def test_detect_generation():
    v1, v2 = _twins()
    assert detect_generation(v1.dtype.names) == "v1"
    assert detect_generation(v2.dtype.names) == "v2"
    assert detect_generation(["RA", "Dec", "e1", "e2", "w"]) is None
    with pytest.raises(ValueError, match="mixes"):
        detect_generation(["SIGMA_PSF_HSM", "HSM_T_STAR"])


def test_v2_and_neutral_tables_pass_through_unchanged():
    _, v2 = _twins()
    assert adapt(v2) is v2
    neutral = np.zeros(3, dtype=[("e1", "f8"), ("e2", "f8"), ("w", "f8")])
    assert adapt(neutral) is neutral
    assert v2_names(v2.dtype.names) == v2.dtype.names


def test_catalogue_presents_a_v1_fits_file_in_v2(tmp_path):
    v1, v2 = _twins()
    _, path = _to_fits_rec(v1, tmp_path)
    with io.Catalogue(path) as catalogue:
        assert catalogue.hdu == 1
        view = catalogue.table()
        assert isinstance(view, V2View)
        _assert_same_columns(view, v2)
        assert catalogue.dtype().names == view.names

    v2_path = tmp_path / "v2.fits"
    fits.BinTableHDU(v2).writeto(v2_path)
    with io.Catalogue(v2_path) as catalogue:
        assert not isinstance(catalogue.table(), V2View)


def test_psf_size_error_matches_between_grammars():
    """shear_psf_leakage's size-residual field sees T, not sigma, from v1.

    ``psf_size_error`` is e_star (T_star - T_psf) / T_star. Computed through
    the adapter from the v1 twin it must equal the v2 value; computed from a
    v1 table merely renamed (sigma under the T names) it must not.
    """
    from shear_psf_leakage.rho_tau_stat import Catalogs

    from sp_validation.rho_tau import get_params_rho_tau

    v1, v2 = _twins()
    entry = {"patch_number": 2, "psf": PSF_BLOCK, "shear": SHEAR_BLOCK}
    catalogs = Catalogs(params=get_params_rho_tau(entry))

    def size_error(cat):
        return np.stack(catalogs.get_cat_fields(cat, "psf_size_error")[2:4])

    expected = size_error(v2)
    np.testing.assert_allclose(size_error(adapt(v1)), expected, rtol=1e-12)

    renamed_only = {name: np.asarray(v2[name]) for name in v2.dtype.names}
    for obj in ("PSF", "STAR"):
        renamed_only[f"HSM_T_{obj}"] = np.asarray(v1[f"SIGMA_{obj}_HSM"])
    assert not np.allclose(size_error(renamed_only), expected)


def test_get_rho_tau_identical_for_v1_and_v2_psf_catalogues(tmp_path):
    """ρ/τ from a v1 PSF catalogue equal those from its v2 twin, end to end."""
    from sp_validation.rho_tau import get_rho_tau

    v1, v2 = _twins()
    v1_hsm = [n for n in v1.dtype.names if n in ("RA", "DEC") or "HSM" in n]
    rng = np.random.default_rng(3)
    shear = np.empty(N, dtype=[(n, "f8") for n in ("RA", "Dec", "e1", "e2", "w")])
    shear["RA"], shear["Dec"] = v2["RA"], v2["DEC"]
    shear["e1"], shear["e2"] = rng.normal(0, 0.3, (2, N))
    shear["w"] = 1.0
    fits.BinTableHDU(shear).writeto(tmp_path / "shear.fits")

    stats = {}
    for label, psf in (("v1", v1[v1_hsm]), ("v2", v2[list(PSF_BLOCK.values())])):
        psf_path = tmp_path / f"psf_{label}.fits"
        fits.BinTableHDU(np.array(psf)).writeto(psf_path)
        config = {
            label: {
                "patch_number": 2,
                "psf": PSF_BLOCK | {"path": str(psf_path), "hdu": 1},
                "shear": SHEAR_BLOCK | {"path": str(tmp_path / "shear.fits")},
            }
        }
        outdir = tmp_path / label
        outdir.mkdir()
        treecorr_config = {
            "ra_units": "deg",
            "dec_units": "deg",
            "sep_units": "arcmin",
            "min_sep": 10,
            "max_sep": 600,
            "nbins": 4,
        }
        get_rho_tau(config, label, treecorr_config, str(outdir), label)
        stats[label] = [
            fits.getdata(outdir / f"{kind}_stats_{label}.fits")
            for kind in ("rho", "tau")
        ]

    for table_v1, table_v2 in zip(stats["v1"], stats["v2"], strict=True):
        assert table_v1.dtype.names == table_v2.dtype.names
        for name in table_v1.dtype.names:
            np.testing.assert_allclose(table_v1[name], table_v2[name], rtol=1e-10)


# -- mask-bit columns: a rule family of their own ---------------------------


def _data_ext_mask_table(n=N, extra=()):
    """A data_ext-style table of {b}_{label} flags, and its MASK_{b}_{label} twin."""
    rng = np.random.default_rng(7)
    old = {
        f"{b}_{label}": rng.integers(0, 2, n).astype(bool)
        for b, label in MASK_LABELS.items()
    }
    new = {
        grammar.mask_column(b): old[f"{b}_{label}"] for b, label in MASK_LABELS.items()
    }
    for name in extra:
        old[name] = new[name] = rng.integers(0, 6, n)
    return _structured_n(old), _structured_n(new)


def _structured_n(columns):
    n = len(next(iter(columns.values())))
    out = np.empty(n, dtype=[(k, v.dtype, v.shape[1:]) for k, v in columns.items()])
    for key, value in columns.items():
        out[key] = value
    return out


@pytest.mark.parametrize("bit", sorted(MASK_LABELS))
def test_every_mask_bit_spelling_maps_to_the_canonical_column(bit):
    canonical = f"MASK_{bit}_{MASK_LABELS[bit]}"
    assert grammar.mask_column(bit) == canonical
    assert v2_names([f"{bit}_{MASK_LABELS[bit]}"]) == (canonical,)
    assert v2_names([f"MASK_n{bit}"]) == (canonical,)
    assert v2_names([canonical]) == (canonical,)


def test_mask_column_rejects_unknown_bit():
    with pytest.raises(KeyError):
        grammar.mask_column(4096)


def test_data_ext_mask_flags_rename_without_a_generation():
    """data_ext mask names are renamed whatever the ShapePipe generation."""
    old, new = _data_ext_mask_table(extra=("npoint3",))
    assert detect_generation(old.dtype.names) is None
    view = adapt(old)
    assert isinstance(view, V2View)
    assert view.names == new.dtype.names
    for name in new.dtype.names:
        np.testing.assert_array_equal(view[name], new[name])

    # Alongside v2 shape columns too: the mask family is not a v1 marker.
    _, v2 = _twins()
    v2_shape = v2[[n for n in v2.dtype.names if not n.startswith("MASK_")]]
    joined = adapt(rfn.repack_fields(v2_shape), old)
    assert detect_generation(joined.names) == "v2"
    np.testing.assert_array_equal(joined["MASK_4_Stars"], new["MASK_4_Stars"])
    np.testing.assert_array_equal(joined["HSM_T_PSF"], v2["HSM_T_PSF"])


@pytest.mark.parametrize(
    "faint, bright",
    [("1_Faint_star_halos", "2_Bright_star_halos"), ("MASK_n1", "MASK_n2")],
)
def test_halo_masks_keep_faint_and_bright_selections_distinct(faint, bright):
    """Producer halo names retain their bit identity through adaptation and cuts."""
    from sp_validation.galaxy import mask_cut

    # Fixed source names and distinct values avoid deriving the fixture from
    # MASK_LABELS, which would hide a reversed faint/bright mapping.
    source = _structured_n(
        {
            faint: np.array([False, True, False, True]),
            bright: np.array([False, False, True, True]),
        }
    )
    view = adapt(source)
    np.testing.assert_array_equal(
        view["MASK_1_Faint_star_halos"], [False, True, False, True]
    )
    np.testing.assert_array_equal(
        view["MASK_2_Bright_star_halos"], [False, False, True, True]
    )
    np.testing.assert_array_equal(
        mask_cut(view, ["MASK_1_Faint_star_halos"]), [True, False, True, False]
    )
    np.testing.assert_array_equal(
        mask_cut(view, ["MASK_2_Bright_star_halos"]), [True, True, False, False]
    )
    np.testing.assert_array_equal(
        mask_cut(view, ["MASK_1_Faint_star_halos", "MASK_2_Bright_star_halos"]),
        [True, False, False, False],
    )


def test_pre_release_mask_names_match_data_ext_names_and_cuts():
    """MASK_n{b} and {b}_{label} present the same columns and give the same cut."""
    from sp_validation.galaxy import DEFAULT_MASK_COLUMNS, MASK_COLUMNS, mask_cut

    old, new = _data_ext_mask_table(extra=("npoint3",))
    pre = rfn.rename_fields(
        old, {f"{b}_{label}": f"MASK_n{b}" for b, label in MASK_LABELS.items()}
    )
    assert detect_generation(pre.dtype.names) is None
    view_old, view_pre = adapt(old), adapt(pre)
    assert view_pre.names == view_old.names == new.dtype.names
    for name in new.dtype.names:
        np.testing.assert_array_equal(view_pre[name], new[name])
    for columns in (DEFAULT_MASK_COLUMNS, MASK_COLUMNS, ["MASK_8_Manual"]):
        expected = mask_cut(new, columns)
        np.testing.assert_array_equal(mask_cut(view_pre, columns), expected)
        np.testing.assert_array_equal(mask_cut(view_old, columns), expected)


def test_new_mask_names_pass_through_and_both_names_conflict():
    old, new = _data_ext_mask_table()
    assert adapt(new) is new
    with pytest.raises(ValueError, match="two names"):
        adapt(old, new[["MASK_4_Stars"]].copy())
    with pytest.raises(ValueError, match="two names"):
        adapt(
            _structured_n({"MASK_n4": new["MASK_4_Stars"]}),
            new[["MASK_4_Stars"]].copy(),
        )


def test_join_requires_equal_lengths_and_disjoint_names():
    old, new = _data_ext_mask_table()
    with pytest.raises(ValueError, match="lengths"):
        adapt(new, old[:10])
    with pytest.raises(ValueError, match="more than one table"):
        adapt(new, new)


# -- V2View mechanics --------------------------------------------------------


def test_dtype_matches_columns(v1_and_v2):
    """dtype reports exactly what each column read returns."""
    view = adapt(v1_and_v2[0])
    for name in view.names:
        column = np.asarray(view[name])
        assert view.dtype[name].base == column.dtype, name
        assert view.dtype[name].shape == column.shape[1:], name


def test_sigma_to_T_keeps_float32():
    v1 = np.zeros(4, dtype=[("SIGMA_PSF_HSM", "f4"), ("E1_PSF_HSM", "f4")])
    v1["SIGMA_PSF_HSM"] = 0.5
    view = adapt(v1)
    assert view.dtype["HSM_T_PSF"] == view["HSM_T_PSF"].dtype
    assert view.to_structured()["HSM_T_PSF"].dtype == view["HSM_T_PSF"].dtype


def test_empty_list_selects_an_empty_view(v1_and_v2):
    view = adapt(v1_and_v2[0])
    empty = view[[]]
    assert isinstance(empty, V2View)
    assert len(empty) == 0
    assert empty["HSM_T_PSF"].shape == (0,)
    assert len(view[np.zeros(len(view), dtype=bool)]) == 0


def test_boolean_selection_holds_only_the_selected_rows():
    """A mask becomes the selected indices, never a full-length index array."""
    old, _ = _data_ext_mask_table(n=1000)
    view = adapt(old)
    mask = np.zeros(1000, dtype=bool)
    mask[[3, 500, 998]] = True
    assert list(view[mask]._rows) == [3, 500, 998]
    window = view[100:900]
    assert isinstance(window._rows, range)
    sub = window[mask[100:900]]
    assert list(sub._rows) == [500]
    np.testing.assert_array_equal(sub["MASK_4_Stars"], old["4_Stars"][[500]])


def _h5py_mask_table(tmp_path, n=1000):
    old, new = _data_ext_mask_table(n=n, extra=("npoint3",))
    with h5py.File(tmp_path / "ext.hdf5", "w") as handle:
        handle.create_dataset("data_ext", data=old)
    return old, new


@pytest.mark.parametrize("block_rows", [1, 7, 10_000])
def test_h5py_selection_reads_each_dataset_once(tmp_path, monkeypatch, block_rows):
    """Selecting rows of an on-disk view reads every column in one pass."""
    old, new = _h5py_mask_table(tmp_path)
    monkeypatch.setattr(grammar, "BLOCK_BYTES", block_rows * old.dtype.itemsize)
    takes = []
    original = grammar._take

    def spy(table, rows, fields=None):
        takes.append(fields)
        return original(table, rows, fields)

    monkeypatch.setattr(grammar, "_take", spy)
    rng = np.random.default_rng(3)
    mask = rng.random(1000) < 0.3
    mask[400:700] = False
    index = np.array([998, 3, 500, 500, 2, 250])
    with h5py.File(tmp_path / "ext.hdf5", "r") as handle:
        view = adapt(handle["data_ext"])
        for key in (mask, index, slice(10, 20, 3), slice(None, None, -7)):
            takes.clear()
            sub = view[key]
            assert takes == [None]
            assert not sub._on_disk
            for name in new.dtype.names:
                np.testing.assert_array_equal(sub[name], new[key][name], err_msg=name)
        takes.clear()
        got = view[mask].to_structured(["MASK_8_Manual", "npoint3"])
        np.testing.assert_array_equal(got["MASK_8_Manual"], new["MASK_8_Manual"][mask])

        # Materialising named columns reads only their fields, in one pass.
        takes.clear()
        got = view.to_structured(
            ["MASK_8_Manual", "npoint3", "MASK_1_Faint_star_halos"]
        )
        assert takes == [["8_Manual", "npoint3", "1_Faint_star_halos"]]
        np.testing.assert_array_equal(
            got["MASK_1_Faint_star_halos"], new["MASK_1_Faint_star_halos"]
        )


@pytest.mark.parametrize("key", [(), Ellipsis])
def test_empty_tuple_and_ellipsis_select_every_row(tmp_path, key):
    old, new = _h5py_mask_table(tmp_path)
    with h5py.File(tmp_path / "ext.hdf5", "r") as handle:
        for view in (adapt(old), adapt(handle["data_ext"])):
            everything = view[key]
            assert len(everything) == len(new)
            np.testing.assert_array_equal(
                everything["MASK_4_Stars"], new["MASK_4_Stars"]
            )
            with pytest.raises(IndexError, match="tuple"):
                view[(0, 1)]


def test_dtype_is_computed_once():
    old, _ = _data_ext_mask_table()
    view = adapt(old)
    assert view.dtype is view.dtype
    assert view[3:9].dtype is view.dtype


def test_v1_no_shear_reconv_psf_reads_the_1P_kernel():
    """v1 NGMIX_Tpsf_NOSHEAR is wrong: RECONV_NOSHEAR reads Tpsf_1P when present."""
    names = [f"NGMIX_Tpsf_{shear}" for shear in SHEARS]
    table = np.zeros(4, dtype=[(name, "f8") for name in names])
    for i, name in enumerate(names):
        table[name] = 0.5 + i
    view = adapt(table)
    np.testing.assert_array_equal(
        view["NGMIX_T_PSF_RECONV_NOSHEAR"], table["NGMIX_Tpsf_1P"]
    )
    np.testing.assert_array_equal(view["NGMIX_T_PSF_RECONV_1P"], table["NGMIX_Tpsf_1P"])
    assert "NGMIX_Tpsf_NOSHEAR" not in view
    assert view.names[0] == "NGMIX_T_PSF_RECONV_NOSHEAR"
    assert sorted(view.names) == sorted(
        f"NGMIX_T_PSF_RECONV_{shear}" for shear in SHEARS
    )

    # A v1 cut catalogue carries only the (already corrected) no-shear size.
    cut = np.zeros(
        3, dtype=[("NGMIX_Tpsf_NOSHEAR", "f8"), ("NGMIX_Tpsf_NOSHEAR_orig", "f8")]
    )
    cut["NGMIX_Tpsf_NOSHEAR"], cut["NGMIX_Tpsf_NOSHEAR_orig"] = 0.4, 0.41
    view = adapt(cut)
    assert view.names == ("NGMIX_T_PSF_RECONV_NOSHEAR", "NGMIX_Tpsf_NOSHEAR_orig")
    np.testing.assert_array_equal(
        view["NGMIX_T_PSF_RECONV_NOSHEAR"], cut["NGMIX_Tpsf_NOSHEAR"]
    )
