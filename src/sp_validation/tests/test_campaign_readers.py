"""Tests for the ShapePipe v2 campaign catalogue readers and mask cut."""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import numpy.testing as npt

from sp_validation import catalog, galaxy
from sp_validation.catalog_builders import JointCat

GAL_DTYPE = np.dtype(
    [
        ("RA", "f8"),
        ("Dec", "f8"),
        ("MAG_AUTO", "f4"),
        ("MASK_n4", "?"),
        ("MASK_n1", "?"),
        ("MASK_n2", "?"),
        ("MASK_n8", "?"),
        ("MASK_n64", "?"),
        ("MASK_n1024", "?"),
    ]
)


def make_galaxy_data(n_obj, offset=0):
    dat = np.zeros(n_obj, dtype=GAL_DTYPE)
    dat["RA"] = np.arange(n_obj) + offset
    dat["Dec"] = np.arange(n_obj) + offset + 0.5
    dat["MAG_AUTO"] = 22.0
    return dat


def write_campaign(path, layout, tiles):
    """Write a campaign hdf5 file in the legacy or flat layout."""
    with h5py.File(path, "w") as f:
        if layout == "legacy":
            group = f.create_group("patches").create_group("CAMPAIGN")
        else:
            group = f.create_group("tiles")
        for tile_id, dat in tiles.items():
            group.create_dataset(tile_id, data=dat)
        f.attrs["n_tiles"] = len(tiles)


class TestCampaignReader(unittest.TestCase):
    """Campaign galaxy catalogue reader."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._dir = Path(self._tmp.name)
        self._tiles = {
            "000.000": make_galaxy_data(3),
            "001.000": make_galaxy_data(2, offset=100),
        }

    def tearDown(self):
        self._tmp.cleanup()

    def _expected(self):
        """Expected concatenation: datasets in sorted-key order."""
        return np.concatenate([self._tiles[key] for key in sorted(self._tiles)])

    def test_legacy_layout(self):
        path = self._dir / "final_cat_CAMPAIGN.hdf5"
        write_campaign(path, "legacy", self._tiles)

        dat = catalog.read_campaign_catalogue(path, verbose=False)

        self.assertEqual(len(dat), 5)
        npt.assert_array_equal(dat["RA"], self._expected()["RA"])

    def test_flat_layout(self):
        path = self._dir / "final_cat_CAMPAIGN.hdf5"
        write_campaign(path, "flat", self._tiles)

        dat = catalog.read_campaign_catalogue(path, verbose=False)

        self.assertEqual(len(dat), 5)
        npt.assert_array_equal(dat["RA"], self._expected()["RA"])

    def test_layouts_agree(self):
        legacy = self._dir / "legacy.hdf5"
        flat = self._dir / "flat.hdf5"
        write_campaign(legacy, "legacy", self._tiles)
        write_campaign(flat, "flat", self._tiles)

        npt.assert_array_equal(
            catalog.read_campaign_catalogue(legacy, verbose=False),
            catalog.read_campaign_catalogue(flat, verbose=False),
        )

    def test_param_list_restriction(self):
        path = self._dir / "final_cat_CAMPAIGN.hdf5"
        write_campaign(path, "flat", self._tiles)

        dat = catalog.read_campaign_catalogue(
            path, param_list=["RA", "Dec"], verbose=False
        )

        self.assertEqual(tuple(dat.dtype.names), ("RA", "Dec"))

    def test_missing_column_raises(self):
        path = self._dir / "final_cat_CAMPAIGN.hdf5"
        write_campaign(path, "flat", self._tiles)

        with self.assertRaises(KeyError) as ctx:
            catalog.read_campaign_catalogue(
                path, param_list=["RA", "NOT_A_COLUMN"], verbose=False
            )
        self.assertIn("NOT_A_COLUMN", str(ctx.exception))

    def test_row_order_is_tile_name_order(self):
        """Keys inserted out of order still concatenate in name order."""
        path = self._dir / "unordered.hdf5"
        tiles = {
            "222.000": make_galaxy_data(2, offset=200),
            "000.000": make_galaxy_data(2, offset=0),
            "111.000": make_galaxy_data(2, offset=100),
        }
        with h5py.File(path, "w") as f:
            group = f.create_group("tiles")
            for tile_id, dat in tiles.items():
                group.create_dataset(tile_id, data=dat)
            f.attrs["n_tiles"] = len(tiles)

        dat = catalog.read_campaign_catalogue(path, verbose=False)

        npt.assert_array_equal(dat["RA"], [0, 1, 100, 101, 200, 201])

    def test_truncated_file_raises(self):
        """n_tiles attribute larger than the number of datasets is fatal."""
        path = self._dir / "truncated.hdf5"
        write_campaign(path, "flat", self._tiles)
        with h5py.File(path, "a") as f:
            f.attrs["n_tiles"] = 10

        with self.assertRaises(ValueError) as ctx:
            catalog.read_campaign_catalogue(path, verbose=False)
        self.assertIn("incomplete", str(ctx.exception))

    def test_column_missing_from_later_tile_raises_clear_error(self):
        """A column absent from a non-first tile is named, with its dataset."""
        path = self._dir / "ragged.hdf5"
        with h5py.File(path, "w") as f:
            group = f.create_group("tiles")
            group.create_dataset("000.000", data=make_galaxy_data(2))
            group.create_dataset(
                "001.000", data=np.zeros(2, dtype=[("RA", "f8"), ("Dec", "f8")])
            )

        with self.assertRaises(KeyError) as ctx:
            catalog.read_campaign_catalogue(
                path, param_list=["RA", "MAG_AUTO"], verbose=False
            )
        message = str(ctx.exception)
        self.assertIn("MAG_AUTO", message)
        self.assertIn("001.000", message)

    def test_dtype_promoted_across_tiles(self):
        """A per-tile dtype difference within a campaign is not truncated."""
        narrow = np.zeros(2, dtype=[("N_EPOCH", "i2"), ("TILE_ID", "S7"), ("RA", "f4")])
        narrow["N_EPOCH"] = [1, 2]
        narrow["TILE_ID"] = [b"123.456", b"123.457"]
        narrow["RA"] = [1.5, 2.5]

        wide = np.zeros(2, dtype=[("N_EPOCH", "i4"), ("TILE_ID", "S12"), ("RA", "f8")])
        wide["N_EPOCH"] = [70000, 3]
        wide["TILE_ID"] = [b"999888.7776", b"123.458"]
        wide["RA"] = [3.123456789, 4.0]

        path = self._dir / "final_cat_MIX.hdf5"
        write_campaign(path, "flat", {"000.000": narrow, "000.001": wide})
        param_list = ["N_EPOCH", "TILE_ID", "RA"]

        dat = catalog.read_campaign_catalogue(
            str(path), param_list=param_list, verbose=False
        )

        self.assertEqual(dat.dtype["N_EPOCH"], np.dtype("i4"))
        self.assertEqual(dat.dtype["TILE_ID"], np.dtype("S12"))
        self.assertEqual(dat.dtype["RA"], np.dtype("f8"))
        npt.assert_array_equal(dat["N_EPOCH"], [1, 2, 70000, 3])
        npt.assert_array_equal(
            dat["TILE_ID"],
            [b"123.456", b"123.457", b"999888.7776", b"123.458"],
        )
        self.assertEqual(dat["RA"][2], 3.123456789)

        # campaign_shape must report the same promoted dtype, since the merge
        # preallocates from it.
        n_rows, dtype_out = catalog.campaign_shape(str(path), param_list=param_list)
        self.assertEqual(n_rows, 4)
        self.assertEqual(dtype_out, dat.dtype)

    def test_iter_campaign_tiles(self):
        """The streaming reader yields one restricted tile at a time."""
        path = self._dir / "final_cat_CAMPAIGN.hdf5"
        write_campaign(path, "legacy", self._tiles)

        tiles = list(
            catalog.iter_campaign_tiles(str(path), param_list=["RA"], verbose=False)
        )

        self.assertEqual([len(tile) for tile in tiles], [3, 2])
        for tile in tiles:
            self.assertEqual(tile.dtype.names, ("RA",))
        npt.assert_array_equal(
            np.concatenate([tile["RA"] for tile in tiles]),
            self._expected()["RA"],
        )

    def test_ambiguous_layout_raises(self):
        path = self._dir / "ambiguous.hdf5"
        with h5py.File(path, "w") as f:
            f.create_group("tiles")
            f.create_group("other")

        with self.assertRaises(ValueError):
            catalog.read_campaign_catalogue(path, verbose=False)


class TestStarCatalogueReader(unittest.TestCase):
    """Campaign star catalogue reader."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._dir = Path(self._tmp.name)

        dtype = np.dtype([(name, "f8") for name in catalog.STAR_CAT_COLUMNS])
        self._exposures = {
            "2110000p": np.zeros(4, dtype=dtype),
            "2110001p": np.ones(6, dtype=dtype),
        }

        self._path = self._dir / "full_starcat_CAMPAIGN.hdf5"
        with h5py.File(self._path, "w") as f:
            group = f.create_group("exposures")
            for exp, dat in self._exposures.items():
                group.create_dataset(exp, data=dat)

    def tearDown(self):
        self._tmp.cleanup()

    def test_read_hdf5(self):
        dat = catalog.read_star_catalogue(self._path, verbose=False)

        self.assertEqual(len(dat), 10)
        self.assertEqual(tuple(dat.dtype.names), catalog.STAR_CAT_COLUMNS)
        npt.assert_array_equal(dat["MAG"][:4], np.zeros(4))
        npt.assert_array_equal(dat["MAG"][4:], np.ones(6))

    def test_read_fits(self):
        from astropy.io import fits

        fits_path = self._dir / "full_starcat-0000000.fits"
        dat = np.concatenate(list(self._exposures.values()))
        fits.BinTableHDU(data=dat).writeto(fits_path)

        out = catalog.read_star_catalogue(str(fits_path), verbose=False)

        self.assertEqual(len(out), 10)
        npt.assert_array_equal(np.asarray(out["MAG"]), dat["MAG"])


class TestMaskCut(unittest.TestCase):
    """Mask-column galaxy selection cut."""

    def setUp(self):
        self._dat = make_galaxy_data(5)

    def test_default_columns(self):
        self._dat["MASK_n4"][0] = True
        self._dat["MASK_n1024"][3] = True

        npt.assert_array_equal(
            galaxy.mask_cut(self._dat),
            np.array([False, True, True, False, True]),
        )

    def test_explicit_column_list(self):
        self._dat["MASK_n4"][0] = True
        self._dat["MASK_n8"][1] = True

        npt.assert_array_equal(
            galaxy.mask_cut(self._dat, ["MASK_n8"]),
            np.array([True, False, True, True, True]),
        )

    def test_empty_column_list_keeps_everything(self):
        self._dat["MASK_n4"][:] = True

        npt.assert_array_equal(
            galaxy.mask_cut(self._dat, []), np.ones(len(self._dat), dtype=bool)
        )

    def test_missing_column_raises(self):
        with self.assertRaises(KeyError) as ctx:
            galaxy.mask_cut(self._dat, ["MASK_n16"])
        self.assertIn("MASK_n16", str(ctx.exception))

    def test_v1_catalogue_raises(self):
        dat = np.zeros(3, dtype=[("IMAFLAGS_ISO", "i2")])

        with self.assertRaises(KeyError):
            galaxy.mask_cut(dat)


class TestCampaignMerge(unittest.TestCase):
    """Merge of several campaign catalogues into a joint catalogue."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._dir = Path(self._tmp.name)

        self._paths = []
        for name, layout, n_obj in (("W3", "legacy", 3), ("SGC", "flat", 2)):
            path = self._dir / f"final_cat_{name}.hdf5"
            write_campaign(path, layout, {"000.000": make_galaxy_data(n_obj)})
            self._paths.append(str(path))

        self._obj = JointCat()
        self._obj._params["verbose"] = False

    def tearDown(self):
        self._tmp.cleanup()

    def test_campaign_name(self):
        self.assertEqual(JointCat.campaign_name("/some/dir/final_cat_W3.hdf5"), "W3")

    def test_merge(self):
        dat = self._obj.merge_catalogues(self._paths)

        self.assertEqual(len(dat), 5)
        self.assertIn("campaign", dat.dtype.names)
        npt.assert_array_equal(dat["campaign"], np.array([b"W3"] * 3 + [b"SGC"] * 2))
        npt.assert_array_equal(dat["RA"][:3], np.arange(3))

    def test_merge_promotes_column_widths(self):
        """A wider string/int column in a later file is not truncated."""
        dtype_narrow = np.dtype([("RA", "f8"), ("TILE_ID", "S7"), ("N", "i4")])
        dtype_wide = np.dtype([("RA", "f8"), ("TILE_ID", "S12"), ("N", "i8")])

        narrow = np.zeros(1, dtype=dtype_narrow)
        narrow["TILE_ID"] = b"123.456"
        narrow["N"] = 7
        wide = np.zeros(1, dtype=dtype_wide)
        wide["TILE_ID"] = b"999888.7776"
        wide["N"] = 2**40

        path_a = self._dir / "final_cat_AA.hdf5"
        path_b = self._dir / "final_cat_BBBBBBBB.hdf5"
        write_campaign(path_a, "flat", {"000.000": narrow})
        write_campaign(path_b, "flat", {"000.000": wide})

        for paths in ([path_a, path_b], [path_b, path_a]):
            dat = self._obj.merge_catalogues([str(path) for path in paths])
            by_campaign = {name: row for name, row in zip(dat["campaign"], dat)}
            self.assertEqual(by_campaign[b"BBBBBBBB"]["TILE_ID"], b"999888.7776")
            self.assertEqual(by_campaign[b"BBBBBBBB"]["N"], 2**40)
            self.assertEqual(by_campaign[b"AA"]["TILE_ID"], b"123.456")

    def test_merge_reduce_mem_overflow_raises(self):
        """reduce_mem never silently wraps out-of-range values."""
        dat = np.zeros(3, dtype=[("RA", "f8"), ("N_EPOCH", "i4")])
        dat["N_EPOCH"] = [3, 200, 300000]
        path = self._dir / "final_cat_Z.hdf5"
        write_campaign(path, "flat", {"000.000": dat})

        self._obj._params["reduce_mem"] = True
        with self.assertRaises(ValueError) as ctx:
            self._obj.merge_catalogues([str(path)])
        self.assertIn("N_EPOCH", str(ctx.exception))

    def test_merge_reduce_mem_in_range_ok(self):
        dat = np.zeros(2, dtype=[("RA", "f8"), ("N_EPOCH", "i4")])
        dat["N_EPOCH"] = [3, 200]
        path = self._dir / "final_cat_Y.hdf5"
        write_campaign(path, "flat", {"000.000": dat})

        self._obj._params["reduce_mem"] = True
        out = self._obj.merge_catalogues([str(path)])

        npt.assert_array_equal(out["N_EPOCH"], [3, 200])
        self.assertEqual(out.dtype["RA"], np.dtype("f8"))  # RA keeps precision

    def test_merge_rejects_multidimensional_column(self):
        dat = np.zeros(2, dtype=[("RA", "f8"), ("XY", "f8", (2,))])
        path = self._dir / "final_cat_M.hdf5"
        write_campaign(path, "flat", {"000.000": dat})

        with self.assertRaises(ValueError) as ctx:
            self._obj.merge_catalogues([str(path)])
        self.assertIn("multi-dimensional", str(ctx.exception))

    def test_merge_incompatible_columns_raises(self):
        other = self._dir / "final_cat_X.hdf5"
        dat = np.zeros(2, dtype=[("RA", "f8")])
        write_campaign(other, "flat", {"000.000": dat})

        with self.assertRaises(ValueError):
            self._obj.merge_catalogues(self._paths + [str(other)])

    def test_no_input_raises(self):
        with self.assertRaises(ValueError):
            self._obj.get_input_paths()


if __name__ == "__main__":
    unittest.main()
