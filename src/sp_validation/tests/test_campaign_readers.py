"""Tests for the ShapePipe v2 campaign catalogue readers and mask cut."""

import unittest

import h5py
import numpy as np
import numpy.testing as npt

from sp_validation import catalog, galaxy
from sp_validation.catalog_builders import JointCat

import tempfile
from pathlib import Path

GAL_DTYPE = np.dtype(
    [
        ("RA", "f8"),
        ("Dec", "f8"),
        ("MAG_AUTO", "f4"),
        ("MASK_n4", "?"),
        ("MASK_n1", "?"),
        ("MASK_n2", "?"),
        ("MASK_n8", "?"),
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
        return np.concatenate(list(self._tiles.values()))

    def test_legacy_layout(self):
        path = self._dir / "final_cat_CAMPAIGN.hdf5"
        write_campaign(path, "legacy", self._tiles)

        dat = catalog.read_campaign_catalogue(path, verbose=False)

        self.assertEqual(len(dat), 5)
        npt.assert_array_equal(np.sort(dat["RA"]), np.sort(self._expected()["RA"]))

    def test_flat_layout(self):
        path = self._dir / "final_cat_CAMPAIGN.hdf5"
        write_campaign(path, "flat", self._tiles)

        dat = catalog.read_campaign_catalogue(path, verbose=False)

        self.assertEqual(len(dat), 5)
        npt.assert_array_equal(np.sort(dat["RA"]), np.sort(self._expected()["RA"]))

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
        self.assertEqual(
            JointCat.campaign_name("/some/dir/final_cat_W3.hdf5"), "W3"
        )

    def test_merge(self):
        dat = self._obj.merge_catalogues(self._paths)

        self.assertEqual(len(dat), 5)
        self.assertIn("campaign", dat.dtype.names)
        npt.assert_array_equal(
            dat["campaign"], np.array([b"W3"] * 3 + [b"SGC"] * 2)
        )
        npt.assert_array_equal(dat["RA"][:3], np.arange(3))

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
