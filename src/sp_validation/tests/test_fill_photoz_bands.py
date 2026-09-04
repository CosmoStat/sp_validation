"""``scripts/fill_photoz_bands.py`` must not trust FITS/HDF5 row order blindly.

The script writes PhotoPipe columns into the ShapePipe comprehensive HDF5
catalogue tile by tile, pairing FITS row ``k`` with the ``k``-th HDF5 row of
that tile (in sorted-index order).  Nothing in the file formats guarantees
that pairing -- only the row *count* used to be checked -- so a tile whose
PhotoPipe catalogue happens to be ordered differently would be filled with
silently mismatched photo-z.

These tests build a tiny synthetic pair of catalogues (three tiles,
deliberately interleaved so the non-contiguous write path is exercised) and
run the script end to end:

* the two tiles whose FITS rows line up are filled;
* the tile whose FITS rows are reversed is *skipped*, counted under
  ``Row-order mismatches``, and left at ``EMPTY_VALUE``, so a resumed run
  retries it rather than treating it as done.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
fits = pytest.importorskip("astropy.io.fits")
pytest.importorskip("cs_util")
pytest.importorskip("tqdm")


def _repo_root() -> Path:
    """Locate the repo root by walking up to the ``pyproject.toml`` marker."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("could not locate repo root (no pyproject.toml above test)")


_SCRIPT = _repo_root() / "scripts" / "fill_photoz_bands.py"

# Tile layout: three tiles, rows interleaved round-robin over the catalogue so
# each tile's HDF5 indices are non-contiguous.
TILES = ("100.100", "200.200", "300.300")
N_PER_TILE = 20
N_ROWS = len(TILES) * N_PER_TILE
FILL_KEYS = ("Z_B", "MAG_GAAP_r")


def _module():
    """Import ``scripts/fill_photoz_bands.py`` (lives outside the package)."""
    spec = importlib.util.spec_from_file_location("fill_photoz_bands", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_catalogues(tmp_path, reversed_tile):
    """Write a synthetic HDF5 catalogue and matching PhotoPipe FITS tiles.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory to write into.
    reversed_tile : str
        Tile whose FITS rows are written in reverse order (row-order breakage).

    Returns
    -------
    tuple
        (input HDF5 path, FITS directory, truth dict tile -> (Z_B, MAG_GAAP_r))

    """
    row = np.arange(N_ROWS)
    tile_of_row = np.array([TILES[i % len(TILES)] for i in row])

    dtype = np.dtype([("TILE_ID", "S7"), ("RA", "<f8"), ("Dec", "<f8"), ("w", "<f4")])
    data = np.empty(N_ROWS, dtype=dtype)
    data["TILE_ID"] = np.array([t.encode() for t in tile_of_row])
    # Well-separated positions: 0.01 deg = 36 arcsec apart, far above the
    # 0.5 arcsec default tolerance, so any reordering is unambiguous.
    data["RA"] = 123.0 + 0.01 * row
    data["Dec"] = 12.0 + 0.01 * row
    data["w"] = 1.0

    hdf5_path = tmp_path / "input.hdf5"
    with h5py.File(hdf5_path, "w") as hf:
        hf.create_dataset("dat", data=data)

    fits_dir = tmp_path / "fits"
    fits_dir.mkdir()

    truth = {}
    for tile in TILES:
        sel = np.where(tile_of_row == tile)[0]  # already in HDF5 row order
        z_b = 0.1 + 0.001 * sel
        mag = 20.0 + 0.001 * sel
        truth[tile] = (z_b, mag)

        order = sel[::-1] if tile == reversed_tile else sel
        columns = fits.ColDefs(
            [
                fits.Column(name="ALPHA_J2000", format="D", array=123.0 + 0.01 * order),
                fits.Column(name="DELTA_J2000", format="D", array=12.0 + 0.01 * order),
                fits.Column(name="Z_B", format="E", array=0.1 + 0.001 * order),
                fits.Column(name="MAG_GAAP_r", format="E", array=20.0 + 0.001 * order),
            ]
        )
        hdu_list = fits.HDUList(
            [fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns, name="OBJECTS")]
        )
        hdu_list.writeto(fits_dir / f"UNIONS.{tile}_SP_ugriz_photoz_ext.cat")

    return hdf5_path, fits_dir, truth


def _run(tmp_path, hdf5_path, fits_dir, *extra):
    """Run the script on the synthetic catalogues; return (result, out_path)."""
    output = tmp_path / "output.hdf5"
    checkpoint = tmp_path / "checkpoint.json"
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "-i",
            str(hdf5_path),
            "-o",
            str(output),
            "-d",
            str(fits_dir),
            "-c",
            str(checkpoint),
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    return result, output, checkpoint


@pytest.mark.fast
def test_reversed_tile_is_skipped_matching_tiles_are_filled(tmp_path):
    """A tile with permuted PhotoPipe rows is skipped; the aligned ones fill."""
    bad_tile = TILES[1]
    hdf5_path, fits_dir, truth = _synthetic_catalogues(tmp_path, bad_tile)

    result, output, checkpoint = _run(tmp_path, hdf5_path, fits_dir)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Row-order mismatches  : 1" in result.stdout, result.stdout
    assert "Tiles processed       : 2" in result.stdout, result.stdout
    assert "row order mismatch" in result.stderr, result.stderr

    empty = _module().EMPTY_VALUE
    with h5py.File(output, "r") as hf:
        dset = hf["dat"][:]

    for tile in TILES:
        sel = np.where(dset["TILE_ID"] == tile.encode())[0]
        z_b, mag = truth[tile]
        if tile == bad_tile:
            assert np.all(dset["Z_B"][sel] == empty), (
                f"tile {tile} has permuted PhotoPipe rows but was filled anyway"
            )
            assert np.all(dset["MAG_GAAP_r"][sel] == empty)
        else:
            np.testing.assert_allclose(dset["Z_B"][sel], z_b, rtol=1e-6)
            np.testing.assert_allclose(dset["MAG_GAAP_r"][sel], mag, rtol=1e-6)

    # The skipped tile must not be checkpointed, so a resume retries it.
    done = set(json.loads(checkpoint.read_text())["done_tiles"])
    assert done == {t for t in TILES if t != bad_tile}


@pytest.mark.fast
def test_check_can_be_disabled(tmp_path):
    """``--n_check_rows 0`` restores the old, unchecked behaviour."""
    bad_tile = TILES[1]
    hdf5_path, fits_dir, _ = _synthetic_catalogues(tmp_path, bad_tile)

    result, output, _ = _run(tmp_path, hdf5_path, fits_dir, "--n_check_rows", "0")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Row-order mismatches  : 0" in result.stdout, result.stdout
    assert "Tiles processed       : 3" in result.stdout, result.stdout

    empty = _module().EMPTY_VALUE
    with h5py.File(output, "r") as hf:
        z_b = hf["dat"]["Z_B"][:]
    assert not np.any(z_b == empty), "all tiles should have been written"


@pytest.mark.fast
def test_sample_row_positions_spans_the_tile():
    """The sample hits both ends and the interior, and never runs off the tile."""
    module = _module()

    assert module.sample_row_positions(0, 10).size == 0
    assert module.sample_row_positions(100, 0).size == 0
    np.testing.assert_array_equal(module.sample_row_positions(4, 10), np.arange(4))

    positions = module.sample_row_positions(1000, 10)
    assert positions[0] == 0
    assert positions[-1] == 999
    assert len(positions) == 10
    assert np.all(np.diff(positions) > 0)
    # Not all at the edges: something is sampled from the bulk.
    assert np.any((positions > 10) & (positions < 989))


@pytest.mark.fast
def test_find_fits_radec_columns():
    """The DR6 PhotoPipe names are recognised; unknown tables yield None."""
    module = _module()

    assert module.find_fits_radec_columns(
        ["SeqNr", "ALPHA_J2000", "DELTA_J2000", "Z_B"]
    ) == ("ALPHA_J2000", "DELTA_J2000")
    assert module.find_fits_radec_columns(["RA", "DEC"]) == ("RA", "DEC")
    assert module.find_fits_radec_columns(["X", "Y"]) == (None, None)


@pytest.mark.fast
def test_angular_separation_arcsec():
    """Separation is correct at the pole-free small-angle limit and wraps RA."""
    module = _module()

    sep = module.angular_separation_arcsec([0.0], [0.0], [1.0 / 3600], [0.0])
    np.testing.assert_allclose(sep, [1.0], rtol=1e-6)

    # Across the RA=0 wrap: 359.999 deg vs 0.001 deg is 0.002 deg, not 360.
    sep = module.angular_separation_arcsec([359.999], [0.0], [0.001], [0.0])
    np.testing.assert_allclose(sep, [0.002 * 3600], rtol=1e-6)
