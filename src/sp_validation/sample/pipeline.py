"""Build a calibrated shear sample from a recipe.

A recipe (``config/calibration/sample/``) names the snapshot, its shape
catalogue, the selection config and the output directory. ``run`` makes two
passes over the catalogue's HEALPix pixels, in parallel:

1. :func:`measure` selects each pixel's galaxies and reduces them to per-cell
   sums (``response.cell_moments``, ``response.cell_sums``) and a per-tile cut
   flow, staging the selected rows. The sums of all pixels give the DES weight
   table, the response ``R`` and the additive bias ``c_e`` of the whole sample.
2. :func:`write` gives each staged row its weight and calibrated ellipticity
   ``R⁻¹ (g − c_e)`` and writes its pixel of the HATS catalogue.

The pixels of the output are those of the input, so no row is ever moved
between files and the catalogue is never held in memory.

Products, under the output directory, laid out as a release:

- ``catalogs/calibrated/``: the HATS catalogue (:data:`COLUMNS`), its
  properties recording the snapshot (:mod:`.snapshot`);
- ``catalogs/response.json``: ``R`` and its shear and selection parts, the
  additive bias, the branch populations and the row counts;
- ``catalogs/des_weights.fits``: the weight table, a product
  ``calibration.load_des_weight_product`` reads;
- ``catalogs/survey.json`` and ``masks/footprint-nside<N>-ring.fits``: area,
  n_eff and σ_e, and the occupied pixels they are taken over;
- ``catalogs/config_mask.yaml`` and ``provenance/calibrated_recipe.yaml``: the
  selection config and the recipe, byte for byte;
- ``provenance/calibrated_tile_accounting.json``: rows of every snapshot tile
  after each cut.
"""

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import healpy as hp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from hats.catalog import PartitionInfo, TableProperties
from hats.io import paths, write_parquet_metadata
from hats.io.skymap import write_skymap
from hats.pixel_math import HealpixPixel
from hats.pixel_math.spatial_index import (
    SPATIAL_INDEX_COLUMN,
    SPATIAL_INDEX_ORDER,
    compute_spatial_index,
)

from .. import calibration, io, masks, response
from ..data_roots import load_yaml
from ..galaxy import classification_galaxy_overlap_ra_dec
from ..object_id import OBJECT_ID, object_id
from ..survey import effective_survey_stats
from . import snapshot

ROOT = Path(__file__).parents[3]

#: The catalogue's columns. ``e1``, ``e2`` are calibrated, ``w_des`` is the
#: shear weight, ``e1_PSF``, ``e2_PSF`` the PSF ellipticity at the galaxy,
#: ``TILE_KEY`` the tile as an integer (194248 for tile 194.248).
COLUMNS = (
    "RA",
    "Dec",
    "e1",
    "e2",
    "w_des",
    "e1_PSF",
    "e2_PSF",
    "SIZE_RATIO",
    "SNR",
    "N_EPOCH",
    "TILE_KEY",
    "TILE_ID",
    "NUMBER",
    OBJECT_ID,
    SPATIAL_INDEX_COLUMN,
)

_SHEARS = ("NOSHEAR", "1P", "1M", "2P", "2M")
_METACAL = [
    f"NGMIX_{quantity}_{shear}"
    for shear in _SHEARS
    for quantity in ("FLAGS", "G1", "G2", "FLUX", "FLUX_ERR", "T", "T_PSF_RECONV")
] + ["NGMIX_T_ERR_NOSHEAR", "NGMIX_G1_ERR_NOSHEAR", "NGMIX_G2_ERR_NOSHEAR"]
_METACAL += [f"NGMIX_T_ERR_{shear}" for shear in _SHEARS[1:]]
_SOURCE = {
    "RA": "XWIN_WORLD",
    "Dec": "YWIN_WORLD",
    "mag": "MAG_AUTO",
    "e1_PSF": "NGMIX_G1_PSF_ORIG_NOSHEAR",
    "e2_PSF": "NGMIX_G2_PSF_ORIG_NOSHEAR",
}
_METACAL_STAGE = "metacal (S/N, size, flags)"


def tile_key(tile_id):
    """Return tile IDs (strings ``"194.248"`` or floats) as integers, 194248."""
    return np.rint(np.asarray(tile_id, dtype=np.float64) * 1000).astype(np.int64)


def source_columns(selection):
    """Return the catalogue columns the build of ``selection`` reads."""
    cuts = [cut["col_name"] for cut in masks.catalogue_cuts(selection)]
    names = [name for name in cuts if name != "overlap" and name not in _SOURCE]
    names += ["TILE_ID", "NUMBER", "N_EPOCH", *_SOURCE.values(), *_METACAL]
    return sorted(set(names))


def with_derived(data, tiles):
    """Return ``data``'s columns with those the selection and the sample derive.

    These are the comprehensive catalogue's (``scripts/calibration/
    extract_info.py``): ``RA``, ``Dec``, ``mag``, ``e1_PSF``, ``e2_PSF``, and
    ``overlap``, True for an object kept by the tile-overlap cut.

    @sc [decision:dr6_catalogue.overlap_cut] overlap-neighbours-are-snapshot-tiles
    A tile's neighbours are looked up among ``tiles``, every tile of the
    snapshot, never among the tiles of the rows in hand.
    """
    columns = {name: data[name] for name in data.dtype.names}
    columns.update({name: data[source] for name, source in _SOURCE.items()})
    columns["overlap"] = classification_galaxy_overlap_ra_dec(data, tiles=tiles)
    return columns


def table_edges(metacal):
    """Return the (S/N, size) edges of the weight table of a ``metacal``
    selection block: logarithmic cells between its cuts."""
    bins = metacal.get("des_num_bins", 20)
    return (
        np.logspace(
            *np.log10([metacal["gal_snr_min"], metacal["gal_snr_max"]]), bins + 1
        ),
        np.logspace(
            *np.log10([metacal["gal_rel_size_min"], metacal["gal_rel_size_max"]]),
            bins + 1,
        ),
    )


def select(data, selection, tiles):
    """Select the galaxies of ``data`` and reduce them to sums and staged rows.

    Parameters
    ----------
    data : numpy.ndarray
        rows of the shape catalogue, with :func:`source_columns`
    selection : dict
        selection config: the ``dat`` cuts and the ``metacal`` block
    tiles : list of str
        every tile ID of the snapshot, sorted

    Returns
    -------
    dict
        ``flow`` (stage, tile), the rows of each tile left after each cut;
        ``moments`` and ``sums``, the weight-table and response sums
        (:mod:`sp_validation.response`); ``step``, the metacal shear step;
        and ``rows``, the selected galaxies with their uncalibrated shapes
        ``g1``, ``g2``

    @sc [label:invariant] sample-sums-add-over-chunks
    ``moments`` and ``sums`` of disjoint sets of rows add to those of their
    union, so the sample's weights, response and additive bias do not depend
    on how the catalogue is cut into pixels.
    """
    keys = tile_key(tiles)
    tile = np.searchsorted(keys, tile_key(data["TILE_ID"]))
    if np.any(keys[np.minimum(tile, len(keys) - 1)] != tile_key(data["TILE_ID"])):
        raise ValueError("the catalogue holds a tile the snapshot does not list")

    def per_tile(index):
        return np.bincount(index, minlength=len(keys))

    columns = with_derived(data, tiles)
    keep = np.ones(len(data), dtype=bool)
    flow = [per_tile(tile)]
    for cut in masks.get_masks_from_config(selection, columns)[0]:
        keep &= np.asarray(cut._mask, dtype=bool)
        flow.append(per_tile(tile[keep]))

    with np.errstate(divide="ignore", invalid="ignore"):
        gm = calibration.metacal_from_config(
            data, keep, selection["metacal"], global_response=False
        )
        branches = {b: getattr(gm, b) for b in response.BRANCHES}
        coordinates = {
            b: (d["flux"] / d["flux_err"], d["T"] / d["Tpsf"])
            for b, d in branches.items()
        }
    shapes = {b: (d["g1"], d["g2"]) for b, d in branches.items()}
    ns = gm.mask_dict["ns"]
    snr, size = (x[ns] for x in coordinates["ns"])
    g1, g2 = (x[ns] for x in shapes["ns"])
    edges = table_edges(selection["metacal"])
    rows = {name: columns[name][keep][ns] for name in ("RA", "Dec", "e1_PSF", "e2_PSF")}
    rows.update(
        {name: data[name][keep][ns] for name in ("N_EPOCH", "TILE_ID", "NUMBER")}
    )
    rows.update(g1=g1, g2=g2, SNR=snr, SIZE_RATIO=size)
    flow.append(per_tile(tile[keep][ns]))
    return dict(
        flow=np.array(flow),
        moments=response.cell_moments(
            snr, size, np.array([g1, g2]), (gm.R11 + gm.R22) / 2, *edges
        ),
        sums=response.cell_sums(shapes, gm.mask_dict, coordinates, gm.R_shear, *edges),
        step=gm._step,
        rows=rows,
    )


def calibrate(rows, table, estimate):
    """Return the catalogue columns of staged ``rows``.

    @sc [decision:dr6_catalogue.additive_bias,decision:dr6_catalogue.leakage_correction] calibrated-e-is-inverse-R-of-centred-g
    ``(e1, e2) = R⁻¹ (g − c_e)`` with the sample's one ``R`` and ``c_e``, and
    ``w_des`` is the sample's table at the galaxy's own S/N and size.
    """
    g = np.array([rows["g1"], rows["g2"]])
    e1, e2 = np.linalg.inv(estimate["total"]) @ (g - estimate["c_e"][:, None])
    healpix = compute_spatial_index(rows["RA"], rows["Dec"])
    out = dict(
        rows,
        e1=e1,
        e2=e2,
        w_des=response.evaluate_w(table, rows["SNR"], rows["SIZE_RATIO"]),
        N_EPOCH=np.asarray(rows["N_EPOCH"], dtype=np.int16),
        TILE_KEY=tile_key(rows["TILE_ID"]).astype(np.int32),
    )
    out[OBJECT_ID] = object_id(rows["TILE_ID"], rows["NUMBER"])
    out[SPATIAL_INDEX_COLUMN] = healpix
    order = np.argsort(healpix, kind="stable")
    return pa.table({name: out[name] for name in COLUMNS}).take(order)


_state = {}


def _start(catalogue, selection, tiles, staging):
    _state.update(
        catalogue=io.Catalogue(catalogue),
        selection=selection,
        tiles=tiles,
        staging=Path(staging),
        columns=source_columns(selection),
    )


def _staged(pixel):
    return _state["staging"] / (pixel.replace("/", "-") + ".parquet")


def measure(pixel):
    """Pass 1 of one pixel: stage its selected rows and return its sums."""
    data = _state["catalogue"].chunks[pixel].to_structured(_state["columns"])
    result = select(data, _state["selection"], _state["tiles"])
    pq.write_table(pa.table(result.pop("rows")), _staged(pixel))
    return result


def write(job):
    """Pass 2 of one pixel: write its file of the catalogue, return its rows."""
    pixel, target, table, estimate = job
    rows = pq.read_table(_staged(pixel)).to_pydict()
    rows = {name: np.asarray(column) for name, column in rows.items()}
    if not len(rows["RA"]):
        return 0
    order, number = (int(part.split("=")[1]) for part in pixel.split("/"))
    path = paths.pixel_catalog_file(target, HealpixPixel(order, number))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        calibrate(rows, table, estimate),
        path,
        compression="zstd",
        compression_level=3,
        row_group_size=100_000,
    )
    return len(rows["RA"])


def write_hats_metadata(target, pixels, record):
    """Make the pixel files under ``target`` a HATS catalogue recording the
    snapshot ``record``. ``pixels`` are the chunk names of the files written."""
    healpix = [
        HealpixPixel(*(int(part.split("=")[1]) for part in pixel.split("/")))
        for pixel in pixels
    ]
    PartitionInfo.from_healpix(healpix).write_to_file(catalog_path=target)
    rows = write_parquet_metadata(target)
    order = max(pixel.order for pixel in healpix)
    TableProperties(
        catalog_name=Path(target).name,
        catalog_type="object",
        total_rows=rows,
        ra_column="RA",
        dec_column="Dec",
        healpix_column=SPATIAL_INDEX_COLUMN,
        healpix_order=SPATIAL_INDEX_ORDER,
        skymap_order=order,
        **{snapshot.PROPERTIES[name]: value for name, value in record.items()},
        **TableProperties.new_provenance_dict(target, builder="sp_validation.sample"),
    ).to_properties_file(target)
    index = io.read_catalogue(target, columns=[SPATIAL_INDEX_COLUMN])
    cells = index[SPATIAL_INDEX_COLUMN] >> (2 * (SPATIAL_INDEX_ORDER - order))
    write_skymap(np.bincount(cells, minlength=hp.order2npix(order)), target)
    return rows


def survey(target, nside, footprint):
    """Write the footprint of the catalogue ``target`` and return its survey
    statistics: the pixels of ``nside`` holding a galaxy, and the area, n_eff
    and σ_e (per component) over them."""
    data = io.read_catalogue(target, columns=["RA", "Dec", "e1", "e2", "w_des"])
    occupied = np.unique(hp.ang2pix(nside, data["RA"], data["Dec"], lonlat=True))
    area = float(len(occupied) * hp.nside2pixarea(nside, degrees=True))
    mask = np.zeros(hp.nside2npix(nside), dtype=np.uint8)
    mask[occupied] = 1
    footprint.parent.mkdir(parents=True, exist_ok=True)
    hp.write_map(
        footprint,
        mask,
        nest=False,
        dtype=np.uint8,
        extra_header=[("COORDSYS", "C"), ("MASKTYPE", "OCCUPIED")],
    )
    stats = effective_survey_stats(data["e1"], data["e2"], data["w_des"], area)
    stats = {name: float(value) for name, value in stats.items()}
    return dict(rows=len(data), area_deg2=area, nside=nside, samples={"all": stats})


def accounting(flow, stages, tiles):
    """Return the tile accounting of a cut ``flow`` (stage, tile): the rows of
    every tile after each of ``stages``, and for each tile left without a
    galaxy the first cut that empties it."""
    emptied = {}
    for column in flow.T[flow[-1] == 0].T.T:
        stage = stages[int(np.argmax(column == 0))]
        emptied[stage] = emptied.get(stage, 0) + 1
    return dict(
        tiles=len(tiles),
        with_selected=int(np.count_nonzero(flow[-1])),
        empty_by_first_emptying_cut=emptied,
        totals={stage: int(total) for stage, total in zip(stages, flow.sum(axis=1))},
        stages=stages,
        rows={tile: column.tolist() for tile, column in zip(tiles, flow.T)},
    )


def _dump(path, value):
    def plain(x):
        return x.tolist() if isinstance(x, np.ndarray) else x.item()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=plain) + "\n")


def run(recipe, output=None, *, workers=1):
    """Build the sample of the recipe file ``recipe``.

    Products go under the recipe's ``output`` directory, or under ``output``.
    The catalogue is never overwritten: a directory already holding one is
    refused.

    @sc [decision:dr6_catalogue.snapshot_record] sample-is-the-whole-snapshot
    The catalogue read is the snapshot's: it declares the manifest's SHA-256
    as its source, and holds exactly the manifest's object count of every
    tile. The sample then records that snapshot.
    """
    config = load_yaml(recipe)
    output = Path(output or config["output"])
    catalogs = output / "catalogs"
    target = catalogs / "calibrated"
    if target.exists():
        raise FileExistsError(f"{target} exists; a sample is built in a new place")
    selection_file = ROOT / config["selection"]
    selection = yaml.safe_load(selection_file.read_text())
    metacal = selection["metacal"]

    snap = snapshot.read(config["snapshot"])
    tiles = sorted(snap["objects"])
    source = io.hats_catalogue_dir(config["catalogue"])
    declared = io._properties(Path(source) / "hats.properties")
    if declared.get("source_manifest_sha256") != snap["manifest_sha256"]:
        raise ValueError(
            f"{source} was not built from the manifest of {config['snapshot']}"
        )

    staging = catalogs / "calibrated.staging"
    staging.mkdir(parents=True)
    with io.Catalogue(source) as catalogue:
        pixels = list(catalogue.chunks)
    start = (source, selection, tiles, staging)
    with ProcessPoolExecutor(workers, initializer=_start, initargs=start) as pool:
        total = None
        for done, result in enumerate(pool.map(measure, pixels), 1):
            step = result.pop("step")
            total = result if total is None else response.add_sums(total, result)
            if done % 100 == 0:
                print(f"measured {done}/{len(pixels)} pixels", flush=True)

        flow = total["flow"]
        expected = np.array([snap["objects"][tile] for tile in tiles])
        if not np.array_equal(flow[0], expected):
            differ = [tile for tile, a, b in zip(tiles, flow[0], expected) if a != b]
            raise ValueError(
                f"{len(differ)} tiles hold other rows than the manifest: {differ[:5]}"
            )
        table = response.table_from_moments(
            *total["moments"],
            *table_edges(metacal),
            min_count=metacal.get("des_min_count", 3),
        )
        estimate = response.responses_from_sums(total["sums"], table, step)

        jobs = [(pixel, target, table, estimate) for pixel in pixels]
        written = list(pool.map(write, jobs))
    shutil.rmtree(staging)
    selected = estimate["counts"]["ns"]
    if sum(written) != selected:
        raise ValueError(f"wrote {sum(written)} rows, selected {selected}")
    kept = [pixel for pixel, rows in zip(pixels, written) if rows]
    write_hats_metadata(target, kept, snapshot.record(snap))

    stages = ["catalogue"]
    stages += [cut["label"] for cut in masks.catalogue_cuts(selection)]
    stages += [_METACAL_STAGE]
    result = dict(
        R=estimate["total"],
        R_shear=estimate["shear"],
        R_selection=estimate["selection"],
        R_joint=estimate["joint"],
        c_e=estimate["c_e"],
        c_g=estimate["c_g"],
        branch_counts=estimate["counts"],
        branch_weight_sums=estimate["weight_sums"],
        raw_rows=int(flow[0].sum()),
        preselection=int(flow[-2].sum()),
        selected_rows=selected,
    )
    _dump(catalogs / "response.json", result)
    calibration.save_des_weight_product(
        catalogs / "des_weights.fits", table, metacal.get("des_min_count", 3)
    )
    nside = config["footprint_nside"]
    footprint = output / "masks" / f"footprint-nside{nside}-ring.fits"
    _dump(catalogs / "survey.json", survey(target, nside, footprint))
    _dump(
        output / "provenance" / "calibrated_tile_accounting.json",
        accounting(flow, stages, tiles),
    )
    shutil.copyfile(selection_file, catalogs / "config_mask.yaml")
    shutil.copyfile(recipe, output / "provenance" / "calibrated_recipe.yaml")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recipe", help="recipe file (config/calibration/sample/)")
    parser.add_argument(
        "--output", help="directory to write under (default: the recipe's)"
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    result = run(args.recipe, args.output, workers=args.workers)
    print(json.dumps({k: result[k] for k in ("selected_rows", "raw_rows")}))
