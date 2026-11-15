"""The v2 catalogue as one HATS collection: shapes, their ID index, PhotoPipe and spec-z.

::

    COLLECTION/
    ├── shapes/                primary: the shape catalogue with OBJECT_ID
    │                          (sp_validation.assembly.object_id)
    ├── shapes_OBJECT_ID/      index: OBJECT_ID -> pixel, the default index
    ├── photopipe/             extension data, a collection of its own, holding
    │   ├── photopipe/         the PhotoPipe catalogue (sp_validation.assembly.photopipe_merge)
    │   └── collection.properties
    ├── specz/                 extension data: the spec-z catalogue
    │   ├── specz/             (sp_validation.assembly.specz_merge)
    │   └── collection.properties
    ├── photopipe.properties   the extensions, each joined to the primary on OBJECT_ID
    ├── specz.properties
    └── collection.properties

This is the layout ``hats_import``'s extension split writes. Each extension
file is on a pixel of the shapes. A PhotoPipe file holds the rows of the shape
file in the same order; a spec-z file holds only the objects with a spectrum,
in the same order, and a pixel without one has no file.

``build`` writes it from a recipe (``config/assembly/``) naming the inputs and
the collection:

- ``shapes``: the shape HATS catalogue;
- ``photopipe_extract``: the PhotoPipe column extract
  (``scripts/assembly/extract_photopipe.py``);
- ``specz_matches``, ``specz_compilation``: PhotoPipe's match of DR6 objects
  to spectra and the redshift compilation it used
  (``sp_validation.assembly.specz_merge``);
- ``collection``: the directory to write.
"""

from pathlib import Path

import pyarrow.parquet as pq
from hats.catalog import CollectionProperties, ExtensionProperties, TableProperties
from hats_import.collection.arguments import CollectionArguments
from hats_import.pipeline import pipeline

from sp_validation import data_roots
from sp_validation.assembly import object_id, photopipe_merge, specz_merge
from sp_validation.assembly.object_id import OBJECT_ID
from sp_validation.assembly.photopipe_merge import KEYS

PRIMARY = object_id.NAME
#: Each extension's name and the kind of data it serves.
EXTENSIONS = {photopipe_merge.NAME: "photometry", specz_merge.NAME: "redshifts"}


def build(recipe, workers=1):
    """Write the collection the recipe file ``recipe`` describes.

    Writes the shapes with ``OBJECT_ID``, then the PhotoPipe extension from
    them, then the spec-z extension from that, then the ``OBJECT_ID`` index
    and every properties file. A part already written (its catalogue's
    ``hats.properties`` exists) is kept, so remove it to rewrite it and what
    is built from it.
    """
    recipe = data_roots.load_yaml(recipe)
    collection = Path(recipe["collection"])

    def written(*parts):
        return (collection.joinpath(*parts) / "hats.properties").is_file()

    if not written(PRIMARY):
        object_id.write_shapes(recipe["shapes"], collection, workers)
    if not written(photopipe_merge.NAME, photopipe_merge.NAME):
        photopipe_merge.merge(collection, recipe["photopipe_extract"], workers)
    if not written(specz_merge.NAME, specz_merge.NAME):
        specz_merge.merge(
            collection,
            recipe["specz_matches"],
            recipe["specz_compilation"],
            workers,
        )
    index(collection, workers)


def index(collection, workers=1):
    """Index ``COLLECTION/shapes`` on ``OBJECT_ID`` and list the extensions.

    Writes the index and every properties file of the collection. An index
    already at ``COLLECTION/shapes_OBJECT_ID`` is kept, so remove it to rebuild.
    """
    collection = Path(collection)
    args = (
        CollectionArguments(
            output_path=collection.parent,
            output_artifact_name=collection.name,
            dask_n_workers=workers,
            progress_bar=False,
        )
        .catalog(output_artifact_name=PRIMARY)
        .add_index(
            indexing_column=OBJECT_ID,
            output_artifact_name=f"{PRIMARY}_{OBJECT_ID}",
            drop_duplicates=False,
        )
    )
    pipeline(args)
    ids = collection / f"{PRIMARY}_{OBJECT_ID}"
    TableProperties.read_from_dir(ids).model_copy(
        update=dict(primary_catalog=PRIMARY)
    ).to_properties_file(ids)

    for name, product in EXTENSIONS.items():
        extension = collection / name
        CollectionProperties(
            name=name,
            hats_primary_table_url=name,
            **args.extra_property_dict(extension),
        ).to_properties_file(extension)
        columns = pq.read_schema(extension / name / "dataset/_common_metadata").names
        ExtensionProperties(
            name=name,
            primary_catalog=collection.name,
            primary_column=OBJECT_ID,
            join_catalog=name,
            join_column=OBJECT_ID,
            extension_columns=[c for c in columns if c not in KEYS],
            extension_join_style="left",
            extension_product_type=product,
            shares_primary_coordinates=True,
            **args.extra_property_dict(extension),
        ).to_properties_file(collection)
    properties = CollectionProperties.read_from_dir(collection)
    properties.model_copy(
        update=dict(
            default_index=OBJECT_ID,
            all_extensions={name: f"{name}.properties" for name in EXTENSIONS},
        )
    ).to_properties_file(collection)
