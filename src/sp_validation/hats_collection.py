"""The v2 catalogue as one HATS collection: shapes, their ID index, PhotoPipe and spec-z.

::

    COLLECTION/
    ├── shapes/                primary: the shape catalogue with OBJECT_ID
    │                          (sp_validation.object_id.add_to_catalogue)
    ├── shapes_OBJECT_ID/      index: OBJECT_ID -> pixel, the default index
    ├── photopipe/             extension data, a collection of its own, holding
    │   ├── photopipe/         the PhotoPipe catalogue (sp_validation.photopipe_merge)
    │   └── collection.properties
    ├── specz/                 extension data: the spec-z catalogue
    │   ├── specz/             (sp_validation.specz_merge)
    │   └── collection.properties
    ├── photopipe.properties   the extensions, each joined to the primary on OBJECT_ID
    ├── specz.properties
    └── collection.properties

This is the layout ``hats_import``'s extension split writes. Each extension
file is on a pixel of the shapes. A PhotoPipe file holds the rows of the shape
file in the same order; a spec-z file holds only the objects with a spectrum,
in the same order, and a pixel without one has no file.
"""

from pathlib import Path

import pyarrow.parquet as pq
from hats.catalog import CollectionProperties, ExtensionProperties, TableProperties
from hats_import.collection.arguments import CollectionArguments
from hats_import.pipeline import pipeline

from sp_validation import photopipe_merge, specz_merge
from sp_validation.object_id import OBJECT_ID
from sp_validation.photopipe_merge import KEYS

PRIMARY = "shapes"
#: Each extension's name and the kind of data it serves.
EXTENSIONS = {photopipe_merge.NAME: "photometry", specz_merge.NAME: "redshifts"}


def build(collection, workers=1):
    """Index ``COLLECTION/shapes`` on ``OBJECT_ID`` and add the extensions.

    ``COLLECTION/shapes`` and the data of every extension in ``EXTENSIONS``
    exist; this writes the index and every properties file of the collection.
    An index already at ``COLLECTION/shapes_OBJECT_ID`` is kept, so remove it
    to rebuild.
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
    index = collection / f"{PRIMARY}_{OBJECT_ID}"
    TableProperties.read_from_dir(index).model_copy(
        update=dict(primary_catalog=PRIMARY)
    ).to_properties_file(index)

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
