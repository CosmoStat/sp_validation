"""The v2 catalogue as one HATS collection: shapes, their ID index, and PhotoPipe.

::

    COLLECTION/
    ├── shapes/                primary: the shape catalogue with OBJECT_ID
    │                          (sp_validation.object_id.add_to_catalogue)
    ├── shapes_OBJECT_ID/      index: OBJECT_ID -> pixel, the default index
    ├── photopipe/             extension data, a collection of its own, holding
    │   ├── photopipe/         the PhotoPipe catalogue (sp_validation.photopipe_merge)
    │   └── collection.properties
    ├── photopipe.properties   the extension, joined to the primary on OBJECT_ID
    └── collection.properties

This is the layout ``hats_import``'s extension split writes. Both catalogues
have the same pixels, and each PhotoPipe pixel file holds the rows of the shape
file in the same order.
"""

from pathlib import Path

import pyarrow.parquet as pq
from hats.catalog import CollectionProperties, ExtensionProperties
from hats_import.collection.arguments import CollectionArguments
from hats_import.pipeline import pipeline

from sp_validation.object_id import OBJECT_ID
from sp_validation.photopipe_merge import KEYS
from sp_validation.photopipe_merge import NAME as EXTENSION

PRIMARY = "shapes"


def build(collection, workers=1):
    """Index ``COLLECTION/shapes`` on ``OBJECT_ID`` and add the PhotoPipe extension.

    ``COLLECTION/shapes`` and ``COLLECTION/photopipe/photopipe`` exist; this
    writes the index and every properties file of the collection. An index
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

    extension = collection / EXTENSION
    CollectionProperties(
        name=EXTENSION,
        hats_primary_table_url=EXTENSION,
        **args.extra_property_dict(extension),
    ).to_properties_file(extension)
    columns = pq.read_schema(extension / EXTENSION / "dataset/_common_metadata").names
    ExtensionProperties(
        name=EXTENSION,
        primary_catalog=collection.name,
        primary_column=OBJECT_ID,
        join_catalog=EXTENSION,
        join_column=OBJECT_ID,
        extension_columns=[c for c in columns if c not in KEYS],
        extension_join_style="left",
        extension_product_type="photometry",
        shares_primary_coordinates=True,
        **args.extra_property_dict(extension),
    ).to_properties_file(collection)
    properties = CollectionProperties.read_from_dir(collection)
    properties.model_copy(
        update=dict(
            default_index=OBJECT_ID,
            all_extensions={EXTENSION: f"{EXTENSION}.properties"},
        )
    ).to_properties_file(collection)
