"""Sphinx extension writing ``catalogue_columns.md`` from the column schema.

The page is generated at ``builder-inited`` from
``config/columns/shapepipe_v2.yaml``, so the documentation cannot drift from the
file the code is checked against.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = ROOT / "config" / "columns" / "shapepipe_v2.yaml"
PAGE = "catalogue_columns.md"

TITLES = {
    "galaxy": "Galaxy catalogue",
    "comprehensive": "Comprehensive catalogue",
    "star": "Star catalogue",
    "shear": "Shear catalogue",
}

LEDE = """\
# Catalogue columns

These are the catalogue columns sp_validation reads by fixed name. Their
canonical names are ShapePipe v2's. Every other column it reads is named in
configuration: the cosmo_val `cat_config.yaml` `*_col` keys pick the shear,
weight, PSF and star columns; masking configs name the columns they cut on
(`col_name`); `mask_columns` overrides the default mask columns listed below.

This page is generated from `config/columns/shapepipe_v2.yaml`, which a test
keeps in step with the code.

## Reading a catalogue with other column names

Give the catalogue's configuration a `column_map`,
`{canonical name: name in the file}`. The reader presents the file's columns
under the canonical names, lazily and without copying the data, so nothing
downstream changes. One `*` on each side maps a family of columns at once;
explicit entries take precedence over patterns.

```yaml
# cat_config.yaml: a shear, psf or star block
shear:
  path: my_catalogue.hdf5
  column_map:
    RA: ALPHA_J2000
    Dec: DELTA_J2000

# calibration config (config/calibration/*.yaml), under params:
params:
  input_path: my_comprehensive.hdf5
  column_map:
    "NGMIX_*": "MYFIT_*"
```

The calibration scripts take `galaxy_column_map` and `star_column_map` in
`params.py`. The `*_col` keys keep naming columns as the reader presents them,
that is after the map.

The container is detected from the file contents: FITS (one table HDU, the
first unless `hdu` says otherwise), or HDF5 holding one dataset, a
comprehensive `data` dataset (joined with `data_ext` when present), or a group
of per-tile or per-exposure datasets, as ShapePipe writes them.

## ShapePipe v1 products

Catalogues released up to v1.6.x name their columns in ShapePipe's v1
grammar (`SIGMA_PSF_HSM`, the 2-vector `NGMIX_ELL_NOSHEAR`, ...). They need no
map: the reader detects them and presents the names below, converting where
the v1 quantity differs (sigma to `T = 2 sigma^2`, 2-vectors to components).
`sp_validation.grammar` holds the rules. A `column_map` applies before that
detection and overrides any rule for the same canonical name.
"""


def render(schema):
    """Return the page text for the parsed ``schema``."""
    parts = [LEDE]
    for kind, entry in schema.items():
        parts.append(f"\n## {TITLES.get(kind, kind.capitalize())}\n")
        parts.append(f"{entry['about'].strip()}\n")
        parts.append("| Column | Meaning |\n|---|---|")
        for name, meaning in entry["columns"].items():
            parts.append(f"| `{name}` | {meaning} |")
    return "\n".join(parts) + "\n"


def write_page(app):
    schema = yaml.safe_load(SCHEMA.read_text())
    (Path(app.srcdir) / PAGE).write_text(render(schema))


def setup(app):
    app.connect("builder-inited", write_page)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
