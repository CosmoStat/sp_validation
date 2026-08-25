"""Rule cv_footprints: plot per-version survey footprints.

Leaf rule. Reads the shear catalog (RA/Dec) for each version; writes
footprint_{version}_{region}.png. Sentinel output keeps the DAG trackable
because region names are internal to FootprintPlotter.
"""

from cv_runner import _unbuffer_streams, make_cv, touch_sentinels

# `snakemake` is injected as a module global by Snakemake's `script:` preamble
# before this file runs; no import is needed (and `from snakemake.script
# import snakemake` is IDE-hint-only -- snakemake.script has no such runtime
# attribute and raises ImportError if actually executed).
_unbuffer_streams()
cv = make_cv(snakemake)
cv.plot_footprints()
touch_sentinels(snakemake)
