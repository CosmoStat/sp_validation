"""Rule cv_footprints: plot per-version survey footprints.

Leaf rule. Reads the shear catalog (RA/Dec) for each version; writes
footprint_{version}_{region}.png. Sentinel output keeps the DAG trackable
because region names are internal to FootprintPlotter.
"""

from cv_runner import _unbuffer_streams, make_cv, touch_sentinels
from snakemake.script import snakemake

_unbuffer_streams()
cv = make_cv(snakemake)
cv.plot_footprints()
touch_sentinels(snakemake)
