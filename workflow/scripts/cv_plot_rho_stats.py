"""Rule cv_plot_rho_stats: rho-statistics overlay across versions.

Reads rho_stats_{base}.fits for every version (declared inputs); writes
rho_stats.png into the leakage output dir. Sentinel-tracked because the figure
lands beside the first version's leakage products, not at a fixed path.
"""

from cv_runner import _unbuffer_streams, make_cv, touch_sentinels

# `snakemake` is injected as a module global by Snakemake's `script:` preamble
# before this file runs; no import is needed (and `from snakemake.script
# import snakemake` is IDE-hint-only -- snakemake.script has no such runtime
# attribute and raises ImportError if actually executed).
_unbuffer_streams()
cv = make_cv(snakemake)
cv.plot_rho_stats()
touch_sentinels(snakemake)
