"""Rule cv_objectwise_leakage: object-wise vs scale-dependent PSF leakage.

plot_objectwise_leakage triggers calculate_objectwise_leakage, which itself
triggers calculate_scale_dependent_leakage — the whole leakage chain runs here.
Writes leakage_{version}/ products (the object-wise regression .pkl is the
durable artifact) and leakage_coefficients.png. Sentinel-tracked because the
per-version leakage product paths are built inside shear_psf_leakage.
"""

from snakemake.script import snakemake

from cv_runner import make_cv, touch_sentinels, _unbuffer_streams

_unbuffer_streams()
cv = make_cv(snakemake)
cv.plot_objectwise_leakage()
touch_sentinels(snakemake)
