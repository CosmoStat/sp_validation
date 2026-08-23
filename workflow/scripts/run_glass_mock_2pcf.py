"""Run treecorr on a GLASS mock galaxy catalog.

Computes fine-binned ξ±(θ) for mock validation. Output is a treecorr
GGCorrelation FITS file readable via gg.read(path).

Mock catalogs have columns: RA, Dec, e1, e2, w.
No response correction or PSF leakage subtraction (these are Gaussian mocks).
"""

import treecorr
from astropy.io import fits

catalog_path = snakemake.input.catalog  # noqa: F821
output_path = snakemake.output.gg  # noqa: F821

min_sep = snakemake.params.min_sep  # noqa: F821
max_sep = snakemake.params.max_sep  # noqa: F821
nbins = snakemake.params.nbins  # noqa: F821
mask_file = getattr(snakemake.params, "mask_file", None)  # noqa: F821

# v1.4.8 = v1.4.6.3 + the faint/bright star-halo cuts (masks.txt in
# /n17data/UNIONS/WL/v1.4.x/v1.4.8/).  mask_r_nside131072.hsp is a bit-packed
# BOOLEAN HealSparse map, True where any structural bit (1|2|4|8|64|1024)
# masks the sky; validated 2026-08-23 against the six per-bit _n{bit}.hsp
# maps (99.74% pixel agreement) and the real catalogs: keeps 85.4% of
# v1.4.6.3 rows vs the real v1.4.8's 87.8% (residual = pixelized mask vs
# per-object flags; bits 4/8/64/1024 confirmed already absent from
# v1.4.6.3).  Out-of-coverage returns the sentinel False = "no hole" = keep.


def apply_v148_mask(ra, dec, mask_path):
    """Return the coordinates' keep mask for the v1.4.8 footprint."""
    import healsparse as hsp
    import numpy as np

    survey_mask = hsp.HealSparseMap.read(mask_path)
    mask_values = survey_mask.get_values_pos(ra, dec, lonlat=True)
    if mask_values.dtype != np.bool_:
        raise TypeError(
            f"{mask_path} expected boolean union mask, got {mask_values.dtype}"
        )
    return ~mask_values


# Load mock catalog
with fits.open(catalog_path) as hdul:
    data = hdul["SOURCE_CATALOGUE"].data
    ra = data["RA"]
    dec = data["Dec"]
    e1 = data["e1"]
    e2 = data["e2"]
    w = data["w"]

if mask_file:
    keep = apply_v148_mask(ra, dec, mask_file)
    n_before = len(ra)
    ra, dec, e1, e2, w = (x[keep] for x in (ra, dec, e1, e2, w))
    print(
        f"Applied v1.4.8 mask {mask_file}: kept {len(ra):,}/{n_before:,} "
        f"galaxies ({len(ra) / n_before:.3%})"
    )
else:
    print(f"Loaded {len(ra)} galaxies from {catalog_path}")

cat = treecorr.Catalog(
    ra=ra,
    dec=dec,
    g1=e1,
    g2=e2,
    w=w,
    ra_units="degrees",
    dec_units="degrees",
)

gg = treecorr.GGCorrelation(
    min_sep=min_sep,
    max_sep=max_sep,
    nbins=nbins,
    sep_units="arcmin",
)

print(f"Running treecorr: {nbins} bins, [{min_sep}, {max_sep}] arcmin...")
gg.process(cat)
gg.write(output_path)
print(f"Saved to {output_path}")
