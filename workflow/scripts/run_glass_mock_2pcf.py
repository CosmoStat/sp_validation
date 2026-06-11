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

# Load mock catalog
with fits.open(catalog_path) as hdul:
    data = hdul["SOURCE_CATALOGUE"].data
    ra = data["RA"]
    dec = data["Dec"]
    e1 = data["e1"]
    e2 = data["e2"]
    w = data["w"]

print(f"Loaded {len(ra)} galaxies from {catalog_path}")

cat = treecorr.Catalog(
    ra=ra, dec=dec, g1=e1, g2=e2, w=w,
    ra_units="degrees", dec_units="degrees",
)

gg = treecorr.GGCorrelation(
    min_sep=min_sep, max_sep=max_sep, nbins=nbins,
    sep_units="arcmin",
)

print(f"Running treecorr: {nbins} bins, [{min_sep}, {max_sep}] arcmin...")
gg.process(cat)
gg.write(output_path)
print(f"Saved to {output_path}")
