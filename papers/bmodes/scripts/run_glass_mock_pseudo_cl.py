"""Run NaMaster pseudo-Cℓ on a GLASS mock galaxy catalog.

Computes mode-coupling-corrected pseudo-Cℓ bandpowers using the catalog-based
NaMaster estimator. Matches the real data pipeline binning exactly.

Mock catalogs have columns: RA, Dec, e1, e2, w.
Polarization convention: pol_factor=True → field=[e1, -e2] (matches pipeline).
"""

import numpy as np
import pymaster as nmt
from astropy.io import fits

catalog_path = snakemake.input.catalog  # noqa: F821
output_path = snakemake.output.pseudo_cl  # noqa: F821

nside = snakemake.params.nside  # noqa: F821
nbins = snakemake.params.nbins  # noqa: F821
power = snakemake.params.power  # noqa: F821
lmin = snakemake.params.lmin  # noqa: F821
lmax = snakemake.params.lmax  # noqa: F821

# Load mock catalog
with fits.open(catalog_path) as hdul:
    data = hdul["SOURCE_CATALOGUE"].data
    ra = data["RA"]
    dec = data["Dec"]
    e1 = data["e1"]
    e2 = data["e2"]
    w = data["w"]

print(f"Loaded {len(ra)} galaxies from {catalog_path}")

# Wrap RA to [0, 360) — NmtFieldCatalog requires longitude in this range
ra = ra % 360

# Powspace binning — matches CosmologyValidation.get_namaster_bin()
b_lmax = lmax - 1
ells = np.arange(lmin, lmax + 1)
start = np.power(lmin, power)
end = np.power(lmax, power)
bins_ell = np.power(np.linspace(start, end, nbins + 1), 1 / power)
bpws = np.digitize(ells.astype(float), bins_ell) - 1
bpws[0] = 0
bpws[-1] = nbins - 1
b = nmt.NmtBin(ells=ells, bpws=bpws, lmax=b_lmax)

ell_eff = b.get_effective_ells()
print(f"Binning: {nbins} powspace bins, ell=[{ell_eff[0]:.1f}, {ell_eff[-1]:.1f}]")

# Create NaMaster spin-2 field from catalog
# pol_factor=True: e2 sign flip to match IAU polarization convention
print("Creating NaMaster field...")
f_all = nmt.NmtFieldCatalog(
    positions=[ra, dec],
    weights=w,
    field=[e1, -e2],
    lmax=b_lmax,
    lmax_mask=b_lmax,
    spin=2,
    lonlat=True,
)

# Compute workspace (mode-coupling matrix)
print("Computing MCM workspace...")
wsp = nmt.NmtWorkspace.from_fields(f_all, f_all, b)

# Compute and decouple pseudo-Cℓ
print("Computing pseudo-Cℓ...")
cl_coupled = nmt.compute_coupled_cell(f_all, f_all)
cl_all = wsp.decouple_cell(cl_coupled)

# cl_all shape: (4, nbins) — EE, EB, BE, BB
# Save as FITS matching the pipeline output format (columns: ELL, EE, EB, BB)
cols = [
    fits.Column(name="ELL", format="D", array=ell_eff),
    fits.Column(name="EE", format="D", array=cl_all[0]),
    fits.Column(name="EB", format="D", array=cl_all[1]),
    fits.Column(name="BB", format="D", array=cl_all[3]),
]
hdu = fits.BinTableHDU.from_columns(cols, name="PSEUDO_CELL")
hdu.writeto(output_path, overwrite=True)

print(f"Saved pseudo-Cℓ to {output_path}")
print(f"  EE range: [{cl_all[0].min():.2e}, {cl_all[0].max():.2e}]")
print(f"  BB range: [{cl_all[3].min():.2e}, {cl_all[3].max():.2e}]")
