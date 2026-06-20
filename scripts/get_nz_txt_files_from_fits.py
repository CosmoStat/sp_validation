# %%
"""
Take as input a FITS file containing the blinded n(z).

Produces a text file for each blind redshift distribution in the format to feed the cosmological inference pipeline.
"""

import sys

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np


nz_hdu = sys.argv[1]
root = sys.argv[2]

hdu = fits.open(nz_hdu)

zmax = 5.0

som_weights = hdu[1].data["SOM_WEIGHT"]

plt.figure()

for blind in ["A", "B", "C"]:
    z = hdu[1].data[f"Z_{blind}"]

    (n, bins, _) = plt.hist(
        z,
        bins=200,
        range=(0, zmax),
        density=True,
        weights=som_weights,
        histtype="step",
        label=f"Blind {blind}",
    )

    print("zmin = ", min(z))
    print("zmax = ", max(z))

    # Take the middle of the bin as the redshift value for the text file
    z_mid = (bins[:-1] + bins[1:]) / 2
    np.savetxt(
        f"/n17data/sguerrini/UNIONS/WL/nz/{root}/nz_SP_{root}_{blind}.txt",
        np.column_stack((z_mid, n)),
    )
plt.legend()
plt.xlabel("Redshift z")
plt.ylabel("n(z)")
plt.savefig(f"/n17data/sguerrini/UNIONS/WL/nz/{root}/nz_SP_{root}.png")
