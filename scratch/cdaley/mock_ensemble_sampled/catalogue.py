"""Write a sampled GLASS catalogue under the data entry's column names.

The generator's e1/e2 are already shear plus intrinsic shape (catalogue e2 =
-HEALPix gamma2, the convention the data and painted mocks share). Galaxies have
unit weight; the PSF columns are zero because no leakage is simulated.
"""
import fitsio
import numpy as np

src = fitsio.read(snakemake.input[0], ext="SOURCE_CATALOGUE",
                  columns=["RA", "Dec", "e1", "e2", "TOM_BIN_ID"])
n = len(src)
out = np.empty(n, dtype=[("RA", "f8"), ("Dec", "f8"), ("e1", "f8"), ("e2", "f8"),
                         ("w_des", "f8"), ("e1_PSF", "f4"), ("e2_PSF", "f4"),
                         ("TOM_BIN_ID", "i2")])
for name in ("RA", "Dec", "e1", "e2", "TOM_BIN_ID"):
    out[name] = src[name]
out["w_des"] = 1.0
out["e1_PSF"] = 0.0
out["e2_PSF"] = 0.0
fitsio.write(snakemake.output[0], out, clobber=True)
print(f"{n} galaxies, per bin:", np.bincount(out["TOM_BIN_ID"]).tolist())
