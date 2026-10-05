"""Read pseudo-Cℓ parts for the B-modes paper workflow."""

from sp_validation import sacc_io


def load_pseudo_cl_data(path):
    """Return a pseudo-Cℓ part's spectra as ``{"ELL", "EE", "EB", "BB"}`` arrays.

    The bandpower covariance is a separate FITS product (``pseudo_cl_cov``).
    Loading goes through ``sacc_io.load``, so an unblinded real-data part is
    refused.
    """
    ell, ee, bb, eb, _window = sacc_io.get_pseudo_cl(sacc_io.load(str(path)), (0, 0))
    missing = [name for name, values in (("BB", bb), ("EB", eb)) if values is None]
    if missing:
        raise ValueError(
            f"{path} lacks required pseudo-Cℓ spectra: {', '.join(missing)}"
        )
    return {"ELL": ell, "EE": ee, "EB": eb, "BB": bb}
