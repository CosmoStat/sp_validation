"""Independent checks of NaMaster covariance ordering."""

import numpy as np
from astropy.io import fits

from sp_validation import sacc_io as sio
from sp_validation.cosmo_val.sacc_writers import namaster_cl_covariance


def test_namaster_covariance_matches_written_point_labels(tmp_path):
    pairs = [(i, j) for i in range(3) for j in range(i, 3)]
    ells = [20.0, 50.0, 90.0]
    types = {sio.CL_EE: "EE", sio.CL_BB: "BB", sio.CL_EB: "EB"}
    nz = {i: (np.array([0.1, 0.5, 1.0]), np.array([0.1, 1.0, 0.1])) for i in range(3)}
    part = sio.new_sacc(nz, {})
    # A noncanonical insertion order makes the lookup depend on point labels.
    for i, j in reversed(pairs):
        sio.add_pseudo_cl(
            part,
            (i, j),
            ells,
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            window_ells=ells,
            window_weights=np.eye(3),
        )
    n = len(pairs) * len(ells)
    rng = np.random.default_rng(4)
    blocks = {}
    hdus = [fits.PrimaryHDU()]
    for pol in types.values():
        a = rng.normal(size=(n, n))
        blocks[pol] = a + a.T
        hdus.append(fits.ImageHDU(blocks[pol], name=f"COVAR_{pol}_{pol}"))
    path = tmp_path / "cov.fits"
    fits.HDUList(hdus).writeto(path)

    labels = [
        (
            types[dp.data_type],
            tuple(int(t.split("_")[1]) for t in dp.tracers),
            dp.tags["ell"],
        )
        for dp in part.data
    ]
    source = [(p, pair, ell) for p in types.values() for pair in pairs for ell in ells]
    matrix = np.zeros((len(source), len(source)))
    for k, pol in enumerate(types.values()):
        matrix[k * n : (k + 1) * n, k * n : (k + 1) * n] = blocks[pol]
    perm = [source.index(label) for label in labels]
    np.testing.assert_array_equal(
        namaster_cl_covariance(part, path), matrix[np.ix_(perm, perm)]
    )
