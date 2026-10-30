"""Independent checks of tomographic covariance ordering."""

import runpy
from pathlib import Path

import numpy as np

from sp_validation import sacc_io as sio
from sp_validation.blinding import Blind
from sp_validation.cosmo_val.sacc_writers import xi_pairs_to_sacc


def test_onecov_assembly_matches_written_point_labels(tmp_path):
    pairs = [(i, j) for i in range(3) for j in range(i, 3)]
    theta = [2.0, 7.0, 20.0]
    nz = {i: (np.array([0.1, 0.5, 1.0]), np.array([0.1, 1.0, 0.1])) for i in range(3)}
    measurements = [
        (pair, dict(theta=theta, xip=np.zeros(3), xim=np.zeros(3)))
        for pair in reversed(pairs)
    ]
    part = xi_pairs_to_sacc(nz, {}, measurements, grid="reporting")
    part_path = tmp_path / "xi.sacc"
    sio.save(part, part_path, blind=Blind("none"))
    labels = [
        (dtype, pair, t)
        for dtype in (sio.XI_PLUS, sio.XI_MINUS)
        for pair in pairs
        for t in theta
    ]
    rng = np.random.default_rng(376)
    a = rng.normal(size=(len(labels), len(labels)))
    source = a + a.T
    cov_path = tmp_path / "onecov.dat"
    np.savetxt(cov_path, source, header="OneCovariance covariance matrix")
    script = Path(__file__).resolve().parents[3] / "workflow/scripts/assemble_sacc.py"
    assemble = runpy.run_path(str(script))["assemble_sacc"]
    output = tmp_path / "assembled.sacc"
    assemble("v", {"xi_reporting": part_path}, output, blind="none", xi_cov=cov_path)
    written = sio.load(output, blind="none")
    perm = [
        labels.index(
            (
                dp.data_type,
                tuple(int(t.removeprefix("source_")) for t in dp.tracers),
                dp.tags["theta"],
            )
        )
        for dp in written.data
    ]
    np.testing.assert_array_equal(written.covariance.dense, source[np.ix_(perm, perm)])
