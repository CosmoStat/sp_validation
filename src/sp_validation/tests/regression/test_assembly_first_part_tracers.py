"""Analysis SACC assembly must reject inconsistent source n(z) and provenance."""

import numpy as np
import pytest

from sp_validation import sacc_io as sio

writers = pytest.importorskip(
    "sp_validation.cosmo_val.sacc_writers",
    reason="The tomography branch lacks the analysis SACC writers module",
)

Z = np.array([0.1, 0.5, 0.9])
NZ_A = np.array([1.0, 2.0, 1.0])
NZ_B = np.array([0.0, 0.0, 10.0])


def _xi_part(nz, metadata):
    theta = np.array([1.0, 10.0])
    return writers.xi_to_sacc(
        {0: (Z, nz)},
        metadata,
        theta,
        np.array([1e-4, 1e-5]),
        np.array([2e-5, 1e-6]),
        grid="reporting",
        variances=np.ones(4),
    )


def _cosebis_part(nz, metadata, en=7.0):
    result = {"En": np.array([en]), "Bn": np.array([0.0]), "cov": np.eye(2)}
    return writers.cosebis_to_sacc({0: (Z, nz)}, metadata, result, (1.0, 250.0))


def _describe(s):
    return (
        f"assembled silently: source_0 nz={sio.get_nz(s, 0)[1].tolist()}, "
        f"catalogue_version={s.metadata.get('catalogue_version')!r}, "
        f"En={sio.get_cosebis(s, (0, 0), (1.0, 250.0))[1].tolist()}"
    )


def test_matching_parts_preserve_all_points_and_source_nz():
    """Compatible parts must retain all four ξ± and two COSEBIs points.

    This passes on develop: identical n(z) and metadata are valid inputs,
    so rejecting inconsistent parts must not break this positive control.
    """
    meta = {"catalogue_version": "v"}
    s = writers.assemble_analysis_sacc(
        [_xi_part(NZ_A, meta), _cosebis_part(NZ_A, meta)]
    )
    assert len(s.mean) == 4 + 2
    np.testing.assert_array_equal(sio.get_nz(s, 0)[1], NZ_A)


@pytest.mark.xfail(
    strict=True,
    reason="#381: assembly silently keeps the first part's n(z)",
)
def test_assembly_rejects_parts_with_different_source_nz():
    """All statistics sharing ``source_0`` must use the same source n(z).

    A ξ± part built with n(z)=[1,2,1] and a COSEBIs part built with
    n(z)=[0,0,10] describe different source samples; combining them must
    raise ValueError, as ``sacc_io.merge`` does for the same situation.
    Otherwise the COSEBIs Eₙ is silently relabelled with the first part's
    n(z), and part order picks which n(z) survives.
    """
    meta = {"catalogue_version": "v"}
    parts = [_xi_part(NZ_A, meta), _cosebis_part(NZ_B, meta)]
    with pytest.raises(ValueError):
        s = writers.assemble_analysis_sacc(parts)
        pytest.fail(_describe(s))


@pytest.mark.xfail(
    strict=True,
    reason="#381: assembly silently keeps conflicting metadata",
)
def test_assembly_rejects_parts_with_conflicting_metadata():
    """Every part of an analysis SACC must describe the same catalogue version.

    Parts stamped ``catalogue_version`` 'A' and 'B' cannot be combined
    under one label. Assembly must raise ValueError rather than keep
    ``parts[0]``'s label over part B's data.
    """
    parts = [
        _xi_part(NZ_A, {"catalogue_version": "A"}),
        _cosebis_part(NZ_A, {"catalogue_version": "B"}, en=99.0),
    ]
    with pytest.raises(ValueError):
        s = writers.assemble_analysis_sacc(parts)
        pytest.fail(_describe(s))
