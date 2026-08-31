"""Tests for :mod:`sp_validation.sacc_io`.

All synthetic, all fast: the OneCovariance fixtures are built in memory
shaped exactly like its real file I/O — a flat ``covariance_list`` table with
the ``(i, j)`` index rows and the Gaussian / Gauss+non-Gaussian value columns
at the indices ``cov_from_one_covariance`` expects (col 10 / col 9), and the
combined n(z) text file (column 0 = z, one column per bin, no edges). No
cluster paths; OneCovariance is not imported.

The two pieces:

- **Piece 1** (n(z) SACC -> OneCovariance input): write the combined n(z)
  file from a SACC's ``source_i`` NZ tracers, read it back, assert the z and
  n(z) columns round-trip and the config stanza names the file.
- **Piece 2** (OneCovariance output -> SACC covariance blocks): reshape the
  flat table to the dense block(s) and prove they feed
  ``sacc_io.assemble_covariance`` cleanly (reshaped block -> assemble ->
  ``s.covariance.dense`` matches the hand-built matrix).
"""

import numpy as np
import numpy.testing as npt
import pytest

from sp_validation import sacc_io as sio


# --------------------------------------------------------------------------- #
# Synthetic OneCovariance-shaped fixtures
# --------------------------------------------------------------------------- #
def _one_cov_table(cov_gauss, cov_all):
    """Flatten two n x n matrices into a OneCovariance ``covariance_list`` table.

    Reproduces the real flat output: one row per ``(i, j)`` element pair in
    row-major order ``k = i·n + j``, with the Gaussian value in column 10 and
    the Gaussian+non-Gaussian value in column 9. Columns 0-8 and the index
    columns are filled with self-documenting placeholder values (the reshape
    only reads cols 9/10, but a realistic width proves it does not spill).
    """
    n = cov_gauss.shape[0]
    rows = []
    for i in range(n):
        for j in range(n):
            row = np.arange(11.0)  # placeholder cols 0-8 (+ overwritten 9,10)
            row[9] = cov_all[i, j]
            row[10] = cov_gauss[i, j]
            rows.append(row)
    return np.array(rows)


def _spd(n, seed):
    """Symmetric positive-definite matrix of size ``n`` (a valid covariance)."""
    a = np.random.default_rng(seed).normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


def _nz(seed, n=40):
    rng = np.random.default_rng(seed)
    z = np.linspace(0.01, 2.0, n)
    return z, rng.uniform(0.1, 1.0, n)


# --------------------------------------------------------------------------- #
# Piece 2 — flat covariance_list -> dense block (reshape correctness + teeth)
# --------------------------------------------------------------------------- #
def test_covariance_blocks_reshapes_to_hand_built_matrix():
    """Pin the reshape and prove gaussian vs gauss+ng select different columns.

    WHAT IS PINNED: ``covariance_blocks`` flattens/reshapes the OneCovariance
    ``covariance_list`` table into a dense square block matching a hand-built
    covariance. It delegates the per-block reshape to
    ``statistics.cov_from_one_covariance``, so column 10 (gaussian) and column 9
    (gauss+ng) must recover the two distinct hand-built matrices.

    WHY TEETH: (a) ``gaussian=True`` vs ``False`` must return the two *different*
    matrices, proving the column flag is load-bearing; (b) perturbing a single
    entry of the flat input must change exactly that entry of the reshaped
    block, proving the reshape actually reads the table (not a constant).
    """
    cov_gauss = _spd(4, seed=1)
    cov_all = _spd(4, seed=2)
    table = _one_cov_table(cov_gauss, cov_all)

    selector = (sio.XI_PLUS, (sio.source_name(0), sio.source_name(0)))

    [(sel_g, block_g)] = sio.covariance_blocks(table, selector, gaussian=True)
    [(sel_a, block_a)] = sio.covariance_blocks(table, selector, gaussian=False)

    assert sel_g == selector and sel_a == selector
    npt.assert_allclose(block_g, cov_gauss, rtol=1e-12)
    npt.assert_allclose(block_a, cov_all, rtol=1e-12)

    # TEETH: gaussian and gauss+ng select different columns -> different blocks.
    assert not np.allclose(block_g, block_a)

    # TEETH: a perturbation of one flat-table entry moves exactly that block
    # entry (row k = i·n + j, col 10 for gaussian).
    perturbed = table.copy()
    perturbed[2 * 4 + 1, 10] += 5.0  # element (i=2, j=1)
    [(_, block_p)] = sio.covariance_blocks(perturbed, selector, gaussian=True)
    npt.assert_allclose(block_p[2, 1] - block_g[2, 1], 5.0, rtol=1e-12)
    block_p[2, 1] = block_g[2, 1]
    npt.assert_allclose(block_p, block_g, rtol=1e-12)  # nothing else moved


def test_covariance_blocks_multiblock_form():
    """Prove the tomography-ready multi-block form reshapes each sub-table.

    WHAT IS PINNED: passing ``selectors=None`` and a sequence of
    ``(selector, sub_table)`` pairs reshapes each sub-table independently and
    returns them paired with their selectors in order — the shape needed to map
    a multi-statistic OneCovariance output onto several SACC selectors.

    WHY TEETH: the two sub-tables carry different matrices; if the function
    reshaped only the first or mixed them, the second block would not match its
    own hand-built matrix.
    """
    cov_a, cov_b = _spd(3, seed=3), _spd(2, seed=4)
    table_a = _one_cov_table(cov_a, _spd(3, seed=5))
    table_b = _one_cov_table(cov_b, _spd(2, seed=6))
    sel_a = (sio.XI_PLUS, (sio.source_name(0), sio.source_name(0)))
    sel_b = (sio.XI_MINUS, (sio.source_name(0), sio.source_name(0)))

    blocks = sio.covariance_blocks(
        [(sel_a, table_a), (sel_b, table_b)], None, gaussian=True
    )

    assert [s for s, _ in blocks] == [sel_a, sel_b]
    npt.assert_allclose(blocks[0][1], cov_a, rtol=1e-12)
    npt.assert_allclose(blocks[1][1], cov_b, rtol=1e-12)


# --------------------------------------------------------------------------- #
# Piece 2 — the reshaped block feeds assemble_covariance cleanly
# --------------------------------------------------------------------------- #
def test_covariance_blocks_feed_assemble_covariance():
    """Round-trip: reshaped block -> assemble_covariance -> dense matches.

    WHAT IS PINNED: the ``(selector, dense)`` pair ``covariance_blocks``
    returns is directly consumable by ``sacc_io.assemble_covariance``: assembled
    onto a SACC whose only statistic is one ξ+ block, ``s.covariance.dense``
    must equal the hand-built OneCovariance matrix. This is the end-to-end
    contract between the two modules.

    WHY TEETH: the block must tile the data vector exactly; if the reshape
    produced the wrong size or the wrong selector, ``assemble_covariance`` would
    raise (its contiguity/tiling/size checks), so a clean assemble + matching
    dense is a real proof.
    """
    theta = np.geomspace(1.0, 100.0, 4)
    xip, xim = np.arange(4) * 1e-5, np.arange(4) * 2e-5
    s = sio.new_sacc({0: _nz(0)})
    sio.add_xi(s, (0, 0), theta, xip, xim, grid="reporting")

    # The ξ block spans ξ+ then ξ− for the pair -> 8 points, one contiguous
    # block (pair-major, matching sacc_io's canonical order).
    cov_gauss = _spd(8, seed=7)
    table = _one_cov_table(cov_gauss, _spd(8, seed=8))
    pair = (sio.source_name(0), sio.source_name(0))
    selector = np.concatenate(
        [s.indices(sio.XI_PLUS, pair), s.indices(sio.XI_MINUS, pair)]
    )

    blocks = sio.covariance_blocks(table, selector, gaussian=True)
    sio.assemble_covariance(s, blocks)

    npt.assert_allclose(s.covariance.dense, cov_gauss, rtol=1e-12)


# --------------------------------------------------------------------------- #
# Piece 1 — n(z) SACC -> OneCovariance input (round-trip + config stanza)
# --------------------------------------------------------------------------- #
def test_write_nz_roundtrips_and_names_file(tmp_path):
    """Round-trip the n(z) file and check the config stanza names it.

    WHAT IS PINNED: ``write_nz`` writes the SACC ``source_i`` NZ tracers as the
    OneCovariance combined file (column 0 = z, one column per bin, no z_low/
    z_high edges). ``read_nz`` recovers the z grid and every per-bin n(z)
    column, and the returned ``[redshift]`` stanza names the exact file written
    (directory + basename) plus ``value_loc_in_lensbin``.

    WHY TEETH: the z grid and each n(z) column must round-trip to the values the
    SACC holds (drawn from a seeded RNG); a transposed write or a dropped column
    would fail the per-bin comparison. The stanza's directory/file must match
    the path actually written.
    """
    z, nz0 = _nz(10)
    _, nz1 = _nz(11)
    s = sio.new_sacc({0: (z, nz0), 1: (z, nz1)})

    path = tmp_path / "nz_onecov.txt"
    stanza = sio.write_nz(s, path, n_bins=2)

    z_read, nz_read = sio.read_nz(path)
    npt.assert_allclose(z_read, z, rtol=1e-12)
    assert nz_read.shape == (len(z), 2)
    npt.assert_allclose(nz_read[:, 0], nz0, rtol=1e-12)
    npt.assert_allclose(nz_read[:, 1], nz1, rtol=1e-12)

    assert stanza["zlens_directory"] == str(tmp_path)
    assert stanza["zlens_file"] == "nz_onecov.txt"
    assert stanza["value_loc_in_lensbin"] == "mid"


def test_write_nz_unions_template_dir_key(tmp_path):
    """The UNIONS-template ``z_directory`` key is selectable via ``dir_key``.

    WHAT IS PINNED: the upstream OneCovariance key is ``zlens_directory``, but
    the UNIONS template (pseudo_cl.py._modify_onecov_config) writes
    ``z_directory``. ``dir_key="z_directory"`` produces that variant so the
    stanza drops straight into the UNIONS template's ``[redshift]`` section.
    """
    z, nz0 = _nz(12)
    s = sio.new_sacc({0: (z, nz0)})
    stanza = sio.write_nz(s, tmp_path / "nz.txt", n_bins=1, dir_key="z_directory")
    assert "z_directory" in stanza and "zlens_directory" not in stanza
    assert stanza["z_directory"] == str(tmp_path)
    assert stanza["zlens_file"] == "nz.txt"


def test_write_nz_fails_on_mismatched_z_grids(tmp_path):
    """Fail fast when source bins do not share one redshift grid.

    WHAT IS PINNED: the OneCovariance combined file has a single redshift
    column, so all bins must share the z grid. A bin on a different grid must
    raise ``ValueError`` at write time, not silently mis-align.
    """
    z0, nz0 = _nz(20)
    z1_shifted, nz1 = _nz(21)
    z1_shifted = z1_shifted + 0.1  # different grid
    s = sio.new_sacc({0: (z0, nz0), 1: (z1_shifted, nz1)})
    with pytest.raises(ValueError, match="differs from source bin 0"):
        sio.write_nz(s, tmp_path / "bad.txt", n_bins=2)


def test_write_nz_no_header_roundtrips(tmp_path):
    """The bare (header=False) file still round-trips numerically.

    WHAT IS PINNED: ``header=False`` writes a purely numeric file (no ``#``
    column-name line); ``read_nz`` recovers the same z grid and n(z) column, so
    the header is cosmetic and never load-bearing for the numeric round-trip.
    """
    z, nz0 = _nz(40)
    s = sio.new_sacc({0: (z, nz0)})
    path = tmp_path / "bare.txt"
    sio.write_nz(s, path, n_bins=1, header=False)
    z_read, nz_read = sio.read_nz(path)
    npt.assert_allclose(z_read, z, rtol=1e-12)
    npt.assert_allclose(nz_read[:, 0], nz0, rtol=1e-12)


def test_nz_config_stanza_rejects_bad_value_loc():
    """``value_loc_in_lensbin`` outside {mid,left,right} fails fast.

    WHAT IS PINNED: OneCovariance only accepts ``mid``/``left``/``right`` for
    the histogram-bin value location; an invalid value is a config bug and must
    raise ``ValueError`` rather than write a stanza OneCovariance will reject.
    """
    with pytest.raises(ValueError, match="value_loc_in_lensbin"):
        sio.nz_config_stanza("/dir", "nz.txt", value_loc="center")


def test_write_nz_fails_on_missing_bin(tmp_path):
    """Fail fast when a requested source bin is absent from the SACC.

    WHAT IS PINNED: requesting more bins than the SACC carries is a real config
    bug; ``write_nz`` raises ``ValueError`` naming the missing tracer rather
    than writing a short file.
    """
    z, nz0 = _nz(30)
    s = sio.new_sacc({0: (z, nz0)})
    with pytest.raises(ValueError, match="source_1"):
        sio.write_nz(s, tmp_path / "short.txt", n_bins=2)
