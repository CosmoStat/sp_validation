"""ONE_COVARIANCE_IO.

:Name: one_covariance_io.py

:Description: File-format glue between the SACC data-product layout
              (:mod:`sp_validation.sacc_io`) and OneCovariance
              (https://github.com/rreischke/OneCovariance). Two directions:

              - **n(z) SACC -> OneCovariance input** (:func:`write_nz`): the
                ``source_i`` NZ tracers of an analysis SACC are written as the
                combined whitespace-delimited redshift file OneCovariance reads
                (column 0 = z grid, then one ``n(z)`` column per tomographic
                bin, no bin edges), and a matching ``[redshift]`` config stanza
                is returned via :func:`nz_config_stanza`.

              - **OneCovariance output -> SACC covariance blocks**
                (:func:`covariance_blocks`): the flat ``covariance_list_*.dat``
                table OneCovariance emits (one row per element pair) is reshaped
                into dense square block(s) — reusing
                :func:`sp_validation.statistics.cov_from_one_covariance` for the
                per-block reshape — and paired with SACC selectors so a caller
                can feed them straight to
                :func:`sp_validation.sacc_io.assemble_covariance`.

              OneCovariance itself is *not* a dependency: this module only
              touches its file formats, verified against the upstream
              ``config.ini`` (``rreischke/OneCovariance`` @ main).

              n(z) file format (upstream ``config.ini`` comment, verbatim):

                  ``redshift   n_1(z) ... n_{N_source}(z)``

              i.e. a plain whitespace-delimited text file, column 0 the shared
              redshift grid and one column per tomographic bin — no ``z_low``/
              ``z_high`` edges (this is the OneCovariance convention, distinct
              from the CosmoSIS NZDATA table which *does* carry edges). All
              source bins must therefore share one z grid.

              ``[redshift]`` config keys (upstream canonical names): a single
              combined file goes in ``zlens_directory`` + ``zlens_file``;
              ``value_loc_in_lensbin`` (``mid``/``left``/``right``) says where
              in each histogram bin the tabulated ``n(z)`` value sits — ``mid``
              for the bin-centred grids the SACC stores. NOTE: the UNIONS
              OneCovariance template driven by
              ``cosmo_val/pseudo_cl.py._modify_onecov_config`` writes the older
              key names ``z_directory``/``zlens_file`` instead; pass
              ``dir_key="z_directory"`` to match that template.
"""

import os

import numpy as np

from . import sacc_io
from .statistics import cov_from_one_covariance


def nz_table(s, n_bins):
    """Stack the SACC ``source_i`` NZ tracers into a OneCovariance n(z) table.

    Parameters
    ----------
    s : sacc.Sacc
        SACC holding ``source_0 … source_{n_bins-1}`` NZ tracers.
    n_bins : int
        Number of tomographic source bins to write.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_z, n_bins + 1)``: column 0 the shared redshift
        grid, columns ``1 … n_bins`` the per-bin ``n(z)``. This is the
        OneCovariance combined-file layout (``redshift  n_1(z) … n_N(z)``).

    Raises
    ------
    ValueError
        If any source bin is missing, or if the bins do not share one z grid
        (OneCovariance's combined file has a single redshift column, so the
        grids must agree bin-for-bin).
    """
    z0, nz0 = sacc_io.get_nz(s, 0)
    z0 = np.asarray(z0, dtype=float)
    columns = [z0]
    for i in range(n_bins):
        if sacc_io.source_name(i) not in s.tracers:
            raise ValueError(
                f"SACC has no NZ tracer {sacc_io.source_name(i)!r}; cannot write "
                f"a {n_bins}-bin OneCovariance n(z) file"
            )
        z_i, nz_i = sacc_io.get_nz(s, i)
        if not np.array_equal(np.asarray(z_i, dtype=float), z0):
            raise ValueError(
                f"source bin {i} n(z) grid differs from source bin 0; the "
                "OneCovariance combined n(z) file has one shared redshift column"
            )
        columns.append(np.asarray(nz_i, dtype=float))
    return np.column_stack(columns)


def write_nz(s, path, n_bins, *, dir_key="zlens_directory", header=True):
    """Write the OneCovariance combined n(z) input file from a SACC.

    OneCovariance reads the source redshift distribution as a plain
    whitespace-delimited text file whose column 0 is the shared redshift grid
    and whose remaining columns are the per-bin ``n(z)`` (``redshift  n_1(z)
    … n_N(z)``) — no ``z_low``/``z_high`` edges. This writes that file from the
    SACC ``source_i`` NZ tracers and returns the ``[redshift]`` config stanza
    that points OneCovariance at it.

    Parameters
    ----------
    s : sacc.Sacc
        Analysis SACC with the ``source_i`` NZ tracers.
    path : str or pathlib.Path
        Output text-file path (overwritten). Its directory + basename become
        the ``[redshift]`` directory/file config values.
    n_bins : int
        Number of tomographic source bins to write.
    dir_key : str, optional
        Config key for the redshift directory. Default ``"zlens_directory"``
        (upstream canonical). Pass ``"z_directory"`` for the UNIONS template
        driven by ``pseudo_cl.py._modify_onecov_config``.
    header : bool, optional
        If ``True`` (default) prepend a ``# redshift  n_1(z) …`` comment header
        naming the columns; OneCovariance's ``genfromtxt``-style reader ignores
        it. Set ``False`` for a bare numeric file.

    Returns
    -------
    dict
        The ``[redshift]`` config stanza (see :func:`nz_config_stanza`), naming
        the file just written.
    """
    table = nz_table(s, n_bins)
    head = ""
    if header:
        cols = " ".join(f"n_{i + 1}(z)" for i in range(n_bins))
        head = f"redshift {cols}"
    np.savetxt(str(path), table, header=head)
    return nz_config_stanza(
        os.path.dirname(os.path.abspath(str(path))),
        os.path.basename(str(path)),
        dir_key=dir_key,
    )


def nz_config_stanza(
    directory, filename, *, dir_key="zlens_directory", value_loc="mid"
):
    """Build the OneCovariance ``[redshift]`` config stanza for an n(z) file.

    Parameters
    ----------
    directory : str
        Directory holding the n(z) file (OneCovariance ``*_directory`` value).
    filename : str
        n(z) file basename (OneCovariance ``zlens_file`` value).
    dir_key : str, optional
        Directory config key — ``"zlens_directory"`` (upstream) or
        ``"z_directory"`` (UNIONS template). Default ``"zlens_directory"``.
    value_loc : str, optional
        ``value_loc_in_lensbin`` — where in each histogram bin the tabulated
        ``n(z)`` value sits (``mid``/``left``/``right``). Default ``"mid"``,
        matching the bin-centred grids the SACC stores.

    Returns
    -------
    dict
        The ``[redshift]`` key/value pairs: ``{dir_key: directory, "zlens_file":
        filename, "value_loc_in_lensbin": value_loc}``. Assign these under
        ``config["redshift"]`` of a OneCovariance ``configparser`` config.
    """
    if value_loc not in ("mid", "left", "right"):
        raise ValueError(
            f"value_loc_in_lensbin must be 'mid', 'left' or 'right'; got {value_loc!r}"
        )
    return {
        dir_key: directory,
        "zlens_file": filename,
        "value_loc_in_lensbin": value_loc,
    }


def read_nz(path):
    """Read a OneCovariance combined n(z) file back to ``(z, nz_columns)``.

    Inverse of :func:`write_nz` (the numeric round-trip; the config stanza is
    not stored in the file). Comment/header lines are skipped.

    Parameters
    ----------
    path : str or pathlib.Path
        n(z) text file (column 0 = z, columns 1… = per-bin n(z)).

    Returns
    -------
    tuple
        ``(z, nz)`` where ``z`` is the shared redshift grid (shape ``(n_z,)``)
        and ``nz`` is the per-bin distributions (shape ``(n_z, n_bins)``).
    """
    table = np.atleast_2d(np.genfromtxt(str(path)))
    return table[:, 0], table[:, 1:]


def covariance_blocks(cov_list, selectors, *, gaussian=True):
    """Reshape a OneCovariance ``covariance_list`` table into SACC cov blocks.

    OneCovariance emits a flat ``covariance_list_*.dat`` table with one row per
    ``(i, j)`` element pair (row-major, ``k = i·n + j``); the covariance value
    lives in column 10 (Gaussian) or column 9 (Gaussian+non-Gaussian). This
    reshapes the flat table into dense square block(s) — reusing
    :func:`sp_validation.statistics.cov_from_one_covariance` for the per-block
    reshape — and pairs each with its SACC selector, ready for
    :func:`sp_validation.sacc_io.assemble_covariance`.

    Single-statistic case: pass the whole table and one selector; you get one
    ``(selector, dense)`` block. Multi-statistic case (tomography-ready): pass a
    sequence of ``(selector, sub_table)`` pairs — each ``sub_table`` a
    contiguous slice of the flat output for one statistic / bin-pair — and each
    is reshaped and re-paired with its selector in order. The API is thus shaped
    to extend to multi-probe blocking without over-fitting the single-bin case.

    Parameters
    ----------
    cov_list : numpy.ndarray or sequence
        Either the flat OneCovariance table (2-D array, one row per pair) for a
        single block, or — for the multi-block form — a sequence of
        ``(selector, sub_table)`` pairs. In the multi-block form ``selectors``
        must be ``None`` (the selectors travel with the sub-tables).
    selectors : selector or None
        For the single-block form, the SACC selector for the whole table (a
        ``(data_type, tracers[, tags])`` tuple or an index array, as
        :func:`sacc_io.assemble_covariance` accepts). Must be ``None`` for the
        multi-block form.
    gaussian : bool, optional
        Select the Gaussian-only column (``True``, default) or the
        Gaussian+non-Gaussian column (``False``); passed straight through to
        ``cov_from_one_covariance``.

    Returns
    -------
    list
        Ordered ``(selector, dense_cov)`` pairs, directly consumable by
        ``sacc_io.assemble_covariance(s, blocks)``.
    """
    if selectors is None:
        # Multi-block form: cov_list is a sequence of (selector, sub_table).
        return [
            (selector, cov_from_one_covariance(np.asarray(sub), gaussian=gaussian))
            for selector, sub in cov_list
        ]
    # Single-block form: one flat table, one selector.
    return [
        (selectors, cov_from_one_covariance(np.asarray(cov_list), gaussian=gaussian))
    ]
