"""Numerical tools for joint errors-in-variables PSF-leakage fits.

This module has no catalogue I/O. It builds the production rho-to-tau design,
profiles noisy templates from a joint delete-one covariance, and provides
TreeCorr jackknife and patch-grouping helpers.
"""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import chi2


def rho_tau_design(rho):
    """Return the linear rho-to-tau design for rho0..rho5.

    ``rho`` has shape ``(6, nbin)``. The output rows are ordered as tau0,
    tau2, tau5, each with ``nbin`` angular bins, and columns are alpha, beta,
    eta. This is the field definition used by ``shear_psf_leakage``.
    """
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 2 or rho.shape[0] != 6:
        raise ValueError("rho must have shape (6, nbin)")
    return np.concatenate(
        [
            np.stack((rho[0], rho[2], rho[5]), axis=-1),
            np.stack((rho[2], rho[1], rho[4]), axis=-1),
            np.stack((rho[5], rho[4], rho[3]), axis=-1),
        ],
        axis=0,
    )


def ward_group_map(centres, ngroups):
    """Group fine patch centres into ``ngroups`` Ward-linkage unions.

    The centre coordinates are used as supplied (TreeCorr spherical patch
    centres are three-dimensional unit vectors). The result maps every fine
    patch index to a contiguous coarse group index.
    """
    centres = np.asarray(centres, dtype=float)
    if centres.ndim != 2:
        raise ValueError("centres must be a two-dimensional array of patch centres")
    if centres.shape[0] < 3:
        raise ValueError(
            f"Ward grouping needs at least 3 fine patch centres, got {centres.shape[0]}"
        )
    if not 1 < ngroups < len(centres):
        raise ValueError("ngroups must be between 2 and the number of fine patches")
    groups = (
        fcluster(linkage(centres, method="ward"), ngroups, criterion="maxclust") - 1
    )
    if len(np.unique(groups)) != ngroups:
        raise ValueError(
            f"Ward grouping produced {len(np.unique(groups))} groups, expected {ngroups}"
        )
    return groups.astype(int, copy=False)


def treecorr_jackknife_rows(correlations, groups=None, check_rows=True):
    """Build exact TreeCorr '+' match-jackknife rows in O(number of pairs).

    When ``groups`` maps fine patch IDs to coarse unions, all fine pairs within
    one coarse union are treated as internal pairs and deleted together.
    Returns ``(rows, max_checked_error)``: the rows of all correlations side by
    side, and the largest difference from TreeCorr's own delete-one result over
    the checked fine rows.
    """
    blocks = []
    max_error = 0.0
    group_map = None if groups is None else np.asarray(groups, dtype=int)

    for corr in correlations:
        fine_n = corr.npatch1
        if fine_n != corr.npatch2 or fine_n < 2:
            raise ValueError(
                "each correlation must have matching patches and npatch > 1"
            )
        if group_map is None:
            n_patch = fine_n
        else:
            if group_map.shape != (fine_n,):
                raise ValueError("groups must contain one index per fine patch")
            unique = np.unique(group_map)
            if not np.array_equal(unique, np.arange(len(unique))):
                raise ValueError("group indices must be contiguous and start at zero")
            n_patch = len(unique)
            if n_patch < 2:
                raise ValueError("grouped jackknife requires at least two groups")

        alpha = n_patch / (2 + np.sqrt(2) * (n_patch - 1))
        keys = np.asarray(list(corr.results), dtype=int)
        if keys.ndim != 2 or keys.shape[1] != 2:
            raise ValueError("TreeCorr correlation has no patch-pair results")
        i, j = keys.T
        if group_map is not None:
            i, j = group_map[i], group_map[j]
        pair_values = np.asarray([corr.results[tuple(key)]._xi[0] for key in keys])
        pair_weights = np.asarray([corr.results[tuple(key)].weight for key in keys])
        lost = np.zeros((n_patch, len(corr.xip)), dtype=float)
        lost_weight = np.zeros_like(lost)
        cross = i != j
        diagonal = ~cross
        for values, output in (
            (pair_values, lost),
            (pair_weights, lost_weight),
        ):
            np.add.at(output, i[cross], alpha * values[cross])
            np.add.at(output, j[cross], alpha * values[cross])
            np.add.at(output, i[diagonal], values[diagonal])

        denominator = pair_weights.sum(axis=0)[None, :] - lost_weight
        rows = (pair_values.sum(axis=0)[None, :] - lost) / denominator
        point = pair_values.sum(axis=0) / pair_weights.sum(axis=0)
        if not np.allclose(point, corr.xip, rtol=1e-9, atol=1e-22):
            raise AssertionError("pair totals do not reproduce TreeCorr's full result")

        if check_rows and group_map is None:
            check = corr.copy()
            pair_iter = corr._jackknife_pairs("match")
            for patch in sorted({0, n_patch // 2, n_patch - 1}):
                check._calculate_xi_from_pairs(
                    check._keep_ok(pair_iter[patch]), corr_only=True
                )
                error = float(np.max(np.abs(check.xip - rows[patch])))
                max_error = max(max_error, error)
                if not np.allclose(check.xip, rows[patch], rtol=1e-8, atol=1e-22):
                    raise AssertionError(
                        f"fast jackknife row {patch} differs from TreeCorr by {error}"
                    )

        blocks.append(rows)

    if not blocks:
        raise ValueError("at least one TreeCorr correlation is required")
    return np.concatenate(blocks, axis=1), max_error


def residual_covariance_model(x_rows, y_rows, cross_term=True):
    """The covariance V(p) of ``r = y - x p`` for a noisy design ``x``.

    ``x_rows`` (shape ``(npatch, ndata, nparam)``) and ``y_rows`` (shape
    ``(npatch, ndata)``) are delete-one jackknife rows of ``x`` and ``y``.
    With the jackknife covariances C_xx, C_xy, C_yy built from them,
    V(p) = C_yy - (B C_xy + C_yx B^T) + B C_xx B^T, B the linear map
    x -> x p. ``cross_term=False`` zeroes C_xy. Returns two functions of
    ``p``: V(p), shape ``(ndata, ndata)``, and dV/dp, shape
    ``(nparam, ndata, ndata)``.
    """
    n_patch = len(x_rows)
    jk_factor = (n_patch - 1) / n_patch
    dx = x_rows - x_rows.mean(axis=0)
    dy = y_rows - y_rows.mean(axis=0)
    cov_yy = jk_factor * dy.T @ dy
    cov_xy = jk_factor * np.einsum("nai,nb->iab", dx, dy)
    if not cross_term:
        cov_xy.fill(0.0)
    linear = -cov_xy - cov_xy.transpose(0, 2, 1)
    cov_xx = jk_factor * np.einsum("nai,nbj->ijab", dx, dx)

    def covariance(parameters):
        return (
            cov_yy
            + np.einsum("i,iab->ab", parameters, linear)
            + np.einsum("i,j,ijab->ab", parameters, parameters, cov_xx)
        )

    def derivative(parameters):
        return (
            linear
            + np.einsum("j,ijab->iab", parameters, cov_xx)
            + np.einsum("j,jiab->iab", parameters, cov_xx)
        )

    return covariance, derivative


def _profile_fit(x, y, x_rows, y_rows, scales, cross_term, start):
    """Fit ``y = x p`` with noisy ``x`` by profiling its latent true values.

    Minimises r^T V(p)^-1 r (no log-determinant term), V from
    `residual_covariance_model`, by BFGS from several starts. The best trial
    is the lowest objective among those BFGS reports converged, or among all
    trials if none did (``bfgs_success`` False).
    """
    n_patch, n_data, n_param = x_rows.shape
    if y_rows.shape != (n_patch, n_data):
        raise ValueError("y_rows must have shape (npatch, ndata)")
    if x.shape != (n_data, n_param):
        raise ValueError("x must have shape (ndata, nparam)")
    if scales.shape != (n_param,) or np.any(scales <= 0):
        raise ValueError("scales must contain one positive value per parameter")
    hartlap = (n_patch - n_data - 2) / (n_patch - 1)
    if hartlap <= 0:
        raise ValueError("Hartlap factor is non-positive for this patch/data count")

    residual_covariance, covariance_derivative = residual_covariance_model(
        x_rows, y_rows, cross_term
    )

    def objective(scaled_parameters):
        parameters = scaled_parameters * scales
        residual = y - x @ parameters
        factor = cho_factor(
            residual_covariance(parameters), lower=True, check_finite=False
        )
        weighted_residual = cho_solve(factor, residual, check_finite=False)
        value = float(residual @ weighted_residual)
        gradient = (
            -2 * x.T @ weighted_residual
            - np.einsum(
                "a,iab,b->i",
                weighted_residual,
                covariance_derivative(parameters),
                weighted_residual,
            )
        ) * scales
        return value, gradient

    # Generalised least squares with the template noise ignored.
    factor = cho_factor(residual_covariance(np.zeros(n_param)), lower=True)
    fixed_start = np.linalg.lstsq(
        x.T @ cho_solve(factor, x), x.T @ cho_solve(factor, y), rcond=None
    )[0]
    starts = [fixed_start, np.zeros(n_param)]
    if n_param >= 3:
        starts.extend(
            [
                np.pad([0.02, 1.7, 3.0], (0, n_param - 3)),
                np.pad([-0.01, 1.0, -2.0], (0, n_param - 3)),
            ]
        )
    if start is not None:
        start = np.asarray(start, dtype=float)
        if start.shape != (n_param,):
            raise ValueError("start must contain one value per fit parameter")
        starts.append(start)
    rng = np.random.default_rng(1028)
    starts.extend(fixed_start + rng.normal(size=n_param) * scales for _ in range(3))

    fitted = [
        minimize(
            objective,
            initial / scales,
            jac=True,
            method="BFGS",
            options={"gtol": 1e-7, "maxiter": 500},
        )
        for initial in starts
    ]
    trials = [
        {
            "parameters": (fit.x * scales).tolist(),
            "chi2_raw": float(fit.fun),
            "success": bool(fit.success),
            "gradient_norm": float(np.linalg.norm(fit.jac)),
        }
        for fit in fitted
    ]
    best = min([fit for fit in fitted if fit.success] or fitted, key=lambda f: f.fun)
    parameters = best.x * scales
    covariance = residual_covariance(parameters)
    residual = y - x @ parameters
    chi2_raw = float(residual @ np.linalg.solve(covariance, residual))
    chi2_value = hartlap * chi2_raw

    step = 1e-4
    hessian = np.column_stack(
        [
            (
                objective(best.x + np.eye(n_param)[index] * step)[1]
                - objective(best.x - np.eye(n_param)[index] * step)[1]
            )
            / (2 * step)
            for index in range(n_param)
        ]
    )
    hessian = (hessian + hessian.T) / 2
    try:
        np.linalg.cholesky(hessian)
        positive = True
    except np.linalg.LinAlgError:
        positive = False
    if positive:
        parameter_covariance = (
            2 * np.linalg.inv(hessian) * scales[:, None] * scales[None, :] / hartlap
        )
        sigma = np.sqrt(np.diag(parameter_covariance)).tolist()
        parameter_covariance = parameter_covariance.tolist()
    else:
        parameter_covariance = None
        sigma = [None] * n_param
    dof = n_data - n_param
    residual_sigma = np.sqrt(np.diag(covariance))
    return {
        "parameters": parameters.tolist(),
        "sigma": sigma,
        "parameter_covariance": parameter_covariance,
        "chi2": chi2_value,
        "chi2_raw": chi2_raw,
        "dof": dof,
        "pte": float(chi2.sf(chi2_value, dof)),
        "hartlap": hartlap,
        "residual": residual.tolist(),
        "prediction": (x @ parameters).tolist(),
        "residual_sigma": residual_sigma.tolist(),
        "residual_sigma_normalized": (residual / residual_sigma).tolist(),
        "profile_hessian_eigenvalues": np.linalg.eigvalsh(hessian).tolist(),
        "profile_hessian_positive": positive,
        "bfgs_success": bool(best.success),
        "gradient_norm": float(np.linalg.norm(best.jac)),
        "trials": trials,
        "objective": "joint noisy-template Gaussian profile; no residual log determinant",
        "cross_term": bool(cross_term),
    }


def fit_joint_eiv(
    vector,
    jk_rows,
    *,
    nbin=None,
    template_vector=None,
    template_rows=None,
    scales=None,
    start=None,
    cross_term=True,
):
    """Fit rho0..rho5 to tau0/tau2/tau5 with the joint EIV covariance.

    ``vector`` is ordered ``[rho0..rho5, tau0, tau2, tau5]`` and ``jk_rows``
    contains the same delete-one measurements, one row per patch. Optional
    template columns add measured predictors such as ``u * delta_e``; their
    point values and rows have shapes ``(ndata, ntemplate)`` and
    ``(npatch, ndata, ntemplate)``.

    The objective is the profile chi-squared with the latent true templates
    profiled out, minimised by multi-start BFGS; it has no log-determinant
    term. The reported chi-squared is scaled by the Hartlap factor
    (N - d - 2) / (N - 1), N the number of rows and d the number of tau data;
    the parameter covariance is twice the inverse numerical profile Hessian,
    divided by the same factor. ``cross_term=False`` zeroes the rho-tau block
    of the covariance.

    Returns a dict with the ``parameters``, their ``sigma`` and
    ``parameter_covariance`` (None where the profile Hessian is not positive
    definite, ``profile_hessian_positive`` False), ``chi2``, ``dof``,
    ``pte``, ``hartlap``, the residuals, and the convergence diagnostics
    ``bfgs_success`` and ``gradient_norm`` of the chosen trial.
    """
    vector = np.asarray(vector, dtype=float)
    jk_rows = np.asarray(jk_rows, dtype=float)
    if vector.ndim != 1 or jk_rows.ndim != 2:
        raise ValueError("vector and jk_rows must be one- and two-dimensional")
    if nbin is None:
        if vector.size % 9:
            raise ValueError("vector length must be divisible by 9")
        nbin = vector.size // 9
    nbin = int(nbin)
    n_rho = 6 * nbin
    n_data = 3 * nbin
    if vector.size != 9 * nbin or jk_rows.shape[1] != vector.size:
        raise ValueError("vector/jk_rows must contain six rho and three tau blocks")

    x = rho_tau_design(vector[:n_rho].reshape(6, nbin))
    y = vector[n_rho:]
    x_rows = np.stack(
        [rho_tau_design(row[:n_rho].reshape(6, nbin)) for row in jk_rows], axis=0
    )
    y_rows = jk_rows[:, n_rho:]
    if template_vector is not None:
        template_vector = np.asarray(template_vector, dtype=float)
        template_rows = np.asarray(template_rows, dtype=float)
        if template_vector.ndim == 1:
            template_vector = template_vector[:, None]
        if template_rows.ndim == 2:
            template_rows = template_rows[:, :, None]
        if template_vector.shape[0] != n_data:
            raise ValueError("template_vector must have one row per tau datum")
        if template_rows.shape != (
            len(jk_rows),
            n_data,
            template_vector.shape[1],
        ):
            raise ValueError("template_rows must have shape (npatch, ndata, ntemplate)")
        x = np.concatenate((x, template_vector), axis=1)
        x_rows = np.concatenate((x_rows, template_rows), axis=2)
    elif template_rows is not None:
        raise ValueError("template_rows requires template_vector")

    n_parameters = x.shape[1]
    if scales is None:
        scales = [0.01, 1.0, 2.0] + [1.0] * (n_parameters - 3)
    scales = np.asarray(scales, dtype=float)
    if scales.shape != (n_parameters,):
        raise ValueError("scales must contain one value per fit parameter")
    return _profile_fit(x, y, x_rows, y_rows, scales, cross_term, start)


def systematic_correlation_matrix(rho, slope=None):
    """Return the matrix M of field correlations behind xi_sys.

    The PSF-error model is g_sys = alpha p + beta de + eta q (+ beta1 u de),
    with p the PSF ellipticity, de the star-minus-PSF ellipticity, q the size
    residual field, and u = T_PSF / <T_PSF> - 1. ``rho`` has shape
    ``(6, nbin)`` (rho0..rho5, one of xi+ or xi-); ``slope``, if given, has
    shape ``(4, nbin)``: the correlations of u de with p, de, q and itself.
    The result has shape ``(k, k, nbin)``, k = 3 or 4, symmetric in its first
    two axes, so that xi_sys = a^T M b for parameter vectors a, b.
    """
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 2 or rho.shape[0] != 6:
        raise ValueError("rho must have shape (6, nbin)")
    pairs = {(0, 0): 0, (1, 1): 1, (0, 1): 2, (2, 2): 3, (1, 2): 4, (0, 2): 5}
    size = 3 if slope is None else 4
    matrix = np.zeros((size, size, rho.shape[1]))
    for (i, j), index in pairs.items():
        matrix[i, j] = matrix[j, i] = rho[index]
    if slope is not None:
        slope = np.asarray(slope, dtype=float)
        if slope.shape != (4, rho.shape[1]):
            raise ValueError("slope must have shape (4, nbin)")
        for i in range(4):
            matrix[3, i] = matrix[i, 3] = slope[i]
    return matrix


def xi_psf_sys(parameters_a, parameters_b, matrix):
    """Return xi_sys = a^T M b for parameter samples a and b.

    ``parameters_a`` and ``parameters_b`` have shape ``(nsample, k)`` or
    ``(k,)`` and ``matrix`` comes from `systematic_correlation_matrix`.
    """
    a = np.atleast_2d(parameters_a)
    b = np.atleast_2d(parameters_b)
    return np.einsum("si,ijn,sj->sn", a, matrix, b)
