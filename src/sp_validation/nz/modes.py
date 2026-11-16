"""Likelihood-aware n(z) modes: the n(z) uncertainty a data vector can see.

Method by William d'Assignies Doumerg (likelihood-aware PCA following
Bernstein et al.). Given an ensemble of tomographic n(z) realisations and the
Fisher information of an observable with respect to n(z), it finds the few
n(z) distortions that carry the observable's χ², ordered by how much each one
contributes, and keeps enough of them to leave a chosen fraction of the χ²
behind.

The ensemble is whitened by its own covariance, C_n = V Λ Vᵀ, so that
n − n̄ = A w with A = V Λ^½ and w white. In those coordinates the observable's
χ² metric is G = Aᵀ F A, whose eigenvalues λ_k are the expected χ² each mode
contributes for a draw from the ensemble. The kept modes give

    n(z | u) = n̄ + Σ_k u_k d_k,    u ~ N(0, 1) under the ensemble,

with d_k the columns of A V_G and u = V_Gᵀ Λ^{-½} Vᵀ (n − n̄). The prior on u
is exactly a unit Gaussian only if the ensemble is Gaussian; plot
``encode(realisations)`` to check.

Only G is needed, so the Fisher matrix is never formed on the full (bin, z)
grid: :func:`observable_metric` differentiates the theory along the columns of
A, one finite difference per ensemble direction, and :func:`sacc_metric` does
so for a SACC data vector with its own covariance and the pyccl theory of
:func:`sp_validation.blinding.shear`. A precomputed Fisher matrix enters
through :func:`fisher_metric`.
"""

import copy
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NzModes:
    """An n(z) model n̄ + Σ u_k d_k on a fixed z grid.

    ``eigenvalues`` holds λ_k for every ensemble direction, not only the kept
    ones, so :attr:`chi2_left` can show what any truncation would discard.
    """

    z: np.ndarray  # (n_z,)
    mean: np.ndarray  # (n_bin, n_z)
    decoder: np.ndarray  # (n_modes, n_bin, n_z): the mode shapes d_k
    encoder: np.ndarray  # (n_modes, n_bin, n_z): u_k = <encoder_k, n − n̄>
    eigenvalues: np.ndarray  # (n_directions,), decreasing

    @property
    def n_modes(self):
        return len(self.decoder)

    @property
    def chi2_left(self):
        """Expected χ² discarded when keeping 0, 1, …, n_directions modes."""
        tail = np.cumsum(self.eigenvalues[::-1])[::-1]
        return np.append(tail, 0.0)

    def decode(self, u):
        """n(z) at mode amplitudes ``u`` (shape ``(..., n_modes)``)."""
        return self.mean + np.tensordot(np.asarray(u), self.decoder, axes=1)

    def encode(self, nz):
        """Mode amplitudes of ``nz`` (shape ``(..., n_bin, n_z)``)."""
        delta = np.asarray(nz) - self.mean
        return np.tensordot(delta, self.encoder, axes=([-2, -1], [1, 2]))

    def save(self, path):
        np.savez(
            path,
            z=self.z,
            mean=self.mean,
            decoder=self.decoder,
            encoder=self.encoder,
            eigenvalues=self.eigenvalues,
        )

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            return cls(**{k: f[k] for k in cls.__dataclass_fields__})


def fit(z, realisations, metric, threshold=0.01, n_modes=None, rtol=1e-12):
    """Find the modes of an n(z) ensemble under an observable's χ² metric.

    Parameters
    ----------
    z : (n_z,) array
    realisations : (n_real, n_bin, n_z) array
        n(z) draws. Each draw holds every bin, so modes span bins jointly.
    metric : callable
        ``metric(mean, directions)`` returns Aᵀ F A for the ``(n_dir, n_bin,
        n_z)`` stack of n(z) directions A about ``mean``; see
        :func:`fisher_metric`, :func:`observable_metric`, :func:`sacc_metric`.
    threshold : float
        Keep the fewest modes whose discarded expected χ² is below
        ``threshold`` times the total.
    n_modes : int, optional
        Keep exactly this many modes instead.
    rtol : float
        Ensemble directions with variance below ``rtol`` times the largest are
        treated as fixed.
    """
    real = np.asarray(realisations, float)
    _, n_bin, n_z = real.shape
    mean = real.mean(axis=0)
    flat = (real - mean).reshape(len(real), -1)

    lam, vec = np.linalg.eigh(np.cov(flat, rowvar=False))
    keep = lam > rtol * lam.max()
    lam, vec = lam[keep][::-1], vec[:, keep][:, ::-1]
    directions = (vec * np.sqrt(lam)).T.reshape(-1, n_bin, n_z)

    g = metric(mean, directions)
    lam_g, vec_g = np.linalg.eigh(0.5 * (g + g.T))
    lam_g, vec_g = np.maximum(lam_g[::-1], 0.0), vec_g[:, ::-1]

    if n_modes is None:
        tail = np.append(np.cumsum(lam_g[::-1])[::-1], 0.0)
        n_modes = int(np.argmax(tail < threshold * tail[0]))
    vg = vec_g[:, :n_modes]
    decoder = np.tensordot(vg.T, directions, axes=1)
    encoder = ((vec / np.sqrt(lam)) @ vg).T.reshape(n_modes, n_bin, n_z)
    return NzModes(np.asarray(z, float), mean, decoder, encoder, lam_g)


def fisher_metric(fisher):
    """Metric from a Fisher matrix over the flattened (bin, z) n(z) values."""
    fisher = np.asarray(fisher)

    def metric(mean, directions):
        a = directions.reshape(len(directions), -1)
        return a @ fisher @ a.T

    return metric


def observable_metric(theory, cov, step=0.1):
    """Metric of a data vector: Jᵀ C⁻¹ J along each ensemble direction.

    ``theory(nz)`` maps an ``(n_bin, n_z)`` n(z) to the data vector, and
    ``cov`` is its covariance. J is a central difference along each direction,
    scaled so the perturbation peaks at ``step`` times the peak of the mean
    n(z). A fixed amplitude keeps every direction in the linear regime and
    well above the theory's numerical noise, however small its variance.
    """
    cov = np.asarray(cov)

    def metric(mean, directions):
        peak = np.abs(mean).max()
        columns = []
        for d in directions:
            h = step * peak / np.abs(d).max()
            columns.append((theory(mean + h * d) - theory(mean - h * d)) / (2 * h))
        jac = np.stack(columns, axis=1)
        return jac.T @ np.linalg.solve(cov, jac)

    return metric


def sacc_metric(s, params=None, step=0.1):
    """Metric of a SACC data vector under :func:`sp_validation.blinding.shear`.

    Uses every data point in ``s`` (cut it to the statistic and scales the
    analysis uses first) and its covariance. Tracers ``source_0`` … carry the
    n(z); the z grid of the realisations must be theirs.
    """
    from .. import blinding

    params = blinding.fiducial() if params is None else params
    work = copy.deepcopy(s)

    def theory(nz):
        set_sacc_nz(work, nz)
        return blinding.shear(params, work)

    return observable_metric(theory, s.covariance.dense, step=step)


def sacc_nz(s):
    """The z grid and ``(n_bin, n_z)`` n(z) of a SACC's source tracers."""
    names = _source_tracers(s)
    z = np.asarray(s.tracers[names[0]].z, float)
    return z, np.stack([np.asarray(s.tracers[n].nz, float) for n in names])


def set_sacc_nz(s, nz):
    """Replace the n(z) of a SACC's source tracers, in place, bin by bin."""
    names = _source_tracers(s)
    if len(nz) != len(names):
        raise ValueError(f"{len(nz)} n(z) for {len(names)} source tracers")
    for name, n in zip(names, nz):
        s.tracers[name].nz = np.asarray(n, float)


def _source_tracers(s):
    names = sorted(
        (n for n, t in s.tracers.items() if getattr(t, "nz", None) is not None),
        key=lambda n: int(n.rsplit("_", 1)[-1]),
    )
    if not names:
        raise ValueError("SACC has no n(z) tracers")
    return names


def shift_stretch(z, nz, shift, stretch):
    """Each bin of ``nz`` shifted by ``shift`` and stretched about its mean.

    ``nz`` is ``(n_bin, n_z)``; ``shift`` and ``stretch`` are ``(..., n_bin)``,
    and the result ``(..., n_bin, n_z)`` keeps each bin's normalisation.
    """
    z, nz = np.asarray(z, float), np.asarray(nz, float)
    norm = np.trapezoid(nz, z, axis=-1)
    zbar = np.trapezoid(nz * z, z, axis=-1) / norm
    shift, stretch = np.broadcast_arrays(
        np.asarray(shift, float), np.asarray(stretch, float)
    )
    source = zbar[:, None] + (z - zbar[:, None] - shift[..., None]) / stretch[..., None]
    out = np.empty(source.shape)
    for idx in np.ndindex(source.shape[:-1]):
        out[idx] = np.interp(source[idx], z, nz[idx[-1]], left=0.0, right=0.0)
    return out * (norm / np.trapezoid(out, z, axis=-1))[..., None]


SHEAR_TYPES = {
    "xi": ("galaxy_shear_xi_plus", "galaxy_shear_xi_minus"),
    "cl": ("galaxy_shear_cl_ee",),
}


def cut_sacc(s, statistic="xi", scales=None):
    """A copy of ``s`` holding one statistic, optionally within a scale range.

    ``statistic`` is ``"xi"`` (ξ±, ``scales`` in arcmin on the ``theta`` tag)
    or ``"cl"`` (C_ℓ^EE, ``scales`` in ℓ on the ``ell`` tag).
    """
    s = s.copy()
    keep = SHEAR_TYPES[statistic]
    for dt in s.get_data_types():
        if dt not in keep:
            s.remove_selection(data_type=dt)
    if scales is not None:
        tag = "theta" if statistic == "xi" else "ell"
        s.remove_selection(**{f"{tag}__lt": scales[0]})
        s.remove_selection(**{f"{tag}__gt": scales[1]})
    return s


def main(argv=None):
    """Write the n(z) modes of a realisation ensemble for a SACC's statistic.

    Reads only the SACC's tracers and covariance, never its data vector, so a
    blinded SACC is safe to use.
    """
    import argparse

    import sacc

    from .. import blinding

    p = argparse.ArgumentParser(prog="python -m sp_validation.nz.modes")
    p.add_argument("sacc", help="SACC with source tracers and a covariance")
    p.add_argument(
        "realisations",
        help="npz with z (n_z,) and realisations (n_real, n_bin, n_z)",
    )
    p.add_argument("-o", "--output", required=True, help="modes .npz")
    p.add_argument("--statistic", choices=sorted(SHEAR_TYPES), default="xi")
    p.add_argument(
        "--scales",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        required=True,
        help="the analysis scale cut: θ in arcmin for xi, ℓ for cl",
    )
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--n-modes", type=int)
    p.add_argument(
        "--A-IA",
        type=float,
        required=True,
        dest="a_ia",
        help="NLA amplitude at which the metric is evaluated; IA changes the modes",
    )
    a = p.parse_args(argv)

    s = cut_sacc(sacc.Sacc.load_fits(a.sacc), a.statistic, a.scales)
    z, _ = sacc_nz(s)
    with np.load(a.realisations) as f:
        z_in, real = f["z"], f["realisations"]
    if not np.array_equal(z_in, z):
        real = np.stack(
            [[np.interp(z, z_in, n, left=0, right=0) for n in r] for r in real]
        )
    params = {**blinding.fiducial(), "A_IA": a.a_ia}
    modes = fit(
        z, real, sacc_metric(s, params), threshold=a.threshold, n_modes=a.n_modes
    )
    modes.save(a.output)
    left = modes.chi2_left
    print(
        f"{modes.n_modes} modes keep {1 - left[modes.n_modes] / left[0]:.2%} of "
        f"the expected Δχ² = {left[0]:.3g} of {a.statistic}; wrote {a.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
