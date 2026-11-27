"""The harmonic 3x2pt forecast of a :class:`~.survey.Survey`, step by step.

Each step reads the products of the ones before it in one output directory::

    python -m sp_validation.cloe.forecast masks      RECIPE OUT  # masks.npz
    python -m sp_validation.cloe.forecast spectra    RECIPE OUT  # spectra.npz
    python -m sp_validation.cloe.forecast covariance RECIPE OUT  # gaussian.npz
    python -m sp_validation.cloe.forecast ssc        RECIPE OUT  # ssc.npz
    python -m sp_validation.cloe.forecast sacc       RECIPE OUT  # forecast.sacc
    python -m sp_validation.cloe.forecast fisher     RECIPE OUT CUT [FAMILIES]
    python -m sp_validation.cloe.forecast chain      RECIPE OUT CUT NAME [options]

``spectra`` is the CLOE JAX theory at the input cosmology at every ell;
``covariance`` the windows of the coupled bandpowers (:mod:`.covariance`) and
their Gaussian covariance; ``ssc`` the super-sample term (run it with
``OMP_NUM_THREADS=1``: its worker processes fork after pyccl has run); ``sacc`` the noiseless data vector, the
windows applied to those spectra, as a SACC with lens and source tracers, windows
and covariance, stamped unblinded. ``fisher`` and ``chain`` read a SACC (by default
``forecast.sacc``, or ``--sacc``, e.g. a GLASS mock's) under the recipe's scale cut
``CUT``, for spectrum families ``GG,GL,LL`` (default all). A chain is cloerun's
nested sampling (nautilus) on :mod:`.probe`, written to ``chains/NAME.npz`` with a
summary ``chains/NAME.json``.

Run the steps inside the container on an allocation; ``--processes`` sets the
covariance's worker processes and ``--cores`` a chain's likelihood workers.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .. import blinding, sacc_io
from .survey import Survey

FAMILIES = ("GG", "GL", "LL")
DERIVED = ("sigma8", "S8", "Omega_m")


def spectra(survey):
    """(n_pair, lmax + 1) C_ell of :attr:`Survey.pairs` at :attr:`Survey.truth`, with
    every prefactor (spin-2, magnification), as the likelihood predicts them."""
    import jax.numpy as jnp

    from .theory.vector import interpolation_matrix
    from .vector import Block, Theory, Vector, reads

    lmax = survey["lmax"]
    labels = [tuple(map(survey.label, p)) for p in survey.pairs]
    nz = {survey.label(f): (survey.z, n) for f, n in survey.nz.items()}
    vector = Vector(
        [Block("cl", labels, np.ones((1, lmax + 1)))],  # one band: the nodes to lmax
        nz,
        ia_z_pivot=survey["probe"]["ia_z_pivot"],
        magnification=survey["probe"]["magnification"],
        engine=survey["probe"]["engine"],
    )
    fixed = {k: v for k, v in survey.truth.items() if reads(vector, k)}
    theory = Theory(vector, [], fixed=fixed)
    nodes, _ = theory.spectra(jnp.zeros(0))
    cl = np.asarray(nodes) @ interpolation_matrix(vector.ells, lmax).T
    return cl, dict(zip(theory.DERIVED, np.asarray(theory.derived(jnp.zeros(0)))))


def data_vector(survey, bands, op, cov, metadata):
    """A SACC of ``bands`` (n_pair, n_band) with windows ``op`` and covariance
    ``cov``, its tracers the survey's n(z)."""
    lens = [survey.nz[f"g{i}"] for i in range(survey.n_lens)]
    source = [survey.nz[f"l{i}"] for i in range(survey.n_source)]
    s = sacc_io.new_sacc(
        [(survey.z, n) for n in source], metadata, [(survey.z, n) for n in lens]
    )
    ell = np.arange(survey["lmax"] + 1.0)
    leff = survey.binning @ ell
    for (a, b), x, w in zip(survey.pairs, bands, op):
        kinds = a[0] + b[0]
        dtype = {"gg": sacc_io.CL_GG, "gl": sacc_io.CL_GE, "ll": sacc_io.CL_EE}[kinds]
        tracers = (survey.sacc_name(b), survey.sacc_name(a))
        if kinds != "gl":
            tracers = tracers[::-1]
        sacc_io.add_cl(s, dtype, tracers, leff, x, window_ells=ell, window_weights=w.T)
    s.add_covariance(cov)
    return s


def fisher(survey, s, cut, families=FAMILIES):
    """Fisher widths of sigma8, S8 and Omega_m from SACC ``s`` under the recipe's
    cut ``cut``, with the priors' widths (flat: (max - min) / sqrt 12) added."""
    import jax
    import jax.numpy as jnp

    from .probe import theory as build

    priors = survey.priors
    model = build(
        survey.probe(),
        dict(survey["cuts"][cut], families=list(families)),
        s,
        list(priors),
        dict(m_nu=survey["cosmology"]["m_nu"]),
    )
    theta = jnp.asarray([survey.truth[n] for n in model.names])
    jac = np.asarray(jax.jacfwd(model.model)(theta))

    def width(p):
        return p["scale"] if p["dist"] == "norm" else (p["max"] - p["min"]) / 12**0.5

    prior = np.diag([width(priors[n]) ** -2 for n in model.names])
    cov = np.linalg.inv(jac.T @ model.icov @ jac + prior)
    d = np.asarray(jax.jacobian(model.derived)(theta))
    sd = np.sqrt(np.diag(d @ cov @ d.T))
    return dict(zip(model.DERIVED, sd.tolist()), bands=len(model.data))


def chain(
    survey,
    sacc_path,
    cut,
    name,
    out,
    families=FAMILIES,
    *,
    n_eff=4000,
    n_live=2000,
    seed=1,
    cores=64,
    expect_blind="none",
):
    """A nautilus chain through cloerun on ``sacc_path`` (:mod:`.probe`); writes
    ``out/NAME.npz`` (cloerun's chain, wall time) and ``out/NAME.json`` (posterior
    mean and standard deviation of sigma8, S8, Omega_m; effective sample size)."""
    import os
    import tempfile

    import yaml
    from cloerun._core.config import RunConfig
    from cloerun.chain import run_chain

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=name, dir=os.environ.get("SLURM_TMPDIR")))
    cfg = dict(
        probe=survey.probe(),
        data=dict(sacc=str(Path(sacc_path).resolve()), expect_blind=expect_blind),
        scale_cuts=dict(survey["cuts"][cut], families=list(families)),
        prune_parameters=True,
        fixed_parameters=dict(m_nu=survey["cosmology"]["m_nu"]),
        sampled_parameters=survey.priors,
        derived_outputs=list(DERIVED),
        sampler=dict(
            kind="nautilus",
            settings=dict(
                n_live=n_live,
                n_batch=1000,
                vectorized=True,
                pool=[cores, 32],
                seed=seed,
                filepath=str(tmp / "nautilus.hdf5"),
                resume=False,
            ),
            run_settings=dict(n_eff=n_eff, discard_exploration=True, verbose=True),
        ),
        output=dict(directory=str(tmp), name=name),
    )
    path = tmp / "run.yaml"
    path.write_text(yaml.safe_dump(json.loads(json.dumps(cfg, default=float))))
    start = time.time()
    result = run_chain(RunConfig.from_yaml(str(path)))
    wall = time.time() - start
    columns = list(result.columns)
    np.savez(
        out / f"{name}.npz",
        samples=result.samples,
        columns=np.array(columns),
        config=path.read_text(),
        wall=wall,
    )
    lw = result.samples[:, columns.index("logweight")]
    w = np.exp(lw - lw.max())
    w /= w.sum()
    summary = dict(name=name, wall=wall, n_eff=float(1 / np.sum(w**2)))
    for k in DERIVED:
        x = result.samples[:, columns.index(k)]
        m = float(w @ x)
        summary[k] = [m, float(np.sqrt(w @ (x - m) ** 2))]
    (out / f"{name}.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m sp_validation.cloe.forecast")
    ap.add_argument("step")
    ap.add_argument("recipe")
    ap.add_argument("out", type=Path)
    ap.add_argument("cut", nargs="?")
    ap.add_argument("name", nargs="?")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--sacc", type=Path)
    ap.add_argument("--processes", type=int, default=48)
    ap.add_argument("--cores", type=int, default=64)
    ap.add_argument("--n-eff", type=int, default=4000)
    ap.add_argument("--n-live", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    survey, out = Survey.load(a.recipe), a.out
    out.mkdir(parents=True, exist_ok=True)
    families = a.families.split(",")
    sacc_path = a.sacc or out / "forecast.sacc"

    if a.step == "masks":
        np.savez(out / "masks.npz", **survey.masks())
    elif a.step == "spectra":
        cl, derived = spectra(survey)
        np.savez(out / "spectra.npz", cl=cl, **derived)
        print("derived at the input point", derived)
    elif a.step == "covariance":
        from . import covariance as cv

        masks = dict(np.load(out / "masks.npz"))
        cl = np.load(out / "spectra.npz")["cl"]
        couplings = cv.Couplings(masks, survey["lmax"], out / "couplings")
        op = cv.operators(survey, masks, couplings)
        gauss = cv.gaussian(survey, masks, cl, couplings, a.processes)
        noise = cv.noise(survey, masks)
        np.savez(
            out / "gaussian.npz",
            op=op,
            gauss=gauss,
            noise=[noise[f] for f in survey.fields],
        )
    elif a.step == "ssc":
        from .covariance import super_sample

        masks = dict(np.load(out / "masks.npz"))
        op = np.load(out / "gaussian.npz")["op"]
        np.savez(out / "ssc.npz", ssc=super_sample(survey, masks, op, a.processes))
    elif a.step == "sacc":
        g = np.load(out / "gaussian.npz")
        cl = np.load(out / "spectra.npz")["cl"]
        bands = np.einsum("pbl,pl->pb", g["op"], cl)
        cov = g["gauss"] + np.load(out / "ssc.npz")["ssc"]
        meta = dict(origin="noiseless forecast", covariance="Gaussian iNKA + SSC")
        s = data_vector(survey, bands, g["op"], cov, meta)
        sacc_io.save(s, out / "forecast.sacc", blind=blinding.NONE)
    elif a.step == "fisher":
        s = sacc_io.load(sacc_path, blind="none")
        result = fisher(survey, s, a.cut, families)
        tag = "".join(sorted(families))
        (out / f"fisher_{a.cut}_{tag}.json").write_text(json.dumps(result))
        print(result)
    elif a.step == "chain":
        print(
            chain(
                survey,
                sacc_path,
                a.cut,
                a.name,
                out / "chains",
                families,
                n_eff=a.n_eff,
                n_live=a.n_live,
                seed=a.seed,
                cores=a.cores,
            )
        )
    else:
        raise SystemExit(f"unknown step {a.step}")


if __name__ == "__main__":
    main()
