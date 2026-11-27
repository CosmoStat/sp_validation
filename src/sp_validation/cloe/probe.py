"""The CLOE JAX likelihood of a SACC, as a cloerun probe.

cloerun (CLOE's chain runner) owns the priors, the sampler and the chain files; this
module is its external-probe contract (``cloerun._core.probes``) over a
:class:`~.vector.Theory` of the harmonic spectra of one SACC::

    probe:
      kind: sp_validation.cloe.probe
      engine: cosmopower_mead2020_feedback   # theory.cloe.ENGINES
      ia_z_pivot: 0.62
      magnification: true                    # slopes mag_slope_<i>
      reference: {Omega_c: ..., Omega_b: ..., h: ..., n_s: ..., A_s: ..., m_nu: ...}
      threads: 1                             # threads per likelihood process
      chunk: 1                               # points JAX evaluates at once
    data:
      sacc: forecast.sacc                    # relative to the config's directory
      expect_blind: none                     # the stamp the SACC must carry
    scale_cuts: {families: [GG, GL, LL], shear: [300, 1600], ...}   # vector.cut
    prune_parameters: true

The ``reference`` cosmology sets the distances of the linear-bias cut. A tracer
with no kept band is not in the model and its parameters are not read; with
``prune_parameters: true`` cloerun drops them. The derived outputs are ``sigma8``,
``S8`` and ``Omega_m``.

The likelihood is fastest as one single-threaded JAX process per core: nautilus's
pool (``sampler.settings.pool: [n_workers, ...]``) splits each batch between
workers, each of which rebuilds the likelihood and pins itself to its own
``threads`` cores. JAX starts threads as soon as it computes, and a process that
forks after that can deadlock, so importing this module makes ``multiprocessing``
spawn its workers.
"""

import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from .. import sacc_io
from .vector import Theory, Vector, cut, reads

multiprocessing.set_start_method("spawn", force=True)


def parse_data(data, config_dir):
    """The SACC (resolved against the config's directory) and its expected stamp."""
    return SimpleNamespace(
        sacc=str(Path(config_dir, data["sacc"])), expect_blind=data["expect_blind"]
    )


def theory(probe, scale_cuts, s, sampled, fixed):
    """The :class:`~.vector.Theory` of SACC ``s`` under ``scale_cuts``, reading the
    ``sampled`` names and ``fixed`` values its tracers use."""
    kept = cut(s, reference=probe["reference"], **scale_cuts)
    vector = Vector.from_sacc(
        kept,
        engine=probe["engine"],
        ia_z_pivot=probe["ia_z_pivot"],
        magnification=probe["magnification"],
    )
    return Theory(
        vector,
        [n for n in sampled if reads(vector, n)],
        fixed={n: v for n, v in fixed.items() if reads(vector, n)},
        correction_reference=probe["reference"],
    )


def build_likelihood(cfg):
    threads = cfg.probe["threads"]
    os.environ["XLA_FLAGS"] = (
        f"--xla_cpu_multi_thread_eigen={str(threads > 1).lower()} "
        f"intra_op_parallelism_threads={threads}"
    )
    os.environ["OMP_NUM_THREADS"] = str(threads)
    worker = multiprocessing.current_process()._identity  # (k,) in the k-th worker
    if worker:
        cores = sorted(os.sched_getaffinity(0))
        first = (worker[-1] - 1) * threads % len(cores)
        os.sched_setaffinity(0, cores[first : first + threads])
    s = sacc_io.load(cfg.data.sacc, blind=cfg.data.expect_blind)
    model = theory(
        cfg.probe,
        cfg.scale_cuts,
        s,
        list(cfg.sampled_parameters),
        cfg.fixed_parameters,
    )
    unread = sorted(set(cfg.sampled_parameters) - set(model.names))
    if unread and not cfg.prune_parameters:
        raise ValueError(
            f"sampled parameters {unread} are not read: set prune_parameters: true"
        )
    model.scalar = jax.jit(model.loglike)
    model.scalar_derived = jax.jit(model.derived)
    model.batch = model._batched(model.loglike, cfg.probe["chunk"])
    model.batch_derived = model._batched(model.derived, cfg.probe["chunk"])
    return model


def obs_labels(cfg):
    return []


def vector_parameter_names(likelihood):
    return set()


def parameter_names(likelihood):
    """The sampled and fixed parameters the model reads."""
    return likelihood.names + list(likelihood.fixed)


def _theta(likelihood, parameters):
    return jnp.stack(
        [jnp.asarray(parameters[n], float) for n in likelihood.names], axis=-1
    )


def loglike(likelihood, parameters):
    return float(likelihood.scalar(_theta(likelihood, parameters)))


def loglike_batch(likelihood, parameters):
    return np.asarray(likelihood.batch(_theta(likelihood, parameters)))


def derived_outputs(likelihood, parameters, names):
    values = np.asarray(likelihood.scalar_derived(_theta(likelihood, parameters)))
    return [float(values[likelihood.DERIVED.index(n)]) for n in names]


def derived_outputs_batch(likelihood, parameters, names):
    values = np.asarray(likelihood.batch_derived(_theta(likelihood, parameters)))
    return [values[:, likelihood.DERIVED.index(n)] for n in names]


def tracked_packages(cfg):
    return ["sp_validation", "cloelib", "jax", "jaxlib", "camb", "interpax"]
