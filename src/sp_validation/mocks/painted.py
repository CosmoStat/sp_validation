"""Catalogue-painted GLASS mocks: a known lensing signal on a real catalogue.

A painted mock keeps everything about a real shear catalogue except the
cosmological signal: its galaxy positions, weights, PSF columns and the
modulus of each calibrated ellipticity. Each ellipticity is rotated to a
random angle, which erases the real shear, and a lognormal GLASS sky is
sheared onto it. Every seed draws an independent sky and independent
rotations; only the geometry is fixed, as it is for the one real data vector,
so the scatter across seeds is the variance conditional on the survey's
geometry. The sampled mocks of ``mocks.glass`` draw synthetic galaxies
instead.

Two stages, each with its own CLI and workflow rule:

1. ``generate_lensing_maps`` makes one sky per seed: lognormal matter shells
   from ``mocks.glass``'s cosmology, projected through a single source n(z).
   Its product is the convergence alm (``kappa_alm.npz``); ``spectra.npz``
   holds that alm's realised spectra beside the shell-ensemble theory.
   Seed-independent shell spectra are solved once and cached
   (``prepare_lensing_spectra``).
2. ``paint_catalogue`` streams a catalogue and evaluates convergence and shear
   at each galaxy position with a non-uniform spherical-harmonic transform
   (``evaluate_alm``). No pixel window enters anywhere: the shells are drawn
   from unwindowed spectra and the sky is never read off a pixel grid.
   Optional injections, PSF leakage ``alpha e_PSF``, a constant additive term
   and a pure-B field, are added on top of the same noise, so a catalogue with
   an injection differs from the clean one by exactly the injection.

Skies are kept as alms in npz: they are inputs to painting, not data vectors.
Anything measured on a painted catalogue goes through cosmo_val to SACC.
Maps follow HEALPix; catalogue e2 is ``pol_factor`` times HEALPix gamma2
(``-1`` for UNIONS, as in cosmo_val), and every output records it.
``glass`` and ``ducc0`` are imported inside the functions that use them.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from sp_validation.mocks.glass import (
    Cosmology_from_camb,
    build_camb_params,
    build_shells,
    generate_convergence,
    matter_shell_cls,
)

__all__ = [
    "source_shell_weights",
    "projected_convergence_weights",
    "projected_theory",
    "solve_shell_spectra",
    "check_gaussian_solve",
    "prepare_lensing_spectra",
    "shear_alm",
    "read_sky",
    "evaluate_alm",
    "generate_lensing_maps",
    "draw_b_mode_alm",
    "b_mode_cl",
    "paint_shear",
    "paint_catalogue",
]


def source_shell_weights(z, nz, shells):
    """Normalized non-negative GLASS partition of a single source n(z)."""
    import glass

    z, nz = np.asarray(z), np.asarray(nz)
    if z.ndim != 1 or nz.shape != z.shape or np.any(np.diff(z) <= 0):
        raise ValueError("n(z) needs matching 1D arrays and increasing redshifts")
    if not np.all(np.isfinite(nz)) or np.any(nz < 0):
        raise ValueError("n(z) must be finite and non-negative")
    norm = np.trapezoid(nz, z)
    if norm <= 0:
        raise ValueError("n(z) has zero integral")
    weights = glass.partition(z, nz / norm, shells)
    return weights / np.sum(weights)


def projected_convergence_weights(pars, shells, weights):
    """Linear response of the n(z)-averaged convergence to matter shells."""
    import glass

    convergence = glass.MultiPlaneConvergence(Cosmology_from_camb(pars))
    transfer = np.zeros(len(shells))
    for i, (shell, weight) in enumerate(zip(shells, weights, strict=True)):
        basis = np.zeros(len(shells))
        basis[i] = 1
        convergence.add_window(basis, shell)
        transfer += weight * convergence.kappa
    return transfer


def projected_theory(cls, transfer, lmax, ncorr=3):
    """Exact shell-ensemble convergence and shear-E spectra.

    The source partition and the finite ``ncorr`` covariance are identical to
    the map generator's. The spin-2 / convergence conversion is spherical,
    including its low-ell factor (not the flat-sky approximation).
    """
    import glass

    kk = np.zeros(lmax + 1)
    for i, ti in enumerate(transfer):
        for j in range(max(0, i - ncorr), i + 1):
            kk += (
                (1 if i == j else 2)
                * ti
                * transfer[j]
                * glass.getcl(cls, i, j, lmax=lmax)
            )
    ell = np.arange(lmax + 1, dtype=float)
    factor = np.zeros_like(ell)
    factor[2:] = (ell[2:] + 2) * (ell[2:] - 1) / (ell[2:] * (ell[2:] + 1))
    return dict(ell=ell.astype(int), cl_kk=kk, cl_ee=kk * factor)


def _solve_shell_spectrum(task):
    """Solve one independent GLASS transformation pair in a worker process."""
    import glass.grf

    index, i, j, cl, first, second = task
    if not len(cl):
        return index, cl.copy(), cl.copy(), 3
    gl, realized, info = glass.grf.solve(
        cl,
        first,
        second,
        pad=2 * len(cl),
        monopole=0.0 if cl[0] == 0 else None,
    )
    return index, gl, realized[: len(cl)], info


def solve_shell_spectra(fields, cls, workers=1):
    """GLASS's padded solver, parallel over independent shell pairs.

    Returns Gaussian spectra, achieved matter spectra, and solver status.
    Spawned processes each use one native thread; the caller's OpenMP setting
    is restored before map generation. No fork follows CAMB's OpenMP runtime.
    """
    import multiprocessing
    import os
    from concurrent.futures import ProcessPoolExecutor

    import glass

    tasks = [
        (index, i, j, cl, fields[i], fields[j])
        for index, (i, j, cl) in enumerate(glass.enumerate_spectra(cls))
    ]
    results = [None] * len(tasks)
    if workers <= 1:
        solved = map(_solve_shell_spectrum, tasks)
        for index, gl, realized, info in solved:
            results[index] = gl, realized, info
    else:
        native_threads = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        saved_threads = {key: os.environ.get(key) for key in native_threads}
        os.environ.update({key: "1" for key in native_threads})
        try:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                for done, (index, gl, realized, info) in enumerate(
                    pool.map(_solve_shell_spectrum, tasks), 1
                ):
                    results[index] = gl, realized, info
                    if len(gl):
                        print(
                            f"Solved shell spectrum {done}/{len(tasks)} (status {info})",
                            flush=True,
                        )
        finally:
            for key, value in saved_threads.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    gaussian, realized, info = zip(*results, strict=True)
    return list(gaussian), list(realized), np.asarray(info)


# Largest tolerated scale-relative difference between achieved and requested.
SOLVER_TOLERANCE = 1e-4


def check_gaussian_solve(requested, achieved):
    """Worst residual of the lognormal solve; raises above ``SOLVER_TOLERANCE``.

    The residual of each pair is ``max_l |achieved - requested| /
    sqrt(C_ii C_jj)``. Scaling by the autos rather than the pair's own
    spectrum keeps it finite where a cross-spectrum crosses zero, which is
    where GLASS's own convergence flag misfires.
    """
    import glass

    pairs = list(glass.enumerate_spectra(requested))
    autos = {i: np.asarray(cl) for i, j, cl in pairs if i == j}
    worst = 0.0
    for (i, j, wanted), got in zip(pairs, achieved, strict=True):
        if not len(wanted):
            continue
        scale = np.sqrt(np.abs(autos[i] * autos[j]))[: len(wanted)]
        keep = scale > 0
        residual = np.abs(np.asarray(got)[: len(wanted)] - wanted)[keep] / scale[keep]
        worst = max(worst, float(np.max(residual, initial=0.0)))
    if worst > SOLVER_TOLERANCE:
        raise ValueError(f"Lognormal solve misses the requested spectra by {worst}")
    return worst


def prepare_lensing_spectra(config, cache_dir, workers=1):
    """Cache seed-independent matter/Gaussian spectra and source theory.

    A lock serializes writers. The key includes cosmology, shell geometry,
    resolution, n(z) contents, GLASS version and the correlation truncation.
    No seed enters the key. Cache files contain numeric arrays only.
    The spectra are unwindowed: the shells are sampled at pixel centres,
    never averaged over pixels, so the map's alm carries no pixel window.
    Every call checks the cached solve (``check_gaussian_solve``) and returns
    its worst residual as ``spectra["solver_residual"]``.
    """
    import fcntl
    import hashlib
    import json
    from pathlib import Path

    import glass

    nz_path = Path(config.nz_path)
    physics = {
        key: getattr(config, key)
        for key in (
            "dx",
            "zmax",
            "h",
            "Om",
            "Ob",
            "ns",
            "sigma8",
            "mnu",
            "log_T_AGN",
            "As_init",
            "kmax",
            "limber",
        )
    }
    physics.update(lmax=config.lmax, ncorr=3, glass=glass.__version__, schema=3)
    digest = hashlib.sha256(
        json.dumps(physics, sort_keys=True).encode() + nz_path.read_bytes()
    ).hexdigest()[:20]
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"cls_{digest}.npz"
    # Shells and the background also define the multi-plane propagation.
    pars = build_camb_params(config)
    shells = build_shells(config, pars)
    with (root / f"cls_{digest}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            print(f"Computing seed-independent matter spectra: {path}", flush=True)
            matter_path = path.with_suffix(".matter.npz")
            if matter_path.exists():
                with np.load(matter_path) as saved:
                    cls = [saved[f"cl_{i}"] for i in range(len(saved.files))]
            else:
                cls = matter_shell_cls(config, pars, shells, limber=config.limber)
                matter_tmp = matter_path.with_suffix(".tmp.npz")
                np.savez(matter_tmp, **{f"cl_{i}": c for i, c in enumerate(cls)})
                matter_tmp.replace(matter_path)
            nz = np.loadtxt(nz_path)
            if nz.ndim != 2 or nz.shape[1] != 2:
                raise ValueError("Projected non-tomographic maps need two-column n(z)")
            if nz[-1, 0] > config.zmax and np.any(nz[nz[:, 0] > config.zmax, 1] > 0):
                raise ValueError("zmax truncates nonzero source n(z)")
            weights = source_shell_weights(nz[:, 0], nz[:, 1], shells)
            transfer = projected_convergence_weights(pars, shells, weights)
            theory = projected_theory(cls, transfer, config.lmax)
            requested = glass.discretized_cls(cls, lmax=config.lmax, ncorr=3)
            gls, achieved, solver_info = solve_shell_spectra(
                glass.lognormal_fields(shells), requested, workers=workers
            )
            arrays = {f"matter_{i}": c for i, c in enumerate(cls)}
            arrays.update({f"achieved_{i}": c for i, c in enumerate(achieved)})
            arrays["solver_info"] = solver_info
            arrays.update({f"gaussian_{i}": c for i, c in enumerate(gls)})
            arrays.update(theory)
            arrays.update(weights=weights, transfer=transfer, nspectra=len(cls))
            temporary = path.with_suffix(".tmp.npz")
            np.savez(temporary, **arrays)
            temporary.replace(path)
            path.with_suffix(".json").write_text(json.dumps(physics, indent=2))
    with np.load(path) as cached:
        spectra = {key: cached[key] for key in cached.files}
    print(f"Using spectra cache: {path}", flush=True)
    count = int(spectra["nspectra"])
    requested = glass.discretized_cls(
        [spectra[f"matter_{i}"] for i in range(count)], lmax=config.lmax, ncorr=3
    )
    spectra["solver_residual"] = check_gaussian_solve(
        requested, [spectra[f"achieved_{i}"] for i in range(count)]
    )
    return pars, shells, spectra, path


def shear_alm(kappa_alm):
    """Shear E-mode alm of a convergence alm.

    The spherical spin factor ``-sqrt((l+2)(l-1)/(l(l+1)))``, as GLASS's
    ``shear_from_convergence`` applies it. The shear maps and the painted
    catalogue both derive from this function.
    """
    import healpy as hp

    lmax = hp.Alm.getlmax(len(kappa_alm))
    ell = np.arange(lmax + 1, dtype=float)
    factor = np.zeros_like(ell)
    factor[2:] = -np.sqrt((ell[2:] + 2) * (ell[2:] - 1) / (ell[2:] * (ell[2:] + 1)))
    return hp.almxfl(kappa_alm, factor)


def read_sky(maps_dir):
    """The convergence alm saved by ``generate_lensing_maps`` in ``maps_dir``."""
    from pathlib import Path

    with np.load(Path(maps_dir) / "kappa_alm.npz") as saved:
        return saved["alm"]


def evaluate_alm(alm, ra, dec, *, epsilon=1e-10, nthreads=1):
    """Non-uniform spherical-harmonic synthesis at sky positions (degrees).

    ``alm`` has shape ``(1, nalm)`` for a scalar field or ``(2, nalm)`` for the
    ``(E, B)`` pair of a spin-2 field, in healpy ordering. Returns ``(1, N)``
    or HEALPix-convention ``(Q, U)`` = ``(gamma1, gamma2)`` of shape ``(2, N)``.
    This is ``ducc0.sht.experimental.synthesis_general``: the band-limited sum
    at each position to relative accuracy ``epsilon``, with no pixel window
    (a nearest-pixel lookup of a map would add one). Each call
    has a fixed setup of about one full-sky synthesis (~4 s at lmax 8192 on 28
    threads, ~75 s on one) and a negligible per-point cost (16M points in
    under 5 s on 28 threads), so callers pass millions of positions at a time.
    """
    import ducc0
    import healpy as hp

    alm = np.atleast_2d(alm)
    if alm.shape[0] not in (1, 2):
        raise ValueError("alm needs one scalar or two (E, B) rows")
    ra, dec = np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)
    if ra.shape != dec.shape or ra.ndim != 1:
        raise ValueError("Positions must be matching 1D arrays")
    if not (np.all(np.isfinite(ra)) and np.all(np.isfinite(dec))):
        raise ValueError("Galaxy positions must be finite")
    if np.any(np.abs(dec) > 90):
        raise ValueError("Declination lies outside [-90, 90]")
    loc = np.empty((len(ra), 2))
    loc[:, 0] = np.radians(90 - dec)
    loc[:, 1] = np.radians(np.mod(ra, 360))
    return ducc0.sht.experimental.synthesis_general(
        alm=np.ascontiguousarray(alm, dtype=complex),
        spin=0 if alm.shape[0] == 1 else 2,
        lmax=hp.Alm.getlmax(alm.shape[1]),
        loc=loc,
        epsilon=epsilon,
        nthreads=nthreads,
    )


def generate_lensing_maps(config, output_dir, cache_dir, workers=1):
    """Save the projected convergence alm, its maps, and its spectra.

    Matter shells are lognormal, with three neighbouring cross-correlations.
    Source-plane convergence is integrated before the linear shear transform,
    avoiding a spherical transform for every source shell.
    ``kappa_alm.npz`` (complex128 healpy-ordered ``alm`` of the convergence
    map up to ``lmax``, and ``nside``) is the sky that ``paint_catalogue``
    evaluates. The float32 ``gamma1``/``gamma2`` maps are its shear E mode
    synthesised on the grid; ``kappa`` is GLASS's pixel convergence, of which
    the alm is the band-limited transform. The ``realization_*`` spectra are
    the alm's own full-sky power (KK, EE, KE; gamma has no B mode), what a
    catalogue painted with ``shear='gamma'`` carries; ``cl_kk``/``cl_ee`` are
    the shell-ensemble theory they scatter around, with no pixel window.
    A statistic measured on a footprint scatters around the realisation by
    that footprint's sample variance; the same statistic of the noise-free
    ``g*_true`` columns is the exact answer for one footprint. Reduced shear
    adds its own EE and BB.
    """
    import json
    import time
    from pathlib import Path

    import healpy as hp

    start = time.perf_counter()
    if config.lmax > 3 * config.nside - 1 or config.lmax < 2:
        raise ValueError("lmax must be between 2 and 3*nside-1")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pars, shells, spectra, cache = prepare_lensing_spectra(
        config, cache_dir, workers=workers
    )
    gls = [spectra[f"gaussian_{i}"] for i in range(int(spectra["nspectra"]))]
    kappa = np.zeros(hp.nside2npix(config.nside))
    shell_maps = generate_convergence(config, pars, shells, gls)
    for i, (_, kappa_i) in enumerate(shell_maps):
        kappa += spectra["weights"][i] * kappa_i
        print(f"Matter shell {i + 1}/{len(shells)}", flush=True)
    kappa_alm = hp.map2alm(kappa, lmax=config.lmax, pol=False, iter=3)
    elm = shear_alm(kappa_alm)
    gamma1, gamma2 = hp.alm2map_spin(
        [elm, np.zeros_like(elm)], config.nside, 2, config.lmax
    )
    np.savez(out / "kappa_alm.npz", alm=kappa_alm, nside=config.nside)
    for name, values in (("kappa", kappa), ("gamma1", gamma1), ("gamma2", gamma2)):
        np.save(out / f"{name}.npy", values.astype(np.float32))
    del kappa, gamma1, gamma2, shell_maps
    np.savez(
        out / "spectra.npz",
        **{key: spectra[key] for key in ("ell", "cl_kk", "cl_ee")},
        realization_kk=hp.alm2cl(kappa_alm),
        realization_ee=hp.alm2cl(elm),
        realization_ke=hp.alm2cl(kappa_alm, elm),
    )
    metadata = dict(
        config=asdict(config),
        lmax=config.lmax,
        spectra_cache=str(cache),
        source_weights=spectra["weights"].tolist(),
        pol_factor=1,
        convention="RING HEALPix maps; gamma2 = pol_factor * HEALPix gamma2",
        sky="kappa_alm.npz; gamma E alm = shear_alm(kappa_alm)",
        theory="GLASS shell ensemble, ncorr=3, spherical spin factor, no pixel window",
        solver_residual=spectra["solver_residual"],
        seconds=time.perf_counter() - start,
    )
    (out / "maps.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def draw_b_mode_alm(cl_bb, seed):
    """Seeded Gaussian B-mode alm with power spectrum ``cl_bb``.

    Painting evaluates the alm at each galaxy, so ``cl_bb`` is the injected
    spectrum with no pixel window.
    This uses a local Generator, not healpy's process-global RNG.
    """
    import healpy as hp

    cl_bb = np.asarray(cl_bb, dtype=float)
    if np.any(~np.isfinite(cl_bb)) or np.any(cl_bb < 0):
        raise ValueError("C_BB must be finite and non-negative")
    lmax = len(cl_bb) - 1
    if lmax < 2 or np.any(cl_bb[:2] != 0):
        raise ValueError("Spin-2 C_BB needs ell 0,1 zero and lmax >= 2")
    rng = np.random.default_rng(seed)
    ell, emm = hp.Alm.getlm(lmax)
    alm = rng.normal(size=len(ell)) + 1j * rng.normal(size=len(ell))
    alm *= np.sqrt(cl_bb[ell] / 2)
    zero = emm == 0
    alm[zero] = alm[zero].real * np.sqrt(2)
    return alm


def b_mode_cl(lmax, amplitude=0.0, path=None):
    """Injected C_BB on ``ell = 0..lmax``, or ``None`` when nothing is injected.

    ``amplitude`` gives ``A exp[-(ell/2000)^2]``; ``path`` names a two-column
    ``ell C_BB`` text file, interpolated linearly and zero outside its range.
    """
    if not (amplitude or path):
        return None
    ell = np.arange(lmax + 1)
    if path:
        supplied = np.loadtxt(path)
        cl = np.interp(ell, supplied[:, 0], supplied[:, 1], left=0, right=0)
    else:
        cl = amplitude * np.exp(-((ell / 2000) ** 2))
    cl[:2] = 0
    return cl


def paint_shear(
    e1,
    e2,
    signal,
    rng,
    *,
    psf=None,
    alpha=0.0,
    additive=(0.0, 0.0),
    b_signal=None,
    shear="gamma",
    pol_factor=-1,
):
    """Paint one block of rows, with all returned components in catalogue convention.

    ``signal=(kappa, gamma1, gamma2)`` and ``b_signal=(b1, b2)`` are the
    HEALPix-convention fields at the rows' positions (``evaluate_alm``).
    The calibrated input |e| is preserved by the uniform spin-2 rotation, one
    ``rng.uniform`` draw per row in row order.
    ``shear='gamma'`` applies gamma directly for pure-E unit-test skies;
    ``shear='reduced'`` applies gamma/(1-kappa). The selected signal is stored
    as ``g1_true/g2_true`` in either case.
    ``e_noise`` is the intrinsic rotated epsilon, not e_mock minus g, because
    ellipticity composition is nonlinear. ``e_injected`` is added afterwards.
    Catalogue e2 is ``pol_factor`` times the HEALPix component.
    """
    kap, gamma1, gamma2 = (np.asarray(v, dtype=float) for v in signal)
    e1, e2 = np.asarray(e1), np.asarray(e2)
    size = len(kap)
    if any(np.shape(v) != (size,) for v in (gamma1, gamma2, e1, e2)):
        raise ValueError("Signal and ellipticity columns must have matching shapes")
    if not (np.all(np.isfinite(e1)) and np.all(np.isfinite(e2))):
        raise ValueError("Galaxy ellipticities must be finite")
    if shear not in ("gamma", "reduced"):
        raise ValueError("shear must be 'gamma' or 'reduced'")
    if pol_factor not in (1, -1):
        raise ValueError("pol_factor must be 1 or -1")
    if shear == "reduced" and np.any(np.abs(1 - kap) < 1e-8):
        raise ValueError("Reduced shear is singular at kappa=1")
    g = gamma1 + 1j * pol_factor * gamma2
    if shear == "reduced":
        g /= 1 - kap
    noise = np.hypot(e1, e2) * np.exp(2j * rng.uniform(0, np.pi, size))
    mock = (g + noise) / (1 + np.conj(g) * noise)
    injection = np.full(size, complex(*additive), dtype=complex)
    if alpha != 0:
        if psf is None:
            raise ValueError("PSF leakage needs two PSF ellipticity columns")
        injection += alpha * (np.asarray(psf[0]) + 1j * np.asarray(psf[1]))
    if b_signal is not None:
        if any(np.shape(v) != (size,) for v in b_signal):
            raise ValueError("B-mode signal and row counts differ")
        injection += b_signal[0] + 1j * pol_factor * np.asarray(b_signal[1])
    mock += injection
    return dict(
        e1=mock.real,
        e2=mock.imag,
        e1_noise=noise.real,
        e2_noise=noise.imag,
        g1_true=g.real,
        g2_true=g.imag,
        kappa_true=kap,
        e1_injected=injection.real,
        e2_injected=injection.imag,
    )


# Mock shapes are written under the shape columns and these calibrated aliases.
ALIASES = ("e1_leak_corrected", "e2_leak_corrected")
# Rows evaluated per synthesis_general call: amortises its ~one-SHT setup.
BLOCK_SIZE = 8_000_000


def default_threads():
    """``OMP_NUM_THREADS`` when set, else the CPUs this process may run on."""
    import os

    if os.environ.get("OMP_NUM_THREADS"):
        return int(os.environ["OMP_NUM_THREADS"])
    return len(os.sched_getaffinity(0))


def paint_catalogue(
    input_path,
    output_path,
    maps_dir,
    seed,
    *,
    ra_col="RA",
    dec_col="Dec",
    e1_col="e1",
    e2_col="e2",
    w_col="w_des",
    psf_cols=("e1_PSF", "e2_PSF"),
    alpha=0.0,
    additive=(0.0, 0.0),
    b_cl=None,
    b_seed=None,
    chunk_size=250_000,
    block_size=BLOCK_SIZE,
    nthreads=None,
    epsilon=1e-10,
    column_map=None,
    overwrite=False,
    shear="gamma",
    pol_factor=-1,
    keep=(),
):
    """Stream a FITS/HDF5 catalogue into an HDF5 mock (root ``data`` table).

    Uses the same lazy ``Catalogue`` reader as ``read_catalogue`` with bounded
    row slices, including single-dataset 138M-row HDF5 catalogues.
    The sky is the convergence alm in ``maps_dir`` (``kappa_alm.npz``). Its
    convergence, shear E mode (``shear_alm``) and any injected B mode are
    evaluated at each galaxy position (``evaluate_alm``), so the sky's
    ``realization_*`` spectra and the B alm's power are what the catalogue
    carries (see ``generate_lensing_maps``).
    Rotation angles come from ``SeedSequence(seed, spawn_key=(1,))``, a stream
    independent of the sky's ``default_rng(seed)``, one uniform per row in row
    order, so a seed gives the same noise whatever the chunk and block sizes.
    Rows are read in ``chunk_size`` slices of the needed columns and evaluated
    in blocks of ``block_size`` rows on ``nthreads`` threads (default
    ``default_threads()``): each evaluation pays a fixed setup of about one
    full-sky synthesis (~4 s at lmax 8192 on 28 threads) and almost nothing
    per row; memory grows by a few hundred bytes per block row on top of an
    lmax-dependent constant (~7 GB at lmax 8192).
    The mock carries an allowlist of input columns, in the reader's canonical
    grammar: position, weight, PSF ellipticities and any ``keep`` names. Every
    other column is dropped. Mock registry entries run with ``blind: none``,
    so a measured shape passed through (``e*_uncal``, say) would reach
    cosmo_val unblinded; the allowlist is what prevents that, and it gives the
    mock a fixed schema. ``e1_col/e2_col`` and ``e*_leak_corrected`` hold the
    mock shapes. Catalogue e2 is ``pol_factor`` times HEALPix gamma2 (``-1``
    for UNIONS). Output is atomic; the HDF5 attributes and a JSON sidecar
    record the injection and ``pol_factor``.
    """
    import json
    import time
    from pathlib import Path

    import h5py
    import healpy as hp

    from sp_validation.grammar import materialise
    from sp_validation.io import Catalogue

    start = time.perf_counter()
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    if Path(input_path).resolve() == output.resolve():
        raise ValueError("Cannot overwrite the input catalogue")
    if chunk_size <= 0 or block_size <= 0:
        raise ValueError("chunk_size and block_size must be positive")
    nthreads = default_threads() if nthreads is None else int(nthreads)
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(maps_dir)
    kappa_alm = read_sky(root)
    sky = dict(
        kappa=kappa_alm[None],
        gamma=np.stack([shear_alm(kappa_alm), np.zeros_like(kappa_alm)]),
    )
    if b_cl is not None:
        if b_seed is None:
            raise ValueError("B-mode injection requires an explicit b_seed")
        b_alm = draw_b_mode_alm(b_cl, b_seed)
        sky["b"] = np.stack([np.zeros_like(b_alm), b_alm])
        b_spectra_path = output.with_suffix(output.suffix + ".bmode_spectra.npz")
        np.savez(
            b_spectra_path,
            ell=np.arange(len(b_cl)),
            target_bb=b_cl,
            realization_bb=hp.alm2cl(b_alm),
            seed=b_seed,
        )
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))
    temporary = output.with_suffix(output.suffix + ".partial")
    extra = (
        "e1_noise",
        "e2_noise",
        "g1_true",
        "g2_true",
        "kappa_true",
        "e1_injected",
        "e2_injected",
    )
    metadata = dict(
        input=str(input_path),
        maps=str(root),
        painting=f"non-uniform SHT (ducc0 synthesis_general), epsilon={epsilon}",
        rotation_seed=seed,
        alpha=alpha,
        additive=list(additive),
        b_seed=b_seed,
        b_cl=None if b_cl is None else np.asarray(b_cl).tolist(),
        b_realization_spectra=None if b_cl is None else str(b_spectra_path),
        columns=dict(
            ra=ra_col, dec=dec_col, e1=e1_col, e2=e2_col, w=w_col, psf=list(psf_cols)
        ),
        kept=list(keep),
        pol_factor=pol_factor,
        convention="catalogue e2 = pol_factor * HEALPix gamma2",
        shear=shear,
        shape_composition="(g+e_noise)/(1+conj(g)*e_noise)+e_injected",
    )
    with (
        Catalogue(input_path, column_map=column_map) as cat,
        h5py.File(temporary, "w") as dst,
    ):
        dtype = cat.dtype()
        required = [ra_col, dec_col, e1_col, e2_col, w_col]
        if alpha != 0:
            required += list(psf_cols)
        missing = set(required) - set(dtype.names)
        if missing:
            raise KeyError(f"Missing painting columns: {sorted(missing)}")
        if set(extra) & set(dtype.names):
            raise ValueError("Input already contains mock truth/noise columns")
        kept = [ra_col, dec_col, w_col, *keep]
        kept += [name for name in psf_cols if name in dtype.names]
        kept = [name for name in dtype.names if name in set(kept)]
        if missing := set(keep) - set(dtype.names):
            raise KeyError(f"Missing kept columns: {sorted(missing)}")
        needed = [name for name in dtype.names if name in {*kept, *required}]
        shapes = [e1_col, e2_col, *ALIASES]
        output_dtype = np.dtype(
            [(name, dtype[name]) for name in kept]
            + [(name, dtype[e1_col]) for name in shapes]
            + [(name, "f4") for name in extra]
        )
        table = dst.create_dataset(
            "data",
            shape=(len(cat),),
            dtype=output_dtype,
            chunks=(min(chunk_size, max(1, len(cat))),),
        )
        offset = 0

        def paint_block(rows):
            nonlocal offset
            fields = {
                key: evaluate_alm(
                    alm, rows[ra_col], rows[dec_col], epsilon=epsilon, nthreads=nthreads
                )
                for key, alm in sky.items()
            }
            painted = paint_shear(
                rows[e1_col],
                rows[e2_col],
                (fields["kappa"][0], *fields["gamma"]),
                rng,
                psf=(rows[psf_cols[0]], rows[psf_cols[1]]) if alpha != 0 else None,
                alpha=alpha,
                additive=additive,
                b_signal=fields.get("b"),
                shear=shear,
                pol_factor=pol_factor,
            )
            del fields
            mock = np.empty(len(rows), dtype=output_dtype)
            for name in kept:
                mock[name] = rows[name]
            for name, component in zip(shapes, ("e1", "e2") * 2, strict=True):
                mock[name] = painted[component]
            for name in extra:
                mock[name] = painted[name]
            del painted
            table[offset : offset + len(rows)] = mock
            offset += len(rows)
            print(f"Painted {offset}/{len(cat)} rows", flush=True)

        pending, size = [], 0
        for part in cat.chunks.values():
            for lo in range(0, len(part), chunk_size):
                pending.append(materialise(part[lo : lo + chunk_size], needed))
                size += len(pending[-1])
                if size >= block_size:
                    paint_block(np.concatenate(pending))
                    pending, size = [], 0
        if size:
            paint_block(np.concatenate(pending))
        metadata.update(rows=offset, seconds=time.perf_counter() - start)
        dst.attrs["mock_metadata"] = json.dumps(metadata)
        dst.attrs["is_mock"] = True
        dst.attrs["pol_factor"] = pol_factor
    temporary.replace(output)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2)
    )
    return metadata
