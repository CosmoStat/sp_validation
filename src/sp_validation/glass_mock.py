"""GLASS mock generation core for UNIONS weak-lensing simulations.

The reproducibility surface of the UNIONS GLASS mocks lives here: the fixed
cosmology (Planck18 + HMCode AGN feedback, ``sigma8``-rescaled CAMB power
spectrum) and the lognormal matter / lensing-map generation that turns a seed
into a sky. Galaxy sampling and catalogue I/O are a thin layer downstream
(``scripts/glass_mock/make_unions_glass_sim.py``), so that the load-bearing
config can be characterized in isolation — see
``tests/test_glass_mock.py``.

Catalogue-painted mocks use ``generate_lensing_maps`` followed by
``paint_catalogue`` (CLI: ``scripts/glass_mock/paint_catalogue.py``).
The maps average the source-plane convergence over one n(z); all galaxies
sample that projected field, not individual source redshifts.
Map gamma2 follows HEALPix, while catalogue e2/g2 have the opposite sign
so that cosmo_val's ``pol_factor=-1`` restores the HEALPix spin-2 field.
The catalogue stage preserves row order, positions, weights, and auxiliary
columns, randomizes each calibrated ellipticity's orientation, and stores
the noise, selected shear signal, and injected terms separately.
Painting defaults to ``shear='gamma'`` for exact pure-E unit-test answers;
``shear='reduced'`` applies gamma/(1-kappa).

This module also collects the reusable two-point and PSF-leakage helpers the
mock post-processing runners in ``scripts/glass_mock/`` share (configuration-
space ξ±, NaMaster pseudo-Cℓ, harmonic rho/tau, footprint masks).

The ``glass`` dependency is imported lazily inside the map-generation functions
so that this module — and therefore the import guard — resolves in any
environment with CAMB, even one without GLASS installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

__all__ = [
    "GlassMockConfig",
    "build_camb_params",
    "camb_sigma8",
    "build_shells",
    "matter_shell_cls",
    "generate_matter_maps",
    "source_shell_weights",
    "projected_convergence_weights",
    "projected_theory",
    "prepare_lensing_spectra",
    "convergence_to_shear",
    "generate_lensing_maps",
    "draw_b_mode_maps",
    "paint_shear",
    "paint_catalogue",
    "downgrade_mask",
    "growth_factor",
    "ia_convergence",
    "create_mask_from_catalogue",
    "compute_two_point_xi",
    "compute_two_point_cl",
    "compute_leakage_harmony",
    "TREECORR_CONFIG",
]


@dataclass(frozen=True)
class GlassMockConfig:
    """Fixed configuration for a UNIONS GLASS mock realization.

    Every field here is a value that must not silently drift: the cosmology
    (a Planck18 base with explicit ``ns``/``sigma8``/AGN-feedback choices),
    the shell construction (``dx``, ``zmax``), and the resolution/seed. The
    characterization test pins these against a committed reference.

    Cosmology defaults reproduce the fiducial UNIONS v1.4.6.3 mock. The
    ``Planck18`` matter/baryon densities are read at construction time so the
    dataclass carries concrete numbers (not an astropy object) and round-trips
    cleanly to a reference.
    """

    # --- resolution / sampling ---
    nside: int = 32
    harmonic_lmax: int | None = None
    seed: int = 42
    # --- shell construction ---
    dx: float = 200.0  # comoving shell thickness [Mpc]
    zmax: float = 3.0
    # --- cosmology (Planck18 base, UNIONS fiducial overrides) ---
    h: float = field(default=0.6766)
    Om: float = field(default=0.30966)
    Ob: float = field(default=0.04897)
    ns: float = 0.9665
    sigma8: float = 0.8102
    mnu: float = 0.06
    log_T_AGN: float = 7.8
    As_init: float = 2.1e-9  # seed As before sigma8 rescaling
    kmax: float = 20.0
    # --- galaxy population (downstream of map generation) ---
    n_arcmin2: float | list = 6.0905
    sigma_e: float | list = 0.2684
    bias: float | list = 1.2  # constant linear galaxy bias b(z)
    phz_sigma_0: float = 0.03
    nbins: int = 1  # number of tomographic bins
    ia_bias: float | None = None
    # --- Runtime options ---
    limber: bool = False
    mask_path: str | None = None
    nz_path: str | None = None
    output_path: str | None = None
    output_prefix: str | None = None

    @classmethod
    def from_planck18(cls, **overrides) -> "GlassMockConfig":
        """Build a config with matter/baryon densities pulled from Planck18.

        Keeps the dataclass defaults and the production script in sync with the
        single source of truth (``astropy.cosmology.Planck18``) without making
        the dataclass depend on astropy at import time.
        """
        from astropy.cosmology import Planck18

        base = dict(
            h=Planck18.H0.value / 100,
            Om=Planck18.Om0,
            Ob=Planck18.Ob0,
        )
        base.update(overrides)
        return cls(**base)

    @classmethod
    def from_yaml(cls, yaml_config, seed=42) -> "GlassMockConfig":
        """
        Build a config for the GLASS mock generation from a YAML configuration file.

        Reads the dataclass fields from the YAML file. If absent, the field is kept at its default value.
        """
        import yaml

        with open(yaml_config, "r") as f:
            data = yaml.safe_load(f)

        overrides = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**{**overrides, "seed": seed})

    @property
    def lmax(self) -> int:
        """Harmonic ceiling; defaults to ``nside`` for the sampled mocks."""
        return self.nside if self.harmonic_lmax is None else self.harmonic_lmax

    @property
    def Oc(self) -> float:
        """Cold dark matter density (CDM = matter - baryons), pre-neutrino."""
        return self.Om - self.Ob

    def check_consistency(self):
        """Check that the configuration in terms of bins is internally consistent"""
        redshift_distributions = np.loadtxt(self.nz_path)
        if redshift_distributions.shape[1] - 1 != self.nbins:
            raise ValueError(
                f"The number of tomographic bins is inconsistent with the redshift distribution: {redshift_distributions.ndim - 1}."
            )

        if isinstance(self.n_arcmin2, list) and len(self.n_arcmin2) != self.nbins:
            raise ValueError(
                f"Number of elements in n_arcmin2 {len(self.n_arcmin2)} "
                f"does not match number of bins {self.nbins}"
            )

        if isinstance(self.sigma_e, list) and len(self.sigma_e) != self.nbins:
            raise ValueError(
                f"Number of elements in sigma_e {len(self.sigma_e)} "
                f"does not match number of bins {self.nbins}"
            )

        if isinstance(self.bias, list) and len(self.bias) != self.nbins:
            raise ValueError(
                f"Number of elements in bias {len(self.bias)} "
                f"does not match number of bins {self.nbins}"
            )

        print("The number of tomographic bins is consistent across inputs...")


def build_camb_params(config: GlassMockConfig):
    """Build the CAMB parameters for a mock, with ``As`` rescaled to ``sigma8``.

    This is the cosmological heart of the mock and is GLASS-free: it reproduces
    the exact CAMB setup of the production script — Planck18 densities, HMCode
    ``mead2020_feedback`` non-linear model with ``HMCode_logT_AGN``, neutrino
    mass subtracted from CDM, and an iterative ``As`` rescaling so the realized
    ``sigma8(z=0)`` matches ``config.sigma8``.

    Returns the CAMB ``CAMBparams`` object; ``camb_sigma8`` reads the realized
    value back out.
    """
    import camb

    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=100 * config.h,
        omch2=config.Oc * config.h**2,
        ombh2=config.Ob * config.h**2,
        mnu=config.mnu,
    )
    # Neutrino density steals from CDM; re-set with the corrected omch2.
    Oc = config.Oc - pars.omeganu
    pars.set_cosmology(
        H0=100 * config.h,
        omch2=Oc * config.h**2,
        ombh2=config.Ob * config.h**2,
        mnu=config.mnu,
    )

    pars.InitPower.set_params(As=config.As_init, ns=config.ns)
    pars.WantTransfer = True
    pars.NonLinear = camb.model.NonLinear_both
    pars.NonLinearModel.set_params(
        halofit_version="mead2020_feedback",
        HMCode_logT_AGN=config.log_T_AGN,
    )
    pars.set_matter_power(nonlinear=True, kmax=config.kmax)

    # Rescale As ∝ sigma8^2 so the realized sigma8(z=0) hits the target.
    sigma8_temp = camb.get_results(pars).get_sigma8_0()
    As_scaled = config.As_init * (config.sigma8 / sigma8_temp) ** 2
    init_power = camb.InitialPowerLaw()
    init_power.set_params(As=As_scaled, ns=config.ns)
    pars.InitPower = init_power

    realized = camb_sigma8(pars)
    assert np.isclose(config.sigma8, realized), (
        f"As rescaling missed target sigma8: wanted {config.sigma8}, got {realized}"
    )
    return pars


def camb_sigma8(pars) -> float:
    """Realized ``sigma8(z=0)`` for a CAMB parameter set."""
    import camb

    return float(camb.get_results(pars).get_sigma8_0())


def build_shells(config: GlassMockConfig, pars):
    """Linear-window matter shells on a comoving distance grid.

    Returns the GLASS shell list (``(z, w, zeff)`` windows). Lazily imports
    GLASS; only callable where GLASS is installed.
    """
    import camb
    import glass
    from cosmology.compat.camb import Cosmology

    results = camb.get_background(pars)
    cosmo = Cosmology(results)
    zb = glass.distance_grid(cosmo, 0.0, config.zmax, dx=config.dx)
    return glass.linear_windows(zb)


def matter_shell_cls(config: GlassMockConfig, pars, shells, limber=False):
    """Matter angular power spectra for the shells, from CAMB via GLASS."""
    import glass.ext.camb

    return glass.ext.camb.matter_cls(pars, config.lmax, shells, limber=limber)


def generate_matter_maps(config: GlassMockConfig, pars, shells, cls):
    """Yield ``(delta, kappa, gamma1, gamma2)`` HEALPix maps, shell by shell.

    This is the deterministic core of the mock at the map level: a seeded RNG
    drives lognormal matter fields (``ncorr=3``), accumulated into convergence
    via ``MultiPlaneConvergence`` and shear via ``shear_from_convergence``. For
    a fixed config + seed the emitted maps are bit-for-bit reproducible within
    a GLASS/NumPy version, which is what the self-consistency test asserts.

    The galaxy sampling that consumes these maps lives in the production script;
    keeping it out of this generator is what makes the map level testable.
    """
    import glass

    rng = np.random.default_rng(config.seed)
    fields = glass.lognormal_fields(shells)
    gls = glass.solve_gaussian_spectra(fields, cls)
    matter = glass.generate(fields, gls, config.nside, ncorr=3, rng=rng)
    convergence = glass.MultiPlaneConvergence(Cosmology_from_camb(pars))

    for i, delta_i in enumerate(matter):
        convergence.add_window(delta_i, shells[i])
        kappa_i = convergence.kappa
        gamma1_i, gamma2_i = glass.shear_from_convergence(kappa_i)
        yield delta_i, kappa_i, gamma1_i, gamma2_i


def Cosmology_from_camb(pars):
    """Thin indirection so the convergence cosmology is built once, lazily."""
    import camb
    from cosmology.compat.camb import Cosmology

    results = camb.get_background(pars)

    return Cosmology(results)


# --- projected maps and catalogue painting ----------------------------------


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


def projected_theory(cls, transfer, nside, lmax, ncorr=3):
    """Exact shell-ensemble spectra, before and after HEALPix pixel windows.

    The source partition and the finite ``ncorr`` covariance are identical to
    the map generator's. The spin-2 / convergence conversion is spherical,
    including its low-ell factor (not the flat-sky approximation).
    """
    import glass
    import healpy as hp

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
    pw0, pw2 = hp.pixwin(nside, pol=True, lmax=lmax)
    return dict(
        ell=ell.astype(int),
        cl_kk=kk,
        cl_ee=kk * factor,
        cl_kk_pixel=kk * pw0**2,
        cl_ee_pixel=kk * factor * pw2**2,
    )


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


def prepare_lensing_spectra(config, cache_dir, workers=1):
    """Cache seed-independent matter/Gaussian spectra and source theory.

    A lock serializes writers. The key includes cosmology, shell geometry,
    resolution, n(z) contents, GLASS version and the correlation truncation.
    No seed enters the key. Cache files contain numeric arrays only.
    """
    import fcntl
    import hashlib
    import json
    from pathlib import Path

    import glass
    import healpy as hp

    nz_path = Path(config.nz_path)
    physics = {
        key: getattr(config, key)
        for key in (
            "nside",
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
    physics.update(lmax=config.lmax, ncorr=3, glass=glass.__version__, schema=2)
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
            theory = projected_theory(cls, transfer, config.nside, config.lmax)
            discrete = glass.discretized_cls(
                cls, nside=config.nside, lmax=config.lmax, ncorr=3
            )
            gls, achieved, solver_info = solve_shell_spectra(
                glass.lognormal_fields(shells), discrete, workers=workers
            )
            # Solver residuals are recorded rather than hidden behind a warning.
            achieved_theory = projected_theory(
                achieved, transfer, config.nside, config.lmax
            )
            theory["cl_kk_achieved_pixel"] = achieved_theory["cl_kk"]
            pw0, pw2 = hp.pixwin(config.nside, pol=True, lmax=config.lmax)
            theory["cl_ee_achieved_pixel"] = achieved_theory["cl_ee"] * (pw2 / pw0) ** 2
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
    return pars, shells, spectra, path


def convergence_to_shear(kappa, lmax):
    """GLASS spherical shear conversion without network pixel-weight downloads."""
    import healpy as hp

    nside = hp.get_nside(kappa)
    alm = hp.map2alm(kappa, lmax=lmax, pol=False, iter=3)
    ell = np.arange(lmax + 1, dtype=float)
    factor = np.zeros_like(ell)
    factor[2:] = -np.sqrt((ell[2:] + 2) * (ell[2:] - 1) / (ell[2:] * (ell[2:] + 1)))
    pw0, pw2 = hp.pixwin(nside, pol=True, lmax=lmax)
    hp.almxfl(alm, factor * pw2 / pw0, inplace=True)
    return hp.alm2map_spin([alm, np.zeros_like(alm)], nside, 2, lmax)


def generate_lensing_maps(config, output_dir, cache_dir, workers=1):
    """Save projected kappa/gamma float32 maps and full-sky known-answer spectra.

    Matter shells are lognormal, with three neighbouring cross-correlations.
    Source-plane convergence is integrated before the linear shear transform,
    avoiding a spherical transform for every source shell.
    """
    import json
    import time
    from pathlib import Path

    import glass
    import healpy as hp

    start = time.perf_counter()
    if config.lmax > 3 * config.nside - 1 or config.lmax < 2:
        raise ValueError("lmax must be between 2 and 3*nside-1")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pars, shells, spectra, cache = prepare_lensing_spectra(
        config, cache_dir, workers=workers
    )
    prepared = time.perf_counter()
    gls = [spectra[f"gaussian_{i}"] for i in range(int(spectra["nspectra"]))]
    matter = glass.generate(
        glass.lognormal_fields(shells),
        gls,
        config.nside,
        ncorr=3,
        rng=np.random.default_rng(config.seed),
    )
    convergence = glass.MultiPlaneConvergence(Cosmology_from_camb(pars))
    kappa = np.zeros(hp.nside2npix(config.nside))
    for i, delta in enumerate(matter):
        convergence.add_window(delta, shells[i])
        kappa += spectra["weights"][i] * convergence.kappa
        print(f"Matter shell {i + 1}/{len(shells)}", flush=True)
    generated = time.perf_counter()
    gamma1, gamma2 = convergence_to_shear(kappa, config.lmax)
    transformed = time.perf_counter()
    for name, values in (("kappa", kappa), ("gamma1", gamma1), ("gamma2", gamma2)):
        np.save(out / f"{name}.npy", values.astype(np.float32))
    # Measure the persisted precision, not just the float64 precursor.
    del kappa, gamma1, gamma2, matter, convergence, delta
    maps = [
        np.load(out / f"{name}.npy", mmap_mode="r")
        for name in ("kappa", "gamma1", "gamma2")
    ]
    measured = hp.anafast(maps, lmax=config.lmax, pol=True, iter=3)
    theory = {
        key: spectra[key]
        for key in (
            "ell",
            "cl_kk",
            "cl_ee",
            "cl_kk_pixel",
            "cl_ee_pixel",
            "cl_kk_achieved_pixel",
            "cl_ee_achieved_pixel",
        )
    }
    np.savez(
        out / "spectra.npz",
        **theory,
        realization_kk=measured[0],
        realization_ee=measured[1],
        realization_bb=measured[2],
        realization_ke=measured[3],
        realization_eb=measured[4],
        realization_kb=measured[5],
    )
    metadata = dict(
        config=asdict(config),
        lmax=config.lmax,
        spectra_cache=str(cache),
        source_weights=spectra["weights"].tolist(),
        convention="RING HEALPix gamma; catalogue e2=-gamma2",
        reduced_shear="gamma/(1-kappa) of n(z)-projected fields",
        theory="GLASS shell ensemble, ncorr=3, spherical spin factor",
        unconverged_spectra=int(np.sum(spectra["solver_info"] == 0)),
        solver_workers=workers,
        seconds=dict(
            prepare=prepared - start,
            matter=generated - prepared,
            shear=transformed - generated,
            spectra_and_io=time.perf_counter() - transformed,
            total=time.perf_counter() - start,
        ),
    )
    (out / "maps.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def draw_b_mode_maps(nside, cl_bb, seed):
    """Seeded pure-B Gaussian spin-2 field in the HEALPix convention.

    ``cl_bb`` is the pixelized target (no additional pixel window is applied).
    This uses a local Generator, not healpy's process-global RNG.
    """
    import healpy as hp

    cl_bb = np.asarray(cl_bb, dtype=float)
    if np.any(~np.isfinite(cl_bb)) or np.any(cl_bb < 0):
        raise ValueError("C_BB must be finite and non-negative")
    lmax = len(cl_bb) - 1
    if lmax < 2 or lmax > 3 * nside - 1 or np.any(cl_bb[:2] != 0):
        raise ValueError("Spin-2 C_BB needs ell 0,1 zero and lmax <= 3*nside-1")
    rng = np.random.default_rng(seed)
    ell, emm = hp.Alm.getlm(lmax)
    alm = rng.normal(size=len(ell)) + 1j * rng.normal(size=len(ell))
    alm *= np.sqrt(cl_bb[ell] / 2)
    zero = emm == 0
    alm[zero] = alm[zero].real * np.sqrt(2)
    return hp.alm2map_spin([np.zeros_like(alm), alm], nside, 2, lmax)


def paint_shear(
    ra,
    dec,
    e1,
    e2,
    maps,
    rng,
    *,
    psf=None,
    alpha=0.0,
    additive=(0.0, 0.0),
    b_maps=None,
    shear="gamma",
):
    """Paint one row chunk, with all returned components in catalogue convention.

    ``maps=(kappa,gamma1,gamma2)`` and ``b_maps`` are RING HEALPix maps.
    The calibrated input |e| is preserved by the uniform spin-2 rotation.
    ``shear='gamma'`` applies gamma directly for pure-E unit-test skies;
    ``shear='reduced'`` applies gamma/(1-kappa). The selected signal is stored
    as ``g1_true/g2_true`` in either case.
    ``e_noise`` is the intrinsic rotated epsilon, not e_mock minus g, because
    ellipticity composition is nonlinear. ``e_injected`` is added afterwards.
    """
    import healpy as hp

    kappa, gamma1, gamma2 = maps
    nside = hp.npix2nside(len(kappa))
    if any(len(m) != len(kappa) for m in maps):
        raise ValueError("Lensing map sizes differ")
    values = [np.asarray(v) for v in (ra, dec, e1, e2)]
    if any(v.shape != values[0].shape for v in values):
        raise ValueError("Galaxy columns must have matching shapes")
    if any(np.any(~np.isfinite(v)) for v in values):
        raise ValueError("Galaxy positions and ellipticities must be finite")
    if np.any(np.abs(values[1]) > 90):
        raise ValueError("Declination lies outside [-90, 90]")
    pix = hp.ang2pix(nside, np.mod(values[0], 360), values[1], lonlat=True)
    kap = np.asarray(kappa[pix], dtype=float)
    if shear not in ("gamma", "reduced"):
        raise ValueError("shear must be 'gamma' or 'reduced'")
    if shear == "reduced" and np.any(np.abs(1 - kap) < 1e-8):
        raise ValueError("Reduced shear is singular at kappa=1")
    # Conjugating HEALPix gamma gives the catalogue's sky-coordinate convention.
    g = np.asarray(gamma1[pix], dtype=float) - 1j * gamma2[pix]
    if shear == "reduced":
        g /= 1 - kap
    noise = np.hypot(values[2], values[3]) * np.exp(
        2j * rng.uniform(0, np.pi, len(pix))
    )
    mock = (g + noise) / (1 + np.conj(g) * noise)
    injection = np.full(len(pix), complex(*additive), dtype=complex)
    if alpha != 0:
        if psf is None:
            raise ValueError("PSF leakage needs two PSF ellipticity columns")
        injection += alpha * (np.asarray(psf[0]) + 1j * np.asarray(psf[1]))
    if b_maps is not None:
        if any(len(m) != len(kappa) for m in b_maps):
            raise ValueError("B-mode and lensing map sizes differ")
        injection += b_maps[0][pix] - 1j * b_maps[1][pix]
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
    column_map=None,
    overwrite=False,
    shear="gamma",
):
    """Stream a FITS/HDF5 catalogue into an HDF5 mock (root ``data`` table).

    Uses the same lazy ``Catalogue`` reader as ``read_catalogue`` with bounded
    row slices, including single-dataset 138M-row HDF5 catalogues.
    Auxiliary columns are preserved in the reader's canonical grammar.
    Calibrated ``e*_leak_corrected`` aliases, when present, also receive the
    mock shapes, so the corrected-column cosmo_val path stays mock-only.
    Output is atomic; a JSON sidecar and HDF5 attributes record the injection.
    """
    import json
    import time
    from pathlib import Path

    import h5py

    from sp_validation.grammar import materialise
    from sp_validation.io import Catalogue

    start = time.perf_counter()
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    if Path(input_path).resolve() == output.resolve():
        raise ValueError("Cannot overwrite the input catalogue")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(maps_dir)
    maps = tuple(
        np.load(root / f"{key}.npy", mmap_mode="r")
        for key in ("kappa", "gamma1", "gamma2")
    )
    b_maps = None
    if b_cl is not None:
        if b_seed is None:
            raise ValueError("B-mode injection requires an explicit b_seed")
        import healpy as hp

        b_maps = draw_b_mode_maps(hp.npix2nside(len(maps[0])), b_cl, b_seed)
        measured_b = hp.anafast(
            [np.zeros_like(b_maps[0]), *b_maps], lmax=len(b_cl) - 1, pol=True, iter=3
        )
        b_spectra_path = output.with_suffix(output.suffix + ".bmode_spectra.npz")
        np.savez(
            b_spectra_path,
            ell=np.arange(len(b_cl)),
            target_bb=b_cl,
            realization_ee=measured_b[1],
            realization_bb=measured_b[2],
            realization_eb=measured_b[4],
            seed=b_seed,
        )
    rng = np.random.default_rng(seed)
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
        rotation_seed=seed,
        alpha=alpha,
        additive=list(additive),
        b_seed=b_seed,
        b_cl=None if b_cl is None else np.asarray(b_cl).tolist(),
        b_realization_spectra=None if b_cl is None else str(b_spectra_path),
        columns=dict(
            ra=ra_col, dec=dec_col, e1=e1_col, e2=e2_col, w=w_col, psf=list(psf_cols)
        ),
        convention="catalogue e2=-HEALPix gamma2; pol_factor=-1",
        shear=shear,
        applied_signal="gamma" if shear == "gamma" else "gamma/(1-kappa)",
        shape_composition="(g+e_noise)/(1+conj(g)*e_noise)+e_injected",
        calibrated_aliases="e1_leak_corrected/e2_leak_corrected = mock",
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
        output_dtype = np.dtype(
            [(name, dtype[name]) for name in dtype.names]
            + [(name, "f4") for name in extra]
        )
        table = dst.create_dataset(
            "data",
            shape=(len(cat),),
            dtype=output_dtype,
            chunks=(min(chunk_size, max(1, len(cat))),),
        )
        offset = 0
        timings = dict(read=0.0, paint=0.0, assemble=0.0, write=0.0)
        weight_sums = np.zeros(2)
        moments = {kind: np.zeros((2, 2, 2)) for kind in ("noise", "mock")}
        for part in cat.chunks.values():
            for lo in range(0, len(part), chunk_size):
                tick = time.perf_counter()
                rows = materialise(part[lo : lo + chunk_size])
                timings["read"] += time.perf_counter() - tick
                tick = time.perf_counter()
                painted = paint_shear(
                    rows[ra_col],
                    rows[dec_col],
                    rows[e1_col],
                    rows[e2_col],
                    maps,
                    rng,
                    psf=(rows[psf_cols[0]], rows[psf_cols[1]]) if alpha != 0 else None,
                    alpha=alpha,
                    additive=additive,
                    b_maps=b_maps,
                    shear=shear,
                )
                timings["paint"] += time.perf_counter() - tick
                tick = time.perf_counter()
                mock = np.empty(len(rows), dtype=output_dtype)
                for name in dtype.names:
                    mock[name] = rows[name]
                mock[e1_col], mock[e2_col] = painted["e1"], painted["e2"]
                for alias, component in (
                    ("e1_leak_corrected", "e1"),
                    ("e2_leak_corrected", "e2"),
                ):
                    if alias in dtype.names:
                        mock[alias] = painted[component]
                for name in extra:
                    mock[name] = painted[name]
                # One-point diagnostics use only rotated or mock shapes.
                # No unrotated data component or rotation angle is retained.
                weight = np.asarray(rows[w_col], dtype=float)
                for index, power in enumerate((1, 2)):
                    wp = weight**power
                    weight_sums[index] += np.sum(wp)
                    for kind, pair in (
                        ("noise", ("e1_noise", "e2_noise")),
                        ("mock", (e1_col, e2_col)),
                    ):
                        for component, name in enumerate(pair):
                            value = np.asarray(mock[name], dtype=float)
                            moments[kind][index, component] += [
                                np.sum(wp * value),
                                np.sum(wp * value**2),
                            ]
                timings["assemble"] += time.perf_counter() - tick
                tick = time.perf_counter()
                table[offset : offset + len(rows)] = mock
                timings["write"] += time.perf_counter() - tick
                offset += len(rows)
                print(f"Painted {offset}/{len(cat)} rows", flush=True)
        diagnostics = {}
        for kind, accum in moments.items():
            diagnostics[kind] = {}
            for index, power in enumerate((1, 2)):
                first, second = accum[index].T
                norm = weight_sums[index]
                diagnostics[kind][f"weight_power_{power}"] = dict(
                    sigma_per_component=np.sqrt(
                        second / norm - (first / norm) ** 2
                    ).tolist(),
                    isotropic_sigma=float(np.sqrt(0.5 * np.sum(second) / norm)),
                )
        metadata.update(
            rows=offset,
            seconds=time.perf_counter() - start,
            seconds_by_stage=timings,
            weight_sums=dict(sum_w=float(weight_sums[0]), sum_w2=float(weight_sums[1])),
            shape_noise=diagnostics,
        )
        dst.attrs["mock_metadata"] = json.dumps(metadata)
        dst.attrs["is_mock"] = True
    temporary.replace(output)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2)
    )
    return metadata


# --- mask / galaxy-sampling helpers (used by the generation runner) ---------


def downgrade_mask(mask, nside):
    """Downgrade a HEALPix mask to a lower ``nside`` (>=0.75 -> 1, else 0).

    Returns ``mask`` unchanged when it is already at ``nside``.
    """
    import healpy as hp

    nside_mask = hp.get_nside(mask)
    if nside_mask == nside:
        return mask
    print(f"[!] Downgrading mask from Nside {nside_mask} to Nside {nside}.")
    print("[!] Pixels with values <0.75 will be set to 0.0")
    print("[!] Pixels with values >=0.75 will be set to 1.0")
    mask_down = hp.ud_grade(mask, nside)
    mask_down[mask_down < 0.75] = 0.0
    mask_down[mask_down >= 0.75] = 1.0
    print("[!] Done.")
    return mask_down


def _growth_E_sq(z, om0):
    """Flat-LCDM ``E^2(z) = H^2(z)/H0^2``."""
    return om0 * (1 + z) ** 3 + 1 - om0


def _growth_integrand(z, om0):
    """Growth-factor redshift integrand."""
    return (z + 1) / (_growth_E_sq(z, om0)) ** 1.5


def _growth_factor_single(z, om0):
    """Normalised linear growth factor at a single redshift."""
    import scipy as sp

    first = sp.integrate.quad(_growth_integrand, z, np.inf, args=(om0,))[0]
    second = sp.integrate.quad(_growth_integrand, 0, np.inf, args=(om0,))[0]
    return (_growth_E_sq(z, om0) ** 0.5) * first / second


def growth_factor(z, om0):
    """Normalised linear growth factor ``D(z)`` (scalar or array ``z``)."""
    if isinstance(z, (float, int)):
        return _growth_factor_single(z, om0)
    return np.array([_growth_factor_single(zi, om0) for zi in list(z)])


def ia_convergence(delta, shell, config):
    """Intrinsic-alignment contribution to the convergence (NLA model).

    Mirrors the production NLA prefactor: ``-A_ia * rho_c1 * Om0 / D(z_eff)``
    applied to the matter overdensity ``delta`` of one shell.
    """
    import astropy.constants as const
    import astropy.units as u

    Om0 = config.Om
    A_ia = config.ia_bias
    _, _, z_eff = shell
    c1 = 5e-14 * (u.Mpc**3.0) / u.solMass
    c1_cgs = (c1 * (1.0 / config.h) ** 2.0).cgs
    H0 = (100 * config.h * u.km / u.s / u.Mpc).to(u.s**-1)
    G = (const.G * u.m**3 / (u.kg * u.s**2)).cgs
    rho_crit = 3 * H0**2 / (8 * np.pi * G)
    rho_c1 = (c1_cgs * rho_crit).value
    prefactor = -A_ia * rho_c1 * Om0
    return prefactor / growth_factor(z_eff, Om0) * delta


def create_mask_from_catalogue(nside, path, output, ra_col="RA", dec_col="DEC"):
    """Write a HEALPix footprint mask from the populated pixels of a catalogue.

    Pixels containing at least one object get 1.0, everything else 0.0.
    """
    import healpy as hp

    from sp_validation.io import read_catalogue

    cat_gal = read_catalogue(path, columns=[ra_col, dec_col])
    ra = cat_gal[ra_col]
    dec = cat_gal[dec_col]

    theta = (90.0 - dec) * np.pi / 180.0
    phi = ra * np.pi / 180.0
    pix = hp.ang2pix(nside, theta, phi)

    _unique_pix, _idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)
    n_gal = np.zeros(hp.nside2npix(nside))
    n_gal[np.unique(pix)] = np.bincount(idx_rep)
    mask = n_gal != 0

    hp.write_map(output, mask, dtype=np.float32, overwrite=True)


# --- shape noise and number density on mock catalogues ----------------------
def shape_noise(e1, e2, w):
    return np.sqrt(
        0.5
        * (np.sum(e1**2 * w**2) / np.sum(w**2) + np.sum(e2**2 * w**2) / np.sum(w**2))
    )


def number_density(w, area):
    return np.sum(w) ** 2 / np.sum(w**2) / area


def validate_shape_noise(cat, tomo=True, e1_col="e1", e2_col="e2", w_col="w"):
    # First compute and print the non-tomographic shape noise
    e1 = cat[e1_col]
    e2 = cat[e2_col]
    w = cat[w_col]
    sigma_e = shape_noise(e1, e2, w)
    print(f"Non-tomographic shape noise: {sigma_e}")

    if tomo:
        # Now compute and print the tomographic shape noise
        for b in range(1, cat["TOM_BIN_ID"].max() + 1):
            e1_bin = e1[cat["TOM_BIN_ID"] == b]
            e2_bin = e2[cat["TOM_BIN_ID"] == b]
            w_bin = w[cat["TOM_BIN_ID"] == b]
            sigma_e_bin = shape_noise(e1_bin, e2_bin, w_bin)
            print(f"Tomographic shape noise (bin {b}): {sigma_e_bin}")


def validate_number_density(cat, mask, tomo=True, w_col="w"):
    import healpy as hp

    # First compute and print the non-tomographic shape noise
    w = cat[w_col]
    nside = hp.npix2nside(mask.shape[0])
    area = (
        np.sum(mask) * hp.nside2pixarea(nside, degrees=True) * 60 * 60
    )  # area in arcmin^2
    n_gal = number_density(w, area)
    print(f"Non-tomographic number density: {n_gal}")

    if tomo:
        for b in range(1, cat["TOM_BIN_ID"].max() + 1):
            w_bin = w[cat["TOM_BIN_ID"] == b]
            n_gal_bin = number_density(w_bin, area)
            print(f"Tomographic number density (bin {b}): {n_gal_bin}")


# --- two-point statistics on mock catalogues --------------------------------

# Configuration-space treecorr defaults for the GLASS-mock 2PCF.
TREECORR_CONFIG = {
    "ra_units": "degrees",
    "dec_units": "degrees",
    "min_sep": 1.0,
    "max_sep": 250.0,
    "nbins": 20,
    "sep_units": "arcmin",
    "num_threads": 4,
}


def _mock_powspace_bin(nside, lmin, n_bins):
    """Powspace NaMaster binning for the mock harmonic stats, via the primitive.

    Returns ``(bin, ell_eff, lmax, b_lmax)`` — the 4-tuple the mock harmonic
    helpers consume. The binning itself is the shared
    ``pseudo_cl.make_namaster_bin`` powspace scheme (``power=0.5`` reproduces the
    old square-root spacing exactly); ``lmax = 2 * nside`` / ``b_lmax = lmax - 1``
    come from ``pseudo_cl.pseudo_cl_geometry``. Note the mock's ``lmin`` floor is
    a call argument here, whereas the geometry helper fixes ``lmin = 8``; the two
    coincide for the production default, and this wrapper keeps the override.
    """
    from sp_validation.pseudo_cl import make_namaster_bin, pseudo_cl_geometry

    _lmin, lmax, b_lmax = pseudo_cl_geometry(nside)
    b = make_namaster_bin(lmin, lmax, b_lmax, "powspace", n_ell_bins=n_bins, power=0.5)
    return b, b.get_effective_ells(), lmax, b_lmax


def compute_two_point_xi(cat, config=None):
    """Configuration-space shear 2PCF (treecorr ``GGCorrelation``) for a mock."""
    import treecorr

    treecorr_config = TREECORR_CONFIG if config is None else config
    treecorr_cat = treecorr.Catalog(
        ra=cat["ra"],
        dec=cat["dec"],
        g1=cat["e1"],
        g2=cat["e2"],
        ra_units="degrees",
        dec_units="degrees",
    )
    gg = treecorr.GGCorrelation(treecorr_config)
    gg.process(treecorr_cat)
    return gg


def compute_two_point_cl(cat, nside=1024, lmin=8, n_bins=32):
    """Catalogue-based NaMaster pseudo-Cl for a mock shear field.

    Returns ``(cl_coupled, cl_all)``, each with the ell axis prepended.

    Deliberately keeps its own ``NmtFieldCatalog`` construction rather than
    delegating to the ``pseudo_cl.get_pseudo_cls_catalog`` primitive: the mock
    field is UNWEIGHTED (``weights = ones``, no per-object shear weight) and this
    function returns the COUPLED pseudo-Cl alongside the decoupled one (both with
    an ell axis prepended), which the primitive's ``(ell_eff, cl_all, wsp)``
    contract does not expose. The shared piece — the powspace binning — is taken
    from ``_mock_powspace_bin`` so the bandpower drift is gone.
    """
    import pymaster as nmt

    e1 = cat["e1"]
    e2 = cat["e2"]
    ra = cat["ra"]
    dec = cat["dec"]
    del cat

    ra[ra < 0] += 360

    b, ell_eff, lmax, b_lmax = _mock_powspace_bin(nside, lmin, n_bins)

    factor = -1
    f_all = nmt.NmtFieldCatalog(
        positions=[ra, dec],
        weights=np.ones_like(e1),
        field=[e1, factor * e2],
        lmax=b_lmax,
        lmax_mask=b_lmax,
        spin=2,
        lonlat=True,
    )

    wsp = nmt.NmtWorkspace.from_fields(f_all, f_all, b)
    cl_coupled = nmt.compute_coupled_cell(f_all, f_all)
    cl_all = wsp.decouple_cell(cl_coupled)

    # compute_coupled_cell runs over ell = 0..b_lmax, index = ell
    cl_coupled = np.concatenate([np.arange(lmax)[np.newaxis, :], cl_coupled])
    cl_all = np.concatenate([ell_eff[np.newaxis, ...], cl_all])
    return cl_coupled, cl_all


def compute_leakage_harmony(cat, cat_star, nside=1024, lmin=8, n_bins=32):
    """Harmonic-space PSF-leakage rho_0 / tau_0 from galaxy + star catalogues.

    Returns ``(rho_cl, tau_cl)``: the PSF-PSF (rho_0) and galaxy-PSF (tau_0)
    decoupled pseudo-Cl, each with the effective-ell axis prepended.
    """
    import pymaster as nmt

    e1 = cat["e1"]
    e2 = cat["e2"]
    e1_psf = cat_star["HSM_G1_PSF"]
    e2_psf = cat_star["HSM_G2_PSF"]

    ra = cat["ra"]
    dec = cat["dec"]
    ra_star = cat_star["RA"]
    dec_star = cat_star["DEC"]
    del cat

    ra[ra < 0] += 360
    ra_star[ra_star < 0] += 360

    b, ell_eff, _lmax, b_lmax = _mock_powspace_bin(nside, lmin, n_bins)

    factor = -1
    f_psf = nmt.NmtFieldCatalog(
        positions=[ra_star, dec_star],
        weights=np.ones_like(ra_star),
        field=[e1_psf, factor * e2_psf],
        lmax=b_lmax,
        lmax_mask=b_lmax,
        spin=2,
        lonlat=True,
    )
    f_gal = nmt.NmtFieldCatalog(
        positions=[ra, dec],
        weights=np.ones_like(ra),
        field=[e1, factor * e2],
        lmax=b_lmax,
        lmax_mask=b_lmax,
        spin=2,
        lonlat=True,
    )

    # rho_0: PSF-PSF
    wsp = nmt.NmtWorkspace.from_fields(f_psf, f_psf, b)
    rho_cl = wsp.decouple_cell(nmt.compute_coupled_cell(f_psf, f_psf))
    rho_cl = np.concatenate([ell_eff[np.newaxis, ...], rho_cl])

    # tau_0: galaxy-PSF
    wsp = nmt.NmtWorkspace.from_fields(f_gal, f_psf, b)
    tau_cl = wsp.decouple_cell(nmt.compute_coupled_cell(f_gal, f_psf))
    tau_cl = np.concatenate([ell_eff[np.newaxis, ...], tau_cl])

    return rho_cl, tau_cl
