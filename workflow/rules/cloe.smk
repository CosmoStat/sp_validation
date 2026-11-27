# The harmonic 3x2pt forecast, GLASS 3x2pt mocks and chains on the CLOE JAX
# likelihood (sp_validation.cloe, sp_validation.mocks.glass_3x2pt).
#
# config["cloe"]: `recipe` (the survey, config/cloe/*.yaml), `output_dir`,
# `chains` ({name: {cut, families, mock, n_eff, seed}}) and `fishers`
# ([[cut, families], ...]). Each rule is one step of the library's CLI.

CLOE = config["cloe"]
CLOE_OUT = CLOE["output_dir"]
CLOE_RECIPE = str(REPO_ROOT / CLOE["recipe"])
FORECAST = "python -m sp_validation.cloe.forecast"
GLASS3 = "python -m sp_validation.mocks.glass_3x2pt"


def cloe_chain(name):
    return {"families": ["GG", "GL", "LL"], "n_eff": 4000, "seed": 1, **CLOE["chains"][name]}


def cloe_chain_sacc(wc):
    mock = cloe_chain(wc.name).get("mock")
    if mock is None:
        return f"{CLOE_OUT}/forecast.sacc"
    return f"{CLOE_OUT}/glass/seed_{int(mock):04d}.sacc"


wildcard_constraints:
    seed=r"\d{4}",


rule cloe_masks:
    """Source and lens coverage-fraction maps at the recipe's nside."""
    input:
        CLOE_RECIPE,
    output:
        f"{CLOE_OUT}/masks.npz",
    resources:
        mem_mb=16000,
        runtime=20,
    shell:
        "{FORECAST} masks {input} {CLOE_OUT}"


rule cloe_spectra:
    """The CLOE JAX theory's C_ell of every pair at the input cosmology."""
    input:
        CLOE_RECIPE,
    output:
        f"{CLOE_OUT}/spectra.npz",
    threads: 16
    resources:
        mem_mb=32000,
        runtime=20,
    shell:
        "{FORECAST} spectra {input} {CLOE_OUT}"


rule cloe_covariance:
    """Windows and Gaussian (iNKA) covariance of the coupled bandpowers;
    NaMaster couplings are cached in couplings/. Minutes on 64 cores."""
    input:
        CLOE_RECIPE,
        f"{CLOE_OUT}/masks.npz",
        f"{CLOE_OUT}/spectra.npz",
    output:
        f"{CLOE_OUT}/gaussian.npz",
    threads: 64
    resources:
        mem_mb=200000,
        runtime=60,
    shell:
        "{FORECAST} covariance {CLOE_RECIPE} {CLOE_OUT} --processes 32"


rule cloe_ssc:
    """Super-sample covariance (pyccl); single-threaded workers, since they
    fork after pyccl has run. Under a minute on 64 cores."""
    input:
        CLOE_RECIPE,
        f"{CLOE_OUT}/masks.npz",
        f"{CLOE_OUT}/gaussian.npz",
    output:
        f"{CLOE_OUT}/ssc.npz",
    threads: 64
    resources:
        mem_mb=100000,
        runtime=30,
    shell:
        "OMP_NUM_THREADS=1 {FORECAST} ssc {CLOE_RECIPE} {CLOE_OUT} --processes {threads}"


rule cloe_sacc:
    """The noiseless data vector, windows and covariance as a SACC."""
    input:
        CLOE_RECIPE,
        f"{CLOE_OUT}/spectra.npz",
        f"{CLOE_OUT}/gaussian.npz",
        f"{CLOE_OUT}/ssc.npz",
    output:
        f"{CLOE_OUT}/forecast.sacc",
    resources:
        mem_mb=16000,
        runtime=10,
    shell:
        "{FORECAST} sacc {CLOE_RECIPE} {CLOE_OUT}"


rule cloe_fisher:
    """Fisher widths of sigma8, S8, Omega_m under a cut, for some families."""
    input:
        CLOE_RECIPE,
        f"{CLOE_OUT}/forecast.sacc",
    output:
        f"{CLOE_OUT}/fisher_{{cut}}_{{families}}.json",
    threads: 16
    resources:
        mem_mb=32000,
        runtime=20,
    params:
        families=lambda wc: ",".join(wc.families[i : i + 2] for i in range(0, len(wc.families), 2)),
    shell:
        "{FORECAST} fisher {CLOE_RECIPE} {CLOE_OUT} {wildcards.cut} --families {params.families}"


rule cloe_chain:
    """A nautilus chain through cloerun on the forecast or a GLASS mock's SACC:
    9-10 min for 3x2pt, 3 for shear alone, on 64 cores (n_eff 4000)."""
    input:
        CLOE_RECIPE,
        cloe_chain_sacc,
    output:
        f"{CLOE_OUT}/chains/{{name}}.npz",
        f"{CLOE_OUT}/chains/{{name}}.json",
    threads: 64
    resources:
        mem_mb=200000,
        runtime=120,
    params:
        cut=lambda wc: cloe_chain(wc.name)["cut"],
        families=lambda wc: ",".join(cloe_chain(wc.name)["families"]),
        n_eff=lambda wc: cloe_chain(wc.name)["n_eff"],
        seed=lambda wc: cloe_chain(wc.name)["seed"],
    shell:
        "OMP_NUM_THREADS=1 {FORECAST} chain {CLOE_RECIPE} {CLOE_OUT} {params.cut} "
        "{wildcards.name} --sacc {input[1]} --families {params.families} "
        "--n-eff {params.n_eff} --seed {params.seed} --cores {threads}"


rule glass3x2_shells:
    """CAMB spectra of the mock's 101 matter shells, rescaled to total matter,
    and the shells' mixing into every field."""
    input:
        CLOE_RECIPE,
        f"{CLOE_OUT}/gaussian.npz",
    output:
        f"{CLOE_OUT}/glass/shells.npz",
    threads: 16
    resources:
        mem_mb=64000,
        runtime=60,
    shell:
        "{GLASS3} shells {CLOE_RECIPE} {CLOE_OUT}"


rule glass3x2_gls:
    """Lognormal shells' Gaussian spectra (clipped where not positive definite)
    and the mock's expectation, the bands of the spectra they realise."""
    input:
        CLOE_RECIPE,
        f"{CLOE_OUT}/glass/shells.npz",
    output:
        f"{CLOE_OUT}/glass/gls.npz",
    threads: 96
    resources:
        mem_mb=200000,
        runtime=60,
    shell:
        "OMP_NUM_THREADS=1 {GLASS3} gls {CLOE_RECIPE} {CLOE_OUT} --processes {threads}"


rule glass3x2_mock:
    """One GLASS 3x2pt mock, measured: about 4 min on 96 cores."""
    input:
        CLOE_RECIPE,
        f"{CLOE_OUT}/glass/gls.npz",
        f"{CLOE_OUT}/masks.npz",
        f"{CLOE_OUT}/forecast.sacc",
    output:
        f"{CLOE_OUT}/glass/seed_{{seed}}.sacc",
        f"{CLOE_OUT}/glass/seed_{{seed}}.npz",
    threads: 96
    resources:
        mem_mb=200000,
        runtime=60,
    shell:
        "{GLASS3} mock {CLOE_RECIPE} {CLOE_OUT} {wildcards.seed}"


rule cloe_all:
    """Every chain and Fisher forecast the config names."""
    input:
        expand(f"{CLOE_OUT}/chains/{{name}}.json", name=CLOE.get("chains", {})),
        [
            f"{CLOE_OUT}/fisher_{c}_{''.join(sorted(f))}.json"
            for c, f in CLOE.get("fishers", [])
        ],
