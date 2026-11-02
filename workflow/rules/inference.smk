# SACC → likelihood FITS → run INI → MPI chain → weighted summary.
# Each named run owns its inputs and priors; no catalogue is remeasured here.
INFERENCE = config.get("inference", {})
INFERENCE_RUNS = INFERENCE.get("runs", {})
INFERENCE_OUT = Path(INFERENCE.get("output_dir", "results/inference")).resolve()
INFERENCE_PATTERN = str(INFERENCE_OUT / "{run}")


def inference_run(w):
    return INFERENCE_RUNS[w.run]


rule inference_fiducial:
    input:
        expand(INFERENCE_PATTERN + "/summary.json", run=list(INFERENCE_RUNS))


rule inference_prep:
    input:
        sacc=lambda w: inference_run(w)["sacc"],
        covariance=lambda w: [inference_run(w)["covariance"]] if inference_run(w).get("covariance") else [],
        rho=lambda w: [inference_run(w)["rho_stats"]] if inference_run(w).get("rho_stats") else [],
        tau=lambda w: [inference_run(w)["tau_stats"]] if inference_run(w).get("tau_stats") else [],
    output:
        fits=INFERENCE_PATTERN + "/data.fits",
        provenance=INFERENCE_PATTERN + "/input.json",
    params:
        run=inference_run,
    threads: 1
    resources:
        mem_mb=8000,
        runtime=10,
    script:
        "../scripts/inference_prepare.py"


rule inference_config:
    input:
        fits=rules.inference_prep.output.fits,
        template=lambda w: inference_run(w)["template"],
        values=lambda w: inference_run(w)["values"],
        priors=lambda w: inference_run(w)["priors"],
    output:
        ini=INFERENCE_PATTERN + "/pipeline.ini",
    params:
        run=inference_run,
        chain=lambda w: INFERENCE_PATTERN.format(run=w.run) + "/chain.txt",
    threads: 1
    resources:
        mem_mb=1000,
        runtime=5,
    script:
        "../scripts/inference_config.py"


rule inference_chain:
    input:
        ini=rules.inference_config.output.ini,
        fits=rules.inference_prep.output.fits,
        values=lambda w: inference_run(w)["values"],
        priors=lambda w: inference_run(w)["priors"],
    output:
        chain=INFERENCE_PATTERN + "/chain.txt",
    log:
        INFERENCE_PATTERN + "/cosmosis.log",
    params:
        mpi_args=INFERENCE.get("mpi_args", ""),
    threads: lambda w: inference_run(w).get("threads", 8)
    resources:
        mem_mb=lambda w: inference_run(w).get("mem_mb", 16000),
        runtime=lambda w: inference_run(w).get("runtime", 720),
    shell:
        "eval \"$(python -m cosmosis.configure)\"; "
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 mpiexec {params.mpi_args} -n {threads} "
        "cosmosis --mpi {input.ini:q} > {log:q} 2>&1"


rule inference_summary:
    input:
        chain=rules.inference_chain.output.chain,
    output:
        summary=INFERENCE_PATTERN + "/summary.json",
    threads: 1
    resources:
        mem_mb=2000,
        runtime=5,
    script:
        "../scripts/inference_summary.py"
