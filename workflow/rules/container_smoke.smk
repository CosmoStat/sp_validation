# Container smoke test -- validates the base container contract every
# composed workflow inherits: profile executor (slurm) + software-deployment
# -method (apptainer) + apptainer-args (binds), no rule-level `container:`
# override and no `apptainer exec` in the shell. See
# workflow/scripts/container_smoke.py for what it actually checks.
#
# Run standalone before trusting the pivot on real compute:
#
#     snakemake --profile workflow/profiles/candide -s workflow/Snakefile \
#         --configfile <run config> container_smoke


rule container_smoke:
    output:
        "results/container_smoke.yaml",
    resources:
        runtime=5,
    script:
        "../scripts/container_smoke.py"
