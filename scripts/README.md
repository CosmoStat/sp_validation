# scripts/ — reduction and runner scripts

Real, standalone scripts for data reduction and for driving the pipeline:
catalogue builders, masking, survey statistics, glass-mock runners, and the
like. Run directly from the command line rather than imported as a library.

What belongs here:

- **Reduction scripts** that turn raw or intermediate products into the inputs
  an analysis needs.
- **Runner scripts** that invoke library code or external tools for a concrete
  job.

What does *not* belong here:

- **Reusable functions** belong in `src/sp_validation/`; a script here should
  import them, not redefine them.
- **Orchestrated, modular analysis** belongs in `workflow/` as Snakemake rules.
- **One-off experiments** belong in `scratch/<you>/`.
