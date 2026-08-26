# workflow/ — shared analysis

The modular Snakemake workflow that produces every analysis product and
diagnostic plot. Generic and reusable: the rules live here once, organized for
several people to run them side by side without stepping on each other.

What belongs here:

- **Reusable rules and their scripts** — `Snakefile`, `rules/`, `scripts/`,
  and the shared helpers in `common.py`. Anything generic enough that a second
  run or a second person would want it.
- **Everything up to a paper figure's inputs.** The boundary is the inputs to a
  figure: catalogues, correlation functions, covariances, chains, and the
  diagnostic plots that vet them — all analysis, all here. Outputs go to the
  top-level `results/`, never alongside the rules.

What does *not* belong here:

- **Final-figure assembly** (PDF, colour, layout for one paper) lives in
  `papers/<paper>/`.
- **One-off, exploratory work** lives in `scratch/<you>/`; promote it here only
  once it is generic and worth sharing.

Runs stay modular, not monolithic: a paper or run composes these rules with
Snakemake's `module` directive under its own config and an output `prefix`, so
each namespaces cleanly under `results/<name>/`.

## Running on the cluster — the candide profile

`profiles/candide/config.yaml` is the committed SLURM profile: it hands
Snakemake the candide executor, account, partition, node excludes, and per-job
floor, so scheduling is repo state rather than an operator's shell. Drive any
target with one command:

```bash
snakemake --profile workflow/profiles/candide \
    -s workflow/image_sims/Snakefile \
    <target> --configfile <run config>
```

For example, the image-sim m-bias chain end to end (`im_mbias` fans out one
SLURM job per branch × tile, MPI-free):

```bash
snakemake --profile workflow/profiles/candide \
    -s workflow/image_sims/Snakefile \
    im_mbias --configfile my_run.yaml
```

Give the target *before* `--configfile`: `--configfile` takes one-or-more
paths, so a target after it is read as a config file ("No such file:
im_mbias"). Always dry-run first with `-n`.

Every rule runs inside the sp_validation container: the profile sets
`software-deployment-method: apptainer` and `apptainer-args` (the bind mounts),
and Snakemake wraps each job's `shell:`/`script:` command in `apptainer exec`
itself — no rule writes its own `apptainer exec` call. The image name comes
from the `container:` directive in `workflow/Snakefile` (or a rule's own
override, e.g. the image-sims `SIF`). Two rules are
explicit, documented exceptions and keep `container: None` with an inline
`apptainer exec`/host-toolchain call — `xi_highres` (multi-node MPI) and
`covariance_cosmocov` (a host-compiled binary) — see their docstrings in
`workflow/rules/`. `OMP_NUM_THREADS` is not set by the profile either: the
slurm executor's `--export=ALL` propagates the driver's env, not a profile
flag, so a rule that needs it pinned sets it itself. Per-rule `mem_mb` /
`runtime` stay on the rules. Off-cluster, drop `--profile` and add `-j N`. See
the profile's own comments for the full rationale.

### Never write `/automnt/nXXdataN` in a path

Use the plain form `/nXXdataN/...` in every rule, config, and invocation
directory. `/automnt/nXXdataN` works only from a node that does *not* own that
disk. On the owning node the disk is mounted directly at `/nXXdataN` and there
is no `/automnt/nXXdataN` entry at all, so a job that lands there dies about one
second after the allocation starts, before any log file is written. This is why
`n17` is in the profile's exclude list. Every canonical path in `common.py`
already uses the plain form; keep new paths the same.

### Run Snakemake from the host, never from inside the container

`snakemake` is a thin host-side tool, pinned once per machine:

```bash
uv tool install snakemake==9.23.1 --with snakemake-executor-plugin-slurm
```

(match the version to `snakemake` in this repo's `uv.lock`). Run every
`snakemake` command directly on the host — do not `apptainer shell` first.
Snakemake itself never touches the science stack; it only reads rule
definitions and submits jobs. Each job carries its own `apptainer exec`
wrapping from the profile (see above), so the container is where the science
code runs, not where the orchestrator runs — one container per job, never a
nested one.

Driving Snakemake from inside a container shell used to be the recommended
path, and is why an old `~/.local/bin/snakemake` (or any host-side `pip
install --user snakemake`) is worth checking for: Apptainer passes your `PATH`
and mounts your `$HOME` by default, so a leftover host install can silently
shadow the one `uv tool install` just set up. Run `which snakemake` and
confirm it resolves under `uv`'s tool directory (`uv tool dir`), not
`~/.local/bin`.

### Container image invariants (not tracked in this repo)

The `script:` directive works by bind-mounting the host orchestrator's own
`snakemake` install into the job's container and `sys.path.extend`-ing it in
(appended, not prepended) — so anything already importable inside the image
under that name wins the lookup instead. The image at
`/n17data/cdaley/containers/containers` is a writable sandbox (see the
top-level UNIONS `CLAUDE.md`), not built from a tracked recipe, so these two
invariants live only in the image itself and must be re-applied by hand after
any rebuild:

- **No `snakemake` (or `snakemake-executor-plugin-slurm`) pip-installed
  inside the image.** A leftover in-image install — from the old
  apptainer-shell-then-snakemake-inside pattern this profile-driven setup
  retired — shadows the host-mounted orchestrator ahead of it on `sys.path`
  and breaks `script:`'s own unpickling preamble (`ModuleNotFoundError: No
  module named 'snakemake.iocontainers'` if the in-image version predates
  that submodule). Check with `apptainer exec ... python3 -m pip show
  snakemake` — `Required-by:` should list nothing outside the snakemake
  family itself before removing it.
- **`/.singularity.d/env/50-bashrc.sh` must not source the host `~/.bashrc`
  for `apptainer exec`/`run`, only for an interactive `apptainer shell`.**
  Apptainer sources every `/.singularity.d/env/*.sh` for all three actions;
  gate any host-dotfile sourcing on `[ "$APPTAINER_COMMAND" = "shell" ]` (set
  by Apptainer itself before these scripts run). Without the guard, a host
  dotfile that mutates `PATH` (e.g. an `asdf` init) runs on every job too and
  can push host tools — including a host-side `~/.local/bin/python` — ahead
  of the image's own `/usr/local/bin`, so a bare `python` in a rule's
  `shell:`/`script:` silently executes outside the container.

### `snakemake` in `script:` files

Every script run via a rule's `script:` directive uses a bare `snakemake`
name (`snakemake.input[...]`, etc.) with no import — Snakemake injects it as
a module global before the script runs. `from snakemake.script import
snakemake` is IDE-hint-only and raises `ImportError` if actually executed.
