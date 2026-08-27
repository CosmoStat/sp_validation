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
override, e.g. the image-sims `SIF`).

A few rules shell out to a host toolchain (CosmoCov, ImageMagick) and keep
`container: None`; each says why in its own docstring.

`OMP_NUM_THREADS` is not set by the profile either: the slurm executor's
`--export=ALL` propagates the driver's env, not a profile flag, so a rule that
needs it pinned sets it itself. Per-rule `mem_mb` / `runtime` stay on the rules.

### Off candide — the default profile

Candide is where the analysis runs, so the candide profile is the one to reach
for. `profiles/default/config.yaml` exists for anywhere else — a laptop, another
cluster — and carries the machine-independent half (container wrapping, rerun
triggers, latency wait) with no SLURM layer:

```bash
snakemake --profile workflow/profiles/default -s workflow/Snakefile \
    <target> --configfile <run config> -j 4
```

Snakemake cannot compose profiles, so both files carry that block (marked
`GENERIC` in each) — change one, change the other. `apptainer-args` is not part
of it: expect to edit the default profile's `--bind` list for your machine.

### Which `sp_validation` a rule imports: the launched checkout

The image is the frozen *dependency stack*; the `sp_validation` that runs is
the one in the checkout you launched from — `common.configure()` puts that
checkout's `src/` on each job's `PYTHONPATH`
(`common.inject_checkout_pythonpath` has the mechanics).

This is the default because the alternative is incoherent: Snakemake's
`script:` directive already runs the checkout's *script files*, so without it a
rule executes new script code against an old `import sp_validation` — the two
halves of one commit, split.

**Caveat:** `rerun-triggers: code` watches rule bodies and `script:` files, not
`src/`. Editing a module under `src/` does not by itself mark outputs stale —
force with `-F` or `--forcerun <rule>`.

To reproduce a run from the image alone, opt out:

```bash
snakemake --profile workflow/profiles/candide --config checkout_pythonpath=false <target>
```

Either way the checkout has to sit under one of the profile's bind mounts to be
visible inside the job.

Most of the time this default is all you need. Reach for a different *image*
only when the dependency stack changed — a new package, a lockfile bump — not
when only `src/` did; "Running an image other than your own" below has the
override.

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

Check for a stray `~/.local/bin/snakemake` (any host-side `pip install --user
snakemake` leaves one): Apptainer passes your `PATH` and mounts your `$HOME` by
default, so it can silently shadow the one `uv tool install` set up. `which
snakemake` should resolve under `uv tool dir`, not `~/.local/bin`.

### The container image — one per person

Everything runs one image, published by CI as a registry tag:

```
docker://ghcr.io/cosmostat/sp_validation:develop
```

**Each person keeps their own copy of it.** There is no shared image directory:
you pull your own file, you refresh it when you want to, and nobody else's
refresh moves the ground under your running jobs. The canonical path is

```
~/.cache/sp_validation/sp_validation.sif
```

and a small CLI, `spv-container`, is what puts it there and tells you about it:

```bash
spv-container pull          # fetch :develop to the canonical path (~1.5 GB / ~15 min)
spv-container status        # what is here, which commit it was built from, how current
spv-container exec <cmd>    # run something inside it, candide binds already applied
```

It ships as a console script with the package, and — being stdlib-only, because
it has to run on the *host* — also works straight from a checkout with nothing
installed:

```bash
python3 src/sp_validation/container.py pull
```

**Do the pull from a compute node**, not the login node: it moves ~1.5 GB and
takes about fifteen minutes. `pull` writes to a temporary name and renames, so a
job either gets the whole old image or the whole new one; jobs already running
hold the old file open and finish against it unharmed.

```bash
salloc -p comp -c 4 --time=01:00:00 --exclude=n17,n09,n36 --no-shell   # note the job id
srun --jobid=<id> spv-container pull
scancel <id>
```

**How the workflow finds it.** One resolution order, shared by the CLI and the
workflow: your **sandbox** if you have built one (below), else your **`.sif`** if
you have pulled one, else the **registry tag** — which Snakemake autopulls into
`.snakemake/singularity` under the working directory. That works, but re-pulls
once per run directory, so `spv-container pull` is the path to prefer. Snakemake
accepts all three forms, a sandbox directory included. The tag itself is written
down once, as
`CONTAINER_URI` in `sp_validation/container.py`, which `workflow/common.py`
re-exports; the image-sims `sif:` config key defaults to `null` and resolves the
same way. Override any of it with `--config container=...` (below), or point
somewhere else entirely with `SPV_CONTAINER`.

At launch the workflow prints one advisory line if your image was built from a
commit behind your checkout. It never fails the run — an older image is normally
fine, since the checkout's `src/` is what rules import (see above). It matters
when the *dependency stack* moved: a new package, a lockfile bump.

#### When you need to install something: the sandbox

The pristine SIF is read-only, which is what you want almost always — it is
exactly the published image, and two people running it run the same thing. But
mid-analysis you sometimes need a package the image does not carry yet, and
rebuilding through CI to find out whether it helps is too slow a loop.

For that, unpack the image into a writable directory once:

```bash
spv-container sandbox                        # ~/.cache/sp_validation/sandbox/
spv-container exec --writable pip install <pkg>
```

Writes into a sandbox persist. It is opt-in — nothing builds one for you — and
once it exists **it takes precedence over the SIF everywhere, workflow jobs
included**, so a package you install this way is available to your Snakemake runs
without any further wiring. Jobs exec it read-only; only `--writable` writes.

The cost is that what you are running is no longer fully described by a revision
label. `spv-container status` says which layer is live and flags that, and the
workflow prints one line at launch when a sandbox is in play — the divergence is
visible, never silent. When you are done exploring, either fold the dependency
into `pyproject.toml` (the real fix) or reset to a clean image:

```bash
spv-container pull                  # refresh the pristine SIF
spv-container sandbox --force       # discard the sandbox, rebuild from it
```

**Where the image comes from.** CI (`.github/workflows/deploy-image.yml`) builds
it on every push, `FROM ghcr.io/cosmostat/shapepipe:im_sims` with `uv sync
--frozen` against `uv.lock`, and publishes to `ghcr.io/cosmostat/sp_validation`
tagged by branch — so `:develop` tracks the tip of `develop`. The package is
public; no credentials are needed. Your pulled file is a *snapshot*: CI
publishing a new image changes nothing until you pull again.

`spv-container status` reads `org.opencontainers.image.revision` — the
sp_validation commit the image was built from — and places it against your
checkout's `HEAD`, naming which layer (sandbox or SIF) it read. The image-sims workflow records the same label in
`m_bias_config.yaml` as `ghcr_revision`, so a result file says which image
produced the number.

#### Running an image other than your own

`container` is a config key read by every entry Snakefile
(`common.resolve_container`), so one flag overrides the default everywhere at
once — a local file, or any CI tag:

```bash
snakemake --profile workflow/profiles/candide --config container=/path/to/my.sif <target>
snakemake --profile workflow/profiles/candide \
    --config container=docker://ghcr.io/cosmostat/sp_validation:my-branch <target>
```

CI tags an image for **every** branch, by sanitized branch name (`/` → `-`), so
the second form is how you test a branch's own stack. Snakemake autopulls a tag
per run directory (~15 minutes — from a compute node); `spv-container pull --tag
<uri>` instead puts it at your canonical path, where it becomes your default.

For the image-sims workflow, set `image_sims: {sif: ...}` in your run config.
Either way the image has to sit under one of the profile's bind mounts to be
visible.

One trap to know: the `script:` directive bind-mounts the host orchestrator's
`snakemake` into the job and *appends* it to `sys.path`, so a `snakemake`
importable inside the image wins the lookup. If `script:` rules start failing
with `ModuleNotFoundError: No module named 'snakemake.iocontainers'` or similar,
an in-image snakemake older than the host's is the first thing to check.

### `snakemake` in `script:` files

Every script run via a rule's `script:` directive uses a bare `snakemake`
name (`snakemake.input[...]`, etc.) with no import — Snakemake injects it as
a module global before the script runs. `from snakemake.script import
snakemake` is IDE-hint-only and raises `ImportError` if actually executed.
