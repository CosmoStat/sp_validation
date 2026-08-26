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
override, e.g. the image-sims `SIF`). One rule is an explicit, documented
exception and keeps `container: None` with an inline host-toolchain call —
`covariance_cosmocov` (a host-compiled binary) — see its docstring in
`workflow/rules/`. `OMP_NUM_THREADS` is not set by the profile either: the
slurm executor's `--export=ALL` propagates the driver's env, not a profile
flag, so a rule that needs it pinned sets it itself. Per-rule `mem_mb` /
`runtime` stay on the rules. Off-cluster, drop `--profile` and add `-j N`. See
the profile's own comments for the full rationale.

### Running your own checkout instead of the image's code

Rules import the `sp_validation` baked into the image. To run a working copy
instead — testing a branch without rebuilding — prepend it to `PYTHONPATH` at
the container boundary. Apptainer forwards `APPTAINERENV_`-prefixed host
variables into the job (this survives the profile's `--cleanenv`, which strips
everything else), so setting it on the `snakemake` invocation reaches every
rule:

```bash
APPTAINERENV_PYTHONPATH=/path/to/your/sp_validation/src \
    snakemake --profile workflow/profiles/candide -s workflow/Snakefile <target>
```

The checkout has to sit under one of the profile's bind mounts to be visible
inside the job. This is a user-side override on purpose: nothing in the
workflow sets it, so a run reproduces from the image alone unless you ask
otherwise.

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

### The container image

Everything runs one image, named once as a registry tag:

```
docker://ghcr.io/cosmostat/sp_validation:develop
```

That tag is what `workflow/Snakefile`, the paper Snakefiles and the image-sims
`sif:` config key declare. Nobody writes a `.sif` path: Snakemake pulls the tag
into the profile's `apptainer-prefix`
(`/n17data/cdaley/containers/snakemake-sif`) on first use and reuses the cached
file forever after. No Snakemake rule needs a file. Host-side callers that do —
the `papers/bmodes/scripts/run_*.sh` drivers, and interactive `apptainer exec` —
get it from `workflow/scripts/container_path.py`, which derives the path from
the same tag (Snakemake names a pulled image `{prefix}/{md5(uri)}.simg`; the
script just recomputes that).

**The first pull is not free.** It happens on the host running `snakemake`,
takes ~15 minutes for ~1.5 GB, and blocks the run — so do the first run of a
new tag from a compute node, not the login node. Every later run finds the
cached file and touches neither cache nor network.

**Where the image comes from.** CI (`.github/workflows/deploy-image.yml`) builds
it on every push, `FROM ghcr.io/cosmostat/shapepipe:im_sims` with `uv sync
--frozen` against `uv.lock`, and publishes to
`ghcr.io/cosmostat/sp_validation` tagged by branch — so `:develop` tracks the
tip of `develop`. The package is public; no credentials are needed. A cached
pull is a *snapshot* of the tag: CI publishing a new image does not change what
your jobs run until someone refreshes.

**Refreshing** — one person does it for everybody. Delete the cached file and
let the next run re-pull it, or pull deliberately:

```bash
# From a compute node (~1.5 GB / ~15 min; never on the login node).
salloc -p comp -c 4 --time=01:00:00 --no-shell     # note the job id
export APPTAINER_CACHEDIR=/n17data/cdaley/containers/.apptainer-cache/cache
export APPTAINER_TMPDIR=/n17data/cdaley/containers/.apptainer-cache/tmp
SIF=$(workflow/scripts/container_path.py)          # prints the cached path
srun --jobid=<id> bash -c "cd \$(dirname $SIF) && \
  apptainer pull --force --name next.simg \
  docker://ghcr.io/cosmostat/sp_validation:develop && \
  mv -f next.simg $SIF"
scancel <id>
```

Pull to `next.simg` and `mv` — never pull straight onto the cached name. `mv`
within one directory is an atomic rename, so a job either gets the whole old
image or the whole new one. Pulling in place would leave the file half-written
for the ~15 minutes the pull takes, and any job starting in that window would
fail. Jobs already running hold the old inode open and finish against it
unharmed.

`snakemake --cleanup-containers` deletes every `*.simg` in the prefix that the
current DAG does not require. With the tag form the cached image *is* required,
so it survives; anything left over from an older tag is what goes.

To check what you have:

```bash
apptainer inspect --labels $(workflow/scripts/container_path.py)
```

`org.opencontainers.image.revision` is the sp_validation commit the image was
built from. The image-sims workflow records it in `m_bias_config.yaml` as
`ghcr_revision`, so a result file says which image produced the number.

**Interactive use** — the same image, resolved the same way:

```bash
apptainer exec --bind /home,/scratch,/automnt,/n17data,/n23data1,/n09data \
  $(workflow/scripts/container_path.py) <command>
```

This is what Cail's `app` shell function points at (the function lives in his
shell config, not this repo). There is one image, not two — a refresh moves
interactive use and the workflow together, with nothing to keep in sync.

**Running your own image** instead of the shared one:

```bash
snakemake --profile workflow/profiles/candide --config container=/path/to/my.sif <target>
```

For the image-sims workflow, set `image_sims: {sif: /path/to/my.sif}` in your run
config — it is already a config key. Your image has to sit under one of the
profile's bind mounts to be visible.

One invariant survives from the old hand-built sandbox and still applies: the
`script:` directive bind-mounts the host orchestrator's `snakemake` into the job
and *appends* it to `sys.path`, so a `snakemake` importable inside the image
wins the lookup. If `script:` rules start failing with `ModuleNotFoundError: No
module named 'snakemake.iocontainers'` or similar, an in-image snakemake older
than the host's is the first thing to check.

### `snakemake` in `script:` files

Every script run via a rule's `script:` directive uses a bare `snakemake`
name (`snakemake.input[...]`, etc.) with no import — Snakemake injects it as
a module global before the script runs. `from snakemake.script import
snakemake` is IDE-hint-only and raises `ImportError` if actually executed.
