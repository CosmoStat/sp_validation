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

## Blinding

Every entry of `cosmo_val/cat_config.yaml` declares `blind: none` or
`blind: <name>`; an entry without one stops the launch. `_leak_corr` versions
take their entry's blind, and entries reading one shear file declare one blind.

A blind is a secret record, `<paths.blinds>/<name>.blind.json` (seed, envelope,
fiducial), drawn once with
`spv-container exec python -m sp_validation.blinding init <name>` and never
committed. On a blinded catalogue, ξ± and Cℓ_EE are shifted by
t(hidden) − t(fiducial) before they are first written: Smokescreen draws the
hidden point (S8 and Ωm) from the seed, and the theory `t` is
`sp_validation.blinding.shear`. Cℓ_BB and Cℓ_EB get zero shift; ρ/τ are saved
with `blinding.no_signal`. Another statistic passes its own theory,
`theory(params, s) -> array the length of s.mean` (see
`src/sp_validation/blinding.py`).

A measurement calculates, then saves: one function computes the signal, saves
it with `sacc_io.save(s, path, blind=...)` (or `sacc_io.seal`), and returns the
concealed part, so raw signal never leaves it
(`CosmologyValidation.calculate_2pcf_version` is the pattern). Statistics computed from
parts (COSEBIs, pure-E/B, the assembled `{version}.sacc`) are saved with
`derived_from=parts` and carry their inputs' stamp, `s.metadata["blind"]`.

Producer rules carry the catalogue's blind in `params.blind`, so unblinding is
flipping the declaration to `blind: none` and rerunning: Snakemake reruns
exactly the parts it touches, and assembly refuses a part whose stamp is not the
catalogue's declared blind.

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

A few rules shell out to a host toolchain (OneCovariance, ImageMagick) and keep
`container: None`; each says why in its own docstring.

The slurm executor submits with `--export=ALL`, so every job starts with the
launching shell's environment. `OMP_NUM_THREADS` is therefore not a profile
setting: a rule that needs it pinned sets it itself. A path in that
environment reaches nodes where it may not exist. For Snakemake's own cache
this is handled (the launch drops `XDG_CACHE_HOME`, and the profile keeps the
source cache off the shared filesystem), so a login shell that points it at
`/scratch` is fine; keep any other path you export on a shared disk. Per-rule
`mem_mb` / `runtime` stay on the rules.

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

### GLASS mock suite

Set `glass_mocks.data_dir` in the paper config to the suite's `results` directory.
Both paper configs use `/n09data/guerrini/glass_mock_v1.4.6.3_v2/results`, the suite supplying Paper II's 350 mocks.
Catalogue measurements, mock covariance, and inference preparation all read this key.
Generated mock products carry the parent directory's suite name (`glass_mock_v1.4.6.3_v2`) in their output paths, so suites don't share products.
`glass_mocks.version` labels the inference product version within a suite; `seed_range` selects the covariance and inference seeds.
The fine-grid measurement aggregators use the first 100 mocks.
Sampled tau statistics depend on the real catalogue, not the mock suite, and retain their catalogue-derived inputs and output paths.

```bash
snakemake -n --profile workflow/profiles/default -s workflow/Snakefile \
    covariance_glass_mock glass_mock_all_xi inference_glass_mocks \
    --configfile papers/bmodes/config/config.yaml
```

### Catalogue-painted GLASS mocks

The `glass_painted_maps` and `glass_painted_catalogue` rules project a single
source n(z) through lognormal matter shells, then paint the resulting shear
onto the positions of a calibrated FITS/HDF5 catalogue.
The default `shear: gamma` makes the signal pure E with the saved C_EE as its
known answer; `shear: reduced` applies gamma/(1-kappa) and includes physical
reduced-shear corrections to EE and BB.
The CLI exposes the same choice through `--shear {gamma,reduced}`, and the
catalogue metadata records it.
Each input ellipticity keeps its amplitude and receives an independent uniform
orientation; positions, weights and auxiliary columns remain row-aligned.
This models the projected source population rather than drawing a source
redshift per object.
The default sky has nside 4096 and lmax 8192; `harmonic_lmax` is explicit so the
painted mocks have power throughout the analysis range, independently of the
sampled mocks' `lmax=nside` default.
The default zmax of 3.5 contains the nonzero tail of the v1.4.6.3 B n(z).

Add this opt-in configuration to a run config (paths are examples):

```yaml
glass_paint:
  output_dir: results/glass_painted
  nz: /path/to/nz_B.txt
  threads: 28
  map_config:
    nside: 4096
    harmonic_lmax: 8192
    zmax: 3.5
    limber: true
  catalogues:
    sample:
      path: /path/to/calibrated_catalogue.fits
      ra_col: RA
      dec_col: Dec
      w_col: w_des
      e1_col: e1
      e2_col: e2
      shear: gamma
      psf_cols: [e1_PSF, e2_PSF]
  injections:
    none: {}
    leakage:
      alpha: 0.02
      additive: [0.0001, 0.0]
    additive_b:
      b_amplitude: 2.6e-11
      b_seed: 401
```

Target `results/glass_painted/seed_00001/sample_none.h5` with the shared
workflow and the run config.
On an existing allocation, run with `srun -n 1 -c 28` and the default profile
(`-j 28`), not the cluster profile that submits new jobs.
The standalone equivalent is:

```bash
python scripts/glass_mock/paint_catalogue.py maps --seed 1 --nz /path/nz_B.txt \
    --output results/glass_painted/seed_00001/maps --cache results/glass_painted/cache
python scripts/glass_mock/paint_catalogue.py paint --seed 1 \
    --input /path/calibrated_catalogue.fits \
    --maps results/glass_painted/seed_00001/maps \
    --output results/glass_painted/seed_00001/sample_none.h5
```

Each sky stores float32 RING `kappa.npy`, `gamma1.npy`, `gamma2.npy`,
`spectra.npz` (measured full-sky KK, EE, BB and cross spectra; shell-ensemble
KK/EE with and without pixel windows), and `maps.json` (cosmology, seed,
source weights, timings and cache key).
The exact shell theory includes the same source partition and three-neighbour
shell covariance truncation as the generator, and the spherical spin-2 factor.
Both the target and the padded lognormal solver's achieved ensemble spectra are
saved, so solver residuals can be checked quantitatively.
The solver runs independent shell pairs in spawned single-thread processes,
bounded by the rule's threads or the CLI's `--workers` (default OMP_NUM_THREADS).
The locked cache holds seed-independent matter and Gaussian spectra.
It is keyed by cosmology, shell geometry, n(z), resolution and GLASS version.

The output HDF5 has a root `data` table, readable by `read_catalogue`.
It stores `e1/e2` (or caller-named columns), `e1_noise/e2_noise` (rotated intrinsic
ellipticities), `g1_true/g2_true` (the selected noise-free signal, gamma by
default), `kappa_true`, and
`e1_injected/e2_injected` (the optional additive terms).
The input calibrated `e1_leak_corrected/e2_leak_corrected` aliases, if present,
also receive the mock shapes so corrected-column diagnostics stay mock-only.
These aliases don't apply a leakage correction to an injected mock.
Input ngmix/uncalibrated shape columns are auxiliary and must not be used as
mock shears; the mock's calibrated shape columns are authoritative.
A JSON sidecar and HDF5 attributes record rotation and injection seeds and
amplitudes.
For a B injection, `<catalogue>.bmode_spectra.npz` stores the target BB and
the realised full-sky EE/BB/EB of the injected field; its path is in the header.
The catalogue header also records weight sums and full-catalogue one-point
moments of the rotated noise and painted shapes, using both w and w² weights.
These diagnostics never use unrotated data components.
Columnar HDF5 inputs (one row-aligned dataset per column) are read lazily by
`Catalogue`, so compact 138M-row inputs stay bounded by `chunk_size`.
For a calibrated v2 input with a `w` column, set `w_col: w` and R=1 in the
consumer; `halo_clear` is copied unchanged for subsequent row selections.
A B-mode injection accepts either `b_amplitude` for
`C_BB=A exp[-(ell/2000)^2]` or a two-column `b_cl_path`; these specify the
pixelized target spectrum, without another pixel-window factor.
Map gamma2 has the HEALPix sign; catalogue e2, g2, noise2 and injected2 have the
opposite sky-coordinate sign, matching cosmo_val's `pol_factor: -1`.
`test_glass_paint.py` proves this with the pipeline's NaMaster map estimator,
and fails the null with the opposite sign.
Only these known-truth mock catalogue entries may use `blind: none`.
The map and catalogue headers include stage timings; catalogue timing splits
reading, signal painting, output assembly and writing.
The known-answer figures can be generated with:

```bash
python scripts/glass_mock/plot_painted_mock.py \
    --maps results/glass_painted/seed_00001/maps \
    --catalogue /path/calibrated_catalogue.fits \
    --mock results/glass_painted/seed_00001/sample_none.h5 \
    --output results/glass_painted/diagnostics --reduced-comparison
```

This diagnostic runner uses the v1.4.6.3 column names (`RA`, `Dec`, `w_des`,
`e1`, `e2`) for its catalogue panels.
It samples up to one million rows in distributed contiguous blocks to avoid a
full multi-GB reread; the figure and JSON identify the evaluated sample size.
Its real-catalogue accesses are only one-point shape-noise distributions;
it never measures a real cosmological vector.
The reduced-shear comparison is entirely mock-map based, with both EE and BB
measured through `healpy.anafast(pol=True)`.
The sign-control panel instead uses cosmo_val's NaMaster primitive.

### Which `sp_validation` a rule imports: the launched checkout

The image is the frozen *dependency stack*; the `sp_validation` that runs is
the one in the checkout you launched from — `common.configure()` puts that
checkout's `src/` on each job's `PYTHONPATH`
(`common.inject_checkout_pythonpath` has the mechanics).

This is the default because the alternative is incoherent: Snakemake's
`script:` directive already runs the checkout's *script files*, so without it a
rule executes new script code against an old `import sp_validation` — the two
halves of one commit, split.

The catalogue config defaults to the launched checkout's `cosmo_val/cat_config.yaml`.
Set `catalog_config: /path/to/catalogues.yaml` in the run config to use an isolated
registry, including its own `paths.blinds`; every producer reads that same file.
Configured catalogue names also work as version wildcards, so a mock entry does
not need a ShapePipe `SP_v...` name.
`COSMO_VAL` defaults to the launched checkout's `cosmo_val/output`, so writing
into another checkout's products means naming it; `COSMO_INFERENCE` defaults to
the shared candide tree, which holds the covariances and only its owner
can write. Anyone else launches with `COSMO_INFERENCE=<tree>` of their own,
holding a link to the shared tree's `data/mask/` (the one input the covariance
rules take from it); the OneCovariance rules then run there:

```bash
mkdir -p <tree>/data
ln -s /n17data/cdaley/unions/code/sp_validation/cosmo_inference/data/mask <tree>/data/
```

**Caveat:** `rerun-triggers: code` watches rule bodies and `script:` files, not
`src/`. Editing a module under `src/` does not by itself mark outputs stale —
force with `-F` or `--forcerun <rule>`.
The `cosmo_val` plot rules are an exception: their plotting modules are explicit inputs, so editing those modules invalidates figures without invalidating computations.

### Re-render the cosmo_val figures from cache

`cosmo_val_all` runs the computations and every companion plot.
`cosmo_val_plots` gathers only figures, including the reporting/integration covariance correlation plots.
Plots read the sealed SACC/NPZ products, sparse footprint counts, weight histograms and PSF/leakage summaries; no plot rule reads a catalogue or fits a model.
Every figure is a declared output and is overwritten when its plot rule runs.

After a full run, use the same config and output roots to render changed plots:

```bash
snakemake --profile workflow/profiles/default -s workflow/Snakefile \
    --configfile <run.yaml> --cores 4 cosmo_val_plots
```

To force every figure to be restyled, while **refusing to run any computation even if a cache is missing**, restrict the allowed rules:

```bash
plot_rules=(cosmo_val_plots cv_plot_rho_stats cv_plot_tau_stats \
    cv_plot_rho_tau_fits cv_plot_footprints cv_plot_objectwise_leakage \
    cv_plot_weights cv_plot_2pcf cv_ratio_xi_sys_xi cv_plot_pure_eb \
    cv_plot_cosebis cv_plot_pseudo_cl covariance_plot)
snakemake --profile workflow/profiles/default -s workflow/Snakefile \
    --configfile <run.yaml> --cores 4 --forceall \
    --allowed-rules "${plot_rules[@]}" -- cosmo_val_plots
```

Run this on an allocation, using the host's container/bind settings as usual.
Without the rule restriction, requesting a missing data product can schedule its producer.
A pre-split run needs one compute pass to create the new caches: old pure-E/B NPZs lack the comparison vectors and old COSEBIs NPZs contain only the fiducial cut.
That pass derives cosmological products only from the already-sealed integration SACC; no unsealed vector is added for plotting.
The objectwise rule now calls the spin-2 regression directly rather than the external all-in-one exploratory driver, so it produces the suite's leakage-coefficient comparison rather than unrelated 1D-fit/scatter side effects.

To reproduce a run from the image alone, opt out:

```bash
snakemake --profile workflow/profiles/candide --config checkout_pythonpath=false <target>
```

Either way the checkout has to sit on a disk the jobs see (next section).

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
`n17` is in the profile's exclude list. `common.py` spells the launched
checkout, `COSMO_VAL` and `COSMO_INFERENCE` in the plain form whatever spelling
it is given (a symlink, a relative path, `/automnt`), and Snakemake matches a
target by its path string, so name file targets in the plain form too; keep new
paths the same. A job sees a plain path only on a disk the profile binds by
name — `/home`, `/n17data`, `/n23data1`, `/n09data` (the `/automnt` bind does
not serve it) — so the checkout and both output roots sit on one of those. To
work on another disk, add it to both bind lists: the candide profile's
`apptainer-args` and `container.DEFAULT_BINDS`.

### Run Snakemake from the host, never from inside the container

`snakemake` is a thin host-side tool, installed once per machine on the
image's Python:

```bash
uv tool install --python 3.12 snakemake --with snakemake-executor-plugin-slurm
```

Any Snakemake version works, but it must run on the image's Python (3.12): a
`script:` job loads the host's `snakemake` package into the image's interpreter,
since the image carries none.

Run every `snakemake` command directly on the host — do not `apptainer shell`
first.
Snakemake itself never touches the science stack; it only reads rule
definitions and submits jobs. Each job carries its own `apptainer exec`
wrapping from the profile (see above), so the container is where the science
code runs, not where the orchestrator runs — one container per job, never a
nested one.

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
installed: run `python3 src/sp_validation/container.py`, or put it on your PATH
once (the README's install step):

```bash
ln -s "$PWD/src/sp_validation/container.py" ~/.local/bin/spv-container
```

**Do the pull from a compute node**, not the login node: it moves ~1.5 GB and
takes about fifteen minutes. `pull` writes to a temporary name and renames, so a
job either gets the whole old image or the whole new one; jobs already running
hold the old file open and finish against it unharmed.

```bash
salloc -p comp -c 4 --time=01:00:00 --exclude=n17,n36 --no-shell   # note the job id
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

### Checking the workflow itself

`workflow/tests/` checks DAG properties through the host launcher, on a toy
checkout and — on candide — on the real papers:

```bash
uv run --isolated --no-project --python 3.12 --with snakemake \
    --with snakemake-executor-plugin-slurm --with pytest \
    pytest workflow/tests
```

CI runs the same suite with `-m "not candide"`. `test_container_smoke`
submits one real SLURM job through the candide profile, so it runs only where
`sbatch` exists — a candide login node — and skips on compute nodes.

### `snakemake` in `script:` files

Every script run via a rule's `script:` directive uses a bare `snakemake`
name (`snakemake.input[...]`, etc.) with no import — Snakemake injects it as
a module global before the script runs. `from snakemake.script import
snakemake` is IDE-hint-only and raises `ImportError` if actually executed.
