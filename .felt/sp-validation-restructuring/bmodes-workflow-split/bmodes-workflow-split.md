---
id: 01KTD18JXS7VK50ECJX3GRWS9W
name: 'Codex reorg: split bmodes into modular workflow/ + papers/bmodes/'
status: open
tags:
    - constitution
    - sp-validation
    - reorg
    - shuttle
created-at: 2026-06-06T01:18:01.913028797+02:00
outcome: 'Steps 1-2 landed on cleanup/restructuring. The B-modes workflow now has common.py for shared constants/helpers and rules/twopoint.smk for xi/rho-tau/pseudo-Cl generation; Snakefile is slimmer but still in-place. Full guard net green at checkpoint: 45 passed, 1 skipped. Next step: slim Snakefile further / prepare top-level workflow move.'
shuttle:
    enabled: false
    kind: oneshot
    interactive: false
    host: candide
    project_dir: /automnt/n17data/cdaley/unions/pure_eb/code/sp_validation
    agent: codex
---

## Desired State

Split the `2D_bmodes_paper_workflow` into the **modular architecture** below — the design
settled live with Cail — moving in **small `git mv`/refactor steps with the back-pressure
net green after EVERY step**. This is a long-running job: work methodically, commit per step,
push to `cleanup/restructuring` (the WIP draft PR #197 — it's our branch) for durability.

### Target structure (top-level in the sp_validation repo)

```
workflow/                     ← generic, reusable analysis — the shared compute base
├── common.py                 ← shared HELPERS + static constants (plain Python; imported everywhere)
├── Snakefile                 ← POINTERS ONLY: from common import *, config, wildcard_constraints, includes
└── rules/
    ├── twopoint.smk          ← xi, xi_highres, rho_tau_stats, run_cosmo_val, pseudo_cl
    │                            (correlation functions AND power spectra — one home, Cail's call)
    ├── covariance.smk        ← covariance rules (+ any covariance-only local helpers)
    ├── inference.smk
    ├── masks.smk
    └── glass_mock.smk

papers/bmodes/                ← THIS paper's epistemic layer + the composition
├── Snakefile                 ← from common import * ; module-import ../../workflow/Snakefile ;
│                               use rule * from analysis ; paper constants ; include epistemic rules
├── config/                   ← this paper's config.yaml (+ cat_config reference)
├── rules/  claims.smk · ecut.smk · synthesis.smk · presentation.smk
└── scripts/                  ← paper-specific scripts
```

### The architecture — DECIDED, do not re-litigate

- **`module` for rules, plain Python `import` for helpers.** The paper composes the workflow's
  rules: `module analysis: snakefile: "../../workflow/Snakefile"; config: config` then
  `use rule * from analysis`. The shared **helper functions** live in `workflow/common.py`,
  imported via `from common import *` in the workflow Snakefile, the rule files, AND the paper
  Snakefile + its epistemic rules. This is the crux: **`module` imports rules, not the Python
  globals of the module's snakefile** — so helpers would be unreachable through `module` alone;
  a plain Python import is namespace-clean and jumps that wall. That is how we "do both."
- **`common.py` holds the CROSS-DOMAIN helpers** — the ones the *epistemic* rules reach into
  (verified by grep): `covariance_path`, `covariance_base`, `covariance_dir`,
  `resolve_covariance_version`, `build_redshift_path`, `get_shear_catalog`,
  `fiducial_binning_suffix`, plus static constants `COSMO_VAL`, `COSMO_INFERENCE`, `CAT_CONFIG`,
  `BLINDS`, `BLOCK_PAIRS`. Domain-only helpers may stay in their topical `.smk`.
- **Config-derived constants** (`FIDUCIAL = config["fiducial"]`, `DEFAULT_MASK_SUFFIX`) cannot be
  module-level in `common.py` (no `config` at import time). Set them in the Snakefile *after*
  `configfile`, or expose `common.get_fiducial(config)`. Helpers that currently close over the
  `FIDUCIAL` global must take it as a parameter instead (they already accept `min_sep`/`max_sep`/
  `nbins` overrides — extend that pattern).
- **The main `workflow/Snakefile` is pointers + constants only** — `from common import *`,
  `configfile`, `wildcard_constraints`, and the `include:` list. **No compute rules in the body**:
  the inline `xi`/`xi_highres`/`rho_tau_stats`/`run_cosmo_val` rules move into `twopoint.smk`.
  Keep it self-documenting — a person should read the Snakefile as a map.
- **Paper-only constants** (`TAPESTRY_DIR`, `PAPER_FIGURES_DIR`, `CONFIG_DIR`) live in the paper
  Snakefile, not `common.py`.
- **`specs.smk` is dead** (included nowhere; epistemic by category). Default: **drop it** (it's in
  git history). Moving it to `papers/bmodes/rules/` is acceptable if you prefer — your call.

### The net — GREEN after EVERY step (run in the container)

```
apptainer exec --bind /home,/scratch,/automnt,/n17data,/n23data1,/n09data \
  /n17data/cdaley/containers/containers/ python3.12 -m pytest \
  src/sp_validation/tests/test_imports.py \
  src/sp_validation/tests/test_bmodes_workflow_dry_run.py \
  src/sp_validation/tests/test_config_paths_exist.py \
  src/sp_validation/tests/test_tracked_symlinks.py \
  src/sp_validation/tests/test_dangling_move_references.py -v
```

(These five guards were built by the [[back-pressure-suite]] agent — read them first; they are
the contract.) **If any guard goes RED: STOP.** Fix it, or if you can't, leave the tree at the
last green commit and write exactly what broke to this fiber's outcome + `felt history`. Never
`git mv` past a red guard. As you move directories, **add `(old → new)` entries to the move-map
in `test_dangling_move_references.py`** so guard ⑥ actively checks each move's references.

### Step sequence (small, committed, net green after each)

1. **Extract `common.py` in place** (still inside `2D_bmodes_paper_workflow/`): pull the
   cross-domain helpers + static constants out of the Snakefile body; `from common import *`
   in the Snakefile + rule files; refactor `FIDUCIAL`-dependent helpers to take it as a param.
   Net green → commit.
2. **Extract `twopoint.smk`**: move the inline two-point rules (`xi`, `xi_highres`,
   `rho_tau_stats`, `run_cosmo_val`) — and the `pseudo_cl` rules from `covariance.smk` — into it;
   `include:` it. Net green → commit.
3. **Slim the Snakefile** to pointers + constants + config + includes. Net green → commit.
   *(Now internally modular, still in place.)*
4. **Create top-level `workflow/`**; `git mv` the generic pieces (`common.py`, `Snakefile`,
   `rules/{twopoint,covariance,inference,masks,glass_mock}.smk`) into it. Fix run-CWD / config-path
   resolution. Net green → commit. (Add the move to ⑥'s map.)
5. **Create `papers/bmodes/`**; `git mv` the epistemic rules + config + scripts there; write
   `papers/bmodes/Snakefile` (`from common import *` + `module`-import `../../workflow/Snakefile`
   + `use rule *` + paper constants + include epistemic rules). Net green → commit.
6. **Repoint the `pure_eb/workflow` symlink** (currently → `…/2D_bmodes_paper_workflow`) to the
   new run location so the run-from-`pure_eb/` flow + guard ⑤ stay green. Net green → commit.
7. **Final verify**: full net green, and `snakemake -n all_tapestry` (or the paper target) exits 0
   from the run location.

### Then (only if green and time remains) — the other paper dirs

Same pattern, simpler (no Snakefile split — mostly dir-move + path-existence), each behind the net,
each committed, each added to ⑥'s map:
- `notebooks/cosmo_val/catalog_paper_plot/ → papers/catalog/`
- `2D_harmonic_space…_plots/ → papers/harmonic/`
- the **untracked** `cosmo_inference/notebooks/2D_cosmic_shear_paper_plots/ → papers/cosmic_shear_2d/`
  (this one is `git add`, not `git mv` — it was never tracked)
- fold `glass_mock` core into `src/sp_validation/`.

**Out of scope for this agent:** notebook curation, `nbstripout`, `scratch/`, the science run. A
later pass owns those.

## Context

**Why this architecture.** The seam is **compute (infrastructure) ↔ epistemic (evidence)** — a line
the Snakefile already half-draws in its `include:` grouping ("Compute rules" vs "Epistemic rules").
The compute rules (`covariance`, `inference`, `masks`, `glass_mock_validation`, plus the inline
two-point rules) are reusable analysis → `workflow/`. The epistemic rules (`claims`, `ecut`,
`synthesis`, `presentation`, dead `specs`) are this paper's argument → `papers/bmodes/`. A coupling
grep showed the epistemic rules reach into the compute helpers (`claims` → `covariance_path`,
`COSMO_VAL`, `COSMO_INFERENCE`, `BLINDS`; `presentation` → `build_redshift_path`, `FIDUCIAL`) — which
is exactly why those cross-domain helpers are centralized in `common.py` and imported on both sides.
We chose `module`+`common.py` over plain `include:` because Cail wants to keep `module` for future
multi-paper composition (per-run `config:`, output `prefix:`, per-rule `with:` override). Full design
thread: parent [[sp-validation-restructuring]].

**Paths stay messy — do NOT relativize.** There are ~117 hardcoded absolute paths
(`/n17data/…`, `/automnt/…`, `/home/…`). Cail's explicit decision: **we do not rewrite them.** They
ride along with their files. Guard ③ (`test_config_paths_exist`) checks they still *exist* on
candide — keep it green; don't introduce NEW missing paths. (Consequence: `module`'s `prefix:`
output-namespacing won't fully bite while outputs are absolute — that payoff is deferred to a future
path-relativization pass. Adopting `module` now still buys the structure + config injection + override.)

**Run mechanics.** Container: `apptainer exec --bind /home,/scratch,/automnt,/n17data,/n23data1,/n09data /n17data/cdaley/containers/containers/ <cmd>`
(`app` is an interactive bash function that won't exist here). Container Python is **3.12**. The
bmodes workflow currently runs from `/automnt/n17data/cdaley/unions/pure_eb/` (where `workflow/`
symlinks to the bmodes dir); `snakemake` is on the login PATH for dry-runs (no execution, safe).
A bare `snakemake -n` fails — there's no `rule all`; use a concrete target like `all_tapestry`.
The Snakefile reads `results/cosmology/planck18.json` at parse time — that file must exist where
snakemake runs.

**Constraints.**
- Branch `cleanup/restructuring`. `git mv` (never `cp`) for `--follow` provenance; small commits per
  step; push to `cleanup/restructuring` for durability (it's our WIP draft PR #197).
- Tree is yours this run (no interactive session, no other agent) — but still: stage only files you
  intend; do **not** `git add -A` blindly (there are unrelated `.felt/` dirs and an untracked
  `2D_cosmic_shear_paper_plots/` — leave those alone unless step "other paper dirs" reaches them).
- Run the net after every step; never proceed past red. Update ⑥'s move-map as you move.

**Exit (autonomous oneshot).** Realize as much as lands green — ideally the full bmodes split
(steps 1–7), then the other paper dirs if time/context remain. Rewrite `outcome` (what's split,
what's green, where you stopped), append a `felt history` event, ensure the tree is green at the
final commit, set `status: closed`, `kill $PPID`. Do not self-`tempered`. If blocked by a red guard
you can't resolve, stop at the last green commit and lead the outcome with "Blocked: …".
