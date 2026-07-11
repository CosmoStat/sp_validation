# Proposed issue tree: SACC + Smokescreen-fork blinding stream

Board model (UNIONS-WL/projects/1): work areas slice; **everything is an issue**; **PRs are never cards — they ride Development links** via same-repo `Closes #N`; issues nest as sub-issues (cross-repo nesting allowed and survives transfer); **file each issue in the repo where the code changes**; the umbrella carries `Kind = Epic` and **no Status**; every item carries a `Work area`.

**Numbering convention in this doc.** Nothing is posted to GitHub yet, so board issue numbers are *not yet assigned* — issues are named with symbolic placeholders (`ISSUE-DEPS`, `ISSUE-FORK`, …). The `#NNN` integers below are the *existing draft PRs* against the old PRD (#243/#245/#249/#251/#253/#255) and refer to PRs only, never issues. #232/#234/#241 are the real, already-existing umbrella/design/PRD numbers.

Two repos are in play:
- **`CosmoStat/sp_validation`** — all format/migration/blinding-wiring/inference work.
- **`UNIONS-WL/Smokescreen`** (the new fork) — the fork + backend-independence work.

## Tree

```
EPIC  sp_validation #232  "SACC data format + Smokescreen blinding"   [Kind=Epic, no Status]
      (existing umbrella; design lives in PRD #241 / issue #234)
│
├─ #234  PRD/design issue (sp_validation)             ← PRD #241 text
│
├─ ISSUE-BOARDCFG  Add UNIONS-WL/Smokescreen to the effort
│               repo: UNIONS-WL/unions-wl.github.io  (board-config home; precedes ISSUE-FORK)
│
├─ ISSUE-FORK   Fork Smokescreen + backend independence
│               repo: UNIONS-WL/Smokescreen    (cross-repo sub-issue under #232)
│               ├─ ISSUE-FORK-PROTO  theory_fn protocol + default CCL backend + fixed draw + firecrown lazy-sidelining + fork-test migration → sub-PR
│               └─ ISSUE-FORK-PKG    packaging/install-identity + fork CI + pip-only docs → sub-PR (unblocks ISSUE-DEPS)
│
├─ ISSUE-DEPS    Dependencies            (sp_validation)  ← draft PR #243
├─ ISSUE-SACCIO  sacc_io writers         (sp_validation)  ← draft PR #245
├─ ISSUE-CONV    Converters              (sp_validation)  ← draft PR #249
├─ ISSUE-MIGRATE cosmo_val migration     (sp_validation)  ← draft PR #251
├─ ISSUE-BLIND   Smokescreen blinding wiring (sp_validation) ← draft PR #253  (reworked onto fork; CAMB↔CCL cross-check folded in)
└─ ISSUE-SACCLIKE Native SACC likelihood (sp_validation)  ← draft PR #255
```

Each `ISSUE-*` gets a fresh board number when filed; each draft PR attaches to its issue via same-repo `Closes #<that issue's number>`. Do not reuse the draft-PR integers as issue numbers — they are already spent as PRs, and typing them as issue IDs would collide.

## The one new issue

**`ISSUE-FORK` — "Fork Smokescreen to UNIONS-WL and make it theory-backend independent."**
- **Repo:** `UNIONS-WL/Smokescreen`. This is where the code changes, so the issue lives there even though its parent epic is in sp_validation. Cross-repo nesting is the board's designed behavior.
- **Parent:** sp_validation #232 (the existing SACC/blinding epic).
- **Body:** fork DESC Smokescreen to the UNIONS-WL org (community-facing blinding object); add a `theory_fn(cosmo_params) -> np.ndarray` backend protocol plus a built-in default CCL backend (so a standard cosmic-shear SACC file blinds out of the box; power users override with their own callable); make the inherited firecrown path lazy and optional (no module-level import) — it stays but is not installed, tested, supported, or maintained here; replace the global-seed/order-dependent draw with a fixed pure draw speaking CCL-native parameter names (no injectable-`draw_fn` abstraction — deferred); declare `pyccl` explicitly in the fork's `pyproject` (the default backend uses it; stock Smokescreen imports it at module level without declaring it) and settle the fork's install identity as a pinned `git+https` tag/SHA (no rename, no PyPI publish); migrate the fork's own test suite off module-level firecrown; add a backend smoke-test as the required CI gate; write pip-only install docs (`pip install git+https://github.com/UNIONS-WL/Smokescreen@<tag>` — no conda, no PyPI) noting the firecrown path is inherited-and-unsupported. No PR against DESC upstream; keep the fork diff small and clean for eventual upstream-PR-ability. **Acceptance:** a synthetic callable blinds a SACC vector end-to-end; the default CCL backend blinds a standard cosmic-shear SACC file; the amplitude envelope is expressed through CCL-native parameters; the fork blinds a real sp_validation `sacc_io` file end-to-end.
- **Decomposed now, not conditionally** — PR 0.1 is the highest-risk PR and one of its parts (packaging/CI) is the board's critical-path unblock. Two sub-issues, each with its own sub-PR:
  - **`ISSUE-FORK-PROTO`** — `theory_fn` protocol + default CCL backend (`datavector.py` + a new backend module) + fixed pure draw (`param_shifts.py`) + firecrown lazy-sidelining + fork-test migration. The draw fix folds in here: with the injectable abstraction dropped it is a small change coupled to the same `datavector` rewrite.
  - **`ISSUE-FORK-PKG`** — install identity (pinned `git+https` tag/SHA) + `pyccl` declaration + fork CI + pip-only docs. **This one unblocks `ISSUE-DEPS` and should not wait on review of the large protocol refactor.**
  Each sub-PR carries `Closes #<its sub-issue>` in `UNIONS-WL/Smokescreen`.
- **Fields:** `Work area` = same blinding/SACC area as the rest of the stream (read the live option off the board; do not hardcode). `Kind = Task`.

**`ISSUE-BOARDCFG` — "Add UNIONS-WL/Smokescreen to the effort."** Per the board model ("everything is an issue"), the board-config work is itself a card, not a floating prerequisite. `UNIONS-WL/Smokescreen` is **not** currently among the repos listed in the board README; adding it to the README and enabling `project item-add` of a `UNIONS-WL/Smokescreen` issue onto `UNIONS-WL/projects/1` is a discrete step that **precedes filing `ISSUE-FORK`**. Home repo: `UNIONS-WL/unions-wl.github.io` (the board README's "general-purpose home for issues that fit no code repo"). `Kind = Task`; same `Work area` as the stream. It has no PR — it is board/README config, closed by hand.

## Existing issues — placement unchanged, content adjusted

The six sp_validation sub-issues live in `CosmoStat/sp_validation` under epic #232, each carrying its implementing draft PR via same-repo `Closes #N`. The standalone CAMB↔CCL cross-check issue is retired — its test folds into `ISSUE-BLIND`. The content edits the ruling forces:
- **`ISSUE-DEPS`** (draft PR #243): body drops firecrown/`patch_firecrown.py`/numpy-ceiling and the `[blinding]` extra; blinding deps go core — the pinned `git+https` fork URL + `pyccl` (already core) + `cryptography` + `sacc>=0.12` (promoted to top-level); Python floor raised `>=3.11` → `>=3.12`. `cryptography` and `sacc>=0.12` come from the fork's `Requires-Dist` and `pyccl` comes from the fork declaring it — but *only after `ISSUE-FORK-PKG` rewrites the fork packaging*; "do not re-pin" is sound solely once that is verified (stock Smokescreen declares neither pyccl nor firecrown despite importing both). Body notes the container/`uv.lock` install-source move to the fork. **Blocked on `ISSUE-FORK-PKG`:** this PR cannot pin a fork SHA earlier than PR 0.1's firecrown-lazy-import + packaging commits, or it drags firecrown back in.
- **`ISSUE-BLIND`** (draft PR #253): body reworks onto the fork protocol (three sp_validation `theory_fn` backends overriding the fork's default CCL backend); the `systm_dict` IA-pin note is removed (firecrown is off our path — lazy and unsupported — and our backends are plain CCL callables that compute the fiducial exactly); the CAMB↔CCL cross-check is folded in as a test in this issue's suite.

## Why the fork issue is a sub-issue, not its own epic

The fork exists **to serve the blinding deliverable** — one enabling step in the SACC/Smokescreen stream, not a parallel work stream. It nests under #232 like the other implementation issues. It is cross-repo (the only one), which the board explicitly supports. If the fork later grows its own community-maintenance life (external users, releases, its own roadmap), promote it to a `Kind = Epic` in `UNIONS-WL/Smokescreen` then — not now.

## Ordering / dependency note (for reviewers, not a board field)

`ISSUE-BOARDCFG` precedes `ISSUE-FORK` (the fork repo must be on the board before its issue can be filed). Within `ISSUE-FORK`, `ISSUE-FORK-PKG` is the earliest sp_validation-facing unblock: `ISSUE-DEPS` is blocked on it (fork install identity + pyccl declaration). `ISSUE-FORK-PROTO` is independent of `ISSUE-DEPS`. The reworked `ISSUE-BLIND` depends on all of `ISSUE-FORK` (protocol + draw) plus the SACC writers (`ISSUE-SACCIO`). The pure-format PRs (`ISSUE-SACCIO`/`ISSUE-CONV`/`ISSUE-MIGRATE`) are independent of the fork and can proceed in parallel.
