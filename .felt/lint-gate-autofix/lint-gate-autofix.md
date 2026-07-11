---
id: 01KX710NNPDHJ0YYXZB4TTN45R
name: 'Server-side ruff autofix bot (PR #264)'
tags:
    - sp-validation
    - ruff
    - ci
created-at: 2026-07-11T00:05:21.462237313+02:00
updated-at: 2026-07-11T00:05:21.462237313+02:00
outcome: 'PR #264 (draft): lint gate now autofixes same-repo PRs — ruff --fix-only + format, bot commit pushed to the branch, re-lint in same run (GITHUB_TOKEN push doesn''t retrigger; no recursion). Fork PRs/develop pushes unchanged. Motivated by PR #236''s Copilot whack-a-mole; kills the failure class without anyone running pre-commit install. Awaiting Cail review+merge, then live test via scratch PR (gate runs base-branch workflow, can''t self-test).'
---
