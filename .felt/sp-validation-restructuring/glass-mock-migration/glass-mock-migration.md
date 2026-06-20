---
id: 01KV8CY7CTFKTWRQFAGT6SEQR1
name: 'Finish glass_mock fold: library to src/, runners to scripts/'
status: closed
tags:
    - reorg
    - sp-validation
    - glass
created-at: 2026-06-16T14:21:35.002488872Z
updated-at: 2026-06-16T14:36:25.639840708Z
closed-at: 2026-06-16T14:36:25.639840033Z
outcome: 'DONE on cleanup/restructuring. Top-level glass_mock/ removed: ~10 reusable fns folded into src/sp_validation/glass_mock.py (heavy deps stay lazily imported, so module resolves without the GLASS stack); runners -> scripts/glass_mock/ as thin argparse wrappers; validate_glass_mock.ipynb deleted (exploratory, hardcoded /home/guerrini paths + broken imports). xfail test_glass_mock untouched (still gated on container [glass] rebuild). ruff green.'
---
