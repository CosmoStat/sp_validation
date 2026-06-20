---
id: 01KV0M4MDDQFKXR6CN9XQG968F
name: GLASS/cosmology API mismatch in glass_mock map path
tags:
    - sp-validation
    - glass
    - bug
created-at: 2026-06-13T15:53:29.51768405+02:00
outcome: 'RESOLVED (pin landed, verified under uv). glass==2025.1 + glass.ext.camb==2023.6 + cosmology==2022.10.9 is the unique compatible set: glass 2025.1 has the flat map-path API AND still calls the legacy cosmo.dc/xm/ef that cosmology 2022.10.9 (its newest release) provides — no cosmology release exposes comoving_distance, so newer glass is incompatible with the cosmology package. matter_cls is a separate package (glass.ext.camb, absent from glass core >=2024.2). Pin is in pyproject [glass]; full map path verified seed-deterministic under bare uv+healpy. REMAINING: drop the test_glass_mock xfail once the container is rebuilt with [glass] (G1 / sandbox swap).'
---
