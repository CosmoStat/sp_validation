---
id: 01KV0M4MDDQFKXR6CN9XQG968F
name: GLASS/cosmology API mismatch in glass_mock map path
tags:
    - sp-validation
    - glass
    - bug
created-at: 2026-06-13T15:53:29.51768405+02:00
outcome: 'Adding GLASS to the container surfaced a latent break: glass_mock.py''s map path calls cosmology.Cosmology.from_camb() -> CambCosmology, which the installed (unpinned) glass+cosmology lacks comoving_distance for, so glass.distance_grid / MultiPlaneConvergence fail. Path was never run before (GLASS absent from the image until 2026-06-13). test_matter_maps_are_seed_deterministic is xfail''d (raises=AttributeError) so CI is green. FIX: pin a compatible glass+cosmology pair in pyproject [glass] extra (or adapt the calls), verify in the fresh image, drop the xfail. Needs the fresh sandbox built to test (glass is only in the published image, not the live candide sandbox).'
---
