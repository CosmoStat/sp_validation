# Painted-mock cosmo_val known-answer suite (v1.4.6.3, seed 1)

**cosmo_val passes every known answer on the painted v1.4.6.3 GLASS mock, with two exceptions, both understood.**
- Fiducial-cut COSEBIs are blind to white-like and PSF-like B by construction.
- ρ/τ recovers only 80% of an injected α, because shear_psf_leakage centres the galaxies but not the PSF fields (CosmoStat/shear_psf_leakage#58).

Run: `run/glassmock-int` @ 20eb4013, which is integration/2026-10-31 @ 63f90fe8 plus the shared integration-covariance reference (fff78fba). All 63 workflow steps plus validation completed 31 Oct 18:12 (drivers `dispatch_63f9.sbatch`, logs `logs/*_63f9*.log`).
- Integration has since moved to 3ca4a2ee (#458), which touches only the tomographic covariance path. I read that from the diff and did not rerun.
- Stars enter ρ/τ on flag cuts only. The v1.4.6.3 config sets no psf `mask_config`, the same as for the data.

No real-data vector and no real blind were opened.

## Entries

Four cat_config entries share one painted sky (seed 1), one shape-noise rotation and hash-identical RA/Dec/w (`geometry_audit.json`), so their differences are the injections:

| Entry | What it is | Truth |
|---|---|---|
| `GLASS1_clean` | γ painted onto v1.4.6.3 positions and weights, \|e\| rotated | input C_EE; B = 0 |
| `GLASS1_blinded` | byte copy under the mock-only blind `glassmock-test` (ΔS8 = +0.0553, ΔΩm = −0.0188) | designed E shift; B unchanged |
| `GLASS1_psfleak` | clean + 0.1 e_PSF | α = 0.1 |
| `GLASS1_addB` | clean + Gaussian B, C_BB = 2.6e-11 exp[−(ℓ/2000)²] | that B |

Pair counts and the Gaussian integration covariance are computed once for clean and shared by explicit reference (`covariance.pair_counts_reference`, `integration_reference`). Every shear, ρ/τ and fit product is measured per entry.

Predeclared: reject at an individual PTE < 0.01; primary cuts ℓ 300–1600 and θ 12–83′.

## Known answers

| Check | Result | Figure |
|---|---|---|
| EE vs the realisation's Cℓ | χ² 6.86/17 (fiducial), 25.3/32 (all); amplitude 1.005 ± 0.035 | `ee_harmonic.png` |
| ξ± vs realisation (direct Bessel sum, NG cov) | χ² 5.07/14 (12–83′), 24.9/40 (all) | `ee_xi.png` |
| Cℓ BB / EB null, clean | PTE 0.23 / 0.94 (all bands) | `bmode_harmonic.png` |
| ξ+B / ξ−B / COSEBIs null, clean | PTE 0.64 / 0.72 / 0.58 (12–83′); COSEBIs 0.58 on 1–250′ | `bmode_decompositions.png` |
| Blind: blinded − clean = designed shift | to 6.6e-24 (Cℓ), 4.4e-18 (ξ) | `blinding_harmonic.png`, `blinding_xi.png` |
| Blind size | ≤ 1.45σ per Cℓ band, ≤ 2.16σ per ξ+ bin, ≤ 1.85σ per ξ− bin; Δχ² 47.7 (Cℓ), 20.7 (ξ±) | same |
| Blind leaves B alone | Cℓ BB/EB to 5e-15σ; ξ_B to 3e-4σ; COSEBIs to 3e-3σ; ξ covariance bit-identical | same |
| Additive B, harmonic | A = (2.52 ± 0.57)e-11 vs 2.6e-11, S/N 4.45; BB null PTE 1e-5 | `additive_harmonic.png` |
| Additive B, real space (excess over clean / clean σ) | ξ−B 3.97σ and ξ+B 0.44σ on 12–83′; 5.6σ and 5.1σ on 1–250′ | `bmode_decompositions.png` |
| Additive B, COSEBIs | 0.71σ on 12–83′ (null PTE 0.68); 4.91σ on 1–250′ (PTE 3.9e-5) | same |
| PSF leak, object-wise | Δα = 0.1 to 6e-11 | `objectwise_psf.png` |
| PSF leak, ρ/τ (EIV), paired | Δα = 0.080, Δβ = −0.07, Δη = −1.05 vs (0.1, 0, 0) | `psf_leakage.png` |
| PSF leak, B-mode summary | ξ+B PTE 5e-23, ξ−B 1e-3, Cℓ BB 6e-52; COSEBIs 0.27 on 12–83′ (4e-14 on 1–250′) | `bmode_decompositions.png` |
| Blind custody | leak gate passes over 53 files | `blind_scan_manifest.json` |

Notes on reading these:

- **EE and ξ± against the realisation are consistency checks, not exact ones.** The ~10%-sky footprint keeps its own cosmic variance around the full-sky realised spectrum, so the high PTEs are expected. Seed 1 is a low draw in both EE and ξ±; the ten-seed PTEs are uniform (`../pte/RESULT.md`).
- **Additive B is white-like up to ℓ ≈ 2000.** Above a few arcmin its ξ+ vanishes and ξ− ∝ θ⁻², an E/B-ambiguous mode. Fiducial-cut COSEBIs annihilate it exactly and ξ+B on 12–83′ barely sees it, while ξ−B and 1–250′ COSEBIs detect it at the predicted level ([[cosebis-white-b-blind]]).
- **Read ρ/τ through the paired difference.** The absolute psfleak α = 0.097 ± 0.011 looks like a pass, but that is the clean mock's chance α = 0.016 (shear–PSF correlation, 1.7σ) added to an under-response of 0.080. The cause is the centring mismatch ([[rho-tau-centring-inconsistency]]): Δα was 0.075 with the earlier star_contamination sky mask, and the flag-only star sample moves it to 0.080.
- **Object-wise α errors omit coherent cosmic-shear variance.** The clean fit's nominal 2.9σ is chance shear–PSF correlation: a ten-sky noise-free audit puts the missing scatter at ~79× the per-object error for the noise-free slope ([[objectwise-alpha-cosmic-shear]]).

## Not null by construction

Real-star ρ; E modes and anything that blinds; footprint and weight diagnostics; the posterior mean of ξ_sys (coefficient variance makes it positive); single-realisation mean shear.

## Open and next

- **Inference round trip** (blinded vs clean chains recovering the mock cosmology): not done; scoped as the next block.
- **v2 footprint:** the suite runs on Nibi, where v2's stars, mask and real config live. A safe painted v2 sky exists on candide (`../mocks/v2/seed_00001/painted.h5`).
- **ξ+ at ~1′:** the ensemble mean squared residual is 2.6. The suspected cause, mock pixel resolution (0.86′), is unverified.

## Reproduction

`README.md` has the commands, image and conventions. Recipe snapshot: `scratch/cdaley/glass_mock_cvpaint/` on `run/glassmock-int`. Machine-readable numbers: `metrics.json`.
