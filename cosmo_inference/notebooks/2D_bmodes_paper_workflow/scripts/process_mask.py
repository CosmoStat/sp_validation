#!/usr/bin/env python3
"""
Process single UNIONS HealSparse pixel mask to specified nside.

This script:
1. Loads high-resolution HealSparse mask
2. Downgrades to single specified nside value
3. Converts to standard HEALPix format
4. Calculates effective survey area

Author: Claude Code
Date: 2025-08-18
"""

import os
import sys
import numpy as np
import healpy as hp
import healsparse as hsp
import yaml


def load_healsparse_mask(mask_path: str, verbose: bool = True) -> hsp.HealSparseMap:
    """Load HealSparse mask from file."""
    if verbose:
        print(f"Loading HealSparse mask from: {mask_path}")
    
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    mask = hsp.HealSparseMap.read(mask_path)
    
    if verbose:
        print(f"Loaded mask with nside_coverage={mask.nside_coverage}, nside_sparse={mask.nside_sparse}")
        print(f"Map size: {mask.valid_pixels.sum()} valid pixels")
        
    return mask


def degrade_mask(mask: hsp.HealSparseMap, target_nside: int, verbose: bool = True) -> np.ndarray:
    """Degrade HealSparse mask to target nside and return as HEALPix array."""
    if verbose:
        print(f"Degrading mask to nside={target_nside}")
    
    # Degrade the mask
    degraded_mask = mask.degrade(target_nside, reduction='mean')
    
    # Convert to full HEALPix array
    healpix_mask = np.zeros(hp.nside2npix(target_nside))
    valid_pixels = degraded_mask.valid_pixels
    healpix_mask[valid_pixels] = degraded_mask[valid_pixels]
    
    if verbose:
        coverage_fraction = np.sum(healpix_mask > 0) / len(healpix_mask)
        print(f"Coverage fraction: {coverage_fraction:.4f}")
        mean_value = np.mean(healpix_mask[healpix_mask > 0])
        print(f"Mean mask value (valid pixels): {mean_value:.4f}")
    
    return healpix_mask


def calculate_effective_area(healpix_mask: np.ndarray, nside: int) -> float:
    """Calculate effective survey area in square degrees."""
    # HEALPix pixel area in steradians
    pixel_area_sr = hp.nside2pixarea(nside)
    
    # Convert to square degrees (1 steradian = (180/π)² square degrees)
    pixel_area_deg2 = pixel_area_sr * (180.0 / np.pi) ** 2
    
    # Sum weighted area
    effective_area = np.sum(healpix_mask) * pixel_area_deg2
    
    return effective_area


def save_healpix_mask(healpix_mask: np.ndarray, output_path: str, nside: int, 
                      nest: bool = False, verbose: bool = True):
    """Save HEALPix mask to FITS file."""
    if verbose:
        print(f"Saving HEALPix mask to: {output_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write map
    hp.write_map(output_path, healpix_mask, nest=nest, overwrite=True,
                 dtype=np.float64)


def save_area_summary(nside: int, effective_area: float, healpix_mask: np.ndarray, 
                      output_path: str, mask_output_path: str):
    """Save area summary for single nside to YAML file."""
    summary = {
        "mask_processing_summary": {
            "source_mask": "UNIONS HealSparse mask",
            "processing_date": "2025-08-18",
            "masks": {
                f"nside_{nside}": {
                    "nside": int(nside),
                    "effective_area_deg2": float(effective_area),
                    "coverage_fraction": float(np.sum(healpix_mask > 0) / len(healpix_mask)),
                    "n_valid_pixels": int(np.sum(healpix_mask > 0)),
                    "total_pixels": int(len(healpix_mask)),
                    "mean_mask_value": float(np.mean(healpix_mask[healpix_mask > 0]) if np.sum(healpix_mask > 0) > 0 else 0.0),
                    "output_file": mask_output_path
                }
            }
        }
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=True)
    
    print(f"Saved area summary to: {output_path}")


def main():
    """Main processing function."""
    # Handle both Snakemake and interactive execution
    if hasattr(sys, 'ps1'):
        # Interactive mode - use example parameters
        from snakemake_helpers import snakemake_interactive
        snakemake = snakemake_interactive(
            output_mask="/n17data/cdaley/unions/pure_eb/output/masks/mask_nside4096.fits",
            output_summary="/n17data/cdaley/unions/pure_eb/output/masks/area_nside4096.yaml",
            nside=4096
        )
    else:
        from snakemake.script import snakemake
    
    # Get parameters from Snakemake
    source_mask_path = snakemake.input.mask
    target_nside = snakemake.params.nside
    output_mask_path = snakemake.output.mask
    output_summary_path = snakemake.output.summary
    
    verbose = True
    print(f"=== Processing Single Mask: nside={target_nside} ===")
    print(f"Source mask: {source_mask_path}")
    print(f"Output mask: {output_mask_path}")
    
    # Load original mask
    original_mask = load_healsparse_mask(source_mask_path, verbose=verbose)
    
    # Degrade mask
    healpix_mask = degrade_mask(original_mask, target_nside, verbose=verbose)
    
    # Calculate effective area
    effective_area = calculate_effective_area(healpix_mask, target_nside)
    print(f"Effective area: {effective_area:.2f} deg²")
    
    # Save mask
    save_healpix_mask(healpix_mask, output_mask_path, target_nside, verbose=verbose)
    
    # Save summary
    save_area_summary(target_nside, effective_area, healpix_mask, 
                      output_summary_path, output_mask_path)
    
    print(f"Successfully processed nside={target_nside}")


if __name__ == "__main__":
    main()