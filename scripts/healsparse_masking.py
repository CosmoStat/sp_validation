#!/usr/bin/env python3
# %%
"""
Interactive Research Exercise: HealSparse Mask Area Analysis
Date: 2025-09-04

Calculates HealSparse coverage and mask areas for the UNIONS cosmic shear analysis.
"""

import os
import numpy as np
import pandas as pd
import tqdm
import healsparse as hsp

# %%
print("\n=== HealSparse Area Analysis Exercise ===")
print("Exploring area calculation algorithms and nside dependencies\n")

print("PART 1: Loading UNIONS HealSparse Mask")
print("=" * 50)

def create_npoint_mask(path):
    """Load coverage file and create boolean mask for coverage > 2."""
    print(f"Loading coverage file: {path}")
    npoint_map = hsp.HealSparseMap.read(path)

    print(f"  ✓ nside_coverage: {npoint_map.nside_coverage}, nside_sparse: {npoint_map.nside_sparse}")
    print(f"  ✓ dtype: {npoint_map.dtype}")
    print(f"  ✓ Total pixels with data: {npoint_map.n_valid:,}")
    
    full_area = npoint_map.get_valid_area(degrees=True)
    print(f"  ✓ Total area: {full_area:.2f} deg²")

    # Create sentinel mask for pixels with npoint > 2
    valid_pixels = npoint_map.valid_pixels[npoint_map[npoint_map.valid_pixels] > 2]
    npoint_mask = hsp.HealSparseMap.make_empty(
        nside_coverage=npoint_map.nside_coverage,
        nside_sparse=npoint_map.nside_sparse,
        dtype=np.int32,
        sentinel=0
    )
    npoint_mask[valid_pixels] = 1
    
    masked_area = npoint_mask.get_valid_area(degrees=True)
    print(f"  ✓ Area after requiring npoint > 2: {masked_area:.2f} deg²")
    print(f"    - Area lost: {full_area - masked_area:.2f} deg²")
    
    return npoint_mask

base_path = "/n17data/UNIONS/WL/masks/"
npoint_mask = create_npoint_mask(f"{base_path}/coverage.hsp")

# %%
# Load quality mask files
n_values = [1, 2, 4, 8, 64, 1024]
individual_masks = {}

for n_val in n_values:
    mask_file = f"mask_r_nside131072_n{n_val}.hsp"
    print(f"\nLoading mask n={n_val}: {mask_file}")
    
    mask = hsp.HealSparseMap.read(f"{base_path}/{mask_file}")
    individual_masks[n_val] = mask
    
    area_deg2 = mask.get_valid_area(degrees=True)
    print(f"  ✓ nside_coverage: {mask.nside_coverage}, nside_sparse: {mask.nside_sparse}")
    print(f"  ✓ Valid pixels: {mask.n_valid:,}")
    print(f"  ✓ Individual area: {area_deg2:.2f} deg²")


# %%
print("\nPART 2: Creating upgraded npoint masks for cumulative analysis")
print("=" * 50)

upgraded_mask_file = f"{base_path}/mask_r_nside131072_npoint.hsp"

if os.path.exists(upgraded_mask_file):
    print(f"Loading cached upgraded mask: {upgraded_mask_file}")
    upgraded_npoint_mask = hsp.HealSparseMap.read(upgraded_mask_file)
    upgraded_area = upgraded_npoint_mask.get_valid_area(degrees=True)
    print(f"  ✓ Loaded upgraded mask area: {upgraded_area:.2f} deg²")
    print(f"  ✓ Pixel count: {upgraded_npoint_mask.n_valid:,} valid pixels")
else:
    print(f"Creating upgraded npoint>2 mask and caching to: {upgraded_mask_file}")
    
    # Upgrade sentinel mask and create boolean mask matching quality mask parameters
    dummy_upgraded = npoint_mask.upgrade(131072)
    upgraded_npoint_mask = hsp.HealSparseMap.make_empty(
        nside_coverage=128, nside_sparse=131072, dtype=bool, bit_packed=True
    )
    
    print("  Processing coverage pixels...")
    for covpix in tqdm.tqdm(
        dummy_upgraded.get_covpix_maps(),
        total=dummy_upgraded.coverage_mask.sum(),
        desc="Coverage pixels"
    ):
        upgraded_npoint_mask[covpix.valid_pixels] = True
        
    upgraded_npoint_mask.write(upgraded_mask_file, clobber=True)
    print("  ✓ Mask created and cached")
            
    # Validate area preservation
    original_area = npoint_mask.get_valid_area(degrees=True)
    upgraded_area = upgraded_npoint_mask.get_valid_area(degrees=True)
    print(f"  ✓ Upgraded area: {upgraded_area:.2f} deg²")
    print(f"  ✓ Area difference: {abs(upgraded_area - original_area):.6f} deg²")
    print(f"  ✓ Pixel count: {upgraded_npoint_mask.n_valid:,} valid pixels")

# Baseline area comparison
npoint_gt0_area = hsp.HealSparseMap.read(f"{base_path}/coverage.hsp").get_valid_area(degrees=True)
print("\nBaseline areas for comparison:")
print(f"  npoint>0 (full coverage): {npoint_gt0_area:.2f} deg²")
print(f"  npoint>2 (analysis mask): {upgraded_area:.2f} deg²")
print(f"  Area lost by npoint>2 cut: {npoint_gt0_area - upgraded_area:.2f} deg²")

# %%
print("\nPART 3: Cumulative mask effect analysis")
print("=" * 50)
print("\nApplying quality masks sequentially to measure cumulative area loss...")

# Initialize cumulative analysis
cumulative_mask = upgraded_npoint_mask.copy()
cumulative_areas = [upgraded_area]
mask_names = ["npoint>2 cut"]

print(f"Starting area (npoint>2): {upgraded_area:.2f} deg²")

# Define mask application order and descriptions
mask_order = [64, 1024, 4, 2, 1]
mask_descriptions = {
    64: "r band",
    1024: "maximask", 
    4: "stars",
    2: "bright star halos",
    1: "faint star halos"
}

# Apply quality masks sequentially in specified order
for n_val in mask_order:
    # Quality masks have opposite definition - negate to get keep-pixel mask
    cumulative_mask = cumulative_mask & (~individual_masks[n_val])
    
    current_area = cumulative_mask.get_valid_area(degrees=True)
    area_lost_step = cumulative_areas[-1] - current_area
    total_area_lost = cumulative_areas[0] - current_area
    
    cumulative_areas.append(current_area)
    mask_names.append(f"+ mask_n{n_val} ({mask_descriptions[n_val]})")
    
    print(f"After applying mask_n{n_val}:")
    print(f"  Current area: {current_area:.2f} deg²")
    print(f"  Area lost this step: {area_lost_step:.2f} deg²")
    print(f"  Total area lost: {total_area_lost:.2f} deg²")
    print(f"  Fraction remaining: {current_area/cumulative_areas[0]:.3f}")

# Create summary DataFrame starting with npoint>0 baseline
table_areas = [npoint_gt0_area] + cumulative_areas
table_names = ["npoint>0 baseline"] + mask_names
step_losses = [0, npoint_gt0_area - upgraded_area] + [cumulative_areas[i] - cumulative_areas[i+1] 
                                                     for i in range(len(cumulative_areas)-1)]

# Calculate individual mask areas
mask_areas = [0, 0]  # npoint>0 baseline and npoint>2 cut don't have individual mask areas
for n_val in mask_order:
    mask_areas.append(individual_masks[n_val].get_valid_area(degrees=True))

cumulative_df = pd.DataFrame({
    'Mask_Applied': table_names,
    'Area_deg2': table_areas,
    'Area_Lost_Step': step_losses,
    'Mask_Area': mask_areas
})

print("\nCumulative Mask Effect Summary:")
print(cumulative_df.to_string(index=False, float_format='%.2f'))

# Area loss summary
npoint2_loss = npoint_gt0_area - cumulative_areas[0]
total_loss = npoint_gt0_area - cumulative_areas[-1]

print(f"\nArea loss summary relative to npoint>0 baseline ({npoint_gt0_area:.2f} deg²):")
print(f"  Area lost by npoint>2 cut: {npoint2_loss:.2f} deg² ({(npoint2_loss/npoint_gt0_area)*100:.1f}%)")
print(f"  Final area after all masks: {cumulative_areas[-1]:.2f} deg²")
print(f"  Total area lost from npoint>0: {total_loss:.2f} deg² ({(total_loss/npoint_gt0_area)*100:.1f}%)")

# %%