"""
Snakemake rules for UNIONS pixel mask processing and analysis.

This file contains rules for:
- Processing HealSparse masks to individual nside values (parallelizable)
- Calculating effective survey areas  
- Computing mask power spectra for CosmoCov integration
- Generating comparison plots and analysis

Author: Claude Code
Date: 2025-08-18
"""

# Wildcard constraints centralized in Snakefile

# Define paths directly in Snakefile 
SOURCE_MASK_FILE = "/n17data/UNIONS/WL/masks/mask_r_nside131072.hsp"
MASK_OUTPUT_DIR = "output/masks"
COSMOCOV_MASK_DIR = "/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/data/masks"

# Get target nside values from config (no fallbacks)
target_nsides = config["pixel_mask"]["target_nsides"]


rule process_pixel_mask:
    """
    Process HealSparse pixel mask for a single nside value.
    
    This rule:
    1. Loads the high-resolution HealSparse mask
    2. Downgrades to the specified nside value
    3. Converts to standard HEALPix format
    4. Calculates effective survey area for this resolution
    """
    input:
        mask=SOURCE_MASK_FILE
    output:
        mask=f"{MASK_OUTPUT_DIR}/mask_nside{{nside}}.fits",
        summary=f"{MASK_OUTPUT_DIR}/area_nside{{nside}}.yaml"
    params:
        nside=lambda wildcards: int(wildcards.nside)
    threads: 1
    resources:
        mem_mb=8000,  # HealSparse processing can be memory intensive
        disk_mb=1000,
        runtime=15    # minutes
    script:
        "../scripts/process_mask.py"


rule combine_area_summaries:
    """
    Combine individual area summaries into a single comprehensive summary.
    """
    input:
        summaries=expand(f"{MASK_OUTPUT_DIR}/area_nside{{nside}}.yaml", nside=target_nsides)
    output:
        combined=f"{MASK_OUTPUT_DIR}/effective_areas.yaml"
    threads: 1
    resources:
        mem_mb=1000,
        runtime=5
    run:
        import yaml
        
        combined_data = {
            "mask_processing_summary": {
                "source_mask": "UNIONS HealSparse mask",
                "processing_date": "2025-08-18",
                "masks": {}
            }
        }
        
        # Load and combine all individual summaries
        for summary_file in input.summaries:
            with open(summary_file, 'r') as f:
                data = yaml.safe_load(f)
            
            # Extract the mask info (assuming single mask per file)
            mask_info = data["mask_processing_summary"]["masks"]
            combined_data["mask_processing_summary"]["masks"].update(mask_info)
        
        # Write combined summary
        with open(output.combined, 'w') as f:
            yaml.dump(combined_data, f, default_flow_style=False, sort_keys=True)
        
        print(f"Combined area summaries: {output.combined}")


rule mask_power_spectrum:
    """
    Calculate angular power spectrum for a single mask.
    
    This rule:
    1. Loads the downgraded HEALPix mask
    2. Calculates power spectrum C_ℓ using healpy.anafast
    3. Exports power spectrum for CosmoCov integration
    """
    input:
        mask=f"{MASK_OUTPUT_DIR}/mask_nside{{nside}}.fits"
    output:
        power_spectrum=f"{MASK_OUTPUT_DIR}/power_spectra/cl_mask_nside{{nside}}.txt"
    params:
        nside=lambda wildcards: int(wildcards.nside),
        target_nsides=target_nsides,
        output_dir=MASK_OUTPUT_DIR
    threads: 1
    resources:
        mem_mb=4000,
        disk_mb=500,
        runtime=10    # minutes
    script:
        "../scripts/analyze_mask_power_spectrum.py"


rule mask_power_spectra_comparison:
    """
    Create comparison plots and convergence analysis across all nside values.
    """
    input:
        masks=expand(f"{MASK_OUTPUT_DIR}/mask_nside{{nside}}.fits", nside=target_nsides),
        power_spectra=expand(f"{MASK_OUTPUT_DIR}/power_spectra/cl_mask_nside{{nside}}.txt", nside=target_nsides),
        summary=f"{MASK_OUTPUT_DIR}/effective_areas.yaml"
    output:
        comparison_plot=f"{MASK_OUTPUT_DIR}/mask_power_spectra_comparison.png"
    params:
        target_nsides=target_nsides,
        output_dir=MASK_OUTPUT_DIR
    threads: 1
    resources:
        mem_mb=4000,
        disk_mb=1000,
        runtime=10
    script:
        "../scripts/create_mask_comparison_plots.py"


rule effective_area_comparison:
    """
    Compare effective areas across different mask resolutions.
    
    This rule creates a summary report comparing:
    1. Effective areas for different nside values
    2. How areas change with mask resolution
    3. Recommendations for optimal nside choice
    """
    input:
        summary=f"{MASK_OUTPUT_DIR}/effective_areas.yaml",
        power_spectra=expand(f"{MASK_OUTPUT_DIR}/power_spectra/cl_mask_nside{{nside}}.txt", nside=target_nsides)
    output:
        report=f"{MASK_OUTPUT_DIR}/area_comparison_report.md"
    params:
        target_nsides=target_nsides
    threads: 1
    resources:
        mem_mb=1000,
        runtime=5
    run:
        import yaml
        import numpy as np
        from pathlib import Path
        
        # Load effective area summary
        with open(input.summary, 'r') as f:
            summary_data = yaml.safe_load(f)
        
        masks_info = summary_data["mask_processing_summary"]["masks"]
        
        # Generate markdown report
        report_content = [
            "# UNIONS Pixel Mask Analysis Report",
            "",
            f"**Generated:** {summary_data['mask_processing_summary']['processing_date']}",
            f"**Source mask:** {summary_data['mask_processing_summary']['source_mask']}",
            "",
            "## Effective Area Summary",
            "",
            "| nside | Effective Area (deg²) | Coverage Fraction | Valid Pixels | Mean Mask Value |",
            "|-------|----------------------|-------------------|--------------|-----------------|"
        ]
        
        # Add table rows
        for nside_key in sorted(masks_info.keys(), key=lambda x: masks_info[x]['nside']):
            data = masks_info[nside_key]
            report_content.append(
                f"| {data['nside']} | {data['effective_area_deg2']:.2f} | "
                f"{data['coverage_fraction']:.1%} | {data['n_valid_pixels']:,} | "
                f"{data['mean_mask_value']:.4f} |"
            )
        
        report_content.extend([
            "",
            "## Resolution Analysis",
            "",
            "### Area Convergence",
            ""
        ])
        
        # Calculate area differences between resolutions
        sorted_data = sorted(masks_info.items(), key=lambda x: x[1]['nside'])
        
        for i in range(1, len(sorted_data)):
            prev_key, prev_data = sorted_data[i-1]
            curr_key, curr_data = sorted_data[i]
            
            area_diff = curr_data['effective_area_deg2'] - prev_data['effective_area_deg2']
            rel_diff = area_diff / prev_data['effective_area_deg2'] * 100
            
            report_content.append(
                f"- **nside {prev_data['nside']} → {curr_data['nside']}**: "
                f"{area_diff:+.2f} deg² ({rel_diff:+.1f}%)"
            )
        
        report_content.extend([
            "",
            "## CosmoCov Integration",
            "",
            "The following power spectrum files are ready for CosmoCov integration:",
            ""
        ])
        
        # List power spectrum files
        for ps_file in input.power_spectra:
            ps_path = Path(ps_file)
            nside = ps_path.stem.split('_')[-1].replace('nside', '')
            report_content.append(f"- **nside {nside}**: `{ps_path.name}`")
        
        report_content.extend([
            "",
            "To use these in CosmoCov, set the `c_footprint_file` parameter to point to the appropriate power spectrum file.",
            "",
            "## Recommendations",
            ""
        ])
        
        # Add basic recommendations
        highest_nside = max(masks_info.keys(), key=lambda x: masks_info[x]['nside'])
        lowest_nside = min(masks_info.keys(), key=lambda x: masks_info[x]['nside'])
        
        report_content.extend([
            f"- **For high accuracy**: Use nside={masks_info[highest_nside]['nside']}",
            f"- **For computational efficiency**: Use nside={masks_info[lowest_nside]['nside']}",
            "- **Recommended**: Check power spectrum convergence plot to determine optimal balance",
            "",
            "---",
            "*Generated by UNIONS pure E/B mode analysis workflow*"
        ])
        
        # Write report
        os.makedirs(os.path.dirname(output.report), exist_ok=True)
        with open(output.report, 'w') as f:
            f.write('\n'.join(report_content))
        
        print(f"Generated area comparison report: {output.report}")


# Convenience rules for different processing stages
rule masks_only:
    """Process masks and calculate effective areas only."""
    input:
        f"{MASK_OUTPUT_DIR}/effective_areas.yaml"


rule masks_full_analysis:
    """Complete mask processing and analysis pipeline."""
    input:
        f"{MASK_OUTPUT_DIR}/effective_areas.yaml",
        f"{MASK_OUTPUT_DIR}/mask_power_spectra_comparison.png",
        f"{MASK_OUTPUT_DIR}/area_comparison_report.md"


# Integration with existing covariance workflow
rule prepare_mask_for_cosmocov:
    """
    Prepare mask power spectra for integration with CosmoCov covariance calculations.
    
    This rule creates copies of mask power spectra in the CosmoCov data directory.
    """
    input:
        power_spectra=expand(f"{MASK_OUTPUT_DIR}/power_spectra/cl_mask_nside{{nside}}.txt", nside=target_nsides),
        report=f"{MASK_OUTPUT_DIR}/area_comparison_report.md"
    output:
        cosmocov_ready=f"{MASK_OUTPUT_DIR}/cosmocov_integration_ready.txt"
    params:
        cosmocov_mask_dir=COSMOCOV_MASK_DIR
    threads: 1
    resources:
        mem_mb=500,
        runtime=2
    run:
        import os
        import shutil
        from pathlib import Path
        
        # Create CosmoCov mask directory if it doesn't exist
        cosmocov_dir = Path(params.cosmocov_mask_dir)
        cosmocov_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy power spectrum files to CosmoCov directory
        copied_files = []
        for ps_file in input.power_spectra:
            src_path = Path(ps_file)
            dst_path = cosmocov_dir / src_path.name
            
            shutil.copy2(src_path, dst_path)
            copied_files.append(str(dst_path))
            print(f"Copied {src_path.name} to CosmoCov directory")
        
        # Create integration status file
        with open(output.cosmocov_ready, 'w') as f:
            f.write("UNIONS Mask Power Spectra - CosmoCov Integration Ready\n")
            f.write(f"Generated: 2025-08-18\n")
            f.write(f"Source: {MASK_OUTPUT_DIR}\n")
            f.write(f"Destination: {params.cosmocov_mask_dir}\n")
            f.write("\nCopied files:\n")
            for file_path in copied_files:
                f.write(f"- {file_path}\n")
            f.write("\nTo use in CosmoCov:\n")
            f.write("Set c_footprint_file parameter to the appropriate power spectrum file path.\n")
        
        print(f"CosmoCov integration prepared: {output.cosmocov_ready}")
