"""
Snakemake rules for UNIONS pixel mask processing and analysis.

NOTE: This subsystem is exploratory/infrastructure. The outputs are NOT currently
wired into the main covariance workflow, which gives OneCovariance each
catalogue's footprint map (its cat_config `mask:` key). See covariance.md spec
for details.

This file contains rules for:
- Processing HealSparse masks to individual nside values (parallelizable)
- Calculating effective survey areas
- Computing mask power spectra for the OneCovariance survey mask
- Generating comparison plots and analysis
"""
from datetime import datetime

# Wildcard constraints centralized in Snakefile

# Paths from config and Snakefile
SOURCE_MASK_FILE = config["pixel_mask"]["source_file"]
MASK_OUTPUT_DIR = "output/masks"

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
                "processing_date": datetime.now().strftime("%Y-%m-%d"),
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
    """Calculate angular power spectrum for a single mask.

    Loads downgraded HEALPix mask, computes C_ell, exports for OneCovariance.
    """
    input:
        mask=f"{MASK_OUTPUT_DIR}/mask_nside{{nside}}.fits"
    output:
        power_spectrum=f"{MASK_OUTPUT_DIR}/power_spectra/cl_mask_nside{{nside}}.txt"
    threads: 1
    resources:
        mem_mb=4000,
        disk_mb=500,
        runtime=10
    script:
        "../scripts/analyze_mask_power_spectrum.py"


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
            "## OneCovariance Integration",
            "",
            "The following power spectrum files are ready for OneCovariance:",
            ""
        ])
        
        # List power spectrum files
        for ps_file in input.power_spectra:
            ps_path = Path(ps_file)
            nside = ps_path.stem.split('_')[-1].replace('nside', '')
            report_content.append(f"- **nside {nside}**: `{ps_path.name}`")
        
        report_content.extend([
            "",
            "To use one in OneCovariance, put it in `mask_directory` and name it in `mask_file_lensing`.",
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
        Path(output.report).parent.mkdir(parents=True, exist_ok=True)
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
        f"{MASK_OUTPUT_DIR}/area_comparison_report.md"


# Fast rules that don't need cluster resources
localrules:
    combine_area_summaries,
    effective_area_comparison,
    masks_only,
    masks_full_analysis,
