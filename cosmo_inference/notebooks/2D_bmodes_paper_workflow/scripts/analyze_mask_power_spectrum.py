#!/usr/bin/env python3
"""
Analyze power spectra of UNIONS pixel masks for CosmoCov integration.

This script:
1. Loads downgraded HEALPix masks
2. Calculates angular power spectra C_ℓ using healpy
3. Analyzes convergence across different nside values  
4. Generates plots for resolution comparison
5. Exports power spectra for CosmoCov integration

Author: Claude Code
"""

from datetime import datetime
import os

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import seaborn as sns
from typing import Dict, Optional, Tuple


# Apply plotting style consistent with project conventions
plt.style.use("/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/notebooks/2D_cosmic_shear_paper_plots/config/paper.mplstyle")


def load_healpix_mask(mask_path: str, verbose: bool = True) -> np.ndarray:
    """Load HEALPix mask from FITS file."""
    if verbose:
        print(f"Loading HEALPix mask: {mask_path}")
    
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    mask = hp.read_map(mask_path, verbose=False)
    nside = hp.npix2nside(len(mask))
    
    if verbose:
        n_valid = np.sum(mask > 0)
        coverage = n_valid / len(mask)
        print(f"  nside={nside}, valid_pixels={n_valid}, coverage={coverage:.1%}")
    
    return mask


def calculate_power_spectrum(mask: np.ndarray, lmax: Optional[int] = None, 
                           verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate angular power spectrum of mask using healpy.anafast."""
    nside = hp.npix2nside(len(mask))
    
    if lmax is None:
        lmax = 3 * nside - 1  # Natural maximum for HEALPix
    
    if verbose:
        print(f"  Calculating power spectrum up to lmax={lmax}")
    
    # Calculate power spectrum
    cl = hp.anafast(mask, lmax=lmax)
    ell = np.arange(len(cl))
    
    if verbose:
        print(f"  Power spectrum calculated: {len(cl)} multipoles")
    
    return ell, cl


def analyze_convergence(masks_data: Dict[str, Dict], verbose: bool = True) -> Dict:
    """Analyze power spectrum convergence across different nside values."""
    if verbose:
        print("\n=== Power Spectrum Convergence Analysis ===")
    
    results = {}
    nsides = sorted([data['nside'] for data in masks_data.values()])
    
    # Find common ell range for comparison
    min_lmax = min([3 * nside - 1 for nside in nsides])
    common_ell = np.arange(min_lmax + 1)
    
    if verbose:
        print(f"Comparing power spectra up to common lmax={min_lmax}")
    
    # Calculate power spectra for all masks
    power_spectra = {}
    for key, data in masks_data.items():
        mask = load_healpix_mask(data['mask_path'], verbose=False)
        ell, cl = calculate_power_spectrum(mask, lmax=min_lmax, verbose=False)
        power_spectra[key] = {'ell': ell, 'cl': cl, 'nside': data['nside']}
    
    # Analyze convergence
    results['power_spectra'] = power_spectra
    results['common_ell'] = common_ell
    results['convergence_metrics'] = {}
    
    # Compare consecutive nside values
    sorted_keys = sorted(power_spectra.keys(), key=lambda x: power_spectra[x]['nside'])
    
    for i in range(1, len(sorted_keys)):
        key1, key2 = sorted_keys[i-1], sorted_keys[i]
        cl1, cl2 = power_spectra[key1]['cl'], power_spectra[key2]['cl']
        
        # Calculate relative difference
        ell_range = slice(1, min(len(cl1), len(cl2)))  # Skip monopole
        rel_diff = np.abs(cl2[ell_range] - cl1[ell_range]) / (cl1[ell_range] + 1e-20)
        
        comparison_key = f"{power_spectra[key1]['nside']}_vs_{power_spectra[key2]['nside']}"
        results['convergence_metrics'][comparison_key] = {
            'mean_rel_diff': np.mean(rel_diff),
            'max_rel_diff': np.max(rel_diff),
            'ell_range': [1, len(rel_diff)]
        }
        
        if verbose:
            print(f"  {comparison_key}: mean_rel_diff={np.mean(rel_diff):.3f}, "
                  f"max_rel_diff={np.max(rel_diff):.3f}")
    
    return results


def plot_power_spectra(analysis_results: Dict, output_dir: str, verbose: bool = True):
    """Create comparative plots of mask power spectra."""
    if verbose:
        print("\n=== Creating Power Spectrum Plots ===")
    
    power_spectra = analysis_results['power_spectra']
    
    # Set up seaborn palette
    n_masks = len(power_spectra)
    sns.set_palette("husl", n_masks)
    
    # Create main comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Power spectra comparison (log-log)
    ax = axes[0, 0]
    for key, data in power_spectra.items():
        ell, cl = data['ell'], data['cl']
        nside = data['nside']
        
        # Skip monopole and dipole for cleaner plot
        ell_plot = ell[2:]
        cl_plot = cl[2:]
        
        ax.loglog(ell_plot, cl_plot, label=f'nside={nside}', linewidth=2)
    
    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(r'$C_\ell^{\rm mask}$')
    ax.set_title('Mask Power Spectra Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Power spectra comparison (linear ell, log C_l)
    ax = axes[0, 1] 
    for key, data in power_spectra.items():
        ell, cl = data['ell'], data['cl']
        nside = data['nside']
        
        ell_plot = ell[2:min(500, len(ell))]  # Focus on low-ell region
        cl_plot = cl[2:min(500, len(cl))]
        
        ax.semilogy(ell_plot, cl_plot, label=f'nside={nside}', linewidth=2)
    
    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(r'$C_\ell^{\rm mask}$') 
    ax.set_title('Low-$\ell$ Power Spectra')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Relative differences between consecutive nside values
    ax = axes[1, 0]
    convergence_metrics = analysis_results['convergence_metrics']
    
    for comparison_key, metrics in convergence_metrics.items():
        # Get the two masks being compared
        nside1_str, nside2_str = comparison_key.split('_vs_')
        nside1, nside2 = int(nside1_str), int(nside2_str)
        
        # Find corresponding power spectra
        key1 = next(k for k, v in power_spectra.items() if v['nside'] == nside1)
        key2 = next(k for k, v in power_spectra.items() if v['nside'] == nside2)
        
        cl1, cl2 = power_spectra[key1]['cl'], power_spectra[key2]['cl']
        ell_range = slice(1, min(len(cl1), len(cl2)))
        ell_vals = np.arange(1, min(len(cl1), len(cl2)))
        
        rel_diff = np.abs(cl2[ell_range] - cl1[ell_range]) / (cl1[ell_range] + 1e-20)
        
        ax.loglog(ell_vals, rel_diff, label=comparison_key, linewidth=2)
    
    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel('Relative Difference')
    ax.set_title('Power Spectrum Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    ax = axes[1, 1]
    nsides = sorted([data['nside'] for data in power_spectra.values()])
    
    # Calculate some summary statistics
    monopoles = [power_spectra[k]['cl'][0] for k in power_spectra 
                 if power_spectra[k]['nside'] in nsides]
    low_ell_power = [np.mean(power_spectra[k]['cl'][2:10]) for k in power_spectra
                     if power_spectra[k]['nside'] in nsides]
    
    ax.semilogx(nsides, monopoles, 'o-', label='Monopole', linewidth=2, markersize=8)
    ax.semilogx(nsides, low_ell_power, 's-', label='Low-$\ell$ mean (2-9)', linewidth=2, markersize=8)
    
    ax.set_xlabel('nside')
    ax.set_ylabel('Power')
    ax.set_title('Summary Statistics vs Resolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'mask_power_spectra_comparison.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print(f"Saved power spectrum comparison plot: {plot_path}")

    return plot_path


def export_power_spectra_for_cosmocov(analysis_results: Dict, output_dir: str, 
                                      verbose: bool = True) -> Dict[str, str]:
    """Export power spectra in format suitable for CosmoCov integration."""
    if verbose:
        print("\n=== Exporting Power Spectra for CosmoCov ===")
    
    power_spectra = analysis_results['power_spectra']
    export_paths = {}
    
    ps_output_dir = os.path.join(output_dir, 'power_spectra')
    os.makedirs(ps_output_dir, exist_ok=True)
    
    for key, data in power_spectra.items():
        ell, cl = data['ell'], data['cl']
        nside = data['nside']
        
        # Create output filename
        filename = f"cl_mask_nside{nside}.txt"
        filepath = os.path.join(ps_output_dir, filename)
        
        # Save in simple two-column format: ell, C_ell
        header = (f"# UNIONS mask power spectrum for nside={nside}\n"
                 f"# Generated by analyze_mask_power_spectrum.py on {datetime.now().strftime('%Y-%m-%d')}\n"
                 f"# Format: ell C_ell\n"
                 f"# lmax = {len(cl)-1}")
        
        np.savetxt(filepath, np.column_stack([ell, cl]), 
                   fmt='%d %.12e', header=header)
        
        export_paths[key] = filepath
        
        if verbose:
            print(f"  Exported {key}: {filepath}")
    
    return export_paths


def main():
    """Main analysis function."""
    # Snakemake script execution only (no interactive mode)
    from snakemake.script import snakemake

    # Parameters must be provided explicitly
    params = snakemake.params
    for key in ("target_nsides", "output_dir"):
        if key not in params:
            raise KeyError(f"Missing required Snakemake param '{key}'")

    target_nsides = params["target_nsides"]
    output_dir = params["output_dir"]
    
    # Ensure output directory is absolute
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)
    
    verbose = True
    print("=== UNIONS Mask Power Spectrum Analysis ===")
    print(f"Target nside values: {target_nsides}")
    print(f"Output directory: {output_dir}")
    
    # Prepare mask data dictionary
    masks_data = {}
    for nside in target_nsides:
        mask_path = os.path.join(output_dir, f"mask_nside{nside}.fits")
        key = f"nside_{nside}"
        masks_data[key] = {
            'nside': nside,
            'mask_path': mask_path
        }
    
    # Analyze power spectra and convergence
    analysis_results = analyze_convergence(masks_data, verbose=verbose)
    
    # Create comparison plots
    plot_path = plot_power_spectra(analysis_results, output_dir, verbose=verbose)
    
    # Export power spectra for CosmoCov
    export_paths = export_power_spectra_for_cosmocov(analysis_results, output_dir, verbose=verbose)
    
    # Print summary
    print("\n=== Analysis Summary ===")
    print(f"Power spectra calculated for {len(target_nsides)} nside values")
    print(f"Comparison plot saved: {plot_path}")
    print("Power spectra exported for CosmoCov integration:")
    for key, path in export_paths.items():
        print(f"  {key}: {os.path.basename(path)}")

    # Provide convergence recommendations
    convergence_metrics = analysis_results['convergence_metrics']
    if convergence_metrics:
        print("\n=== Convergence Recommendations ===")
        for comparison, metrics in convergence_metrics.items():
            if metrics['mean_rel_diff'] < 0.01:
                print(f"✓ {comparison}: Well converged (mean rel. diff: {metrics['mean_rel_diff']:.3f})")
            elif metrics['mean_rel_diff'] < 0.05:
                print(f"~ {comparison}: Moderately converged (mean rel. diff: {metrics['mean_rel_diff']:.3f})")
            else:
                print(f"⚠ {comparison}: Poor convergence (mean rel. diff: {metrics['mean_rel_diff']:.3f})")
    
    print("\nPower spectrum analysis completed successfully!")


if __name__ == "__main__":
    main()
