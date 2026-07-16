"""Plot GAIA stars on sky maps by magnitude bins."""

import glob
import os
import re

import matplotlib.pyplot as plt
from astropy.table import Table
from cs_util.plots import FootprintPlotter


def load_gaia_fits(file_path):
    """Load GAIA FITS file and return RA, DEC arrays.

    Parameters
    ----------
    file_path : str
        Path to FITS file

    Returns
    -------
    ra : np.ndarray
        Right ascension values
    dec : np.ndarray
        Declination values
    n_objects : int
        Number of objects in file
    """
    table = Table.read(file_path, format="fits")
    print(f"Loaded {len(table)} objects from {file_path}")
    return table["ra"], table["dec"], len(table)


def extract_label_from_filename(filename):
    """Extract magnitude information from filename to create label.

    Parameters
    ----------
    filename : str
        FITS filename

    Returns
    -------
    str
        Label describing the magnitude bin
    """
    basename = os.path.basename(filename)

    # Match pattern: g_smaller_XX.X
    match = re.search(r"g_smaller_([0-9.]+)", basename)
    if match:
        mag = match.group(1)
        return f"G < {mag} (bright)"

    # Match pattern: g_in_XX.X_YY.Y
    match = re.search(r"g_in_([0-9.]+)_([0-9.]+)", basename)
    if match:
        mag1, mag2 = match.group(1), match.group(2)
        return f"{mag1} ≤ G < {mag2} (medium)"

    # Match pattern: g_larger_XX.X
    match = re.search(r"g_larger_([0-9.]+)", basename)
    if match:
        mag = match.group(1)
        return f"G ≥ {mag} (faint)"

    # Fallback
    return basename.replace(".fits", "")


def main():
    """Create sky plots for GAIA stars in three magnitude bins."""

    # Auto-detect GAIA files using patterns
    pattern_smaller = "gaia_stars_g_smaller_*.fits"
    pattern_in = "gaia_stars_g_in_*.fits"
    pattern_larger = "gaia_stars_g_larger_*.fits"

    files = []
    files.extend(sorted(glob.glob(pattern_smaller)))
    files.extend(sorted(glob.glob(pattern_in)))
    files.extend(sorted(glob.glob(pattern_larger)))

    if not files:
        print("ERROR: No GAIA FITS files found matching patterns:")
        print(f"  - {pattern_smaller}")
        print(f"  - {pattern_in}")
        print(f"  - {pattern_larger}")
        return

    print(f"Found {len(files)} GAIA FITS files:")
    for f in files:
        print(f"  - {f}")
    print()

    # Generate labels from filenames
    labels = [extract_label_from_filename(f) for f in files]

    # Initialize the footprint plotter
    plotter = FootprintPlotter(nside_coverage=32, nside_map=2048)

    # Process each magnitude bin
    hsp_maps = []
    for file_path, label in zip(files, labels):
        print(f"Processing: {label}")

        # Load data
        ra, dec, n_objects = load_gaia_fits(file_path)

        # Create healsparse map
        hsp_map = plotter.create_hsp_map(ra, dec)
        hsp_maps.append(hsp_map)

        # Plot all regions (NGC, SGC, fullsky)
        plotter.plot_all_regions(
            hsp_map, outbase=f"gaia_sky_{file_path.replace('.fits', '')}"
        )
        print(f"Created plots for {label}")

    # Create combined plot showing all bins
    n_files = len(files)
    if n_files > 0:
        print("Creating combined plot")

        fig, axes = plt.subplots(1, n_files, figsize=(10 * n_files, 10))

        # Handle case of single file
        if n_files == 1:
            axes = [axes]

        for idx, (hsp_map, label) in enumerate(zip(hsp_maps, labels)):
            # Use fullsky region parameters
            region = plotter._regions["fullsky"]

            plotter.plot_area(
                hsp_map,
                ra_0=region["ra_0"],
                extend=region["extend"],
                vmax=region["vmax"],
                title=label,
            )

        plt.tight_layout()
        plt.savefig("gaia_sky_combined_all_bins.png", dpi=150, bbox_inches="tight")
        print("Saved combined plot: gaia_sky_combined_all_bins.png")

    print("All plots created successfully!")


if __name__ == "__main__":
    main()
