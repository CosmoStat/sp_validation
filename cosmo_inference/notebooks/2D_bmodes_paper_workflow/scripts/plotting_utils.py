"""Shared plotting utilities for claims scripts."""

import matplotlib.scale as mscale
import matplotlib.ticker as ticker
import matplotlib.transforms as mtransforms
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns
from scipy import stats


# Shared constants
PAPER_MPLSTYLE = "/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/notebooks/2D_cosmic_shear_paper_plots/config/paper.mplstyle"
FIG_WIDTH_FULL = 7.24  # Two-column A&A format (textwidth)
FIG_WIDTH_SINGLE = 3.54  # Single-column A&A format (columnwidth)
MARKER_STYLES = ["o", "s", "D", "^"]

# Default errorbar styling for version comparison plots
ERRORBAR_DEFAULTS = {
    "markeredgewidth": 0.5,
    "markersize": 4,
    "capsize": 2,
    "capthick": 0.8,
    "elinewidth": 0.8,
}

# Default version box styling
VERSION_BOX_DEFAULTS = {
    "edge_color": "black",
    "edge_linewidth": 0.8,
    "fiducial_line_color": "black",
    "fiducial_line_width": 1.0,
}


def make_pte_colormap(low=0.05, high=0.95, gradient_range=(0.15, 0.85)):
    """Create discrete colormap with sharp breaks at significance thresholds.

    Provides clear visual distinction between passing (middle) and failing
    (extreme) PTE regions without requiring contour overlays.

    Parameters
    ----------
    low, high : float
        PTE thresholds. Solid blue below `low`, solid red above `high`.
    gradient_range : tuple
        Range of vlag colormap (0-1) to use for the gradient portion.
        Narrower range = shallower gradient = sharper contrast at boundaries.

    Returns
    -------
    cmap : LinearSegmentedColormap
        Discrete colormap with sharp breaks at thresholds.
    """
    vlag = sns.color_palette("vlag", as_cmap=True)

    # Solid regions use extreme vlag colors for sharp contrast
    solid_blue = vlag(0.0)
    solid_red = vlag(1.0)

    # Build colormap: [0, low] solid blue, [low, high] compressed gradient, [high, 1] solid red
    n_total = 256
    n_low = int(low * n_total)
    n_high = int((1 - high) * n_total)
    n_mid = n_total - n_low - n_high

    # Gradient samples from compressed range of vlag
    g_lo, g_hi = gradient_range
    gradient_colors = [vlag(g_lo + (g_hi - g_lo) * i / (n_mid - 1)) for i in range(n_mid)]

    all_colors = [solid_blue] * n_low + gradient_colors + [solid_red] * n_high
    cmap = LinearSegmentedColormap.from_list("pte_discrete", all_colors, N=256)
    cmap.set_bad(color="lightgray")
    return cmap


def compute_chi2_pte(data, covariance, n_samples=None):
    """Compute chi-squared and PTE for null test.

    Parameters
    ----------
    data : array_like
        Data vector (e.g., B-mode signal).
    covariance : array_like
        Covariance matrix.
    n_samples : int, optional
        Number of samples used to estimate the covariance (e.g., MC samples
        or jackknife patches). When provided, the Hartlap correction factor
        (n_samples - n_bins - 2) / (n_samples - 1) is applied to debias the
        inverse covariance estimate. Leave as None for analytical covariances.

    Returns
    -------
    chi2 : float
        Chi-squared value.
    pte : float
        Probability to exceed (survival function).
    dof : int
        Degrees of freedom (length of data).
    """
    chi2 = float(data @ np.linalg.solve(covariance, data))
    dof = len(data)
    if n_samples is not None:
        hartlap_factor = (n_samples - dof - 2) / (n_samples - 1)
        chi2 *= hartlap_factor
    pte = stats.chi2.sf(chi2, dof)
    return chi2, pte, dof


class SquareRootScale(mscale.ScaleBase):
    """Square root scale for x-axis (matches bandpower binning)."""

    name = "squareroot"

    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self, axis, **kwargs)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(0.0, vmin), vmax

    class SquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return np.array(a) ** 0.5

        def inverted(self):
            return SquareRootScale.InvertedSquareRootTransform()

    class InvertedSquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return np.array(a) ** 2

        def inverted(self):
            return SquareRootScale.SquareRootTransform()

    def get_transform(self):
        return self.SquareRootTransform()


# Register at import time
mscale.register_scale(SquareRootScale)


def version_label(version, version_labels):
    """Get human-readable label for version from config.

    Parameters
    ----------
    version : str
        Version string (e.g., "SP_v1.4.11.2_leak_corr").
    version_labels : dict
        Mapping from version strings to human-readable labels.

    Returns
    -------
    str
        Human-readable label, or cleaned version string if not in mapping.
    """
    return version_labels.get(version, version.replace("SP_", "").replace("_leak_corr", ""))


def find_fiducial_index(datasets, fiducial_version, key="version"):
    """Find index of fiducial version in datasets list.

    Parameters
    ----------
    datasets : list of dict
        List of dataset dictionaries with version key.
    fiducial_version : str
        Version string to find.
    key : str
        Key to use for matching (default: "version").

    Returns
    -------
    int
        Index of fiducial version, or 0 if not found.
    """
    return next(
        (i for i, d in enumerate(datasets) if d[key] == fiducial_version),
        0
    )


def get_version_alpha(version, fiducial_version, plotting_config):
    """Get alpha value for version based on whether it's fiducial.

    Parameters
    ----------
    version : str
        Version string.
    fiducial_version : str
        Fiducial version string for comparison.
    plotting_config : dict
        Plotting configuration with 'version_alpha' key containing
        'fiducial' and 'comparison' values.

    Returns
    -------
    float
        Alpha value (1.0 for fiducial, 0.5 for others by default).
    """
    version_alpha = plotting_config.get("version_alpha", {})
    if version == fiducial_version:
        return version_alpha.get("fiducial", 1.0)
    return version_alpha.get("comparison", 0.5)


def get_box_style(box_style=None):
    """Get version box styling with defaults.

    Parameters
    ----------
    box_style : dict, optional
        User-provided style overrides.

    Returns
    -------
    dict
        Style dict with keys: edge_color, edge_linewidth,
        fiducial_line_color, fiducial_line_width.
    """
    style = box_style or {}
    return {
        "edge_color": style.get("edge_color", VERSION_BOX_DEFAULTS["edge_color"]),
        "edge_linewidth": style.get("edge_linewidth", VERSION_BOX_DEFAULTS["edge_linewidth"]),
        "fiducial_line_color": style.get("fiducial_line_color", VERSION_BOX_DEFAULTS["fiducial_line_color"]),
        "fiducial_line_width": style.get("fiducial_line_width", VERSION_BOX_DEFAULTS["fiducial_line_width"]),
    }


def draw_normalized_version_box(ax, x_left, x_right, y_vals, fiducial_val, style):
    """Draw a single version box for normalized data.

    Draws a rectangle spanning the range of values +/- 1 sigma (where sigma=1
    by construction for normalized data), with a fiducial reference line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    x_left, x_right : float
        Left and right edges of box.
    y_vals : array-like
        Normalized y values for all versions at this bin.
    fiducial_val : float
        Fiducial version's y value for the reference line.
    style : dict
        Styling dict from get_box_style().
    """
    y_vals = np.asarray(y_vals)
    box_bottom = y_vals.min()
    box_top = y_vals.max()

    rect = Rectangle(
        (x_left, box_bottom),
        x_right - x_left,
        box_top - box_bottom,
        facecolor='none',
        edgecolor=style["edge_color"],
        linewidth=style["edge_linewidth"],
        zorder=1,
    )
    ax.add_patch(rect)

    ax.hlines(
        fiducial_val, x_left, x_right,
        colors=style["fiducial_line_color"],
        linewidth=style["fiducial_line_width"],
        zorder=1
    )


def draw_normalized_boxes_log_scale(ax, x_centers, datasets, y_norm_key, fiducial_idx,
                                     x_offset_range, box_style=None):
    """Draw version boxes for log-scale x-axis (e.g., theta).

    For each bin, draws a box spanning the range of values across versions,
    with a horizontal line at the fiducial version's value.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    x_centers : array-like
        Bin centers (theta values).
    datasets : list of dict
        Dataset dicts containing y_norm_key.
    y_norm_key : str
        Key for normalized y values in datasets.
    fiducial_idx : int
        Index of fiducial version in datasets.
    x_offset_range : tuple
        (min, max) multiplicative offsets for data points. Box covers
        this range plus 10% padding.
    box_style : dict, optional
        Style overrides for boxes.
    """
    style = get_box_style(box_style)
    offset_min, offset_max = x_offset_range
    padding = 0.25 * (offset_max - offset_min)
    box_left_factor = offset_min - padding
    box_right_factor = offset_max + padding

    for i, x_i in enumerate(x_centers):
        y_vals = np.array([data[y_norm_key][i] for data in datasets])
        x_left = x_i * box_left_factor
        x_right = x_i * box_right_factor
        draw_normalized_version_box(ax, x_left, x_right, y_vals, y_vals[fiducial_idx], style)


def draw_normalized_boxes_linear_scale(ax, x_centers, datasets, y_norm_key, fiducial_idx,
                                        x_offsets, box_style=None):
    """Draw version boxes for linear-scale x-axis (e.g., COSEBIS modes).

    For each bin, draws a box spanning the range of values across versions,
    with a horizontal line at the fiducial version's value.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    x_centers : array-like
        Bin centers (mode numbers).
    datasets : list of dict
        Dataset dicts containing y_norm_key.
    y_norm_key : str
        Key for normalized y values in datasets.
    fiducial_idx : int
        Index of fiducial version in datasets.
    x_offsets : array-like
        Array of x-offsets used for jittering data points. Box width covers
        this range plus 25% padding.
    box_style : dict, optional
        Style overrides for boxes.
    """
    style = get_box_style(box_style)
    x_offsets = np.asarray(x_offsets)
    box_half_width = np.max(np.abs(x_offsets)) * 1.7

    for i, x_i in enumerate(x_centers):
        y_vals = np.array([data[y_norm_key][i] for data in datasets])
        x_left = x_i - box_half_width
        x_right = x_i + box_half_width
        draw_normalized_version_box(ax, x_left, x_right, y_vals, y_vals[fiducial_idx], style)


def draw_normalized_boxes_ell_scale(ax, ell, ell_widths, datasets, y_norm_key, fiducial_idx,
                                     jitter_fraction, n_versions, box_style=None):
    """Draw version boxes for ell-space plots with variable bin widths.

    For each multipole bin, draws a box spanning the range of values across
    versions, with a horizontal line at the fiducial version's value.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    ell : array-like
        Multipole bin centers.
    ell_widths : array-like
        Width of each ell bin.
    datasets : list of dict
        Dataset dicts containing y_norm_key.
    y_norm_key : str
        Key for normalized y values in datasets.
    fiducial_idx : int
        Index of fiducial version in datasets.
    jitter_fraction : float
        Jitter fraction used for offsetting data points.
    n_versions : int
        Number of versions being plotted.
    box_style : dict, optional
        Style overrides for boxes.
    """
    style = get_box_style(box_style)
    # Max jitter is ((n-1)/2) * jitter_fraction; add 15% padding
    box_half_width_factor = ((n_versions - 1) / 2) * jitter_fraction * 1.4

    for i, ell_i in enumerate(ell):
        y_vals = np.array([data[y_norm_key][i] for data in datasets])
        half_width = ell_widths[i] * box_half_width_factor
        x_left = ell_i - half_width
        x_right = ell_i + half_width
        draw_normalized_version_box(ax, x_left, x_right, y_vals, y_vals[fiducial_idx], style)


def extract_version_number(version_string):
    """Extract short version number from full version string.

    Examples:
        SP_v1.4.5_leak_corr -> v1.4.5
        SP_v1.4.11.2_leak_corr -> v1.4.11.2
        SP_v1.4.5 -> v1.4.5

    Parameters
    ----------
    version_string : str
        Full version string (e.g., "SP_v1.4.5_leak_corr").

    Returns
    -------
    str
        Short version number (e.g., "v1.4.5").
    """
    import re
    match = re.search(r'(v[\d.]+)', version_string)
    return match.group(1) if match else version_string


def iter_version_figures(version_labels, fiducial_version):
    """Iterate over the 9 figure specs for per-version data vector plots.

    Produces specs for:
    - 1 paper figure: fiducial version, leak-corrected, no title
    - 8 dashboard figures: all versions × (corrected + uncorrected), with titles

    Parameters
    ----------
    version_labels : dict
        Mapping from leak-corrected version strings to human-readable labels.
        E.g., {"SP_v1.4.5_leak_corr": "Initial", ...}
    fiducial_version : str
        Fiducial version string (e.g., "SP_v1.4.11.2_leak_corr").

    Yields
    ------
    dict
        Figure specification with keys:
        - filename: output filename (e.g., "figure_v1.4.5.png")
        - version_leak_corr: leak-corrected version string
        - version_uncorr: uncorrected version string
        - leak_corrected: bool
        - title: title string or None (for paper figure)
        - is_paper_figure: bool
    """
    # Paper figure: fiducial, leak-corrected, no title
    yield {
        "filename": "figure.png",
        "version_leak_corr": fiducial_version,
        "version_uncorr": fiducial_version.replace("_leak_corr", ""),
        "leak_corrected": True,
        "title": None,
        "is_paper_figure": True,
    }

    # Dashboard figures: all versions × (corrected + uncorrected)
    # Sort by version string length descending to ensure proper matching
    for version_lc in sorted(version_labels.keys(), key=lambda v: -len(v)):
        version_num = extract_version_number(version_lc)
        label = version_labels[version_lc]
        version_uncorr = version_lc.replace("_leak_corr", "")

        # Leak-corrected figure
        yield {
            "filename": f"figure_{version_num}.png",
            "version_leak_corr": version_lc,
            "version_uncorr": version_uncorr,
            "leak_corrected": True,
            "title": label,
            "is_paper_figure": False,
        }

        # Uncorrected figure
        yield {
            "filename": f"figure_{version_num}_uncorrected.png",
            "version_leak_corr": version_lc,
            "version_uncorr": version_uncorr,
            "leak_corrected": False,
            "title": f"{label} (uncorrected)",
            "is_paper_figure": False,
        }


def get_version_figure_outputs(version_labels):
    """Get list of all per-version figure filenames for Snakemake outputs.

    Parameters
    ----------
    version_labels : dict
        Mapping from leak-corrected version strings to human-readable labels.

    Returns
    -------
    list of str
        List of 9 filenames (1 paper + 8 dashboard figures).
    """
    filenames = ["figure.png"]
    for version_lc in sorted(version_labels.keys(), key=lambda v: -len(v)):
        version_num = extract_version_number(version_lc)
        filenames.append(f"figure_{version_num}.png")
        filenames.append(f"figure_{version_num}_uncorrected.png")
    return filenames
