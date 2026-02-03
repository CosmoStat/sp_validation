"""Shared plotting utilities for claims scripts."""

import matplotlib.scale as mscale
import matplotlib.ticker as ticker
import matplotlib.transforms as mtransforms
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import seaborn as sns
from scipy import stats


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


def compute_chi2_pte(data, covariance):
    """Compute chi-squared and PTE for null test.

    Parameters
    ----------
    data : array_like
        Data vector (e.g., B-mode signal).
    covariance : array_like
        Covariance matrix.

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
