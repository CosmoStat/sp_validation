"""Shared plotting utilities for claims scripts."""

import matplotlib.scale as mscale
import matplotlib.ticker as ticker
import matplotlib.transforms as mtransforms
import numpy as np
from scipy import stats


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
