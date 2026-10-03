"""Scalar S8 tension divides the signed mean difference by the quadrature sum of
symmetrized error widths. Swapping analyses reverses its sign; equal means give
zero, and common positive scaling leaves it unchanged.
"""

import pytest

from cosmo_inference.scripts.chain_postprocessing import get_sigma_tension

pytestmark = [pytest.mark.fast, pytest.mark.decision("inference.posterior_summary")]


def test_tension_combines_independent_errors_in_quadrature():
    """The rtol=1e-12 and atol=1e-14 tolerance has deterministic false-alarm bound 0."""
    # Positive error widths average to 0.03 and 0.04; quadrature gives the 3-4-5
    # denominator 0.05 and signed tension +2.
    inputs = (0.8, 0.02, 0.04, 0.7, 0.03, 0.05)

    assert get_sigma_tension(*inputs) == pytest.approx(2.0, rel=1e-12, abs=1e-14)
    assert get_sigma_tension(*inputs[3:], *inputs[:3]) == pytest.approx(
        -2.0, rel=1e-12, abs=1e-14
    )
    assert get_sigma_tension(0.8, 0.02, 0.04, 0.8, 0.03, 0.05) == pytest.approx(
        0.0, rel=1e-12, abs=1e-14
    )
    assert get_sigma_tension(*(10 * value for value in inputs)) == pytest.approx(
        2.0, rel=1e-12, abs=1e-14
    )
