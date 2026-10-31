"""Catalogues without galaxy PSF shapes still support PSF cross-correlations."""

from types import SimpleNamespace

from sp_validation.cosmo_val.core import CosmologyValidation
from sp_validation.cosmo_val.diagnostic_plots import plot_objectwise_leakage


def test_missing_galaxy_psf_skips_objectwise_regression(tmp_path):
    messages = []
    cv = SimpleNamespace(
        versions=["mock"],
        cc={"mock": {"shear": {"e1_col": "e1", "e2_col": "e2"}}},
        print_magenta=messages.append,
        # The builder must not be called for an inapplicable regression.
        set_params_leakage_object=lambda _: (_ for _ in ()).throw(AssertionError()),
        set_params_leakage_scale=None,
    )
    assert CosmologyValidation.init_results(cv, objectwise=True) == {}
    assert "no galaxy PSF ellipticity columns" in messages[0]
    output = tmp_path / "unavailable.png"
    plot_objectwise_leakage({}, {}, output)
    assert output.is_file()
