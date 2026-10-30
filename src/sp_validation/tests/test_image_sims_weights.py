"""Image-simulation numerator weights and paired-estimator restrictions."""

import numpy as np
import pytest
from astropy.io import fits

from sp_validation.image_sims import ImageSimMBias


def test_modern_headline_uses_branch_weights(tmp_path):
    """Symmetric weights remove a real numerator contribution on matched rows."""
    for branch, sign in (("plus", 1), ("minus", -1)):
        directory = tmp_path / f"{branch}_grid_1"
        directory.mkdir()
        arrays = dict(
            RA=[30.0, 31.0],
            Dec=[0.0, 0.0],
            e1=np.array([-0.1, 0.1]) + sign * 0.02,
            e2=[0.0, 0.0],
            w_des=[1.0, 3.0] if sign == 1 else [3.0, 1.0],
        )
        arrays["e1_uncal"] = arrays["e1"]
        arrays["e2_uncal"] = arrays["e2"]
        table = fits.BinTableHDU.from_columns(
            [fits.Column(name=k, array=v, format="D") for k, v in arrays.items()]
            + [fits.Column(name="TILE_ID", array=["a", "b"], format="1A")]
        )
        jk = fits.BinTableHDU.from_columns(
            [
                fits.Column(name="TILE_ID", array=["b", "a"], format="1A"),
                *[
                    fits.Column(name=f"R_{i}{j}", array=[float(i == j)] * 2, format="D")
                    for i in (1, 2)
                    for j in (1, 2)
                ],
            ],
            name="R_JK",
        )
        primary = fits.PrimaryHDU()
        primary.header["WGHTSCH"] = "des"
        primary.header["RAPPLIED"] = True
        fits.HDUList([primary, table, jk]).writeto(directory / "cat.fits")
    cfg = dict(
        grids_dir=str(tmp_path),
        num=1,
        catalog_name="cat.fits",
        shear_amplitude=0.02,
        match_radius_deg=0.0002,
        w_cols=["w_des"],
        n_bootstrap=10,
        bootstrap_seed=42,
        pair_match=False,
        branches=["plus", "minus"],
        pairs=[dict(plus="plus", minus="minus", component=0)],
    )
    estimator = ImageSimMBias(cfg)
    estimator.load_catalogs(False)
    result = estimator.run(False)
    assert result["m1"] == pytest.approx(2.5)
    # All objects match, but symmetric weighting would yield m=0.
    symmetric_m = (
        np.mean((estimator.cats["plus"]["e1"] - estimator.cats["minus"]["e1"]) / 0.04)
        - 1
    )
    assert symmetric_m == pytest.approx(0)
    with pytest.raises(ValueError, match="pair_match=False"):
        ImageSimMBias({**cfg, "pair_match": True}).load_catalogs(False)
