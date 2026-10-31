"""OneCovariance inputs for tomographic and whole-sample covariances."""

from contextlib import contextmanager
from pathlib import Path

import yaml
from conftest import VERSIONS


def _ini(toy, tomo_suffix=""):
    # Unmasked target: no site-specific footprint needed by this test.
    grid = toy.common.xi_grids(toy.config, toy.config["fiducial"])["integration"]
    return toy.common.covariance_path(
        VERSIONS[0],
        "g",
        grid["min_sep"],
        grid["max_sep"],
        grid["nbins"],
        "",
        tomo_suffix,
        suffix=".ini",
    )


@contextmanager
def _two_bin_catalogue(toy):
    """The toy version as a two-bin catalogue: a 2-column n(z), cov_th.tomo."""
    nz = toy.root / "data" / "nz_SP_v0.1_A.txt"
    cat_config = toy.root / "cosmo_val" / "cat_config.yaml"
    originals = {p: p.read_text() if p.exists() else None for p in (nz, cat_config)}
    nz.write_text("0.0 0.0 0.0\n0.5 1.0 0.5\n1.0 0.0 1.5\n")
    entries = yaml.safe_load(originals[cat_config])
    entries[VERSIONS[0]]["cov_th"]["tomo"] = {
        "n_e": [2.0, 3.0],
        "sigma_e": [0.25, 0.35],
        "A": 7.5,
    }
    entries[VERSIONS[0]]["shear"]["tomo_bin_col"] = "TOM_BIN_ID"
    cat_config.write_text(yaml.safe_dump(entries))
    try:
        yield entries[VERSIONS[0]]["cov_th"]
    finally:
        for path, text in originals.items():
            if text is None:
                path.unlink(missing_ok=True)
            else:
                path.write_text(text)


def test_tomographic_catalogue_covariance_inputs(toy):
    """One multi-bin entry serves both covariances: the tomographic ini reads its
    per-bin n(z) and cov_th.tomo (with the binned sample's own area), the
    non-tomographic one the bin-summed n(z) and the whole-sample cov_th."""
    unmasked = ['covariance={"shape_noise":"analytic","default_masked":false}']
    with _two_bin_catalogue(toy) as cov_th:
        texts = {}
        for suffix in ("", "_tomo"):
            target = _ini(toy, suffix)
            result = toy.snakemake(
                "-F",
                target,
                "--allowed-rules",
                "covariance_ini",
                "nz_bin_sum",
                config=unmasked,
            )
            assert result.returncode == 0, result.stdout
            texts[suffix] = Path(target).read_text()
    assert "zlens_file = nz_SP_v0.1_A.txt" in texts["_tomo"]
    assert "ellipticity_dispersion = 0.25, 0.35" in texts["_tomo"]
    assert "n_eff_lensing = 2.0, 3.0" in texts["_tomo"]
    assert "survey_area_lensing_in_deg2 = 7.5\n" in texts["_tomo"]
    assert f"survey_area_lensing_in_deg2 = {cov_th['A']}\n" in texts[""]
    assert f"ellipticity_dispersion = {float(cov_th['sigma_e'])!r}\n" in texts[""]
    assert f"n_eff_lensing = {float(cov_th['n_e'])!r}\n" in texts[""]
    summed = toy.cosmo_val / "nz" / f"{VERSIONS[0]}_nz_bin_sum.txt"
    assert f"zlens_file = {summed.name}" in texts[""]
    table = [
        list(map(float, line.split()))
        for line in summed.read_text().splitlines()
        if not line.startswith("#")
    ]
    assert table == [[0.0, 0.0], [0.5, 1.5], [1.0, 1.5]]
