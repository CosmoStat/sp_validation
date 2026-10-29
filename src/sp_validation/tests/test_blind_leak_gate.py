"""Synthetic workflow no-leak regression: real estimators, real blind, real files.

Run in the scientific container on an allocated compute node. No fixture reads a
survey catalogue or a user's blind registry. The twin has a physical FITS copy,
not a symlink: one catalogue cannot legitimately declare two different blinds.
"""

import copy
import logging
import runpy
import shutil
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml
from astropy.io import fits
from smokescreen.datavector import draw_param_shifts

from sp_validation import b_modes, blinding, sacc_io
from sp_validation.cosmo_val import CosmologyValidation
from sp_validation.cosmo_val.sacc_writers import cosebis_to_sacc, pure_eb_to_sacc
from sp_validation.tests import test_cosmo_val as synthetic_helpers
from sp_validation.tests.blind_scan import assert_no_raw_values

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "workflow/scripts"
INTEGRATION = dict(min_sep=1.0, max_sep=150.0, nbins=100)
REPORTING = dict(min_sep=5.0, max_sep=100.0, nbins=4)
CUT = (5.0, 100.0)
CUTS = [CUT, (10.0, 70.0)]


class Named(dict):
    """The attribute, key and iteration access used by Snakemake scripts."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error

    def __getitem__(self, key):
        return (
            list(self.values())[key]
            if isinstance(key, int)
            else super().__getitem__(key)
        )

    def __iter__(self):
        return iter(self.values())


def script(name, *, params, inputs=None, outputs=None):
    smk = SimpleNamespace(
        params=Named(params), input=Named(inputs or {}), output=Named(outputs or {})
    )
    return runpy.run_path(
        str(SCRIPTS / name), run_name="__main__", init_globals={"snakemake": smk}
    )


def make_inputs(tmp_path):
    params, version = (
        synthetic_helpers.TestCosmologyValidation._write_synthetic_catalogs(
            tmp_path, n_gal=3000, coherent_shear=True, with_psf=True
        )
    )
    config_path = Path(params["catalog_config"])
    config = yaml.safe_load(config_path.read_text())
    config["paths"]["blinds"] = str(tmp_path / "test-blinds")
    config[version]["blind"] = "gate"
    config[version]["patch_number"] = 1
    config["Twin"] = copy.deepcopy(config[version])
    config["Twin"]["blind"] = "none"
    config["Twin"]["shear"]["path"] = "twin.fits"
    shutil.copyfile(tmp_path / "catalog/shear.fits", tmp_path / "catalog/twin.fits")
    config_path.write_text(yaml.safe_dump(config))
    # Exercise the same CLI entry point as `python -m ... blinding init`.
    blinding.main(["init", "gate", "--cat-config", str(config_path)])
    return config_path, config, version


def produce(config_path, version, blind, output):
    """Execute the workflow's actual non-tomographic producer script bodies."""
    output.mkdir()
    paths = {key: output / f"{key}.sacc" for key in ("reporting", "integration", "cl")}
    for grid, binning in (("reporting", REPORTING), ("integration", INTEGRATION)):
        script(
            "run_2pcf.py",
            params=dict(
                ver=version,
                **binning,
                npatch=1,
                grid=grid,
                cat_config=str(config_path),
                output_dir=str(output),
            ),
            outputs={"sacc": str(paths[grid])},
        )
    script(
        "generate_pseudo_cl.py",
        params=dict(
            version=version,
            cat_config=str(config_path),
            nside=32,
            npatch=1,
            binning="powspace",
            nbins=4,
            power=0.5,
        ),
        outputs={"pseudo_cl": str(paths["cl"])},
    )
    # Covariance is exempt from blinding. A positive synthetic analytic matrix
    # avoids OneCovariance/external survey masks and keeps this a small gate.
    covariance = output / "cov.txt"
    np.savetxt(covariance, np.eye(2 * INTEGRATION["nbins"]) * 1e-10)
    pure = script(
        "cv_pure_eb.py",
        params=dict(
            version=version,
            blind=blind,
            fiducial_scale_cut=CUT,
            integration=INTEGRATION,
            **REPORTING,
        ),
        inputs={
            "xi_integration": str(paths["integration"]),
            "cov_integration": str(covariance),
        },
        outputs={
            "npz": str(output / "pure.npz"),
            "sacc": str(output / "pure.sacc"),
            # Before the compute/render split, the producer also draws figures.
            **(
                pure_figures(output)
                if not (SCRIPTS / "cv_plot_pure_eb.py").exists()
                else {}
            ),
        },
    )
    cosebis = script(
        "cv_cosebis.py",
        params=dict(
            version=version,
            blind=blind,
            fiducial_scale_cut=CUT,
            nmodes=2,
            scale_cuts=CUTS,
            **INTEGRATION,
        ),
        inputs={"xi": str(paths["integration"]), "cov": str(covariance)},
        outputs={
            "npz": str(output / "cosebis.npz"),
            "sacc": str(output / "cosebis.sacc"),
            **(
                cosebis_figures(output)
                if not (SCRIPTS / "cv_plot_cosebis.py").exists()
                else {}
            ),
        },
    )
    return paths, pure["results"], cosebis["results"]


def shift_for(part, blind):
    """Independent oracle: evaluate t(hidden)-t(fiducial), not conceal/save."""
    record = blind.record()  # A newly generated tmp test blind ONLY.
    fiducial = record["fiducial"]
    draw = draw_param_shifts(record["envelope"], record["seed"])
    hidden = {key: value + draw.get(key, 0.0) for key, value in fiducial.items()}
    return blinding.shear(hidden, part) - blinding.shear(fiducial, part)


def references_and_shifts(
    raw_paths, sealed_paths, raw_pure, sealed_pure, raw_cosebis, sealed_cosebis, blind
):
    references, concealed = {}, {}

    def reference(label, raw, sealed):
        raw, sealed = np.asarray(raw), np.asarray(sealed)
        valid = np.isfinite(raw) & (raw != 0)
        changed = ~np.isclose(raw, sealed, rtol=5e-6, atol=1e-25)
        assert np.any(valid & changed), f"Fixture has no shifted nonzero {label} values"
        # Retain weakly shifted bins too. The scanner chooses references at the
        # decoder's precision rather than dropping every <5e-6 relative shift.
        references[label], concealed[label] = raw[valid], sealed[valid]

    for grid in ("reporting", "integration", "cl"):
        raw = sacc_io.load(raw_paths[grid], blind="none")
        sealed = sacc_io.load(sealed_paths[grid], blind=blind)
        delta = shift_for(raw, blind)
        np.testing.assert_allclose(
            sealed.mean, raw.mean + delta, rtol=1e-10, atol=1e-25
        )
        kinds = (
            (sacc_io.CL_EE,) if grid == "cl" else (sacc_io.XI_PLUS, sacc_io.XI_MINUS)
        )
        for kind in kinds:
            rows = raw.indices(kind)
            reference(f"{grid}/{kind}", raw.mean[rows], sealed.mean[rows])
        if grid == "cl":
            for kind in (sacc_io.CL_BB, sacc_io.CL_EB):
                rows = raw.indices(kind)
                # Independent NaMaster runs differ at roundoff (the small
                # footprint's coupling matrix is ill-conditioned at low ell).
                np.testing.assert_allclose(
                    sealed.mean[rows], raw.mean[rows], rtol=1e-9, atol=1e-18
                )
        else:
            np.testing.assert_allclose(
                sealed.covariance.dense, raw.covariance.dense, rtol=1e-12, atol=1e-25
            )

    raw = sacc_io.load(raw_paths["integration"])
    shifted = raw.copy()
    for dp, delta in zip(shifted.data, shift_for(raw, blind), strict=True):
        dp.value += delta
    gg = sacc_io.xi_correlation(shifted, grid="integration")
    edges = np.geomspace(
        INTEGRATION["min_sep"], INTEGRATION["max_sep"], INTEGRATION["nbins"] + 1
    )
    expected_pure = b_modes.calculate_pure_eb_correlation(
        gg.meanr,
        gg.xip,
        gg.xim,
        gg.weight,
        edges,
        np.eye(2 * INTEGRATION["nbins"]) * 1e-10,
        np.geomspace(*CUT, REPORTING["nbins"] + 1),
    )
    pure_path = sealed_paths["cl"].parent / "pure.npz"
    extended_pure_cache = hasattr(b_modes, "load_pure_eb_results")
    if extended_pure_cache:
        cached_pure = b_modes.load_pure_eb_results(pure_path, blind=blind)
    else:
        # develop's pre-split cache has the pure modes but not the comparison ξ±
        # arrays or a stamp. Still check every persisted signal value numerically.
        with np.load(pure_path) as data:
            cached_pure = dict(data)
    for key in (*sacc_io.PURE_KEYS, "xip", "xim", "xip_int", "xim_int"):
        np.testing.assert_allclose(
            sealed_pure[key], expected_pure[key], rtol=1e-10, atol=1e-20
        )
    cache_keys = (
        (*sacc_io.PURE_KEYS, "xip", "xim", "xip_int", "xim_int")
        if extended_pure_cache
        else sacc_io.PURE_KEYS
    )
    for key in cache_keys:
        np.testing.assert_allclose(
            cached_pure[key], expected_pure[key], rtol=1e-10, atol=1e-20
        )
    for key in (
        "xip_E",
        "xim_E",
        "xip_amb",
        "xim_amb",
        "xip",
        "xim",
        "xip_int",
        "xim_int",
    ):
        reference(f"pure/{key}", raw_pure[key], sealed_pure[key])
    expected_cosebis = b_modes.cosebis_scan_from_xi(
        gg.meanr,
        gg.xip,
        gg.xim,
        np.eye(2 * INTEGRATION["nbins"]) * 1e-10,
        edges[:-1],
        edges[1:],
        nmodes=2,
        scale_cuts=CUTS,
    )
    cosebis_path = sealed_paths["cl"].parent / "cosebis.npz"
    if hasattr(b_modes, "load_cosebis_results"):
        cached_cosebis, _ = b_modes.load_cosebis_results(cosebis_path, blind=blind)
        assert set(cached_cosebis) == set(expected_cosebis)
    else:
        # Before #430, only the fiducial cut is persisted. Both cuts are still
        # computed by the real script and checked in memory below.
        with np.load(cosebis_path) as data:
            assert tuple(data["scale_cut"]) == CUT
            cached_cosebis = {CUT: dict(data)}
    for cut, result in sealed_cosebis.items():
        for key in ("En", "Bn", "chi2_E"):
            np.testing.assert_allclose(
                result[key], expected_cosebis[cut][key], rtol=1e-10, atol=1e-20
            )
            if cut in cached_cosebis:
                np.testing.assert_allclose(
                    cached_cosebis[cut][key],
                    expected_cosebis[cut][key],
                    rtol=1e-10,
                    atol=1e-20,
                )
        for key in ("En", "chi2_E"):
            reference(f"cosebis/{cut}/{key}", raw_cosebis[cut][key], result[key])
    # Check SACC numerical values too: a stamp and nonempty output aren't enough.
    nz = {0: sacc_io.get_nz(raw, 0)}
    expected_parts = {
        "pure": pure_eb_to_sacc(nz, {}, expected_pure["theta"], expected_pure),
        "cosebis": cosebis_to_sacc(nz, {}, expected_cosebis[CUT], CUT),
    }
    for name, expected in expected_parts.items():
        part = sacc_io.load(sealed_paths["cl"].parent / f"{name}.sacc", blind=blind)
        np.testing.assert_allclose(part.mean, expected.mean, rtol=1e-10, atol=1e-20)
        assert len(part.mean) > 0 and np.all(np.isfinite(part.mean))
    return references, concealed


def pure_figures(output):
    return {
        f"figure_{key}": str(output / f"pure_{key}.png")
        for key in ("integration_vs_reporting", "xis", "ptes", "covariance")
    }


def cosebis_figures(output):
    return {
        f"figure_{key}": str(output / f"cosebis_{key}.png")
        for key in ("modes", "covariance", "scalecut_ptes")
    }


def observe_renderers(observed, monkeypatch):
    """Record real renderer inputs, including pre-split compute-time figures."""
    from sp_validation.cosmo_val import pseudo_cl

    def record(module, name):
        original = getattr(module, name)

        def wrapped(*args, **kwargs):
            observed.append((f"renderer {module.__name__}.{name}", (args, kwargs)))
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, wrapped)

    for module, names in (
        (pseudo_cl, ("plot_pseudo_cl_spectrum",)),
        (
            b_modes,
            (
                "plot_integration_vs_reporting",
                "plot_pure_eb_correlations",
                "plot_cosebis_modes",
            ),
        ),
    ):
        for name in names:
            record(module, name)
    if find_spec("sp_validation.cosmo_val.diagnostic_plots") is not None:
        from sp_validation.cosmo_val import diagnostic_plots

        record(diagnostic_plots, "plot_2pcf")
    if find_spec("sp_validation.dashboard") is not None:
        from sp_validation.dashboard import data_vectors

        for name in ("xi_pm", "pure_eb", "cosebis"):
            record(data_vectors, name)


def render(paths, version, blind):
    """Always assemble/plot Cℓ; run #430/#434-only figures when present.

    No importorskip here: a missing optional surface must not skip the core
    estimators, assembly, numerical assertions or final leak scan.
    """
    out = paths["cl"].parent
    cov_cl = out / "cl_cov.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            *[
                fits.ImageHDU(np.eye(4) * 1e-12, name=f"COVAR_{k}_{k}")
                for k in ("EE", "BB", "EB")
            ],
        ]
    ).writeto(cov_cl)
    cov_xi = out / "reporting_cov.txt"
    np.savetxt(cov_xi, np.eye(2 * REPORTING["nbins"]) * 1e-10)
    assembled = out / "assembled.sacc"
    script(
        "assemble_sacc.py",
        params=dict(
            version=version,
            blind=blind,
            expected=["xi_reporting", "pseudo_cl", "cosebis", "pure_eb"],
        ),
        inputs=dict(
            xi_reporting=str(paths["reporting"]),
            pseudo_cl=str(paths["cl"]),
            cosebis=str(out / "cosebis.sacc"),
            pure_eb=str(out / "pure.sacc"),
            xi_cov=str(cov_xi),
            pseudo_cl_cov=str(cov_cl),
        ),
        outputs={"sacc": str(assembled)},
    )
    n_figures = 10  # Four pure-E/B, three COSEBIs, three Cℓ on both APIs.
    if (SCRIPTS / "cv_plot_diagnostic.py").exists():
        script(
            "cv_plot_diagnostic.py",
            params=dict(
                kind="xi",
                styles={version: {"colour": "C0", "marker": "o"}},
                blinds=[blind],
            ),
            inputs={"xi": [str(paths["reporting"])]},
            outputs={"figures": [str(out / f"xi_{i}.png") for i in range(2)]},
        )
        n_figures += 2
    script(
        "cv_plot_pseudo_cl.py",
        params=dict(versions=[version], blinds=[blind], markers=["o"], colours=["C0"]),
        inputs={"pseudo_cl": [str(paths["cl"])], "pseudo_cl_cov": [str(cov_cl)]},
        outputs={f"figure_{k}": str(out / f"cl_{k}.png") for k in ("ee", "bb", "eb")},
    )
    for name, outputs in (
        ("pure_eb", pure_figures(out)),
        ("cosebis", cosebis_figures(out)),
    ):
        if (SCRIPTS / f"cv_plot_{name}.py").exists():
            script(
                f"cv_plot_{name}.py",
                params=dict(version=version, blind=blind, fiducial_scale_cut=CUT),
                inputs={
                    "npz": str(
                        out / ("pure.npz" if name == "pure_eb" else "cosebis.npz")
                    )
                },
                outputs=outputs,
            )
    if find_spec("sp_validation.dashboard") is not None:
        from sp_validation.dashboard import data_vectors, products

        xi = products.load_xi(assembled, blind)
        pure = products.load_pure_eb(out / "pure.npz", blind)
        cosebis = products.load_cosebis(out / "cosebis.npz", blind)
        data_vectors.xi_pm(xi, out / "dashboard_xi.png", blind=blind)
        data_vectors.pure_eb(pure, out / "dashboard_pure.png", CUT, blind=blind)
        data_vectors.cosebis(
            cosebis, out / "dashboard_cosebis.png", full_cut=CUT, n_subset=2
        )
        n_figures += 3
    figures = list(out.glob("*.png"))
    assert len(figures) == n_figures and all(p.stat().st_size > 100 for p in figures)
    # All constituent signal vectors survive assembly unchanged (not re-shifted).
    expected_mean = np.concatenate(
        [
            sacc_io.load(path).mean
            for path in (
                paths["reporting"],
                paths["cl"],
                out / "cosebis.sacc",
                out / "pure.sacc",
            )
        ]
    )
    np.testing.assert_array_equal(
        sacc_io.load(assembled, blind=blind).mean, expected_mean
    )


@pytest.mark.slow
def test_non_tomographic_workflow_emits_no_unblinded_signal(
    tmp_path, monkeypatch, capfd, caplog
):
    """Paired catalogue runs must emit shifted signal and no copied raw values."""
    assert (
        Path(blinding.__file__).resolve().parent == Path(__file__).resolve().parents[1]
    ), (
        "Gate imported the image's baked package, not the checkout under test; "
        "set PYTHONPATH=$PWD/src inside spv-container exec"
    )
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import cv_runner

    # Only stream ownership is replaced; estimators, theories, I/O and renderers
    # remain real. Reopening pytest's fd would otherwise close its capture.
    monkeypatch.setattr(cv_runner, "_unbuffer_streams", lambda: None)
    config_path, config, version = make_inputs(tmp_path)
    blind = blinding.open_blind("gate", config)
    raw_paths, raw_pure, raw_cosebis = produce(
        config_path, "Twin", "none", tmp_path / "public"
    )
    capfd.readouterr()
    caplog.clear()
    observed = []
    observe_renderers(observed, monkeypatch)
    # Observe high-level returns and retained state without replacing computation.
    for name in ("calculate_2pcf_version", "calculate_pseudo_cl_catalog"):
        original = getattr(CosmologyValidation, name)

        def watch(self, *args, _original=original, **kwargs):
            result = _original(self, *args, **kwargs)
            observed.append((f"producer {_original.__name__} return", result))
            observed.append((f"producer {_original.__name__} state", dict(vars(self))))
            return result

        monkeypatch.setattr(CosmologyValidation, name, watch)
    with caplog.at_level(logging.INFO):
        sealed_paths, sealed_pure, sealed_cosebis = produce(
            config_path, version, "gate", tmp_path / "sealed"
        )
        render(sealed_paths, version, "gate")
        cv = CosmologyValidation(
            versions=[version],
            catalog_config=str(config_path),
            output_dir=str(tmp_path / "sealed"),
            npatch=1,
        )
        with pytest.raises(blinding.BlindingError, match="tomographic"):
            cv.calculate_2pcf_version(version, compute_tomography=True)
        with pytest.raises(blinding.BlindingError, match="tomographic"):
            cv.calculate_pseudo_cl(compute_tomography=True)
    captured = capfd.readouterr()
    logs = captured.out + captured.err + caplog.text
    references, concealed = references_and_shifts(
        raw_paths,
        sealed_paths,
        raw_pure,
        sealed_pure,
        raw_cosebis,
        sealed_cosebis,
        blind,
    )
    assert_no_raw_values(
        tmp_path / "sealed",
        references,
        expected=concealed,
        logs=logs,
        objects=[
            ("pure producer return", sealed_pure),
            ("COSEBIs producer return", sealed_cosebis),
            *observed,
        ],
    )
    plt.close("all")
