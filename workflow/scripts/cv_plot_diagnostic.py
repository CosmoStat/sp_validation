"""Catalogue-free plotting entry point for cached scalar/PSF/map diagnostics."""

import json

import numpy as np
from astropy.io import fits

from sp_validation import sacc_io
from sp_validation.cosmo_val import diagnostic_plots as plots
from sp_validation.cosmo_val.diagnostic_products import load_footprint

p, inputs, outputs = snakemake.params, snakemake.input, snakemake.output
styles = dict(p.styles)
versions = list(styles)


def psf_products():
    products = {}
    for ver, path in zip(versions, inputs.products, strict=True):
        with np.load(path, allow_pickle=False) as data:
            products[ver] = dict(data)
    return products


if p.kind == "weights":
    with np.load(inputs.histograms, allow_pickle=False) as data:
        histograms = {
            ver: (data[f"counts_{i}"], data[f"edges_{i}"])
            for i, ver in enumerate(data["versions"].tolist())
        }
    plots.plot_weights(histograms, styles, outputs.figure)
elif p.kind == "footprints":
    for i, (ver, path) in enumerate(zip(versions, inputs.maps, strict=True)):
        plots.plot_footprint(
            load_footprint(path),
            {
                region: outputs.figures[i * len(p.regions) + j]
                for j, region in enumerate(p.regions)
            },
        )
elif p.kind == "objectwise_leakage":
    with open(inputs.coefficients) as stream:
        coefficients = json.load(stream)
    plots.plot_objectwise_leakage(coefficients, styles, outputs.figure)
elif p.kind == "rho_tau_fits":
    plots.plot_rho_tau_fits(
        psf_products(), styles, outputs.contours, outputs.samples, outputs.summary
    )
elif p.kind in ("ratio", "xi"):
    correlations = {
        ver: sacc_io.xi_correlation(sacc_io.load(path, blind=blind), grid="reporting")
        for ver, path, blind in zip(versions, inputs.xi, p.blinds, strict=True)
    }
    if p.kind == "ratio":
        plots.plot_ratio_xi_sys_xi(psf_products(), correlations, styles, outputs.figure)
    else:
        plots.plot_2pcf(correlations, styles, outputs.figures)
elif p.kind in ("rho", "tau"):
    tables = {
        ver: fits.getdata(path)
        for ver, path in zip(versions, inputs.stats, strict=True)
    }
    covariances = [np.load(path) for path in inputs.covariances]
    plots.plot_psf_stats(tables, styles, outputs.figure, p.kind, covariances)
else:
    raise ValueError(f"Unknown diagnostic: {p.kind}")
