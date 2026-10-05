"""Layout-independent full-sample means with patched covariance measurements."""

import json
from pathlib import Path

import numpy as np
import treecorr

from sp_validation.angular_binning import validate_nested_grids
from sp_validation.b_modes import _reporting_binning


def _measurement_configs(config):
    """Return the explicit means settings and default-tolerance patched settings."""
    means = dict(config)
    patched = dict(config)
    patched.pop("bin_slop", None)
    return {"means": means, "patched": patched}


def measurement_matches(path, config):
    """Whether a cache records the current means and patched configurations."""
    metadata = Path(str(path) + ".json")
    if not Path(path).exists() or not metadata.exists():
        return False
    recorded = json.loads(metadata.read_text())
    recorded_configs = recorded.get("configs")
    if recorded.get("means") != "unpatched" or not isinstance(recorded_configs, dict):
        return False
    expected_configs = _measurement_configs(config)
    if config.get("min_top") is not None:
        # With the root depth pinned, threads affect reduction round-off,
        # not the approximation. A plotting job can reuse a multi-core product.
        for settings in (*expected_configs.values(), *recorded_configs.values()):
            if isinstance(settings, dict):
                settings.pop("num_threads", None)
    return recorded_configs == expected_configs


def write_measurement_metadata(path, config):
    """Record both TreeCorr configurations beside a cached product."""
    Path(str(path) + ".json").write_text(
        json.dumps({"means": "unpatched", "configs": _measurement_configs(config)})
        + "\n"
    )


def _unpatched_catalog(cat):
    """The same positions, shears and weights, with no spatial partition."""
    if cat.ra is not None:
        positions = dict(ra=cat.ra, dec=cat.dec, ra_units="rad", dec_units="rad")
        if cat.r is not None:
            positions["r"] = cat.r
    else:
        positions = dict(x=cat.x, y=cat.y)
        if cat.z is not None:
            positions["z"] = cat.z
    return treecorr.Catalog(
        **positions, g1=cat.g1, g2=cat.g2, w=cat.w, wpos=cat.wpos, npatch=1
    )


def measure_with_patches(measure, catalogs, config, means=None):
    """Return means and a patched measurement with distinct TreeCorr settings.

    ``measure(catalogs, config)`` must return a fresh measurement of the
    supplied catalogue mapping. ``config`` describes the means pass, including
    its explicit ``bin_slop``. The patched pass omits ``bin_slop`` so TreeCorr
    uses its standard default. A cached ``means`` can be reused for repeated
    covariance draws of the same catalogue. With no patches, one means-config
    measurement supplies both products unless supplied means are reused.
    """
    configs = _measurement_configs(config)
    if all(cat.npatch == 1 for cat in catalogs.values()):
        patched = measure(catalogs, configs["means"])
        return (patched if means is None else means), patched

    patched = measure(catalogs, configs["patched"])
    if means is None:
        unpatched = {}
        # Preserve aliases, including an auto-correlation passed as a cross-pair.
        copies = {}
        for key, cat in catalogs.items():
            if id(cat) not in copies:
                copies[id(cat)] = _unpatched_catalog(cat)
            unpatched[key] = copies[id(cat)]
        means = measure(unpatched, configs["means"])
    return means, patched


def rebin_gg_means(fine, config):
    """Pair-weighted fine ξ± means on exactly nested reporting edges.

    The same operator used by pure-E/B averages the complex correlations and
    separations. Pair counts and weights are sums over the corresponding fine
    bins. This object supplies means only, not a covariance or patch results.
    """
    reporting = treecorr.GGCorrelation({**config, "var_method": "shot"})
    if fine.sep_units != reporting.sep_units:
        raise ValueError("fine and reporting ξ± must use the same separation units")
    validate_nested_grids(
        config, dict(min_sep=fine.min_sep, max_sep=fine.max_sep, nbins=fine.nbins)
    )
    fine_edges = np.append(fine.left_edges, fine.right_edges[-1])
    report_edges = np.append(reporting.left_edges, reporting.right_edges[-1])
    operator, edges = _reporting_binning(fine.weight, fine_edges, report_edges)
    for name in ("xip", "xim", "xip_im", "xim_im", "meanr", "meanlogr"):
        getattr(reporting, name)[:] = operator @ getattr(fine, name)
    indices = np.searchsorted(fine_edges, edges)
    for i, (lo, hi) in enumerate(zip(indices[:-1], indices[1:])):
        reporting.weight[i] = np.sum(fine.weight[lo:hi])
        reporting.npairs[i] = np.sum(fine.npairs[lo:hi])
    return reporting


def process_gg(config, cat1, cat2=None, means=None):
    """Measure GG means and patched resampling products at distinct tolerances.

    The configured ``bin_slop`` applies to unpatched means; the shared
    two-pass mechanism omits it from the patched configuration so TreeCorr's
    default sets the covariance and resampling tolerance. Supplied ``means``
    replace the unpatched reporting pass, for example with pair-weighted fine-grid
    means. The patched covariance is materialized before the means fields change.
    """
    catalogs = {"cat1": cat1}
    if cat2 is not None:
        catalogs["cat2"] = cat2

    def measure(cats, measurement_config):
        cfg = dict(measurement_config)
        if all(cat.npatch == 1 for cat in cats.values()):
            cfg["var_method"] = "shot"
        gg = treecorr.GGCorrelation(cfg)
        gg.process(cats["cat1"], cat2=cats.get("cat2"))
        return gg

    means, gg = measure_with_patches(measure, catalogs, config, means=means)
    if means is not gg:
        # Freeze the patched covariance and variances before replacing means.
        _ = gg.varxip, gg.varxim, gg.cov
        for name in (
            "xip",
            "xim",
            "xip_im",
            "xim_im",
            "meanr",
            "meanlogr",
            "npairs",
            "weight",
        ):
            getattr(gg, name)[:] = getattr(means, name)
    return gg
