"""Layout-independent full-sample means with patched covariance measurements."""

import json
from pathlib import Path

import treecorr


def measurement_matches(path, config):
    """Whether a cached product records these settings and unpatched means."""
    metadata = Path(str(path) + ".json")
    if not Path(path).exists() or not metadata.exists():
        return False
    recorded = json.loads(metadata.read_text())
    previous_config = recorded.get("config", {})
    if config.get("min_top") is not None:
        # With the root depth pinned, threads affect reduction round-off,
        # not the approximation. A plotting job can reuse a multi-core product.
        config = {k: v for k, v in config.items() if k != "num_threads"}
        previous_config = {
            k: v for k, v in previous_config.items() if k != "num_threads"
        }
    return recorded.get("means") == "unpatched" and previous_config == config


def write_measurement_metadata(path, config):
    """Record the full configuration and mean source beside a cached product."""
    Path(str(path) + ".json").write_text(
        json.dumps({"means": "unpatched", "config": config}) + "\n"
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


def measure_with_patches(measure, catalogs, means=None):
    """Return unpatched means and a measurement retaining the patch results.

    ``measure(catalogs)`` must return a fresh measurement of the supplied
    catalogue mapping. The patched measurement defines all resampling products;
    the unpatched measurement defines all full-sample means. A cached ``means``
    can be reused for repeated covariance draws of the same catalogue.
    With no patches, one measurement supplies both products.
    """
    patched = measure(catalogs)
    if all(cat.npatch == 1 for cat in catalogs.values()):
        return patched, patched
    if means is None:
        unpatched = {}
        # Preserve aliases, including an auto-correlation passed as a cross-pair.
        copies = {}
        for key, cat in catalogs.items():
            if id(cat) not in copies:
                copies[id(cat)] = _unpatched_catalog(cat)
            unpatched[key] = copies[id(cat)]
        means = measure(unpatched)
    return means, patched


def process_gg(config, cat1, cat2=None):
    """Measure GG means without patches and keep the patched resampling state.

    Covariance estimates, including joint and derived-statistic jackknives,
    continue to use TreeCorr's per-patch results. Published pair counts,
    weights, separations and complex correlations use the unpatched tree.
    """
    catalogs = {"cat1": cat1}
    if cat2 is not None:
        catalogs["cat2"] = cat2

    def measure(cats):
        cfg = dict(config)
        if all(cat.npatch == 1 for cat in cats.values()):
            cfg["var_method"] = "shot"
        gg = treecorr.GGCorrelation(cfg)
        gg.process(cats["cat1"], cat2=cats.get("cat2"))
        return gg

    means, gg = measure_with_patches(measure, catalogs)
    if means is not gg:
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
