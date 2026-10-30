"""Layout-independent full-sample means with patched covariance measurements."""

import hashlib
import json
from pathlib import Path

import numpy as np
import treecorr


def selection_digest(*masks):
    """Digest of row selections; ``None`` stands for every row.

    Recorded beside a cached measurement so a product made from different rows,
    e.g. before or after a sky mask, is never reused.
    """
    digest = hashlib.sha1()
    for mask in masks:
        if mask is None:
            digest.update(b"all;")
        else:
            mask = np.asarray(mask, dtype=bool)
            digest.update(f"{mask.size};".encode())
            digest.update(np.packbits(mask).tobytes())
    return digest.hexdigest()


def _measurement_configs(config):
    """Return the explicit means settings and default-tolerance patched settings."""
    means = dict(config)
    patched = dict(config)
    patched.pop("bin_slop", None)
    return {"means": means, "patched": patched}


def measurement_matches(path, config, selection=None):
    """Whether a cache records the current configurations and row selection."""
    metadata = Path(str(path) + ".json")
    if not Path(path).exists() or not metadata.exists():
        return False
    recorded = json.loads(metadata.read_text())
    recorded_configs = recorded.get("configs")
    if recorded.get("means") != "unpatched" or not isinstance(recorded_configs, dict):
        return False
    if recorded.get("selection") != selection:
        return False
    expected_configs = _measurement_configs(config)
    if config.get("min_top") is not None:
        # With the root depth pinned, threads affect reduction round-off,
        # not the approximation. A plotting job can reuse a multi-core product.
        for settings in (*expected_configs.values(), *recorded_configs.values()):
            if isinstance(settings, dict):
                settings.pop("num_threads", None)
    return recorded_configs == expected_configs


def write_measurement_metadata(path, config, selection=None):
    """Record both TreeCorr configurations and the row selection beside a product."""
    record = {"means": "unpatched", "configs": _measurement_configs(config)}
    if selection is not None:
        record["selection"] = selection
    Path(str(path) + ".json").write_text(json.dumps(record) + "\n")


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
    measurement supplies both products.
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


def process_gg(config, cat1, cat2=None):
    """Measure GG means with ``config`` and patched resampling at TreeCorr's default.

    The shared two-pass mechanism omits ``bin_slop`` only from the patched
    configuration. Covariance estimates, including joint and derived-statistic
    jackknives, retain TreeCorr's per-patch results. Published pair counts,
    weights, separations and complex correlations come from the means pass.
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

    means, gg = measure_with_patches(measure, catalogs, config)
    if means is not gg:
        # TreeCorr estimates covariance lazily from fields that the means pass
        # replaces below, so freeze the patched-pass estimates first.
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
