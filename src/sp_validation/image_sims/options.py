"""What each option of an image-simulation decision sets.

The ASTRA spec (``analyses/image_sims/astra.yaml``) names every decision, its
options and why one is chosen; ``analyses/image_sims/config/options.yaml``
gives the concrete settings of each option, keyed the same way. A recipe hands
the selected option ids over as ``decision=option`` pairs, and :func:`settings`
merges what they set. An option the code does not implement has no entry.
"""

import yaml


def parse(pairs):
    """``["field=fir_b0269", ...]`` as a dict."""
    return dict(pair.split("=", 1) for pair in pairs)


def settings(path, selected):
    """The merged settings of the selected options.

    Parameters
    ----------
    path : str
        The options table (YAML: decision -> option -> settings).
    selected : dict
        Decision id -> option id.
    """
    with open(path) as f:
        table = yaml.safe_load(f)
    out = {}
    for decision, option in selected.items():
        options = table[decision]
        if option not in options:
            raise ValueError(f"{decision}={option} is not implemented in {path}")
        out.update(options[option] or {})
    return out
