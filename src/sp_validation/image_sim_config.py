"""Image-simulation calibration policy (no catalogue I/O)."""

from copy import deepcopy


def calibration_config(config, w_cols, source="external", product=None):
    """Use a named data calibration table by default for DES image sims.

    Explicit ``source='self'`` retains sample-fitted W as a diagnostic.
    Non-DES runs retain their existing calibration configuration.
    """
    config = deepcopy(config)
    if "w_des" not in w_cols:
        return config
    if source not in ("external", "self"):
        raise ValueError("des_weight_source must be external or self")
    if source == "external" and (
        not isinstance(product, dict)
        or not product.get("name")
        or not product.get("path")
    ):
        raise ValueError(
            "DES image sims require des_weight_product with name and path; "
            "des_weight_source: self is diagnostic only"
        )
    cm = config["metacal"]
    cm.pop("global_R_weight", None)
    cm.update(weight_scheme="des", w_col="w_des", additive_correction=False)
    if source == "external":
        cm["des_weight_product"] = product.copy()
    else:
        cm.pop("des_weight_product", None)
    return config
