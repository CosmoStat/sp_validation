"""The matched-cut recalibration recipe is scientifically incomplete.

A matched cut catalogue contains only R_g and cannot reconstruct the selection
and weight response of the new population. Calibrate from aligned parent
five-branch inputs via sp_validation.calibration.metacal_from_config instead.
PSF leakage and additive bias are distinct corrections.
"""


def process_matched_catalogue(
    input_path, output_path=None, weight_column="w_iv", num_bins=20, weight_type="iv"
):
    """Refuse to invent the missing selection response of a matched sample."""
    raise ValueError(
        "A matched cut catalogue lacks the five-branch selection/weight "
        "response for its changed population. Calibrate the parent with "
        "metacal_from_config before using a matched sample diagnostically."
    )


if __name__ == "__main__":
    process_matched_catalogue(None)
