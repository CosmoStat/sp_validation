# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: sp_validation
#     language: python
#     name: sp_validation
# ---

# # Leakage of minimal catalogue

from sp_validation import run_joint_cat as sp_joint


# Initialize calibration class instance
obj = sp_joint.CalibrateCat()

config = obj.read_config_set_params("config_minimal.yaml")

obj._params

# Get data. Set load_into_memory to False for very large files
dat, dat_ext = obj.read_cat(load_into_memory=False)

# ## Masking

masks_to_apply = [
    "N_EPOCH",
    "FLAGS",
    "4_Stars",
    "npoint3",
]

# Gather mask information for above list from config
masks, labels = sp_joint.get_masks_from_config(config, dat, dat_ext, masks_to_apply=masks_to_apply, verbose=obj._params["verbose"])

# Combine masks
mask_combined = sp_joint.Mask.from_list(
    masks,
    label="combined",
    verbose=obj._params["verbose"],
)

# Output some mask statistics
sp_joint.print_mask_stats(dat.shape[0], masks, masks_combined)


