# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Match UNIONS and HSC PSF stars

# %% [markdown]
# References:
# - Li et al. (2022), HSC-SSP Y3 shear catalogue
#   https://arxiv.org/abs/2107.00136
# - Aihara et al. (2021), HSC-SSP Y3 data release
#   https://arxiv.org/abs/2108.13045

# %%

from astropy.io import fits

from sp_validation.plots import *

# %%
# Tolerance: Set in ShapePipe config file of the match_external module.
# Largest distance to define matched pair

tol = 0.5  # in arcsec

# %%
# Open matched catalogue
hdu_list = fits.open(
    "output/run_sp_match_ext/match_external_runner/output/cat_matched-0000000.fits"
)

# %%
data = hdu_list[1].data

# %%
data.dtype.names

# %%
len(data)

# %%
# Use only objects with valid PSF flag
m_flag = data["FLAG_STAR_HSM"] == 0
len(np.where(m_flag)[0])

# %%
# Mask out objects with invalid coordinates (due to previous MCCD bug)
m_coord = (data["RA"] != 0) & (data["DEC"] != 0)
len(np.where(m_coord)[0])

# %%
# Remove unmatched objects which have been marked by -99
# (see ShapePipe config file)
m_match = data["ira"] != -99
len(np.where(m_match)[0])

# %%
m_val = m_flag & m_match

# %%
# Plot spatial distribution
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(15, 7.5))
plt.tight_layout()

ms = [0.2, 0.5]

axes[0].plot(
    data["RA"][m_coord], data["DEC"][m_coord], ".", markersize=ms[0], label="SP"
)
axes[1].plot(data["RA"][m_val], data["DEC"][m_val], ".", markersize=ms[1])

for idx, ax in enumerate(axes):
    ax.plot(
        data["ira"][m_val], data["idec"][m_val], ".", markersize=ms[idx], label="HSC"
    )


axes[0].legend(markerscale=30)

for ax in axes:
    ax.set_xlabel("R.A. [deg]")
    ax.set_ylabel("dec [deg]")

# %%
print(data["distance"].max(), data["distance"][m_val].max())

# %%
len(np.where(data["distance"] < tol)[0])

# %%
title = "UNIONS-r HSC PSF stars"
n_bin = 200

# %%
plot_histograms(
    [data["distance"], data["distance"][m_val]],
    ["all", "matched"],
    title,
    "distance [arcsec]",
    "frequency",
    [0, 1.5],
    n_bin,
    "hist_dist",
    density=False,
)

# %%
n_bin = 50
plot_histograms(
    [np.log10(data["distance"]), np.log10(data["distance"][m_val])],
    ["all", "matched"],
    title,
    "log10(distance [arcsec])",
    "frequency",
    [-2, 3],
    n_bin,
    "log_hist_dist",
)

# %%
# Upper limiting distance to define potential match,
tol_up = 3  # in arcsec

# %%
m_unmatched = (data["distance"] > tol) & (data["distance"] < tol_up) & m_flag

# %%
print(len(data["RA"][m_unmatched]), len(data["RA"][m_val]))

# %%
# Plot spatial distribution
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))

ms = 5

ax.plot(
    data["RA"][m_unmatched],
    data["DEC"][m_unmatched],
    ".",
    markersize=ms,
    label="unmatched",
)
ax.plot(data["ira"][m_val], data["idec"][m_val], ".", markersize=ms, label="matched")

ax.legend(markerscale=3)

ax.set_xlabel("R.A. [deg]")
ax.set_ylabel("dec [deg]")

# %%
# Open original HSC star catalogue
hdu_list_hsc = fits.open(
    "/n17data/mkilbing/astro/data/CFIS/imaging_surveys/HECTOMAP_stars.fits"
)

# %%
data_hsc = hdu_list_hsc[1].data

# %%
data_hsc.dtype.names

# %%
ra_min = data["RA"][m_val].min()
ra_max = data["RA"][m_val].max()
dec_min = data["DEC"][m_val].min()
dec_max = data["DEC"][m_val].max()

print("UNIONS coordinate limits")
print(ra_min, ra_max, dec_min, dec_max)


ira_min = data_hsc["ira"].min()
ira_max = data_hsc["ira"].max()
idec_min = data_hsc["idec"].min()
idec_max = data_hsc["idec"].max()

print("HSC coordinate limits")
print(ira_min, ira_max, idec_min, idec_max)

# Mask HSC stars in area of UNIONS - HSC match (rough estimate)
m_area = (
    (data_hsc["ira"] > ra_min)
    & (data_hsc["ira"] < ra_max)
    & (data_hsc["idec"] > dec_min)
    & (data_hsc["idec"] < dec_max)
)

# %%
print(len(data_hsc["ira"]), len(data_hsc["ira"][m_area]))

# %%
# Plot spatial distribution
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))

ms = 1

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

ax.plot(
    data_hsc["ira"][m_area],
    data_hsc["idec"][m_area],
    ".",
    markersize=ms,
    color=colors[1],
)
ax.plot(data_hsc["ira"], data_hsc["idec"], ".", markersize=ms, color=colors[2])


ax.set_title("HSC-SSP stars")
ax.set_xlabel("R.A. [deg]")
ax.set_ylabel("dec [deg]")

# %%
ids_matched = data["object_id"][m_val]

# %%
m_ids_matched = np.in1d(data_hsc["object_id"], data["object_id"][m_val])

# %%
n_bin = 50
title = "HSC stars"
plot_histograms(
    [data_hsc["imag_psf"][m_area], data_hsc["imag_psf"][m_ids_matched]],
    ["all HSC", "matched to UNIONS"],
    title,
    "$i$",
    "frequency",
    [18, 22.5],
    n_bin,
    "mag_i_hsc",
    density=True,
)

# %%
