# %%
# Calibrate comprehensive catalogue

# %%
from IPython import get_ipython

# %%
# enable autoreload for interactive sessions
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("reload_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
    ipython.run_line_magic("reload_ext", "log_cell_time")

import numpy as np
import time
from astroquery.gaia import Gaia
import astropy.units
from astropy.coordinates import SkyCoord
from textwrap import dedent


def get_bins(results, key="phot_g_mean_mag", n_bins=3):

    quantiles_raw = np.percentile(results[key], np.linspace(0, 100, n_bins + 1))
    # round
    quantiles = np.round(quantiles_raw, 1)

    print("Magnitude bin edges:")
    subsets = []
    for i in range(n_bins):
        this_subset = results[
            (results[key] >= quantiles[i]) & (results[key] < quantiles[i + 1])
        ]
        subsets.append(this_subset)

        print(
            f"  Bin {i+1}: {quantiles[i]:.2f}, {quantiles[i+1]:.2f}, N={len(this_subset)}"
        )

    return subsets, quantiles


def do_query(do_async=True, nmax=3_000_000):

    # Define query
    query = dedent(
        f"""
        SELECT TOP {nmax} ra, dec, phot_g_mean_mag, ruwe
		FROM gaiadr3.gaia_source
		WHERE dec > 30
			AND dec < 75
			AND phot_g_mean_mag < 20
    		AND phot_g_mean_mag IS NOT NULL
            AND ruwe < 1.4
        ORDER BY random_index
	"""
    ).strip()


    # Launch the query
    if not do_async:
        print("Retrieveing GAIA data (sync)")
        job = Gaia.launch_job(query)
    else:
        print("Retrieveing GAIA data (async)")
        job = Gaia.launch_job_async(query)

        while not job.is_finished():
            print(f"Job status: {job.get_phase()}")
            time.sleep(10)

    results = job.get_results()

    print(f"Retrieved {len(results)} sources")
    print(results[:10])  # Show first 10 results

    return results


def write_subsets(subsets, quantiles):

    for idx, subset in enumerate(subsets):

        if idx == 0:
            out_path = f"gaia_stars_g_smaller_{quantiles[idx + 1]}.fits"
        elif idx == len(subsets) - 1:
            out_path = f"gaia_stars_g_larger_{quantiles[idx]}.fits"
        else:
            out_path = f"gaia_stars_g_in_{quantiles[idx]}_{quantiles[idx + 1]}.fits"
        print(f"Writing {len(subset)} objects to {out_path}")
        subset.write(out_path, format="fits", overwrite=True)


# %% Main program
results = do_query(do_async=True)

subsets, quantiles = get_bins(results)

write_subsets(subsets, quantiles)
