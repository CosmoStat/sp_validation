"""IMAGE_SIMS.

:Description: Multiplicative and additive shear bias from image simulations.

:Author: Martin Kilbinger

"""

import numpy as np
from astropy.io import fits

from sp_validation.catalog import match_catalogs_radec

# Shear component for each simulation pair
_PAIRS = [
    ("1p2z", "1m2z", 0),  # g1 component, index 0 → e1
    ("1z2p", "1z2m", 1),  # g2 component, index 1 → e2
]


def _load_cat(path, e_col, w_col):
    """Load RA, Dec, ellipticity component and weight from a FITS catalogue."""
    with fits.open(path) as hdul:
        data = hdul[1].data
        return {
            "ra": data["RA"].copy(),
            "dec": data["Dec"].copy(),
            "e1": data["e1"].copy(),
            "e2": data["e2"].copy(),
            "w": data[w_col].copy(),
        }


class ImageSimMBias:
    """Compute multiplicative and additive shear bias from image simulations.

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys:
        - grids_dir : str, path to the grids directory
        - num : int, run number (e.g. 2 for *_grid_2)
        - catalog_name : str, filename of the cut catalogue
        - shear_amplitude : float, input shear |g| (e.g. 0.02)
        - match_radius_deg : float, matching radius in degrees
        - w_col : str, weight column name (default 'w_des')
        - n_bootstrap : int, number of bootstrap resamples for errors
    """

    def __init__(self, config):
        self.cfg = config
        self.g_in = config["shear_amplitude"]
        self.thresh = config.get("match_radius_deg", 0.0002)
        self.w_col = config.get("w_col", "w_des")
        self.n_boot = config.get("n_bootstrap", 500)
        self.cats = {}

    def load_catalogs(self, verbose=True):
        """Load the 5 sheared and reference catalogues."""
        grids_dir = self.cfg["grids_dir"]
        num = self.cfg["num"]
        cat_name = self.cfg["catalog_name"]
        sim_names = ["1z2z", "1p2z", "1m2z", "1z2p", "1z2m"]

        for name in sim_names:
            path = f"{grids_dir}/{name}_grid_{num}/{cat_name}"
            if verbose:
                print(f"  Loading {path}")
            self.cats[name] = _load_cat(path, "e1", self.w_col)
            if verbose:
                print(f"    {len(self.cats[name]['ra'])} objects")

    def _m_c_pair(self, name_p, name_m, comp, verbose=True):
        """Compute m and c for one shear pair and component (0=g1, 1=g2).

        Paired ("pool") estimator. The +g and -g simulations inject opposite
        input shear on the *same* galaxies, so matching them directly by
        RA/Dec yields a one-to-one correspondence. Differencing the two
        ellipticities per object,

            m = <(e_+ - e_-) / (2 g_in) - 1> ,   c = <(e_+ + e_-) / 2> ,

        cancels the intrinsic shape (sigma_e ~ 0.3) object-by-object in the
        multiplicative term, leaving only measurement noise -- so sigma(m)
        shrinks by ~sigma_e/sigma_meas relative to differencing two
        independent means. (The additive term c is a *sum*, so intrinsic
        shape does not cancel there and its error stays shape-noise limited.)
        """
        e_key = f"e{comp + 1}"

        # Match the +g and -g sims to each other: same galaxies, opposite shear.
        idx_p, idx_m = match_catalogs_radec(
            self.cats[name_p]["ra"],
            self.cats[name_p]["dec"],
            self.cats[name_m]["ra"],
            self.cats[name_m]["dec"],
            thresh_deg=self.thresh,
        )

        if verbose:
            print(f"  {name_p} <-> {name_m}: {len(idx_p)} paired objects")

        e_p = self.cats[name_p][e_key][idx_p]
        w_p = self.cats[name_p]["w"][idx_p]
        e_m = self.cats[name_m][e_key][idx_m]
        w_m = self.cats[name_m]["w"][idx_m]

        # Per-object shear-differenced (-> m) and summed (-> c) ellipticity,
        # with a symmetric per-pair weight.
        w = 0.5 * (w_p + w_m)
        d = (e_p - e_m) / (2 * self.g_in) - 1
        s = (e_p + e_m) / 2

        m = np.average(d, weights=w)
        c = np.average(s, weights=w)

        # Paired bootstrap: resample objects once and apply the same draw to
        # both sims, so the per-object cancellation in `d` is preserved in
        # the error estimate.
        rng = np.random.default_rng(seed=42)
        n = len(d)
        m_boot = np.empty(self.n_boot)
        c_boot = np.empty(self.n_boot)
        for i in range(self.n_boot):
            ib = rng.integers(0, n, n)
            m_boot[i] = np.average(d[ib], weights=w[ib])
            c_boot[i] = np.average(s[ib], weights=w[ib])

        return m, np.std(m_boot), c, np.std(c_boot)

    def run(self, verbose=True):
        """Compute m and c for both shear components.

        Returns
        -------
        dict with keys m1, m1_err, c1, c1_err, m2, m2_err, c2, c2_err
        """
        results = {}
        for name_p, name_m, comp in _PAIRS:
            label = f"g{comp + 1}"
            if verbose:
                print(f"\n--- {label}: {name_p} / {name_m} ---")
            m, m_err, c, c_err = self._m_c_pair(name_p, name_m, comp, verbose=verbose)
            results[f"m{comp + 1}"] = m
            results[f"m{comp + 1}_err"] = m_err
            results[f"c{comp + 1}"] = c
            results[f"c{comp + 1}_err"] = c_err
            if verbose:
                print(f"  m{comp + 1} = {m:.4f} ± {m_err:.4f}")
                print(f"  c{comp + 1} = {c:.4f} ± {c_err:.4f}")
        return results
