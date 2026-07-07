"""IMAGE_SIMS.

:Description: Multiplicative and additive shear bias from image simulations.

:Author: Martin Kilbinger

"""

import numpy as np
from astropy.io import fits

from sp_validation.cat import match_catalogs_radec


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
            "ra":  data["RA"].copy(),
            "dec": data["Dec"].copy(),
            "e1":  data["e1"].copy(),
            "e2":  data["e2"].copy(),
            "w":   data[w_col].copy(),
        }


class ImageSimMBias:
    """Compute multiplicative and additive shear bias from image simulations.

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys:
        - base : str, path to the grids base directory
        - num : int, run number (e.g. 2 for *_grid_2)
        - catalog_name : str, filename of the cut catalogue
          (default 'shape_catalog_cut_ngmix.fits')
        - shear_amplitude : float, input shear |g| (e.g. 0.02)
        - match_radius_deg : float, matching radius in degrees
        - pair_match : bool, match objects between sheared catalogues
          (default True); if False, use all objects of each catalogue
        - w_col : str, weight column name (default 'w_des')
        - n_bootstrap : int, number of bootstrap resamples for errors
    """

    def __init__(self, config):
        self.cfg = config
        self.g_in = config["shear_amplitude"]
        self.thresh = config.get("match_radius_deg", 0.0002)
        self.pair_match = config.get("pair_match", True)
        self.w_col = config.get("w_col", "w_des")
        self.n_boot = config.get("n_bootstrap", 500)
        self.cats = {}

    def load_catalogs(self, verbose=True):
        """Load the 4 sheared catalogues (1p2z, 1m2z, 1z2p, 1z2m)."""
        grids_dir = self.cfg["base"]
        num = self.cfg["num"]
        cat_name = self.cfg.get("catalog_name", "shape_catalog_cut_ngmix.fits")
        sim_names = ["1p2z", "1m2z", "1z2p", "1z2m"]

        for name in sim_names:
            path = f"{grids_dir}/{name}_grid_{num}/{cat_name}"
            if verbose:
                print(f"  Loading {path}")
            self.cats[name] = _load_cat(path, "e1", self.w_col)
            if verbose:
                print(f"    {len(self.cats[name]['ra'])} objects")

    def print_mean_ellipticities(self):
        """Print weighted mean e1, e2 for each sheared catalogue, as check."""
        print("\nMean weighted ellipticities (all objects):")
        for name, cat in self.cats.items():
            mean_e1 = np.average(cat["e1"], weights=cat["w"])
            mean_e2 = np.average(cat["e2"], weights=cat["w"])
            print(f"  {name}:  <e1> = {mean_e1:+.5f}   <e2> = {mean_e2:+.5f}")

    def _m_c_pair(self, name_p, name_m, comp, verbose=True):
        """Compute m and c for one shear pair and component (0=g1, 1=g2)."""
        e_key = f"e{comp + 1}"

        if self.pair_match:
            idx_p, idx_m = match_catalogs_radec(
                self.cats[name_p]["ra"], self.cats[name_p]["dec"],
                self.cats[name_m]["ra"], self.cats[name_m]["dec"],
                thresh_deg=self.thresh,
            )
            if verbose:
                print(f"  {name_p}: {len(idx_p)} matched  |  {name_m}: {len(idx_m)} matched")
        else:
            idx_p = slice(None)
            idx_m = slice(None)
            if verbose:
                print(
                    f"  no pair-matching: {name_p}: {len(self.cats[name_p][e_key])}"
                    f"  |  {name_m}: {len(self.cats[name_m][e_key])} objects"
                )

        e_p = self.cats[name_p][e_key][idx_p]
        w_p = self.cats[name_p]["w"][idx_p]
        e_m = self.cats[name_m][e_key][idx_m]
        w_m = self.cats[name_m]["w"][idx_m]

        mean_ep = np.average(e_p, weights=w_p)
        mean_em = np.average(e_m, weights=w_m)

        m = (mean_ep - mean_em) / (2 * self.g_in) - 1
        c = (mean_ep + mean_em) / 2

        # Bootstrap errors
        rng = np.random.default_rng(seed=42)
        m_boot = np.empty(self.n_boot)
        c_boot = np.empty(self.n_boot)
        n_p, n_m = len(e_p), len(e_m)
        for i in range(self.n_boot):
            ib_p = rng.integers(0, n_p, n_p)
            ib_m = rng.integers(0, n_m, n_m)
            ep_b = np.average(e_p[ib_p], weights=w_p[ib_p])
            em_b = np.average(e_m[ib_m], weights=w_m[ib_m])
            m_boot[i] = (ep_b - em_b) / (2 * self.g_in) - 1
            c_boot[i] = (ep_b + em_b) / 2

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
            results[f"m{comp + 1}"]     = m
            results[f"m{comp + 1}_err"] = m_err
            results[f"c{comp + 1}"]     = c
            results[f"c{comp + 1}_err"] = c_err
            if verbose:
                print(f"  m{comp+1} = {m:.4f} ± {m_err:.4f}")
                print(f"  c{comp+1} = {c:.4f} ± {c_err:.4f}")
        return results
