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


def _load_cat(path, w_cols):
    """Load RA, Dec, ellipticities and weight columns from a FITS catalogue.

    The weight scheme "none" gives uniform weights.
    """
    with fits.open(path) as hdul:
        data = hdul[1].data
        cat = {
            "ra":  data["RA"].copy(),
            "dec": data["Dec"].copy(),
            "e1":  data["e1"].copy(),
            "e2":  data["e2"].copy(),
            "w":   {},
        }
        for w_col in w_cols:
            if w_col == "none":
                cat["w"][w_col] = np.ones(len(cat["ra"]))
            else:
                cat["w"][w_col] = data[w_col].copy()
        return cat


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
        - w_cols : list of str, weight schemes to compute; "none" means
          uniform weights (default ["none", "w_iv"]); the first
          entry is the primary result
        - w_col : str, deprecated single weight column; used as
          [w_col] if w_cols is not given
        - n_bootstrap : int, number of bootstrap resamples for errors
    """

    def __init__(self, config):
        self.cfg = config
        self.g_in = config["shear_amplitude"]
        self.thresh = config.get("match_radius_deg", 0.0002)
        self.pair_match = config.get("pair_match", True)
        w_cols = config.get("w_cols")
        if not w_cols:
            w_col = config.get("w_col")
            w_cols = [w_col] if w_col else ["none", "w_iv"]
        self.w_cols = [str(w) for w in w_cols]
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
            self.cats[name] = _load_cat(path, self.w_cols)
            if verbose:
                print(f"    {len(self.cats[name]['ra'])} objects")

    def print_mean_ellipticities(self):
        """Print mean e1, e2 for each sheared catalogue and weight scheme."""
        for w_col in self.w_cols:
            print(f"\nMean ellipticities (all objects, weights: {w_col}):")
            for name, cat in self.cats.items():
                mean_e1 = np.average(cat["e1"], weights=cat["w"][w_col])
                mean_e2 = np.average(cat["e2"], weights=cat["w"][w_col])
                print(f"  {name}:  <e1> = {mean_e1:+.5f}   <e2> = {mean_e2:+.5f}")

    def _m_c_pair(self, name_p, name_m, comp, verbose=True):
        """Compute m and c for one shear pair and component (0=g1, 1=g2).

        Matches the pair once, then computes results for each weight
        scheme. Returns a dict weight scheme -> (m, m_err, c, c_err).
        """
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
        e_m = self.cats[name_m][e_key][idx_m]
        n_p, n_m = len(e_p), len(e_m)

        # Same bootstrap indices for all weight schemes
        rng = np.random.default_rng(seed=42)
        ib_p = rng.integers(0, n_p, (self.n_boot, n_p))
        ib_m = rng.integers(0, n_m, (self.n_boot, n_m))

        res = {}
        for w_col in self.w_cols:
            w_p = self.cats[name_p]["w"][w_col][idx_p]
            w_m = self.cats[name_m]["w"][w_col][idx_m]

            mean_ep = np.average(e_p, weights=w_p)
            mean_em = np.average(e_m, weights=w_m)

            m = (mean_ep - mean_em) / (2 * self.g_in) - 1
            c = (mean_ep + mean_em) / 2

            m_boot = np.empty(self.n_boot)
            c_boot = np.empty(self.n_boot)
            for i in range(self.n_boot):
                ep_b = np.average(e_p[ib_p[i]], weights=w_p[ib_p[i]])
                em_b = np.average(e_m[ib_m[i]], weights=w_m[ib_m[i]])
                m_boot[i] = (ep_b - em_b) / (2 * self.g_in) - 1
                c_boot[i] = (ep_b + em_b) / 2

            res[w_col] = (m, np.std(m_boot), c, np.std(c_boot))

        return res

    def run(self, verbose=True):
        """Compute m and c for both shear components and all weight schemes.

        Returns
        -------
        dict with keys m1, m1_err, c1, c1_err, m2, m2_err, c2, c2_err
        for the primary (first) weight scheme, plus "weights" holding
        the same keys per weight scheme.
        """
        results = {"weights": {w: {} for w in self.w_cols}}
        for name_p, name_m, comp in _PAIRS:
            label = f"g{comp + 1}"
            if verbose:
                print(f"\n--- {label}: {name_p} / {name_m} ---")
            res = self._m_c_pair(name_p, name_m, comp, verbose=verbose)
            for w_col, (m, m_err, c, c_err) in res.items():
                results["weights"][w_col][f"m{comp + 1}"]     = m
                results["weights"][w_col][f"m{comp + 1}_err"] = m_err
                results["weights"][w_col][f"c{comp + 1}"]     = c
                results["weights"][w_col][f"c{comp + 1}_err"] = c_err
                if verbose:
                    print(f"  [{w_col}] m{comp+1} = {m:.4f} ± {m_err:.4f}"
                          f"   c{comp+1} = {c:.4f} ± {c_err:.4f}")

        # Primary scheme results at top level
        results.update(results["weights"][self.w_cols[0]])
        return results
