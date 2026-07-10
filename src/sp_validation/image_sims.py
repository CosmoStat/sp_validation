"""IMAGE_SIMS.

:Description: Multiplicative and additive shear bias from image simulations.

:Author: Martin Kilbinger

"""

import numpy as np
from astropy.io import fits

from sp_validation.catalog import match_catalogs_radec

# Conventional campaign layout, used only when the config carries no branch map
# (e.g. the synthetic-recovery tests).  In a workflow run the branches and pairs
# come from manifest.yaml via the m_bias config; nothing about the injected
# shear is hard-coded on the estimator's side.
_DEFAULT_BRANCHES = ["1z2z", "1p2z", "1m2z", "1z2p", "1z2m"]
_DEFAULT_PAIRS = [
    ("1p2z", "1m2z", 0),  # g1 component, index 0 → e1
    ("1z2p", "1z2m", 1),  # g2 component, index 1 → e2
]


def _load_cat(path, e_col, w_col):
    """Load RA, Dec, ellipticity component and weight from a FITS catalogue.

    ``w_col=None`` gives every object unit weight — the no-weighting mode for
    m-bias runs (#227: shape weights are excluded from sim calibration).
    """
    with fits.open(path) as hdul:
        data = hdul[1].data
        return {
            "ra": data["RA"].copy(),
            "dec": data["Dec"].copy(),
            "e1": data["e1"].copy(),
            "e2": data["e2"].copy(),
            "w": data[w_col].copy() if w_col else np.ones(len(data["RA"])),
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
          (default 'shape_catalog_cut_ngmix.fits')
        - shear_amplitude : float, input shear |g| (from manifest.yaml)
        - branches : list of str, branch names in load order (incl. the
          unsheared reference); defaults to the conventional 5-branch layout
        - pairs : list of dicts {plus, minus, component}, the +/- sheared
          branch pairing per component; defaults to the conventional pairs
        - match_radius_deg : float, matching radius in degrees
        - pair_match : bool, match objects between the +g and -g sheared
          catalogues (default True); if False, use all objects of each
          catalogue (the paired per-object cancellation is then unavailable)
        - w_col : str or None, weight column name (default 'w_des');
          None → unit weights (no weighting, per the #227 verdict)
        - n_bootstrap : int, number of bootstrap resamples for errors
    """

    def __init__(self, config):
        self.cfg = config
        self.g_in = config["shear_amplitude"]
        self.thresh = config.get("match_radius_deg", 0.0002)
        self.pair_match = config.get("pair_match", True)
        self.w_col = config.get("w_col", "w_des")
        self.n_boot = config.get("n_bootstrap", 500)
        # Branch list and pairing come from the manifest-derived config
        # (``branches`` / ``pairs``); fall back to the conventional layout only
        # when neither is given.  ``branches`` fixes the catalogue load order;
        # ``pairs`` fixes which sims difference into which component.
        self.sim_names = list(config.get("branches", _DEFAULT_BRANCHES))
        if config.get("pairs"):
            self.pairs = [
                (p["plus"], p["minus"], p["component"]) for p in config["pairs"]
            ]
        else:
            self.pairs = list(_DEFAULT_PAIRS)
        self.cats = {}

    def load_catalogs(self, verbose=True):
        """Load the 5 sheared and reference catalogues."""
        grids_dir = self.cfg["grids_dir"]
        num = self.cfg["num"]
        cat_name = self.cfg.get("catalog_name", "shape_catalog_cut_ngmix.fits")
        # ``sim_names`` (incl. the unsheared reference) comes from the config's
        # branch map. The +g/-g pool estimator pairs the sheared sims directly;
        # the reference is loaded for completeness and null-test diagnostics.
        for name in self.sim_names:
            path = f"{grids_dir}/{name}_grid_{num}/{cat_name}"
            if verbose:
                print(f"  Loading {path}")
            self.cats[name] = _load_cat(path, "e1", self.w_col)
            if verbose:
                print(f"    {len(self.cats[name]['ra'])} objects")

    def print_mean_ellipticities(self):
        """Print weighted mean e1, e2 for each catalogue, as a check.

        Works with unit weights too (``w_col=None`` → w all ones), in which
        case these are the plain unweighted means.
        """
        print("\nMean weighted ellipticities (all objects):")
        for name, cat in self.cats.items():
            mean_e1 = np.average(cat["e1"], weights=cat["w"])
            mean_e2 = np.average(cat["e2"], weights=cat["w"])
            print(f"  {name}:  <e1> = {mean_e1:+.5f}   <e2> = {mean_e2:+.5f}")

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

        With ``pair_match=False`` the +g and -g sims are *not* matched: every
        object of each catalogue is used, so the per-object cancellation is
        lost and m, c fall back to differencing/summing the two independent
        weighted means. The paired bootstrap likewise cannot be applied (the
        two arrays generally have different lengths), so each side is resampled
        independently per replicate.
        """
        e_key = f"e{comp + 1}"

        if self.pair_match:
            # Match the +g and -g sims to each other: same galaxies, opposite
            # shear. This is a nearest-neighbour match within `thresh`, not a
            # strict bijection -- on grid sims galaxies are well separated so
            # pairs are effectively 1:1 (verified ~99% co-located to <0.05" on
            # SKiLLS grid_1); on denser fields a small fraction could share a
            # +g partner and dilute the cancellation.
            idx_p, idx_m = match_catalogs_radec(
                self.cats[name_p]["ra"],
                self.cats[name_p]["dec"],
                self.cats[name_m]["ra"],
                self.cats[name_m]["dec"],
                thresh_deg=self.thresh,
            )
            if verbose:
                print(f"  {name_p} <-> {name_m}: {len(idx_p)} paired objects")
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

        rng = np.random.default_rng(seed=42)
        m_boot = np.empty(self.n_boot)
        c_boot = np.empty(self.n_boot)

        if self.pair_match:
            # Per-object shear-differenced (-> m) and summed (-> c)
            # ellipticity, with a symmetric per-pair weight.
            w = 0.5 * (w_p + w_m)
            d = (e_p - e_m) / (2 * self.g_in) - 1
            s = (e_p + e_m) / 2

            m = np.average(d, weights=w)
            c = np.average(s, weights=w)

            # Paired bootstrap: resample objects once and apply the same draw
            # to both sims, so the per-object cancellation in `d` is preserved
            # in the error estimate.
            n = len(d)
            for i in range(self.n_boot):
                ib = rng.integers(0, n, n)
                m_boot[i] = np.average(d[ib], weights=w[ib])
                c_boot[i] = np.average(s[ib], weights=w[ib])
        else:
            # No matching: difference/sum the two independent weighted means.
            mean_ep = np.average(e_p, weights=w_p)
            mean_em = np.average(e_m, weights=w_m)

            m = (mean_ep - mean_em) / (2 * self.g_in) - 1
            c = (mean_ep + mean_em) / 2

            # Unpaired bootstrap: the +g and -g arrays generally differ in
            # length, so resample each side independently per replicate.
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
        for name_p, name_m, comp in self.pairs:
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
