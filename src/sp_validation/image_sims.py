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


# Weight-scheme name that means "no weighting": every object gets unit weight.
# ``None`` (from a YAML ``null``) is accepted as an alias, so the fiducial
# unweighted primary scheme can be written either ``none`` or ``null``.
_UNWEIGHTED = "none"


def _is_unweighted(scheme):
    """True for the unit-weight scheme (``"none"`` or ``None``)."""
    return scheme is None or scheme == _UNWEIGHTED


def _load_cat(path, w_cols):
    """Load RA, Dec, ellipticities and per-scheme weights from a FITS catalogue.

    Reads the ``e1``/``e2`` columns, which the calibration stage writes as the
    *calibrated* shear estimate ``g = R^-1 g_uncal - c`` (metacal response and
    additive-bias corrected) -- not the raw ``e1_uncal``/``e2_uncal`` columns
    that sit alongside them in the same catalogue.  The bias this estimator
    measures is therefore the *residual* m/c left after the chain's own metacal
    calibration, not the raw pre-calibration bias.

    ``w_cols`` is the list of weight schemes to load. The scheme ``"none"``
    (equivalently a ``None``/``null`` entry) gives every object unit weight --
    the no-weighting mode for m-bias runs (#227: shape weights are excluded from
    sim calibration); any other entry is read as a FITS column name. The weights
    come back as a dict keyed by scheme so one catalogue load serves every
    scheme in a multi-weight run.
    """
    with fits.open(path) as hdul:
        data = hdul[1].data
        cat = {
            "ra": data["RA"].copy(),
            "dec": data["Dec"].copy(),
            "e1": data["e1"].copy(),
            "e2": data["e2"].copy(),
            "w": {},
        }
        for scheme in w_cols:
            cat["w"][scheme] = (
                np.ones(len(cat["ra"]))
                if _is_unweighted(scheme)
                else data[scheme].copy()
            )
        return cat


class ImageSimMBias:
    """Compute multiplicative and additive shear bias from image simulations.

    The estimator consumes the *calibrated* ``e1``/``e2`` columns (the metacal
    response- and additive-bias-corrected shear ``g = R^-1 g_uncal - c``), so
    the headline m/c is the **residual** bias remaining after the chain's own
    metacal calibration, not the raw pre-calibration bias.

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
        - match_radius_deg : float, matching radius in degrees (required)
        - pair_match : bool, match objects between the +g and -g sheared
          catalogues (required); if False, use all objects of each
          catalogue (the paired per-object cancellation is then unavailable)
        - w_cols : list of str, weight schemes to compute in one run
          (required); ``"none"`` (or a ``null`` entry) means unit weights,
          any other entry is a FITS column name. The **first** entry is the
          primary result surfaced at the top level of ``run()``'s output. Our
          fiducial run leads with the unweighted scheme (``["none", ...]``),
          per the #227 verdict that shape weights are excluded from sim
          calibration; the unweighted m also avoids the ``cov(w, e)`` residual
          weighted estimators carry on constant-shear sims.
        - w_col : str or None, *deprecated* single weight scheme; accepted for
          back-compat and used as ``[w_col]`` only when ``w_cols`` is absent.
        - n_bootstrap : int, number of bootstrap resamples for errors (required)
        - bootstrap_seed : int, seed for the per-pair bootstrap RNG (required);
          makes the bootstrap errors bit-reproducible. The resample indices are
          drawn once per pair and shared across every weight scheme, so the
          schemes differ only in their weighting, never in their draws.

    The science knobs (``match_radius_deg``, ``pair_match``, ``w_cols``,
    ``n_bootstrap``, ``bootstrap_seed``) are read with no in-code default: a
    missing one is a config bug and raises ``KeyError`` at construction, per the
    fail-fast contract (the workflow emits every one into the m_bias config).
    The lone exception is the deprecated ``w_col``, which is honoured as a
    fallback so pre-``w_cols`` configs still run.
    """

    def __init__(self, config):
        self.cfg = config
        self.g_in = config["shear_amplitude"]
        self.thresh = config["match_radius_deg"]
        self.pair_match = config["pair_match"]
        # ``w_cols`` is the required science key. A pre-``w_cols`` config that
        # still carries the deprecated scalar ``w_col`` is honoured as a
        # single-scheme run; only a config with neither raises (fail-fast).
        if "w_cols" in config:
            w_cols = config["w_cols"]
        else:
            w_cols = [config["w_col"]]
        # Normalise a ``None``/``null`` entry to the canonical "none" name so
        # results key off a string; downstream still treats it as unit weights.
        self.w_cols = [_UNWEIGHTED if _is_unweighted(w) else str(w) for w in w_cols]
        self.n_boot = config["n_bootstrap"]
        self.boot_seed = config["bootstrap_seed"]
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
            self.cats[name] = _load_cat(path, self.w_cols)
            if verbose:
                print(f"    {len(self.cats[name]['ra'])} objects")

    def print_mean_ellipticities(self):
        """Print the mean e1, e2 for each catalogue and weight scheme, as a check.

        The unweighted scheme (``"none"``) gives the plain unweighted means.
        """
        for scheme in self.w_cols:
            print(f"\nMean ellipticities (all objects, weights: {scheme}):")
            for name, cat in self.cats.items():
                mean_e1 = np.average(cat["e1"], weights=cat["w"][scheme])
                mean_e2 = np.average(cat["e2"], weights=cat["w"][scheme])
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
        e_m = self.cats[name_m][e_key][idx_m]

        # Draw the bootstrap resample indices *once*, before the weight-scheme
        # loop, and reuse them for every scheme -- the schemes then differ only
        # in their weighting, never in their draws (so a scheme comparison is a
        # clean weighting comparison). Pre-drawing the full ``(n_boot, n)`` block
        # in one call is bit-identical to drawing ``rng.integers(0, n, n)`` once
        # per replicate (numpy fills the block row-major), so the numbers match a
        # single-scheme, per-iteration bootstrap to the last bit.
        rng = np.random.default_rng(seed=self.boot_seed)
        if self.pair_match:
            n = len(e_p)
            ib = rng.integers(0, n, (self.n_boot, n))
        else:
            n_p, n_m = len(e_p), len(e_m)
            ib_p = rng.integers(0, n_p, (self.n_boot, n_p))
            ib_m = rng.integers(0, n_m, (self.n_boot, n_m))

        res = {}
        for scheme in self.w_cols:
            w_p = self.cats[name_p]["w"][scheme][idx_p]
            w_m = self.cats[name_m]["w"][scheme][idx_m]
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

                # Paired bootstrap: the same object draw is applied to both
                # sims, so the per-object cancellation in `d` is preserved in
                # the error estimate.
                for i in range(self.n_boot):
                    m_boot[i] = np.average(d[ib[i]], weights=w[ib[i]])
                    c_boot[i] = np.average(s[ib[i]], weights=w[ib[i]])
            else:
                # No matching: difference/sum the two independent weighted means.
                mean_ep = np.average(e_p, weights=w_p)
                mean_em = np.average(e_m, weights=w_m)

                m = (mean_ep - mean_em) / (2 * self.g_in) - 1
                c = (mean_ep + mean_em) / 2

                # Unpaired bootstrap: the +g and -g arrays generally differ in
                # length, so each side is resampled independently per replicate.
                for i in range(self.n_boot):
                    ep_b = np.average(e_p[ib_p[i]], weights=w_p[ib_p[i]])
                    em_b = np.average(e_m[ib_m[i]], weights=w_m[ib_m[i]])
                    m_boot[i] = (ep_b - em_b) / (2 * self.g_in) - 1
                    c_boot[i] = (ep_b + em_b) / 2

            res[scheme] = (m, np.std(m_boot), c, np.std(c_boot))

        return res

    def run(self, verbose=True):
        """Compute m and c for both shear components and every weight scheme.

        Returns
        -------
        dict
            ``results["weights"][scheme]`` holds ``m1, m1_err, c1, c1_err,
            m2, m2_err, c2, c2_err`` for each weight scheme. The primary
            (first) scheme's keys are also mirrored at the top level, so a
            reader that wants the headline m/c never has to know the scheme
            name.
        """
        results = {"weights": {scheme: {} for scheme in self.w_cols}}
        for name_p, name_m, comp in self.pairs:
            label = f"g{comp + 1}"
            if verbose:
                print(f"\n--- {label}: {name_p} / {name_m} ---")
            res = self._m_c_pair(name_p, name_m, comp, verbose=verbose)
            for scheme, (m, m_err, c, c_err) in res.items():
                w = results["weights"][scheme]
                w[f"m{comp + 1}"] = m
                w[f"m{comp + 1}_err"] = m_err
                w[f"c{comp + 1}"] = c
                w[f"c{comp + 1}_err"] = c_err
                if verbose:
                    print(
                        f"  [{scheme}] m{comp + 1} = {m:.4f} ± {m_err:.4f}"
                        f"   c{comp + 1} = {c:.4f} ± {c_err:.4f}"
                    )

        # Mirror the primary (first) scheme's m/c at the top level: the headline
        # result reads out without knowing the scheme name, and a downstream
        # gate keyed on the old flat keys still finds them.
        results.update(results["weights"][self.w_cols[0]])
        return results
