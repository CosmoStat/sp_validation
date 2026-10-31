"""Loaders from cached cosmo_val products to the plain arrays the figures draw.

Every loader takes explicit paths and returns a dict; none measures, fits or
transforms anything. Cosmological products are read through the blind-stamp
checks of :mod:`sp_validation.sacc_io`, so a figure can only draw vectors
saved under the blind its catalogue is declared under
(:func:`declared_blind`).

:class:`RunProducts` resolves those paths inside a cosmo_val run directory for
one catalogue version; the figure functions never see the directory layout.
"""

import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sp_validation import sacc_io
from sp_validation.b_modes import load_cosebis_results, load_pure_eb_results


def _one(pattern):
    matches = sorted(glob.glob(str(pattern)))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one match for {pattern}, found {matches}")
    return Path(matches[0])


def _maybe(pattern):
    try:
        return _one(pattern)
    except FileNotFoundError:
        return None


@dataclass
class RunProducts:
    """Paths of one version's cached products in a cosmo_val run directory."""

    run_dir: Path
    version: str
    extra_dir: Path | None = None

    @property
    def cv(self):
        return Path(self.run_dir) / "cosmo_val"

    @property
    def assembled(self):
        return self.cv / f"{self.version}.sacc"

    @property
    def pure_eb(self):
        return _one(self.cv / f"{self.version}_eb_minsep=*_data.npz")

    @property
    def cosebis(self):
        return _one(self.cv / f"{self.version}_cosebis_minsep=*_data.npz")

    @property
    def cl_cov(self):
        """NaMaster covariance of the C_ell band powers."""
        prefix = f"pseudo_cl_cov_{self.version}_"
        binning = re.compile(r"(powspace|logspace|linear)_nbins=\d+\.fits")
        matches = [
            p
            for p in self.cv.glob(prefix + "*.fits")
            if binning.fullmatch(p.name[len(prefix) :])
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one {prefix}<binning>.fits, found {matches}"
            )
        return matches[0]

    @property
    def psf_fit(self):
        return _one(self.cv / f"psf_fit_{self.version}_minsep=*.npz")

    @property
    def tau_cov(self):
        return _one(
            self.cv / "rho_tau_stats" / f"cov_tau_{self.version}_tomo_bin_all_*.npy"
        )

    @property
    def leakage(self):
        return self.cv / "leakage_coefficients.json"

    @property
    def summary(self):
        return self.cv / "bmode_summary.json"

    @property
    def xi_covariance(self):
        """OneCovariance reporting-grid ξ± covariance (.dat), if present."""
        return _maybe(
            Path(self.run_dir)
            / "cosmo_inference/data/covariance"
            / f"covariance_{self.version}_ng_minsep=*_masked"
            / f"covariance_{self.version}_ng_minsep=*_masked.dat"
        )

    @property
    def field_maps(self):
        return _maybe(
            Path(self.extra_dir or self.cv) / f"field_maps_{self.version}.npz"
        )

    @property
    def cosebis_grid(self):
        """COSEBIs B PTEs over every reporting-bin cut (:mod:`.cosebis_grid`)."""
        return _maybe(
            Path(self.extra_dir or self.cv) / f"cosebis_grid_{self.version}.npz"
        )

    @property
    def weight_cells(self):
        return _maybe(
            Path(self.extra_dir or self.cv) / f"weight_cells_{self.version}.npz"
        )


def catalogue_name(catalogues, version):
    """The catalogue-config entry describing ``version``: its own, or the one
    its ``_leak_corr`` / ``_seed<N>`` variant derives from (as the workflow
    resolves it)."""
    if version in catalogues:
        return version
    return re.sub(r"_seed\d+$", "", re.sub(r"_leak_corr$", "", version))


def declared_blind(cat_config, version):
    """The blind ``version`` is declared under in the catalogue config.

    The expected blind comes from the declaration, never from a product's own
    stamp, so a stale product of a catalogue that has since been put under a
    blind is refused by the loaders rather than drawn.
    """
    import yaml

    from sp_validation import blinding

    catalogues = yaml.safe_load(Path(cat_config).read_text())
    return blinding.blind_of(catalogues, catalogue_name(catalogues, version))


def load_xi(sacc_path, blind):
    """Reporting-grid ξ± with the covariance carried by the assembled SACC."""
    s = sacc_io.load(sacc_path, blind=blind)
    view = sacc_io.xi_correlation(s, grid="reporting")
    n = len(view.xip)
    cov = np.asarray(view.cov)
    return {
        "theta": view.meanr,
        "xip": view.xip,
        "xim": view.xim,
        "sigma_xip": np.sqrt(np.diag(cov)[:n]),
        "sigma_xim": np.sqrt(np.diag(cov)[n:]),
        "cov": cov,
        "blind": blind,
    }


def load_pure_eb(npz_path, blind):
    r = load_pure_eb_results(npz_path, blind=blind)
    return {
        "theta": r["theta"],
        "left_edges": r["left_edges"],
        "right_edges": r["right_edges"],
        "xip": r["xip"],
        "xim": r["xim"],
        "sigma_xip": np.sqrt(r["var_xip"]),
        "sigma_xim": np.sqrt(r["var_xim"]),
        **{
            f"{xi}_{m}": r[f"{xi}_{m}"]
            for xi in ("xip", "xim")
            for m in ("E", "B", "amb")
        },
        **{
            f"sigma_{xi}_{m}": r[f"std_{xi}_{m}"]
            for xi in ("xip", "xim")
            for m in ("E", "B", "amb")
        },
        "cov": r["cov"],
        "pte_matrices": r["pte_matrices"],
    }


def load_cosebis(npz_path, blind):
    """Every cached COSEBIs scale cut, keyed by ``(theta_min, theta_max)``."""
    cuts, _ = load_cosebis_results(npz_path, blind=blind)
    data = sacc_io.load_npz(npz_path, blind=blind)
    out = {}
    for cut, r in cuts.items():
        n = len(r["Bn"])
        cov = r["cov"]
        out[tuple(float(c) for c in cut)] = {
            "En": r["En"],
            "Bn": r["Bn"],
            "sigma_E": np.sqrt(np.diag(cov)[:n]),
            "sigma_B": np.sqrt(np.diag(cov)[n:]),
            "cov": cov,
            "pte_B": float(r["pte_B"]),
        }
    return {
        "cuts": out,
        "fiducial": tuple(float(c) for c in data["scale_cut"]),
        "nmodes": len(data["Bn"]),
    }


def powspace_edges(ell_eff, lmin=None, lmax=2048, power=0.5):
    """Inclusive multipole edges of the powspace band powers.

    Reconstructs the NaMaster binning of
    :func:`sp_validation.pseudo_cl.make_namaster_bin` and checks that each
    band's mean multipole reproduces the stored effective ell.
    """
    from sp_validation.pseudo_cl import LMIN

    lmin = LMIN if lmin is None else lmin
    n = len(ell_eff)
    ells = np.arange(lmin, lmax + 1)
    bins_ell = np.power(np.linspace(lmin**power, lmax**power, n + 1), 1 / power)
    bpws = np.digitize(ells.astype(float), bins_ell) - 1
    bpws[0], bpws[-1] = 0, n - 1
    lo = np.array([ells[bpws == i].min() for i in range(n)])
    hi = np.array([ells[bpws == i].max() for i in range(n)])
    means = np.array([ells[bpws == i].mean() for i in range(n)])
    if not np.allclose(means, ell_eff, atol=0.51):
        raise ValueError("powspace edges do not reproduce the stored ell_eff")
    return lo, hi


def load_cosebis_grid(npz_path, blind):
    """The COSEBIs scale-cut PTE grid, checked against the expected blind."""
    with np.load(npz_path) as f:
        d = {k: f[k] for k in f.files}
    if str(d.pop("blind")) != blind:
        raise ValueError(f"{npz_path} was computed under another blind")
    return d


def load_cl(sacc_path, cov_path, blind):
    """Band powers, their multipole edges, and the NaMaster covariance blocks."""
    from astropy.io import fits

    s = sacc_io.load(sacc_path, blind=blind)
    ell, cl_ee, cl_bb, cl_eb, _ = sacc_io.get_pseudo_cl(s, (0, 0))
    ell = np.asarray(ell)
    lo, hi = powspace_edges(ell)
    with fits.open(cov_path) as hdus:
        cov = {
            h.name.removeprefix("COVAR_"): np.asarray(h.data)
            for h in hdus
            if h.name.startswith("COVAR_")
        }
    return {
        "ell": ell,
        "ell_lo": lo,
        "ell_hi": hi,
        "EE": cl_ee,
        "BB": cl_bb,
        "EB": cl_eb,
        "cov": cov,
        "sigma_BB": np.sqrt(np.diag(cov["BB_BB"])),
        "sigma_EB": np.sqrt(np.diag(cov["EB_EB"])),
        "sigma_EE": np.sqrt(np.diag(cov["EE_EE"])),
    }


def load_rho_tau(sacc_path, tau_cov_path, psf_fit_path, blind):
    """ρ/τ statistics with their covariance blocks, and the PSF-fit samples."""
    s = sacc_io.load(sacc_path, blind=blind)
    cov = s.covariance.dense
    rho = {}
    for k in range(6):
        theta, p, m = sacc_io.get_rho(s, k)
        idx = s.indices(sacc_io.RHO_PLUS.format(k=k))
        rho[k] = {"p": p, "m": m, "sigma_p": np.sqrt(np.abs(np.diag(cov)[idx]))}
    result = load_tau(sacc_path, tau_cov_path, blind)
    result.update(theta=theta, rho=rho, fit=dict(np.load(psf_fit_path)))
    return result


def load_tau(sacc_path, tau_cov_path, blind):
    """τ-only SACC part and theory covariance; no cosmology or fit required."""
    s = sacc_io.load(sacc_path, blind=blind)
    tau = {}
    for k in (0, 2, 5):
        theta_tau, p, m = sacc_io.get_tau(s, (0,), k)
        tau[k] = {"p": p, "m": m}
    tau_cov = np.load(tau_cov_path)
    n = len(theta_tau)
    for i, k in enumerate((0, 2, 5)):
        var = np.diag(tau_cov)[i * n : (i + 1) * n]
        tau[k]["var_p"] = var
    return {"theta_tau": theta_tau, "tau": tau, "tau_cov": tau_cov}


def load_corrected_tau(run_dir, version, cat_config):
    """Resolve a τ-only run explicitly, avoiding a corrected full-DAG run."""
    stats = Path(run_dir) / "cosmo_val" / "rho_tau_stats"
    part = _one(stats / f"rho_tau_{version}_tomo_bin_all_*.sacc")
    cov = _one(stats / f"cov_tau_{version}_tomo_bin_all_*_th.npy")
    return load_tau(part, cov, declared_blind(cat_config, version))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_correlation(path):
    """A :mod:`sp_validation.dashboard.correlations` product as
    ``{label: {quantity: array}}`` plus ``"provenance"``."""
    with np.load(path, allow_pickle=False) as data:
        out = {"provenance": json.loads(str(data["provenance"]))}
        for label in data["labels"]:
            prefix = f"{label}__"
            out[str(label)] = {
                k[len(prefix) :]: data[k] for k in data.files if k.startswith(prefix)
            }
    return out
