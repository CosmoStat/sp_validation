"""A harmonic 3x2pt survey: tracers, n(z), densities, noise, footprints, bands,
input cosmology, priors and scale cuts, read from one recipe YAML.

The recipe (``config/cloe/unions_v2_lsx.yaml`` is UNIONS v2 sources with LSx
lenses) names each input with its source, its paths through
:mod:`sp_validation.data_roots`. Sources are read from a release: n(z) on its
nodes (linear between them), ``n_eff`` and ``sigma_e`` per component from
``survey_tomo.json``, and the binary footprint, averaged onto ``nside`` as a
coverage fraction. Lenses are given by their total count, area, bin edges, linear
biases and relative counts per bin; their n(z) is a top hat in estimated redshift
times ``dN/dz ∝ z^2 exp[-(z/z0)^1.5]``, smoothed by a Gaussian scatter
``sigma_z (1 + z)``; their footprint is the source footprint above the galactic
latitude that leaves ``area_deg2``.

Fields are ordered lenses then sources, ``g0..g4, l0..l3``, and spectra are the
upper triangle of that order (:attr:`Survey.pairs`).
"""

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from .. import sacc_io
from ..data_roots import load_yaml, localise

ARCMIN2_PER_SR = (180 * 60 / np.pi) ** 2
SR_TO_DEG2 = (180 / np.pi) ** 2


@dataclass
class Survey:
    recipe: dict

    @classmethod
    def load(cls, path):
        return cls(localise(load_yaml(path)))

    def __getitem__(self, key):
        return self.recipe[key]

    # ---- tracers ---------------------------------------------------------- #
    @property
    def n_lens(self):
        return len(self["lenses"]["bias"])

    @cached_property
    def n_source(self):
        return len(self.source_survey)

    @property
    def fields(self):
        """``g<i>`` (lens bins) then ``l<i>`` (source bins)."""
        return [f"g{i}" for i in range(self.n_lens)] + [
            f"l{i}" for i in range(self.n_source)
        ]

    @property
    def pairs(self):
        """Every pair of :attr:`fields`, upper triangle in field order."""
        f = self.fields
        return [(f[i], f[j]) for i in range(len(f)) for j in range(i, len(f))]

    @staticmethod
    def label(field):
        """The theory's tracer label: ``g2 -> G2``, ``l0 -> L0``."""
        return {"g": "G", "l": "L"}[field[0]] + field[1:]

    @staticmethod
    def sacc_name(field):
        """The SACC tracer name: ``g2 -> lens_2``, ``l0 -> source_0``."""
        i = int(field[1:])
        return sacc_io.lens_name(i) if field[0] == "g" else sacc_io.source_name(i)

    # ---- n(z) ------------------------------------------------------------- #
    @cached_property
    def z(self):
        lo, hi, step = self["z_grid"]
        return np.arange(lo, hi + step / 2, step)

    @cached_property
    def nz(self):
        """{field: n(z) on :attr:`z`}, each normalised (trapezoid)."""
        from scipy.special import erf

        z, lens = self.z, self["lenses"]
        out = {}
        edges = lens["edges"]
        width = lens["sigma_z"] * (1 + z) * np.sqrt(2)
        for i in range(self.n_lens):
            top = erf((edges[i + 1] - z) / width) - erf((edges[i] - z) / width)
            out[f"g{i}"] = self.dndz(z) * 0.5 * top
        table = np.loadtxt(self["sources"]["nz"])
        for i in range(self.n_source):
            out[f"l{i}"] = np.interp(z, table[:, 0], table[:, i + 1])
        return {
            k: np.maximum(v, 0) / np.trapezoid(np.maximum(v, 0), z)
            for k, v in out.items()
        }

    def dndz(self, z):
        """The lenses' redshift distribution before binning, unnormalised."""
        return z**2 * np.exp(-((z / self["lenses"]["dndz_z0"]) ** 1.5))

    # ---- densities and noise ---------------------------------------------- #
    @cached_property
    def source_survey(self):
        samples = json.loads(Path(self["sources"]["survey"]).read_text())["samples"]
        return [samples[f"bin{i + 1}"] for i in range(len(samples))]

    @property
    def n_eff(self):
        """Source effective densities, arcmin^-2."""
        return np.array([b["n_eff"] for b in self.source_survey])

    @property
    def sigma_e(self):
        """Source shape noise per component."""
        return np.array([b["sigma_e_per_component"] for b in self.source_survey])

    @property
    def lens_density(self):
        """Lens densities per bin, arcmin^-2."""
        lens = self["lenses"]
        frac = np.asarray(lens["fractions"], float)
        return lens["count"] * frac / frac.sum() / lens["area_deg2"] / 3600

    def density(self, field):
        """Density of ``field`` per steradian."""
        i = int(field[1:])
        n = self.lens_density[i] if field[0] == "g" else self.n_eff[i]
        return n * ARCMIN2_PER_SR

    def variance(self, field):
        """Per-object variance: 1 for counts, sigma_e^2 per shear component."""
        return 1.0 if field[0] == "g" else self.sigma_e[int(field[1:])] ** 2

    # ---- footprints ------------------------------------------------------- #
    def masks(self):
        """{"g": lens, "l": source} coverage-fraction maps at ``nside``."""
        import healpy as hp

        nside = self["nside"]
        binary = hp.read_map(self["sources"]["footprint"], dtype=np.float32)
        source = hp.ud_grade(binary, nside).astype(float)
        theta, phi = hp.pix2ang(nside, np.arange(len(source)))
        theta_g, _ = hp.Rotator(coord=["C", "G"])(theta, phi)
        b = np.abs(90 - np.degrees(theta_g))

        def area(m):
            return 4 * np.pi * SR_TO_DEG2 * m.mean()

        cut = next(
            c
            for c in np.arange(10, 60, 0.25)
            if area(source * (b > c)) <= self["lenses"]["area_deg2"]
        )
        return {"g": source * (b > cut), "l": source}

    # ---- bands ------------------------------------------------------------ #
    @cached_property
    def edges(self):
        """Band edges: ``n`` bands uniform in sqrt(ell), ``ell_min..ell_max``."""
        b = self["bands"]
        root = np.linspace(np.sqrt(b["ell_min"]), np.sqrt(b["ell_max"]), b["n"] + 1)
        return np.round(root**2).astype(int)

    @cached_property
    def binning(self):
        """(n_band, lmax + 1): band i averages ell in [edge_i, edge_i+1) over 2l+1."""
        e, ell = self.edges, np.arange(self["lmax"] + 1)
        out = np.zeros((len(e) - 1, len(ell)))
        for i in range(len(e) - 1):
            s = slice(e[i], e[i + 1])
            out[i, s] = (2 * ell[s] + 1) / np.sum(2 * ell[s] + 1)
        return out

    # ---- parameters ------------------------------------------------------- #
    @property
    def truth(self):
        """The input point: cosmology, lens biases and slopes, zero shifts and m."""
        out = dict(self["cosmology"])
        for i in range(self.n_lens):
            out[f"bias_{i}"] = self["lenses"]["bias"][i]
            out[f"mag_slope_{i}"] = self["lenses"]["mag_slope"]
            out[f"delta_z_G_{i}"] = 0.0
        for i in range(self.n_source):
            out[f"m_{i}"] = 0.0
            out[f"delta_z_L_{i}"] = 0.0
        return out

    @property
    def reference(self):
        """The input cosmology in the physical basis of the probe's ``reference``."""
        c = self["cosmology"]
        h = c["h"]
        return dict(
            Omega_c=c["omch2"] / h**2,
            Omega_b=c["ombh2"] / h**2,
            h=h,
            n_s=c["n_s"],
            A_s=1e-10 * np.exp(c["logA"]),
            m_nu=c["m_nu"],
        )

    @property
    def priors(self):
        """cloerun priors: ``priors.cosmology`` as given, then each entry of
        ``priors.lenses`` and ``priors.sources`` once per bin (``bias_{i}``)."""
        p = self["priors"]
        out = dict(p["cosmology"])
        for group, n in (("lenses", self.n_lens), ("sources", self.n_source)):
            for name, prior in p[group].items():
                out.update({name.format(i=i): prior for i in range(n)})
        return out

    def probe(self, **kw):
        """The probe section of a cloerun run (:mod:`.probe`)."""
        return dict(
            kind="sp_validation.cloe.probe",
            reference=self.reference,
            threads=1,
            chunk=1,
            **self["probe"],
            **kw,
        )
