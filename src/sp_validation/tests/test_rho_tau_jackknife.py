"""Jackknife layouts and covariance averaging on small synthetic catalogues."""

from pathlib import Path

import numpy as np
from astropy.table import Table

from sp_validation import rho_tau


def test_jackknife_draws_use_distinct_shared_layouts(tmp_path, monkeypatch):
    rng = np.random.default_rng(37)
    n = 400
    star = Table(
        dict(
            RA=rng.uniform(10, 14, n),
            Dec=rng.uniform(10, 14, n),
            p1=rng.normal(0, 0.03, n),
            p2=rng.normal(0, 0.03, n),
            s1=rng.normal(0, 0.04, n),
            s2=rng.normal(0, 0.04, n),
            tp=np.ones(n),
            ts=np.full(n, 1.1),
        )
    )
    gal = Table(
        dict(
            RA=rng.uniform(10, 14, n),
            Dec=rng.uniform(10, 14, n),
            e1=rng.normal(0, 0.2, n),
            e2=rng.normal(0, 0.2, n),
            w=rng.uniform(0.2, 1, n),
        )
    )
    star.write(tmp_path / "stars.fits")
    gal.write(tmp_path / "galaxies.fits")
    config = {
        "v": dict(
            patch_number=7,
            psf=dict(
                path=str(tmp_path / "stars.fits"),
                ra_col="RA",
                dec_col="Dec",
                e1_PSF_col="p1",
                e2_PSF_col="p2",
                e1_star_col="s1",
                e2_star_col="s2",
                PSF_size="tp",
                star_size="ts",
            ),
            shear=dict(
                path=str(tmp_path / "galaxies.fits"),
                ra_col="RA",
                dec_col="Dec",
                e1_col="e1",
                e2_col="e2",
                w_col="w",
            ),
        )
    }
    layouts, catalogs, chunks = [], [], {"rho": [], "tau": []}
    compute = rho_tau.RhoStat.compute_rho_stats
    save = np.save

    def record_compute(self, catalog_id, *args, **kwargs):
        cats = self.catalogs.catalogs_dict
        psf = cats[f"psf_{catalog_id}"]
        layouts.append(psf.patch_centers.copy())
        catalogs.append(psf)
        for prefix in ("psf_error", "psf_size_error", "gal"):
            np.testing.assert_array_equal(
                cats[f"{prefix}_{catalog_id}"].patch_centers, psf.patch_centers
            )
        # Materialise TreeCorr's patch cache before the next draw.
        assert len(psf.patches) == 7
        return compute(self, catalog_id, *args, **kwargs)

    def record_save(path, arr, *args, **kwargs):
        for kind in chunks:
            if Path(path).stem in {f"cov_{kind}_tau{i}" for i in range(3)}:
                chunks[kind].append(np.array(arr).copy())
        return save(path, arr, *args, **kwargs)

    monkeypatch.setattr(rho_tau.RhoStat, "compute_rho_stats", record_compute)
    monkeypatch.setattr(np, "save", record_save)
    rho_tau.get_jackknife_cov(
        config,
        "v",
        dict(
            min_sep=1,
            max_sep=150,
            nbins=2,
            sep_units="arcmin",
            num_threads=1,
            cross_patch_weight="match",
        ),
        str(tmp_path),
        "rho",
        "tau",
        npatch=7,
        ncov=3,
    )
    assert len(layouts) == 3
    assert all(
        not np.array_equal(layouts[i], layouts[j]) for i in range(3) for j in range(i)
    )
    assert len({id(cat) for cat in catalogs}) == 3
    for kind in chunks:
        assert len(chunks[kind]) == 3
        assert any(not np.array_equal(chunks[kind][0], cov) for cov in chunks[kind][1:])
        np.testing.assert_allclose(
            np.load(tmp_path / f"cov_{kind}_{kind}_jk.npy"),
            np.mean(chunks[kind], axis=0),
            rtol=1e-13,
            atol=0,
        )
