"""Linear pseudo-Cl generators must produce the number of bands they are asked for."""

import importlib.util
import inspect
from pathlib import Path

import pytest

import sp_validation
from sp_validation.pseudo_cl import make_namaster_bin, pseudo_cl_geometry

REPO = Path(__file__).resolve().parents[4]
ACTIVE_REPO = Path(sp_validation.__file__).resolve().parents[2]
if ACTIVE_REPO != REPO:
    REPO = ACTIVE_REPO
SCRIPTS = ["generate_pseudo_cl.py", "generate_pseudo_cl_cov.py"]


class _Captured(Exception):
    pass


def _forwarded_kwargs(script, tmp_path, nside, nbins):
    """Run the generator up to CosmologyValidation(...) and return its kwargs."""
    path = REPO / "workflow" / "scripts" / script
    if not path.exists():
        pytest.skip(f"Workflow generator missing: {script}")
    spec = importlib.util.spec_from_file_location(f"gen_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    seen = {}

    def fake_cv(**kw):
        seen.update(kw)
        raise _Captured

    mod.CosmologyValidation = fake_cv
    func = getattr(mod, path.stem)
    params = inspect.signature(func).parameters
    kw = dict(
        version="SP_test",
        cat_config="unused",
        nside=nside,
        binning="linear",
        nbins=nbins,
    )
    if "out_path" in params:
        kw["out_path"] = str(tmp_path / "out" / "pseudo_cl.sacc")
    if "output_dir" in params:
        kw["output_dir"] = str(tmp_path / "out")
    with pytest.raises(_Captured):
        func(**kw)
    return seen


_WRONG_BAND_COUNT = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=("#381: linear nbins maps to the wrong band count"),
)


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize(
    "nside, nbins",
    [
        pytest.param(1024, 32, marks=_WRONG_BAND_COUNT),
        pytest.param(512, 32, marks=_WRONG_BAND_COUNT),
        (1024, 20),
        pytest.param(512, 20, marks=_WRONG_BAND_COUNT),
    ],
)
def test_linear_nbins_yields_nbins_bands(script, nside, nbins, tmp_path):
    """A ``binning=linear, nbins=N`` request must yield an N-band NmtBin.

    The workflow names products ``pseudo_cl_{ver}_linear_nbins=N`` and the
    generators document ``nbins`` as the number of ell bins, so the bandpower
    count the estimator builds from the forwarded kwargs (via the package's own
    ``pseudo_cl_geometry`` + ``make_namaster_bin``, exactly as
    ``CosmologyValidation.get_namaster_bin`` does) must equal N. The generators
    instead convert N to ``ell_step = (2048 - 2) // N``, a span that ignores
    both LMIN=8 and nside, so N=32 gives ell_step 63 -> 33 bands at nside 1024
    (2040 ells / 63 rounded up) and 17 at nside 512. The nside=1024,
    nbins=20 case happens to produce 20 bands on develop; leave that control
    unmarked so an unrelated binning regression is still caught.
    """
    kw = _forwarded_kwargs(script, tmp_path, nside, nbins)
    lmin, lmax, b_lmax = pseudo_cl_geometry(kw["nside"])
    b = make_namaster_bin(
        lmin,
        lmax,
        b_lmax,
        kw["binning"],
        ell_step=kw.get("ell_step", 10),
        n_ell_bins=kw.get("n_ell_bins", 32),
        power=kw.get("power", 0.5),
    )
    n_bands = b.get_n_bands()
    assert n_bands == nbins, (
        f"{script}: binning=linear nbins={nbins} nside={nside} forwarded "
        f"ell_step={kw.get('ell_step')} -> NmtBin has {n_bands} bands, not {nbins}"
    )
