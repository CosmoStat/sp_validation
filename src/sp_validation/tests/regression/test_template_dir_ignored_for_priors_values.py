"""--template-dir must select the priors/values files, not only the pipeline ini."""

import configparser
import importlib.util
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import sp_validation

REPO = Path(__file__).resolve().parents[4]
# Cross-checkout runs must exercise the package selected by PYTHONPATH.
ACTIVE_REPO = Path(sp_validation.__file__).resolve().parents[2]
if ACTIVE_REPO != REPO:
    REPO = ACTIVE_REPO

SCRIPT = REPO / "cosmo_inference" / "scripts" / "cosmosis_fitting.py"
CHECKOUT_TEMPLATES = REPO / "cosmo_inference" / "cosmosis_config" / "templates"


def _load_script():
    if not SCRIPT.exists():
        pytest.skip(f"{SCRIPT} not found (run from an sp_validation checkout)")
    spec = importlib.util.spec_from_file_location("cosmosis_fitting", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.xfail(
    strict=True,
    reason="#379: template-dir is ignored for priors and values paths",
)
def test_template_dir_custom_priors_reach_generated_pipeline(tmp_path):
    """A custom --template-dir must supply the priors and values CosmoSIS reads.

    We copy the checkout's PSF pipeline template, values and priors into a custom
    template directory and change the priors there (a sentinel line), then
    generate the pipeline ini through the script's own ``_generate_ini_file``.
    CosmoSIS resolves the ``[pipeline] values/priors`` paths relative to its
    launch directory; for every launch directory, including the checkout's
    ``cosmo_inference/``, those paths must land on the files in the custom
    directory. Otherwise the user's m/Delta-z/alpha priors are silently replaced
    by the checkout's defaults, which shifts every posterior run with
    ``--template-dir``.
    """
    mod = _load_script()

    custom = tmp_path / "custom_templates"
    custom.mkdir()
    for name in ("cosmosis_pipeline_A_psf.ini", "values_psf.ini", "priors_psf.ini"):
        shutil.copy(CHECKOUT_TEMPLATES / name, custom / name)
    with open(custom / "priors_psf.ini", "a") as f:
        f.write("\n; CUSTOM_PRIOR_SENTINEL\n")

    args = SimpleNamespace(
        template_dir=str(custom),
        output_config_dir=str(tmp_path / "out"),
        config_name_base="t",
        data_dir=str(tmp_path / "data"),
        config_relative_fits="x.fits",
        cosmosis_root="t",
        use_rho_tau=True,
    )
    mod._generate_ini_file(
        args, "cosmosis_pipeline_A_psf.ini", "priors_psf.ini", "values_psf.ini"
    )

    cp = configparser.ConfigParser(interpolation=None, strict=False)
    cp.read(tmp_path / "out" / "cosmosis_pipeline_t.ini")
    launch_dirs = [REPO / "cosmo_inference", tmp_path / "elsewhere"]
    for launch in launch_dirs:
        for key, fname in (("priors", "priors_psf.ini"), ("values", "values_psf.ini")):
            raw = cp["pipeline"][key]
            resolved = (launch / raw).resolve()
            assert resolved == (custom / fname).resolve(), (
                f"[pipeline] {key} = {raw!r} resolves (launch dir {launch}) to "
                f"{resolved}, not the --template-dir file {custom / fname}"
            )
