"""Tests for the CosmoSIS inference-config generator.

The generator (:mod:`workflow.scripts.generate_inference_config`) fills a
pipeline ini template's ``[DEFAULT]`` section with concrete paths so CosmoSIS's
ConfigParser resolves the template's ``%(KEY)s`` placeholders. These tests need
no cosmosis: they check that the substituted DEFAULT keys land, that the module
file paths the templates reference exist on disk, and that no ``%(...)s``
placeholder is left unresolved after filling.
"""

import configparser
import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "workflow" / "scripts" / "generate_inference_config.py"
_CONFIG_DIR = _REPO / "cosmo_inference" / "cosmosis_config"
_SACC_TEMPLATE = _CONFIG_DIR / "cosmosis_pipeline_A_ia_sacc.ini"
_FITS_TEMPLATE = _CONFIG_DIR / "cosmosis_pipeline_A_ia.ini"


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_inference_cfg", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load_generator()


def _read_interpolated(ini_path):
    """Parse the generated ini with interpolation ON (the way CosmoSIS reads it).

    CosmoSIS uses ``%(KEY)s`` BasicInterpolation with case-*preserved* keys, so
    set ``optionxform = str`` (stdlib configparser lowercases keys by default,
    which would break the uppercase ``%(FITS_FILE)s`` / ``%(SACC_FILE)s``
    lookups). The default BasicInterpolation then raises if a referenced key is
    missing — exactly the failure we want to catch.
    """
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(ini_path)
    return parser


def test_sacc_template_defaults_filled(tmp_path):
    """The generated sacc ini carries the substituted DEFAULT keys and resolves."""
    out = tmp_path / "gen_sacc.ini"
    gen.generate_inference_config(
        _SACC_TEMPLATE,
        out,
        gen._substitutions(
            scratch="/scratch/run",
            cosmosis_dir="/csl",
            sacc_file="/data/v1.sacc",
        ),
    )
    text = out.read_text()
    assert "SCRATCH = /scratch/run" in text
    assert "SACC_FILE = /data/v1.sacc" in text
    assert "COSMOSIS_DIR = /csl" in text
    assert "SP_VALIDATION_MODULES = " in text
    # FITS_FILE is None for the sacc path — must be dropped, not written as "None".
    assert "FITS_FILE" not in text

    parser = _read_interpolated(out)
    # The sacc_like data_file interpolates SACC_FILE; load_nz_sacc too.
    assert parser["sacc_like"]["data_file"] == "/data/v1.sacc"
    assert parser["load_nz_sacc"]["nz_file"] == "/data/v1.sacc"
    assert parser["sacc_like"]["csl_dir"] == "/csl"


def test_fits_template_defaults_filled(tmp_path):
    """The generated 2pt_like ini carries the substituted DEFAULT keys and resolves."""
    out = tmp_path / "gen_fits.ini"
    gen.generate_inference_config(
        _FITS_TEMPLATE,
        out,
        gen._substitutions(
            scratch="/scratch/run",
            cosmosis_dir="/csl",
            fits_file="/data/v1.fits",
        ),
    )
    text = out.read_text()
    assert "SCRATCH = /scratch/run" in text
    assert "FITS_FILE = /data/v1.fits" in text
    assert "SACC_FILE" not in text

    parser = _read_interpolated(out)
    assert parser["2pt_like"]["data_file"] == "/data/v1.fits"
    assert parser["load_nz_fits"]["nz_file"] == "/data/v1.fits"


def test_sacc_template_module_file_exists():
    """The sacc_like module file the generated ini points at exists on disk.

    ``SP_VALIDATION_MODULES`` resolves to the installed package dir;
    ``sacc_like_unions.py`` must live there (it is the shim CosmoSIS loads).
    """
    modules = Path(gen._sp_validation_modules())
    assert (modules / "sacc_like_unions.py").is_file()


def test_all_placeholders_resolve(tmp_path):
    """No ``%(...)s`` placeholder survives interpolation in either template.

    A missing DEFAULT key would make ConfigParser raise on access; iterate every
    option in every section to force resolution of all placeholders.
    """
    for template, subs in (
        (
            _SACC_TEMPLATE,
            gen._substitutions(scratch="/s", cosmosis_dir="/csl", sacc_file="/d.sacc"),
        ),
        (
            _FITS_TEMPLATE,
            gen._substitutions(scratch="/s", cosmosis_dir="/csl", fits_file="/d.fits"),
        ),
    ):
        out = tmp_path / (template.stem + ".gen.ini")
        gen.generate_inference_config(template, out, subs)
        parser = _read_interpolated(out)
        for section in parser.sections():
            for key in parser[section]:
                value = parser[section][key]  # raises if a placeholder is unresolved
                assert "%(" not in value, f"[{section}] {key} = {value}"


def test_no_default_section_raises(tmp_path):
    """A template with no [DEFAULT] section is a loud error."""
    bad = tmp_path / "bad.ini"
    bad.write_text("[pipeline]\nmodules = a b c\n")
    with pytest.raises(ValueError, match="DEFAULT"):
        gen.generate_inference_config(bad, tmp_path / "out.ini", {"SCRATCH": "/s"})
