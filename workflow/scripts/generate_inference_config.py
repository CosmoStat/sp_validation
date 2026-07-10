"""Generate a CosmoSIS pipeline ini by filling a template's ``[DEFAULT]`` section.

Dual-mode, like ``assemble_sacc.py``. Under Snakemake (``script:`` directive) the
injected ``snakemake`` object supplies the template, output path and DEFAULT
substitutions; as a standalone CLI (argparse) the same fill runs from explicit
flags.

The template carries ``%(KEY)s`` interpolation placeholders (SCRATCH, FITS_FILE
or SACC_FILE, COSMOSIS_DIR, SP_VALIDATION_MODULES) in its module sections; this
script prepends the concrete ``KEY = value`` lines into ``[DEFAULT]`` so
CosmoSIS's ConfigParser resolves them at load. It is deliberately plain text
processing — appending lines after the ``[DEFAULT]`` header, the same idiom as
``pipeline.sh``'s ``sed -i "/^\\[DEFAULT\\]/a\\KEY = value"`` — rather than a
configparser round-trip, which would strip the template's comments and its
``%(...)s`` interpolation.

``SP_VALIDATION_MODULES`` is resolved from ``sp_validation.__file__``'s parent so
the generated ini points at the installed package's module directory (where
``sacc_like_unions.py`` lives) regardless of checkout location.
"""

import argparse
from pathlib import Path


def _sp_validation_modules():
    """The directory holding the sp_validation CosmoSIS module files.

    Resolved from the installed package so the generated ini finds
    ``sacc_like_unions.py`` wherever sp_validation is installed.
    """
    import sp_validation

    return str(Path(sp_validation.__file__).resolve().parent)


def generate_inference_config(template_path, out_path, substitutions):
    """Write ``out_path`` from ``template_path`` with ``substitutions`` in DEFAULT.

    Parameters
    ----------
    template_path : str or Path
        The pipeline ini template (carries ``%(KEY)s`` placeholders).
    out_path : str or Path
        Destination ini.
    substitutions : dict
        ``{KEY: value}`` lines prepended into the template's ``[DEFAULT]``
        section. Every referenced ``%(KEY)s`` in the template must have a value
        here (COSMOSIS_DIR already sits in the template's DEFAULT and may be
        overridden). ``None`` values are dropped (an absent optional key).
    """
    lines = Path(template_path).read_text().splitlines(keepends=True)

    header = "[DEFAULT]"
    default_idx = next(
        (i for i, line in enumerate(lines) if line.strip() == header), None
    )
    if default_idx is None:
        raise ValueError(f"template {template_path} has no [DEFAULT] section")

    # The end of the DEFAULT section: the next `[section]` header, or EOF.
    section_end = next(
        (
            i
            for i in range(default_idx + 1, len(lines))
            if lines[i].lstrip().startswith("[")
        ),
        len(lines),
    )

    # A key the template already declares in DEFAULT is REPLACED in place (e.g. the
    # template's placeholder COSMOSIS_DIR); a genuinely-new key is prepended just
    # after the header. This avoids a duplicate DEFAULT key, which CosmoSIS's
    # ConfigParser (strict) rejects.
    wanted = {key: value for key, value in substitutions.items() if value is not None}
    remaining = dict(wanted)
    for i in range(default_idx + 1, section_end):
        stripped = lines[i].lstrip()
        if not stripped or stripped.startswith(("#", ";", "[")):
            continue
        existing_key = stripped.split("=", 1)[0].strip()
        if existing_key in remaining:
            lines[i] = f"{existing_key} = {remaining.pop(existing_key)}\n"

    prepended = [f"{key} = {value}\n" for key, value in remaining.items()]
    out_lines = lines[: default_idx + 1] + prepended + lines[default_idx + 1 :]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(out_lines))
    print(
        f"Wrote {out_path} from {template_path} "
        f"({len(wanted)} DEFAULT keys, {len(prepended)} new)"
    )
    return str(out_path)


def _substitutions(scratch, cosmosis_dir, *, fits_file=None, sacc_file=None):
    """Assemble the DEFAULT substitution dict, resolving SP_VALIDATION_MODULES.

    ``fits_file`` (2pt_like path) and ``sacc_file`` (sacc_like path) are mutually
    the data-file placeholder for their template; whichever the template
    references is filled, the other left absent.
    """
    return {
        "SCRATCH": scratch,
        "FITS_FILE": fits_file,
        "SACC_FILE": sacc_file,
        "COSMOSIS_DIR": cosmosis_dir,
        "SP_VALIDATION_MODULES": _sp_validation_modules(),
    }


def _from_snakemake(smk):
    p = smk.params
    generate_inference_config(
        template_path=smk.input[0]
        if not hasattr(smk.input, "template")
        else smk.input.template,
        out_path=str(smk.output[0]),
        substitutions=_substitutions(
            scratch=p["scratch"],
            cosmosis_dir=p["cosmosis_dir"],
            fits_file=p.get("fits_file", None),
            sacc_file=p.get("sacc_file", None),
        ),
    )


def _from_cli(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate a CosmoSIS pipeline ini from a template + DEFAULT subs."
    )
    ap.add_argument("--template", required=True, help="Pipeline ini template")
    ap.add_argument("--out", required=True, help="Output ini path")
    ap.add_argument("--scratch", required=True, help="SCRATCH value")
    ap.add_argument("--cosmosis-dir", required=True, help="COSMOSIS_DIR value")
    ap.add_argument("--fits-file", default=None, help="FITS_FILE (2pt_like path)")
    ap.add_argument("--sacc-file", default=None, help="SACC_FILE (sacc_like path)")
    a = ap.parse_args(argv)
    generate_inference_config(
        template_path=a.template,
        out_path=a.out,
        substitutions=_substitutions(
            scratch=a.scratch,
            cosmosis_dir=a.cosmosis_dir,
            fits_file=a.fits_file,
            sacc_file=a.sacc_file,
        ),
    )


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821 — injected by Snakemake's script: directive
    except NameError:
        _from_cli()
    else:
        _from_snakemake(snakemake)  # noqa: F821
