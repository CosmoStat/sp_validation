"""Generate LaTeX macros from claim evidence.

Reads evidence.json files and produces:
- claims_macros.tex: LaTeX macro definitions for paper values
- pte_table_results.tex: PTE results table for main text
- pte_table_appendix.tex: PTE table for appendix
- evidence.json: Dashboard dependency tracking
"""

import json
import math
from datetime import datetime
from pathlib import Path

# Version number to word mapping for TeX-safe macro names
# (avoids cleveref/siunitx conflict with numeric names)
VERSION_WORDS = {"5": "Five", "6": "Six", "6.3": "SixThree", "8": "Eight", "11.2": "ElevenTwo", "11.3": "ElevenThree"}


def _parse_version_short(version: str) -> str:
    """Extract short version number from full version string.

    Examples:
        "SP_v1.4.6_leak_corr" → "6"
        "SP_v1.4.11.2" → "11.2"
    """
    return version.split("v1.4.")[1].split("_")[0]


def _format_value(value, bold_threshold=None, italic_threshold=None) -> str:
    """Format a value for LaTeX.

    Parameters
    ----------
    bold_threshold : float, optional
        If set and value is a float below this threshold, wrap in \\textbf{}.
    italic_threshold : float, optional
        If set and value is a float below this threshold, wrap in \\textit{}.
        Checked after bold_threshold, so bold takes precedence.
    """
    if isinstance(value, float):
        if math.isnan(value):
            return "--"
        if abs(value) < 0.001 and value != 0:
            text = rf"\num{{{value:.2e}}}"
        elif abs(value) < 0.1:
            text = rf"\num{{{value:.3f}}}"
        else:
            text = rf"\num{{{value:.2f}}}"
        if bold_threshold is not None and value < bold_threshold:
            return rf"\textbf{{{text}}}"
        if italic_threshold is not None and value < italic_threshold:
            return rf"\textit{{{text}}}"
        return text
    elif isinstance(value, int):
        return rf"\num{{{value}}}"
    elif isinstance(value, str):
        return value.replace("_", r"\_")
    elif isinstance(value, list):
        return ", ".join(_format_value(v) for v in value)
    else:
        return str(value)


def generate_macros(claims_dir: Path, output_paths: list[Path], fiducial_version: str):
    """Generate LaTeX macros from evidence files.

    Macro names are kept simple. The spec (bmodes_paper.md)
    determines which values go into the paper. Fiducial version from config.
    """
    macros = []
    macros.append("% Auto-generated from claim evidence")
    macros.append("% Regenerate: snakemake paper_macros")
    macros.append("% See workflow/config/bmodes_paper.md for paper choices")
    macros.append("")

    # COSEBIS version comparison - extract fiducial version, n=6
    cosebis_path = claims_dir / "cosebis_version_comparison" / "evidence.json"
    if cosebis_path.exists():
        with open(cosebis_path) as f:
            cosebis_ev = json.load(f).get("evidence", {})

        macros.append(f"% cosebis ({fiducial_version}, n=6)")

        # Fiducial scale cut - use pte_6_min (conservative across blinds)
        fiducial = cosebis_ev.get("fiducial", {})
        fid_versions = fiducial.get("versions", {})
        fid_data = fid_versions.get(fiducial_version, {})
        if "pte_6_min" in fid_data:
            macros.append(f"\\newcommand{{\\cosebisfiducialPte}}{{{_format_value(fid_data['pte_6_min'], bold_threshold=0.05)}}}")

        # Full range
        full = cosebis_ev.get("full", {})
        full_versions = full.get("versions", {})
        full_data = full_versions.get(fiducial_version, {})
        if "pte_6_min" in full_data:
            macros.append(f"\\newcommand{{\\cosebisfullPte}}{{{_format_value(full_data['pte_6_min'], bold_threshold=0.05)}}}")

        # Scale cuts from fiducial
        if "scale_cut_arcmin" in fiducial:
            cuts = fiducial["scale_cut_arcmin"]
            macros.append(f"\\newcommand{{\\cosebisthetaMin}}{{{_format_value(cuts[0])}}}")
            macros.append(f"\\newcommand{{\\cosebisthetaMax}}{{{_format_value(cuts[1])}}}")

        macros.append("")

    # Pure E/B data vector - use min across blinds (cache for reuse below)
    eb_path = claims_dir / "pure_eb_data_vector" / "evidence.json"
    eb_fid = {}  # cached for PTE variation section
    if eb_path.exists():
        with open(eb_path) as f:
            eb_ev = json.load(f).get("evidence", {})

        macros.append("% pure_eb_data_vector (min across blinds per spec)")

        # Fiducial PTEs - use pte_joint_min (conservative across blinds)
        eb_fid = eb_ev.get("fiducial", {})
        if "pte_joint_min" in eb_fid:
            macros.append(f"\\newcommand{{\\ebfiducialPte}}{{{_format_value(eb_fid['pte_joint_min'], bold_threshold=0.05)}}}")

        # Full range PTEs
        full = eb_ev.get("full", {})
        if "pte_joint_min" in full:
            macros.append(f"\\newcommand{{\\ebfullPte}}{{{_format_value(full['pte_joint_min'], bold_threshold=0.05)}}}")

        # Scale cuts from fiducial
        if "scale_cut_xip" in eb_fid:
            cuts = eb_fid["scale_cut_xip"]
            macros.append(f"\\newcommand{{\\ebthetaXipMin}}{{\\num{{{cuts[0]}}}}}")
            macros.append(f"\\newcommand{{\\ebthetaXipMax}}{{\\num{{{cuts[1]}}}}}")
        if "scale_cut_xim" in eb_fid:
            cuts = eb_fid["scale_cut_xim"]
            macros.append(f"\\newcommand{{\\ebthetaXimMin}}{{\\num{{{cuts[0]}}}}}")
            macros.append(f"\\newcommand{{\\ebthetaXimMax}}{{\\num{{{cuts[1]}}}}}")

        macros.append("")

    # Pure E/B covariance structure
    eb_cov_path = claims_dir / "pure_eb_covariance" / "evidence.json"
    if eb_cov_path.exists():
        with open(eb_cov_path) as f:
            data = json.load(f)
        ev = data.get("evidence", {})

        macros.append("% pure_eb_covariance (block condition numbers)")

        # Block condition numbers
        block_analysis = ev.get("block_analysis", {})
        if "xi_E" in block_analysis:
            cond = block_analysis["xi_E"]["condition_number"]
            macros.append(f"\\newcommand{{\\ebcovCondE}}{{\\num{{{cond:.1e}}}}}")
        if "xi_B" in block_analysis:
            cond = block_analysis["xi_B"]["condition_number"]
            macros.append(f"\\newcommand{{\\ebcovCondB}}{{\\num{{{cond:.1e}}}}}")
        if "xi_amb" in block_analysis:
            cond = block_analysis["xi_amb"]["condition_number"]
            macros.append(f"\\newcommand{{\\ebcovCondAmb}}{{\\num{{{cond:.1e}}}}}")

        # Full matrix
        if "condition_number" in ev:
            macros.append(f"\\newcommand{{\\ebcovCondFull}}{{\\num{{{ev['condition_number']:.1e}}}}}")
        if "n_bins" in ev:
            macros.append(f"\\newcommand{{\\ebcovNbins}}{{\\num{{{ev['n_bins']}}}}}")

        macros.append("")

    # Covariance blind consistency
    cov_path = claims_dir / "covariance_blind_consistency" / "evidence.json"
    if cov_path.exists():
        with open(cov_path) as f:
            data = json.load(f)
        ev = data.get("evidence", {})

        macros.append("% covariance_blind_consistency")

        # Max deviations across blinds
        xip = ev.get("xip", {})
        xim = ev.get("xim", {})
        xip_max = max(xip.get("B_to_A", {}).get("max_dev", 0), xip.get("C_to_A", {}).get("max_dev", 0))
        xim_max = max(xim.get("B_to_A", {}).get("max_dev", 0), xim.get("C_to_A", {}).get("max_dev", 0))
        macros.append(f"\\newcommand{{\\covXipMaxDev}}{{{_format_value(xip_max * 100)}\\%}}")
        macros.append(f"\\newcommand{{\\covXimMaxDev}}{{{_format_value(xim_max * 100)}\\%}}")

        macros.append("")

    # PTE variation across blinds (reuse eb_fid cached above)
    if eb_fid:
        joint_ptes = [eb_fid.get(f"pte_joint_{b}") for b in ["A", "B", "C"] if f"pte_joint_{b}" in eb_fid]
        if joint_ptes:
            macros.append("% PTE variation across blinds (fiducial scale cuts)")
            joint_delta = max(joint_ptes) - min(joint_ptes)
            macros.append(f"\\newcommand{{\\ebJointPteDelta}}{{{_format_value(joint_delta)}}}")
            macros.append("")

    # Config-space PTE matrices - generate table
    config_pte_path = claims_dir / "config_space_pte_matrices" / "evidence.json"
    if config_pte_path.exists():
        with open(config_pte_path) as f:
            data = json.load(f)
        ev = data.get("evidence", {})
        versions = ev.get("versions", {})

        macros.append("% config_space_pte_matrices (all versions)")
        macros.append("")

        # Filter to leak_corr versions only to avoid duplicate macro definitions
        # (both SP_v1.4.6 and SP_v1.4.6_leak_corr map to configPteSix)
        leak_corr_versions = {k: v for k, v in versions.items() if "leak_corr" in k}

        # Generate individual macros for each version
        for ver, ver_data in leak_corr_versions.items():
            short_ver = _parse_version_short(ver)
            prefix = f"configPte{VERSION_WORDS.get(short_ver, short_ver)}"

            xip = ver_data.get("xip_stats", {})
            xim = ver_data.get("xim_stats", {})
            combined = ver_data.get("combined_stats", {})
            cosebis = ver_data.get("cosebis_stats", {})
            cosebis_20 = ver_data.get("cosebis_20_stats", {})

            # Fiducial PTEs
            bold = 0.05
            if "pte_at_fiducial" in xip:
                macros.append(f"\\newcommand{{\\{prefix}Xip}}{{{_format_value(xip['pte_at_fiducial'], bold_threshold=bold)}}}")
            if "pte_at_fiducial" in xim:
                macros.append(f"\\newcommand{{\\{prefix}Xim}}{{{_format_value(xim['pte_at_fiducial'], bold_threshold=bold)}}}")
            if "pte_at_fiducial" in combined:
                macros.append(f"\\newcommand{{\\{prefix}Combined}}{{{_format_value(combined['pte_at_fiducial'], bold_threshold=bold)}}}")
            if "pte_at_fiducial" in cosebis:
                macros.append(f"\\newcommand{{\\{prefix}Cosebis}}{{{_format_value(cosebis['pte_at_fiducial'], bold_threshold=bold)}}}")
            if "pte_at_fiducial" in cosebis_20:
                macros.append(f"\\newcommand{{\\{prefix}CosebisTwenty}}{{{_format_value(cosebis_20['pte_at_fiducial'], bold_threshold=bold)}}}")

            # Full-range PTEs
            if "pte_at_full_range" in xip:
                macros.append(f"\\newcommand{{\\{prefix}XipFull}}{{{_format_value(xip['pte_at_full_range'], bold_threshold=bold)}}}")
            if "pte_at_full_range" in xim:
                macros.append(f"\\newcommand{{\\{prefix}XimFull}}{{{_format_value(xim['pte_at_full_range'], bold_threshold=bold)}}}")
            if "pte_at_full_range" in combined:
                macros.append(f"\\newcommand{{\\{prefix}CombinedFull}}{{{_format_value(combined['pte_at_full_range'], bold_threshold=bold)}}}")
            if "pte_at_full_range" in cosebis:
                macros.append(f"\\newcommand{{\\{prefix}CosebisFull}}{{{_format_value(cosebis['pte_at_full_range'], bold_threshold=bold)}}}")
            if "pte_at_full_range" in cosebis_20:
                macros.append(f"\\newcommand{{\\{prefix}CosebisTwentyFull}}{{{_format_value(cosebis_20['pte_at_full_range'], bold_threshold=bold)}}}")

        macros.append("")

    # Harmonic-space PTE matrices - generate table
    harmonic_pte_path = claims_dir / "harmonic_space_pte_matrices" / "evidence.json"
    if harmonic_pte_path.exists():
        with open(harmonic_pte_path) as f:
            data = json.load(f)
        ev = data.get("evidence", {})
        versions = ev.get("versions", {})

        macros.append("% harmonic_space_pte_matrices (all versions)")
        macros.append("")

        # Filter to leak_corr versions only (excluding ecut variants) to avoid
        # duplicate macro definitions — both SP_v1.4.6_ecut07_leak_corr and
        # SP_v1.4.6_leak_corr would map to the same clPteSix prefix.
        leak_corr_versions = {
            k: v for k, v in versions.items()
            if "leak_corr" in k and "ecut" not in k
        }

        for ver, ver_data in leak_corr_versions.items():
            short_ver = _parse_version_short(ver)
            prefix = f"clPte{VERSION_WORDS.get(short_ver, short_ver)}"

            # Fiducial PTEs
            if "pte_at_fiducial" in ver_data:
                macros.append(f"\\newcommand{{\\{prefix}Fid}}{{{_format_value(ver_data['pte_at_fiducial'], bold_threshold=0.05)}}}")

            # Full-range PTEs
            if "pte_at_full_range" in ver_data:
                macros.append(f"\\newcommand{{\\{prefix}Full}}{{{_format_value(ver_data['pte_at_full_range'], bold_threshold=0.05)}}}")

        macros.append("")

    # Harmonic-config COSEBIS comparison - B-mode PTEs (modes 1-N) per angular range
    for angular_range in ["full", "fiducial"]:
        harm_cosebis_path = claims_dir / f"harmonic_config_cosebis_comparison_{angular_range}" / "evidence.json"
        if not harm_cosebis_path.exists():
            continue
        with open(harm_cosebis_path) as f:
            data = json.load(f)
        ev = data.get("evidence", {})
        range_suffix = "Full" if angular_range == "full" else "Fid"

        macros.append(f"% harmonic_config_cosebis_comparison_{angular_range} (B-mode COSEBIS PTEs)")
        macros.append("")

        for method, method_prefix in [("harmonic_b_mode_ptes", "harmCosebis"), ("config_b_mode_ptes", "cfgCosebis")]:
            ptes = ev.get(method, {})
            for ver, pte_data in ptes.items():
                short_ver = _parse_version_short(ver)
                word = VERSION_WORDS.get(short_ver, short_ver)
                pte = pte_data.get("pte")
                chi2 = pte_data.get("chi2")
                if pte is not None:
                    macros.append(f"\\newcommand{{\\{method_prefix}Pte{word}{range_suffix}}}{{{_format_value(pte, bold_threshold=0.05)}}}")
                if chi2 is not None:
                    macros.append(f"\\newcommand{{\\{method_prefix}Chisq{word}{range_suffix}}}{{{_format_value(chi2)}}}")

        macros.append("")

    content = "\n".join(macros)
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        print(f"  → {output_path}")


_CONFIG_STATS = ["cosebis_stats", "cosebis_20_stats", "xip_stats", "xim_stats", "combined_stats"]


def _pte_row_cells(pte_key, cfg, harm, bold, italic=None):
    """Build the 6 PTE table cells for one row.

    Column order: COSEBIS n=6, n=20, ξ+^B, ξ-^B, ξ_tot^B, C_ℓ^BB.

    Parameters
    ----------
    pte_key : str
        Evidence key ("pte_at_fiducial" or "pte_at_full_range").
    cfg : dict
        Config-space evidence for one version (may be empty).
    harm : dict
        Harmonic-space evidence for one version (may be empty).
    bold : float
        Bold threshold for PTE formatting.
    italic : float, optional
        Italic threshold for PTE formatting (values between bold and italic).

    Returns
    -------
    list of str
        Six cell strings, each prefixed with "& ".
    """
    cells = []
    for stat in _CONFIG_STATS:
        s = cfg.get(stat, {})
        cells.append(f"& {_format_value(s.get(pte_key, float('nan')), bold_threshold=bold, italic_threshold=italic)}")
    cells.append(f"& {_format_value(harm.get(pte_key, float('nan')), bold_threshold=bold, italic_threshold=italic)}")
    return cells


def generate_pte_tables(claims_dir: Path, output_dir: Path, fiducial_version: str, versions: list, version_labels: dict, config: dict):
    """Generate LaTeX table files from PTE evidence.

    Creates:
    - pte_table_results.tex: Fiducial version PTE summary
    - pte_table_appendix.tex: All versions PTE comparison
    """
    bold = 0.05  # bold PTE values below this threshold

    # Paper-consistent table labels (distinct from short plot labels)
    table_labels = {
        "SP_v1.4.5_leak_corr": "Initial (v1.4.5)",
        "SP_v1.4.6_leak_corr": "Size cut (v1.4.6)",
        "SP_v1.4.8_leak_corr": "Masked (v1.4.8)",
        "SP_v1.4.11.3_leak_corr": "Relaxed flags (v1.4.11.3)",
    }

    # Load config-space PTE evidence
    config_pte_path = claims_dir / "config_space_pte_matrices" / "evidence.json"
    config_data = {}
    if config_pte_path.exists():
        with open(config_pte_path) as f:
            config_data = json.load(f).get("evidence", {}).get("versions", {})

    # Load harmonic-space PTE evidence
    harmonic_pte_path = claims_dir / "harmonic_space_pte_matrices" / "evidence.json"
    harmonic_data = {}
    if harmonic_pte_path.exists():
        with open(harmonic_pte_path) as f:
            harmonic_data = json.load(f).get("evidence", {}).get("versions", {})

    # Get table label for caption
    fid_label = table_labels.get(fiducial_version, version_labels.get(fiducial_version, fiducial_version))

    # Results table (fiducial only) — grouped by statistic family
    if fiducial_version in config_data or fiducial_version in harmonic_data:
        results_table = []
        results_table.append("% Auto-generated PTE summary tabular (Results section)")
        results_table.append("% Regenerate: snakemake paper_macros")
        results_table.append(r"% Wrap in table*/table environment in main tex for float control")
        # Column layout: Scale | COSEBIS n≤6 | COSEBIS n≤20 || ξ+^B | ξ-^B | ξ_tot^B ||| C_ℓ^BB
        results_table.append(r"\begin{tabular}{l cc @{\hskip 8pt} ccc @{\hskip 8pt} c}")
        results_table.append(r"    \hline")
        results_table.append(r"    & \multicolumn{2}{c}{COSEBIS} & \multicolumn{3}{c}{Pure E/B} & Pseudo-$C_\ell$ \\")
        results_table.append(r"    \cmidrule(lr){2-3} \cmidrule(lr){4-6} \cmidrule(l){7-7}")
        results_table.append(r"    Scale cuts & $B_n$ ($n \leq 6$) & $B_n$ ($n \leq 20$) & $\xi_+^{\mathrm{B}}$ & $\xi_-^{\mathrm{B}}$ & $\xi_{\mathrm{tot}}^{\mathrm{B}}$ & $C_\ell^{BB}$ \\")
        results_table.append(r"    \hline")

        cfg = config_data.get(fiducial_version, {})
        harm = harmonic_data.get(fiducial_version, {})

        for pte_key, cut_label in [("pte_at_fiducial", "Fiducial"), ("pte_at_full_range", "Full range")]:
            row = [f"    {cut_label}"]
            row.extend(_pte_row_cells(pte_key, cfg, harm, bold))
            row.append(r" \\")
            results_table.append(" ".join(row))

        results_table.append(r"  \hline")
        results_table.append(r"\end{tabular}")

        results_path = output_dir / "pte_table_results.tex"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text("\n".join(results_table))
        print(f"  → {results_path}")

    # Appendix table (all versions, fiducial + full-range rows) — grouped by statistic family
    if config_data or harmonic_data:
        appendix_table = []
        appendix_table.append("% Auto-generated PTE comparison tabular (Appendix)")
        appendix_table.append("% Regenerate: snakemake paper_macros")
        appendix_table.append(r"% Wrap in table*/table environment in main tex for float control")
        appendix_table.append(r"\begin{tabular}{ll cc @{\hskip 8pt} ccc @{\hskip 8pt} c}")
        appendix_table.append(r"    \hline")
        appendix_table.append(r"    & & \multicolumn{2}{c}{COSEBIS} & \multicolumn{3}{c}{Pure E/B} & Pseudo-$C_\ell$ \\")
        appendix_table.append(r"    \cmidrule(lr){3-4} \cmidrule(lr){5-7} \cmidrule(l){8-8}")
        appendix_table.append(r"    Version & Scale cuts & $B_n$ ($n \leq 6$) & $B_n$ ($n \leq 20$) & $\xi_+^{\mathrm{B}}$ & $\xi_-^{\mathrm{B}}$ & $\xi_{\mathrm{tot}}^{\mathrm{B}}$ & $C_\ell^{BB}$ \\")
        appendix_table.append(r"    \hline")

        # Filter to only leak_corr versions (those with labels in version_labels)
        table_versions = [v for v in versions if v in version_labels]
        for i, ver in enumerate(table_versions):
            label = table_labels.get(ver, version_labels.get(ver, ver))
            if ver == fiducial_version:
                label = f"{label} (fiducial)"

            for pte_key, cut_label in [("pte_at_fiducial", "Fiducial"), ("pte_at_full_range", "Full range")]:
                # Only show version label on first row of each pair
                row_label = label if pte_key == "pte_at_fiducial" else ""
                row = [f"    {row_label} & {cut_label}"]
                cfg = config_data.get(ver, {})
                harm = harmonic_data.get(ver, {})
                row.extend(_pte_row_cells(pte_key, cfg, harm, bold))
                row.append(r" \\")
                appendix_table.append(" ".join(row))

            # Visual separator between versions (except after last)
            if i < len(table_versions) - 1:
                appendix_table.append(r"    \noalign{\vskip 2pt}")

        appendix_table.append(r"  \hline")
        appendix_table.append(r"\end{tabular}")

        appendix_path = output_dir / "pte_table_appendix.tex"
        appendix_path.parent.mkdir(parents=True, exist_ok=True)
        appendix_path.write_text("\n".join(appendix_table))
        print(f"  → {appendix_path}")


def generate_evidence(
    spec_id: str,
    spec_path: str,
    depends_on: list[str],
    claims_dir: Path,
    output_path: Path,
):
    """Generate evidence.json for dashboard dependency tracking."""
    # Collect summary from dependent claims
    summary = {}
    for dep in depends_on:
        dep_evidence = claims_dir / dep / "evidence.json"
        if dep_evidence.exists():
            with open(dep_evidence) as f:
                data = json.load(f)
            summary[dep] = {
                "has_evidence": True,
                "generated": data.get("generated"),
            }
        else:
            summary[dep] = {"has_evidence": False}

    evidence = {
        "spec_id": spec_id,
        "spec_path": spec_path,
        "depends_on": depends_on,
        "generated": datetime.now().isoformat(),
        "evidence": {
            "type": "paper_integration",
            "description": "Aggregates quantitative claims for paper reporting",
            "dependencies_summary": summary,
        },
        "output": {"macros": "claims_macros.tex"},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(evidence, f, indent=2)
    print(f"  → {output_path}")


if __name__ == "__main__":
    # When run via snakemake
    claims_dir = Path(snakemake.params.claims_dir)  # noqa: F821
    config = snakemake.config  # noqa: F821
    fiducial_version = config["fiducial"]["version"]
    versions = config["versions"]
    version_labels = config["plotting"]["version_labels"]

    # Separate macro file from PTE tables and evidence
    # Only claims_macros.tex gets macro content; PTE tables generated separately
    macro_file = [Path(p) for p in snakemake.output if p.endswith("claims_macros.tex")]  # noqa: F821
    evidence_outputs = [Path(p) for p in snakemake.output if p.endswith("evidence.json")]  # noqa: F821

    print(f"Generating macros from {claims_dir}")
    generate_macros(claims_dir, macro_file, fiducial_version)

    # Generate PTE tables (separate files, not macro content)
    if macro_file:
        paper_dir = macro_file[0].parent
        print(f"Generating PTE tables to {paper_dir}")
        generate_pte_tables(claims_dir, paper_dir, fiducial_version, versions, version_labels, config)

    # Generate evidence.json if requested
    # Dependencies derived from snakemake inputs (rules.X.output declarations)
    rule_inputs = snakemake.input.keys()  # noqa: F821
    input_deps = [k for k in rule_inputs if k.endswith("_evidence") or k == "covariance_evidence"]
    depends_on = [d.replace("_evidence", "") for d in input_deps]

    for evidence_path in evidence_outputs:
        spec_id = evidence_path.parent.name  # e.g., xi_cosmology_paper
        generate_evidence(
            spec_id=spec_id,
            spec_path=f"workflow/config/{spec_id}.md",
            depends_on=depends_on,
            claims_dir=claims_dir,
            output_path=evidence_path,
        )
