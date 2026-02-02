"""Generate LaTeX macros from claim evidence.

Reads evidence.json files and produces claims_macros.tex.
Macro names are simple; the spec determines what goes in the paper.
Also generates evidence.json for dashboard dependency tracking.
"""

import json
from datetime import datetime
from pathlib import Path

# Version number to word mapping for TeX-safe macro names
# (avoids cleveref/siunitx conflict with numeric names)
VERSION_WORDS = {"5": "Five", "6": "Six", "8": "Eight", "11.2": "ElevenTwo"}


def _format_value(value) -> str:
    """Format a value for LaTeX."""
    if isinstance(value, float):
        if value != value:  # NaN check
            return "--"
        if abs(value) < 0.001 and value != 0:
            return f"{value:.2e}"
        elif abs(value) < 0.1:
            return f"{value:.3f}"
        else:
            return f"{value:.2f}"
    elif isinstance(value, int):
        return str(value)
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
            data = json.load(f)
        ev = data.get("evidence", {})

        macros.append(f"% cosebis ({fiducial_version}, n=6)")

        # Fiducial scale cut - use pte_6_min (conservative across blinds)
        fiducial = ev.get("fiducial", {})
        fid_versions = fiducial.get("versions", {})
        fid_data = fid_versions.get(fiducial_version, {})
        if "pte_6_min" in fid_data:
            macros.append(f"\\newcommand{{\\cosebisfiducialPte}}{{{_format_value(fid_data['pte_6_min'])}}}")

        # Full range
        full = ev.get("full", {})
        full_versions = full.get("versions", {})
        full_data = full_versions.get(fiducial_version, {})
        if "pte_6_min" in full_data:
            macros.append(f"\\newcommand{{\\cosebisfullPte}}{{{_format_value(full_data['pte_6_min'])}}}")

        # Scale cuts from fiducial
        if "scale_cut_arcmin" in fiducial:
            cuts = fiducial["scale_cut_arcmin"]
            macros.append(f"\\newcommand{{\\cosebisthetaMin}}{{{_format_value(cuts[0])}}}")
            macros.append(f"\\newcommand{{\\cosebisthetaMax}}{{{_format_value(cuts[1])}}}")

        macros.append("")

    # Pure E/B data vector - use min across blinds
    eb_path = claims_dir / "pure_eb_data_vector" / "evidence.json"
    if eb_path.exists():
        with open(eb_path) as f:
            data = json.load(f)
        ev = data.get("evidence", {})

        macros.append("% pure_eb_data_vector (min across blinds per spec)")

        # Fiducial PTEs - use pte_joint_min (conservative across blinds)
        fid = ev.get("fiducial", {})
        if "pte_joint_min" in fid:
            macros.append(f"\\newcommand{{\\ebfiducialPte}}{{{_format_value(fid['pte_joint_min'])}}}")

        # Full range PTEs
        full = ev.get("full", {})
        if "pte_joint_min" in full:
            macros.append(f"\\newcommand{{\\ebfullPte}}{{{_format_value(full['pte_joint_min'])}}}")

        # Scale cuts from fiducial
        if "scale_cut_xip" in fid:
            cuts = fid["scale_cut_xip"]
            macros.append(f"\\newcommand{{\\ebthetaXipMin}}{{{cuts[0]}}}")
            macros.append(f"\\newcommand{{\\ebthetaXipMax}}{{{cuts[1]}}}")
        if "scale_cut_xim" in fid:
            cuts = fid["scale_cut_xim"]
            macros.append(f"\\newcommand{{\\ebthetaXimMin}}{{{cuts[0]}}}")
            macros.append(f"\\newcommand{{\\ebthetaXimMax}}{{{cuts[1]}}}")

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
            macros.append(f"\\newcommand{{\\ebcovCondE}}{{{cond:.1e}}}")
        if "xi_B" in block_analysis:
            cond = block_analysis["xi_B"]["condition_number"]
            macros.append(f"\\newcommand{{\\ebcovCondB}}{{{cond:.1e}}}")
        if "xi_amb" in block_analysis:
            cond = block_analysis["xi_amb"]["condition_number"]
            macros.append(f"\\newcommand{{\\ebcovCondAmb}}{{{cond:.1e}}}")

        # Full matrix
        if "condition_number" in ev:
            macros.append(f"\\newcommand{{\\ebcovCondFull}}{{{ev['condition_number']:.1e}}}")
        if "n_bins" in ev:
            macros.append(f"\\newcommand{{\\ebcovNbins}}{{{ev['n_bins']}}}")

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

    # PTE variation across blinds (from pure_eb_data_vector, not covariance_blind_consistency)
    eb_path = claims_dir / "pure_eb_data_vector" / "evidence.json"
    if eb_path.exists():
        with open(eb_path) as f:
            eb_data = json.load(f)
        fid = eb_data.get("evidence", {}).get("fiducial", {})
        joint_ptes = [fid.get(f"pte_joint_{b}") for b in ["A", "B", "C"] if f"pte_joint_{b}" in fid]
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

        # Generate individual macros for each version
        for ver, ver_data in versions.items():
            short_ver = ver.split("v1.4.")[1].split("_")[0]  # e.g. "5", "6", "8", "11.2"
            prefix = f"configPte{VERSION_WORDS.get(short_ver, short_ver)}"

            xip = ver_data.get("xip_stats", {})
            xim = ver_data.get("xim_stats", {})
            cosebis = ver_data.get("cosebis_stats", {})

            # Fiducial PTEs
            if "pte_at_fiducial" in xip:
                macros.append(f"\\newcommand{{\\{prefix}Xip}}{{{_format_value(xip['pte_at_fiducial'])}}}")
            if "pte_at_fiducial" in xim:
                macros.append(f"\\newcommand{{\\{prefix}Xim}}{{{_format_value(xim['pte_at_fiducial'])}}}")
            if "pte_at_fiducial" in cosebis:
                macros.append(f"\\newcommand{{\\{prefix}Cosebis}}{{{_format_value(cosebis['pte_at_fiducial'])}}}")

            # Full-range PTEs
            if "pte_at_full_range" in xip:
                macros.append(f"\\newcommand{{\\{prefix}XipFull}}{{{_format_value(xip['pte_at_full_range'])}}}")
            if "pte_at_full_range" in xim:
                macros.append(f"\\newcommand{{\\{prefix}XimFull}}{{{_format_value(xim['pte_at_full_range'])}}}")
            if "pte_at_full_range" in cosebis:
                macros.append(f"\\newcommand{{\\{prefix}CosebisFull}}{{{_format_value(cosebis['pte_at_full_range'])}}}")

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

        for ver, ver_data in versions.items():
            short_ver = ver.split("v1.4.")[1].split("_")[0]
            prefix = f"clPte{VERSION_WORDS.get(short_ver, short_ver)}"

            # Fiducial PTEs
            if "pte_at_fiducial" in ver_data:
                macros.append(f"\\newcommand{{\\{prefix}Fid}}{{{_format_value(ver_data['pte_at_fiducial'])}}}")

            # Full-range PTEs
            if "pte_at_full_range" in ver_data:
                macros.append(f"\\newcommand{{\\{prefix}Full}}{{{_format_value(ver_data['pte_at_full_range'])}}}")

        macros.append("")

    content = "\n".join(macros)
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        print(f"  → {output_path}")


def generate_pte_tables(claims_dir: Path, output_dir: Path, fiducial_version: str, versions: list, version_labels: dict, config: dict):
    """Generate LaTeX table files from PTE evidence.

    Creates:
    - pte_table_results.tex: Fiducial version PTE summary
    - pte_table_appendix.tex: All versions PTE comparison
    """
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

    # Get short version label for caption
    fid_label = version_labels.get(fiducial_version, fiducial_version)

    # Results table (fiducial only)
    if fiducial_version in config_data or fiducial_version in harmonic_data:
        results_table = []
        results_table.append("% Auto-generated PTE summary table (Results section)")
        results_table.append("% Regenerate: snakemake paper_macros")
        results_table.append(r"\begin{table}")
        results_table.append(r"  \centering")
        results_table.append(rf"  \caption{{B-mode PTE values for {fid_label} at fiducial and full-range scale cuts.}}")
        results_table.append(r"  \label{tab:pte_results}")
        results_table.append(r"  \begin{tabular}{lccc}")
        results_table.append(r"    \hline")
        results_table.append(r"    Statistic & PTE (fiducial) & PTE (full range) & Fiducial cut \\")
        results_table.append(r"    \hline")

        if fiducial_version in config_data:
            cfg = config_data[fiducial_version]
            xip = cfg.get("xip_stats", {})
            xim = cfg.get("xim_stats", {})
            cosebis = cfg.get("cosebis_stats", {})

            # Config-space scale cuts from config (Paper II, Goh et al.)
            config_cut = f"[{config['fiducial']['fiducial_min_scale']}--{config['fiducial']['fiducial_max_scale']}]$'$"
            if xip:
                pte_fid = _format_value(xip.get("pte_at_fiducial", float("nan")))
                pte_full = _format_value(xip.get("pte_at_full_range", float("nan")))
                results_table.append(f"    $\\xi_+^B$ & {pte_fid} & {pte_full} & {config_cut} \\\\")
            if xim:
                pte_fid = _format_value(xim.get("pte_at_fiducial", float("nan")))
                pte_full = _format_value(xim.get("pte_at_full_range", float("nan")))
                results_table.append(f"    $\\xi_-^B$ & {pte_fid} & {pte_full} & {config_cut} \\\\")
            if cosebis:
                pte_fid = _format_value(cosebis.get("pte_at_fiducial", float("nan")))
                pte_full = _format_value(cosebis.get("pte_at_full_range", float("nan")))
                results_table.append(f"    COSEBIS $B_n$ & {pte_fid} & {pte_full} & {config_cut} \\\\")

        if fiducial_version in harmonic_data:
            harm = harmonic_data[fiducial_version]
            pte_fid = _format_value(harm.get("pte_at_fiducial", float("nan")))
            pte_full = _format_value(harm.get("pte_at_full_range", float("nan")))
            harmonic_cut = f"$\\ell$=[{config['cl']['fiducial_ell_min']}--{config['cl']['fiducial_ell_max']}]"
            results_table.append(f"    $C_\\ell^{{BB}}$ & {pte_fid} & {pte_full} & {harmonic_cut} \\\\")

        results_table.append(r"    \hline")
        results_table.append(r"  \end{tabular}")
        results_table.append(r"\end{table}")

        results_path = output_dir / "pte_table_results.tex"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text("\n".join(results_table))
        print(f"  → {results_path}")

    # Appendix table (all versions)
    if config_data or harmonic_data:
        appendix_table = []
        appendix_table.append("% Auto-generated PTE comparison table (Appendix)")
        appendix_table.append("% Regenerate: snakemake paper_macros")
        appendix_table.append(r"\begin{table*}")
        appendix_table.append(r"  \centering")
        appendix_table.append(r"  \caption{B-mode PTE values at fiducial scale cuts across catalog versions.}")
        appendix_table.append(r"  \label{tab:pte_appendix}")
        appendix_table.append(r"  \begin{tabular}{lcccc}")
        appendix_table.append(r"    \hline")
        appendix_table.append(r"    Version & $\xi_+^B$ & $\xi_-^B$ & COSEBIS $B_n$ & $C_\ell^{BB}$ \\")
        appendix_table.append(r"    \hline")

        for ver in versions:
            label = version_labels.get(ver, ver)
            # Mark fiducial version in table
            if ver == fiducial_version:
                label = f"{label} (fiducial)"
            row = [f"    {label}"]

            # Config-space - fiducial PTEs only
            if ver in config_data:
                cfg = config_data[ver]
                for stat in ["xip_stats", "xim_stats", "cosebis_stats"]:
                    s = cfg.get(stat, {})
                    pte = _format_value(s.get("pte_at_fiducial", float("nan")))
                    row.append(f"& {pte}")
            else:
                row.extend(["& --"] * 3)

            # Harmonic-space - fiducial PTE
            if ver in harmonic_data:
                harm = harmonic_data[ver]
                pte = _format_value(harm.get("pte_at_fiducial", harm.get("pte_at_full_range", float("nan"))))
                row.append(f"& {pte}")
            else:
                row.append("& --")

            row.append(r" \\")
            appendix_table.append(" ".join(row))

        appendix_table.append(r"    \hline")
        appendix_table.append(r"  \end{tabular}")
        appendix_table.append(r"\end{table*}")

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
            # Extract key values for the summary
            ev = data.get("evidence", {})
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
        "artifacts": {"macros": "claims_macros.tex"},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(evidence, f, indent=2)
    print(f"  → {output_path}")


if __name__ == "__main__":
    # When run via snakemake
    claims_dir = Path(snakemake.params.claims_dir)
    config = snakemake.config
    fiducial_version = config["fiducial"]["version"]
    versions = config["versions"]
    version_labels = config["plotting"]["version_labels"]

    # Separate macro outputs from evidence output
    macro_outputs = [Path(p) for p in snakemake.output if p.endswith(".tex")]
    evidence_outputs = [Path(p) for p in snakemake.output if p.endswith("evidence.json")]

    print(f"Generating macros from {claims_dir}")
    generate_macros(claims_dir, macro_outputs, fiducial_version)

    # Generate PTE tables
    if macro_outputs:
        paper_dir = macro_outputs[0].parent
        print(f"Generating PTE tables to {paper_dir}")
        generate_pte_tables(claims_dir, paper_dir, fiducial_version, versions, version_labels, config)

    # Generate evidence.json if requested
    for evidence_path in evidence_outputs:
        spec_id = evidence_path.parent.name  # e.g., xi_cosmology_paper
        generate_evidence(
            spec_id=spec_id,
            spec_path=f"workflow/config/{spec_id}.md",
            depends_on=["pure_eb_data_vector", "cosebis_version_comparison", "covariance_blind_consistency"],
            claims_dir=claims_dir,
            output_path=evidence_path,
        )
